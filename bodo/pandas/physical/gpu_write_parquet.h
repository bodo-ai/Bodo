#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../io/parquet_write.h"
#include "../../libs/_bodo_to_arrow.h"
#include "../../libs/streaming/_shuffle.h"
#include "../libs/_table_builder.h"
#include "_bodo_write_function.h"
#include "physical/operator.h"
#include "physical/write_parquet_utils.h"

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>

#include <filesystem>
namespace fs = std::filesystem;

struct PhysicalGPUWriteParquetMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t max_buffer_rows = 0;
    stat_t n_files_written = 0;

    time_t init_time = 0;
    time_t consume_time = 0;
    time_t accumulate_time = 0;
    time_t file_write_time = 0;
};

/// @brief GPU-backed parquet writer for streaming pipelines.
///
/// Mirrors the CPU writer in write_parquet.h but accepts GPU_DATA as input
/// (GPU_DATA == std::pair<std::unique_ptr<cudf::table>,
/// std::shared_ptr<arrow::Schema>>). Accumulates incoming GPU tables into an
/// internal buffer and only emits a parquet file part when the accumulated
/// bytes exceeds `chunk_bytes` or when the stream is finished. File names are
/// generated using the same get_fname_prefix() logic as the CPU writer so parts
/// are lexicographically ordered.
class PhysicalGPUWriteParquet : public PhysicalGPUSink {
   private:
    std::vector<std::string> get_names_from_arrow(
        std::shared_ptr<arrow::Schema> schema) {
        std::vector<std::string> names;
        names.reserve(schema->num_fields());
        for (int i = 0; i < schema->num_fields(); i++) {
            names.push_back(schema->field(i)->name());
        }
        return names;
    }

    std::shared_ptr<StreamAndEvent> prev_batch_se;

   public:
    explicit PhysicalGPUWriteParquet(std::shared_ptr<bodo::Schema> in_schema,
                                     ParquetWriteFunctionData &bind_data)
        : path(std::move(bind_data.path)),
          compression(std::move(bind_data.compression)),
          row_group_size(bind_data.row_group_size),
          bucket_region(std::move(bind_data.bucket_region)),
          arrow_schema(std::move(bind_data.arrow_schema)),
          is_last_state(std::make_shared<IsLastState>()),
          finished(false),
          chunk_bytes(static_cast<int64_t>(get_parquet_chunk_size())) {
        time_pt start_init = start_timer();

        pq_write_create_dir(this->path.c_str());

        column_names = get_names_from_arrow(arrow_schema);

        // Keep dict builders for API parity with CPU writer (not used for GPU
        // write logic here, but may be useful for metrics/compatibility).
        for (auto &col : in_schema->column_types) {
            dict_builders.emplace_back(
                create_dict_builder_for_array(col->copy(), false));
        }

        // Start with no buffer; we'll adopt the first incoming table.
        buffer_table.reset();
        buffer_rows = 0;

        this->metrics.init_time += end_timer(start_init);
    }

    virtual ~PhysicalGPUWriteParquet() = default;

    size_t column_bytes(const cudf::column_view &col,
                        std::shared_ptr<StreamAndEvent> se) {
        size_t total = 0;

        // null mask
        if (col.nullable()) {
            total += cudf::bitmask_allocation_size_bytes(col.size());
        }

        // FIXED-WIDTH TYPES
        if (cudf::is_fixed_width(col.type())) {
            total += col.size() * cudf::size_of(col.type());
            return total;
        }

        // STRINGS
        if (col.type().id() == cudf::type_id::STRING) {
            cudf::strings_column_view scv(col);

            // offsets buffer
            total += scv.offsets().size() * sizeof(int32_t);

            // chars buffer (requires a stream)
            total += scv.chars_size(se->stream);

            return total;
        }

        // LISTS
        if (col.type().id() == cudf::type_id::LIST) {
            cudf::lists_column_view lcv(col);

            // offsets buffer
            total += lcv.offsets().size() * sizeof(int32_t);

            // recurse into child column
            total += column_bytes(lcv.child(), se);

            return total;
        }

        // STRUCTS
        if (col.type().id() == cudf::type_id::STRUCT) {
            for (int i = 0; i < col.num_children(); ++i) {
                total += column_bytes(col.child(i), se);
            }
            return total;
        }

        // fallback
        return total;
    }

    size_t table_bytes(const cudf::table_view &tv,
                       std::shared_ptr<StreamAndEvent> se) {
        size_t total = 0;
        for (auto const &col : tv) {
            total += column_bytes(col, se);
        }
        return total;
    }

    size_t row_size_bytes(const cudf::table_view &tv,
                          std::shared_ptr<StreamAndEvent> se) {
        if (tv.num_rows() == 0)
            return 0;
        return table_bytes(tv, se) / tv.num_rows();
    }

    // ConsumeBatch signature using GPU_DATA
    OperatorResult ConsumeBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        if (prev_batch_se) {
            // Write batches to file in pipeline order.
            // Wait for last write pipeline batch to finish before running
            // this one.
            prev_batch_se->event.wait(se->stream);
        }
        prev_batch_se = se;

        if (finished) {
            cudaStreamSynchronize(se->stream);
            return OperatorResult::FINISHED;
        }

        std::shared_ptr<cudf::table> incoming_tbl = input_batch.table;
        std::shared_ptr<arrow::Schema> incoming_schema = input_batch.schema;

        // adopt or concatenate
        if (!buffer_table) {
            // adopt incoming shared_ptr directly
            buffer_table = incoming_tbl;
            buffer_rows = buffer_table
                              ? static_cast<int64_t>(buffer_table->num_rows())
                              : 0;
        } else if (incoming_tbl && incoming_tbl->num_rows() > 0) {
            // concatenate: build views, get unique_ptr result
            std::vector<cudf::table_view> views{buffer_table->view(),
                                                incoming_tbl->view()};
            std::unique_ptr<cudf::table> concat_uptr =
                cudf::concatenate(views, se->stream);

            // convert unique_ptr -> shared_ptr by moving ownership
            buffer_table = std::shared_ptr<cudf::table>(std::move(concat_uptr));
            buffer_rows = static_cast<int64_t>(buffer_table->num_rows());
        }

        // decide flush by rows
        bool is_last = (prev_op_result == OperatorResult::FINISHED);
        size_t row_size = row_size_bytes(buffer_table->view(), se);
        size_t buffer_bytes = buffer_rows * row_size;
        bool should_flush = is_last || (buffer_bytes >= chunk_bytes);

        if (should_flush && buffer_table && buffer_rows > 0) {
            std::string fname_prefix = get_fname_prefix(iter);
            int myrank, num_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            std::string fname = gen_pieces_file_name(myrank, num_ranks,
                                                     fname_prefix, ".parquet");
            std::string out_path = (fs::path(path) / fname).string();

            cudf::table_view bttv = buffer_table->view();
            cudf::io::table_input_metadata meta{bttv};
            for (int i = 0; i < bttv.num_columns(); ++i) {
                meta.column_metadata[i].set_name(column_names[i]);
            }

            // build writer options (row_group_size, compression, etc.)
            auto sink = cudf::io::sink_info(out_path);
            auto builder = cudf::io::parquet_writer_options::builder(sink, bttv)
                               .metadata(meta);
            if (row_group_size > 0) {
                builder = builder.row_group_size_rows(row_group_size);
            }
            builder = builder.stats_level(
                cudf::io::statistics_freq::STATISTICS_ROWGROUP);
            try {
                builder.compression(static_cast<cudf::io::compression_type>(
                    pq_compression_from_string(compression)));
            } catch (...) {
                throw std::runtime_error(
                    "PhysicalGPUWriteParquet(): pq_compression_from_string "
                    "failed.");
            }

            auto options = builder.build();

            // do the actual write
            cudf::io::write_parquet(options, se->stream);

            // reset buffer
            buffer_table.reset();
            buffer_rows = 0;
            ++iter;
            ++metrics.n_files_written;
        }

        if (is_last) {
            finished = true;
            cudaStreamSynchronize(se->stream);
            return OperatorResult::FINISHED;
        }

        return OperatorResult::NEED_MORE_INPUT;
    }

    std::variant<std::shared_ptr<table_info>, PyObject *> GetResult() override {
        return std::shared_ptr<table_info>(nullptr);
    }

    void FinalizeSink() override {
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().SubmitOperatorName(getOpId(),
                                                            ToString());
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 0),
            this->metrics.init_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.consume_time);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1), 0);
    }

   private:
    void ReportMetrics(std::vector<MetricBase> &metrics_out) {
        metrics_out.emplace_back(
            StatMetric("max_buffer_rows", this->metrics.max_buffer_rows));
        metrics_out.emplace_back(
            StatMetric("n_files_written", this->metrics.n_files_written));
        metrics_out.emplace_back(
            TimerMetric("accumulate_time", this->metrics.accumulate_time));
        metrics_out.emplace_back(
            TimerMetric("file_write_time", this->metrics.file_write_time));

        for (size_t i = 0; i < this->dict_builders.size(); ++i) {
            auto dict_builder = this->dict_builders[i];
            if (dict_builder) {
                dict_builder->GetMetrics().add_to_metrics(
                    metrics_out, fmt::format("dict_builder_{}", i));
            }
        }
    }

    // Helper to map compression string to cudf enum (extend mapping as needed)
    int pq_compression_from_string(const std::string &c) {
        if (c == "snappy")
            return static_cast<int>(cudf::io::compression_type::SNAPPY);
        if (c == "gzip")
            return static_cast<int>(cudf::io::compression_type::GZIP);
        if (c == "brotli")
            return static_cast<int>(cudf::io::compression_type::BROTLI);
        if (c == "lz4")
            return static_cast<int>(cudf::io::compression_type::LZ4);
        if (c == "zstd")
            return static_cast<int>(cudf::io::compression_type::ZSTD);
        return static_cast<int>(cudf::io::compression_type::SNAPPY);
    }

    const std::string path;
    const std::string compression;
    const int64_t row_group_size;
    const std::string bucket_region;
    const std::shared_ptr<arrow::Schema> arrow_schema;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    const std::shared_ptr<IsLastState> is_last_state;
    bool finished;
    std::vector<std::string> column_names;
    const uint64_t chunk_bytes;
    int64_t iter = 0;

    PhysicalGPUWriteParquetMetrics metrics;

    // GPU buffer: owning cudf::table that accumulates incoming batches
    std::shared_ptr<cudf::table> buffer_table;
    int64_t buffer_rows = 0;
};
