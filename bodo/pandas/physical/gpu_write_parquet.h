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

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
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
/// internal buffer and only emits a parquet file part when the accumulated row
/// count exceeds `chunk_rows` or when the stream is finished. File names are
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

    void recreate_output_dir(const std::string &path) {
        if (fs::exists(path)) {
            fs::remove_all(path);
        }
        fs::create_directories(path);
    }

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
          // Interpret the configured chunk threshold as a row count threshold.
          // The existing CPU writer uses get_parquet_chunk_size() for bytes;
          // here we reuse the same configuration entry but treat it as rows.
          chunk_rows(static_cast<int64_t>(get_parquet_chunk_size())) {
        time_pt start_init = start_timer();

        recreate_output_dir(this->path);

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

    // ConsumeBatch signature using GPU_DATA
    OperatorResult ConsumeBatch(GPU_DATA input_batch,
                                OperatorResult prev_op_result) override {
        if (finished)
            return OperatorResult::FINISHED;

        auto incoming_tbl = input_batch.table;  // std::shared_ptr<cudf::table>
        auto incoming_schema =
            input_batch.schema;  // std::shared_ptr<arrow::Schema>

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
            std::unique_ptr<cudf::table> concat_uptr = cudf::concatenate(views);

            // convert unique_ptr -> shared_ptr by moving ownership
            buffer_table = std::shared_ptr<cudf::table>(std::move(concat_uptr));
            buffer_rows = static_cast<int64_t>(buffer_table->num_rows());
        }

        // decide flush by rows
        bool is_last = (prev_op_result == OperatorResult::FINISHED);
        bool should_flush = is_last || (buffer_rows >= chunk_rows);

        if (should_flush && buffer_table && buffer_rows > 0) {
            std::string fname_prefix = get_fname_prefix();
            std::string out_path =
                (fs::path(path) / (fname_prefix + "0.parquet")).string();

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
            }

            auto options = builder.build();

            // do the actual write
            cudf::io::write_parquet(options);

            // reset buffer
            buffer_table.reset();
            buffer_rows = 0;
            ++iter;
            ++metrics.n_files_written;
        }

        if (is_last) {
            finished = true;
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
    // Same get_fname_prefix() logic as CPU writer to keep file ordering
    // identical.
    std::string get_fname_prefix() {
        std::string base_prefix = "part-";

        int MAX_ITER = 1000;
        int n_max_digits = static_cast<int>(std::ceil(std::log10(MAX_ITER)));

        // Number of prefix characters to add ("batch" number)
        int n_prefix = (iter == 0) ? 0
                                   : static_cast<int>(std::floor(
                                         std::log(iter) / std::log(MAX_ITER)));

        std::string iter_str = std::to_string(iter);
        int n_zeros = ((n_prefix + 1) * n_max_digits) -
                      static_cast<int>(iter_str.length());
        iter_str = std::string(n_zeros, '0') + iter_str;

        return base_prefix + std::string(n_prefix, 'b') + iter_str + "-";
    }

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
    // chunk_rows is the row-count threshold for emitting a parquet part.
    const int64_t chunk_rows;
    int64_t iter = 0;

    PhysicalGPUWriteParquetMetrics metrics;

    // GPU buffer: owning cudf::table that accumulates incoming batches
    std::shared_ptr<cudf::table> buffer_table;
    int64_t buffer_rows = 0;
};
