#pragma once

#include "../libs/_table_builder.h"

#include "../../io/parquet_write.h"
#include "../../libs/_bodo_to_arrow.h"
#include "../../libs/streaming/_shuffle.h"
#include "_bodo_write_function.h"
#include "physical/operator.h"
#include "physical/write_parquet_utils.h"

struct PhysicalWriteParquetMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t max_buffer_size = 0;
    stat_t n_files_written = 0;

    time_t init_time = 0;
    time_t consume_time = 0;
    time_t accumulate_time = 0;
    time_t file_write_time = 0;
};

class PhysicalWriteParquet : public PhysicalSink {
   public:
    explicit PhysicalWriteParquet(std::shared_ptr<bodo::Schema> in_schema,
                                  ParquetWriteFunctionData& bind_data)
        : path(std::move(bind_data.path)),
          compression(std::move(bind_data.compression)),
          row_group_size(bind_data.row_group_size),
          bucket_region(std::move(bind_data.bucket_region)),
          arrow_schema(std::move(bind_data.arrow_schema)),
          is_last_state(std::make_shared<IsLastState>()),
          finished(false) {
        // Similar to streaming parquet write in Bodo JIT
        // https://github.com/bodo-ai/Bodo/blob/9902c4bd19f0c1f85ef0c971c58e42cf84a35fc7/bodo/io/stream_parquet_write.py#L269
        time_pt start_init = start_timer();

        pq_write_create_dir(this->path.c_str());

        // Initialize the buffer and dictionary builders
        for (auto& col : in_schema->column_types) {
            dict_builders.emplace_back(
                create_dict_builder_for_array(col->copy(), false));
        }
        buffer = std::make_shared<TableBuildBuffer>(in_schema, dict_builders);
        this->metrics.init_time += end_timer(start_init);
    }

    virtual ~PhysicalWriteParquet() = default;

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        // Similar to streaming parquet write in Bodo JIT
        // https://github.com/bodo-ai/Bodo/blob/3741f4e05e4236b5c3cc35ef5ecccad921f17dc4/bodo/io/stream_parquet_write.py#L433
        if (finished) {
            return OperatorResult::FINISHED;
        }

        // ===== Part 1: Accumulate batch in writer and compute total size =====
        time_pt start_accumulate = start_timer();
        buffer->UnifyTablesAndAppend(input_batch, dict_builders);
        int64_t buffer_nbytes =
            table_local_memory_size(buffer->data_table, true);
        this->metrics.accumulate_time += end_timer(start_accumulate);
        this->metrics.max_buffer_size =
            std::max(this->metrics.max_buffer_size, buffer_nbytes);

        // Sync is_last flag
        bool is_last = prev_op_result == OperatorResult::FINISHED;
        is_last = static_cast<bool>(sync_is_last_non_blocking(
            is_last_state.get(), static_cast<int32_t>(is_last)));

        // === Part 2: Write Parquet file if file size threshold is exceeded ===
        time_pt start_write = start_timer();
        if (is_last || buffer_nbytes >= this->chunk_size) {
            std::shared_ptr<table_info> data = buffer->data_table;

            if (data->nrows() > 0) {
                std::shared_ptr<arrow::Table> arrow_table =
                    bodo_table_to_arrow(data, arrow_schema->field_names(),
                                        arrow_schema->metadata());

                std::vector<bodo_array_type::arr_type_enum> bodo_array_types;
                for (auto& col : data->columns) {
                    bodo_array_types.emplace_back(col->arr_type);
                }

                pq_write(path.c_str(), arrow_table, compression.c_str(), true,
                         bucket_region.c_str(), row_group_size,
                         get_fname_prefix(iter).c_str(), bodo_array_types,
                         false, "", nullptr);
            }
            // Reset the buffer for the next batch
            buffer->Reset();
        }
        this->metrics.file_write_time += end_timer(start_write);
        this->metrics.consume_time += end_timer(start_accumulate);

        if (is_last) {
            finished = true;
        }

        iter++;
        return is_last ? OperatorResult::FINISHED
                       : OperatorResult::NEED_MORE_INPUT;
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
        // Write doesn't produce rows
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1), 0);
    }

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        return std::shared_ptr<table_info>(nullptr);
    }

   private:
    // State similar to streaming parquet write in Bodo JIT
    // https://github.com/bodo-ai/Bodo/blob/1ab0dedf37f7b2ae8551bb95a1e3d4cfc70c553f/bodo/io/stream_parquet_write.py#L289
    const std::string path;
    const std::string compression;
    const int64_t row_group_size;
    const std::string bucket_region;
    const std::shared_ptr<arrow::Schema> arrow_schema;
    std::shared_ptr<TableBuildBuffer> buffer;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    const std::shared_ptr<IsLastState> is_last_state;
    bool finished = false;
    const int64_t chunk_size = get_parquet_chunk_size();
    int64_t iter = 0;

    PhysicalWriteParquetMetrics metrics;

    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        metrics_out.emplace_back(
            StatMetric("max_buffer_size", this->metrics.max_buffer_size));
        metrics_out.emplace_back(
            StatMetric("n_files_written", this->metrics.n_files_written));
        metrics_out.emplace_back(
            TimerMetric("accumulate_time", this->metrics.accumulate_time));
        metrics_out.emplace_back(
            TimerMetric("file_write_time", this->metrics.file_write_time));

        // Add the dict builder metrics if they exist
        for (size_t i = 0; i < this->dict_builders.size(); ++i) {
            auto dict_builder = this->dict_builders[i];
            if (dict_builder) {
                dict_builder->GetMetrics().add_to_metrics(
                    metrics_out, fmt::format("dict_builder_{}", i));
            }
        }
    }
};
