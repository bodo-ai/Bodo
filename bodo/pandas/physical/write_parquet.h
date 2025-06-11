#pragma once

#include "../libs/_table_builder.h"

#include "../../io/parquet_write.h"
#include "../../libs/_bodo_to_arrow.h"
#include "../../libs/streaming/_shuffle.h"
#include "_bodo_scan_function.h"
#include "physical/operator.h"

class PhysicalWriteParquet : public PhysicalSink {
   public:
    explicit PhysicalWriteParquet(std::shared_ptr<bodo::Schema> in_schema,
                                  ParquetWriteFunctionData& bind_data)
        : path(std::move(bind_data.path)),
          compression(std::move(bind_data.compression)),
          row_group_size(bind_data.row_group_size),
          bucket_region(std::move(bind_data.bucket_region)) {
        // Similar to streaming parquet write in Bodo JIT
        // https://github.com/bodo-ai/Bodo/blob/9902c4bd19f0c1f85ef0c971c58e42cf84a35fc7/bodo/io/stream_parquet_write.py#L269

        pq_write_create_dir(this->path.c_str());

        // Initialize the buffer and dictionary builders
        for (auto& col : in_schema->column_types) {
            dict_builders.emplace_back(
                create_dict_builder_for_array(col->copy(), false));
        }
        buffer = std::make_shared<TableBuildBuffer>(in_schema, dict_builders);

        is_last_state = std::make_shared<IsLastState>();
        finished = false;
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
        buffer->UnifyTablesAndAppend(input_batch, dict_builders);
        int64_t buffer_nbytes =
            table_local_memory_size(buffer->data_table, true);

        // Sync is_last flag
        bool is_last = prev_op_result == OperatorResult::FINISHED;
        is_last = static_cast<bool>(sync_is_last_non_blocking(
            is_last_state.get(), static_cast<int32_t>(is_last)));

        // === Part 2: Write Parquet file if file size threshold is exceeded ===
        if (is_last || buffer_nbytes >= this->chunk_size) {
            std::shared_ptr<table_info> data = buffer->data_table;

            if (data->nrows() > 0) {
                std::shared_ptr<arrow::Table> arrow_table =
                    bodo_table_to_arrow(data);

                std::vector<bodo_array_type::arr_type_enum> bodo_array_types;
                for (auto& col : data->columns) {
                    bodo_array_types.emplace_back(col->arr_type);
                }

                pq_write(path.c_str(), arrow_table, compression.c_str(), true,
                         bucket_region.c_str(), row_group_size, "part-",
                         bodo_array_types, false, "", nullptr);
            }
            // Reset the buffer for the next batch
            buffer->Reset();
        }

        if (is_last) {
            finished = true;
        }

        return is_last ? OperatorResult::FINISHED
                       : OperatorResult::NEED_MORE_INPUT;
    }

    void Finalize() override {}

    std::shared_ptr<table_info> GetResult() override { return nullptr; }

   private:
    // State similar to streaming parquet write in Bodo JIT
    // https://github.com/bodo-ai/Bodo/blob/1ab0dedf37f7b2ae8551bb95a1e3d4cfc70c553f/bodo/io/stream_parquet_write.py#L289
    const std::string path;
    const std::string compression;
    const int64_t row_group_size;
    const std::string bucket_region;
    std::shared_ptr<TableBuildBuffer> buffer;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    std::shared_ptr<IsLastState> is_last_state;
    bool finished = false;
    int64_t chunk_size = get_parquet_chunk_size();
};
