#pragma once

#include "../libs/_table_builder.h"

#include "../../io/parquet_write.h"
#include "../../libs/_bodo_to_arrow.h"
#include "_bodo_scan_function.h"
#include "physical/operator.h"

class PhysicalWriteParquet : public PhysicalSink {
   private:
    std::shared_ptr<TableBuildBuffer> buffer;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;

   public:
    explicit PhysicalWriteParquet(ParquetWriteFunctionData& bind_data)
        : path(std::move(bind_data.path)) {
        // Similar to streaming parquet write in Bodo JIT
        // https://github.com/bodo-ai/Bodo/blob/9902c4bd19f0c1f85ef0c971c58e42cf84a35fc7/bodo/io/stream_parquet_write.py#L269

        pq_write_create_dir(bind_data.path.c_str());
    }

    virtual ~PhysicalWriteParquet() = default;

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        std::shared_ptr<arrow::Table> arrow_table =
            bodo_table_to_arrow(input_batch);

        std::vector<bodo_array_type::arr_type_enum> bodo_array_types;
        for (auto& col : input_batch->columns) {
            bodo_array_types.emplace_back(col->arr_type);
        }

        pq_write(path.c_str(), arrow_table, "snappy", true, "us-east-1", -1,
                 "part-", bodo_array_types, false, "", nullptr);

        return prev_op_result == OperatorResult::FINISHED
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    void Finalize() override {}

    std::shared_ptr<table_info> GetResult() override { return nullptr; }

   private:
    const std::string path;
};
