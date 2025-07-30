#pragma once

#include <string>
#include "../../libs/streaming/_shuffle.h"
#include "_bodo_write_function.h"
#include "physical/operator.h"

class PhysicalWriteS3Vectors : public PhysicalSink {
   public:
    explicit PhysicalWriteS3Vectors(
        std::shared_ptr<bodo::Schema> in_bodo_schema,
        S3VectorsWriteFunctionData& bind_data)
        : vector_bucket_name(std::move(bind_data.vector_bucket_name)),
          index_name(std::move(bind_data.index_name)) {}

    virtual ~PhysicalWriteS3Vectors() = default;

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        if (finished) {
            return OperatorResult::FINISHED;
        }

        // TODO: write data

        // Sync is_last flag
        bool is_last = prev_op_result == OperatorResult::FINISHED;
        is_last = static_cast<bool>(sync_is_last_non_blocking(
            is_last_state.get(), static_cast<int32_t>(is_last)));

        if (is_last) {
            finished = true;
        }

        iter++;
        return is_last ? OperatorResult::FINISHED
                       : OperatorResult::NEED_MORE_INPUT;
    }

    void Finalize() override {}

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        return std::shared_ptr<table_info>(nullptr);
    }

   private:
    std::string vector_bucket_name;
    std::string index_name;

    const std::shared_ptr<IsLastState> is_last_state;
    bool finished = false;
    int64_t iter = 0;
};
