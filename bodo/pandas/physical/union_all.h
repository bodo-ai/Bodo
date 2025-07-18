#pragma once

#include <memory>
#include <utility>
#include "../../libs/streaming/_join.h"
#include "../libs/_array_utils.h"
#include "../libs/_distributed.h"
#include "../libs/_table_builder.h"
#include "operator.h"

/**
 * @brief Physical node for union all.
 *
 */
class PhysicalUnionAll : public PhysicalSourceSink, public PhysicalSink {
   public:
    explicit PhysicalUnionAll(std::shared_ptr<bodo::Schema> input_schema)
        : output_schema(input_schema) {}

    virtual ~PhysicalUnionAll() = default;

    void Finalize() override {}

    /**
     * @brief Do union all.
     *
     * @return std::pair<std::shared_ptr<table_info>, OperatorResult>
     * The output table from the current operation and whether there is more
     * output.
     */
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        if (!collected_rows) {
            collected_rows = std::make_unique<ChunkedTableBuilderState>(
                input_batch->schema(), get_streaming_batch_size());
        }
        collected_rows->builder->AppendBatch(input_batch);
        return (prev_op_result == OperatorResult::FINISHED)
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        // Union all should be between pipelines and act alternatively as a sink
        // then source but there should never be the need to ask for the result
        // all in one go.
        throw std::runtime_error("GetResult called on a union all node.");
    }

    /**
     * @brief ProcessBatch streaming through union
     *
     * returns std::pair<std::shared_ptr<table_info>, OperatorResult>
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) override {
        if (!first_processed_batch) {
            first_processed_batch = input_batch;
        }
        if (collected_rows && !collected_rows->builder->empty()) {
            auto next_batch = collected_rows->builder->PopChunk(true);
            return {std::get<0>(next_batch),
                    //(prev_op_result == OperatorResult::FINISHED &&
                    // collected_rows->builder->empty())
                    //    ? OperatorResult::FINISHED
                    //    : OperatorResult::HAVE_MORE_OUTPUT};
                    OperatorResult::HAVE_MORE_OUTPUT};
        } else {
            if (first_processed_batch) {
                input_batch = first_processed_batch;
                first_processed_batch = nullptr;
            }
            return {input_batch, prev_op_result == OperatorResult::FINISHED
                                     ? OperatorResult::FINISHED
                                     : OperatorResult::NEED_MORE_INPUT};
        }
    }

    /**
     * @brief Get the physical schema of the output data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

   private:
    std::unique_ptr<ChunkedTableBuilderState> collected_rows;
    std::shared_ptr<table_info> first_processed_batch;
    const std::shared_ptr<bodo::Schema> output_schema;
};
