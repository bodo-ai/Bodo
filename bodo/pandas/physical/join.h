
#pragma once

#include "../../libs/streaming/_join.h"
#include "operator.h"

/**
 * @brief Physical node for join.
 *
 */
class PhysicalJoin : public PhysicalSourceSink, public PhysicalSink {
   public:
    explicit PhysicalJoin() {}

    virtual ~PhysicalJoin() = default;

    void InitializeJoinState(
        const std::shared_ptr<bodo::Schema> build_table_schema,
        const std::shared_ptr<bodo::Schema> probe_table_schema) {
        this->join_state = std::make_shared<HashJoinState>(
            build_table_schema, probe_table_schema,
            // TODO: handle keys properly
            1,
            // TODO: handle outer joins properly
            false, false,
            // TODO: handle broadcast join properly
            false, nullptr, true, true, get_streaming_batch_size(), -1,
            // TODO: add op_id
            -1);
    }

    void Finalize() override {}

    /**
     * @brief process input tables to build side of join (populate the hash
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(
        std::shared_ptr<table_info> input_batch) override {
        return OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief Run join probe on the input batch
     *
     * @param input_batch input batch to probe
     * @return output batch of probe and return flag
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override {
        std::shared_ptr<table_info> out_table;
        return {out_table, OperatorResult::NEED_MORE_INPUT};
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::shared_ptr<table_info> GetResult() override {
        // Join build doesn't return output results
        throw std::runtime_error("GetResult called on a join node.");
    }

    std::shared_ptr<bodo::Schema> getOutputSchema() override {
        // TODO
        return nullptr;
    }

   private:
    std::shared_ptr<HashJoinState> join_state;
};
