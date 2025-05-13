
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
        return {out_table, OperatorResult::NEED_MORE_INPUT};
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::shared_ptr<table_info> GetResult() override {
        // Join build doesn't return output results
        throw std::runtime_error("GetResult called on a join node.");
    }

   private:
    std::shared_ptr<HashJoinState> join_state;
};
