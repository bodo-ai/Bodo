#pragma once

#include <cstdint>
#include "../../libs/streaming/_join.h"
#include "../_util.h"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "expression.h"
#include "operator.h"

struct PhysicalGPUJoinMetrics {
    using time_t = MetricBase::TimerValue;
    using stat_t = MetricBase::StatValue;

    time_t init_time = 0;
    time_t consume_time = 0;
    time_t process_batch_time = 0;

    stat_t output_row_count = 0;
};

/**
 * @brief Physical node for join.
 *
 */
class PhysicalGPUJoin : public PhysicalGPUProcessBatch, public PhysicalGPUSink {
   public:
    explicit PhysicalGPUJoin(duckdb::LogicalComparisonJoin& logical_join)
        : has_non_equi_cond(false),
          is_mark_join(logical_join.join_type == duckdb::JoinType::MARK),
          is_anti_join(logical_join.join_type == duckdb::JoinType::ANTI ||
                       logical_join.join_type == duckdb::JoinType::RIGHT_ANTI) {
        throw std::runtime_error("Not implemented.");
    }

    /**
     * @brief Physical Join constructor for cross join.
     *
     */
    PhysicalGPUJoin(duckdb::LogicalCrossProduct& logical_join,
                    const std::shared_ptr<bodo::Schema> build_table_schema,
                    const std::shared_ptr<bodo::Schema> probe_table_schema)
        : has_non_equi_cond(false) {
        throw std::runtime_error("Not implemented.");
    }

    void buildProbeSchemas(
        duckdb::LogicalComparisonJoin& logical_join,
        duckdb::vector<duckdb::JoinCondition>& conditions,
        const std::shared_ptr<bodo::Schema> build_table_schema,
        const std::shared_ptr<bodo::Schema> probe_table_schema) {}

    virtual ~PhysicalGPUJoin() = default;

    void FinalizeSink() override {}

    void FinalizeProcessBatch() override {
        throw std::runtime_error("Not implemented.");
    }

    JoinState* getJoinStatePtr() const {
        throw std::runtime_error("Not implemented.");
    }

    /**
     * @brief process input tables to build side of join (populate the hash
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(GPU_DATA input_batch,
                                OperatorResult prev_op_result) override {
        throw std::runtime_error("Not implemented.");
    }

    /**
     * @brief Run join probe on the input batch
     *
     * @param input_batch input batch to probe
     * @return output batch of probe and return flag
     */
    std::pair<GPU_DATA, OperatorResult> ProcessBatch(
        GPU_DATA input_batch, OperatorResult prev_op_result) override {
        throw std::runtime_error("Not implemented.");
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::variant<GPU_DATA, PyObject*> GetResult() override {
        // Join build doesn't return output results
        throw std::runtime_error("GetResult called on a join node.");
    }

    /**
     * @brief Get the output schema of join probe
     *
     * @return std::shared_ptr<bodo::Schema>
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

    std::string ToString() override { return PhysicalGPUSink::ToString(); }

    int64_t getOpId() const { return PhysicalGPUSink::getOpId(); }

   private:
    std::shared_ptr<bodo::Schema> output_schema;

    bool has_non_equi_cond;
    bool is_mark_join = false;
    bool is_anti_join = false;
    bool use_cudf = false;

    PhysicalGPUJoinMetrics metrics;
};
