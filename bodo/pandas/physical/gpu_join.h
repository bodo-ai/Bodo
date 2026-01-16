#pragma once

#include <cstdint>
#include <cudf/types.hpp>
#include "../../libs/streaming/_join.h"
#include "../../libs/streaming/cuda_join.h"
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
        assert(logical_join.join_type == duckdb::JoinType::INNER);

        // Probe side
        duckdb::vector<duckdb::ColumnBinding> left_bindings =
            logical_join.children[0]->GetColumnBindings();
        // Build side
        duckdb::vector<duckdb::ColumnBinding> right_bindings =
            logical_join.children[1]->GetColumnBindings();

        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
            left_col_ref_map = getColRefMap(left_bindings);
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
            right_col_ref_map = getColRefMap(right_bindings);

        std::vector<cudf::size_type> probe_keys;
        std::vector<cudf::size_type> build_keys;

        for (const duckdb::JoinCondition& cond : logical_join.conditions) {
            if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL) {
                throw std::runtime_error(
                    "Non-equi join conditions are not supported in GPU join.");
            }
            if (cond.left->GetExpressionClass() !=
                duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "Join condition left side is not a column reference.");
            }
            if (cond.right->GetExpressionClass() !=
                duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "Join condition right side is not a column reference.");
            }
            if (cond.comparison == duckdb::ExpressionType::COMPARE_EQUAL) {
                auto& left_bce =
                    cond.left->Cast<duckdb::BoundColumnRefExpression>();
                auto& right_bce =
                    cond.right->Cast<duckdb::BoundColumnRefExpression>();
                probe_keys.push_back(
                    left_col_ref_map[{left_bce.binding.table_index,
                                      left_bce.binding.column_index}]);
                build_keys.push_back(
                    right_col_ref_map[{right_bce.binding.table_index,
                                       right_bce.binding.column_index}]);
            }
        }

        this->cuda_join =
            CudaHashJoin(build_keys, probe_keys, cudf::null_equality::EQUAL);
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

    void FinalizeSink() override { cuda_join.FinalizeBuild(); }

    void FinalizeProcessBatch() override {
        // throw std::runtime_error("Not implemented.");
    }

    /**
     * @brief process input tables to build side of join (populate the hash
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(GPU_DATA input_batch,
                                OperatorResult prev_op_result) override {
        cuda_join.BuildConsumeBatch(input_batch.table);
        return OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief Run join probe on the input batch
     *
     * @param input_batch input batch to probe
     * @return output batch of probe and return flag
     */
    std::pair<GPU_DATA, OperatorResult> ProcessBatch(
        GPU_DATA input_batch, OperatorResult prev_op_result) override {
        std::unique_ptr<cudf::table> output_table =
            cuda_join.ProbeProcessBatch(input_batch.table);
        GPU_DATA output_gpu_data = {std::move(output_table),
                                    this->getOutputSchema()->ToArrowSchema()};
        return {output_gpu_data, prev_op_result};
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

    PhysicalGPUJoinMetrics metrics;

    CudaHashJoin cuda_join;
};
