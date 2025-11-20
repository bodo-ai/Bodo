#include "optimizer/runtime_join_filter.h"
#include <iostream>
#include "_plan.h"
#include "_util.h"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"

RuntimeJoinFilterPushdownOptimizer::RuntimeJoinFilterPushdownOptimizer(
    bododuckdb::Optimizer &_optimizer)
    : optimizer(_optimizer) {}

bododuckdb::unique_ptr<bododuckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::VisitOperator(
    bododuckdb::unique_ptr<bododuckdb::LogicalOperator> &op) {
    if (op->type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
        duckdb::LogicalComparisonJoin &join_op =
            op->Cast<duckdb::LogicalComparisonJoin>();
        JoinFilterProgramState old_join_state_map = this->join_state_map;
        JoinFilterProgramState left_join_state_map = this->join_state_map;
        JoinFilterProgramState right_join_state_map = this->join_state_map;
        if (join_op.join_type == duckdb::JoinType::RIGHT ||
            join_op.join_type == duckdb::JoinType::INNER) {
            // Insert runtime join filter
            std::vector<int64_t> left_eq_cols;
            for (duckdb::JoinCondition &cond : join_op.conditions) {
                // TODO Support non-equalities for pushing to I/O
                // e.g. if T1.A < T2.B we can push A < the minimum of B to
                // T1's scan operator
                if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL) {
                    // Only support equality join conditions for now
                    continue;
                }
                if (cond.left->GetExpressionType() ==
                        duckdb::ExpressionType::BOUND_COLUMN_REF &&
                    cond.right->GetExpressionType() ==
                        duckdb::ExpressionType::BOUND_COLUMN_REF) {
                    auto &left_colref =
                        cond.left->Cast<duckdb::BoundColumnRefExpression>();
                    left_eq_cols.push_back(left_colref.binding.column_index);
                } else {
                    // Only support simple column references for now
                    continue;
                }
            }

            join_op.join_id = this->cur_join_filter_id++;
            left_join_state_map[join_op.join_id] = {
                .filter_columns = left_eq_cols,
                .is_first_locations =
                    std::vector<bool>(true, left_eq_cols.size())};
            this->join_state_map = left_join_state_map;
            // Visit probe side
            op->children[0] = VisitOperator(op->children[0]);
            this->join_state_map = right_join_state_map;
            // Visit build side
            op->children[1] = VisitOperator(op->children[1]);
            this->join_state_map = old_join_state_map;
            return std::move(op);
        }
    } else {
        if (this->join_state_map.size() > 0) {
            op = this->insert_join_filters(op, this->join_state_map);
            this->join_state_map.clear();
        }
    }
    for (auto &i : op->children) {
        i = VisitOperator(i);
    }
    return std::move(op);
}

bododuckdb::unique_ptr<bododuckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::insert_join_filters(
    bododuckdb::unique_ptr<bododuckdb::LogicalOperator> &op,
    JoinFilterProgramState &join_state_map) {
    std::vector<int> filter_ids;
    std::vector<std::vector<int64_t>> filter_columns;
    std::vector<std::vector<bool>> is_first_locations;
    for (const auto &[join_id, join_info] : join_state_map) {
        filter_ids.push_back(join_id);
        filter_columns.push_back(join_info.filter_columns);
        is_first_locations.push_back(join_info.is_first_locations);
    }
    auto join_filter =
        make_join_filter(op,
                         /*filter_ids=*/filter_ids,
                         /*filter_columns=*/filter_columns,
                         /*is_first_locations=*/is_first_locations);
    // Replace current operator with join filter
    return std::move(join_filter);
}
