#include "optimizer/runtime_join_filter.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include "_plan.h"
#include "_util.h"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

// RuntimeJoinFilterPushdownOptimizer::RuntimeJoinFilterPushdownOptimizer(
//     duckdb::Optimizer &_optimizer)
//     : optimizer(_optimizer) {}

duckdb::unique_ptr<duckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::VisitOperator(
    duckdb::unique_ptr<duckdb::LogicalOperator> &op) {
    switch (op->type) {
        case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
            return this->VisitCompJoin(op);
        } break;
        case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
            op = this->VisitProjection(op);
        } break;
        default: {
            // If we don't know how to handle this operator, insert any pending
            // join filters and clear the state
            op = this->insert_join_filters(op, this->join_state_map);
            this->join_state_map.clear();
        }
    }
    for (auto &i : op->children) {
        i = VisitOperator(i);
    }
    return std::move(op);
}

duckdb::unique_ptr<duckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::insert_join_filters(
    duckdb::unique_ptr<duckdb::LogicalOperator> &op,
    JoinFilterProgramState &join_state_map) {
    if (this->join_state_map.size() == 0) {
        return std::move(op);
    }
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

duckdb::unique_ptr<duckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::VisitCompJoin(
    duckdb::unique_ptr<duckdb::LogicalOperator> &op) {
    JoinFilterProgramState out_join_state_map;
    JoinFilterProgramState left_join_state_map;
    JoinFilterProgramState right_join_state_map;
    auto &join_op = op->Cast<duckdb::LogicalComparisonJoin>();

    const size_t numLeftColumns = op->children[0]->GetColumnBindings().size();

    std::vector<int64_t> left_key_indices;
    std::vector<int64_t> right_key_indices;
    for (const auto &cond : join_op.conditions) {
        if (cond.left->GetExpressionType() ==
                duckdb::ExpressionType::BOUND_COLUMN_REF &&
            cond.right->GetExpressionType() ==
                duckdb::ExpressionType::BOUND_COLUMN_REF) {
            auto &left_colref =
                cond.left->Cast<duckdb::BoundColumnRefExpression>();
            auto &right_colref =
                cond.right->Cast<duckdb::BoundColumnRefExpression>();
            left_key_indices.push_back(left_colref.binding.column_index);
            right_key_indices.push_back(right_colref.binding.column_index);
        }
    }

    for (const auto &[join_id, join_info] : this->join_state_map) {
        std::vector<int64_t> leftKeys;
        std::vector<bool> leftIsFirstLocations;
        std::vector<int64_t> rightKeys;
        std::vector<bool> rightIsFirstLocations;
        for (const int64_t &col_idx : join_info.filter_columns) {
            if (col_idx == -1) {
                // Column is -1 so propagate, these filters were already
                // evaluated
                leftKeys.push_back(-1);
                leftIsFirstLocations.push_back(false);
                rightKeys.push_back(-1);
                rightIsFirstLocations.push_back(false);
            } else if (col_idx < numLeftColumns) {
                // Column is on the left
                leftKeys.push_back(col_idx);
                leftIsFirstLocations.push_back(true);

                // If the column is a join key, push to right side
                if (int64_t key_idx =
                        std::ranges::find(left_key_indices, col_idx) !=
                        left_key_indices.end()) {
                    rightKeys.push_back(right_key_indices[key_idx]);
                    rightIsFirstLocations.push_back(true);
                } else {
                    rightKeys.push_back(-1);
                    rightIsFirstLocations.push_back(false);
                }
            } else {
                // Column is on the right
                int64_t right_col_idx = col_idx - numLeftColumns;
                rightKeys.push_back(right_col_idx);
                rightIsFirstLocations.push_back(true);

                // If the column is a join key, push to left side
                if (int64_t key_idx =
                        std::ranges::find(right_key_indices, right_col_idx) !=
                        right_key_indices.end()) {
                    leftKeys.push_back(left_key_indices[key_idx]);
                    leftIsFirstLocations.push_back(true);
                } else {
                    leftKeys.push_back(-1);
                    leftIsFirstLocations.push_back(false);
                }
            }
        }
        bool keep_left_equality =
            std::ranges::find(leftIsFirstLocations, true) !=
            leftIsFirstLocations.end();
        bool keep_right_equality =
            std::ranges::find(rightIsFirstLocations, true) !=
            rightIsFirstLocations.end();

        if (keep_left_equality) {
            left_join_state_map[join_id] = {
                .filter_columns = leftKeys,
                .is_first_locations = leftIsFirstLocations};
        }
        if (keep_right_equality) {
            right_join_state_map[join_id] = {
                .filter_columns = rightKeys,
                .is_first_locations = rightIsFirstLocations};
        }
        if (keep_left_equality && keep_right_equality) {
            bool all_cols = std::ranges::all_of(join_info.is_first_locations,
                                                [](bool v) { return v; });
            if (all_cols) {
                bool all_left = std::ranges::all_of(leftIsFirstLocations,
                                                    [](bool v) { return v; });
                bool all_right = std::ranges::all_of(rightIsFirstLocations,
                                                     [](bool v) { return v; });
                if (!all_left && !all_right) {
                    // Both sides dropped some columns so insert a filter on top
                    // of this join to apply the bloom filter since the the
                    // bloom filter requires all columns
                    JoinColumnInfo join_info_copy = join_info;
                    join_info_copy.is_first_locations = std::vector<bool>(
                        false, join_info.filter_columns.size());
                    out_join_state_map[join_id] = join_info_copy;
                }
            }
        }
    }

    if (join_op.join_type == duckdb::JoinType::RIGHT ||
        join_op.join_type == duckdb::JoinType::INNER) {
        // Insert runtime join filter on the probe side
        std::vector<int64_t> left_eq_cols;
        for (duckdb::JoinCondition &cond : join_op.conditions) {
            // TODO Support non-equalities for pushing to I/O
            // e.g. if T1.A < T2.B is a join conditition we can push A < the
            // minimum of B to T1's scan operator
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
                continue;
            }
        }

        join_op.join_id = this->cur_join_filter_id++;
        left_join_state_map[join_op.join_id] = {
            .filter_columns = left_eq_cols,
            .is_first_locations = std::vector<bool>(true, left_eq_cols.size())};
    }
    this->join_state_map = left_join_state_map;
    // Visit probe side
    op->children[0] = VisitOperator(op->children[0]);
    this->join_state_map = right_join_state_map;
    // Visit build side
    op->children[1] = VisitOperator(op->children[1]);
    return this->insert_join_filters(op, out_join_state_map);
}

duckdb::unique_ptr<duckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::VisitProjection(
    duckdb::unique_ptr<duckdb::LogicalOperator> &op) {
    duckdb::LogicalProjection &proj_op = op->Cast<duckdb::LogicalProjection>();
    // Remap the columns from join_state through the projection
    // then propagate down
    JoinFilterProgramState new_join_state_map;
    for (const auto &[join_id, join_info] : this->join_state_map) {
        std::vector<int64_t> new_filter_columns;
        std::vector<bool> new_is_first_locations;
        for (long long col_idx : join_info.filter_columns) {
            if (col_idx == -1) {
                new_filter_columns.push_back(-1);
                new_is_first_locations.push_back(false);
                continue;
            }
            auto &expr = proj_op.expressions[col_idx];
            if (expr->GetExpressionType() ==
                duckdb::ExpressionType::BOUND_COLUMN_REF) {
                auto &colref_expr =
                    expr->Cast<duckdb::BoundColumnRefExpression>();
                new_filter_columns.push_back(colref_expr.binding.column_index);
                new_is_first_locations.push_back(true);
            } else {
                // Projection expression is not a column ref, cannot push down
                new_filter_columns.push_back(-1);
                new_is_first_locations.push_back(false);
            }
        }
        new_join_state_map[join_id] = {
            .filter_columns = new_filter_columns,
            .is_first_locations = new_is_first_locations};
    }
    this->join_state_map = new_join_state_map;

    return std::move(op);
}
