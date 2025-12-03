#include "optimizer/runtime_join_filter.h"
#include <algorithm>
#include "_plan.h"
#include "_util.h"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"

duckdb::unique_ptr<duckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::VisitOperator(
    duckdb::unique_ptr<duckdb::LogicalOperator> &op) {
    switch (op->type) {
        case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
            return this->VisitCompJoin(op);
        } break;
        case duckdb::LogicalOperatorType::LOGICAL_FILTER: {
            op = this->VisitFilter(op);
        } break;
        case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
            op = this->VisitProjection(op);
        } break;
        case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
            return this->VisitAggregate(op);
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
    if (join_state_map.size() == 0) {
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
    auto left_child_colref_map =
        getColRefMap(join_op.children[0]->GetColumnBindings());
    auto right_child_colref_map =
        getColRefMap(join_op.children[1]->GetColumnBindings());

    std::vector<duckdb::ColumnBinding> left_keys;
    std::vector<duckdb::ColumnBinding> right_keys;
    // Get the column bindings for join equality keys. If a filter column is an
    // equality key, we can push the filter to the other side of the join.
    for (const auto &cond : join_op.conditions) {
        if (cond.comparison == bododuckdb::ExpressionType::COMPARE_EQUAL &&
            cond.left->GetExpressionType() ==
                duckdb::ExpressionType::BOUND_COLUMN_REF &&
            cond.right->GetExpressionType() ==
                duckdb::ExpressionType::BOUND_COLUMN_REF) {
            auto &left_colref =
                cond.left->Cast<duckdb::BoundColumnRefExpression>();
            auto &right_colref =
                cond.right->Cast<duckdb::BoundColumnRefExpression>();
            left_keys.push_back(left_colref.binding);
            right_keys.push_back(right_colref.binding);
        }
    }

    for (const auto &[join_id, join_info] : this->join_state_map) {
        std::vector<int64_t> left_filter_cols;
        std::vector<bool> left_is_first_locations;
        std::vector<int64_t> right_filter_cols;
        std::vector<bool> right_is_first_locations;

        for (const int64_t &col_idx : join_info.filter_columns) {
            // Remap each filter column to left/right child based on column
            // binding
            duckdb::ColumnBinding col_binding =
                col_idx == -1 ? duckdb::ColumnBinding()
                              : join_op.GetColumnBindings()[col_idx];

            // Get the table indices of the left child so we can
            // check whether a column binding belongs to the left or right child
            auto &left_child = *join_op.children[0];
            duckdb::unordered_set<duckdb::idx_t> left_table_indices;
            join_op.GetTableReferences(left_child, left_table_indices);

            if (col_idx == -1) {
                // Column is -1 so propagate, these filters were already
                // evaluated
                left_filter_cols.push_back(-1);
                left_is_first_locations.push_back(false);
                right_filter_cols.push_back(-1);
                right_is_first_locations.push_back(false);
            } else if (left_table_indices.find(col_binding.table_index) !=
                       left_table_indices.end()) {
                // Column is on the left, remap to the left child
                left_filter_cols.push_back(left_child_colref_map[{
                    col_binding.table_index, col_binding.column_index}]);
                left_is_first_locations.push_back(true);

                // If the column is a join key, push to right side
                auto key_iter = std::ranges::find(left_keys, col_binding);
                if (key_iter != left_keys.end()) {
                    // This column is a key so figure out which right key this
                    // matches to
                    int64_t key_idx =
                        std::distance(left_keys.begin(), key_iter);
                    // Remap the right key column onto the right child
                    right_filter_cols.push_back(right_child_colref_map[{
                        right_keys[key_idx].table_index,
                        right_keys[key_idx].column_index}]);
                    right_is_first_locations.push_back(true);
                } else {
                    right_filter_cols.push_back(-1);
                    right_is_first_locations.push_back(false);
                }
            } else {
                // Column is on the right, remap to the right child
                right_filter_cols.push_back(right_child_colref_map[{
                    col_binding.table_index, col_binding.column_index}]);
                right_is_first_locations.push_back(true);

                // If the column is a join key, push to left side
                auto key_iter = std::ranges::find(right_keys, col_binding);
                if (key_iter != right_keys.end()) {
                    // This column is a key so figure out which left key this
                    // matches to
                    int64_t key_idx =
                        std::distance(right_keys.begin(), key_iter);
                    // Remap the left key column onto the left child
                    left_filter_cols.push_back(left_child_colref_map[{
                        left_keys[key_idx].table_index,
                        left_keys[key_idx].column_index}]);
                    left_is_first_locations.push_back(true);
                } else {
                    left_filter_cols.push_back(-1);
                    left_is_first_locations.push_back(false);
                }
            }
        }
        bool keep_left_equality =
            std::ranges::find(left_is_first_locations, true) !=
            left_is_first_locations.end();
        bool keep_right_equality =
            std::ranges::find(right_is_first_locations, true) !=
            right_is_first_locations.end();

        // If we have any live columns, add to the respective child join state
        // map
        if (keep_left_equality) {
            left_join_state_map[join_id] = {
                .filter_columns = left_filter_cols,
                .is_first_locations = left_is_first_locations};
        }
        if (keep_right_equality) {
            right_join_state_map[join_id] = {
                .filter_columns = right_filter_cols,
                .is_first_locations = right_is_first_locations};
        }
        // If we're pushing into both sides, we may need to add a filter on top
        // of the join to evaluate the bloom filter if some columns were dropped
        // on either side since bloom filters require all key columns
        if (keep_left_equality && keep_right_equality) {
            bool all_cols = std::ranges::all_of(join_info.is_first_locations,
                                                [](bool v) { return v; });
            if (all_cols) {
                bool all_left = std::ranges::all_of(left_is_first_locations,
                                                    [](bool v) { return v; });
                bool all_right = std::ranges::all_of(right_is_first_locations,
                                                     [](bool v) { return v; });
                if (!all_left && !all_right) {
                    // Both sides dropped some columns so insert a filter on top
                    // of this join to apply the bloom filter since the the
                    // bloom filter requires all columns
                    if (join_info.filter_columns.size()) {
                        JoinColumnInfo join_info_copy = join_info;
                        // Is first is false since they were remapped into the
                        // children so this isn't the first time the column will
                        // be filtered at a column level
                        join_info_copy.is_first_locations = std::vector<bool>(
                            join_info.filter_columns.size(), false);
                        out_join_state_map[join_id] = join_info_copy;
                    }
                }
            }
        }
    }

    // If we have a right or inner join we can add a join filter on the probe
    // (left) side based on the join equality keys
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
                left_eq_cols.push_back(
                    left_child_colref_map[{left_colref.binding.table_index,
                                           left_colref.binding.column_index}]);
            } else {
                continue;
            }
        }

        join_op.join_id = this->cur_join_filter_id++;
        if (left_eq_cols.size()) {
            left_join_state_map[join_op.join_id] = {
                .filter_columns = left_eq_cols,
                .is_first_locations =
                    std::vector<bool>(left_eq_cols.size(), true)};
        }
    }
    // Visit probe side
    this->join_state_map = left_join_state_map;
    op->children[0] = VisitOperator(op->children[0]);
    // Visit build side
    this->join_state_map = right_join_state_map;
    op->children[1] = VisitOperator(op->children[1]);
    // Insert any join filters needed on top of this join
    return this->insert_join_filters(op, out_join_state_map);
}

duckdb::unique_ptr<duckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::VisitProjection(
    duckdb::unique_ptr<duckdb::LogicalOperator> &op) {
    // Remap the columns from join_state through the projection
    // then propagate down
    JoinFilterProgramState new_join_state_map;
    auto child_colref_map = getColRefMap(op->children[0]->GetColumnBindings());
    for (const auto &[join_id, join_info] : this->join_state_map) {
        std::vector<int64_t> new_filter_columns;
        std::vector<bool> new_is_first_locations;
        for (long long col_idx : join_info.filter_columns) {
            if (col_idx == -1) {
                new_filter_columns.push_back(-1);
                new_is_first_locations.push_back(false);
                continue;
            }
            auto &expr = op->expressions[col_idx];
            if (expr->GetExpressionType() ==
                duckdb::ExpressionType::BOUND_COLUMN_REF) {
                auto &colref_expr =
                    expr->Cast<duckdb::BoundColumnRefExpression>();
                int64_t child_col =
                    child_colref_map[{colref_expr.binding.table_index,
                                      colref_expr.binding.column_index}];
                if (std::ranges::find(new_filter_columns, child_col) ==
                    new_filter_columns.end()) {
                    new_filter_columns.push_back(child_col);
                    new_is_first_locations.push_back(true);
                } else {
                    // Duplicate column reference in projection, cannot push
                    new_filter_columns.push_back(-1);
                    new_is_first_locations.push_back(false);
                }
            } else {
                // Projection expression is not a column ref, cannot push down
                // TODO: Potentiually we could create the inverse expression
                // here and push that down but the cost to evaluate would need
                // to be weighed against the extra filtered rows, probably a
                // hard decision to make correctly
                new_filter_columns.push_back(-1);
                new_is_first_locations.push_back(false);
            }
        }
        if (new_filter_columns.size()) {
            new_join_state_map[join_id] = {
                .filter_columns = new_filter_columns,
                .is_first_locations = new_is_first_locations};
        }
    }
    this->join_state_map = new_join_state_map;

    return std::move(op);
}

duckdb::unique_ptr<duckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::VisitFilter(
    duckdb::unique_ptr<duckdb::LogicalOperator> &op) {
    // Remap the columns from join_state through the filter
    // then propagate down
    JoinFilterProgramState new_join_state_map;
    auto child_colref_map = getColRefMap(op->children[0]->GetColumnBindings());
    for (const auto &[join_id, join_info] : this->join_state_map) {
        std::vector<int64_t> new_filter_columns;
        std::vector<bool> new_is_first_locations;
        for (long long col_idx : join_info.filter_columns) {
            if (col_idx == -1) {
                new_filter_columns.push_back(-1);
                new_is_first_locations.push_back(false);
                continue;
            }
            duckdb::ColumnBinding binding = op->GetColumnBindings()[col_idx];
            int64_t child_col =
                child_colref_map[{binding.table_index, binding.column_index}];
            if (std::ranges::find(new_filter_columns, child_col) ==
                new_filter_columns.end()) {
                new_filter_columns.push_back(child_col);
                new_is_first_locations.push_back(true);
            } else {
                // Duplicate column reference in filter, cannot push
                new_filter_columns.push_back(-1);
                new_is_first_locations.push_back(false);
            }
        }
        if (new_filter_columns.size()) {
            new_join_state_map[join_id] = {
                .filter_columns = new_filter_columns,
                .is_first_locations = new_is_first_locations};
        }
    }
    this->join_state_map = new_join_state_map;

    return std::move(op);
}

duckdb::unique_ptr<duckdb::LogicalOperator>
RuntimeJoinFilterPushdownOptimizer::VisitAggregate(
    duckdb::unique_ptr<duckdb::LogicalOperator> &op) {
    auto &agg_op = op->Cast<duckdb::LogicalAggregate>();
    JoinFilterProgramState new_join_state_map;
    JoinFilterProgramState out_join_state_map;
    auto child_colref_map = getColRefMap(op->children[0]->GetColumnBindings());
    for (const auto &[join_id, join_info] : this->join_state_map) {
        std::vector<int64_t> push_filter_columns;
        std::vector<bool> push_is_first_locations;
        std::vector<int64_t> out_filter_columns;
        std::vector<bool> out_is_first_locations;

        for (long long col_idx : join_info.filter_columns) {
            if (col_idx == -1) {
                push_filter_columns.push_back(-1);
                push_is_first_locations.push_back(false);
                continue;
            }
            size_t group_idx = std::distance(
                agg_op.groups.begin(),
                std::ranges::find(
                    agg_op.groups, agg_op.GetColumnBindings()[col_idx],
                    [](auto &expr) {
                        assert(expr->GetExpressionType() ==
                               duckdb::ExpressionType::BOUND_COLUMN_REF);
                        auto &colref_expr = expr->template Cast<
                            duckdb::BoundColumnRefExpression>();
                        return colref_expr.binding;
                    }));

            if (group_idx < agg_op.groups.size()) {
                auto &expr = agg_op.groups[group_idx];
                assert(expr->GetExpressionType() ==
                       duckdb::ExpressionType::BOUND_COLUMN_REF);
                auto &colref_expr =
                    expr->template Cast<duckdb::BoundColumnRefExpression>();
                int64_t child_col =
                    child_colref_map[{colref_expr.binding.table_index,
                                      colref_expr.binding.column_index}];
                if (std::ranges::find(push_filter_columns, child_col) ==
                    push_filter_columns.end()) {
                    push_filter_columns.push_back(child_col);
                    push_is_first_locations.push_back(true);
                } else {
                    // Duplicate column reference in projection, cannot push
                    push_filter_columns.push_back(-1);
                    push_is_first_locations.push_back(false);
                }
            } else {
                // Aggregate expression is not a group key, cannot push down
                out_filter_columns.push_back(col_idx);
                out_is_first_locations.push_back(false);
            }
        }
        if (push_filter_columns.size()) {
            new_join_state_map[join_id] = {
                .filter_columns = push_filter_columns,
                .is_first_locations = push_is_first_locations};
        }
        if (out_filter_columns.size()) {
            out_join_state_map[join_id] = {
                .filter_columns = out_filter_columns,
                .is_first_locations = out_is_first_locations};
        }
    }

    this->join_state_map = new_join_state_map;
    op->children[0] = VisitOperator(op->children[0]);

    return this->insert_join_filters(op, out_join_state_map);
}
