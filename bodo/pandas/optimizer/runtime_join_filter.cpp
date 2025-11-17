#include "optimizer/runtime_join_filter.h"
#include <iostream>
#include "_plan.h"
#include "_util.h"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"

RuntimeJoinFilterPushdownOptimizer::RuntimeJoinFilterPushdownOptimizer(
    bododuckdb::Optimizer &_optimizer)
    : duckdb::LogicalOperatorVisitor(), optimizer(_optimizer) {}

void RuntimeJoinFilterPushdownOptimizer::VisitOperator(
    bododuckdb::LogicalOperator &op) {
    if (op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
        duckdb::LogicalComparisonJoin &join_op =
            op.Cast<duckdb::LogicalComparisonJoin>();
        if (join_op.join_type == duckdb::JoinType::RIGHT ||
            join_op.join_type == duckdb::JoinType::INNER) {
            // Insert runtime join filter
            std::vector<int64_t> left_eq_cols;
            auto colref_map = getColRefMap(op.GetColumnBindings());
            for (duckdb::JoinCondition &cond : join_op.conditions) {
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
                    // auto &right_colref =
                    // cond.right->Cast<duckdb::BoundColumnRefExpression>();
                    left_eq_cols.push_back(
                        colref_map[{left_colref.binding.table_index,
                                    left_colref.binding.column_index}]);
                } else {
                    // Only support simple column references for now
                    continue;
                }
            }

            join_op.join_id = this->cur_join_filter_id++;
            std::cout << "Inserting join filter nodes for join id "
                      << join_op.join_id << " with " << left_eq_cols.size()
                      << " equality columns." << std::endl;

            auto left_join_filter = make_join_filter(
                join_op.children[0],
                /*filter_ids=*/{join_op.join_id},
                /*filter_columns=*/{left_eq_cols},  // TODO: support multiple
                                                    // join keys
                                                    /*is_first_locations=*/
                {std::vector<bool>(true, left_eq_cols.size())}
                // TODO: support multiple join keys
            );

            join_op.children[0] = std::move(left_join_filter);
        }
    }
    duckdb::LogicalOperatorVisitor::VisitOperator(op);
}
