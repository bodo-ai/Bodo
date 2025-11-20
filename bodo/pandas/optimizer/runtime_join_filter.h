#pragma once
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

struct JoinColumnInfo {
    std::vector<int64_t> filter_columns;
    std::vector<bool> is_first_locations;
};

class RuntimeJoinFilterPushdownOptimizer {
   public:
    // explicit RuntimeJoinFilterPushdownOptimizer(
    //     duckdb::Optimizer &optimizer);
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitOperator(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);

   private:
    //[[maybe_unused]] duckdb::Optimizer &optimizer;
    size_t cur_join_filter_id = 0;
    using JoinFilterProgramState = std::unordered_map<int, JoinColumnInfo>;
    JoinFilterProgramState join_state_map;
    duckdb::unique_ptr<duckdb::LogicalOperator> insert_join_filters(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op,
        JoinFilterProgramState &join_state_map);

    duckdb::unique_ptr<duckdb::LogicalOperator> VisitCompJoin(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitProjection(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);
};
