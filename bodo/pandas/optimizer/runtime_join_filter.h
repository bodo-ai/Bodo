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
    explicit RuntimeJoinFilterPushdownOptimizer(
        bododuckdb::Optimizer &optimizer);
    bododuckdb::unique_ptr<bododuckdb::LogicalOperator> VisitOperator(
        bododuckdb::unique_ptr<bododuckdb::LogicalOperator> &op);

   private:
    [[maybe_unused]] bododuckdb::Optimizer &optimizer;
    size_t cur_join_filter_id = 0;
    using JoinFilterProgramState = std::unordered_map<int, JoinColumnInfo>;
    JoinFilterProgramState join_state_map;
    bododuckdb::unique_ptr<bododuckdb::LogicalOperator> insert_join_filters(
        bododuckdb::unique_ptr<bododuckdb::LogicalOperator> &op,
        JoinFilterProgramState &join_state_map);

    bododuckdb::unique_ptr<bododuckdb::LogicalOperator> VisitCompJoin(
        bododuckdb::unique_ptr<bododuckdb::LogicalOperator> &op);
};
