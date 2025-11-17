#pragma once
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

class RuntimeJoinFilterPushdownOptimizer
    : public duckdb::LogicalOperatorVisitor {
   public:
    explicit RuntimeJoinFilterPushdownOptimizer(
        bododuckdb::Optimizer &optimizer);
    void VisitOperator(bododuckdb::LogicalOperator &op) override;

   private:
    [[maybe_unused]] bododuckdb::Optimizer &optimizer;
    size_t cur_join_filter_id = 0;
};
