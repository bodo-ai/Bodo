#pragma once
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

class RuntimeJoinFilterPushdownOptimizer
    : public duckdb::LogicalOperatorVisitor {
   public:
    explicit RuntimeJoinFilterPushdownOptimizer(
        bododuckdb::Optimizer &optimizer);

   private:
    [[maybe_unused]] bododuckdb::Optimizer &optimizer;
};
