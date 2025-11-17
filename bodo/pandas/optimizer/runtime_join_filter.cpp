#include "optimizer/runtime_join_filter.h"

RuntimeJoinFilterPushdownOptimizer::RuntimeJoinFilterPushdownOptimizer(
    bododuckdb::Optimizer &_optimizer)
    : duckdb::LogicalOperatorVisitor(), optimizer(_optimizer) {}
