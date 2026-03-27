#pragma once

#include <arrow/array/util.h>
#include <arrow/compute/expression.h>
#include <arrow/scalar.h>
#include <arrow/type_fwd.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include "../libs/_bodo_to_arrow.h"
#include "../libs/_query_profile_collector.h"
#include "operator.h"
#include "physical/expression.h"

inline bool gpu_capable_reduce(duckdb::LogicalAggregate& logical_aggregate) {
    assert(logical_aggregate.groups.empty());
    for (size_t i = 0; i < logical_aggregate.expressions.size(); i++) {
        const auto& expr = logical_aggregate.expressions[i];

        if (expr->type != duckdb::ExpressionType::BOUND_AGGREGATE) {
            throw std::runtime_error(
                "Aggregate expression is not a bound aggregate: " +
                expr->ToString());
        }
        auto& agg_expr = expr->Cast<duckdb::BoundAggregateExpression>();

        // Only support sum for now
        if (agg_expr.function.name != "sum") {
            return false;
        }
    }
    return true;
}
