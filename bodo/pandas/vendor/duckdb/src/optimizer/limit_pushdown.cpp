#include "duckdb/optimizer/limit_pushdown.hpp"

#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

namespace duckdb {

bool LimitPushdown::CanOptimize(duckdb::LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_LIMIT &&
	    op.children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &limit = op.Cast<LogicalLimit>();

		if (limit.offset_val.Type() == LimitNodeType::EXPRESSION_PERCENTAGE ||
		    limit.offset_val.Type() == LimitNodeType::EXPRESSION_VALUE) {
			// Offset cannot be an expression
			return false;
		}

		if (limit.limit_val.Type() == LimitNodeType::CONSTANT_VALUE && limit.limit_val.GetConstantValue() < 8192) {
			// Push down only when limit value is smaller than 8192.
			// when physical_limit is introduced, it will end a parallel pipeline
			// restrict the limit value to be small so that remaining operations run fast without parallelization.
			return true;
		}
	}
	return false;
}

unique_ptr<LogicalOperator> LimitPushdown::Optimize(unique_ptr<LogicalOperator> op) {
	if (CanOptimize(*op)) {
		auto projection = std::move(op->children[0]);
		op->children[0] = std::move(projection->children[0]);
		projection->SetEstimatedCardinality(op->estimated_cardinality);
		projection->children[0] = std::move(op);
		swap(projection, op);
	}
    // Bodo Change: limit_pushdown to data source not previously supported by DuckDB
	if (op->type == LogicalOperatorType::LOGICAL_LIMIT &&
		op->Cast<LogicalLimit>().limit_val.Type() == LimitNodeType::CONSTANT_VALUE &&
	    op->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
	    op->children[0]->Cast<LogicalGet>().function.limit_pushdown && op->children[0]->children.empty()) {
		auto &get = op->children[0]->Cast<LogicalGet>();
		// set limit options
		get.extra_info.limit_val = make_uniq<BoundLimitNode>(std::move(op->Cast<LogicalLimit>().limit_val));
		get.extra_info.offset_val = make_uniq<BoundLimitNode>(std::move(op->Cast<LogicalLimit>().offset_val));
		op = std::move(op->children[0]);
	}
	for (auto &child : op->children) {
		child = Optimize(std::move(child));
	}
	return op;
}

} // namespace duckdb
