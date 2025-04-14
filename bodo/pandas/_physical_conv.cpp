#include "_physical_conv.h"
#include <stdexcept>
#include "_plan.h"

#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "physical/project.h"
#include "physical/run_udf.h"

void PhysicalPlanBuilder::Visit(duckdb::LogicalGet& op) {
    // Get selected columns from LogicalGet to pass to physical
    // operators
    std::vector<int> selected_columns;
    for (auto& ci : op.GetColumnIds()) {
        selected_columns.push_back(ci.GetPrimaryIndex());
    }
    auto physical_op =
        op.bind_data->Cast<BodoScanFunctionData>().CreatePhysicalOperator(
            selected_columns);
    if (this->active_pipeline != nullptr) {
        throw std::runtime_error(
            "LogicalGet operator should be the first operator in the pipeline");
    }
    this->active_pipeline = std::make_shared<PipelineBuilder>(physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalProjection& op) {
    // Process the source of this projection.
    this->Visit(*op.children[0]);

    // Handle UDF execution case
    if (op.expressions.size() == 1 &&
        op.expressions[0]->type == duckdb::ExpressionType::BOUND_FUNCTION) {
        this->active_pipeline->AddOperator(std::make_shared<PhysicalRunUDF>(
            op.expressions[0]
                ->Cast<duckdb::BoundFunctionExpression>()
                .bind_info->Cast<BodoUDFFunctionData>()
                .func));
        return;
    }

    std::vector<int64_t> selected_columns;

    // Convert BoundColumnRefExpressions in LogicalOperator.expresssions field
    // to integer selected columns.
    for (const auto& expr : op.expressions) {
        duckdb::BoundColumnRefExpression& colref =
            expr->Cast<duckdb::BoundColumnRefExpression>();
        selected_columns.push_back(colref.binding.column_index);
    }

    auto physical_op = std::make_shared<PhysicalProjection>(selected_columns);
    this->active_pipeline->AddOperator(physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalFilter& op) {
    throw std::runtime_error(
        "Not supported on the physical side yet: LogicalFilter");
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalComparisonJoin& op) {
    throw std::runtime_error(
        "Not supported on the physical side yet: LogicalComparisonJoin");
}
