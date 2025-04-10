#include "_physical_conv.h"
#include <stdexcept>
#include "_plan.h"

#include "physical/project.h"

void PhysicalPlanBuilder::Visit(duckdb::LogicalGet& op) {
    auto physical_op =
        op.bind_data->Cast<BodoScanFunctionData>().CreatePhysicalOperator();
    if (this->active_pipeline != nullptr) {
        throw std::runtime_error(
            "LogicalGet operator should be the first operator in the pipeline");
    }
    this->active_pipeline = std::make_shared<PipelineBuilder>(physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalProjection& op) {
    // Process the source of this projection.
    this->Visit(*op.children[0]);

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
