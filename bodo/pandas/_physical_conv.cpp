#include "_physical_conv.h"
#include <stdexcept>
#include "_plan.h"

#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "physical/filter.h"
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

// Generic dynamic_cast for std::unique_ptr
template <typename Derived, typename Base>
duckdb::unique_ptr<Derived> dynamic_cast_unique_ptr(
    duckdb::unique_ptr<Base>& base_ptr) {
    // Perform dynamic_cast on the raw pointer
    if (Derived* derived_raw = dynamic_cast<Derived*>(base_ptr.get())) {
        // Release ownership from the base_ptr and transfer it to a new
        // unique_ptr
        base_ptr.release();  // Release the ownership of the raw pointer
        return duckdb::unique_ptr<Derived>(derived_raw);
    }
    // If the cast fails, return a nullptr unique_ptr
    return nullptr;
}

std::variant<int, float, double> extractValue(const duckdb::Value& value) {
    duckdb::LogicalTypeId type = value.type().id();
    switch (type) {
        case duckdb::LogicalTypeId::INTEGER:
            return value.GetValue<int>();
        case duckdb::LogicalTypeId::FLOAT:
            return value.GetValue<float>();
        case duckdb::LogicalTypeId::DOUBLE:
            return value.GetValue<double>();
        default:
            throw std::runtime_error("extractValue unhandled type.");
    }
}

std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
    duckdb::unique_ptr<duckdb::Expression>& expr) {
    duckdb::ExpressionClass expr_class = expr->GetExpressionClass();
    duckdb::ExpressionType expr_type = expr->GetExpressionType();

    switch (expr_class) {
        case duckdb::ExpressionClass::BOUND_COMPARISON: {
            auto bce =
                dynamic_cast_unique_ptr<duckdb::BoundComparisonExpression>(
                    expr);
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalComparisonExpression>(
                    buildPhysicalExprTree(bce->left),
                    buildPhysicalExprTree(bce->right), expr_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_COLUMN_REF: {
            auto bce =
                dynamic_cast_unique_ptr<duckdb::BoundColumnRefExpression>(expr);
            auto binding = bce->binding;
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalColumnRefExpression>(
                    binding.table_index, binding.column_index));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CONSTANT: {
            auto bce =
                dynamic_cast_unique_ptr<duckdb::BoundConstantExpression>(expr);
            auto extracted_value = extractValue(bce->value);
            return std::visit(
                [](const auto& value) {
                    return std::static_pointer_cast<PhysicalExpression>(
                        std::make_shared<PhysicalConstantExpression<
                            std::decay_t<decltype(value)>>>(value));
                },
                extracted_value);
        } break;  // suppress wrong fallthrough error
        default:
            throw std::runtime_error(
                "Unsupported duckdb expression type" +
                std::to_string(static_cast<int>(expr_class)));
    }
    throw std::logic_error("Control should never reach here");
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalFilter& op) {
    // Process the source of this projection.
    this->Visit(*op.children[0]);

    if (op.expressions.size() != 1) {
        throw std::runtime_error(
            "LogicalFilter only supports expressions of size 1");
    }

    auto physExprTree = buildPhysicalExprTree(op.expressions[0]);
    auto physical_op = std::make_shared<PhysicalFilter>(physExprTree);
    this->active_pipeline->AddOperator(physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalComparisonJoin& op) {
    throw std::runtime_error(
        "Not supported on the physical side yet: LogicalComparisonJoin");
}
