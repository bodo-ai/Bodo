#pragma once

#include <arrow/type.h>
#include <utility>
#include "../../libs/_bodo_to_arrow.h"
#include "../_util.h"
#include "../libs/_array_utils.h"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "expression.h"
#include "operator.h"

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalProjection : public PhysicalSourceSink {
   public:
    explicit PhysicalProjection(
        std::vector<duckdb::ColumnBinding>& source_cols,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs,
        std::shared_ptr<bodo::Schema> input_schema) {
        // Map of column bindings to column indices in physical input table
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map =
            getColRefMap(source_cols);

        // Create the output schema from expressions
        this->output_schema = std::make_shared<bodo::Schema>();
        for (auto& expr : exprs) {
            physical_exprs.emplace_back(
                buildPhysicalExprTree(expr, col_ref_map, true));

            if (expr->type == duckdb::ExpressionType::BOUND_COLUMN_REF) {
                auto& colref = expr->Cast<duckdb::BoundColumnRefExpression>();
                size_t col_idx = col_ref_map[{colref.binding.table_index,
                                              colref.binding.column_index}];
                std::unique_ptr<bodo::DataType> col_type =
                    input_schema->column_types[col_idx]->copy();
                this->output_schema->append_column(std::move(col_type));
                if (input_schema->column_names.size() > 0) {
                    col_names.emplace_back(input_schema->column_names[col_idx]);
                } else {
                    col_names.emplace_back(colref.GetName());
                }
            } else if (expr->type == duckdb::ExpressionType::BOUND_FUNCTION) {
                auto& func_expr = expr->Cast<duckdb::BoundFunctionExpression>();
                if (func_expr.bind_info) {
                    BodoPythonScalarFunctionData& scalar_func_data =
                        func_expr.bind_info
                            ->Cast<BodoPythonScalarFunctionData>();
                    std::unique_ptr<bodo::DataType> col_type =
                        bodo::Schema::FromArrowSchema(
                            scalar_func_data.out_schema)
                            ->column_types[0]
                            ->copy();
                    this->output_schema->append_column(std::move(col_type));
                    col_names.emplace_back(
                        scalar_func_data.out_schema->field(0)->name());
                } else {
                    // Will use types from LogicalProjection here eventually.
                    throw std::runtime_error(
                        "Unsupported bound_function in projection " +
                        expr->ToString());
                }
            } else if (expr->type == duckdb::ExpressionType::VALUE_CONSTANT) {
                auto& const_expr =
                    expr->Cast<duckdb::BoundConstantExpression>();

                std::unique_ptr<bodo::DataType> col_type =
                    arrow_type_to_bodo_data_type(
                        duckdbValueToArrowType(const_expr.value))
                        ->copy();
                this->output_schema->append_column(std::move(col_type));
                col_names.emplace_back(const_expr.value.ToString());
            } else if (expr->type == duckdb::ExpressionType::COMPARE_EQUAL ||
                       expr->type == duckdb::ExpressionType::COMPARE_NOTEQUAL ||
                       expr->type == duckdb::ExpressionType::COMPARE_LESSTHAN ||
                       expr->type ==
                           duckdb::ExpressionType::COMPARE_GREATERTHAN ||
                       expr->type ==
                           duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO ||
                       expr->type == duckdb::ExpressionType::
                                         COMPARE_GREATERTHANOREQUALTO) {
                std::unique_ptr<bodo::DataType> col_type =
                    std::make_unique<bodo::DataType>(
                        bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::CTypeEnum::_BOOL);
                this->output_schema->append_column(std::move(col_type));
                if (input_schema->column_names.size() > 0) {
                    col_names.emplace_back(input_schema->column_names[0]);
                } else {
                    col_names.emplace_back("comp");
                }
            } else if (expr->type == duckdb::ExpressionType::CONJUNCTION_AND ||
                       expr->type == duckdb::ExpressionType::CONJUNCTION_OR) {
                std::unique_ptr<bodo::DataType> col_type =
                    std::make_unique<bodo::DataType>(
                        bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::CTypeEnum::_BOOL);
                this->output_schema->append_column(std::move(col_type));
                if (input_schema->column_names.size() > 0) {
                    col_names.emplace_back(input_schema->column_names[0]);
                } else {
                    col_names.emplace_back("conjunction");
                }
            } else {
                throw std::runtime_error(
                    "Unsupported expression type in projection " +
                    std::to_string(static_cast<int>(expr->type)) + " " +
                    expr->ToString());
            }
        }
        this->output_schema->column_names = col_names;
        this->output_schema->metadata = input_schema->metadata;
    }

    virtual ~PhysicalProjection() = default;

    void Finalize() override {}

    /**
     * @brief Do projection.
     *
     * @return std::pair<std::shared_ptr<table_info>, OperatorResult>
     * The output table from the current operation and whether there is more
     * output.
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) override {
        std::vector<std::shared_ptr<array_info>> out_cols;

        for (auto& phys_expr : this->physical_exprs) {
            std::shared_ptr<ExprResult> phys_res =
                phys_expr->ProcessBatch(input_batch);
            std::shared_ptr<ArrayExprResult> res_as_array =
                std::dynamic_pointer_cast<ArrayExprResult>(phys_res);
            if (!res_as_array) {
                throw std::runtime_error(
                    "Expression in projection did not result in an array");
            }
            out_cols.emplace_back(res_as_array->result);
        }

        uint64_t out_size =
            out_cols.size() > 0 ? out_cols[0]->length : input_batch->nrows();
        if (out_size != input_batch->nrows()) {
            throw std::runtime_error(
                "Output size does not match input size in Projection");
        }

        std::shared_ptr<table_info> out_table_info =
            std::make_shared<table_info>(out_cols, out_size, col_names,
                                         input_batch->metadata);

        // Just propagate the FINISHED flag to other operators (like join) or
        // accept more input
        return {out_table_info, prev_op_result == OperatorResult::FINISHED
                                    ? OperatorResult::FINISHED
                                    : OperatorResult::NEED_MORE_INPUT};
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

   private:
    std::shared_ptr<bodo::Schema> output_schema;
    std::vector<std::string> col_names;
    std::vector<std::shared_ptr<PhysicalExpression>> physical_exprs;
};
