#pragma once

#include <utility>
#include "../_plan.h"
#include "../libs/_array_utils.h"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "operator.h"

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalProjection : public PhysicalSourceSink {
   public:
    explicit PhysicalProjection(
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
        : exprs(std::move(exprs)) {}

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
        std::shared_ptr<table_info> input_batch) override {
        std::vector<std::shared_ptr<array_info>> out_cols;
        std::vector<std::string> col_names;

        for (const auto& expr : this->exprs) {
            if (expr->type == duckdb::ExpressionType::BOUND_COLUMN_REF) {
                auto& colref = expr->Cast<duckdb::BoundColumnRefExpression>();
                size_t col_idx = colref.binding.column_index;
                out_cols.emplace_back(input_batch->columns[col_idx]);
                if (input_batch->column_names.size() > 0) {
                    col_names.emplace_back(input_batch->column_names[col_idx]);
                } else {
                    col_names.emplace_back(colref.GetName());
                }
            } else if (expr->type == duckdb::ExpressionType::BOUND_FUNCTION) {
                auto& func_expr = expr->Cast<duckdb::BoundFunctionExpression>();
                BodoPythonScalarFunctionData& scalar_func_data =
                    func_expr.bind_info->Cast<BodoPythonScalarFunctionData>();
                std::vector<int64_t> selected_columns;
                for (const auto& child_expr : func_expr.children) {
                    if (child_expr->type ==
                        duckdb::ExpressionType::BOUND_COLUMN_REF) {
                        auto& colref =
                            child_expr
                                ->Cast<duckdb::BoundColumnRefExpression>();
                        size_t col_idx = colref.binding.column_index;
                        selected_columns.emplace_back(col_idx);
                    } else {
                        throw std::runtime_error(
                            "Unsupported expression type in function input " +
                            child_expr->ToString());
                    }
                }
                std::shared_ptr<table_info> udf_input =
                    ProjectTable(input_batch, selected_columns);
                std::shared_ptr<table_info> udf_output =
                    runPythonScalarFunction(udf_input, scalar_func_data.args);
                // Extracting the data column only assuming Index columns are
                // the same as input and already included as column refs in
                // exprs.
                out_cols.emplace_back(udf_output->columns[0]);
                col_names.emplace_back(func_expr.GetName());
            } else {
                throw std::runtime_error(
                    "Unsupported expression type in projection " +
                    expr->ToString());
            }
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
        return {out_table_info, OperatorResult::NEED_MORE_INPUT};
    }

    std::shared_ptr<bodo::Schema> getOutputSchema() override {
        // TODO
        return nullptr;
    }

   private:
    /**
     * @brief Run Python scalar function on the input batch and return the
     * output table (single data column plus Index columns).
     *
     * @param input_batch input table batch
     * @param args Python arguments for the function
     * @return std::shared_ptr<table_info> output table from the Python function
     */
    static std::shared_ptr<table_info> runPythonScalarFunction(
        std::shared_ptr<table_info> input_batch, PyObject* args);

    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
};
