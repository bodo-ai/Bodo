#pragma once

#include <utility>
#include "../libs/_array_utils.h"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "operator.h"

/**
 * @brief UDF plan node data to pass around in DuckDB plans in
 * BoundFunctionExpression.
 *
 */
struct BodoPythonScalarFunctionData : public duckdb::FunctionData {
    BodoPythonScalarFunctionData(PyObject* args,
                                 std::shared_ptr<arrow::Schema> out_schema)
        : args(args), out_schema(std::move(out_schema)) {
        Py_INCREF(args);
    }
    ~BodoPythonScalarFunctionData() override { Py_DECREF(args); }
    bool Equals(const FunctionData& other_p) const override {
        const BodoPythonScalarFunctionData& other =
            other_p.Cast<BodoPythonScalarFunctionData>();
        return (other.args == this->args);
    }
    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        return duckdb::make_uniq<BodoPythonScalarFunctionData>(this->args,
                                                               out_schema);
    }

    PyObject* args;
    std::shared_ptr<arrow::Schema> out_schema;
};

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalProjection : public PhysicalSourceSink {
   public:
    explicit PhysicalProjection(
        duckdb::unique_ptr<duckdb::LogicalOperator>& source,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs,
        std::shared_ptr<bodo::Schema> input_schema)
        : exprs(std::move(exprs)) {
        // Initialize map of column bindings to column indices in physical input
        // table.
        std::vector<duckdb::ColumnBinding> source_cols =
            source->GetColumnBindings();
        for (size_t i = 0; i < source_cols.size(); i++) {
            duckdb::ColumnBinding& col = source_cols[i];
            col_ref_map[{col.table_index, col.column_index}] = i;
        }

        // Create the output schema from expressions
        this->output_schema = std::make_shared<bodo::Schema>();
        std::vector<std::string> col_names;
        for (const auto& expr : this->exprs) {
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
                BodoPythonScalarFunctionData& scalar_func_data =
                    func_expr.bind_info->Cast<BodoPythonScalarFunctionData>();

                std::unique_ptr<bodo::DataType> col_type =
                    bodo::Schema::FromArrowSchema(scalar_func_data.out_schema)
                        ->column_types[0]
                        ->copy();
                this->output_schema->append_column(std::move(col_type));
                col_names.emplace_back(
                    scalar_func_data.out_schema->field(0)->name());
            } else {
                throw std::runtime_error(
                    "Unsupported expression type in projection " +
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
        std::vector<std::string> col_names;

        for (const auto& expr : this->exprs) {
            if (expr->type == duckdb::ExpressionType::BOUND_COLUMN_REF) {
                auto& colref = expr->Cast<duckdb::BoundColumnRefExpression>();
                size_t col_idx = col_ref_map[{colref.binding.table_index,
                                              colref.binding.column_index}];
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
                        size_t col_idx =
                            col_ref_map[{colref.binding.table_index,
                                         colref.binding.column_index}];
                        selected_columns.emplace_back(col_idx);
                    } else {
                        throw std::runtime_error(
                            "Unsupported expression type in function input " +
                            child_expr->ToString());
                    }
                }
                const std::shared_ptr<arrow::DataType>& result_type =
                    scalar_func_data.out_schema->field(0)->type();
                std::shared_ptr<table_info> udf_input =
                    ProjectTable(input_batch, selected_columns);
                std::shared_ptr<table_info> udf_output =
                    runPythonScalarFunction(udf_input, result_type,
                                            scalar_func_data.args);
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
    /**
     * @brief Run Python scalar function on the input batch and return the
     * output table (single data column plus Index columns).
     *
     * @param input_batch input table batch
     * @param result_type The expected result type of the function
     * @param args Python arguments for the function
     * @return std::shared_ptr<table_info> output table from the Python function
     */
    static std::shared_ptr<table_info> runPythonScalarFunction(
        std::shared_ptr<table_info> input_batch,
        const std::shared_ptr<arrow::DataType>& result_type, PyObject* args);

    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
    std::shared_ptr<bodo::Schema> output_schema;

    // Map of column bindings to column indices in physical input table
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map;
};
