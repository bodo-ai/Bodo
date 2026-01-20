#pragma once

#include <arrow/type.h>
#include <utility>
#include "../../libs/_bodo_to_arrow.h"
#include "../../libs/_query_profile_collector.h"
#include "../_util.h"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "gpu_expression.h"
#include "operator.h"

struct PhysicalGPUProjectionMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    stat_t output_row_count = 0;
    stat_t n_exprs = 0;

    time_t init_time = 0;
    time_t expr_eval_time = 0;
};

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalGPUProjection : public PhysicalGPUProcessBatch {
   public:
    explicit PhysicalGPUProjection(
        std::vector<duckdb::ColumnBinding>& source_cols,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs,
        std::shared_ptr<bodo::Schema> input_schema) {
        time_pt start_init_time = start_timer();
        // Map of column bindings to column indices in physical input table
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map =
            getColRefMap(source_cols);

        // Create the output schema from expressions
        this->output_schema = std::make_shared<bodo::Schema>();
        for (size_t expr_idx = 0; expr_idx < exprs.size(); ++expr_idx) {
            auto& expr = exprs[expr_idx];
            physical_exprs.emplace_back(
                buildPhysicalGPUExprTree(expr, col_ref_map, true));

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
                    BodoScalarFunctionData& scalar_func_data =
                        func_expr.bind_info->Cast<BodoScalarFunctionData>();

                    std::unique_ptr<bodo::DataType> col_type =
                        bodo::Schema::FromArrowSchema(
                            scalar_func_data.out_schema)
                            ->column_types[0]
                            ->copy();
                    this->output_schema->append_column(std::move(col_type));
                    col_names.emplace_back(
                        scalar_func_data.out_schema->field(0)->name());
                } else {
                    if (func_expr.function.name == "floor") {
                        this->output_schema->append_column(
                            input_schema->column_types[expr_idx]->copy());
                        col_names.emplace_back("floor");
                    } else {
                        // Will use types from LogicalProjection here
                        // eventually.
                        throw std::runtime_error(
                            "Unsupported bound_function in projection " +
                            expr->ToString());
                    }
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
            } else if (expr->type == duckdb::ExpressionType::CASE_EXPR) {
                auto& case_expr = expr->Cast<duckdb::BoundCaseExpression>();

                // TODO(ehsan): cast to common type of inputs if needed to match
                // compute?
                std::unique_ptr<bodo::DataType> col_type =
                    arrow_type_to_bodo_data_type(
                        duckdbTypeToArrow(case_expr.return_type))
                        ->copy();
                this->output_schema->append_column(std::move(col_type));
                if (input_schema->column_names.size() > 0) {
                    col_names.emplace_back(input_schema->column_names[0]);
                } else {
                    col_names.emplace_back("Case");
                }
            } else if (expr->type == duckdb::ExpressionType::OPERATOR_CAST) {
                auto& bce = expr->Cast<duckdb::BoundCastExpression>();

                std::unique_ptr<bodo::DataType> col_type =
                    arrow_type_to_bodo_data_type(
                        duckdbTypeToArrow(bce.return_type))
                        ->copy();
                this->output_schema->append_column(std::move(col_type));
                if (input_schema->column_names.size() > 0) {
                    col_names.emplace_back(input_schema->column_names[0]);
                } else {
                    col_names.emplace_back("Cast");
                }
            } else if (expr->type == duckdb::ExpressionType::OPERATOR_NOT) {
                std::unique_ptr<bodo::DataType> col_type =
                    arrow_type_to_bodo_data_type(
                        duckdbTypeToArrow(duckdb::LogicalType::BOOLEAN))
                        ->copy();
                this->output_schema->append_column(std::move(col_type));
                if (input_schema->column_names.size() > 0) {
                    col_names.emplace_back(input_schema->column_names[0]);
                } else {
                    col_names.emplace_back("Not");
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
        this->metrics.init_time = end_timer(start_init_time);
        this->metrics.n_exprs = this->physical_exprs.size();
        arrow_output_schema = this->output_schema->ToArrowSchema();
    }

    virtual ~PhysicalGPUProjection() = default;

    void FinalizeProcessBatch() override {
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().SubmitOperatorName(getOpId(),
                                                            ToString());
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 0),
            this->metrics.init_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.expr_eval_time);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.output_row_count);
    }

    /**
     * @brief Do projection.
     *
     * @return std::pair<std::shared_ptr<table_info>, OperatorResult>
     * The output table from the current operation and whether there is more
     * output.
     */
    std::pair<GPU_DATA, OperatorResult> ProcessBatch(
        GPU_DATA input_batch, OperatorResult prev_op_result) override {
        std::vector<GPU_COLUMN> out_cols;

        time_pt start_process_exprs = start_timer();
        for (auto& phys_expr : this->physical_exprs) {
            std::shared_ptr<ExprGPUResult> phys_res =
                phys_expr->ProcessBatch(input_batch);
            std::shared_ptr<ArrayExprGPUResult> res_as_array =
                std::dynamic_pointer_cast<ArrayExprGPUResult>(phys_res);
            if (!res_as_array) {
                throw std::runtime_error(
                    "Expression in projection did not result in an array");
            }
            out_cols.emplace_back(std::move(res_as_array->result));
        }
        this->metrics.expr_eval_time += end_timer(start_process_exprs);

        uint64_t out_size = out_cols.size() > 0 ? out_cols[0]->size()
                                                : input_batch.table->num_rows();
        if (out_size != (uint64_t)input_batch.table->num_rows()) {
            throw std::runtime_error(
                "Output size does not match input size in Projection");
        }

        auto out_table = std::make_unique<cudf::table>(std::move(out_cols));
        GPU_DATA out_table_info(std::move(out_table), arrow_output_schema);

        this->metrics.output_row_count += out_size;

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
    std::shared_ptr<arrow::Schema> arrow_output_schema;
    std::vector<std::string> col_names;
    std::vector<std::shared_ptr<PhysicalGPUExpression>> physical_exprs;
    PhysicalGPUProjectionMetrics metrics;
    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        metrics_out.emplace_back(StatMetric("n_exprs", this->metrics.n_exprs));
        metrics_out.emplace_back(
            TimerMetric("init_time", this->metrics.init_time));
        metrics_out.emplace_back(
            TimerMetric("expr_eval_time", this->metrics.expr_eval_time));
        for (auto& expr : this->physical_exprs) {
            expr->ReportMetrics(metrics_out);
        }
    }
};
