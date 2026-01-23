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
#include "expression.h"
#include "operator.h"
#include "project_utils.h"

struct PhysicalProjectionMetrics {
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
class PhysicalProjection : public PhysicalProcessBatch {
   public:
    explicit PhysicalProjection(
        std::vector<duckdb::ColumnBinding>& source_cols,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs,
        std::shared_ptr<bodo::Schema> input_schema) {
        time_pt start_init_time = start_timer();
        this->output_schema = getProjectionOutputSchema(
            source_cols, exprs, input_schema, col_names, physical_exprs,
            [](duckdb::unique_ptr<duckdb::Expression>& expr,
               std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
                   col_ref_map) {
                return buildPhysicalExprTree(expr, col_ref_map, true);
            });
        this->metrics.init_time = end_timer(start_init_time);
        this->metrics.n_exprs = this->physical_exprs.size();
    }

    virtual ~PhysicalProjection() = default;

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
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) override {
        std::vector<std::shared_ptr<array_info>> out_cols;

        time_pt start_process_exprs = start_timer();
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
        this->metrics.expr_eval_time += end_timer(start_process_exprs);

        uint64_t out_size =
            out_cols.size() > 0 ? out_cols[0]->length : input_batch->nrows();
        if (out_size != input_batch->nrows()) {
            throw std::runtime_error(
                "Output size does not match input size in Projection");
        }

        std::shared_ptr<table_info> out_table_info =
            std::make_shared<table_info>(out_cols, out_size, col_names,
                                         input_batch->metadata);

        this->metrics.output_row_count += out_size;

        out_table_info->column_names = this->output_schema->column_names;
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
    PhysicalProjectionMetrics metrics;
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
