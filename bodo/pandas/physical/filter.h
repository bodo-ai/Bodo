#pragma once

#include <memory>
#include <utility>
#include "../../libs/_query_profile_collector.h"
#include "../libs/_array_utils.h"
#include "expression.h"
#include "operator.h"

struct PhysicalFilterMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t output_row_count = 0;
    stat_t input_row_count = 0;
    time_t expr_eval_time = 0;
    time_t filtering_time = 0;
};
/**
 * @brief Physical node for filter.
 *
 */
class PhysicalFilter : public PhysicalProcessBatch {
   public:
    explicit PhysicalFilter(
        duckdb::LogicalFilter& logical_filter,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs,
        std::shared_ptr<bodo::Schema> input_schema,
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>&
            col_ref_map) {
        this->output_schema = std::make_shared<bodo::Schema>();
        if (logical_filter.projection_map.empty()) {
            for (size_t i = 0; i < input_schema->ncols(); i++) {
                this->kept_cols.push_back(i);
            }
        } else {
            for (const auto& c : logical_filter.projection_map) {
                this->kept_cols.push_back(c);
            }
        }
        for (size_t i = 0; i < this->kept_cols.size(); i++) {
            std::unique_ptr<bodo::DataType> col_type =
                input_schema->column_types[this->kept_cols[i]]->copy();
            this->output_schema->append_column(std::move(col_type));
            this->output_schema->column_names.push_back(
                input_schema->column_names[this->kept_cols[i]]);
        }
        if (this->kept_cols.size() !=
            logical_filter.GetColumnBindings().size()) {
            throw std::runtime_error(
                "Filter output schema has different number of columns than "
                "LogicalFilter");
        }
        this->output_schema->metadata = std::make_shared<bodo::TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));

        expression = buildPhysicalExprTree(exprs[0], col_ref_map);
        for (size_t i = 1; i < exprs.size(); ++i) {
            std::shared_ptr<PhysicalExpression> subExprTree =
                buildPhysicalExprTree(exprs[i], col_ref_map);
            expression = std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalConjunctionExpression>(
                    expression, subExprTree,
                    duckdb::ExpressionType::CONJUNCTION_AND));
        }
    }

    virtual ~PhysicalFilter() = default;

    void FinalizeProcessBatch() override {
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.output_row_count);
    }

    /**
     * @brief The logic for filtering input batches.
     *
     * @param input_batch - the input data to be filtered
     * @return the filtered batch from applying the expression to the input.
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) override {
        this->metrics.input_row_count += input_batch->nrows();
        time_pt start_expr_eval = start_timer();
        // Evaluate the Physical expression tree with the given input batch.
        std::shared_ptr<ExprResult> expr_output =
            expression->ProcessBatch(input_batch);
        // Make sure that the output of the expression tree is a bitmask in
        // the form of a boolean array.
        std::shared_ptr<ArrayExprResult> arr_output =
            std::dynamic_pointer_cast<ArrayExprResult>(expr_output);
        if (!arr_output) {
            throw std::runtime_error(
                "Filter expression tree did not result in an array");
        }
        std::shared_ptr<array_info> bitmask = arr_output->result;
        if (bitmask->dtype != Bodo_CTypes::_BOOL) {
            throw std::runtime_error(
                "Filter expression tree did not result in a boolean array " +
                std::to_string(static_cast<int>(bitmask->dtype)));
        }
        this->metrics.expr_eval_time += end_timer(start_expr_eval);

        time_pt start_filtering = start_timer();
        // Apply the bitmask to the input_batch to do row filtering.
        std::shared_ptr<table_info> filtered_table =
            RetrieveTable(input_batch, bitmask);

        std::vector<std::shared_ptr<array_info>> out_cols;
        for (size_t i = 0; i < this->kept_cols.size(); i++) {
            out_cols.emplace_back(filtered_table->columns[this->kept_cols[i]]);
        }
        std::shared_ptr<table_info> out_table = std::make_shared<table_info>(
            out_cols, filtered_table->nrows(), output_schema->column_names,
            output_schema->metadata);
        this->metrics.filtering_time += end_timer(start_filtering);

        this->metrics.output_row_count += out_table->nrows();

        // Just propagate the FINISHED flag to other operators (like join) or
        // accept more input
        return {out_table, prev_op_result == OperatorResult::FINISHED
                               ? OperatorResult::FINISHED
                               : OperatorResult::NEED_MORE_INPUT};
    }

    /**
     * @brief Get the physical schema of the output data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        // Filter's output schema is the same as the input schema
        return output_schema;
    }

   private:
    std::shared_ptr<PhysicalExpression> expression;
    std::shared_ptr<bodo::Schema> output_schema;
    std::vector<uint64_t> kept_cols;
    PhysicalFilterMetrics metrics;
    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        metrics_out.reserve(3);
        metrics_out.emplace_back(
            TimerMetric("expr_eval_time", this->metrics.expr_eval_time));
        metrics_out.emplace_back(
            TimerMetric("filtering_time", this->metrics.filtering_time));
        MetricBase::BlobValue selectivity_value(
            std::to_string(static_cast<double>(this->metrics.output_row_count) /
                           static_cast<double>(this->metrics.input_row_count)));
        metrics_out.emplace_back(BlobMetric("selectivity", selectivity_value));
    }
};
