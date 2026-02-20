#pragma once

#include <cudf/stream_compaction.hpp>
#include <memory>
#include <utility>
#include "../../libs/_query_profile_collector.h"
#include "../libs/_array_utils.h"
#include "gpu_expression.h"
#include "operator.h"

struct PhysicalGPUFilterMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t output_row_count = 0;
    stat_t input_row_count = 0;
    time_t expr_eval_time = 0;
    time_t filtering_time = 0;
};

inline bool gpu_capable(duckdb::LogicalFilter& logical_filter) {
    for (auto& expr : logical_filter.expressions) {
        if (!gpu_capable(expr)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Physical node for filter.
 *
 */
class PhysicalGPUFilter : public PhysicalGPUProcessBatch {
   public:
    explicit PhysicalGPUFilter(
        duckdb::LogicalFilter& logical_filter,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs,
        std::shared_ptr<bodo::Schema> input_schema,
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>&
            col_ref_map) {
        this->output_schema = std::make_shared<bodo::Schema>();
        if (logical_filter.projection_map.empty()) {
            has_projection_map = false;
            for (size_t i = 0; i < input_schema->ncols(); i++) {
                this->kept_cols.push_back(i);
            }
        } else {
            has_projection_map = true;
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
        arrow_output_schema = this->output_schema->ToArrowSchema();

        expression = buildPhysicalGPUExprTree(exprs[0], col_ref_map);
        for (size_t i = 1; i < exprs.size(); ++i) {
            std::shared_ptr<PhysicalGPUExpression> subExprTree =
                buildPhysicalGPUExprTree(exprs[i], col_ref_map);
            expression = std::static_pointer_cast<PhysicalGPUExpression>(
                std::make_shared<PhysicalGPUConjunctionExpression>(
                    expression, subExprTree,
                    duckdb::ExpressionType::CONJUNCTION_AND,
                    exprs[i]->return_type));
        }
    }

    virtual ~PhysicalGPUFilter() = default;

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
    std::pair<GPU_DATA, OperatorResult> ProcessBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        this->metrics.input_row_count += input_batch.table->num_rows();
        time_pt start_expr_eval = start_timer();
        // Evaluate the Physical expression tree with the given input batch.
        std::shared_ptr<ExprGPUResult> expr_output =
            expression->ProcessBatch(input_batch, se);
        // Make sure that the output of the expression tree is a bitmask in
        // the form of a boolean array.
        std::shared_ptr<ArrayExprGPUResult> arr_output =
            std::dynamic_pointer_cast<ArrayExprGPUResult>(expr_output);
        if (!arr_output) {
            throw std::runtime_error(
                "Filter expression tree did not result in an array");
        }
        GPU_COLUMN bitmask = std::move(arr_output->result);
        if (bitmask->type().id() != cudf::type_id::BOOL8) {
            throw std::runtime_error(
                "Filter expression tree did not result in a boolean array " +
                std::to_string(static_cast<int>(bitmask->type().id())));
        }
        this->metrics.expr_eval_time += end_timer(start_expr_eval);

        time_pt start_filtering = start_timer();
        // Apply the bitmask to the input_batch to do row filtering.
        std::unique_ptr<cudf::table> filtered_table = cudf::apply_boolean_mask(
            input_batch.table->view(), bitmask->view(), se->stream);

        if (has_projection_map) {
            std::vector<GPU_COLUMN> out_cols;
            out_cols.reserve(this->kept_cols.size());
            for (size_t i = 0; i < this->kept_cols.size(); i++) {
                out_cols.push_back(std::make_unique<cudf::column>(
                    filtered_table->get_column(this->kept_cols[i])));
            }
            filtered_table =
                std::make_unique<cudf::table>(std::move(out_cols), se->stream);
        }

        GPU_DATA out_table_info(std::move(filtered_table), arrow_output_schema,
                                se);
        this->metrics.filtering_time += end_timer(start_filtering);

        this->metrics.output_row_count += out_table_info.table->num_rows();

        // Just propagate the FINISHED flag to other operators (like join) or
        // accept more input
        return {out_table_info, prev_op_result == OperatorResult::FINISHED
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
    std::shared_ptr<PhysicalGPUExpression> expression;
    std::shared_ptr<bodo::Schema> output_schema;
    std::shared_ptr<arrow::Schema> arrow_output_schema;
    std::vector<uint64_t> kept_cols;
    PhysicalGPUFilterMetrics metrics;
    bool has_projection_map;
    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        metrics_out.reserve(3);
        metrics_out.emplace_back(
            TimerMetric("expr_eval_time", this->metrics.expr_eval_time));
        metrics_out.emplace_back(
            TimerMetric("filtering_time", this->metrics.filtering_time));
        double selectivity = 0.0;
        if (this->metrics.input_row_count > 0) {
            selectivity = static_cast<double>(this->metrics.output_row_count) /
                          static_cast<double>(this->metrics.input_row_count);
        }
        MetricBase::BlobValue selectivity_value(std::to_string(selectivity));
        metrics_out.emplace_back(BlobMetric("selectivity", selectivity_value));
    }
};
