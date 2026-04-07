#pragma once

#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include "../../libs/streaming/cuda_sort.h"
#include "../_util.h"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "operator.h"

#ifdef USE_CUDF

inline bool gpu_capable(duckdb::LogicalOrder& op) { return true; }

inline bool gpu_capable(duckdb::LogicalTopN& op) { return true; }

struct PhysicalGPUSortMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    time_t init_time = 0;
    time_t consume_time = 0;
    time_t produce_time = 0;

    stat_t output_row_count = 0;
};

/**
 * @brief Physical node for GPU sort.
 *
 */
class PhysicalGPUSortOperator : public PhysicalGPUSource,
                                public PhysicalGPUSink {
   private:
    explicit PhysicalGPUSortOperator(
        duckdb::vector<duckdb::BoundOrderByNode>& orders,
        std::shared_ptr<bodo::Schema> input_schema,
        std::vector<duckdb::ColumnBinding>& source_cols, int64_t limit,
        int64_t offset, unsigned node_cols,
        const std::vector<duckdb::idx_t>& projection_map = {}) {
        time_pt start_init = start_timer();

        if (limit != -1 || offset != -1) {
            throw std::runtime_error(
                "PhysicalGPUSortOperator: limit and offset are not yet "
                "supported.");
        }

        // Calculate the output schema and kept columns.
        this->output_schema = std::make_shared<bodo::Schema>();
        if (projection_map.empty()) {
            for (size_t i = 0; i < input_schema->ncols(); i++) {
                this->kept_cols.push_back(i);
            }
        } else {
            for (const auto& c : projection_map) {
                this->kept_cols.push_back(c);
            }
        }
        for (long kept_col : this->kept_cols) {
            std::unique_ptr<bodo::DataType> col_type =
                input_schema->column_types[kept_col]->copy();
            this->output_schema->append_column(std::move(col_type));
            this->output_schema->column_names.push_back(
                input_schema->column_names[kept_col]);
        }
        if (this->kept_cols.size() != node_cols) {
            throw std::runtime_error(
                "Sort output schema has different number of columns than "
                "LogicalOrder or LogicalTopN");
        }
        this->output_schema->metadata = std::make_shared<bodo::TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));
        this->arrow_output_schema = this->output_schema->ToArrowSchema();

        std::vector<cudf::order> column_order;
        std::vector<cudf::null_order> null_precedence;
        std::vector<cudf::size_type> key_indices;
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map;
        col_ref_map = getColRefMap(source_cols);

        // Convert BoundOrderByNode's to keys, order, and null precedence.
        for (const auto& order : orders) {
            switch (order.type) {
                case duckdb::OrderType::ASCENDING:
                    column_order.push_back(cudf::order::ASCENDING);
                    break;
                case duckdb::OrderType::DESCENDING:
                    column_order.push_back(cudf::order::DESCENDING);
                    break;
                default:
                    throw std::runtime_error(
                        "PhysicalGPUSortOperator order type not ascending or "
                        "descending.");
            }
            switch (order.null_order) {
                case duckdb::OrderByNullType::NULLS_FIRST:
                    null_precedence.push_back(cudf::null_order::BEFORE);
                    break;
                case duckdb::OrderByNullType::NULLS_LAST:
                    null_precedence.push_back(cudf::null_order::AFTER);
                    break;
                default:
                    throw std::runtime_error(
                        "PhysicalGPUSortOperator orderbynull type not first or "
                        "last.");
            }
            duckdb::ExpressionClass expr_class =
                order.expression->GetExpressionClass();
            if (expr_class != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "PhysicalGPUSortOperator expression is not column ref.");
            }
            auto& bce =
                order.expression->Cast<duckdb::BoundColumnRefExpression>();
            key_indices.push_back(static_cast<cudf::size_type>(col_ref_map[{
                bce.binding.table_index, bce.binding.column_index}]));
        }

        this->cuda_sort_state = std::make_unique<CudaSortState>(
            input_schema, key_indices, column_order, null_precedence);

        this->metrics.init_time = end_timer(start_init);
    }

   public:
    explicit PhysicalGPUSortOperator(
        duckdb::LogicalOrder& logical_order,
        std::shared_ptr<bodo::Schema> input_schema,
        std::vector<duckdb::ColumnBinding>& source_cols, int64_t limit = -1,
        int64_t offset = -1)
        : PhysicalGPUSortOperator(logical_order.orders, input_schema,
                                  source_cols, limit, offset,
                                  logical_order.GetColumnBindings().size(),
                                  logical_order.projection_map) {}

    explicit PhysicalGPUSortOperator(
        duckdb::LogicalTopN& logical_topn,
        std::shared_ptr<bodo::Schema> input_schema,
        std::vector<duckdb::ColumnBinding>& source_cols, int64_t limit = -1,
        int64_t offset = -1)
        : PhysicalGPUSortOperator(logical_topn.orders, input_schema,
                                  source_cols, limit, offset,
                                  logical_topn.GetColumnBindings().size()) {}

    virtual ~PhysicalGPUSortOperator() = default;

    void FinalizeSink() override {}

    void FinalizeSource() override {
        QueryProfileCollector::Default().SubmitOperatorName(getOpId(),
                                                            ToString());
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 0),
            this->metrics.init_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.consume_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 2),
            this->metrics.produce_time);
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 2),
            this->metrics.output_row_count);
    }

    OperatorResult ConsumeBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        time_pt start_consume = start_timer();
        bool local_is_last = prev_op_result == OperatorResult::FINISHED;

        if (input_batch.table != nullptr && input_batch.table->num_rows() > 0) {
            cuda_sort_state->ConsumeBatch(input_batch.table, se);
        }

        bool global_is_last =
            cuda_sort_state->FinalizeAccumulation(local_is_last, se);

        this->metrics.consume_time += end_timer(start_consume);
        return global_is_last ? OperatorResult::FINISHED
                              : OperatorResult::NEED_MORE_INPUT;
    }

    std::pair<GPU_DATA, OperatorResult> ProduceBatchGPU(
        std::shared_ptr<StreamAndEvent> se) override {
        time_pt start_produce = start_timer();
        bool out_is_last = false;
        std::unique_ptr<cudf::table> sorted_table =
            cuda_sort_state->GetOutputBatch(out_is_last, se->stream);

        if (sorted_table == nullptr) {
            this->metrics.produce_time += end_timer(start_produce);
            return {GPU_DATA(nullptr, arrow_output_schema, se),
                    out_is_last ? OperatorResult::FINISHED
                                : OperatorResult::HAVE_MORE_OUTPUT};
        }

        // Keep only the selected columns in desired order from kept_cols.
        std::unique_ptr<cudf::table> out_table;
        if (this->kept_cols.size() == (size_t)sorted_table->num_columns()) {
            bool identity = true;
            for (size_t i = 0; i < this->kept_cols.size(); i++) {
                if (this->kept_cols[i] != (int64_t)i) {
                    identity = false;
                    break;
                }
            }
            if (identity) {
                out_table = std::move(sorted_table);
            }
        }

        if (out_table == nullptr) {
            std::vector<cudf::size_type> cudf_kept_cols;
            for (auto c : kept_cols) {
                cudf_kept_cols.push_back(static_cast<cudf::size_type>(c));
            }
            out_table = std::make_unique<cudf::table>(
                sorted_table->select(cudf_kept_cols));
        }

        this->metrics.output_row_count += out_table->num_rows();
        this->metrics.produce_time += end_timer(start_produce);

        return {GPU_DATA(std::move(out_table), arrow_output_schema, se),
                out_is_last ? OperatorResult::FINISHED
                            : OperatorResult::HAVE_MORE_OUTPUT};
    }

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        throw std::runtime_error("GetResult called on a sort node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

    std::string ToString() override { return PhysicalGPUSink::ToString(); }

    int64_t getOpId() const { return PhysicalGPUSink::getOpId(); }

   private:
    std::shared_ptr<bodo::Schema> output_schema;
    std::shared_ptr<arrow::Schema> arrow_output_schema;
    std::unique_ptr<CudaSortState> cuda_sort_state;
    std::vector<int64_t> kept_cols;

    PhysicalGPUSortMetrics metrics;
};

#endif
