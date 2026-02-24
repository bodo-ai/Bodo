#pragma once

#include <cstdint>
#include "../../libs/streaming/_sort.h"
#include "../_util.h"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "operator.h"

struct PhysicalSortMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    time_t init_time = 0;
    time_t consume_time = 0;
    time_t produce_time = 0;

    stat_t output_row_count = 0;
};

/**
 * @brief Physical node for sort.
 *
 */
class PhysicalSort : public PhysicalSource, public PhysicalSink {
   private:
    explicit PhysicalSort(
        duckdb::vector<duckdb::BoundOrderByNode>& orders,
        std::shared_ptr<bodo::Schema> input_schema,
        std::vector<duckdb::ColumnBinding>& source_cols, int64_t limit,
        int64_t offset, unsigned node_cols,
        const std::vector<duckdb::idx_t>& projection_map = {}) {
        time_pt start_init = start_timer();
        // Calculate the output schema.
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
        for (size_t i = 0; i < this->kept_cols.size(); i++) {
            std::unique_ptr<bodo::DataType> col_type =
                input_schema->column_types[this->kept_cols[i]]->copy();
            this->output_schema->append_column(std::move(col_type));
            this->output_schema->column_names.push_back(
                input_schema->column_names[this->kept_cols[i]]);
        }
        if (this->kept_cols.size() != node_cols) {
            throw std::runtime_error(
                "Sort output schema has different number of columns than "
                "LogicalOrder or LogicalTopN");
        }
        this->output_schema->metadata = std::make_shared<bodo::TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));

        std::vector<int64_t> ascending;
        std::vector<int64_t> na_last;
        std::vector<uint64_t> keys;
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map;
        col_ref_map = getColRefMap(source_cols);

        // Convert BoundOrderByNode's to keys, asc, and na_last.
        for (const auto& order : orders) {
            switch (order.type) {
                case duckdb::OrderType::ASCENDING:
                    ascending.push_back(1);
                    break;
                case duckdb::OrderType::DESCENDING:
                    ascending.push_back(0);
                    break;
                default:
                    throw std::runtime_error(
                        "PhysicalSort order type not ascending or descending.");
            }
            switch (order.null_order) {
                case duckdb::OrderByNullType::NULLS_FIRST:
                    na_last.push_back(0);
                    break;
                case duckdb::OrderByNullType::NULLS_LAST:
                    na_last.push_back(1);
                    break;
                default:
                    throw std::runtime_error(
                        "PhysicalSort orderbynull type not first or last.");
            }
            duckdb::ExpressionClass expr_class =
                order.expression->GetExpressionClass();
            if (expr_class != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "PhysicalSort expression is not column ref.");
            }
            auto& bce =
                order.expression->Cast<duckdb::BoundColumnRefExpression>();
            keys.push_back(col_ref_map[{bce.binding.table_index,
                                        bce.binding.column_index}]);
        }

        // Establish table reordering so key are at beginning.
        bidirectionalColumnMapping(col_inds, inverse_col_inds, keys,
                                   input_schema->ncols());

        std::shared_ptr<bodo::Schema> build_table_schema_reordered =
            input_schema->Project(col_inds);

        if (limit == -1 && offset == -1) {
            stream_sorter = std::make_unique<StreamSortState>(
                getOpId(), keys.size(), std::move(ascending),
                std::move(na_last), build_table_schema_reordered,
                /*parallel*/ true);
        } else {
            stream_sorter = std::make_unique<StreamSortLimitOffsetState>(
                getOpId(), keys.size(), std::move(ascending),
                std::move(na_last), build_table_schema_reordered,
                /*parallel*/ true, limit, offset);
        }
        this->metrics.init_time = end_timer(start_init);
    }

   public:
    explicit PhysicalSort(duckdb::LogicalOrder& logical_order,
                          std::shared_ptr<bodo::Schema> input_schema,
                          std::vector<duckdb::ColumnBinding>& source_cols,
                          int64_t limit = -1, int64_t offset = -1)
        : PhysicalSort(logical_order.orders, input_schema, source_cols, limit,
                       offset, logical_order.GetColumnBindings().size(),
                       logical_order.projection_map) {}

    explicit PhysicalSort(duckdb::LogicalTopN& logical_topn,
                          std::shared_ptr<bodo::Schema> input_schema,
                          std::vector<duckdb::ColumnBinding>& source_cols,
                          int64_t limit = -1, int64_t offset = -1)
        : PhysicalSort(logical_topn.orders, input_schema, source_cols, limit,
                       offset, logical_topn.GetColumnBindings().size()) {}

    virtual ~PhysicalSort() = default;

    void FinalizeSink() override {
        time_pt start_finalize_build = start_timer();
        stream_sorter->FinalizeBuild();
        this->metrics.consume_time += end_timer(start_finalize_build);
    }

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

    /**
     * @brief process input tables to sort
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        time_pt start_consume = start_timer();
        bool local_is_last = prev_op_result == OperatorResult::FINISHED;

        std::shared_ptr<table_info> input_batch_reordered =
            ProjectTable(input_batch, col_inds);

        stream_sorter->ConsumeBatch(input_batch_reordered);
        this->metrics.consume_time += end_timer(start_consume);

        return local_is_last ? OperatorResult::FINISHED
                             : OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief Act as data source producing sorted rows.
     *
     * @param input_batch input batch to sort
     * @return output batch of sorted data and return flag
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        time_pt start_produce = start_timer();
        auto sorted_res = stream_sorter->GetOutput();

        // Undo column reordering for sorting.
        std::shared_ptr<table_info> sorted_table =
            ProjectTable(sorted_res.first, inverse_col_inds);

        // Keep only the selected columns in desired order from
        // projection_map.
        std::vector<std::shared_ptr<array_info>> out_cols;
        for (size_t i = 0; i < this->kept_cols.size(); i++) {
            out_cols.emplace_back(sorted_table->columns[this->kept_cols[i]]);
        }
        std::shared_ptr<table_info> out_table = std::make_shared<table_info>(
            out_cols, sorted_table->nrows(), output_schema->column_names,
            output_schema->metadata);
        this->metrics.output_row_count += sorted_table->nrows();
        this->metrics.produce_time += end_timer(start_produce);

        return {out_table, sorted_res.second
                               ? OperatorResult::FINISHED
                               : OperatorResult::HAVE_MORE_OUTPUT};
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        // Sort build doesn't return output results
        throw std::runtime_error("GetResult called on a sort node.");
    }

    /**
     * @brief Get the output schema
     *
     * @return std::shared_ptr<bodo::Schema>
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

    std::string ToString() override { return PhysicalSink::ToString(); }

    int64_t getOpId() const { return PhysicalSink::getOpId(); }

   private:
    static void bidirectionalColumnMapping(
        std::vector<int64_t>& col_inds, std::vector<int64_t>& inverse_col_inds,
        const std::vector<uint64_t>& keys, uint64_t ncols) {
        initInputColumnMapping(col_inds, keys, ncols);

        inverse_col_inds.resize(ncols);
        for (uint64_t i = 0; i < ncols; i++) {
            inverse_col_inds[col_inds[i]] = i;
        }
    }

    std::vector<int64_t> col_inds;
    std::vector<int64_t> inverse_col_inds;
    std::shared_ptr<bodo::Schema> output_schema;
    std::unique_ptr<StreamSortState> stream_sorter;
    std::vector<uint64_t> kept_cols;

    PhysicalSortMetrics metrics;
};
