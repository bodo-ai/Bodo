#pragma once

#include <algorithm>
#include <utility>
#include "../../libs/_query_profile_collector.h"
#include "../libs/_array_utils.h"
#include "../libs/streaming/_join.h"
#include "_plan.h"
#include "_util.h"
#include "operator.h"

struct PhysicalJoinFilterMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t output_row_count = 0;
    stat_t input_row_count = 0;
    time_t filtering_time = 0;
};

/**
 * @brief Physical node for join filter.
 *
 */
class PhysicalJoinFilter : public PhysicalProcessBatch {
   public:
    explicit PhysicalJoinFilter(
        bodo::LogicalJoinFilter& logical_filter,
        std::shared_ptr<bodo::Schema>& input_schema,
        std::shared_ptr<std::unordered_map<int, join_state_t>>&
            join_filter_states)
        : filter_ids(std::move(logical_filter.filter_ids)),
          filter_columns(std::move(logical_filter.filter_columns)),
          is_first_locations(std::move(logical_filter.is_first_locations)),
          join_filter_states(join_filter_states) {
        this->output_schema = std::make_shared<bodo::Schema>();

        this->materialize_after_each_filter = true;
        for (size_t i = 0; i < input_schema->ncols(); i++) {
            std::unique_ptr<bodo::DataType> col_type =
                input_schema->column_types[i]->copy();
            // TODO[BSE-5176]: Handle DICT column casting for bloom filters
            // See
            // https://github.com/bodo-ai/Bodo/blob/4b6e5830cc9f16bba5fc40ba495f11955bbf15af/bodo/libs/streaming/join.py#L1477
            if (col_type->array_type == bodo_array_type::DICT) {
                throw std::runtime_error(
                    "Join filter does not support input tables with DICT "
                    "columns yet");
            }

            // Avoid materializing after each filter if there is any variable
            // length column (similar to Python join filter code)
            if (col_type->is_array() || col_type->is_map() ||
                col_type->array_type == bodo_array_type::STRING) {
                this->materialize_after_each_filter = false;
            }

            this->output_schema->append_column(std::move(col_type));
            this->output_schema->column_names.push_back(
                input_schema->column_names[i]);
        }

        this->output_schema->metadata = std::make_shared<bodo::TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));

        this->can_apply_bloom_filters.reserve(this->filter_columns.size());
        for (const auto& join_key_idxs : this->filter_columns) {
            bool can_apply =
                !join_key_idxs.empty() &&
                std::all_of(join_key_idxs.begin(), join_key_idxs.end(),
                            [](int idx) { return idx != -1; });
            this->can_apply_bloom_filters.push_back(can_apply);
        }

        // TODO[BSE-5176]: support column level filters (only on DICT columns
        // currently)
    }

    virtual ~PhysicalJoinFilter() = default;

    void FinalizeProcessBatch() override {
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().SubmitOperatorName(getOpId(),
                                                            ToString());
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

        // No filters can be applied, just pass through
        if (std::ranges::all_of(this->can_apply_bloom_filters,
                                [](bool v) { return !v; }) ||
            input_batch->nrows() == 0) {
            return {input_batch, prev_op_result == OperatorResult::FINISHED
                                     ? OperatorResult::FINISHED
                                     : OperatorResult::NEED_MORE_INPUT};
        }

        time_pt start_filtering = start_timer();

        // Allocate bitmask initialized to all true
        if (!row_bitmask || row_bitmask->length < input_batch->nrows()) {
            row_bitmask = alloc_nullable_array_no_nulls(input_batch->nrows(),
                                                        Bodo_CTypes::_BOOL);
            // Initialize all bits to true
            memset(row_bitmask
                       ->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t*>(),
                   0xff, arrow::bit_util::BytesForBits(input_batch->nrows()));
        }

        bool applied_any_filter = false;

        // Apply filters
        for (size_t i = 0; i < this->filter_ids.size(); i++) {
            int filter_id = this->filter_ids[i];
            if (!this->can_apply_bloom_filters[i]) {
                continue;
            }

            join_state_t join_state_ = (*join_filter_states)[filter_id];
            // GPU Joins don't create bloom filters so only run against
            // CPU JoinStates
            std::visit(
                [&](const auto& join_state) {
                    if constexpr (std::is_same_v<
                                      std::decay_t<decltype(join_state)>,
                                      JoinState*>) {
                        if (join_state->IsNestedLoopJoin()) {
                            return;
                        }
                        HashJoinState* hash_join_state =
                            (HashJoinState*)join_state;

                        applied_any_filter = applied_any_filter ||
                                             hash_join_state->RuntimeFilter(
                                                 input_batch, row_bitmask,
                                                 this->filter_columns[i],
                                                 this->is_first_locations[i]);

                        if (this->materialize_after_each_filter &&
                            applied_any_filter) {
                            input_batch =
                                RetrieveTable(input_batch, row_bitmask);
                            // Reset the bitmask for the next filter
                            // we could potentially do something smarter here if
                            // row_bitmask's length is way bigger than input
                            // batch by only resetting the first
                            // input_batch->nrows() bits and then resetting the
                            // whole thing at the end but this should be good
                            // enough for now, we don't always want to do that
                            // because it's an extra reset
                            memset(
                                row_bitmask
                                    ->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                            uint8_t*>(),
                                0xff,
                                arrow::bit_util::BytesForBits(
                                    row_bitmask->length));

                            applied_any_filter = false;
                        }
                    }
                },
                join_state_);
        }
        if (applied_any_filter) {
            input_batch = RetrieveTable(input_batch, row_bitmask);
            // Reset the bitmask for the next batch
            memset(row_bitmask
                       ->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t*>(),
                   0xff, arrow::bit_util::BytesForBits(row_bitmask->length));
        }

        this->metrics.filtering_time += end_timer(start_filtering);
        this->metrics.output_row_count += input_batch->nrows();

        // Just propagate the FINISHED flag to other operators (like join) or
        // accept more input
        return {input_batch, prev_op_result == OperatorResult::FINISHED
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
    std::shared_ptr<bodo::Schema> output_schema;

    // IDs of joins creating each filter
    const std::vector<int> filter_ids;
    // Mapping columns of the join to the columns in the current table
    const std::vector<std::vector<int64_t>> filter_columns;
    // Indicating for which of the columns is it the first filtering site
    const std::vector<std::vector<bool>> is_first_locations;

    std::vector<bool> can_apply_bloom_filters;

    bool materialize_after_each_filter;

    std::shared_ptr<array_info> row_bitmask;

    // Mapping of join ids to their JoinState pointers for join filter operators
    // (filled during physical plan construction). Using loose pointers since
    // PhysicalJoinFilter only needs to access the JoinState during execution
    std::shared_ptr<std::unordered_map<int, join_state_t>> join_filter_states;

    PhysicalJoinFilterMetrics metrics;
    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        metrics_out.reserve(2);
        metrics_out.emplace_back(
            TimerMetric("filtering_time", this->metrics.filtering_time));
        MetricBase::BlobValue selectivity_value(
            std::to_string(static_cast<double>(this->metrics.output_row_count) /
                           static_cast<double>(this->metrics.input_row_count)));
        metrics_out.emplace_back(BlobMetric("selectivity", selectivity_value));
    }
};
