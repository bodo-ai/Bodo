#pragma once

#include <algorithm>
#include <utility>
#include "../../libs/_query_profile_collector.h"
#include "../libs/_array_utils.h"
#include "../libs/streaming/_join.h"
#include "_plan.h"
#include "_util.h"
#include "operator.h"

struct PhysicalGPUJoinFilterMetrics {
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
class PhysicalGPUJoinFilter : public PhysicalGPUProcessBatch {
   public:
    explicit PhysicalGPUJoinFilter(
        bodo::LogicalJoinFilter& logical_filter,
        std::shared_ptr<bodo::Schema>& input_schema,
        std::shared_ptr<std::unordered_map<int, join_state_t>>&
            join_filter_states)
        : filter_ids(std::move(logical_filter.filter_ids)),
          is_first_locations(std::move(logical_filter.is_first_locations)),
          join_filter_states(join_filter_states) {
        this->output_schema = input_schema;

        this->can_apply_bloom_filters.reserve(
            logical_filter.filter_columns.size());
        for (const auto& join_key_idxs : logical_filter.filter_columns) {
            std::vector<cudf::size_type> cudf_idxs;
            for (const auto& join_key : join_key_idxs) {
                cudf_idxs.push_back(join_key);
            }
            filter_columns.emplace_back(std::move(cudf_idxs));
            bool can_apply =
                !join_key_idxs.empty() &&
                std::all_of(join_key_idxs.begin(), join_key_idxs.end(),
                            [](int idx) { return idx != -1; });
            this->can_apply_bloom_filters.push_back(can_apply);
        }

        // TODO[BSE-5176]: support column level filters (only on DICT columns
        // currently)
    }

    virtual ~PhysicalGPUJoinFilter() = default;

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
    std::pair<GPU_DATA, OperatorResult> ProcessBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        if (!is_gpu_rank()) {
            return {GPU_DATA(nullptr, input_batch.schema, se),
                    prev_op_result == OperatorResult::FINISHED
                        ? OperatorResult::FINISHED
                        : OperatorResult::NEED_MORE_INPUT};
        }

        this->metrics.input_row_count +=
            input_batch.table ? input_batch.table->num_rows() : 0;

        // No filters can be applied, just pass through
        if (std::ranges::all_of(this->can_apply_bloom_filters,
                                [](bool v) { return !v; }) ||
            input_batch.table->num_rows() == 0) {
            GPU_DATA out_table_info(std::move(input_batch.table),
                                    input_batch.schema, se);
            return {out_table_info, prev_op_result == OperatorResult::FINISHED
                                        ? OperatorResult::FINISHED
                                        : OperatorResult::NEED_MORE_INPUT};
        }

        time_pt start_filtering = start_timer();

        std::unique_ptr<cudf::column> row_bitmask;  // start off empty

        // Apply filters
        for (size_t i = 0; i < this->filter_ids.size(); i++) {
            int filter_id = this->filter_ids[i];
            if (!this->can_apply_bloom_filters[i]) {
                continue;
            }

            join_state_t join_state_ = (*join_filter_states)[filter_id];
            std::visit(
                [&](const auto& join_state) {
                    if constexpr (std::is_same_v<
                                      std::decay_t<decltype(join_state)>,
                                      CudaJoin*>) {
                        join_state->runtime_filter(input_batch.table->view(),
                                                   filter_columns[i],
                                                   row_bitmask, se->stream);
                    }
                },
                join_state_);
        }

        std::shared_ptr<cudf::table> result;
        if (row_bitmask) {
            result = cudf::apply_boolean_mask(input_batch.table->view(),
                                              *row_bitmask, se->stream);
        } else {
            result = input_batch.table;
        }

        this->metrics.filtering_time += end_timer(start_filtering);
        this->metrics.output_row_count += result->num_rows();

        GPU_DATA out_table_info(std::move(result), input_batch.schema, se);

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
    std::shared_ptr<bodo::Schema> output_schema;

    // IDs of joins creating each filter
    const std::vector<int> filter_ids;
    // Mapping columns of the join to the columns in the current table
    std::vector<std::vector<cudf::size_type>> filter_columns;
    // Indicating for which of the columns is it the first filtering site
    const std::vector<std::vector<bool>> is_first_locations;

    std::vector<bool> can_apply_bloom_filters;

    // Mapping of join ids to their join state info for join filter operators
    // (filled during physical plan construction). Using loose pointers since
    // PhysicalGPUJoinFilter only needs to access the join state during
    // execution
    std::shared_ptr<std::unordered_map<int, join_state_t>> join_filter_states;

    PhysicalGPUJoinFilterMetrics metrics;
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
