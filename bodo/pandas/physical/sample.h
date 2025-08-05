
#pragma once

#include <memory>
#include <utility>
#include "../libs/_array_utils.h"

#include "operator.h"

struct PhysicalSampleMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    stat_t output_row_count = 0;

    time_t sample_time = 0;
};

/**
 * @brief Physical node for sampling.
 *
 */
class PhysicalSample : public PhysicalProcessBatch {
   public:
    explicit PhysicalSample(float percent,
                            std::shared_ptr<bodo::Schema> input_schema)
        : percentage(percent), output_schema(input_schema) {}

    virtual ~PhysicalSample() = default;

    void FinalizeProcessBatch() override {
        std::vector<MetricBase> metrics_out;
        metrics_out.emplace_back(
            TimerMetric("sample_time", this->metrics.sample_time));
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.output_row_count);
    }

    /**
     * @brief Do limit.
     *
     * @return std::pair<int64_t, OperatorResult> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), state of operator after processing the
     * batch
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) override {
        time_pt start_sample_time = start_timer();
        uint64_t select_this_time = stochasticRound(input_batch->nrows());

        // Perhaps we should randomly select rather than just
        // take the first select_this_time rows.
        std::vector<int64_t> rowInds(select_this_time);
        for (uint64_t i = 0; i < select_this_time; ++i) {
            rowInds[i] = i;
        }
        std::shared_ptr<table_info> out_table_info =
            RetrieveTable(input_batch, rowInds);
        this->metrics.output_row_count += out_table_info->nrows();
        this->metrics.sample_time += end_timer(start_sample_time);
        return {out_table_info, OperatorResult::NEED_MORE_INPUT};
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

   private:
    const float percentage;
    const std::shared_ptr<bodo::Schema> output_schema;
    PhysicalSampleMetrics metrics;

    uint64_t stochasticRound(uint64_t nrows) {
        double scaled = nrows * percentage;
        int base = static_cast<int>(scaled);
        double remainder = scaled - base;

        // Decide whether to add 1 based on remainder probability
        return base +
               (static_cast<double>(rand()) / RAND_MAX < remainder ? 1 : 0);
    }
};
