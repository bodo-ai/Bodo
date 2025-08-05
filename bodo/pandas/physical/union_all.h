#pragma once

#include <fmt/format.h>
#include <memory>
#include <utility>
#include "../libs/_table_builder.h"
#include "operator.h"

struct PhysicalUnionAllMetrics {
    using timer_t = MetricBase::TimerValue;
    using stat_t = MetricBase::StatValue;
    stat_t output_row_count = 0;
};

/**
 * @brief Physical node for union all.
 *
 */
class PhysicalUnionAll : public PhysicalProcessBatch, public PhysicalSink {
   public:
    explicit PhysicalUnionAll(std::shared_ptr<bodo::Schema> input_schema)
        : output_schema(input_schema) {}

    virtual ~PhysicalUnionAll() = default;

    void Finalize() override {
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(PhysicalSink::getOpId(),
                                                       1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(PhysicalSink::getOpId(),
                                                       2),
            this->metrics.output_row_count);
    }

    /**
     * @brief Do union all.
     *
     * @return std::pair<std::shared_ptr<table_info>, OperatorResult>
     * The output table from the current operation and whether there is more
     * output.
     */
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        if (!collected_rows) {
            collected_rows = std::make_unique<ChunkedTableBuilderState>(
                input_batch->schema(), get_streaming_batch_size());
        }
        collected_rows->builder->AppendBatch(input_batch);
        return (prev_op_result == OperatorResult::FINISHED)
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::variant<std::shared_ptr<table_info>, PyObject *> GetResult() override {
        // Union all should be between pipelines and act alternatively as a sink
        // then source but there should never be the need to ask for the result
        // all in one go.
        throw std::runtime_error("GetResult called on a union all node.");
    }

    /**
     * @brief ProcessBatch streaming through union
     *
     * returns std::pair<std::shared_ptr<table_info>, OperatorResult>
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) override {
        if (!first_processed_batch) {
            first_processed_batch = input_batch;
        }
        if (collected_rows && !collected_rows->builder->empty()) {
            auto next_batch = collected_rows->builder->PopChunk(true);
            this->metrics.output_row_count += std::get<0>(next_batch)->nrows();
            return {std::get<0>(next_batch),
                    //(prev_op_result == OperatorResult::FINISHED &&
                    // collected_rows->builder->empty())
                    //    ? OperatorResult::FINISHED
                    //    : OperatorResult::HAVE_MORE_OUTPUT};
                    OperatorResult::HAVE_MORE_OUTPUT};
        } else {
            if (first_processed_batch) {
                input_batch = first_processed_batch;
                first_processed_batch = nullptr;
            }
            this->metrics.output_row_count += input_batch->nrows();
            return {input_batch, prev_op_result == OperatorResult::FINISHED
                                     ? OperatorResult::FINISHED
                                     : OperatorResult::NEED_MORE_INPUT};
        }
    }

    /**
     * @brief Get the physical schema of the output data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

   private:
    std::unique_ptr<ChunkedTableBuilderState> collected_rows;
    std::shared_ptr<table_info> first_processed_batch;
    const std::shared_ptr<bodo::Schema> output_schema;
    PhysicalUnionAllMetrics metrics;
    void ReportMetrics(std::vector<MetricBase> &metrics_out) {
        MetricBase::TimerValue append_time =
            this->collected_rows ? this->collected_rows->builder->append_time
                                 : MetricBase::TimerValue(0);
        metrics_out.push_back(TimerMetric("append_time", append_time));

        // Add the dict builder metrics if they exist
        for (size_t i = 0; i < this->output_schema->ncols(); ++i) {
            auto dict_builder = this->collected_rows
                                    ? this->collected_rows->dict_builders[i]
                                    : nullptr;
            if (dict_builder) {
                dict_builder->GetMetrics().add_to_metrics(
                    metrics_out, fmt::format("dict_builder_{}", i));
            }
        }

        // No metrics to report for union all
    }
};
