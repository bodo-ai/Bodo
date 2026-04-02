#pragma once

#include <fmt/format.h>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "duckdb/planner/operator/logical_set_operation.hpp"
#include "operator.h"

inline bool gpu_capable(duckdb::LogicalSetOperation &op) {
    if (op.type == duckdb::LogicalOperatorType::LOGICAL_UNION && op.setop_all) {
        return true;
    }
    return false;
}

struct PhysicalGPUUnionAllMetrics {
    using timer_t = MetricBase::TimerValue;
    using stat_t = MetricBase::StatValue;
    stat_t output_row_count = 0;
};

/**
 * @brief Physical node for union all on GPU.
 *
 */
class PhysicalGPUUnionAll : public PhysicalGPUProcessBatch,
                            public PhysicalGPUSink {
   public:
    explicit PhysicalGPUUnionAll(std::shared_ptr<bodo::Schema> input_schema)
        : output_schema(input_schema),
          arrow_output_schema(input_schema->ToArrowSchema()) {}

    virtual ~PhysicalGPUUnionAll() = default;

    void FinalizeSink() override {}

    void FinalizeProcessBatch() override {
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(
                PhysicalGPUSink::getOpId(), 1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(
                PhysicalGPUSink::getOpId(), 2),
            this->metrics.output_row_count);
    }

    /**
     * @brief Consume a batch of GPU data for union all and populate the
     * internal buffer.
     *
     */
    OperatorResult ConsumeBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        if (is_gpu_rank()) {
            collected_rows.emplace_back(input_batch);
        }
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
     */
    std::pair<GPU_DATA, OperatorResult> ProcessBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        if (!is_gpu_rank()) {
            return {GPU_DATA(nullptr, arrow_output_schema, se),
                    prev_op_result == OperatorResult::FINISHED
                        ? OperatorResult::FINISHED
                        : OperatorResult::NEED_MORE_INPUT};
        }

        if (!first_processed_batch.has_value()) {
            first_processed_batch = input_batch;
        }
        if (!collected_rows.empty()) {
            auto next_batch = collected_rows.back();
            collected_rows.pop_back();
            return {next_batch, OperatorResult::HAVE_MORE_OUTPUT};
        } else {
            if (first_processed_batch.has_value()) {
                input_batch = first_processed_batch.value();
                first_processed_batch = std::nullopt;
            }
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
    std::vector<GPU_DATA> collected_rows;
    std::optional<GPU_DATA> first_processed_batch;
    const std::shared_ptr<bodo::Schema> output_schema;
    const std::shared_ptr<arrow::Schema> arrow_output_schema;
    PhysicalGPUUnionAllMetrics metrics;
    void ReportMetrics(std::vector<MetricBase> &metrics_out) {
        // No metrics to report for GPU union all yet
    }
};
