#pragma once

#include <arrow/compute/expression.h>
#include <arrow/scalar.h>
#include <arrow/type_fwd.h>
#include <kll_sketch.hpp>
#include <memory>
#include <stdexcept>
#include <utility>
#include "../../libs/_distributed.h"
#include "operator.h"
#include "physical/expression.h"

#undef CHECK_ARROW
#define CHECK_ARROW(expr, msg)                                             \
    if (!(expr.ok())) {                                                    \
        std::string err_msg = std::string("PhysicalKLL::ConsumeBatch: ") + \
                              msg + " " + expr.ToString();                 \
        throw std::runtime_error(err_msg);                                 \
    }

struct PhysicalQuantileMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    stat_t output_row_count = 0;

    time_t consume_time = 0;
    time_t finalize_time = 0;
};

template <typename T>
class PhysicalQuantile : public PhysicalSource, public PhysicalSink {
    using KllSketch = datasketches::kll_sketch<T>;

   public:
    explicit PhysicalQuantile(
        std::shared_ptr<bodo::Schema> out_schema, std::vector<double> quantiles,
        uint16_t k = datasketches::kll_constants::DEFAULT_K)
        : out_schema(std::move(out_schema)),
          quantiles(std::move(quantiles)),
          sketch(std::make_shared<KllSketch>(k)) {
        assert(this->out_schema->ncols() == 1);
    }

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        time_pt start_consume_time = start_timer();
        assert(input_batch->ncols() == 1 &&
               input_batch->columns[0]->arr_type ==
                   bodo_array_type::NULLABLE_INT_BOOL);

        auto col = input_batch->columns[0];
        for (uint64_t i = 0; i < input_batch->nrows(); i++) {
            sketch->update(input_batch->columns[0]
                               ->at<T, bodo_array_type::NULLABLE_INT_BOOL>(i));
        }

        this->metrics.consume_time += end_timer(start_consume_time);
        return prev_op_result == OperatorResult::FINISHED
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        return {final_result, OperatorResult::FINISHED};
    }

    virtual ~PhysicalQuantile() = default;

    void FinalizeSink() override {
        if (collected) {
            return;
        }
        time_pt start_finalize_time = start_timer();
        // Toggle collected flag to prevent multiple collections
        collected = true;
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        auto bytes = sketch->serialize();
        int local_size = (int)bytes.size();

        std::vector<int> recv_counts(size);
        CHECK_MPI(MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1,
                             MPI_INT, 0, MPI_COMM_WORLD),
                  "quantile.h::Finalize(): MPI error on MPI_Gather:");

        std::vector<int> displs(size);
        std::vector<uint8_t> all_bytes;

        if (rank == 0) {
            int total_size = 0;
            for (int i = 0; i < size; i++) {
                displs[i] = total_size;
                total_size += recv_counts[i];
            }
            all_bytes.resize(total_size);
        }

        CHECK_MPI(MPI_Gatherv(bytes.data(), local_size, MPI_BYTE,
                              all_bytes.data(), recv_counts.data(),
                              displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD),
                  "quantile.h::Finalize(): MPI error on MPI_Gatherv:");

        if (rank == 0) {
            auto merged_sketch = KllSketch();
            for (int i = 0; i < size; i++) {
                auto start = all_bytes.data() + displs[i];
                auto other = KllSketch::deserialize(start, recv_counts[i]);
                merged_sketch.merge(std::move(other));
            }

            *sketch = std::move(merged_sketch);

            std::vector<std::shared_ptr<array_info>> results{};
            for (auto it : quantiles) {
                double qval;
                if (!sketch->is_empty()) {
                    // For queries q=0.0 and q=1.0, return exact min/max values
                    if (it == 0.0) {
                        qval = sketch->get_min_item();
                    } else if (it == 1.0) {
                        qval = sketch->get_max_item();
                    } else {
                        qval = sketch->get_quantile(it);
                    }
                } else {
                    qval = std::numeric_limits<double>::quiet_NaN();
                }
                auto arrow_scalar = arrow::MakeScalar(qval);
                auto arr = ScalarToArrowArray(arrow_scalar);
                auto bodo_arr =
                    arrow_array_to_bodo(arr, bodo::BufferPool::DefaultPtr());
                this->metrics.output_row_count += 1;
                results.push_back(bodo_arr);
            }
            final_result = std::make_shared<table_info>(results);
        } else {
            // Produce empty result for rank != 0
            final_result = std::make_shared<table_info>(
                std::vector<std::shared_ptr<array_info>>{});
        }
        this->metrics.finalize_time += end_timer(start_finalize_time);

        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(PhysicalSink::getOpId(),
                                                       1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(PhysicalSink::getOpId(),
                                                       1),
            this->metrics.output_row_count);
    }

    void FinalizeSource() override {}

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        throw std::runtime_error("GetResult called on a quantile node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

   private:
    std::shared_ptr<bodo::Schema> out_schema;
    const std::vector<double> quantiles;
    std::shared_ptr<KllSketch> sketch;
    std::shared_ptr<table_info> final_result;
    bool collected{false};
    PhysicalQuantileMetrics metrics;
    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        metrics_out.emplace_back(
            TimerMetric("consume_time", this->metrics.consume_time));
        metrics_out.emplace_back(
            TimerMetric("finalize_time", this->metrics.finalize_time));
    }
};
