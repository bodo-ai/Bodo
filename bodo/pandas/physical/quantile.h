#pragma once

#include <arrow/compute/expression.h>
#include <arrow/scalar.h>
#include <arrow/type_fwd.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include "../../libs/_distributed.h"
#include "../libs/_bodo_to_arrow.h"
#include "/tmp/datasketches-prefix/include/DataSketches/kll_sketch.hpp"
#include "operator.h"
#include "physical/expression.h"

using KLLDoubleSketch = datasketches::kll_sketch<double>;

#undef CHECK_ARROW
#define CHECK_ARROW(expr, msg)                                             \
    if (!(expr.ok())) {                                                    \
        std::string err_msg = std::string("PhysicalKLL::ConsumeBatch: ") + \
                              msg + " " + expr.ToString();                 \
        throw std::runtime_error(err_msg);                                 \
    }

class PhysicalQuantile : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalQuantile(
        std::shared_ptr<bodo::Schema> out_schema, std::vector<double> quantiles,
        uint16_t k = datasketches::kll_constants::DEFAULT_K)
        : out_schema(out_schema),
          quantiles(quantiles),
          sketch(std::make_shared<KLLDoubleSketch>(k)) {}

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        auto col = input_batch->columns[0];
        arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
        std::shared_ptr<arrow::Array> in_arrow_array = bodo_array_to_arrow(
            bodo::BufferPool::DefaultPtr(), input_batch->columns[0],
            false /*convert_timedelta_to_int64*/, "", time_unit,
            false, /*downcast_time_ns_to_us*/
            bodo::default_buffer_memory_manager());

        for (auto i = 0; i < in_arrow_array->length(); i++) {
            auto elem_result = in_arrow_array->GetScalar(i);
            CHECK_ARROW(elem_result.status(),
                        "Error getting scalar from Arrow array");
            auto scalar = elem_result.ValueOrDie();
            if (scalar->is_valid) {
                // Cast to DoubleScalar and extract value
                auto double_scalar =
                    std::dynamic_pointer_cast<arrow::DoubleScalar>(scalar);
                if (!double_scalar) {
                    auto casted_result = arrow::compute::Cast(
                        scalar, arrow::float64(),
                        arrow::compute::CastOptions::Safe(),
                        bodo::default_buffer_exec_context());
                    CHECK_ARROW(casted_result.status(),
                                "Failed to cast scalar to double");
                    auto casted_scalar = casted_result.ValueOrDie().scalar();
                    double_scalar =
                        std::dynamic_pointer_cast<arrow::DoubleScalar>(
                            casted_scalar);
                    if (!double_scalar) {
                        throw std::runtime_error(
                            "Expected DoubleScalar after cast");
                    }
                }
                sketch->update(double_scalar->value);
            }
        }

        return prev_op_result == OperatorResult::FINISHED
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
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

        // TODO: consider possible optimization of merging and querying sketches
        // across ranks.
        if (rank == 0) {
            auto merged_sketch = KLLDoubleSketch();
            for (int i = 0; i < size; i++) {
                auto start = all_bytes.data() + displs[i];
                auto other =
                    KLLDoubleSketch::deserialize(start, recv_counts[i]);
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
                results.push_back(bodo_arr);
            }
            final_result = std::make_shared<table_info>(results);
        } else {
            // Produce empty result for rank != 0
            final_result = std::make_shared<table_info>(
                std::vector<std::shared_ptr<array_info>>{});
        }
        return {final_result, OperatorResult::FINISHED};
    }

    virtual ~PhysicalQuantile() = default;

    void Finalize() override {}

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        throw std::runtime_error("GetResult called on a quantile node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

   private:
    std::shared_ptr<bodo::Schema> out_schema;
    const std::vector<double> quantiles;
    std::shared_ptr<KLLDoubleSketch> sketch;
    std::shared_ptr<table_info> final_result;
};
