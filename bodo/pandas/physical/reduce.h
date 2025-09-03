#pragma once

#include <arrow/array/util.h>
#include <arrow/compute/expression.h>
#include <arrow/scalar.h>
#include <arrow/type_fwd.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include "../libs/_bodo_to_arrow.h"
#include "../libs/_query_profile_collector.h"
#include "../libs/_shuffle.h"
#include "operator.h"
#include "physical/expression.h"

#undef CHECK_ARROW
#define CHECK_ARROW(expr, msg)                                                \
    if (!(expr.ok())) {                                                       \
        std::string err_msg = std::string("PhysicalReduce::ConsumeBatch: ") + \
                              msg + " " + expr.ToString();                    \
        throw std::runtime_error(err_msg);                                    \
    }

enum class ReductionType {
    COMPARISON,
    AGGREGATION,
};

struct PhysicalReduceMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    stat_t output_row_count = 0;

    time_t consume_time = 0;
    time_t produce_time = 0;
};

struct ReductionFunction {
    virtual void Finalize() = 0;
    virtual ~ReductionFunction() = default;
    std::vector<std::string> function_names;
    std::vector<std::string> reduction_names;
    std::vector<ReductionType> reduction_types;
    arrow::ScalarVector results;
    arrow::DataTypeVector result_types;
};

struct ReductionFunctionMax : public ReductionFunction {
    ReductionFunctionMax(arrow::ScalarVector initial_results) {
        assert(initial_results.size() == 1);
        function_names = {"max"};
        reduction_names = {"greater"};
        results = initial_results;
        reduction_types = {ReductionType::COMPARISON};
    }
    void Finalize() override {
        std::vector<std::shared_ptr<arrow::Array>> global_results;
        int n_ranks, rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
        arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;

        for (size_t i = 0; i < n_ranks; i++) {
            auto arrow_array_builder_res = arrow::MakeBuilder(results[0]->type);
            auto arrow_array_builder =
                arrow_array_builder_res.MoveValueUnsafe();
            CHECK_ARROW(arrow_array_builder_res.status(),
                        "Error in MakeBuilder for max reduction");
            auto append_status = arrow_array_builder->AppendScalars(results);
            CHECK_ARROW(append_status,
                        "Error in AppendScalars for max reduction");
            arrow::Result<std::shared_ptr<arrow::Array>> array_res =
                arrow_array_builder->Finish();
            CHECK_ARROW(array_res.status(),
                        "Error in AppendScalars for max reduction");
            auto arrow_array = array_res.ValueOrDie();
            std::shared_ptr<array_info> send_array =
                i == rank ? arrow_array_to_bodo(arrow_array,
                                                bodo::BufferPool::DefaultPtr())
                          : nullptr;
            std::shared_ptr<array_info> array =
                broadcast_array(nullptr, send_array, nullptr, false, i, rank);
            std::shared_ptr<arrow::Array> received_arrow_array =
                bodo_array_to_arrow(bodo::BufferPool::DefaultPtr(), array,
                                    false /*convert_timedelta_to_int64*/, "",
                                    time_unit, false, /*downcast_time_ns_to_us*/
                                    bodo::default_buffer_memory_manager());
            global_results.push_back(received_arrow_array);
        }
        // Find global max, each receieved array has size 1
        std::shared_ptr<arrow::Scalar> global_max = results[0];
        for (const auto& arr : global_results) {
            std::shared_ptr<arrow::Scalar> arr_scalar =
                arr->GetScalar(0).ValueOrDie();
            if (!arr_scalar->is_valid) {
                continue;
            } else if (!global_max->is_valid) {
                global_max = arr_scalar;
            } else {
                arrow::Result<arrow::Datum> cmp_res_scalar =
                    arrow::compute::CallFunction("greater",
                                                 {arr_scalar, global_max});
                CHECK_ARROW(cmp_res_scalar.status(),
                            "Error in Arrow compute scalar comparison");
                const std::shared_ptr<arrow::Scalar> cmp_scalar =
                    cmp_res_scalar.ValueOrDie().scalar();
                if (cmp_scalar->Equals(arrow::BooleanScalar(true))) {
                    global_max = arr_scalar;
                }
            }
            results[0] = global_max;
        }
    }
};

/**
 * @brief Physical node for reductions like max.
 *
 */
class PhysicalReduce : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalReduce(std::shared_ptr<bodo::Schema> out_schema,
                            std::vector<std::string> function_names)
        // Drop Index columns since not necessary in output
        : out_schema(std::move(out_schema)),
          function_names(std::move(function_names)) {}

    virtual ~PhysicalReduce() = default;

    void FinalizeSink() override {
        for (auto& reduction_function : reduction_functions) {
            reduction_function->Finalize();
        }
    }

    void FinalizeSource() override {
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

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        time_pt start_consume_time = start_timer();
        // Convert to Arrow array
        arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
        std::shared_ptr<arrow::Array> in_arrow_array = bodo_array_to_arrow(
            bodo::BufferPool::DefaultPtr(), input_batch->columns[0],
            false /*convert_timedelta_to_int64*/, "", time_unit,
            false, /*downcast_time_ns_to_us*/
            bodo::default_buffer_memory_manager());

        if (iter == 0) {
            // Initialize reduction functions on first batch
            for (const auto& func_name : function_names) {
                if (func_name == "max") {
                    // Initialize with first value in the batch
                    reduction_functions.push_back(
                        std::make_unique<ReductionFunctionMax>(
                            arrow::ScalarVector(
                                {in_arrow_array->GetScalar(0).ValueOr(
                                    arrow::MakeNullScalar(
                                        in_arrow_array->type()))})));

                } else {
                    throw std::runtime_error(
                        "Unsupported reduction function: " + func_name);
                }
            }
        }

        for (auto& reduction_function : reduction_functions) {
            for (size_t i = 0; i < reduction_function->function_names.size();
                 i++) {
                const std::string& function_name =
                    reduction_function->function_names[i];
                const std::string& scalar_cmp_names =
                    reduction_function->reduction_names[i];
                const ReductionType& reduction_type =
                    reduction_function->reduction_types[i];
                // Current reduction result
                std::shared_ptr<arrow::Scalar>& output_scalar =
                    reduction_function->results[i];

                // Reduce Arrow array using compute function
                arrow::Result<arrow::Datum> cmp_res =
                    arrow::compute::CallFunction(function_name,
                                                 {in_arrow_array});
                CHECK_ARROW(cmp_res.status(), "Error in Arrow compute kernel");
                std::shared_ptr<arrow::Scalar> out_scalar_batch =
                    cmp_res.ValueOrDie().scalar();

                // Update reduction result
                if (!out_scalar_batch->is_valid) {
                    // If we get an empty batch which results in invalid
                    // arrow::Scalar result then just ignore it.
                    continue;
                } else if (!output_scalar->is_valid) {
                    // The last result can be null if there have been no rows
                    // seen thus far.  In which case, use the current result
                    // just on this batch as the result thus far.
                    output_scalar = out_scalar_batch;
                } else {
                    arrow::Result<arrow::Datum> cmp_res_scalar =
                        arrow::compute::CallFunction(
                            scalar_cmp_names,
                            {out_scalar_batch, output_scalar});
                    CHECK_ARROW(cmp_res_scalar.status(),
                                "Error in Arrow compute scalar comparison");
                    const std::shared_ptr<arrow::Scalar> cmp_scalar =
                        cmp_res_scalar.ValueOrDie().scalar();
                    if (reduction_type == ReductionType::COMPARISON) {
                        if (cmp_scalar->Equals(arrow::BooleanScalar(true))) {
                            output_scalar = out_scalar_batch;
                        }
                    } else if (reduction_type == ReductionType::AGGREGATION) {
                        output_scalar = cmp_scalar;
                    } else {
                        throw std::runtime_error(
                            "Unsupported reduction function: " + function_name);
                    }
                }
            }
        }

        iter++;
        this->metrics.consume_time += end_timer(start_consume_time);
        return prev_op_result == OperatorResult::FINISHED
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        throw std::runtime_error("GetResult called on a PhysicalReduce node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        time_pt start_produce_time = start_timer();
        // Create a vector of Arrow arrays from output_scalars
        std::vector<std::shared_ptr<arrow::Array>> arrow_arrays;
        for (const auto& reduction_function : reduction_functions) {
            // Every reduction function has one global output scalar after
            // Finalize which is called in FinalizeSink
            const auto& output_scalar = reduction_function->results[0];
            std::shared_ptr<arrow::Array> array =
                ScalarToArrowArray(output_scalar);
            arrow_arrays.push_back(array);
        }
        // Wrap Arrow arrays into Bodo arrays
        std::vector<std::shared_ptr<array_info>> bodo_arrays;
        for (const auto& arr : arrow_arrays) {
            std::shared_ptr<array_info> bodo_arr =
                arrow_array_to_bodo(arr, bodo::BufferPool::DefaultPtr());
            bodo_arrays.push_back(bodo_arr);
        }
        std::shared_ptr<table_info> next_batch =
            std::make_shared<table_info>(bodo_arrays);
        this->metrics.output_row_count += next_batch->nrows();
        this->metrics.produce_time += end_timer(start_produce_time);

        return {next_batch, OperatorResult::FINISHED};
    }

   private:
    /**
     * @brief Get comparison operator for comparing scalars for the reduction
     * function
     *
     * @param func_name reduction function name (e.g., "max", "min")
     * @return std::string scalar comparison operator name like "greater" or
     * "less"
     */
    static std::string getScalarOpName(std::string func_name) {
        if (func_name == "max") {
            return "greater";
        } else if (func_name == "min") {
            return "less";
        } else if (func_name == "sum") {
            return "add";
        } else if (func_name == "product") {
            return "multiply";
        } else if (func_name == "count") {
            return "add";
        } else {
            throw std::runtime_error("Unsupported reduction function: " +
                                     func_name);
        }
    }

    static std::vector<std::string> getScalarOpNames(
        const std::vector<std::string>& func_names) {
        std::vector<std::string> scalar_cmp_names;
        for (const auto& func_name : func_names) {
            scalar_cmp_names.push_back(getScalarOpName(func_name));
        }
        return scalar_cmp_names;
    }

    static ReductionType getReductionType(std::string func_name) {
        if (func_name == "max") {
            return ReductionType::COMPARISON;
        } else if (func_name == "min") {
            return ReductionType::COMPARISON;
        } else if (func_name == "sum") {
            return ReductionType::AGGREGATION;
        } else if (func_name == "product") {
            return ReductionType::AGGREGATION;
        } else if (func_name == "count") {
            return ReductionType::AGGREGATION;
        } else {
            throw std::runtime_error("Unsupported reduction function: " +
                                     func_name);
        }
    }

    static std::vector<ReductionType> getReductionTypes(
        const std::vector<std::string>& func_names) {
        std::vector<ReductionType> reduction_types;
        for (const auto& func_name : func_names) {
            reduction_types.push_back(getReductionType(func_name));
        }
        return reduction_types;
    }

    const std::shared_ptr<bodo::Schema> out_schema;
    std::vector<std::unique_ptr<ReductionFunction>> reduction_functions;
    std::vector<std::string> function_names;

    int64_t iter = 0;
    PhysicalReduceMetrics metrics;
    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        metrics_out.emplace_back(
            TimerMetric("consume_time", this->metrics.consume_time));
        metrics_out.emplace_back(
            TimerMetric("produce_time", this->metrics.produce_time));
    }
};
