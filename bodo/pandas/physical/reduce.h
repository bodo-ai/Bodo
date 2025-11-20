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
    std::vector<std::string> function_names;
    std::vector<std::string> reduction_names;
    std::vector<ReductionType> reduction_types;
    arrow::ScalarVector results;
    arrow::DataTypeVector result_types;
    ReductionFunction(std::vector<std::string> function_names,
                      std::vector<std::string> reduction_names,
                      std::vector<ReductionType> reduction_types,
                      arrow::ScalarVector initial_results)
        : function_names(std::move(function_names)),
          reduction_names(std::move(reduction_names)),
          reduction_types(std::move(reduction_types)),
          results(std::move(initial_results)) {
        assert(this->function_names.size() == this->reduction_names.size());
        assert(this->function_names.size() == this->reduction_types.size());
        assert(this->function_names.size() == this->results.size());
        assert(!this->function_names.empty());
    }
    virtual void Finalize();
    void ConsumeBatch(std::shared_ptr<arrow::Array> in_arrow_array);
    virtual void CombineResults(const arrow::ScalarVector& other_results);
    virtual ~ReductionFunction() = default;
};

struct ReductionFunctionMax : public ReductionFunction {
    ReductionFunctionMax()
        : ReductionFunction({"max"}, {"greater"}, {ReductionType::COMPARISON},
                            {nullptr}) {}
};

struct ReductionFunctionMin : public ReductionFunction {
    ReductionFunctionMin()
        : ReductionFunction({"min"}, {"less"}, {ReductionType::COMPARISON},
                            {nullptr}) {}
};

struct ReductionFunctionSum : public ReductionFunction {
    ReductionFunctionSum(std::shared_ptr<arrow::DataType> dt)
        : ReductionFunction({"sum"}, {"add"}, {ReductionType::AGGREGATION},
                            {arrow::MakeScalar(dt, 0).ValueOrDie()}) {}
};

struct ReductionFunctionProduct : public ReductionFunction {
    ReductionFunctionProduct(std::shared_ptr<arrow::DataType> dt)
        : ReductionFunction({"product"}, {"multiply"},
                            {ReductionType::AGGREGATION},
                            {arrow::MakeScalar(dt, 1).ValueOrDie()}) {}
};

struct ReductionFunctionCount : public ReductionFunction {
    ReductionFunctionCount(std::shared_ptr<arrow::DataType> dt)
        : ReductionFunction({"count"}, {"add"}, {ReductionType::AGGREGATION},
                            {arrow::MakeScalar(dt, 0).ValueOrDie()}) {}
};

struct ReductionFunctionMean : public ReductionFunction {
    ReductionFunctionMean()
        : ReductionFunction(
              {"sum", "count"}, {"add", "add"},
              {ReductionType::AGGREGATION, ReductionType::AGGREGATION},
              {nullptr, nullptr}) {}
    void Finalize() override;
};

struct ReductionFunctionStd : public ReductionFunction {
    ReductionFunctionStd(int _ddof)
        : ReductionFunction(
              {"sum", "count", "sum_of_squares"}, {"add", "add", "add"},
              {ReductionType::AGGREGATION, ReductionType::AGGREGATION,
               ReductionType::AGGREGATION},
              {nullptr, nullptr, nullptr}),
          ddof(_ddof) {}
    void Finalize() override;

    int ddof;
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
        QueryProfileCollector::Default().SubmitOperatorName(
            PhysicalSink::getOpId(), PhysicalSink::ToString());
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
                                OperatorResult prev_op_result) override;

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        throw std::runtime_error("GetResult called on a PhysicalReduce node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        time_pt start_produce_time = start_timer();
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Create a vector of Arrow arrays from output_scalars
        std::vector<std::shared_ptr<arrow::Array>> arrow_arrays;
        for (size_t i = 0; i < reduction_functions.size(); i++) {
            const std::unique_ptr<ReductionFunction>& reduction_function =
                this->reduction_functions[i];
            // Every reduction function has one global output scalar after
            // Finalize which is called in FinalizeSink
            auto output_scalar = reduction_function->results[0];
            if (output_scalar == nullptr) {
                output_scalar = arrow::MakeNullScalar(
                    out_schema->column_types[i]->ToArrowDataType());
            }

            std::shared_ptr<arrow::Array> array =
                ScalarToArrowArray(output_scalar);
            arrow_arrays.push_back(array);
        }
        // Wrap Arrow arrays into Bodo arrays
        std::vector<std::shared_ptr<array_info>> bodo_arrays;
        for (const auto& arr : arrow_arrays) {
            // Only rank 0 returns the single row with the result, other ranks
            // return an empty array
            auto sliced_arr = arr->Slice(0, 1 ? rank == 0 : 0);
            std::shared_ptr<array_info> bodo_arr =
                arrow_array_to_bodo(sliced_arr, bodo::BufferPool::DefaultPtr());
            bodo_arrays.push_back(bodo_arr);
        }
        std::shared_ptr<table_info> next_batch =
            std::make_shared<table_info>(bodo_arrays);
        this->metrics.output_row_count += next_batch->nrows();
        this->metrics.produce_time += end_timer(start_produce_time);

        return {next_batch, OperatorResult::FINISHED};
    }

   private:
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
