#pragma once

#include <arrow/array/util.h>
#include <mpi.h>
#include <cudf/column/column_factories.hpp>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../_util.h"
#include "../libs/_query_profile_collector.h"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "operator.h"

inline bool gpu_capable_reduce(duckdb::LogicalAggregate& logical_aggregate) {
    assert(logical_aggregate.groups.empty());
    for (const auto& expr : logical_aggregate.expressions) {
        if (expr->type != duckdb::ExpressionType::BOUND_AGGREGATE) {
            throw std::runtime_error(
                "Aggregate expression is not a bound aggregate: " +
                expr->ToString());
        }
        auto& agg_expr = expr->Cast<duckdb::BoundAggregateExpression>();

        // Only support sum, max, min, product, and count for now
        if (agg_expr.function.name != "sum" &&
            agg_expr.function.name != "max" &&
            agg_expr.function.name != "min" &&
            agg_expr.function.name != "product" &&
            agg_expr.function.name != "count") {
            return false;
        }
    }
    return true;
}

enum class GPUReductionType {
    COMPARISON,
    AGGREGATION,
};

struct PhysicalGPUReduceMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    stat_t output_row_count = 0;

    time_t consume_time = 0;
    time_t produce_time = 0;
};

struct GPUReductionFunction {
    std::vector<std::string> function_names;
    std::vector<std::string> reduction_names;
    std::vector<GPUReductionType> reduction_types;
    std::vector<std::unique_ptr<cudf::scalar>> results;
    cudf::data_type out_dtype;
    MPI_Op mpi_reduce_op;
    GPUReductionFunction(
        std::vector<std::string> function_names,
        std::vector<std::string> reduction_names,
        std::vector<GPUReductionType> reduction_types,
        std::vector<std::unique_ptr<cudf::scalar>> initial_results,
        std::shared_ptr<arrow::DataType> dt, MPI_Op mpi_reduce_op)
        : function_names(std::move(function_names)),
          reduction_names(std::move(reduction_names)),
          reduction_types(std::move(reduction_types)),
          results(std::move(initial_results)),
          out_dtype(arrow_to_cudf_type(dt)),
          mpi_reduce_op(mpi_reduce_op) {
        assert(this->function_names.size() == this->reduction_names.size());
        assert(this->function_names.size() == this->reduction_types.size());
        assert(this->function_names.size() == this->results.size());
        assert(!this->function_names.empty());
    }
    virtual void Finalize(MPI_Comm comm);
    void ConsumeBatch(std::shared_ptr<cudf::table> input_table,
                      rmm::cuda_stream_view& output_stream);
    virtual void CombineResults(
        std::vector<std::unique_ptr<cudf::scalar>>& other,
        rmm::cuda_stream_view& output_stream);
    virtual ~GPUReductionFunction() = default;
};

/**
 * @brief Create a vector of one unique_ptr<cudf::scalar> that is a nullptr.
 *
 */
std::vector<std::unique_ptr<cudf::scalar>> make_vector_of_one_nullptr();

/**
 * @brief Create a vector of one unique_ptr<cudf::scalar> that contains the
 * given scalar.
 *
 */
std::vector<std::unique_ptr<cudf::scalar>> make_vector_of_cudf_scalar(
    std::unique_ptr<cudf::scalar> scalar);

struct GPUReductionFunctionMax : public GPUReductionFunction {
    GPUReductionFunctionMax(std::shared_ptr<arrow::DataType> dt,
                            rmm::cuda_stream_view& output_stream)
        : GPUReductionFunction({"max"}, {"greater"},
                               {GPUReductionType::COMPARISON},
                               make_vector_of_one_nullptr(), dt, MPI_MAX) {}
};

struct GPUReductionFunctionMin : public GPUReductionFunction {
    GPUReductionFunctionMin(std::shared_ptr<arrow::DataType> dt,
                            rmm::cuda_stream_view& output_stream)
        : GPUReductionFunction({"min"}, {"less"},
                               {GPUReductionType::COMPARISON},
                               make_vector_of_one_nullptr(), dt, MPI_MIN) {}
};

struct GPUReductionFunctionSum : public GPUReductionFunction {
    GPUReductionFunctionSum(std::shared_ptr<arrow::DataType> dt,
                            rmm::cuda_stream_view& output_stream)
        : GPUReductionFunction(
              {"sum"}, {"add"}, {GPUReductionType::AGGREGATION},
              make_vector_of_cudf_scalar(arrow_scalar_to_cudf(
                  arrow::MakeScalar(dt, 0).ValueOrDie(), output_stream)),
              dt, MPI_SUM) {}
};

struct GPUReductionFunctionProduct : public GPUReductionFunction {
    GPUReductionFunctionProduct(std::shared_ptr<arrow::DataType> dt,
                                rmm::cuda_stream_view& output_stream)
        : GPUReductionFunction(
              {"product"}, {"multiply"}, {GPUReductionType::AGGREGATION},
              make_vector_of_cudf_scalar(arrow_scalar_to_cudf(
                  arrow::MakeScalar(dt, 1).ValueOrDie(), output_stream)),
              dt, MPI_PROD) {}
};

struct GPUReductionFunctionCount : public GPUReductionFunction {
    GPUReductionFunctionCount(std::shared_ptr<arrow::DataType> dt,
                              rmm::cuda_stream_view& output_stream)
        : GPUReductionFunction(
              {"count"}, {"add"}, {GPUReductionType::AGGREGATION},
              make_vector_of_cudf_scalar(arrow_scalar_to_cudf(
                  arrow::MakeScalar(dt, 0).ValueOrDie(), output_stream)),
              dt, MPI_SUM) {}
};

/**
 * @brief Physical node for reductions like max.
 *
 */
class PhysicalGPUReduce : public PhysicalGPUSource, public PhysicalGPUSink {
   public:
    explicit PhysicalGPUReduce(std::shared_ptr<bodo::Schema> out_schema,
                               std::vector<std::string> function_names)
        : out_schema(std::move(out_schema)),
          function_names(std::move(function_names)) {}

    virtual ~PhysicalGPUReduce() = default;

    void FinalizeSink() override {
        MPI_Comm comm = get_gpu_mpi_comm(get_gpu_id());
        for (auto& reduction_function : reduction_functions) {
            reduction_function->Finalize(comm);
        }
    }

    void FinalizeSource() override {
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().SubmitOperatorName(
            PhysicalGPUSink::getOpId(), PhysicalGPUSink::ToString());
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(
                PhysicalGPUSink::getOpId(), 1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(
                PhysicalGPUSink::getOpId(), 1),
            this->metrics.output_row_count);
    }

    /**
     * @brief process input tables to groupby build (populate the build
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatchGPU(GPU_DATA input_batch,
                                   OperatorResult prev_op_result,
                                   std::shared_ptr<StreamAndEvent> se) override;

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        throw std::runtime_error(
            "GetResult called on a PhysicalGPUReduce node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

    std::pair<GPU_DATA, OperatorResult> ProduceBatchGPU(
        std::shared_ptr<StreamAndEvent> se) override {
        time_pt start_produce_time = start_timer();

        if (!is_gpu_rank()) {
            return {GPU_DATA(nullptr, out_schema->ToArrowSchema(), se),
                    OperatorResult::FINISHED};
        }

        std::vector<std::unique_ptr<cudf::column>> cols;
        for (size_t i = 0; i < reduction_functions.size(); i++) {
            std::unique_ptr<GPUReductionFunction>& reduction_function =
                this->reduction_functions[i];
            // Every reduction function has one global output scalar after
            // Finalize which is called in FinalizeSink
            auto output_scalar = std::move(reduction_function->results[0]);
            if (output_scalar == nullptr) {
                output_scalar = arrow_scalar_to_cudf(arrow::MakeNullScalar(
                    out_schema->column_types[i]->ToArrowDataType()));
            }

            std::unique_ptr<cudf::column> col1 =
                cudf::make_column_from_scalar(*output_scalar, 1, se->stream);
            cols.push_back(std::move(col1));
        }

        auto scalar_table = std::make_unique<cudf::table>(std::move(cols));
        std::pair<GPU_DATA, OperatorResult> ret = std::make_pair(
            GPU_DATA(std::move(scalar_table), out_schema->ToArrowSchema(), se),
            OperatorResult::FINISHED);
        this->metrics.produce_time += end_timer(start_produce_time);
        return ret;
    }

   private:
    const std::shared_ptr<bodo::Schema> out_schema;
    std::vector<std::unique_ptr<GPUReductionFunction>> reduction_functions;
    std::vector<std::string> function_names;

    int64_t iter = 0;
    PhysicalGPUReduceMetrics metrics;
    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        metrics_out.emplace_back(
            TimerMetric("consume_time", this->metrics.consume_time));
        metrics_out.emplace_back(
            TimerMetric("produce_time", this->metrics.produce_time));
    }
};
