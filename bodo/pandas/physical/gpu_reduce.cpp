#include "physical/gpu_reduce.h"
#include <arrow/array/util.h>
#include <mpi.h>
#include <cstdint>
#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <memory>

std::vector<std::unique_ptr<cudf::scalar>> make_vector_of_one_nullptr() {
    std::vector<std::unique_ptr<cudf::scalar>> out;
    out.emplace_back(nullptr);
    return out;
}

std::vector<std::unique_ptr<cudf::scalar>> make_vector_of_cudf_scalar(
    std::unique_ptr<cudf::scalar> scalar) {
    std::vector<std::unique_ptr<cudf::scalar>> out;
    out.emplace_back(std::move(scalar));
    return out;
}

/**
 * @brief Get cudf reduce_aggregation object corresponding to the given
 * reduction function name.
 *
 */
std::unique_ptr<cudf::reduce_aggregation> get_reduce_agg(
    const std::string& function_name) {
    if (function_name == "max") {
        return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    } else if (function_name == "min") {
        return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    } else if (function_name == "sum") {
        return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    } else if (function_name == "product") {
        return cudf::make_product_aggregation<cudf::reduce_aggregation>();
    } else if (function_name == "count") {
        return cudf::make_count_aggregation<cudf::reduce_aggregation>();
    } else {
        throw std::runtime_error("Unsupported reduction function: " +
                                 function_name);
    }
}

void GPUReductionFunction::ConsumeBatch(
    std::shared_ptr<cudf::table> input_table,
    rmm::cuda_stream_view& output_stream) {
    std::vector<std::unique_ptr<cudf::scalar>> reductions;
    for (const auto& function_name : this->function_names) {
        std::unique_ptr<cudf::reduce_aggregation> agg =
            get_reduce_agg(function_name);

        cudf::column_view input_column = input_table->get_column(0).view();
        std::unique_ptr<cudf::scalar> out_scalar =
            cudf::reduce(input_column, *agg, out_dtype, output_stream);

        reductions.push_back(std::move(out_scalar));
    }
    // Combine the batch reduction results with the current result
    this->CombineResults(reductions, output_stream);
}

void GPUReductionFunction::CombineResults(
    std::vector<std::unique_ptr<cudf::scalar>>& other,
    rmm::cuda_stream_view& output_stream) {
    assert(other.size() == this->results.size());

    for (size_t i = 0; i < this->function_names.size(); i++) {
        const std::string& function_name = this->function_names[i];
        const GPUReductionType& reduction_type = this->reduction_types[i];
        // Current reduction result
        std::unique_ptr<cudf::scalar>& result = this->results[i];
        std::unique_ptr<cudf::scalar>& other_result = other[i];

        if (other_result == nullptr || !other_result->is_valid(output_stream)) {
            // Nothing to combine from other
            continue;
        } else if (result == nullptr || !result->is_valid(output_stream)) {
            // If our current result is null then just take the other
            result = std::move(other_result);
            continue;
        }

        std::unique_ptr<cudf::reduce_aggregation> agg =
            get_reduce_agg(function_name);

        std::unique_ptr<cudf::column> col1 =
            cudf::make_column_from_scalar(*other_result, 1, output_stream);
        std::unique_ptr<cudf::column> col2 =
            cudf::make_column_from_scalar(*result, 1, output_stream);
        std::unique_ptr<cudf::column> combined = cudf::concatenate(
            std::vector<cudf::column_view>{col1->view(), col2->view()},
            output_stream);

        std::unique_ptr<cudf::scalar> cmp_scalar =
            cudf::reduce(combined->view(), *agg, out_dtype, output_stream);

        if (reduction_type == GPUReductionType::COMPARISON) {
            if (static_cast<cudf::numeric_scalar<bool>*>(cmp_scalar.get())
                    ->value(output_stream)) {
                result = std::move(other_result);
            }
        } else if (reduction_type == GPUReductionType::AGGREGATION) {
            result = std::move(cmp_scalar);
        } else {
            throw std::runtime_error("Unsupported reduction function: " +
                                     function_name);
        }
    }
}

void GPUReductionFunction::Finalize() {
    if (!is_gpu_rank()) {
        return;
    }

    MPI_Comm comm = get_gpu_mpi_comm(get_gpu_id());

    for (size_t i = 0; i < this->function_names.size(); i++) {
        // TODO(ehsan): handle empty and all null cases
        std::unique_ptr<cudf::scalar>& result = this->results[i];

        void* result_ptr =
            static_cast<cudf::numeric_scalar<int64_t>*>(result.get())->data();
        MPI_Datatype mpi_dtype = cudf_dtype_to_mpi(out_dtype);
        MPI_Allreduce(MPI_IN_PLACE, result_ptr, 1, mpi_dtype,
                      this->mpi_reduce_op, comm);
    }
}

OperatorResult PhysicalGPUReduce::ConsumeBatchGPU(
    GPU_DATA input_batch, OperatorResult prev_op_result,
    std::shared_ptr<StreamAndEvent> se) {
    time_pt start_consume_time = start_timer();

    if (iter == 0) {
        for (size_t i = 0; i < this->function_names.size(); i++) {
            const std::string func_name = this->function_names[i];
            if (func_name == "max") {
                reduction_functions.push_back(
                    std::make_unique<GPUReductionFunctionMax>(
                        this->out_schema->column_types[i]->ToArrowDataType(),
                        se->stream));

            } else if (func_name == "min") {
                reduction_functions.push_back(
                    std::make_unique<GPUReductionFunctionMin>(
                        this->out_schema->column_types[i]->ToArrowDataType(),
                        se->stream));
            } else if (func_name == "sum") {
                reduction_functions.push_back(
                    std::make_unique<GPUReductionFunctionSum>(
                        this->out_schema->column_types[i]->ToArrowDataType(),
                        se->stream));
            } else if (func_name == "product") {
                reduction_functions.push_back(
                    std::make_unique<GPUReductionFunctionProduct>(
                        this->out_schema->column_types[i]->ToArrowDataType(),
                        se->stream));
            } else if (func_name == "count") {
                reduction_functions.push_back(
                    std::make_unique<GPUReductionFunctionCount>(
                        this->out_schema->column_types[i]->ToArrowDataType(),
                        se->stream));
            } else {
                throw std::runtime_error("Unsupported reduction function: " +
                                         func_name);
            }
        }
    }

    for (auto& reduction_function : reduction_functions) {
        reduction_function->ConsumeBatch(input_batch.table, se->stream);
    }

    iter++;
    this->metrics.consume_time += end_timer(start_consume_time);
    return prev_op_result == OperatorResult::FINISHED
               ? OperatorResult::FINISHED
               : OperatorResult::NEED_MORE_INPUT;
}
