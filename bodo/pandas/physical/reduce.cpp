#include "physical/reduce.h"
#include <arrow/array/util.h>
#include "../libs/_shuffle.h"

void ReductionFunction::ConsumeBatch(
    std::shared_ptr<arrow::Array> in_arrow_array) {
    arrow::ScalarVector reductions;
    std::shared_ptr<arrow::Scalar> out_scalar_batch;
    for (const auto& function_name : this->function_names) {
        if (function_name == "sum_of_squares") {
            // Special case for sum_of_squares since Arrow does not have this
            // function we need to compute it manually as sum(x*x)
            arrow::Result<arrow::Datum> square_res =
                arrow::compute::CallFunction("multiply",
                                             {in_arrow_array, in_arrow_array});
            CHECK_ARROW(square_res.status(),
                        "Error in Arrow compute multiply for sum_of_squares");
            std::shared_ptr<arrow::Array> square_array =
                square_res.ValueOrDie().make_array();
            arrow::Result<arrow::Datum> sum_res =
                arrow::compute::CallFunction("sum", {square_array});
            CHECK_ARROW(sum_res.status(),
                        "Error in Arrow compute sum for sum_of_squares");
            out_scalar_batch = sum_res.ValueOrDie().scalar();
        } else {
            // Reduce Arrow array using compute function
            arrow::Result<arrow::Datum> cmp_res =
                arrow::compute::CallFunction(function_name, {in_arrow_array});
            CHECK_ARROW(cmp_res.status(), "Error in Arrow compute kernel");
            out_scalar_batch = cmp_res.ValueOrDie().scalar();
        }

        reductions.push_back(out_scalar_batch);
    }
    // Combine the batch reduction results with the current result
    this->CombineResults(reductions);
}

void ReductionFunction::CombineResults(const arrow::ScalarVector& other) {
    assert(other.size() == this->results.size());

    for (size_t i = 0; i < this->function_names.size(); i++) {
        const std::string& function_name = this->function_names[i];
        const std::string& scalar_cmp_names = this->reduction_names[i];
        const ReductionType& reduction_type = this->reduction_types[i];
        // Current reduction result
        std::shared_ptr<arrow::Scalar>& result = this->results[i];
        const std::shared_ptr<arrow::Scalar>& other_result = other[i];

        if (other_result == nullptr || !other_result->is_valid) {
            // Nothing to combine from other
            continue;
        } else if (result == nullptr || !result->is_valid) {
            // If our current result is null then just take the other
            result = other_result;
            continue;
        }

        arrow::Result<arrow::Datum> cmp_res_scalar =
            arrow::compute::CallFunction(scalar_cmp_names,
                                         {other_result, result});
        CHECK_ARROW(cmp_res_scalar.status(),
                    "Error in Arrow compute scalar comparison");
        const std::shared_ptr<arrow::Scalar> cmp_scalar =
            cmp_res_scalar.ValueOrDie().scalar();

        if (reduction_type == ReductionType::COMPARISON) {
            if (cmp_scalar->Equals(arrow::BooleanScalar(true))) {
                result = other_result;
            }
        } else if (reduction_type == ReductionType::AGGREGATION) {
            result = cmp_scalar;
        } else {
            throw std::runtime_error("Unsupported reduction function: " +
                                     function_name);
        }
    }
}

void ReductionFunction::Finalize() {
    std::vector<std::shared_ptr<arrow::Array>> global_results;
    int n_ranks, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;

    // Convert scalars to array so we can convert to bodo array for broadcast
    std::shared_ptr<arrow::Array> arrow_array;
    if (results[0] == nullptr) {
        auto null_res = arrow::MakeArrayOfNull(arrow::null(), results.size());
        CHECK_ARROW(null_res.status(),
                    "Error in MakeArrayOfNull during reduction");
        arrow_array = null_res.ValueOrDie();

    } else {
        auto arrow_array_builder_res = arrow::MakeBuilder(results[0]->type);
        auto arrow_array_builder = arrow_array_builder_res.MoveValueUnsafe();
        CHECK_ARROW(arrow_array_builder_res.status(),
                    "Error in MakeBuilder during reduction");
        auto append_status = arrow_array_builder->AppendScalars(results);
        CHECK_ARROW(append_status, "Error in AppendScalars during reduction");
        arrow::Result<std::shared_ptr<arrow::Array>> array_res =
            arrow_array_builder->Finish();
        CHECK_ARROW(array_res.status(),
                    "Error in builder finalization during reduction");
        arrow_array = array_res.ValueOrDie();
    }
    // Broadcast local reduction from each rank to all other ranks
    for (int i = 0; i < n_ranks; i++) {
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
        if (i != rank) {
            global_results.push_back(received_arrow_array);
        }
    }
    for (const auto& arr : global_results) {
        arrow::ScalarVector scalar_global_results;
        for (int64_t j = 0; j < arr->length(); j++) {
            scalar_global_results.push_back(arr->GetScalar(j).ValueOrDie());
        }
        this->CombineResults(scalar_global_results);
    }
}

OperatorResult PhysicalReduce::ConsumeBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    time_pt start_consume_time = start_timer();
    // Convert to Arrow array
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    std::shared_ptr<arrow::Array> in_arrow_array = bodo_array_to_arrow(
        bodo::BufferPool::DefaultPtr(), input_batch->columns[0],
        false /*convert_timedelta_to_int64*/, "", time_unit,
        false, /*downcast_time_ns_to_us*/
        bodo::default_buffer_memory_manager());

    if (iter == 0) {
        // Initialize reduction functions on first batch, we need to wait until
        // we have the input array to get the type for initializing the result
        // scalar.
        for (size_t i = 0; i < this->function_names.size(); i++) {
            const std::string func_name = this->function_names[i];
            if (func_name == "max") {
                // Initialize with first value in the batch, we really should do
                // this from the result of the first reduction but reductions
                // happen within the reduciton_functions and for all the
                // functions we support the result is the same type as the input
                // so this is ok for now.
                reduction_functions.push_back(
                    std::make_unique<ReductionFunctionMax>());

            } else if (func_name == "min") {
                reduction_functions.push_back(
                    std::make_unique<ReductionFunctionMin>());
            } else if (func_name == "sum") {
                reduction_functions.push_back(
                    std::make_unique<ReductionFunctionSum>(
                        this->out_schema->column_types[i]->ToArrowDataType()));
            } else if (func_name == "product") {
                reduction_functions.push_back(
                    std::make_unique<ReductionFunctionProduct>(
                        this->out_schema->column_types[i]->ToArrowDataType()));
            } else if (func_name == "count") {
                reduction_functions.push_back(
                    std::make_unique<ReductionFunctionCount>(
                        this->out_schema->column_types[i]->ToArrowDataType()));
            } else if (func_name == "mean") {
                reduction_functions.push_back(
                    std::make_unique<ReductionFunctionMean>());
            } else if (func_name == "std") {
                reduction_functions.push_back(
                    std::make_unique<ReductionFunctionStd>(1));
            } else if (func_name == "std_pop") {
                reduction_functions.push_back(
                    std::make_unique<ReductionFunctionStd>(0));
            } else {
                throw std::runtime_error("Unsupported reduction function: " +
                                         func_name);
            }
        }
    }

    for (auto& reduction_function : reduction_functions) {
        reduction_function->ConsumeBatch(in_arrow_array);
    }

    iter++;
    this->metrics.consume_time += end_timer(start_consume_time);
    return prev_op_result == OperatorResult::FINISHED
               ? OperatorResult::FINISHED
               : OperatorResult::NEED_MORE_INPUT;
}

void ReductionFunctionMean::Finalize() {
    const std::shared_ptr<arrow::Scalar>& sum_scalar = this->results[0];
    const std::shared_ptr<arrow::Scalar>& count_scalar = this->results[1];
    if (sum_scalar == nullptr || !sum_scalar->is_valid ||
        count_scalar == nullptr || !count_scalar->is_valid) {
        this->results = {std::make_shared<arrow::DoubleScalar>(0.0),
                         std::make_shared<arrow::DoubleScalar>(0.0)};
    } else {
        // Mean should always be float64 so convert sum and count to float64
        std::shared_ptr<arrow::Scalar> sum_float_scalar;
        if (sum_scalar->type->id() != arrow::Type::DOUBLE) {
            arrow::Result<arrow::Datum> cast_res = arrow::compute::Cast(
                arrow::Datum(sum_scalar), arrow::float64());
            CHECK_ARROW(cast_res.status(), "Error in Arrow compute cast");
            sum_float_scalar = cast_res.ValueOrDie().scalar();
        } else {
            sum_float_scalar = sum_scalar;
        }
        arrow::Result<arrow::Datum> cast_res =
            arrow::compute::Cast(arrow::Datum(count_scalar), arrow::float64());
        CHECK_ARROW(cast_res.status(), "Error in Arrow compute cast");
        std::shared_ptr<arrow::Scalar> count_float_scalar =
            cast_res.ValueOrDie().scalar();

        this->results = {sum_float_scalar, count_float_scalar};
    }

    // Combine across ranks
    ReductionFunction::Finalize();

    // Compute mean from sum and count
    arrow::Result<arrow::Datum> mean_res = arrow::compute::CallFunction(
        "divide",
        {arrow::Datum(this->results[0]), arrow::Datum(this->results[1])});
    CHECK_ARROW(mean_res.status(), "Error in Arrow compute divide");
    this->results = {mean_res.ValueOrDie().scalar()};
}

void ReductionFunctionStd::Finalize() {
    const std::shared_ptr<arrow::Scalar>& sum_scalar = this->results[0];
    const std::shared_ptr<arrow::Scalar>& count_scalar = this->results[1];
    const std::shared_ptr<arrow::Scalar>& sumsq_scalar = this->results[2];
    if (sum_scalar == nullptr || !sum_scalar->is_valid ||
        sumsq_scalar == nullptr || !sumsq_scalar->is_valid ||
        count_scalar == nullptr || !count_scalar->is_valid) {
        this->results = {std::make_shared<arrow::DoubleScalar>(0.0),
                         std::make_shared<arrow::DoubleScalar>(0.0),
                         std::make_shared<arrow::DoubleScalar>(0.0)};
    } else {
        // Std should always be float64 so convert sum, sumsq and count to
        // float64
        std::shared_ptr<arrow::Scalar> sum_float_scalar;
        if (sum_scalar->type->id() != arrow::Type::DOUBLE) {
            arrow::Result<arrow::Datum> cast_res = arrow::compute::Cast(
                arrow::Datum(sum_scalar), arrow::float64());
            CHECK_ARROW(cast_res.status(), "Error in Arrow compute cast");
            sum_float_scalar = cast_res.ValueOrDie().scalar();
        } else {
            sum_float_scalar = sum_scalar;
        }
        std::shared_ptr<arrow::Scalar> sumsq_float_scalar;
        if (sumsq_scalar->type->id() != arrow::Type::DOUBLE) {
            arrow::Result<arrow::Datum> cast_res = arrow::compute::Cast(
                arrow::Datum(sumsq_scalar), arrow::float64());
            CHECK_ARROW(cast_res.status(), "Error in Arrow compute cast");
            sumsq_float_scalar = cast_res.ValueOrDie().scalar();
        } else {
            sumsq_float_scalar = sumsq_scalar;
        }
        arrow::Result<arrow::Datum> cast_res =
            arrow::compute::Cast(arrow::Datum(count_scalar), arrow::float64());
        CHECK_ARROW(cast_res.status(), "Error in Arrow compute cast");
        std::shared_ptr<arrow::Scalar> count_float_scalar =
            cast_res.ValueOrDie().scalar();

        this->results = {sum_float_scalar, sumsq_float_scalar,
                         count_float_scalar};
    }

    // Combine across ranks
    ReductionFunction::Finalize();
    arrow::Datum sum_datum(this->results[0]);
    arrow::Datum sum_sq_datum(this->results[1]);
    arrow::Datum count_datum(this->results[2]);

    // --- Calculate sum^2 ---
    arrow::Result<arrow::Datum> sum_squared_res =
        arrow::compute::CallFunction("multiply", {sum_datum, sum_datum});
    CHECK_ARROW(sum_squared_res.status(), "Error in Arrow compute 'multiply'");
    arrow::Datum sum_squared_datum = sum_squared_res.ValueUnsafe();

    // --- Calculate (sum^2) / count ---
    arrow::Result<arrow::Datum> term_res = arrow::compute::CallFunction(
        "divide", {sum_squared_datum, count_datum});
    CHECK_ARROW(term_res.status(), "Error in Arrow compute 'divide'");
    arrow::Datum term_datum = term_res.ValueUnsafe();

    // --- Calculate the numerator: sum_sq - (sum^2 / count) ---
    arrow::Result<arrow::Datum> numerator_res =
        arrow::compute::CallFunction("subtract", {sum_sq_datum, term_datum});
    CHECK_ARROW(numerator_res.status(), "Error in Arrow compute 'subtract'");
    arrow::Datum numerator_datum = numerator_res.ValueUnsafe();

    // --- Calculate the denominator: count - ddof ---
    arrow::Result<arrow::Datum> denominator_res = arrow::compute::CallFunction(
        "subtract", {count_datum, arrow::Datum(ddof)});
    CHECK_ARROW(denominator_res.status(), "Error in Arrow compute 'subtract'");
    arrow::Datum denominator_datum = denominator_res.ValueUnsafe();

    // --- Calculate variance: numerator / denominator ---
    arrow::Result<arrow::Datum> variance_res = arrow::compute::CallFunction(
        "divide", {numerator_datum, denominator_datum});
    CHECK_ARROW(variance_res.status(), "Error in Arrow compute 'divide'");
    arrow::Datum variance_datum = variance_res.ValueUnsafe();

    // --- Calculate standard deviation: sqrt(variance) ---
    arrow::Result<arrow::Datum> std_dev_res =
        arrow::compute::CallFunction("sqrt", {variance_datum});
    CHECK_ARROW(std_dev_res.status(), "Error in Arrow compute 'sqrt'");
    arrow::Datum std_dev_datum = std_dev_res.ValueUnsafe();

    this->results = {std_dev_datum.scalar()};
}
