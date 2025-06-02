#include "expression.h"

std::shared_ptr<arrow::Array> prepare_arrow_compute(
    std::shared_ptr<array_info> arr) {
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    return bodo_array_to_arrow(bodo::BufferPool::DefaultPtr(), arr,
                               false /*convert_timedelta_to_int64*/, "",
                               time_unit, false, /*downcast_time_ns_to_us*/
                               bodo::default_buffer_memory_manager());
}

// String specialization
std::shared_ptr<arrow::Array> CreateOneElementArrowArray(
    const std::string &value) {
    arrow::StringBuilder builder;
    arrow::Status status;
    status = builder.Append(value);
    if (!status.ok()) {
        throw std::runtime_error("builder.Append failed.");
    }
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }
    return array;
}

std::shared_ptr<arrow::Array> CreateOneElementArrowArray(
    const std::shared_ptr<arrow::Scalar> &value) {
    arrow::Result<std::shared_ptr<arrow::Array>> array_result =
        arrow::MakeArrayFromScalar(*value, 1);
    if (!array_result.ok()) {
        throw std::runtime_error("MakeArrayFromScalar failed: " +
                                 array_result.status().message());
    }
    return array_result.ValueOrDie();
}

std::shared_ptr<arrow::Array> CreateOneElementArrowArray(bool value) {
    arrow::BooleanBuilder builder;

    // Append boolean value
    arrow::Status status = builder.Append(value);
    if (!status.ok()) {
        throw std::runtime_error("builder.Append failed.");
    }

    // Finalize the Arrow array
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }

    return array;
}

std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, std::shared_ptr<ExprResult> right_res,
    const std::string &comparator) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(left_res);
    std::shared_ptr<ScalarExprResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(left_res);
    std::shared_ptr<ArrayExprResult> right_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(right_res);
    std::shared_ptr<ScalarExprResult> right_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(right_res);

    arrow::Datum src1;
    if (left_as_array) {
        src1 = arrow::Datum(prepare_arrow_compute(left_as_array->result));
    } else if (left_as_scalar) {
        src1 = arrow::MakeScalar(prepare_arrow_compute(left_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute left is neither array nor scalar.");
    }

    arrow::Datum src2;
    if (right_as_array) {
        src2 = arrow::Datum(prepare_arrow_compute(right_as_array->result));
    } else if (right_as_scalar) {
        src2 = arrow::MakeScalar(prepare_arrow_compute(right_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute right is neither array nor scalar.");
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {src1, src2});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error("do_array_compute: Error in Arrow compute: " +
                                 cmp_res.status().message());
    }

    return arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                               bodo::BufferPool::DefaultPtr());
}

std::shared_ptr<array_info> do_arrow_compute_unary(
    std::shared_ptr<ExprResult> left_res, const std::string &comparator) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(left_res);
    std::shared_ptr<ScalarExprResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(left_res);

    arrow::Datum src1;
    if (left_as_array) {
        src1 = arrow::Datum(prepare_arrow_compute(left_as_array->result));
    } else if (left_as_scalar) {
        src1 = arrow::MakeScalar(prepare_arrow_compute(left_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute left is neither array nor scalar.");
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {src1});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error("do_array_compute: Error in Arrow compute: " +
                                 cmp_res.status().message());
    }

    return arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                               bodo::BufferPool::DefaultPtr());
}

std::shared_ptr<array_info> do_arrow_compute_cast(
    std::shared_ptr<ExprResult> left_res,
    const duckdb::LogicalType &return_type) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprResult> left_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(left_res);
    std::shared_ptr<ScalarExprResult> left_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(left_res);

    arrow::Datum src1;
    if (left_as_array) {
        src1 = arrow::Datum(prepare_arrow_compute(left_as_array->result));
    } else if (left_as_scalar) {
        src1 = arrow::MakeScalar(prepare_arrow_compute(left_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute left is neither array nor scalar.");
    }

    std::shared_ptr<arrow::DataType> arrow_ret_type =
        duckdbTypeToArrow(return_type);
    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::Cast(src1, arrow_ret_type);
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error("do_array_compute: Error in Arrow compute: " +
                                 cmp_res.status().message());
    }

    return arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                               bodo::BufferPool::DefaultPtr());
}
