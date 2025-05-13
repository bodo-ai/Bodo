#include "expression.h"

duckdb::ExpressionType exprSwitchLeftRight(duckdb::ExpressionType etype) {
    switch (etype) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return etype;
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return duckdb::ExpressionType::COMPARE_LESSTHAN;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO;
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return duckdb::ExpressionType::COMPARE_GREATERTHAN;
        default:
            throw std::runtime_error(
                "switchLeftRight doesn't handle expression type " +
                std::to_string(static_cast<int>(etype)));
    }
}

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
    const std::string& value) {
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
