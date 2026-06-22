#include "expression.h"
#include <arrow/type_fwd.h>
#include "_util.h"
#include "duckdb/common/types/interval.hpp"

std::shared_ptr<arrow::Array> prepare_arrow_compute(
    std::shared_ptr<array_info> arr) {
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    return bodo_array_to_arrow(bodo::BufferPool::DefaultPtr(), arr,
                               false /*convert_timedelta_to_int64*/, "",
                               time_unit, false, /*downcast_time_ns_to_us*/
                               bodo::default_buffer_memory_manager());
}

#define CHECK_ARROW(expr, msg)                                \
    {                                                         \
        arrow::Status __status = expr;                        \
        if (!__status.ok()) {                                 \
            std::string err_msg =                             \
                std::string(msg) + " " + __status.ToString(); \
            throw std::runtime_error(err_msg);                \
        }                                                     \
    }

// String specialization
std::shared_ptr<arrow::Array> ScalarToArrowArray(const std::string& value,
                                                 size_t num_elements) {
    arrow::StringBuilder builder;
    arrow::Status status;
    for (size_t i = 0; i < num_elements; ++i) {
        status = builder.Append(value);
        if (!status.ok()) {
            throw std::runtime_error("builder.Append failed.");
        }
    }
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }
    return array;
}

std::shared_ptr<arrow::Array> ScalarToArrowArray(
    const std::shared_ptr<arrow::Scalar>& value, size_t num_elements) {
    arrow::Result<std::shared_ptr<arrow::Array>> array_result;
    if (value == nullptr || value->is_valid == false) {
        array_result = arrow::MakeArrayOfNull(
            value ? value->type : arrow::null(), num_elements);
    } else {
        array_result = arrow::MakeArrayFromScalar(*value, num_elements);
    }
    if (!array_result.ok()) {
        throw std::runtime_error("MakeArrayFromScalar failed: " +
                                 array_result.status().message());
    }
    return array_result.ValueOrDie();
}

std::shared_ptr<arrow::Array> ScalarToArrowArray(bool value,
                                                 size_t num_elements) {
    arrow::BooleanBuilder builder;
    arrow::Status status;

    for (size_t i = 0; i < num_elements; ++i) {
        // Append boolean value
        status = builder.Append(value);
        if (!status.ok()) {
            throw std::runtime_error("builder.Append failed.");
        }
    }

    // Finalize the Arrow array
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }

    return array;
}

// String specialization
std::shared_ptr<arrow::Array> NullArrowArray(const std::string& value,
                                             size_t num_elements) {
    arrow::StringBuilder builder;
    arrow::Status status;
    status = builder.AppendNulls(num_elements);
    if (!status.ok()) {
        throw std::runtime_error("builder.AppendNulls failed.");
    }
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }
    return array;
}

std::shared_ptr<arrow::Array> NullArrowArray(
    const std::shared_ptr<arrow::Scalar>& value, size_t num_elements) {
    arrow::Result<std::shared_ptr<arrow::Array>> array_result =
        arrow::MakeArrayOfNull(value->type, num_elements);
    if (!array_result.ok()) {
        throw std::runtime_error("MakeArrayFromScalar failed: " +
                                 array_result.status().message());
    }
    return array_result.ValueOrDie();
}

std::shared_ptr<arrow::Array> NullArrowArray(bool value, size_t num_elements) {
    arrow::BooleanBuilder builder;
    arrow::Status status;

    status = builder.AppendNulls(num_elements);
    if (!status.ok()) {
        throw std::runtime_error("builder.AppendNulls failed.");
    }

    // Finalize the Arrow array
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }

    return array;
}

arrow::Datum do_arrow_compute_multi_input_datum(
    const std::vector<std::shared_ptr<ExprResult>>& in_expr_results,
    const std::string& arrow_func_name) {
    std::vector<arrow::Datum> arg_datums;
    for (auto& expr_res : in_expr_results) {
        arrow::Datum arg_datum = ConvertExprResultToDatum(
            expr_res, "do_arrow_compute_multi_input input");
        arg_datums.push_back(arg_datum);
    }

    arrow::Result<arrow::Datum> func_res;

    if (arrow_func_name == "bodo_dateadd") {
        // DATEADD(unit, amount, date) has two pieces of behavior that Arrow
        // compute does not provide directly: calendar month arithmetic
        // (including month-end clamping) and Snowflake's rounding of fractional
        // amounts before applying the interval.  Calcite lowering passes the
        // unit as either a month multiplier or a nanosecond multiplier.
        if (arg_datums.size() != 4) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_multi_input: bodo_dateadd expects exactly 4 "
                "arguments.");
        }
        int64_t num_rows = 1;
        for (auto& datum : arg_datums) {
            if (!datum.is_scalar()) {
                num_rows = datum.length();
                break;
            }
        }

        std::shared_ptr<arrow::Array> date_arr =
            arg_datums[0].is_scalar()
                ? arrow::MakeArrayFromScalar(*arg_datums[0].scalar(), num_rows)
                      .ValueOrDie()
                : arg_datums[0].make_array();
        arrow::Result<arrow::Datum> amount_datum_res =
            arrow::compute::Cast(arg_datums[1], arrow::float64());
        if (!amount_datum_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_multi_input: Error in Arrow compute "
                "(bodo_dateadd/amount_cast): " +
                amount_datum_res.status().message());
        }
        arrow::Datum amount_datum = amount_datum_res.ValueOrDie();
        std::shared_ptr<arrow::Array> amount_arr =
            amount_datum.is_scalar()
                ? arrow::MakeArrayFromScalar(*amount_datum.scalar(), num_rows)
                      .ValueOrDie()
                : amount_datum.make_array();
        auto amount = std::static_pointer_cast<arrow::DoubleArray>(amount_arr);
        int64_t month_scale =
            arg_datums[2].is_scalar()
                ? arg_datums[2].scalar_as<arrow::Int64Scalar>().value
                : std::static_pointer_cast<arrow::Int64Array>(
                      arg_datums[2].make_array())
                      ->Value(0);
        int64_t nanos_scale =
            arg_datums[3].is_scalar()
                ? arg_datums[3].scalar_as<arrow::Int64Scalar>().value
                : std::static_pointer_cast<arrow::Int64Array>(
                      arg_datums[3].make_array())
                      ->Value(0);
        auto round_amount = [](double value) -> int64_t {
            // Snowflake rounds DATEADD amounts half away from zero before
            // applying the unit, e.g. 0.5 -> 1 and -9.5 -> -10.
            return static_cast<int64_t>(value + (value >= 0 ? 0.5 : -0.5));
        };
        auto nanos_per_unit = [](arrow::TimeUnit::type unit) -> int64_t {
            switch (unit) {
                case arrow::TimeUnit::SECOND:
                    return 1000000000LL;
                case arrow::TimeUnit::MILLI:
                    return 1000000LL;
                case arrow::TimeUnit::MICRO:
                    return 1000LL;
                case arrow::TimeUnit::NANO:
                    return 1LL;
                default:
                    throw std::runtime_error("Unknown time unit");
            }
        };

        if (date_arr->type_id() == arrow::Type::TIMESTAMP) {
            auto ts_type = std::static_pointer_cast<arrow::TimestampType>(
                date_arr->type());
            if (!ts_type->timezone().empty()) {
                throw std::runtime_error(
                    "bodo_dateadd does not support timezone-aware timestamps");
            }
            auto ts_arr =
                std::static_pointer_cast<arrow::TimestampArray>(date_arr);
            int64_t mult = nanos_per_unit(ts_type->unit());
            arrow::TimestampBuilder ts_builder(
                arrow::timestamp(arrow::TimeUnit::NANO),
                arrow::default_memory_pool());
            for (int64_t i = 0; i < num_rows; i++) {
                if (ts_arr->IsNull(i) || amount->IsNull(i)) {
                    (void)ts_builder.AppendNull();
                } else {
                    int64_t rounded = round_amount(amount->Value(i));
                    int64_t ns_val = ts_arr->Value(i) * mult;
                    if (month_scale != 0) {
                        // DuckDB's interval arithmetic gives the calendar-month
                        // semantics needed for YEAR/QUARTER/MONTH units.  Keep
                        // the nanosecond remainder because DuckDB timestamps
                        // are microsecond-based.
                        duckdb::timestamp_t ts(ns_val / 1000);
                        duckdb::interval_t interval;
                        interval.months =
                            static_cast<int32_t>(rounded * month_scale);
                        interval.days = 0;
                        interval.micros = 0;
                        duckdb::timestamp_t result =
                            duckdb::Interval::Add(ts, interval);
                        (void)ts_builder.Append(result.value * 1000 +
                                                ns_val % 1000);
                    } else {
                        (void)ts_builder.Append(ns_val + rounded * nanos_scale);
                    }
                }
            }
            auto res_arr = ts_builder.Finish();
            if (!res_arr.ok()) {
                throw std::runtime_error(res_arr.status().ToString());
            }
            return res_arr.ValueOrDie();
        }
        if (date_arr->type_id() == arrow::Type::TIME64) {
            if (month_scale != 0) {
                throw std::runtime_error(
                    "bodo_dateadd does not support calendar units for TIME");
            }
            auto time_arr =
                std::static_pointer_cast<arrow::Time64Array>(date_arr);
            const int64_t nanos_per_day = 86400000000000LL;
            auto time_type =
                std::static_pointer_cast<arrow::Time64Type>(date_arr->type());
            int64_t mult = nanos_per_unit(time_type->unit());
            arrow::Time64Builder time_builder(
                arrow::time64(arrow::TimeUnit::NANO),
                arrow::default_memory_pool());
            for (int64_t i = 0; i < num_rows; i++) {
                if (time_arr->IsNull(i) || amount->IsNull(i)) {
                    (void)time_builder.AppendNull();
                } else {
                    int64_t out =
                        (time_arr->Value(i) * mult +
                         round_amount(amount->Value(i)) * nanos_scale) %
                        nanos_per_day;
                    if (out < 0) {
                        out += nanos_per_day;
                    }
                    (void)time_builder.Append(out);
                }
            }
            auto res_arr = time_builder.Finish();
            if (!res_arr.ok()) {
                throw std::runtime_error(res_arr.status().ToString());
            }
            return res_arr.ValueOrDie();
        }
        if (date_arr->type_id() == arrow::Type::DATE32) {
            auto date32_arr =
                std::static_pointer_cast<arrow::Date32Array>(date_arr);
            const int64_t nanos_per_day = 86400000000000LL;
            // Snowflake preserves DATE output for calendar units and whole-day
            // offsets, but promotes DATE to TIMESTAMP for time/subsecond units.
            bool output_date =
                month_scale != 0 || nanos_scale % nanos_per_day == 0;
            if (output_date) {
                arrow::Date32Builder date_builder(arrow::default_memory_pool());
                for (int64_t i = 0; i < num_rows; i++) {
                    if (date32_arr->IsNull(i) || amount->IsNull(i)) {
                        (void)date_builder.AppendNull();
                    } else {
                        int64_t rounded = round_amount(amount->Value(i));
                        if (month_scale != 0) {
                            duckdb::date_t date(date32_arr->Value(i));
                            duckdb::interval_t interval;
                            interval.months =
                                static_cast<int32_t>(rounded * month_scale);
                            interval.days = 0;
                            interval.micros = 0;
                            duckdb::date_t result =
                                duckdb::Interval::Add(date, interval);
                            (void)date_builder.Append(result.days);
                        } else {
                            (void)date_builder.Append(
                                date32_arr->Value(i) +
                                rounded * (nanos_scale / nanos_per_day));
                        }
                    }
                }
                auto res_arr = date_builder.Finish();
                if (!res_arr.ok()) {
                    throw std::runtime_error(res_arr.status().ToString());
                }
                return res_arr.ValueOrDie();
            }
            arrow::TimestampBuilder ts_builder(
                arrow::timestamp(arrow::TimeUnit::NANO),
                arrow::default_memory_pool());
            for (int64_t i = 0; i < num_rows; i++) {
                if (date32_arr->IsNull(i) || amount->IsNull(i)) {
                    (void)ts_builder.AppendNull();
                } else {
                    (void)ts_builder.Append(
                        date32_arr->Value(i) * nanos_per_day +
                        round_amount(amount->Value(i)) * nanos_scale);
                }
            }
            auto res_arr = ts_builder.Finish();
            if (!res_arr.ok()) {
                throw std::runtime_error(res_arr.status().ToString());
            }
            return res_arr.ValueOrDie();
        }
        throw std::runtime_error(
            "do_arrow_compute_multi_input: bodo_dateadd unsupported input "
            "type " +
            date_arr->type()->ToString());
    } else if (arrow_func_name == "nullif") {
        // SQL NULLIF(a, b): returns NULL when a == b, else a.
        // Arrow has no direct nullif kernel, so implement as:
        //   case_when(equal(a, b), null_scalar_of_a_type, a)
        if (arg_datums.size() != 2) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_multi_input: nullif expects exactly 2 "
                "arguments.");
        }
        auto eq_res = arrow::compute::CallFunction("equal", arg_datums);
        if (!eq_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_multi_input: Error in Arrow compute "
                "(nullif/equal): " +
                eq_res.status().message());
        }

        // Build a null scalar with the same type as the first argument.
        auto null_scalar_res = arrow::MakeNullScalar(arg_datums[0].type());
        arrow::Datum null_datum(null_scalar_res);

        auto cond = eq_res.ValueOrDie();
        // Use struct array condition for case_when
        if (!cond.is_scalar()) {
            auto struct_type =
                arrow::struct_({arrow::field("cond", arrow::boolean())});
            auto cond_arr = std::make_shared<arrow::StructArray>(
                struct_type, cond.length(),
                std::vector<std::shared_ptr<arrow::Array>>{cond.make_array()});
            cond = arrow::Datum(cond_arr);
        }

        func_res = arrow::compute::CallFunction(
            "case_when", {cond, null_datum, arg_datums[0]});
    } else if (arrow_func_name == "binary_join_element_wise") {
        // binary_join_element_wise appears to require all arguments to have the
        // same type. Cast all arguments to match the first argument's type
        auto target_type = arg_datums[0].type();
        std::vector<arrow::Datum> casted_datums;
        for (auto& datum : arg_datums) {
            if (datum.type()->Equals(target_type)) {
                casted_datums.push_back(datum);
            } else {
                auto cast_opts = arrow::compute::CastOptions::Safe(target_type);
                auto cast_res =
                    arrow::compute::CallFunction("cast", {datum}, &cast_opts);
                if (!cast_res.ok()) [[unlikely]] {
                    throw std::runtime_error(
                        "do_arrow_compute_multi_input: Error casting argument "
                        "to match "
                        "binary_join_element_wise: " +
                        cast_res.status().message());
                }
                casted_datums.push_back(cast_res.ValueOrDie());
            }
        }
        func_res = arrow::compute::CallFunction(arrow_func_name, casted_datums);
    } else {
        func_res = arrow::compute::CallFunction(arrow_func_name, arg_datums);
    }

    if (!func_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_arrow_compute_multi_input: Error in Arrow compute: " +
            func_res.status().message());
    }

    return func_res.ValueOrDie();
}

std::shared_ptr<array_info> do_arrow_compute_multi_input(
    const std::vector<std::shared_ptr<ExprResult>>& in_expr_results,
    const std::string& arrow_func_name) {
    arrow::Datum result_datum =
        do_arrow_compute_multi_input_datum(in_expr_results, arrow_func_name);
    return ConvertDatumToArrayInfo(result_datum);
}

std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, std::shared_ptr<ExprResult> right_res,
    const std::string& comparator,
    const std::shared_ptr<arrow::DataType> result_type,
    bool sync_input_int_types) {
    arrow::Datum src1 =
        ConvertExprResultToDatum(left_res, "do_arrow_compute left");
    arrow::Datum src2 =
        ConvertExprResultToDatum(right_res, "do_arrow_compute right");
    arrow::Datum cmp_res_datum = do_arrow_compute_binary(
        src1, src2, comparator, result_type, sync_input_int_types);
    return ConvertDatumToArrayInfo(cmp_res_datum);
}

std::shared_ptr<array_info> do_arrow_compute_binary(
    arrow::Datum left_res, std::shared_ptr<ExprResult> right_res,
    const std::string& comparator,
    const std::shared_ptr<arrow::DataType> result_type,
    bool sync_input_int_types) {
    arrow::Datum src2 =
        ConvertExprResultToDatum(right_res, "do_arrow_compute right");
    arrow::Datum cmp_res_datum = do_arrow_compute_binary(
        left_res, src2, comparator, result_type, sync_input_int_types);
    return ConvertDatumToArrayInfo(cmp_res_datum);
}

std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, arrow::Datum right_res,
    const std::string& comparator,
    const std::shared_ptr<arrow::DataType> result_type,
    bool sync_input_int_types) {
    arrow::Datum src1 =
        ConvertExprResultToDatum(left_res, "do_arrow_compute left");
    arrow::Datum cmp_res_datum = do_arrow_compute_binary(
        src1, right_res, comparator, result_type, sync_input_int_types);
    return ConvertDatumToArrayInfo(cmp_res_datum);
}

std::shared_ptr<array_info> do_arrow_compute_unary(
    std::shared_ptr<ExprResult> left_res, const std::string& comparator,
    const arrow::compute::FunctionOptions* func_options) {
    arrow::Datum src1 =
        ConvertExprResultToDatum(left_res, "do_arrow_compute left");
    arrow::Datum cmp_res =
        do_arrow_compute_unary(src1, comparator, func_options);
    return ConvertDatumToArrayInfo(cmp_res);
}

std::shared_ptr<array_info> do_arrow_compute_cast(
    std::shared_ptr<ExprResult> left_res,
    const duckdb::LogicalType& return_type) {
    arrow::Datum src1 =
        ConvertExprResultToDatum(left_res, "do_arrow_compute left");

    std::shared_ptr<arrow::DataType> arrow_ret_type =
        duckdbTypeToArrow(return_type);
    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::Cast(src1, arrow_ret_type);
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_cast: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return ConvertDatumToArrayInfo(cmp_res.ValueOrDie());
}

arrow::Datum do_arrow_compute_binary(
    arrow::Datum left_res, arrow::Datum right_res,
    const std::string& comparator,
    const std::shared_ptr<arrow::DataType> result_type,
    bool sync_input_int_types) {
    if (sync_input_int_types) {
        std::tie(left_res, right_res) = CastIntDatumsToCommonType(
            "do_arrow_compute_binary arg", left_res, right_res);
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {left_res, right_res});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_binary: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    arrow::Datum cmp_datum = cmp_res.ValueOrDie();

    std::shared_ptr<arrow::DataType> cmp_dtype = cmp_datum.type();
    if (result_type && cmp_dtype != result_type) {
        // Cast to result type if available and different from current type.
        arrow::compute::CastOptions cast_opts;
        cast_opts.allow_int_overflow = true;
        arrow::Result<arrow::Datum> cast_res =
            arrow::compute::Cast(cmp_datum, result_type, cast_opts);
        if (!cast_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_binary cast_res: Error in Arrow compute: " +
                cast_res.status().message());
        }
        cmp_res = cast_res;
    }

    return cmp_res.ValueOrDie();
}

arrow::Datum do_arrow_compute_unary(
    arrow::Datum src1, const std::string& comparator,
    const arrow::compute::FunctionOptions* func_options) {
    // Special handling for is_not_null since it is not supported directly
    // by Arrow compute.
    if (comparator == "is_not_null") {
        arrow::Result<arrow::Datum> is_null_res =
            arrow::compute::CallFunction("is_null", {src1}, func_options);
        if (!is_null_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_unary: Error in Arrow compute: " +
                is_null_res.status().message());
        }

        // Invert the boolean array
        arrow::Result<arrow::Datum> invert_res =
            arrow::compute::CallFunction("invert", {is_null_res.ValueOrDie()});
        if (!invert_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_unary: Error in Arrow compute Invert: " +
                invert_res.status().message());
        }
        return invert_res.ValueOrDie();
    }

    // Special handling for is_true since it is not supported directly
    // by Arrow compute.
    if (comparator == "is_true") {
        auto arrow_false = arrow::MakeScalar(false);
        arrow::Result<arrow::Datum> is_true_res = arrow::compute::CallFunction(
            "coalesce", {src1, arrow_false}, func_options);
        if (!is_true_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_unary: Error in Arrow compute: " +
                is_true_res.status().message());
        }
        return is_true_res.ValueOrDie();
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {src1}, func_options);
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_arrow_compute_unary: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return cmp_res.ValueOrDie();
}

arrow::Datum do_arrow_compute_cast(arrow::Datum left_res,
                                   const duckdb::LogicalType& return_type) {
    std::shared_ptr<arrow::DataType> arrow_ret_type =
        duckdbTypeToArrow(return_type);
    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::Cast(left_res, arrow_ret_type);
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_array_compute_cast: Error in Arrow compute: " +
            cmp_res.status().message());
    }

    return cmp_res.ValueOrDie();
}

std::shared_ptr<array_info> do_arrow_compute_case(
    std::shared_ptr<ExprResult> when_res, std::shared_ptr<ExprResult> then_res,
    std::shared_ptr<ExprResult> else_res,
    const std::shared_ptr<arrow::DataType> result_type) {
    // Try to convert the results of our children into array
    // or scalar results to see which one they are.
    std::shared_ptr<ArrayExprResult> when_as_array =
        std::dynamic_pointer_cast<ArrayExprResult>(when_res);
    std::shared_ptr<ScalarExprResult> when_as_scalar =
        std::dynamic_pointer_cast<ScalarExprResult>(when_res);

    arrow::Datum src1;
    if (when_as_array) {
        std::shared_ptr<arrow::Array> arr =
            prepare_arrow_compute(when_as_array->result);

        // Wrap the boolean array into a struct array with one child,
        // as required by Arrow's "case_when" kernel.
        auto struct_type = arrow::struct_({arrow::field("cond", arr->type())});
        arr = std::make_shared<arrow::StructArray>(
            struct_type, arr->length(),
            std::vector<std::shared_ptr<arrow::Array>>{arr});

        src1 = arrow::Datum(arr);
    } else if (when_as_scalar) {
        src1 = arrow::MakeScalar(prepare_arrow_compute(when_as_scalar->result)
                                     ->GetScalar(0)
                                     .ValueOrDie());
    } else {
        throw std::runtime_error(
            "do_arrow_compute when is neither array nor scalar.");
    }

    arrow::Datum src2 =
        ConvertExprResultToDatum(then_res, "do_arrow_compute then");
    arrow::Datum src3 =
        ConvertExprResultToDatum(else_res, "do_arrow_compute else");

    // Make input integer datums unsigned so that Arrow doesn't attempt a
    // safe conversion to signed int that isn't possible
    std::tie(src2, src3) =
        CastIntDatumsToCommonType("do_arrow_compute_case arg", src2, src3);

    // NOTE: Arrow's "if_else" doesn't match our Python and SQL semantics since
    // it propagates nulls in the condition.
    arrow::Result<arrow::Datum> case_res =
        arrow::compute::CallFunction("case_when", {src1, src2, src3});
    if (!case_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_arrow_compute_case case_when: Error in Arrow compute: " +
            case_res.status().message());
    }

    arrow::Datum case_datum = case_res.ValueOrDie();
    std::shared_ptr<arrow::DataType> case_dtype = case_datum.type();
    if (result_type && case_dtype != result_type) {
        // Cast to result type if available and different from current type.
        arrow::compute::CastOptions cast_opts;
        cast_opts.allow_int_overflow = true;
        arrow::Result<arrow::Datum> cast_res =
            arrow::compute::Cast(case_datum, result_type, cast_opts);
        if (!cast_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_case cast_res: Error in Arrow compute: " +
                cast_res.status().message());
        }
        case_res = cast_res;
    }

    return ConvertDatumToArrayInfo(case_res.ValueOrDie());
}

std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
    duckdb::unique_ptr<duckdb::Expression>& expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>& col_ref_map,
    bool no_scalars);

std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
    duckdb::Expression& expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>& col_ref_map,
    bool no_scalars) {
    // Class and type here are really like the general type of the
    // expression node (expr_class) and a sub-type of that general
    // type (expr_type).
    duckdb::ExpressionClass expr_class = expr.GetExpressionClass();
    duckdb::ExpressionType expr_type = expr.GetExpressionType();

    switch (expr_class) {
        case duckdb::ExpressionClass::BOUND_COMPARISON: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundComparisonExpression>();
            // This node type has left and right children which are recursively
            // processed first and then the resulting Bodo Physical expression
            // subtrees are combined with the expression sub-type (e.g., equal,
            // greater_than, less_than) to make the Bodo PhysicalComparisonExpr.
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalComparisonExpression>(
                    buildPhysicalExprTree(bce.left, col_ref_map, no_scalars),
                    buildPhysicalExprTree(bce.right, col_ref_map, no_scalars),
                    expr_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_COLUMN_REF: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundColumnRefExpression>();
            duckdb::ColumnBinding binding = bce.binding;
            size_t col_idx = col_ref_map_lookup(
                col_ref_map, binding.table_index, binding.column_index);
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalColumnRefExpression>(col_idx, binding,
                                                              bce.GetName()));
            // binding.table_index, binding.column_index));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CONSTANT: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundConstantExpression>();
            if (bce.value.IsNull()) {
                // Get the constant out of the duckdb node as a C++ variant.
                // Using auto since variant set will be extended.
                auto extracted_value =
                    getDefaultValueForDuckdbValueType(bce.value);
                // Return a PhysicalConstantExpression<T> where T is the actual
                // type of the value contained within bce.value.
                auto ret = std::visit(
                    [no_scalars](const auto& value) {
                        return std::static_pointer_cast<PhysicalExpression>(
                            std::make_shared<PhysicalNullExpression<
                                std::decay_t<decltype(value)>>>(value,
                                                                no_scalars));
                    },
                    extracted_value);
                return ret;
            } else {
                // Get the constant out of the duckdb node as a C++ variant.
                // Using auto since variant set will be extended.
                auto extracted_value = extractValue(bce.value);
                // Return a PhysicalConstantExpression<T> where T is the actual
                // type of the value contained within bce.value.
                auto ret = std::visit(
                    [no_scalars](const auto& value) {
                        return std::static_pointer_cast<PhysicalExpression>(
                            std::make_shared<PhysicalConstantExpression<
                                std::decay_t<decltype(value)>>>(value,
                                                                no_scalars));
                    },
                    extracted_value);
                return ret;
            }
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CONJUNCTION: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundConjunctionExpression>();
            // This node type has left and right children which are recursively
            // processed first and then the resulting Bodo Physical expression
            // subtrees are combined with the expression sub-type (e.g., equal,
            // greater_than, less_than) to make the Bodo PhysicalComparisonExpr.
            int left_child = 0;
            int right_child = 1;
            // With short-circuit evaluation, make expensive bound_function
            // operators be on the right side.
            if (bce.children[0]->GetExpressionClass() ==
                duckdb::ExpressionClass::BOUND_FUNCTION) {
                left_child = 1;
                right_child = 0;
            }
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalConjunctionExpression>(
                    buildPhysicalExprTree(bce.children[left_child], col_ref_map,
                                          no_scalars),
                    buildPhysicalExprTree(bce.children[right_child],
                                          col_ref_map, no_scalars),
                    expr_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_OPERATOR: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& boe = expr.Cast<duckdb::BoundOperatorExpression>();
            switch (boe.children.size()) {
                case 1: {
                    return std::static_pointer_cast<PhysicalExpression>(
                        std::make_shared<PhysicalUnaryExpression>(
                            buildPhysicalExprTree(boe.children[0], col_ref_map,
                                                  no_scalars),
                            expr_type));
                } break;
                case 2: {
                    return std::static_pointer_cast<PhysicalExpression>(
                        std::make_shared<PhysicalBinaryExpression>(
                            buildPhysicalExprTree(boe.children[0], col_ref_map,
                                                  no_scalars),
                            buildPhysicalExprTree(boe.children[1], col_ref_map,
                                                  no_scalars),
                            expr_type));
                } break;
                default:
                    throw std::runtime_error(
                        "Unsupported number of children for bound operator");
            }
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_FUNCTION: {
            // Convert the base duckdb::Expression node to its actual derived
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bfe = expr.Cast<duckdb::BoundFunctionExpression>();
            std::shared_ptr<arrow::DataType> result_type = nullptr;

            if (bfe.bind_info) {
                BodoScalarFunctionData& scalar_func_data =
                    bfe.bind_info->Cast<BodoScalarFunctionData>();
                result_type = scalar_func_data.out_schema->field(0)->type();
            }

            if (bfe.bind_info &&
                (bfe.bind_info->Cast<BodoScalarFunctionData>().args ||
                 !bfe.bind_info->Cast<BodoScalarFunctionData>()
                      .arrow_func_name.empty())) {
                BodoScalarFunctionData& scalar_func_data =
                    bfe.bind_info->Cast<BodoScalarFunctionData>();

                std::vector<std::shared_ptr<PhysicalExpression>> phys_children;
                for (auto& child_expr : bfe.children) {
                    phys_children.emplace_back(buildPhysicalExprTree(
                        child_expr, col_ref_map, no_scalars));
                }

                if (!scalar_func_data.arrow_func_name.empty()) {
                    return std::static_pointer_cast<PhysicalExpression>(
                        std::make_shared<PhysicalArrowExpression>(
                            phys_children, scalar_func_data, result_type));
                } else if (scalar_func_data.args) {
                    return std::static_pointer_cast<PhysicalExpression>(
                        std::make_shared<PhysicalUDFExpression>(
                            phys_children, scalar_func_data, result_type));
                }
            } else {
                switch (bfe.children.size()) {
                    case 1: {
                        return std::static_pointer_cast<PhysicalExpression>(
                            std::make_shared<PhysicalUnaryExpression>(
                                buildPhysicalExprTree(bfe.children[0],
                                                      col_ref_map, no_scalars),
                                bfe.function.name));
                    } break;
                    case 2: {
                        // Check for calendar interval constants that
                        // Arrow's duration-based add cannot handle
                        // (because Arrow always promotes DATE to TIMESTAMP,
                        // and cannot handle month-bearing intervals at all).
                        for (int ci = 0; ci < 2; ci++) {
                            if (bfe.children[ci]->GetExpressionClass() ==
                                duckdb::ExpressionClass::BOUND_CONSTANT) {
                                auto& const_expr =
                                    bfe.children[ci]
                                        ->Cast<
                                            duckdb::BoundConstantExpression>();
                                if (!const_expr.value.IsNull() &&
                                    const_expr.value.type().id() ==
                                        duckdb::LogicalTypeId::INTERVAL) {
                                    duckdb::interval_t interval =
                                        const_expr.value
                                            .GetValue<duckdb::interval_t>();
                                    if (interval.months != 0) {
                                        if (bfe.function.name != "add" &&
                                            bfe.function.name != "+" &&
                                            bfe.function.name != "subtract" &&
                                            bfe.function.name != "-") {
                                            throw std::runtime_error(
                                                "Only addition and subtraction "
                                                "are supported for "
                                                "month-bearing calendar "
                                                "intervals.");
                                        }
                                        int date_child_idx = 1 - ci;
                                        bool is_sub =
                                            (bfe.function.name == "subtract" ||
                                             bfe.function.name == "-");
                                        return std::static_pointer_cast<
                                            PhysicalExpression>(
                                            std::make_shared<
                                                PhysicalCalendarIntervalExpression>(
                                                buildPhysicalExprTree(
                                                    bfe.children
                                                        [date_child_idx],
                                                    col_ref_map, no_scalars),
                                                interval, is_sub, result_type));
                                    }
                                }
                            }
                        }
                        return std::static_pointer_cast<PhysicalExpression>(
                            std::make_shared<PhysicalBinaryExpression>(
                                buildPhysicalExprTree(bfe.children[0],
                                                      col_ref_map, no_scalars),
                                buildPhysicalExprTree(bfe.children[1],
                                                      col_ref_map, no_scalars),
                                bfe.function.name, result_type));
                    } break;
                    default:
                        throw std::runtime_error(
                            "Unsupported number of children " +
                            std::to_string(bfe.children.size()) +
                            " for bound function");
                }
            }
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CAST: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundCastExpression>();
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalCastExpression>(
                    buildPhysicalExprTree(bce.child, col_ref_map, no_scalars),
                    bce.return_type));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_BETWEEN: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bbe = expr.Cast<duckdb::BoundBetweenExpression>();
            // Convert to conjunction and comparison nodes.
            std::shared_ptr<PhysicalExpression> input_expr =
                buildPhysicalExprTree(bbe.input, col_ref_map, no_scalars);
            std::shared_ptr<PhysicalExpression> lower_expr =
                buildPhysicalExprTree(bbe.lower, col_ref_map, no_scalars);
            std::shared_ptr<PhysicalExpression> upper_expr =
                buildPhysicalExprTree(bbe.upper, col_ref_map, no_scalars);

            std::shared_ptr<PhysicalExpression> left = std::static_pointer_cast<
                PhysicalExpression>(
                std::make_shared<PhysicalComparisonExpression>(
                    input_expr, lower_expr,
                    bbe.lower_inclusive
                        ? duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO
                        : duckdb::ExpressionType::COMPARE_GREATERTHAN));

            std::shared_ptr<PhysicalExpression> right =
                std::static_pointer_cast<PhysicalExpression>(
                    std::make_shared<PhysicalComparisonExpression>(
                        upper_expr, input_expr,
                        bbe.upper_inclusive
                            ? duckdb::ExpressionType::
                                  COMPARE_GREATERTHANOREQUALTO
                            : duckdb::ExpressionType::COMPARE_GREATERTHAN));

            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalConjunctionExpression>(
                    left, right, duckdb::ExpressionType::CONJUNCTION_AND));
        } break;  // suppress wrong fallthrough error
        case duckdb::ExpressionClass::BOUND_CASE: {
            // Convert the base duckdb::Expression node to its actual derived
            // type.
            auto& bce = expr.Cast<duckdb::BoundCaseExpression>();
            if (bce.case_checks.size() != 1) {
                throw std::runtime_error(
                    "Only single WHEN case expressions are supported.");
            }
            auto& caseCheck = bce.case_checks[0];
            return std::static_pointer_cast<PhysicalExpression>(
                std::make_shared<PhysicalCaseExpression>(
                    buildPhysicalExprTree(caseCheck.when_expr, col_ref_map,
                                          no_scalars),
                    buildPhysicalExprTree(caseCheck.then_expr, col_ref_map,
                                          no_scalars),
                    buildPhysicalExprTree(bce.else_expr, col_ref_map,
                                          no_scalars),
                    duckdbTypeToArrow(bce.return_type)));
        } break;  // suppress wrong fallthrough error
        default:
            throw std::runtime_error(
                "Unsupported duckdb expression class " +
                std::to_string(static_cast<int>(expr_class)));
    }
    throw std::logic_error("Control should never reach here");
}

std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
    duckdb::unique_ptr<duckdb::Expression>& expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>& col_ref_map,
    bool no_scalars) {
    return buildPhysicalExprTree(*expr, col_ref_map, no_scalars);
}

std::shared_ptr<ExprResult> PhysicalUDFExpression::ProcessBatch(
    std::shared_ptr<table_info> input_batch) {
    std::vector<std::shared_ptr<array_info>> child_results;
    std::vector<std::string> column_names;

    // All the sources of the UDF will be separate projections.
    // Create each one of them here.
    for (const auto& child : children) {
        std::shared_ptr<ExprResult> child_res =
            child->ProcessBatch(input_batch);

        std::shared_ptr<ArrayExprResult> child_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(child_res);
        std::shared_ptr<ScalarExprResult> child_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(child_res);

        if (child_as_array) {
            child_results.emplace_back(child_as_array->result);
            column_names.emplace_back(child_as_array->column_name);
        } else if (child_as_scalar) {
            child_results.emplace_back(child_as_scalar->result);
            column_names.emplace_back("scalar");
        } else {
            throw std::runtime_error(
                "Child of UDF did not return an array or scalar.");
        }
    }
    // Put them all back together for the UDF to process.
    std::shared_ptr<table_info> udf_input = std::make_shared<table_info>(
        child_results, column_names, input_batch->metadata);

    // Actually run the UDF.
    std::shared_ptr<table_info> udf_output;
    if (cfunc_ptr) {
        if (cfunc_ptr == (table_udf_t)1) {
            PyThreadState* save = PyEval_SaveThread();
            cfunc_ptr = compile_future.get();
            PyEval_RestoreThread(save);
        }
        time_pt start_init_time = start_timer();
        udf_output = runCfuncScalarFunction(udf_input, cfunc_ptr);
        this->metrics.udf_execution_time += end_timer(start_init_time);
    } else {
        auto [out_temp, cpp_to_py_time, udf_time, py_to_cpp_time] =
            runPythonScalarFunction(udf_input, result_type,
                                    scalar_func_data.args,
                                    scalar_func_data.has_state, init_state);
        udf_output = out_temp;
        // Update the metrics.
        this->metrics.cpp_to_py_time += cpp_to_py_time;
        this->metrics.udf_execution_time += udf_time;
        this->metrics.py_to_cpp_time += py_to_cpp_time;
    }

    return std::make_shared<ArrayExprResult>(udf_output->columns[0],
                                             udf_output->column_names[0]);
}

std::shared_ptr<ExprResult> PhysicalArrowExpression::ProcessBatch(
    std::shared_ptr<table_info> input_batch) {
    std::shared_ptr<array_info> result;
    // BodoSQL functions may have multiple arguments. TODO(Ehsan): refactor
    // various Arrow compute call code paths.
    if (children.size() > 1) {
        std::vector<std::shared_ptr<ExprResult>> in_expr_results;
        for (const auto& child : children) {
            in_expr_results.emplace_back(child->ProcessBatch(input_batch));
        }
        time_pt start_init_time = start_timer();
        result = do_arrow_compute_multi_input(in_expr_results,
                                              scalar_func_data.arrow_func_name);
        this->metrics.arrow_compute_time += end_timer(start_init_time);
    } else {
        std::shared_ptr<ExprResult> res =
            children[0]->ProcessBatch(input_batch);
        time_pt start_init_time = start_timer();
        result = this->do_arrow_compute(res);
        this->metrics.arrow_compute_time += end_timer(start_init_time);
    }

    // Broadcast scalar result to batch size
    if (result->length == 1 && input_batch->nrows() > 1) {
        std::shared_ptr<arrow::Array> arrow_arr = prepare_arrow_compute(result);
        auto scalar = arrow_arr->GetScalar(0).ValueOrDie();
        auto broadcast =
            arrow::MakeArrayFromScalar(*scalar, input_batch->nrows());
        if (!broadcast.ok()) {
            throw std::runtime_error("Failed to broadcast scalar: " +
                                     broadcast.status().message());
        }
        result = arrow_array_to_bodo(broadcast.ValueOrDie(),
                                     bodo::BufferPool::DefaultPtr());
    }

    return std::make_shared<ArrayExprResult>(result, "Arrow Scalar");
}

bool PhysicalExpression::join_expr(array_info** left_table,
                                   array_info** right_table, void** left_data,
                                   void** right_data, void** left_null_bitmap,
                                   void** right_null_bitmap, int64_t left_index,
                                   int64_t right_index) {
    arrow::Datum res = cur_join_expr->join_expr_internal(
        left_table, right_table, left_data, right_data, left_null_bitmap,
        right_null_bitmap, left_index, right_index);
    if (!res.is_scalar()) {
        throw std::runtime_error("join_expr_internal did not return scalar.");
    }
    if (res.scalar()->type->id() != arrow::Type::BOOL) {
        throw std::runtime_error("join_expr_internal did not return bool.");
    }
    auto bool_scalar =
        std::dynamic_pointer_cast<arrow::BooleanScalar>(res.scalar());
    if (bool_scalar && bool_scalar->is_valid) {
        return bool_scalar->value;
    } else {
        throw std::runtime_error("join_expr_internal bool is null or invalid.");
    }
}

void PhysicalExpression::join_expr_batch(
    array_info** left_table, array_info** right_table, void** left_data,
    void** right_data, void** left_null_bitmap, void** right_null_bitmap,
    uint8_t* match_arr, int64_t left_index_start, int64_t left_index_end,
    int64_t right_index_start, int64_t right_index_end) {
    for (int64_t j = right_index_start; j < right_index_end; j++) {
        for (int64_t i = left_index_start; i < left_index_end; i++) {
            SetBitTo(match_arr,
                     (i - left_index_start) + (j - right_index_start),
                     join_expr(left_table, right_table, left_data, right_data,
                               left_null_bitmap, right_null_bitmap, i, j));
        }
    }
}

PhysicalExpression* PhysicalExpression::cur_join_expr = nullptr;

template <typename ArrowType, typename ModOp>
arrow::Status ModImpl(arrow::compute::KernelContext* ctx,
                      const arrow::compute::ExecSpan& batch,
                      arrow::compute::ExecResult* out) {
    using CType = typename ArrowType::c_type;
    using ScalarType = typename arrow::TypeTraits<ArrowType>::ScalarType;

    // Extract left array (it has to be an array).
    const arrow::ArraySpan& left = batch[0].array;
    // Extract right element...could be scalar or array.
    const arrow::compute::ExecValue& right_span = batch[1];

    // Make sure it's an array.
    if (!left.type) {
        throw std::runtime_error("ModInt left.type not valid.");
    }

    arrow::Status status;
    // Get raw pointers to values and null bits for left array.
    const CType* left_values = left.GetValues<CType>(1);
    const uint8_t* left_valid_bits = left.buffers[0].data;

    auto is_valid_bit = [](const uint8_t* bits, int64_t offset,
                           int64_t i) -> bool {
        return !bits || arrow::bit_util::GetBit(bits, offset + i);
    };

    // Output array is preallocated and comes in as an ArraySpan in the
    // out->value variant.
    arrow::ArraySpan& out_span = std::get<arrow::ArraySpan>(out->value);
    // Get raw pointers to values and null bits of the output array.
    CType* out_values = out_span.GetValues<CType>(1);
    uint8_t* out_valid_bits = out_span.buffers[0].data;
    int64_t offset = out_span.offset;

    auto set_valid = [](uint8_t* bits, int64_t i) {
        if (bits)
            arrow::bit_util::SetBit(bits, i);
    };

    auto clear_valid = [](uint8_t* bits, int64_t i) {
        if (bits)
            arrow::bit_util::ClearBit(bits, i);
    };

    if (right_span.is_scalar()) {
        // Right side is a scalar
        const arrow::Scalar* scalar = right_span.scalar;
        if (!scalar || !scalar->is_valid) {
            // If scalar is null, all outputs are null
            for (int64_t i = 0; i < left.length; ++i) {
                clear_valid(out_valid_bits, offset + i);
            }
        } else {
            // Get right value as a scalar.
            const ScalarType& sc = right_span.scalar_as<ScalarType>();
            // Extract value from the scalar.
            CType r = sc.value;
            // For each element of the left array.
            for (int64_t i = 0; i < left.length; ++i) {
                if (!is_valid_bit(left_valid_bits, left.offset, i)) {
                    clear_valid(out_valid_bits, offset + i);
                } else {
                    // Get ith element of left.
                    CType l = left_values[i];
                    // Calculate modulus operator.
                    CType res = ModOp::apply(l, r);
                    // Assign result.
                    out_values[i] = res;
                    // Indicate index has valid data.
                    set_valid(out_valid_bits, offset + i);
                }
            }
        }
    } else {
        // Right side is an array so extract it.
        const arrow::ArraySpan& right = right_span.array;
        // Get raw pointers to values and null bits for right array.
        const CType* right_values = right.GetValues<CType>(1);
        const uint8_t* right_valid_bits = right.buffers[0].data;
        // For each element of the left array.
        for (int64_t i = 0; i < left.length; ++i) {
            if (!is_valid_bit(left_valid_bits, left.offset, i) ||
                !is_valid_bit(right_valid_bits, right.offset, i)) {
                clear_valid(out_valid_bits, offset + i);
            } else {
                // Get corresponding ith elements of left and right arrays.
                CType l = left_values[i];
                CType r = right_values[i];
                // Calculate modulus operator.
                CType res = ModOp::apply(l, r);
                // Assign result.
                out_values[i] = res;
                // Indicate index has valid data.
                set_valid(out_valid_bits, offset + i);
            }
        }
    }

    return arrow::Status::OK();
}

struct NativeMod {
    template <typename T>
    static T apply(T l, T r) {
        return (r == 0 ? 0 : (l % r));
    }
};

struct AltMod {
    template <typename T>
    static T apply(T l, T r) {
        return (r == 0 ? 0 : (l - ((int64_t)(l / r) * r)));
    }
};

void RegisterMod(arrow::compute::FunctionRegistry* registry) {
    // Declare the binary arrow compute function named "bodo_mod".
    auto func = std::make_shared<arrow::compute::ScalarFunction>(
        "bodo_mod", arrow::compute::Arity::Binary(),
        arrow::compute::FunctionDoc{
            "Modulo of two arrays", "Returns lhs % rhs", {"lhs", "rhs"}});

    // Declare int32,int32->int32 mod kernel.
    arrow::compute::ScalarKernel kernel32(
        {arrow::compute::InputType(arrow::int32()),
         arrow::compute::InputType(arrow::int32())},
        arrow::compute::OutputType(arrow::int32()),
        ModImpl<arrow::Int32Type, NativeMod>);
    // Declare int64,int64->int64 mod kernel.
    arrow::compute::ScalarKernel kernel64(
        {arrow::compute::InputType(arrow::int64()),
         arrow::compute::InputType(arrow::int64())},
        arrow::compute::OutputType(arrow::int64()),
        ModImpl<arrow::Int64Type, NativeMod>);
    // Declare float,float->float mod kernel.
    arrow::compute::ScalarKernel floatkernel32(
        {arrow::compute::InputType(arrow::float32()),
         arrow::compute::InputType(arrow::float32())},
        arrow::compute::OutputType(arrow::float32()),
        ModImpl<arrow::FloatType, AltMod>);
    // Declare double,double->double mod kernel.
    arrow::compute::ScalarKernel floatkernel64(
        {arrow::compute::InputType(arrow::float64()),
         arrow::compute::InputType(arrow::float64())},
        arrow::compute::OutputType(arrow::float64()),
        ModImpl<arrow::DoubleType, AltMod>);

    arrow::Status status;
    // Add all the above kernels to the function.
    status = func->AddKernel(kernel32);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod 32 AddKernel failed.");
    }
    status = func->AddKernel(kernel64);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod 64 AddKernel failed.");
    }
    status = func->AddKernel(floatkernel32);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod 32 AddKernel failed.");
    }
    status = func->AddKernel(floatkernel64);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod 64 AddKernel failed.");
    }
    // Register the function.
    status = registry->AddFunction(std::move(func));
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod AddFunction failed.");
    }
}

void EnsureModRegistered() {
    static std::once_flag flag;
    // Register the mod arrow compute function only once.
    std::call_once(flag, [] {
        auto* registry = arrow::compute::GetFunctionRegistry();
        RegisterMod(registry);
    });
}

std::shared_ptr<ExprResult> PhysicalCalendarIntervalExpression::ProcessBatch(
    std::shared_ptr<table_info> input_batch) {
    // Evaluate the date-side child expression to get the operand.
    auto child_res = date_child->ProcessBatch(input_batch);

    // If this is a subtraction (date - interval), invert the interval so we
    // can always use DuckDB's Interval::Add() for uniform handling.
    duckdb::interval_t effective_interval = calendar_interval;
    if (is_subtract) {
        effective_interval = duckdb::Interval::Invert(effective_interval);
    }

    // Extract the typed concrete result (array or scalar) from the child.
    auto child_arr = std::dynamic_pointer_cast<ArrayExprResult>(child_res);
    auto child_scalar = std::dynamic_pointer_cast<ScalarExprResult>(child_res);

    // Convert the child result to an Arrow array for element-wise processing.
    // Scalars are wrapped into single-element arrays but we track `is_scalar`
    // so we can return the same result shape as the input.
    std::shared_ptr<arrow::Array> arrow_arr;
    bool is_scalar = false;
    if (child_arr) {
        arrow_arr = prepare_arrow_compute(child_arr->result);
    } else if (child_scalar) {
        arrow_arr = prepare_arrow_compute(child_scalar->result);
        is_scalar = true;
    } else {
        throw std::runtime_error(
            "PhysicalCalendarIntervalExpression: child is neither array "
            "nor scalar");
    }

    int64_t num_rows = arrow_arr->length();
    auto arrow_type = arrow_arr->type();

    if (arrow_type->id() == arrow::Type::TIMESTAMP) {
        auto ts_arr =
            std::static_pointer_cast<arrow::TimestampArray>(arrow_arr);
        auto ts_unit =
            std::static_pointer_cast<arrow::TimestampType>(arrow_type)->unit();
        // Arrow timestamps can be in different units; convert to nanoseconds
        // for uniform handling since DuckDB's timestamp_t is microsecond-based
        // (via `value` which is microseconds since epoch).
        auto nanos_per_unit = [](arrow::TimeUnit::type unit) -> int64_t {
            switch (unit) {
                case arrow::TimeUnit::SECOND:
                    return 1000000000LL;
                case arrow::TimeUnit::MILLI:
                    return 1000000LL;
                case arrow::TimeUnit::MICRO:
                    return 1000LL;
                case arrow::TimeUnit::NANO:
                    return 1LL;
                default:
                    throw std::runtime_error("Unknown time unit");
            }
        };
        int64_t mult = nanos_per_unit(ts_unit);
        // Build result as nanosecond timestamps.
        arrow::TimestampBuilder ts_builder(
            arrow::timestamp(arrow::TimeUnit::NANO),
            arrow::default_memory_pool());
        for (int64_t i = 0; i < num_rows; i++) {
            if (ts_arr->IsNull(i)) {
                (void)ts_builder.AppendNull();
            } else {
                // Convert Arrow timestamp → nanoseconds → DuckDB timestamp_t
                // (microseconds). DuckDB's Interval::Add handles month-end
                // clamping (e.g., Jan 31 + 1 month → Feb 28).
                int64_t ns_val = ts_arr->Value(i) * mult;
                duckdb::timestamp_t ts(ns_val / 1000);
                duckdb::timestamp_t result =
                    duckdb::Interval::Add(ts, effective_interval);
                // Convert DuckDB result (microseconds) back to nanoseconds.
                (void)ts_builder.Append(result.value * 1000);
            }
        }
        arrow::Result<std::shared_ptr<arrow::Array>> res_arr =
            ts_builder.Finish();
        if (!res_arr.ok()) {
            throw std::runtime_error(res_arr.status().ToString());
        }
        auto bodo_arr = arrow_array_to_bodo(res_arr.ValueOrDie(),
                                            bodo::BufferPool::DefaultPtr());
        if (is_scalar) {
            return std::make_shared<ScalarExprResult>(std::move(bodo_arr));
        }
        return std::make_shared<ArrayExprResult>(std::move(bodo_arr),
                                                 "CalendarInterval");
    } else if (arrow_type->id() == arrow::Type::DATE32) {
        auto date_arr = std::static_pointer_cast<arrow::Date32Array>(arrow_arr);
        if (effective_interval.micros == 0) {
            // Day/month interval with no time component → produce DATE32.
            arrow::Date32Builder date_builder(arrow::default_memory_pool());
            for (int64_t i = 0; i < num_rows; i++) {
                if (date_arr->IsNull(i)) {
                    (void)date_builder.AppendNull();
                } else {
                    int32_t days = date_arr->Value(i);
                    duckdb::date_t date(days);
                    duckdb::date_t date_res =
                        duckdb::Interval::Add(date, effective_interval);
                    (void)date_builder.Append(date_res.days);
                }
            }
            arrow::Result<std::shared_ptr<arrow::Array>> res_arr =
                date_builder.Finish();
            if (!res_arr.ok()) {
                throw std::runtime_error(res_arr.status().ToString());
            }
            auto bodo_arr = arrow_array_to_bodo(res_arr.ValueOrDie(),
                                                bodo::BufferPool::DefaultPtr());
            if (is_scalar) {
                return std::make_shared<ScalarExprResult>(std::move(bodo_arr));
            }
            return std::make_shared<ArrayExprResult>(std::move(bodo_arr),
                                                     "CalendarInterval");
        } else {
            // Time-bearing interval → produce TIMESTAMP at midnight.
            arrow::TimestampBuilder ts_builder(
                arrow::timestamp(arrow::TimeUnit::NANO),
                arrow::default_memory_pool());
            for (int64_t i = 0; i < num_rows; i++) {
                if (date_arr->IsNull(i)) {
                    (void)ts_builder.AppendNull();
                } else {
                    int32_t days = date_arr->Value(i);
                    duckdb::date_t date(days);
                    duckdb::date_t date_res =
                        duckdb::Interval::Add(date, effective_interval);
                    int64_t ts_ns = int64_t(date_res.days) * 86400000000000LL;
                    (void)ts_builder.Append(ts_ns);
                }
            }
            arrow::Result<std::shared_ptr<arrow::Array>> res_arr =
                ts_builder.Finish();
            if (!res_arr.ok()) {
                throw std::runtime_error(res_arr.status().ToString());
            }
            auto bodo_arr = arrow_array_to_bodo(res_arr.ValueOrDie(),
                                                bodo::BufferPool::DefaultPtr());
            if (is_scalar) {
                return std::make_shared<ScalarExprResult>(std::move(bodo_arr));
            }
            return std::make_shared<ArrayExprResult>(std::move(bodo_arr),
                                                     "CalendarInterval");
        }
    } else if (arrow_type->id() == arrow::Type::INT64) {
        // Some operations (e.g., TO_DATE) may produce an int64 array
        // representing days since epoch instead of native DATE32.
        auto int_arr = std::static_pointer_cast<arrow::Int64Array>(arrow_arr);
        if (effective_interval.micros == 0) {
            // Day/month interval with no time component → produce DATE32.
            arrow::Date32Builder date_builder(arrow::default_memory_pool());
            for (int64_t i = 0; i < num_rows; i++) {
                if (int_arr->IsNull(i)) {
                    (void)date_builder.AppendNull();
                } else {
                    int32_t days = static_cast<int32_t>(int_arr->Value(i));
                    duckdb::date_t date(days);
                    duckdb::date_t date_res =
                        duckdb::Interval::Add(date, effective_interval);
                    (void)date_builder.Append(date_res.days);
                }
            }
            arrow::Result<std::shared_ptr<arrow::Array>> res_arr =
                date_builder.Finish();
            if (!res_arr.ok()) {
                throw std::runtime_error(res_arr.status().ToString());
            }
            auto bodo_arr = arrow_array_to_bodo(res_arr.ValueOrDie(),
                                                bodo::BufferPool::DefaultPtr());
            if (is_scalar) {
                return std::make_shared<ScalarExprResult>(std::move(bodo_arr));
            }
            return std::make_shared<ArrayExprResult>(std::move(bodo_arr),
                                                     "CalendarInterval");
        } else {
            // Time-bearing interval → produce TIMESTAMP at midnight.
            arrow::TimestampBuilder ts_builder(
                arrow::timestamp(arrow::TimeUnit::NANO),
                arrow::default_memory_pool());
            for (int64_t i = 0; i < num_rows; i++) {
                if (int_arr->IsNull(i)) {
                    (void)ts_builder.AppendNull();
                } else {
                    int32_t days = static_cast<int32_t>(int_arr->Value(i));
                    duckdb::date_t date(days);
                    duckdb::date_t date_res =
                        duckdb::Interval::Add(date, effective_interval);
                    int64_t ts_ns = int64_t(date_res.days) * 86400000000000LL;
                    (void)ts_builder.Append(ts_ns);
                }
            }
            arrow::Result<std::shared_ptr<arrow::Array>> res_arr =
                ts_builder.Finish();
            if (!res_arr.ok()) {
                throw std::runtime_error(res_arr.status().ToString());
            }
            auto bodo_arr = arrow_array_to_bodo(res_arr.ValueOrDie(),
                                                bodo::BufferPool::DefaultPtr());
            if (is_scalar) {
                return std::make_shared<ScalarExprResult>(std::move(bodo_arr));
            }
            return std::make_shared<ArrayExprResult>(std::move(bodo_arr),
                                                     "CalendarInterval");
        }
    }

    throw std::runtime_error(
        "PhysicalCalendarIntervalExpression: unsupported input type " +
        arrow_type->ToString());
}

arrow::Datum PhysicalCalendarIntervalExpression::join_expr_internal(
    array_info** left_table, array_info** right_table, void** left_data,
    void** right_data, void** left_null_bitmap, void** right_null_bitmap,
    int64_t left_index, int64_t right_index) {
    throw std::runtime_error(
        "PhysicalCalendarIntervalExpression::join_expr_internal not "
        "implemented");
}

arrow::compute::CalendarUnit getArrowCalendarUnit(const char* unit_str) {
    if (strcmp(unit_str, "nanosecond") == 0) {
        return arrow::compute::CalendarUnit::NANOSECOND;
    } else if (strcmp(unit_str, "microsecond") == 0) {
        return arrow::compute::CalendarUnit::MICROSECOND;
    } else if (strcmp(unit_str, "millisecond") == 0) {
        return arrow::compute::CalendarUnit::MILLISECOND;
    } else if (strcmp(unit_str, "second") == 0) {
        return arrow::compute::CalendarUnit::SECOND;
    } else if (strcmp(unit_str, "minute") == 0) {
        return arrow::compute::CalendarUnit::MINUTE;
    } else if (strcmp(unit_str, "hour") == 0) {
        return arrow::compute::CalendarUnit::HOUR;
    } else if (strcmp(unit_str, "day") == 0) {
        return arrow::compute::CalendarUnit::DAY;
    } else if (strcmp(unit_str, "week") == 0) {
        return arrow::compute::CalendarUnit::WEEK;
    } else if (strcmp(unit_str, "month") == 0) {
        return arrow::compute::CalendarUnit::MONTH;
    } else if (strcmp(unit_str, "quarter") == 0) {
        return arrow::compute::CalendarUnit::QUARTER;
    } else if (strcmp(unit_str, "year") == 0) {
        return arrow::compute::CalendarUnit::YEAR;
    } else {
        throw std::runtime_error("Unsupported calendar unit: " +
                                 std::string(unit_str));
    }
}

#undef CHECK_ARROW
