#include "expression.h"
#include <arrow/type_fwd.h>
#include <cmath>
#include "../libs/_decimal_ext.h"
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
    const std::vector<arrow::Datum>& arg_datums,
    const std::string& arrow_func_name,
    const arrow::compute::FunctionOptions* func_options) {
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
    } else if (arrow_func_name == "month_interval_between") {
        if (arg_datums.size() != 2) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_multi_input: month_interval_between expects "
                "exactly 2 "
                "arguments.");
        }
        auto mib_res =
            arrow::compute::CallFunction("month_interval_between", arg_datums);
        if (!mib_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_multi_input: Error in Arrow compute "
                "(month_interval_between): " +
                mib_res.status().message());
        }

        // Cast MonthInterval result to int32, that is all we need
        arrow::Datum month_interval_datum = mib_res.ValueOrDie();
        std::shared_ptr<arrow::MonthIntervalArray> mi_arr =
            std::static_pointer_cast<arrow::MonthIntervalArray>(
                month_interval_datum.make_array());

        // MonthIntervalArray stores months as int32.
        // Extract raw buffers and create Int32Array with the same buffers.
        auto mi_arr_int32 = std::make_shared<arrow::Int32Array>(
            arrow::int32(), mi_arr->length(), mi_arr->values(),
            mi_arr->null_bitmap(), mi_arr->null_count());

        return arrow::Datum(mi_arr_int32);
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
    } else if (arrow_func_name == "zip") {
        return do_arrow_compute_zip(arg_datums);
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
        func_res = arrow::compute::CallFunction(arrow_func_name, casted_datums,
                                                func_options);
    } else if (arrow_func_name == "bodo_substr_three" ||
               arrow_func_name == "utf8_slice_codeunits") {
        // If the arrow func name is bodo_substr_three or utf8_slice_codeunits,
        // we are computing a substring. The main difference between them is
        // that utf8_slice_codeunits takes an end index whereas
        // bodo_substr_three takes a length. bodo_substr_three only accepts
        // arrays whereas utf8_slice_codeunits only accepts scalars, so we call
        // utf8_slice_codeunits if both the start index and length / end index
        // are scalars, otherwise we call bodo_substr_three.

        // We only currently support array-based substring with two/three
        // arguments. If two arguments are provided, we pass the max value of
        // int64 as the third (end index or length) parameter to slice until the
        // end of the string. In the future we could add a "step" parameter.
        if (arg_datums.size() != 2 && arg_datums.size() != 3) {
            throw std::runtime_error(
                "do_arrow_compute_multi_input: " + arrow_func_name +
                " expects 2 or 3 "
                "arguments.");
        }

        if (arg_datums[1].is_scalar() &&
            (arg_datums.size() == 2 || arg_datums[2].is_scalar())) {
            // The index arguments are scalars: use utf8_slice_codeunits
            int64_t arg1_scalar =
                arg_datums[1].scalar_as<arrow::Int64Scalar>().value;

            arrow::compute::SliceOptions slice_opts;
            slice_opts.start = arg1_scalar;
            if (arg_datums.size() == 3) {
                int64_t arg2_scalar =
                    arg_datums[2].scalar_as<arrow::Int64Scalar>().value;
                if (arrow_func_name == "bodo_substr_three") {
                    // Convert from length argument to end index argument
                    arg2_scalar = arg1_scalar + arg2_scalar;
                }
                slice_opts.stop = arg2_scalar;
            }
            // If arg_datums.size() == 2, no need to do anything more since
            // the `stop` option defaults to the max value of int64.

            func_res = arrow::compute::CallFunction(
                "utf8_slice_codeunits", {arg_datums[0]}, &slice_opts);
        } else {
            // At least one of the second and third arguments are arrays: use
            // bodo_substr_three

            // Copy so we can modify datum args (pre-sized to 3 elements)
            std::vector<arrow::Datum> array_arg_datums(3);
            std::copy(arg_datums.begin(), arg_datums.end(),
                      array_arg_datums.begin());

            if (arrow_func_name == "utf8_slice_codeunits" &&
                arg_datums.size() == 3) {
                // Convert from end index argument to length argument
                array_arg_datums[2] = do_arrow_compute_binary(
                    array_arg_datums[2], array_arg_datums[1], "subtract");
            }

            // Get the scalar object from a possible scalar datum (start index
            // or length)
            int scalar_arg_index = -1;
            arrow::Int64Scalar arg_scalar;
            if (arg_datums.size() == 3) {
                // 0 or 1 of the arguments could be scalars
                for (int arg_datum_index = 1; arg_datum_index <= 2;
                     arg_datum_index++) {
                    if (array_arg_datums[arg_datum_index].is_scalar()) {
                        scalar_arg_index = arg_datum_index;
                        arg_scalar = array_arg_datums[arg_datum_index]
                                         .scalar_as<arrow::Int64Scalar>();
                        break;
                    }
                }
            } else {
                // If length was not provided, use the max value of int64.
                scalar_arg_index = 2;
                arg_scalar =
                    arrow::Int64Scalar(std::numeric_limits<int64_t>::max());
            }

            // Broadcast the scalar argument (if there was one) to array since
            // bodo_substr_three requires arrays
            if (scalar_arg_index != -1) {
                auto arg_scalar_array = arrow::MakeArrayFromScalar(
                    arg_scalar, array_arg_datums[0].make_array()->length());
                if (!arg_scalar_array.ok()) {
                    throw std::runtime_error(
                        "do_arrow_compute_multi_input: Failed to make array "
                        "from scalar: " +
                        arg_scalar_array.status().message());
                }
                array_arg_datums[scalar_arg_index] =
                    arrow::Datum(arg_scalar_array.ValueOrDie());
            }

            // Call our custom substring function that can handle array indices
            EnsureSubstrRegistered();
            func_res = arrow::compute::CallFunction("bodo_substr_three",
                                                    array_arg_datums);
        }
    } else {
        func_res = arrow::compute::CallFunction(arrow_func_name, arg_datums,
                                                func_options);
    }

    if (!func_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_arrow_compute_multi_input: Error in Arrow compute (" +
            arrow_func_name + "): " + func_res.status().message());
    }

    return func_res.ValueOrDie();
}

std::shared_ptr<array_info> do_arrow_compute_multi_input(
    const std::vector<std::shared_ptr<ExprResult>>& in_expr_results,
    const std::string& arrow_func_name,
    const arrow::compute::FunctionOptions* func_options) {
    std::vector<arrow::Datum> arg_datums;
    for (auto& expr_res : in_expr_results) {
        arrow::Datum arg_datum = ConvertExprResultToDatum(
            expr_res, "do_arrow_compute_multi_input input");
        arg_datums.push_back(arg_datum);
    }
    arrow::Datum result_datum = do_arrow_compute_multi_input_datum(
        arg_datums, arrow_func_name, func_options);
    return ConvertDatumToArrayInfo(result_datum);
}

std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, std::shared_ptr<ExprResult> right_res,
    const std::string& comparator,
    const arrow::compute::FunctionOptions* func_options,
    const std::shared_ptr<arrow::DataType> result_type) {
    arrow::Datum src1 =
        ConvertExprResultToDatum(left_res, "do_arrow_compute left");
    arrow::Datum src2 =
        ConvertExprResultToDatum(right_res, "do_arrow_compute right");
    arrow::Datum cmp_res_datum = do_arrow_compute_binary(
        src1, src2, comparator, func_options, result_type);
    return ConvertDatumToArrayInfo(cmp_res_datum);
}

std::shared_ptr<array_info> do_arrow_compute_binary(
    arrow::Datum left_res, std::shared_ptr<ExprResult> right_res,
    const std::string& comparator,
    const arrow::compute::FunctionOptions* func_options,
    const std::shared_ptr<arrow::DataType> result_type) {
    arrow::Datum src2 =
        ConvertExprResultToDatum(right_res, "do_arrow_compute right");
    arrow::Datum cmp_res_datum = do_arrow_compute_binary(
        left_res, src2, comparator, func_options, result_type);
    return ConvertDatumToArrayInfo(cmp_res_datum);
}

std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, arrow::Datum right_res,
    const std::string& comparator,
    const arrow::compute::FunctionOptions* func_options,
    const std::shared_ptr<arrow::DataType> result_type) {
    arrow::Datum src1 =
        ConvertExprResultToDatum(left_res, "do_arrow_compute left");
    arrow::Datum cmp_res_datum = do_arrow_compute_binary(
        src1, right_res, comparator, func_options, result_type);
    return ConvertDatumToArrayInfo(cmp_res_datum);
}

std::shared_ptr<array_info> do_arrow_compute_unary(
    std::shared_ptr<ExprResult> left_res, const std::string& comparator,
    const arrow::compute::FunctionOptions* func_options,
    const std::shared_ptr<arrow::DataType> result_type) {
    arrow::Datum src1 =
        ConvertExprResultToDatum(left_res, "do_arrow_compute left");
    arrow::Datum cmp_res =
        do_arrow_compute_unary(src1, comparator, func_options, result_type);
    return ConvertDatumToArrayInfo(cmp_res);
}

std::shared_ptr<array_info> do_arrow_compute_cast(
    std::shared_ptr<ExprResult> left_res,
    const std::shared_ptr<arrow::DataType>& return_type) {
    arrow::Datum src1 =
        ConvertExprResultToDatum(left_res, "do_arrow_compute left");

    arrow::Datum casted = do_arrow_compute_cast(src1, return_type);
    return ConvertDatumToArrayInfo(casted);
}

void do_result_type_cast(arrow::Result<arrow::Datum>& out_res,
                         const std::shared_ptr<arrow::DataType> result_type) {
    arrow::Datum out_datum = out_res.ValueOrDie();
    std::shared_ptr<arrow::DataType> out_dtype = out_datum.type();
    if (result_type && !out_dtype->Equals(result_type)) {
        // Cast to result type if available and different from current type.
        arrow::compute::CastOptions cast_opts;
        cast_opts.allow_int_overflow = true;
        cast_opts.allow_float_truncate = true;
        arrow::Result<arrow::Datum> cast_res =
            arrow::compute::Cast(out_datum, result_type, cast_opts);
        if (!cast_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_result_type_cast cast_res: Error in Arrow compute: " +
                cast_res.status().message());
        }
        out_res = cast_res;
    }
}

static arrow::Datum decimal_arithmetic(
    const arrow::Datum& left_res, const arrow::Datum& right_res,
    const std::string& op,  // "add","subtract","multiply","divide"
    int result_precision, int result_scale, int left_precision, int left_scale,
    int right_precision, int right_scale) {
    // Determine length and whether inputs are arrays or scalars
    std::shared_ptr<arrow::Array> left_arr =
        left_res.is_array() ? left_res.make_array() : nullptr;
    std::shared_ptr<arrow::Array> right_arr =
        right_res.is_array() ? right_res.make_array() : nullptr;
    bool left_is_scalar = left_res.is_scalar();
    bool right_is_scalar = right_res.is_scalar();

    if (left_is_scalar || right_is_scalar) {
        throw std::runtime_error(
            "decimal_arithmetic: don't handle scalars yet");
    }

    int64_t length = 0;
    length = left_arr->length();
    if (length != right_arr->length()) {
        throw std::runtime_error("decimal_arithmetic: array length mismatch");
    }

    auto left_dec_arr =
        std::static_pointer_cast<arrow::Decimal128Array>(left_arr);
    auto right_dec_arr =
        std::static_pointer_cast<arrow::Decimal128Array>(right_arr);
    std::shared_ptr<arrow::Array> result = arrow_array_decimal_arithmetic_util(
        left_dec_arr, left_precision, left_scale, right_dec_arr,
        right_precision, right_scale, length, result_precision, result_scale,
        op);
    // check for overflow
    if (result == nullptr) {
        throw std::runtime_error("Decimal overflow in operation " + op);
    }
    return arrow::Datum(result);
}

// Main function with added decimal check and Snowflake rules
arrow::Datum do_arrow_compute_binary(
    arrow::Datum left_res, arrow::Datum right_res,
    const std::string& comparator,
    const arrow::compute::FunctionOptions* func_options,
    const std::shared_ptr<arrow::DataType> result_type) {
    // --- New: if both are decimal types, compute Snowflake-style result
    // precision/scale ---
    bool left_is_decimal =
        left_res.type() && left_res.type()->id() == arrow::Type::DECIMAL128;
    bool right_is_decimal =
        right_res.type() && right_res.type()->id() == arrow::Type::DECIMAL128;

    if (left_is_decimal || right_is_decimal) {
        if (left_res.is_scalar()) {
            throw std::runtime_error(
                "do_arrow_compute_binary decimal operator not supported with "
                "scalar yet for left arg");
        }
        if (right_res.is_scalar()) {
            throw std::runtime_error(
                "do_arrow_compute_binary decimal operator not supported with "
                "scalar yet for right arg");
        }
        int p1, s1, l1, p2, s2, l2;
        if (left_is_decimal) {
            auto left_dec_type =
                std::static_pointer_cast<arrow::Decimal128Type>(
                    left_res.type());
            p1 = left_dec_type->precision();
            s1 = left_dec_type->scale();
        } else {
            std::tie(p1, s1) = getPrecisionScaleNonDecimal(left_res);
        }
        if (right_is_decimal) {
            auto right_dec_type =
                std::static_pointer_cast<arrow::Decimal128Type>(
                    right_res.type());
            p2 = right_dec_type->precision();
            s2 = right_dec_type->scale();
        } else {
            std::tie(p2, s2) = getPrecisionScaleNonDecimal(right_res);
        }

        l1 = p1 - s1;
        l2 = p2 - s2;

        int result_precision = 0;
        int result_scale = 0;

        // Map comparator to operation name used in Snowflake rules
        // We handle add, subtract, multiply, divide
        std::string op = comparator;  // assume comparator is
                                      // "add","subtract","multiply","divide"
        // If comparator is an Arrow function name like "multiply", "add", etc.,
        // adapt accordingly.

        if (op == "add" || op == "subtract") {
            // Snowflake rule:
            // scale = max(s1, s2)
            // precision = max(p1 - s1, p2 - s2) + scale + 1
            result_scale = std::max(s1, s2);
            result_precision = std::max(l1, l2) + result_scale + 1;
        } else if (op == "multiply") {
            // Snowflake rule:
            // precision = p1 + p2
            // scale = s1 + s2
            result_precision = p1 + p2;
            result_scale = s1 + s2;
        } else if (op == "divide") {
            // Snowflake rule (one common variant):
            // scale = max(6, s1 + p2 + 1)
            // precision = p1 - s1 + s2 + scale
            result_scale = std::max(6, s1 + p2 + 1);
            result_precision = l1 + s2 + result_scale;
        } else if (op == "equal" || op == "not_equal" || op == "less" ||
                   op == "greater" || op == "less_equal" ||
                   op == "greater_equal") {
            result_scale = std::max(s1, s2);
            result_precision = std::max(l1, l2) + result_scale;
        } else {
            // Not a decimal arithmetic op we know; fall back to normal
            // CallFunction (or you can throw) fall through to normal path below
            result_precision = -1;
        }

        if (result_precision > 38) {
            if (!left_is_decimal) {
                left_res =
                    do_arrow_compute_cast(left_res, arrow::decimal128(p1, s1));
            }
            if (!right_is_decimal) {
                right_res =
                    do_arrow_compute_cast(right_res, arrow::decimal128(p2, s2));
            }
            // Use decimal_arithmetic elementwise with overflow checking
            return decimal_arithmetic(left_res, right_res, op, 38, result_scale,
                                      p1, s1, p2, s2);
        }
    }

    // --- Default path: not both decimals or unknown op: call Arrow compute
    // directly ---
    arrow::Result<arrow::Datum> cmp_res = arrow::compute::CallFunction(
        comparator, {left_res, right_res}, func_options);
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_arrow_compute_binary: Error in Arrow compute (" + comparator +
            "): " + cmp_res.status().message());
    }

    arrow::Datum cmp_datum = cmp_res.ValueOrDie();
    return do_arrow_compute_cast(cmp_datum, result_type);
}

arrow::Datum do_arrow_compute_unary(
    arrow::Datum src1, const std::string& comparator,
    const arrow::compute::FunctionOptions* func_options,
    const std::shared_ptr<arrow::DataType> result_type) {
    // Special handling for is_not_null since it is not supported directly
    // by Arrow compute.
    if (comparator == "is_not_null") {
        arrow::Result<arrow::Datum> is_null_res =
            arrow::compute::CallFunction("is_null", {src1}, func_options);
        if (!is_null_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_unary: Error in Arrow compute (is_null): " +
                is_null_res.status().message());
        }

        // Invert the boolean array
        arrow::Result<arrow::Datum> invert_res =
            arrow::compute::CallFunction("invert", {is_null_res.ValueOrDie()});
        if (!invert_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_unary: Error in Arrow compute (invert): " +
                invert_res.status().message());
        }
        do_result_type_cast(invert_res, result_type);
        return invert_res.ValueOrDie();
    }

    // Special handling for is_true, is_not_true, is_false, is_not_false since
    // they are not supported directly by Arrow compute.
    if (comparator == "is_true" || comparator == "is_not_true" ||
        comparator == "is_false" || comparator == "is_not_false") {
        const bool test_true =
            comparator == "is_true" || comparator == "is_not_true";
        const bool invert =
            comparator == "is_not_true" || comparator == "is_false";

        auto na_fill_value = arrow::MakeScalar(test_true ? false : true);

        arrow::Result<arrow::Datum> result = arrow::compute::CallFunction(
            "coalesce", {src1, na_fill_value}, func_options);
        if (!result.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_unary: Error in Arrow compute (coalesce): " +
                result.status().message());
        }

        // is_not_true: Invert so that null/false -> true and true -> false.
        // is_false: Invert so that null/true -> false and true -> false.
        if (invert) {
            result =
                arrow::compute::CallFunction("invert", {result.ValueOrDie()});
            if (!result.ok()) [[unlikely]] {
                throw std::runtime_error(
                    "do_arrow_compute_unary: Error in Arrow compute "
                    "(invert): " +
                    result.status().message());
            }
        }
        do_result_type_cast(result, result_type);
        return result.ValueOrDie();
    }

    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::CallFunction(comparator, {src1}, func_options);
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_arrow_compute_unary: Error in Arrow compute (" + comparator +
            "): " + cmp_res.status().message());
    }

    do_result_type_cast(cmp_res, result_type);
    return cmp_res.ValueOrDie();
}

arrow::Datum do_arrow_compute_cast(
    arrow::Datum left_res,
    const std::shared_ptr<arrow::DataType>& return_type) {
    // No need to cast if type is already the target type.
    // Note that arrow::DataType.Equals() also compares type parameters such
    // as time units and timezones.
    if (!return_type || left_res.type()->Equals(return_type)) {
        return left_res;
    }

    // Globally set the allow_int_overflow cast option to true; in the future,
    // CastExpressions should support these options.
    arrow::compute::CastOptions cast_opts;
    cast_opts.allow_int_overflow = true;
    cast_opts.allow_float_truncate = true;
    arrow::Result<arrow::Datum> cmp_res =
        arrow::compute::Cast(left_res, return_type, cast_opts);
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_arrow_compute_cast: Error in Arrow compute: " +
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
    arrow::Datum src3;
    if (else_res != nullptr) {
        src3 = ConvertExprResultToDatum(else_res, "do_arrow_compute else");
    }

    // NOTE: Arrow's "if_else" doesn't match our Python and SQL semantics since
    // it propagates nulls in the condition.
    arrow::Result<arrow::Datum> case_res = arrow::compute::CallFunction(
        "case_when", else_res ? std::vector<arrow::Datum>{src1, src2, src3}
                              : std::vector<arrow::Datum>{src1, src2});
    if (!case_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            "do_arrow_compute_case case_when: Error in Arrow compute: " +
            case_res.status().message());
    }

    do_result_type_cast(case_res, result_type);
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
                                bfe.function.name, result_type));
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
                    getCastReturnType(bce)));
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

        // Special handling for multi-input Arrow functions with options
        if (scalar_func_data.arrow_func_name == "max_element_wise" ||
            scalar_func_data.arrow_func_name == "min_element_wise") {
            auto [skip_nulls] = get_var_py_args_as_types<0, 1>(
                scalar_func_data.args, scalar_func_data.arrow_func_name.c_str(),
                get_py_object_as_bool);

            arrow::compute::ElementWiseAggregateOptions opts;
            if (skip_nulls.has_value()) {
                opts.skip_nulls = *skip_nulls;
            } else {
                // Avoid skipping nulls to match SQL semantics.
                // This is True by default in Arrow.
                opts.skip_nulls = false;
            }

            result = do_arrow_compute_multi_input(
                in_expr_results, scalar_func_data.arrow_func_name, &opts);
        } else {
            result = do_arrow_compute_multi_input(
                in_expr_results, scalar_func_data.arrow_func_name);
        }

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

// ------------------ Custom Arrow Kernels ------------------

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
                } else if (r == 0) {
                    // Return NULL when modulus is 0
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
                if (r == 0) {
                    // Return NULL when modulus is 0
                    clear_valid(out_valid_bits, offset + i);
                } else {
                    // Calculate modulus operator.
                    CType res = ModOp::apply(l, r);
                    // Assign result.
                    out_values[i] = res;
                    // Indicate index has valid data.
                    set_valid(out_valid_bits, offset + i);
                }
            }
        }
    }

    return arrow::Status::OK();
}

struct NativeMod {
    template <typename T>
    static T apply(T l, T r) {
        return l % r;
    }
};

struct AltMod {
    template <typename T>
    static T apply(T l, T r) {
        return l - ((int64_t)(l / r) * r);
    }
};

void RegisterMod(arrow::compute::FunctionRegistry* registry) {
    // Declare the binary arrow compute function named "bodo_mod".
    auto func = std::make_shared<arrow::compute::ScalarFunction>(
        "bodo_mod", arrow::compute::Arity::Binary(),
        arrow::compute::FunctionDoc{
            "Modulo of two arrays", "Returns lhs % rhs", {"lhs", "rhs"}});

    // Declare int8,int8->int8 mod kernel.
    arrow::compute::ScalarKernel kernel8(
        {arrow::compute::InputType(arrow::int8()),
         arrow::compute::InputType(arrow::int8())},
        arrow::compute::OutputType(arrow::int8()),
        ModImpl<arrow::Int8Type, NativeMod>);
    kernel8.null_handling = arrow::compute::NullHandling::COMPUTED_PREALLOCATE;

    // Declare int16,int16->int16 mod kernel.
    arrow::compute::ScalarKernel kernel16(
        {arrow::compute::InputType(arrow::int16()),
         arrow::compute::InputType(arrow::int16())},
        arrow::compute::OutputType(arrow::int16()),
        ModImpl<arrow::Int16Type, NativeMod>);
    kernel16.null_handling = arrow::compute::NullHandling::COMPUTED_PREALLOCATE;

    // Declare int32,int32->int32 mod kernel.
    arrow::compute::ScalarKernel kernel32(
        {arrow::compute::InputType(arrow::int32()),
         arrow::compute::InputType(arrow::int32())},
        arrow::compute::OutputType(arrow::int32()),
        ModImpl<arrow::Int32Type, NativeMod>);
    kernel32.null_handling = arrow::compute::NullHandling::COMPUTED_PREALLOCATE;

    // Declare int64,int64->int64 mod kernel.
    arrow::compute::ScalarKernel kernel64(
        {arrow::compute::InputType(arrow::int64()),
         arrow::compute::InputType(arrow::int64())},
        arrow::compute::OutputType(arrow::int64()),
        ModImpl<arrow::Int64Type, NativeMod>);
    kernel64.null_handling = arrow::compute::NullHandling::COMPUTED_PREALLOCATE;

    // Declare uint64,uint64->uint64 mod kernel.
    arrow::compute::ScalarKernel kernelu64(
        {arrow::compute::InputType(arrow::uint64()),
         arrow::compute::InputType(arrow::uint64())},
        arrow::compute::OutputType(arrow::uint64()),
        ModImpl<arrow::UInt64Type, NativeMod>);
    kernelu64.null_handling =
        arrow::compute::NullHandling::COMPUTED_PREALLOCATE;

    // Declare float,float->float mod kernel.
    arrow::compute::ScalarKernel floatkernel32(
        {arrow::compute::InputType(arrow::float32()),
         arrow::compute::InputType(arrow::float32())},
        arrow::compute::OutputType(arrow::float32()),
        ModImpl<arrow::FloatType, AltMod>);
    floatkernel32.null_handling =
        arrow::compute::NullHandling::COMPUTED_PREALLOCATE;

    // Declare double,double->double mod kernel.
    arrow::compute::ScalarKernel floatkernel64(
        {arrow::compute::InputType(arrow::float64()),
         arrow::compute::InputType(arrow::float64())},
        arrow::compute::OutputType(arrow::float64()),
        ModImpl<arrow::DoubleType, AltMod>);
    floatkernel64.null_handling =
        arrow::compute::NullHandling::COMPUTED_PREALLOCATE;

    arrow::Status status;
    // Add all the above kernels to the function.
    status = func->AddKernel(kernel8);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod int8 AddKernel failed.");
    }
    status = func->AddKernel(kernel16);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod int16 AddKernel failed.");
    }
    status = func->AddKernel(kernel32);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod int32 AddKernel failed.");
    }
    status = func->AddKernel(kernel64);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod int64 AddKernel failed.");
    }
    status = func->AddKernel(kernelu64);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod uint64 AddKernel failed.");
    }
    status = func->AddKernel(floatkernel32);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod float32 AddKernel failed.");
    }
    status = func->AddKernel(floatkernel64);
    if (!status.ok()) {
        throw std::runtime_error("RegisterMod float64 AddKernel failed.");
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

template <typename StringType>
struct SubstrTraits;

template <>
struct SubstrTraits<arrow::LargeStringType> {
    using Builder = arrow::LargeStringBuilder;

    // Helper to count UTF-8 characters in a byte string
    static int64_t GetLogicalLength(const char* data, int64_t byte_len) {
        int64_t char_count = 0;
        for (int64_t i = 0; i < byte_len; i++) {
            // Only count bytes that are not continuation bytes (10xxxxxx)
            if ((data[i] & 0xC0) != 0x80) {
                char_count++;
            }
        }
        return char_count;
    }

    // Helper to convert UTF-8 character number offset to byte offset
    static int64_t LogicalToByteOffset(const char* data, int64_t byte_len,
                                       int64_t char_offset) {
        int64_t char_count = 0;
        for (int64_t i = 0; i < byte_len; i++) {
            // Only count bytes that are not continuation bytes (10xxxxxx)
            if ((data[i] & 0xC0) != 0x80) {
                if (char_count == char_offset)
                    return i;
                char_count++;
            }
        }
        // If the string has less than char_offset characters, just return the
        // string length in bytes. We don't expect to hit this case.
        return byte_len;
    }
};

template <>
struct SubstrTraits<arrow::LargeBinaryType> {
    using Builder = arrow::LargeBinaryBuilder;

    // For binary input, logical length is the same as byte length
    static int64_t GetLogicalLength(const char* data, int64_t byte_len) {
        return byte_len;
    }

    // For binary input, logical offset is the same as byte offset
    static int64_t LogicalToByteOffset(const char* data, int64_t byte_len,
                                       int64_t logical_offset) {
        // Assume that logical_offset < byte_len, since we check that
        // 0 <= logical_offset < byte_len before calling this function
        return logical_offset;
    }
};

// The execution function for the kernel.
// Signature: Status Exec(KernelContext*, const ExecBatch&, Datum* out)
template <typename StringType>
static arrow::Status SubstrThreeImpl(arrow::compute::KernelContext* ctx,
                                     const arrow::compute::ExecSpan& batch,
                                     arrow::compute::ExecResult* out) {
    using Traits = SubstrTraits<StringType>;

    // Expect exactly 3 inputs
    if (batch.values.size() != 3) {
        throw std::runtime_error("bodo_substr_three expected 3 inputs.");
    }

    const arrow::ArraySpan& src = batch[0].array;
    const arrow::ArraySpan& start = batch[1].array;
    const arrow::ArraySpan& len = batch[2].array;

    if (!src.type) {
        throw std::runtime_error("bodo_substr_three src not valid.");
    }
    if (!start.type) {
        throw std::runtime_error("bodo_substr_three start not valid.");
    }
    if (!len.type) {
        throw std::runtime_error("bodo_substr_three len not valid.");
    }

    // Number of rows
    int64_t n = static_cast<int64_t>(batch.length);

    // Pointers to src buffers
    const uint8_t* src_null_bitmap = src.buffers[0].data;
    const int64_t* src_offsets =
        reinterpret_cast<const int64_t*>(src.buffers[1].data);
    const char* src_data = reinterpret_cast<const char*>(src.buffers[2].data);
    int64_t src_offset = src.offset;  // element offset into offsets array

    // Pointers to start/len buffers (int64)
    const uint8_t* start_null_bitmap = start.buffers[0].data;
    const int64_t* start_values =
        reinterpret_cast<const int64_t*>(start.buffers[1].data);
    int64_t start_offset = start.offset;

    const uint8_t* len_null_bitmap = len.buffers[0].data;
    const int64_t* len_values =
        reinterpret_cast<const int64_t*>(len.buffers[1].data);
    int64_t len_offset = len.offset;

    // Prepare builder for output strings
    typename Traits::Builder builder(ctx->memory_pool());

    auto is_valid_bit = [](const uint8_t* bits, int64_t offset,
                           int64_t i) -> bool {
        return !bits || arrow::bit_util::GetBit(bits, offset + i);
    };

    for (int64_t i = 0; i < n; ++i) {
        // Check nulls: bitmaps may be null (no nulls) -> treat as all-valid
        bool src_is_null = false;
        if (src_null_bitmap) {
            src_is_null = !is_valid_bit(src_null_bitmap, src_offset, i);
        }
        bool start_is_null = false;
        if (start_null_bitmap) {
            start_is_null = !is_valid_bit(start_null_bitmap, start_offset, i);
        }
        bool len_is_null = false;
        if (len_null_bitmap) {
            len_is_null = !is_valid_bit(len_null_bitmap, len_offset, i);
        }

        if (src_is_null || start_is_null || len_is_null) {
            ARROW_RETURN_NOT_OK(builder.AppendNull());
            continue;
        }

        // Read start and len values.
        // These are in units of UTF-8 characters, so we will need
        // to convert to a raw byte offset.
        int64_t start_chars_val = start_values[start_offset + i];
        int64_t len_chars_val = len_values[len_offset + i];

        // Read string offsets: offsets are int64 (common Arrow layout)
        // offsets array length is n + 1; offsets index = src_offset + i
        int64_t off0 = src_offsets[src_offset + i];
        int64_t off1 = src_offsets[src_offset + i + 1];
        int64_t byte_len = off1 - off0;

        // If input type is large_string, get number of UTF-8 characters in
        // string (necessary to handle multi-byte characters). Else (input type
        // is large_binary), we use the raw byte length.
        int64_t char_len = Traits::GetLogicalLength(src_data + off0, byte_len);

        // Normalize negative length
        if (len_chars_val < 0) {
            len_chars_val = 0;
        }

        // If start index is negative, count backwards from end of string
        if (start_chars_val < 0) {
            start_chars_val = char_len + start_chars_val;
            if (start_chars_val < 0) {
                // If start index is still negative (meaning the index wrapped
                // around and still exceeded the string length), set it to 0 and
                // adjust the length for the number of characters left of the
                // beginning of the string the start index is.
                len_chars_val =
                    std::max<int64_t>(len_chars_val + start_chars_val, 0);
                start_chars_val = 0;
            }
        }

        // Return an empty string if:
        //   Start index is beyond string length,
        //   start index is negative and its absolute value is greater than the
        //   string length plus the requested substring length, or a
        //   negative/zero substring length was provided.
        if (start_chars_val >= char_len || len_chars_val == 0) {
            ARROW_RETURN_NOT_OK(builder.Append(std::string()));
            continue;
        }

        // Convert start index and length to byte offsets depending on input
        // type
        int64_t start_val = Traits::LogicalToByteOffset(
            src_data + off0, byte_len, start_chars_val);
        int64_t len_val = Traits::LogicalToByteOffset(src_data + off0, byte_len,
                                                      len_chars_val);

        // Compute take and append substring
        int64_t take = std::min<int64_t>(len_val, byte_len - start_val);
        const char* substr_ptr =
            src_data + off0 + static_cast<size_t>(start_val);
        ARROW_RETURN_NOT_OK(
            builder.Append(std::string(substr_ptr, static_cast<size_t>(take))));
    }

    // Finish builder to produce Array
    std::shared_ptr<arrow::Array> out_array;
    ARROW_RETURN_NOT_OK(builder.Finish(&out_array));

    out->value = out_array->data();
    return arrow::Status::OK();
}

void RegisterSubstr(arrow::compute::FunctionRegistry* registry) {
    auto func = std::make_shared<arrow::compute::ScalarFunction>(
        "bodo_substr_three", arrow::compute::Arity::Ternary(),
        arrow::compute::FunctionDoc{"substr of source, start, len",
                                    "Returns substr(src, start, len)",
                                    {"src", "start", "len"}});

    arrow::compute::ScalarKernel kernel_utf8(
        {arrow::compute::InputType(arrow::large_utf8()),
         arrow::compute::InputType(arrow::int64()),
         arrow::compute::InputType(arrow::int64())},
        arrow::compute::OutputType(arrow::large_utf8()),
        SubstrThreeImpl<arrow::LargeStringType>);

    arrow::Status status;
    status = func->AddKernel(kernel_utf8);
    if (!status.ok()) {
        throw std::runtime_error("RegisterSubstr utf8 AddKernel failed.");
    }

    arrow::compute::ScalarKernel kernel_binary(
        {arrow::compute::InputType(arrow::large_binary()),
         arrow::compute::InputType(arrow::int64()),
         arrow::compute::InputType(arrow::int64())},
        arrow::compute::OutputType(arrow::large_binary()),
        SubstrThreeImpl<arrow::LargeBinaryType>);

    status = func->AddKernel(kernel_binary);
    if (!status.ok()) {
        throw std::runtime_error("RegisterSubstr binary AddKernel failed.");
    }

    // Register the function.
    status = registry->AddFunction(std::move(func));
    if (!status.ok()) {
        throw std::runtime_error("RegisterSubstr AddFunction failed.");
    }
}

// Register the function in Arrow's global function registry.
void EnsureSubstrRegistered() {
    static std::once_flag once_flag_;
    // Register the Bodo substr arrow compute function only once.
    std::call_once(once_flag_, [&]() {
        auto* registry = arrow::compute::GetFunctionRegistry();
        RegisterSubstr(registry);
    });
}

// ----------------------------------------------------------

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

// ----- Translations of custom function names to Arrow computations -----

/**
 * @brief Return a tuple of the input regex pattern with its capturing
 *   groups converted to named groups, and the number of groups that were
 *   found in the pattern.
 */
std::tuple<std::string, int> convert_to_named_regexp(std::string pattern_str) {
    std::string named_pattern;
    // Number of groups found in regex pattern so far
    int num_groups = 0;

    // Convert all groups to _groupN format for extract_regex
    for (size_t i = 0; i < pattern_str.length(); i++) {
        // Handle escaped characters by reading the backslash and the
        // following character together
        if (pattern_str[i] == '\\' && i + 1 < pattern_str.length()) {
            named_pattern += pattern_str[i];
            named_pattern += pattern_str[i + 1];
            i++;
        } else if (pattern_str[i] == '(') {  // Start of group
            // Check if it's a named group of the form (?<name>...)
            // or (?P<name>...)
            if (i + 1 < pattern_str.length() && pattern_str[i + 1] == '?') {
                // Offset by 1 if it is of the form (?P<name>...)
                int p_offset =
                    (i + 2 < pattern_str.length() && pattern_str[i + 2] == 'P')
                        ? 1
                        : 0;
                if (i + 2 + p_offset < pattern_str.length() &&
                    pattern_str[i + 2 + p_offset] == '<') {
                    // Rename existing name to _groupN
                    size_t close = pattern_str.find('>', i + 3 + p_offset);
                    if (close != std::string::npos) {
                        named_pattern +=
                            "(?<_group" + std::to_string(num_groups++) + ">";
                        i = close;  // Skip to after the >
                    } else {
                        named_pattern += pattern_str[i];
                    }
                } else {
                    // Non-capturing or other special group, keep as-is
                    named_pattern += pattern_str[i];
                }
            } else {
                // Unnamed group: convert to named group
                named_pattern +=
                    "(?<_group" + std::to_string(num_groups++) + ">";
            }
        } else {
            named_pattern += pattern_str[i];
        }
    }

    return std::tuple(named_pattern, num_groups);
}

arrow::Datum PhysicalArrowExpression::do_arrow_compute_regexp_substr(
    arrow::Datum res_datum, std::string pattern_str, bool extract_submatches,
    int64_t group_to_extract) {
    // Ensure all groups are named so that we can pass to extract_regex.
    // Do this once per operator.
    if (!named_regexp) {
        named_regexp = std::make_shared<std::tuple<std::string, int>>(
            convert_to_named_regexp(pattern_str));
    }

    std::string named_pattern = std::get<0>(*named_regexp);
    int num_groups = std::get<1>(*named_regexp);

    if (!extract_submatches || num_groups == 0) {
        // Wrap the whole pattern in a group
        named_pattern = "(?<_whole>" + named_pattern + ")";
        extract_submatches = false;
    }

    arrow::compute::ExtractRegexOptions opts(named_pattern);
    auto extract_regex_result =
        do_arrow_compute_unary(res_datum, "extract_regex", &opts);

    // Convert to Arrow array (StructArray)
    std::shared_ptr<arrow::Array> extract_array =
        extract_regex_result.make_array();
    std::shared_ptr<arrow::StructArray> struct_result =
        std::static_pointer_cast<arrow::StructArray>(extract_array);

    // Extract the requested field
    std::shared_ptr<arrow::Array> captured_field;
    if (extract_submatches && group_to_extract >= num_groups) {
        // Return null array if requested group is greater than number
        // of groups in regex
        captured_field =
            arrow::MakeArrayOfNull(arrow::utf8(), struct_result->length())
                .ValueOrDie();
    } else {
        std::string field_name;
        if (!extract_submatches) {
            // No group extraction requested, return whole match
            field_name = "_whole";
        } else {
            // extract_submatches && group_to_extract < num_groups:
            // valid group number requested.
            field_name = "_group" + std::to_string(group_to_extract);
        }

        // NOTE: StructArray.GetFieldByName() exists, but does not propagate
        // the validity bitmap to the child fields. We need to retain the
        // nulls that are emitted when there is no match for the regexp, so
        // we use GetFlattenedField() instead.
        auto struct_type =
            std::static_pointer_cast<arrow::StructType>(struct_result->type());
        int field_index = struct_type->GetFieldIndex(field_name);
        auto captured_field_res = struct_result->GetFlattenedField(field_index);
        if (!captured_field_res.ok()) {
            throw std::runtime_error(
                "do_arrow_compute_regexp_substr: Error getting flattened "
                "field from struct array: " +
                captured_field_res.status().message());
        }
        captured_field = captured_field_res.ValueOrDie();
    }

    return arrow::Datum(captured_field);
}

arrow::Datum PhysicalArrowExpression::do_arrow_compute_regexp_instr(
    arrow::Datum res_datum, std::string pattern_str, bool get_start_index,
    bool extract_submatches, int64_t group_to_extract) {
    // Ensure all groups are named so that we can pass to
    // extract_regex_span. Do this once per operator.
    if (!named_regexp) {
        named_regexp = std::make_shared<std::tuple<std::string, int>>(
            convert_to_named_regexp(pattern_str));
    }

    std::string named_pattern = std::get<0>(*named_regexp);
    int num_groups = std::get<1>(*named_regexp);

    if (!extract_submatches || num_groups == 0) {
        // Wrap the whole pattern in a group
        named_pattern = "(?<_whole>" + named_pattern + ")";
        extract_submatches = false;
    }

    arrow::compute::ExtractRegexSpanOptions opts(named_pattern);
    auto extract_regex_span_result =
        do_arrow_compute_unary(res_datum, "extract_regex_span", &opts);

    // Convert to Arrow StructArray so we can extract the
    // field corresponding to the requested group
    std::shared_ptr<arrow::StructArray> struct_result =
        std::static_pointer_cast<arrow::StructArray>(
            extract_regex_span_result.make_array());

    // Extract the requested field
    std::shared_ptr<arrow::Array> captured_field_span = nullptr;
    if (extract_submatches && group_to_extract < num_groups) {
        // Valid group number requested
        captured_field_span = struct_result->GetFieldByName(
            "_group" + std::to_string(group_to_extract));
    } else if (!extract_submatches) {
        // No group extraction requested, return whole match
        captured_field_span = struct_result->GetFieldByName("_whole");
    }

    arrow::Datum captured_field_index_datum;
    if (!captured_field_span) {
        // Return array of -1 if requested group is greater than number
        // of groups in regex
        arrow::Int64Scalar negative_one_scalar(-1);
        captured_field_index_datum =
            arrow::Datum(arrow::MakeArrayFromScalar(negative_one_scalar,
                                                    struct_result->length())
                             .ValueOrDie());
    } else {
        // The values of the each StructArray field are two-element
        // fixed_size_lists. The first element of the FixedSizeList is the
        // start index of the substring matched by the group, and the second
        // element is the length of that substring.

        // Get zero-based start indices
        auto first_element_index = std::make_shared<arrow::Int64Scalar>(0);
        arrow::Datum start_indices = do_arrow_compute_binary(
            arrow::Datum(captured_field_span),
            arrow::Datum(first_element_index), "list_element");

        if (!get_start_index) {
            // If get_start_index is False, we should return the index of
            // the first character after the substring matched by the group.
            // So we need to add the substring length to the start index.

            // Get lengths
            auto second_element_index = std::make_shared<arrow::Int64Scalar>(1);
            arrow::Datum lengths = do_arrow_compute_binary(
                arrow::Datum(captured_field_span),
                arrow::Datum(second_element_index), "list_element");

            // Add lengths of to start indices
            captured_field_index_datum =
                do_arrow_compute_binary(start_indices, lengths, "add");
        } else {
            captured_field_index_datum = start_indices;
        }

        // The validity bitmap of the parent StructArray is not propagated
        // automatically to the child fields (see
        // https://github.com/apache/arrow/issues/41833), so we have to
        // check it manually and return -1 if not valid.
        auto negative_one_scalar = std::make_shared<arrow::Int64Scalar>(-1);
        arrow::Datum valid_mask =
            do_arrow_compute_unary(extract_regex_span_result, "is_valid");
        auto nulled_index_res = arrow::compute::CallFunction(
            "if_else",
            {valid_mask, captured_field_index_datum, negative_one_scalar});
        if (!nulled_index_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_regexp_instr: Error in Arrow compute "
                "(if_else): " +
                nulled_index_res.status().message());
        }
        captured_field_index_datum = nulled_index_res.ValueOrDie();
    }

    return captured_field_index_datum;
}

arrow::Datum do_arrow_compute_replace_substring_regex_single(
    arrow::Datum res_datum, std::string pattern_str,
    std::string replacement_str, int occurrence_num) {
    if (occurrence_num < 1) {
        throw std::invalid_argument(
            "occurrences_num argument to replace_substring_regex_single "
            "must be one or greater.");
    }

    // Use the regexp to split the input string occurrence_num - 1 times.
    // Thus the occurrence we are looking for will be in the final piece in
    // the list.
    arrow::compute::SplitPatternOptions split_opts{pattern_str,
                                                   occurrence_num - 1};
    // split_string is a ListArray
    arrow::Datum split_string =
        do_arrow_compute_unary(res_datum, "split_pattern_regex", &split_opts);

    // Get the final list element, which would contain the (occurrence_num)
    // occurrence if it existed. We can't use list_element directly, because
    // the lists in the ListArray could have different lengths depending on
    // the number of occurrences in the string, and list_element raises an
    // error instead of returning NULL in that case. Therefore, we first get
    // a slice of the (occurrence_num) element with the
    // return_fixed_size_list option set to true, which substitutes NULL
    // when the original list has no element corresponding to a particular
    // index.
    arrow::compute::ListSliceOptions list_slice_opts{occurrence_num - 1,
                                                     occurrence_num, 1, true};
    arrow::Datum string_tail_list =
        do_arrow_compute_unary(split_string, "list_slice", &list_slice_opts);

    // Get the singular list element from our slice of the last element in
    // split_string
    auto zero_scalar = std::make_shared<arrow::Int64Scalar>(0);
    arrow::Datum string_tail =
        do_arrow_compute_binary(string_tail_list, zero_scalar, "list_element");

    // Replace the first occurrence in the tail, which is the
    // (occurrence_num) occurrence in the full string
    arrow::compute::ReplaceSubstringOptions replace_substring_opts{
        pattern_str, replacement_str, 1};
    arrow::Datum replaced_tail = do_arrow_compute_unary(
        string_tail, "replace_substring_regex", &replace_substring_opts);

    // Calculate the length of the prefix string (unchanged portion with
    // (occurrence_num - 1) occurrences). prefix length = total length -
    // tail length
    arrow::Datum full_length = do_arrow_compute_unary(res_datum, "utf8_length");
    arrow::Datum tail_length =
        do_arrow_compute_unary(string_tail, "utf8_length");
    arrow::Datum prefix_length =
        do_arrow_compute_binary(full_length, tail_length, "subtract");

    // Take the substring up to the prefix length to get the prefix string
    arrow::Datum prefix_string = do_arrow_compute_multi_input_datum(
        {res_datum, zero_scalar, prefix_length}, "bodo_substr_three");

    // Concatenate the prefix string to the modified tail
    auto empty_string_scalar = std::make_shared<arrow::StringScalar>("");
    arrow::Datum combined_replaced_string = do_arrow_compute_multi_input_datum(
        {prefix_string, replaced_tail, empty_string_scalar},
        "binary_join_element_wise");

    // If the string tail is null, there were fewer than (occurrence_num)
    // occurrences
    arrow::Datum string_tail_is_null =
        do_arrow_compute_unary(string_tail, "is_null");
    // If occurrence doesn't exist, return original string unchanged
    arrow::Datum final_replaced_string = do_arrow_compute_multi_input_datum(
        {string_tail_is_null, res_datum, combined_replaced_string}, "if_else");

    return final_replaced_string;
}

/**
 * @brief If `count` == 0, returns empty string. If `count` > 0, returns the
 * substring that is left of the `count`-th occurrence of `delim_str` in
 * `res_datum`. If `count` < 0, returns the substring that is right of the
 * abs(count)-th ocurrence of `delim_str` in `res_datum`. If abs(count) is
 * higher than the number of occurrences of `delim_str`, returns the original
 * string (`res_datum`).
 */
arrow::Datum do_arrow_compute_substring_index(arrow::Datum res_datum,
                                              std::string delim_str,
                                              int count) {
    // Return empty string if count is 0 or delimiter is empty
    if (count == 0 || delim_str == "") {
        auto empty_string_scalar =
            (res_datum.type()->id() == arrow::Type::LARGE_STRING)
                ? std::static_pointer_cast<arrow::Scalar>(
                      std::make_shared<arrow::LargeStringScalar>(""))
                : std::static_pointer_cast<arrow::Scalar>(
                      std::make_shared<arrow::StringScalar>(""));
        auto empty_string_arr = ScalarToArrowArray(
            empty_string_scalar, res_datum.make_array()->length());
        return arrow::Datum(empty_string_arr);
    }

    int count_abs = std::abs(count);

    // Split the string on the delimiter count_abs times. The sign of `count`
    // controls the direction we search for `count_abs` delimiter occurrences.
    arrow::compute::SplitPatternOptions split_opts{delim_str, count_abs,
                                                   count < 0};
    // split_string is a ListArray. If `count` is positive, the last element of
    // each list is the part of the string we DON'T want to return. If `count`
    // is negative, it is the first element of each list we don't want to
    // return. Of course, if the string has more than `count_abs` occurrences of
    // the delimiter, we should return the full string.
    arrow::Datum split_string =
        do_arrow_compute_unary(res_datum, "split_pattern", &split_opts);

    // Our approach here is to extract the part of the string we DON'T want to
    // keep and determine its length and start position so we can use
    // bodo_substr_three to slice it off, leaving us with the substring to the
    // left of the (count_abs)-th delimiter occurrence from the left or the
    // substring to the right of the (count_abs)-th delimiter occurrence from
    // the right.

    // Get the first or the (count_abs + 1)-th list element from the
    // split_string. We can't use list_element directly, because the lists in
    // the ListArray could have different lengths depending on the number of
    // delimiter occurrences in the string, and list_element raises an error
    // instead of returning NULL in that case. Therefore, we first get a slice
    // containing only the first or the (count_abs + 1)-th element with the
    // return_fixed_size_list option set to true, which substitutes NULL when
    // the original list has no element corresponding to a particular index.
    arrow::compute::ListSliceOptions list_slice_opts;
    if (count > 0) {
        // Get the part of the string we would want to throw away if the string
        // has at least count_abs delimiters. (In this case, the last element,
        // since we want the substring to the left)
        list_slice_opts.start = count_abs;
        list_slice_opts.stop = count_abs + 1;
    } else {
        // Get the part of the string we would want to throw away if the string
        // has at least count_abs delimiters. (In this case, the first element,
        // since we want the substring to the right)
        list_slice_opts.start = 0;
        list_slice_opts.stop = 1;
    }
    list_slice_opts.return_fixed_size_list = true;
    arrow::Datum element_to_remove_list =
        do_arrow_compute_unary(split_string, "list_slice", &list_slice_opts);

    // Get the singular list element from our list slice.
    // If there are fewer than `count_abs` occurrences of the delimiter,
    // this will be NULL if count > 0 but not if count < 0.
    auto zero_scalar = std::make_shared<arrow::Int64Scalar>(0);
    arrow::Datum element_to_remove = do_arrow_compute_binary(
        element_to_remove_list, zero_scalar, "list_element");

    arrow::Datum to_remove_length =
        do_arrow_compute_unary(element_to_remove, "utf8_length");
    auto delim_len_scalar =
        std::make_shared<arrow::Int64Scalar>(delim_str.length());

    // Get the substring result (assuming there were enough delimiter
    // occurrences)
    arrow::Datum substring;
    if (count > 0) {
        // Substring left of the (count_abs)-th occurrence of delimiter from the
        // left res_datum[0: len(res_datum) - len(to_remove_part) - len(delim)]
        arrow::Datum full_length =
            do_arrow_compute_unary(res_datum, "utf8_length");
        arrow::Datum to_keep_length =
            do_arrow_compute_binary(full_length, to_remove_length, "subtract");
        to_keep_length = do_arrow_compute_binary(to_keep_length,
                                                 delim_len_scalar, "subtract");
        substring = do_arrow_compute_multi_input_datum(
            {res_datum, zero_scalar, to_keep_length}, "bodo_substr_three");
    } else {
        // Substring right of the (count_abs)-th occurrence of delimiter from
        // the right res_datum[len(to_remove_part) + len(delim): ]
        arrow::Datum start_index =
            do_arrow_compute_binary(to_remove_length, delim_len_scalar, "add");
        substring = do_arrow_compute_multi_input_datum({res_datum, start_index},
                                                       "bodo_substr_three");
    }

    // To be able to tell whether we should return the full string or not,
    // we need to get the actual length of the split_string list.
    // The length of each list is the minimum of `count_abs` + 1 and
    // number of delimiter occurrences + 1.
    arrow::Datum list_length =
        do_arrow_compute_unary(split_string, "list_value_length");

    // Make sure the full string is returned if there are fewer than `count_abs`
    // occurrences of the delimiter.
    auto count_abs_scalar = std::make_shared<arrow::Int32Scalar>(count_abs + 1);
    arrow::Datum return_full_string =
        do_arrow_compute_binary(list_length, count_abs_scalar, "less");
    return do_arrow_compute_multi_input_datum(
        {return_full_string, res_datum, substring}, "if_else");
}

/**
 * @brief Occurrences of `delim_str` divide the input string `res_datum`
 * into parts. SPLIT_PART returns the substring corresponding to a part
 * number, where 1 is the first part. If the part number is negative,
 * the counting happens from the right. If `delim_str` is empty,
 * `res_datum` is returned as is. If there are fewer than abs(part_num)
 * parts in the string, an empty string is emitted.
 */
arrow::Datum do_arrow_compute_split_part(arrow::Datum res_datum,
                                         std::string delim_str, int part_num) {
    // Snowflake returns the original string when the delimiter is empty
    if (delim_str == "") {
        return res_datum;
    }

    int part_num_abs = std::abs(part_num);

    // Split the string on the delimiter (part_num_abs) times. The sign of
    // `part_num` controls the direction we search for (part_num_abs) delimiter
    // occurrences. It's important than we don't split on all the occurrences of
    // the delimiter in the string. If we did, a slew of Arrow limitations would
    // make it much harder to pick the right list element when part_num is
    // negative. This way, we know the second element is the one we are looking
    // for.
    arrow::compute::SplitPatternOptions split_opts{delim_str, part_num_abs,
                                                   part_num < 0};
    // split_string is a ListArray
    arrow::Datum split_string =
        do_arrow_compute_unary(res_datum, "split_pattern", &split_opts);

    // Get a single-element list of the part we want, with
    // return_fixed_size_list = true. We need to do this since calling
    // list_element directly will throw an error if there are fewer than
    // `part_num` parts in the split string (for positive `part_num`).
    arrow::compute::ListSliceOptions list_slice_opts;
    if (part_num > 0) {
        // part_num is one-based, so part_num - 1 is the index corresponding to
        // the part
        list_slice_opts.start = part_num - 1;
        list_slice_opts.stop = part_num;
    } else {
        // When the part number is negative, we count right to left.
        // We split the string into (at most) part_num_abs + 1 parts, so the
        // leftmost part is the remaining non-split segment, and the second
        // part is the requested part.
        list_slice_opts.start = 1;
        list_slice_opts.stop = 2;
    }
    list_slice_opts.return_fixed_size_list = true;
    arrow::Datum part_list =
        do_arrow_compute_unary(split_string, "list_slice", &list_slice_opts);

    // Get the actual string part. May or may not be NULL.
    // The reliable test for whether the requested part exists
    // is the length of split_string.
    auto zero_scalar = std::make_shared<arrow::Int64Scalar>(0);
    arrow::Datum part =
        do_arrow_compute_binary(part_list, zero_scalar, "list_element");

    // To be able to tell whether we should return an empty string or not,
    // we need to get the actual length of the split_string list.
    // The length of each list is the minimum of `part_num_abs` + 1 and
    // number of delimiter occurrences + 1.
    arrow::Datum list_length =
        do_arrow_compute_unary(split_string, "list_value_length");

    // If there's only one element / part, just use that.
    // The above code for part_num < 0 assumes two or more parts
    // (at least one delimiter occurrence).
    arrow::Datum first_part =
        do_arrow_compute_binary(split_string, zero_scalar, "list_element");
    auto one_scalar = std::make_shared<arrow::Int32Scalar>(1);
    arrow::Datum more_than_one_part =
        do_arrow_compute_binary(list_length, one_scalar, "greater");
    part = do_arrow_compute_multi_input_datum(
        {more_than_one_part, part, first_part}, "if_else");

    // Make sure an empty string is returned if there are fewer than
    // `part_num_abs` parts.
    auto part_num_abs_scalar =
        std::make_shared<arrow::Int32Scalar>(part_num_abs);
    arrow::Datum return_empty_string =
        do_arrow_compute_binary(list_length, part_num_abs_scalar, "less");

    auto empty_string_scalar =
        (res_datum.type()->id() == arrow::Type::LARGE_STRING)
            ? std::static_pointer_cast<arrow::Scalar>(
                  std::make_shared<arrow::LargeStringScalar>(""))
            : std::static_pointer_cast<arrow::Scalar>(
                  std::make_shared<arrow::StringScalar>(""));
    return do_arrow_compute_multi_input_datum(
        {return_empty_string, empty_string_scalar, part}, "if_else");
}

/**
 * @brief Occurrences matched by the regex `delim_str` divide the input string
 * `res_datum` into tokens. STRTOK returns the substring corresponding to a part
 * number, where 1 is the first token. The part number cannot be negative.
 * NULL is emitted if the requested token does not exist.
 *
 * Note: If called directly, this function differs from Snowflake's STRTOK in
 * that `delim_str` is a regex pattern instead of a string where each character
 * is considered a delimiter. The Python side converts the raw delimiter input
 * to a regexp that matches any of the given delimeters. The Python side is also
 * in charge of handling the edge cases (such as when the delimiter is empty).
 */
arrow::Datum do_arrow_compute_strtok(arrow::Datum res_datum,
                                     std::string delim_regexp, int part_num) {
    // Split the string on the delimiter `part_num` times, which is the minimum
    // number of times to find the desired token.
    arrow::compute::SplitPatternOptions split_opts{delim_regexp, part_num};
    // string_tokens is a ListArray
    arrow::Datum string_tokens =
        do_arrow_compute_unary(res_datum, "split_pattern_regex", &split_opts);

    // Get a single-element list of the token we want, with
    // return_fixed_size_list = true. We need to do this since calling
    // list_element directly will throw an error if there are fewer than
    // `part_num` token in the split string.
    arrow::compute::ListSliceOptions list_slice_opts;
    // part_num is one-based, so part_num - 1 is the index corresponding to
    // the token
    list_slice_opts.start = part_num - 1;
    list_slice_opts.stop = part_num;
    list_slice_opts.return_fixed_size_list = true;
    arrow::Datum token_list =
        do_arrow_compute_unary(string_tokens, "list_slice", &list_slice_opts);

    // Get the actual string part. This will be NULL if
    // the string has fewer than `part_num` tokens, which
    // isn't a problem since STRTOK is supposed to return
    // NULL in that case.
    auto zero_scalar = std::make_shared<arrow::Int64Scalar>(0);
    arrow::Datum token =
        do_arrow_compute_binary(token_list, zero_scalar, "list_element");
    return token;
}

arrow::Datum do_arrow_compute_dow_num(arrow::Datum res_datum) {
    // We strip off the leading spaces and only look at the first two
    // characters of the input string in accordance with Snowflake (e.g.
    // NEXT_DAY/PREVIOUS_DAY)

    // Create Arrow array containing the days of the week. The order (index)
    // determines the result DoW number.
    arrow::StringBuilder builder;
    arrow::Status status =
        builder.AppendValues({"mo", "tu", "we", "th", "fr", "sa", "su"});
    if (!status.ok()) {
        throw std::runtime_error(
            "do_arrow_compute_dow_num: Failed to append values to "
            "StringBuilder");
    }
    std::shared_ptr<arrow::Array> dow_array = builder.Finish().ValueOrDie();

    // Normalize string to two lowercase characters representing the day of
    // the week
    arrow::Datum trimmed_dow_string =
        do_arrow_compute_unary(res_datum, "utf8_ltrim_whitespace");
    arrow::compute::SliceOptions slice_opts(0, 2, 1);
    arrow::Datum sliced_dow_string = do_arrow_compute_unary(
        trimmed_dow_string, "utf8_slice_codeunits", &slice_opts);
    arrow::Datum lowered_dow_string =
        do_arrow_compute_unary(sliced_dow_string, "utf8_lower");

    // Get index of string into DoW array, which equals the DoW number
    arrow::compute::SetLookupOptions set_lookup_opts(dow_array);
    arrow::Datum dow_num = do_arrow_compute_unary(lowered_dow_string,
                                                  "index_in", &set_lookup_opts);

    return dow_num;
}

arrow::Datum PhysicalArrowExpression::do_arrow_compute_random_int64(
    arrow::Datum res_datum) {
    // Get dummy input as an Arrow array.
    // We need the result to match the length of this array.
    std::shared_ptr<arrow::Array> res_array = res_datum.make_array();

    // Only create PRNG once so we keep state across batches
    // and don't start from the same position for each batch.
    if (!gen) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        auto [seed_arg] = get_var_py_args_as_types<0, 1>(
            scalar_func_data.args, scalar_func_data.arrow_func_name.c_str(),
            get_py_object_as_int64);

        if (seed_arg.has_value()) {
            // Seed was explicitly provided, so use it
            int64_t seed = *seed_arg;

            if (rank == 0) {
                // Create psuedo-random number generator.
                // For rank 0 we use the input seed directly
                gen = std::make_shared<std::mt19937_64>(std::mt19937_64(seed));
            } else {
                // seed_seq applies a transformation on the master seed and
                // the rank number to scramble and break linear correlation
                // between the seeds used by each rank.
                // Because of this, the generated sequence will be different
                // depending on the number of workers used, but it should
                // still be deterministic.
                // The Dynamic Creator algorithm might be theoretically
                // better here, but seed_seq is probably sufficient in
                // practice.

                // Note that we have to split the 64-bit input seed because
                // seed_seq only accepts 32-bit integers
                std::seed_seq rank_seed{
                    static_cast<uint32_t>(seed >> 32),
                    static_cast<uint32_t>(seed & 0xFFFFFFFF),
                    static_cast<uint32_t>(rank)};

                // Create PRNG with a (hopefully) independent seed
                gen = std::make_shared<std::mt19937_64>(
                    std::mt19937_64(rank_seed));
            }
        } else {
            // Generate a seed with 96-bits of entropy from system's
            // random_device and the current system time. time(NULL) is
            // mainly a backup in case std::random_device falls back
            // to a deterministic implementation.
            // We also integrate the rank number in the seed calculation
            // to ensure that two ranks do not get the same seed by chance.
            std::random_device rd;
            std::seed_seq seed{rd(), static_cast<uint32_t>(time(NULL)), rd(),
                               static_cast<uint32_t>(rank)};
            // Create psuedo-random number generator
            gen = std::make_shared<std::mt19937_64>(std::mt19937_64(seed));
        }
    }

    // Full-range uniform int64 distribution
    std::uniform_int_distribution<int64_t> int64_dist(0x8000000000000000,
                                                      0x7FFFFFFFFFFFFFFF);

    // Compute array of random integers based on the length of the dummy
    // input
    arrow::Int64Builder builder;
    for (int i = 0; i < res_array->length(); i++) {
        arrow::Status status = builder.Append(int64_dist(*gen));
        if (!status.ok()) {
            throw std::runtime_error(
                "do_arrow_compute (random_int64): Failed to append "
                "value to "
                "Int64Builder");
        }
    }
    std::shared_ptr<arrow::Array> random_int64_array =
        builder.Finish().ValueOrDie();

    return arrow::Datum(random_int64_array);
}

/**
 * @brief Zip the Datums into a ListArray, provided they all have the
 * same datatype. If N datums are passed, and each datum has R rows,
 * the result will be an array with R rows where each array value is
 * a list containing N elements.
 *
 * Scalars are accepted. If all arguments are scalars, the result will
 * be a ListArray with one value (one list).
 * `datums` must not be empty.
 */
arrow::Datum do_arrow_compute_zip(const std::vector<arrow::Datum>& datums) {
    if (datums.empty()) {
        throw std::invalid_argument(
            "do_arrow_compute_zip does not accept an empty vector of datums.");
    }

    // First, loop over the datums to get the number of rows and
    // verify that all array datums have the same length.
    // Another option would be to pad with nulls when the lengths
    // are not equal.
    int64_t num_rows = 1;
    for (const arrow::Datum& datum : datums) {
        if (!datum.is_scalar()) {
            int64_t datum_length = datum.length();
            if (datum_length == -1) {
                throw std::runtime_error(
                    "do_arrow_compute_zip: Failed to get length of input "
                    "datum.");
            }
            if (datum_length != num_rows && num_rows != 1) {
                throw std::invalid_argument(
                    "do_arrow_compute_zip: Input array datums must have the "
                    "same length.");
            }
            num_rows = datum_length;
        }
    }

    // Ensure all datums have the same datatype
    std::shared_ptr<arrow::DataType> value_type = datums[0].type();
    for (size_t i = 1; i < datums.size(); i++) {
        if (!value_type->Equals(datums[i].type())) {
            throw std::invalid_argument(
                "do_arrow_compute_zip: Input datums must have the same "
                "datatype.");
        }
    }

    std::shared_ptr<arrow::Array> values_array;

    // The interleaving step is only necessary if more than one datum is passed.
    if (datums.size() > 1) {
        // First, turn all datums into arrays. Scalar datums will become
        // single-element arrays. This way we can make an ArraySpan and use the
        // efficient AppendArraySlice() for both array and scalar datums.
        std::vector<std::shared_ptr<arrow::ArrayData>> input_array_datas(
            datums.size());
        for (size_t i = 0; i < datums.size(); i++) {
            if (datums[i].is_scalar()) {
                input_array_datas[i] =
                    arrow::MakeArrayFromScalar(*datums[i].scalar(), 1)
                        .ValueOrDie()
                        ->data();
            } else {
                input_array_datas[i] = datums[i].array();
            }
        }

        // Helper struct for performance and to accommodate both arrays and
        // scalars
        struct InputView {
            arrow::ArraySpan span;
            // What to multiply the row index variable by in the builder loop.
            // 0 for scalars and 1 for arrays. This avoids the branching of
            // checking whether input is an array or scalar every iteration.
            int index_multiplier;
        };
        std::vector<InputView> inputs(datums.size());

        // Package input data into InputViews for faster memory
        // access in the builder loop.
        for (size_t i = 0; i < datums.size(); i++) {
            InputView input_view;
            // Convert input ArrayData to ArraySpan
            input_view.span = arrow::ArraySpan(*input_array_datas[i]);
            if (datums[i].is_scalar()) {
                // Scalars are not broadcasted to number of rows, so
                // the index multiplier should be 0 to grab the first
                // element over and over.
                input_view.index_multiplier = 0;
            } else {
                input_view.index_multiplier = 1;
            }
            inputs[i] = input_view;
        }

        // Make generic builder
        std::unique_ptr<arrow::ArrayBuilder> interleaved_arr_builder;
        auto make_builder_status = arrow::MakeBuilder(
            arrow::default_memory_pool(), value_type, &interleaved_arr_builder);
        if (!make_builder_status.ok()) {
            throw std::runtime_error(
                "do_arrow_compute_zip: Failed to create ArrayBuilder for " +
                value_type->ToString() + ": " + make_builder_status.ToString());
        }

        // Presize array to minimize resize operations. If `value_type` is a
        // variable-sized type, this doesn't guarantee that no reallocation will
        // occur.
        auto presize_builder_status =
            interleaved_arr_builder->Resize(num_rows * datums.size());
        if (!presize_builder_status.ok()) {
            throw std::runtime_error(
                "do_arrow_compute_zip: Failed to presize ArrayBuilder: " +
                presize_builder_status.ToString());
        }

        // Append elements one row at a time (the first elements of arrays 1..N,
        // then the second elements, etc.). This is not going to be ideal for
        // cache locality, but there may not be a better generic solution that
        // handles variable-width datatypes.
        for (int64_t i = 0; i < num_rows; i++) {
            for (const InputView& input : inputs) {
                // Append element at index i (or 0, for scalars) from the array.
                // AppendArraySlice should be faster than getting the scalar at
                // position i and then appending that.
                arrow::Status append_status =
                    interleaved_arr_builder->AppendArraySlice(
                        input.span, input.index_multiplier * i, 1);
                if (!append_status.ok()) {
                    throw std::runtime_error(
                        "do_arrow_compute_zip: Failed to append scalar to "
                        "builder: " +
                        append_status.ToString());
                }
            }
        }
        auto values_array_res = interleaved_arr_builder->Finish();
        if (!values_array_res.ok()) {
            throw std::runtime_error(
                "do_arrow_compute_zip: Failed to finish ArrayBuilder: " +
                values_array_res.status().message());
        }
        values_array = values_array_res.ValueOrDie();
    } else {
        // If only one datum is passed, this function is the equivalent of
        // turning the array into an array of single-element lists. We should be
        // able to create a ListArray from the original array without any
        // copying.
        values_array =
            datums[0].is_scalar()
                ? arrow::MakeArrayFromScalar(*datums[0].scalar(), num_rows)
                      .ValueOrDie()
                : datums[0].make_array();
    }

    // Make offsets array. The offsets represent the indices of the boundaries
    // between the lists, so they are separated by the number of input datums.
    arrow::Int64Builder offsets_builder;
    auto presize_builder_status = offsets_builder.Resize(num_rows + 1);
    if (!presize_builder_status.ok()) {
        throw std::runtime_error(
            "do_arrow_compute_zip: Failed to presize Int64Builder: " +
            presize_builder_status.ToString());
    }
    for (int64_t i = 0; i <= num_rows; i++) {
        offsets_builder.UnsafeAppend(i * datums.size());
    }
    // Contains 0, N, 2N, ... R*N
    auto offsets = offsets_builder.Finish().ValueOrDie();

    // Make LargeListArray from interleaved values array and offsets array
    auto list_type = arrow::large_list(value_type);
    auto list_array_res =
        arrow::LargeListArray::FromArrays(list_type, *offsets, *values_array);
    if (!list_array_res.ok()) {
        throw std::runtime_error(
            "do_arrow_compute_zip: Failed to make LargeListArray: " +
            list_array_res.status().message());
    }

    return arrow::Datum(list_array_res.ValueOrDie());
}

// -----------------------------------------------------------------------

#undef CHECK_ARROW
