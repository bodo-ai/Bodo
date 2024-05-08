// Copyright (C) 2020 Bodo Inc. All rights reserved.
// C/C++ code for DecimalArray handling
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>

#include <arrow/compute/cast.h>
#include <arrow/python/pyarrow.h>
#include <arrow/util/bit_util.h>
#include <arrow/util/decimal.h>
#include <fmt/format.h>
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_gandiva_decimal_copy.h"

// using scale 18 when converting from Python Decimal objects (same as Spark)
#define PY_DECIMAL_SCALE 18

#pragma pack(1)
struct decimal_value {
    int64_t low;
    int64_t high;
};

/**
 * @brief enum for designating comparison operators passed from Python.
 * Most match with definition in decimal_arr_ext.py.
 */
enum CmpOp {
    LT = 0,
    LE = 1,
    EQ = 2,
    NE = 3,
    GT = 4,
    GE = 5,
};

std::string decimal_to_std_string(arrow::Decimal128 const& arrow_decimal,
                                  int const& scale) {
    // TODO(srilman): I think Arrow Decimal128::ToString() had trailing zeros in
    // past, but when I tested, it didn't. Maybe we can get rid of this
    // function entirely
    std::string str = arrow_decimal.ToString(scale);
    if (str.find('.') == std::string::npos) {
        return str;
    }

    // str may be of the form 0.45000000000 or 4.000000000
    size_t last_char = str.length();
    while (true) {
        if (str[last_char - 1] != '0')
            break;
        last_char--;
    }
    // position reduce str to 0.45  or 4.
    if (str[last_char - 1] == '.')
        last_char--;

    // Slice String to New Range
    str = str.substr(0, last_char);
    if (str == "0.E-18")
        return "0";
    return str;
}

std::string int128_decimal_to_std_string(__int128 const& val,
                                         int const& scale) {
    arrow::Decimal128 arrow_decimal((int64_t)(val >> 64), (int64_t)(val));
    return decimal_to_std_string(arrow_decimal, scale);
}

double decimal_to_double(__int128 const& val, uint8_t scale) {
    // TODO: Zero-copy (cast __int128 to int64[2] for Decimal128 constructor)
    // Can't figure out how to do this in C++
    arrow::Decimal128 dec((int64_t)(val >> 64), (int64_t)(val));
    return dec.ToDouble(scale);
}

arrow::Decimal128 shift_decimal_scalar(arrow::Decimal128 val,
                                       int64_t shift_amount) {
    if (shift_amount == 0) {
        return val;
    } else if (shift_amount > 0) {
        return val.IncreaseScaleBy(shift_amount);
    } else {
        // We always round "half up".
        return val.ReduceScaleBy(-shift_amount, true);
    }
}

arrow::Decimal128 cast_decimal_to_decimal_scalar_unsafe_py_entry(
    arrow::Decimal128 val, int64_t shift_amount) {
    try {
        return shift_decimal_scalar(val, shift_amount);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return arrow::Decimal128(0);
    }
}

arrow::Decimal128 cast_decimal_to_decimal_scalar_safe_py_entry(
    arrow::Decimal128 val, int64_t shift_amount, int64_t max_power_of_ten,
    bool* safe) {
    try {
        bool safe_cast = val.FitsInPrecision(max_power_of_ten);
        *safe = safe_cast;
        if (!safe_cast) {
            return val;
        } else {
            return shift_decimal_scalar(val, shift_amount);
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return arrow::Decimal128(0);
    }
}

/**
 * @brief Check if the input array can be safely cast to the output array. If
 * null_on_error is true, then the output array's null bitmap is updated to
 * reflect whether the input array can be safely cast. If null_on_error is
 * false, then an exception is thrown if the input array cannot be safely cast.
 *
 * @param[in] input_array The input array.
 * @param max_power_of_ten The maximum power of ten that every value must fit
 * inside.
 * @param null_on_error Should we throw an exception if the input array cannot
 * be safely cast.
 * @param[out] output_array The output array for updating the null bitmap if
 * null_on_error is true.
 */
void check_cast_safe(std::shared_ptr<array_info> input_array,
                     int64_t max_power_of_ten, bool null_on_error,
                     std::shared_ptr<array_info> output_array) {
    arrow::Decimal128* input_data = reinterpret_cast<arrow::Decimal128*>(
        input_array->data1<bodo_array_type::NULLABLE_INT_BOOL>());
    uint8_t* input_null_bitmap = reinterpret_cast<uint8_t*>(
        input_array->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>());
    if (null_on_error) {
        uint8_t* output_null_bitmap = reinterpret_cast<uint8_t*>(
            output_array->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>());
        for (size_t i = 0; i < input_array->length; i++) {
            if (arrow::bit_util::GetBit(input_null_bitmap, i)) {
                arrow::bit_util::SetBitTo(
                    output_null_bitmap, i,
                    input_data[i].FitsInPrecision(max_power_of_ten));
            }
        }
    } else {
        for (size_t i = 0; i < input_array->length; i++) {
            if (arrow::bit_util::GetBit(input_null_bitmap, i) &&
                !input_data[i].FitsInPrecision(max_power_of_ten)) {
                throw std::runtime_error("Number out of representable range");
            }
        }
    }
}

/**
 * @brief Shift all non-null values in a decimal array by a given amount into
 * output array. This is done to enable a shared implementation between safe
 * and unsafe casts.
 *
 * @param[in] input_array The input array to shift.
 * @param shift_amount The amount to shift by.
 * @param[out] output_array The output array for updating the shifted values. We
 * assume that the null bitmap has been preallocated to either all 1s if doing
 * an unsafe cast, or some nulls if doing a safe cast and the values are not
 * safe to cast.
 */
void shift_decimal_array(std::shared_ptr<array_info> input_array,
                         int64_t shift_amount,
                         std::shared_ptr<array_info> output_array) {
    arrow::Decimal128* input_data = reinterpret_cast<arrow::Decimal128*>(
        input_array->data1<bodo_array_type::NULLABLE_INT_BOOL>());
    uint8_t* input_null_bitmap = reinterpret_cast<uint8_t*>(
        input_array->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>());
    arrow::Decimal128* output_data = reinterpret_cast<arrow::Decimal128*>(
        output_array->data1<bodo_array_type::NULLABLE_INT_BOOL>());
    uint8_t* output_null_bitmap = reinterpret_cast<uint8_t*>(
        output_array->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>());
    if (shift_amount == 0) {
        // TODO: Remove this copy. We can avoid the copy in the unsafe code path
        // because it doesn't modify the null bit mask. However, adding a
        // no_copy argument seems too risky because someone may modify the data
        // inplace.
        memcpy((char*)output_data, (char*)input_data,
               input_array->length * sizeof(arrow::Decimal128));
        for (size_t i = 0; i < input_array->length; i++) {
            bool bit = arrow::bit_util::GetBit(input_null_bitmap, i) &&
                       arrow::bit_util::GetBit(output_null_bitmap, i);
            arrow::bit_util::SetBitTo(output_null_bitmap, i, bit);
        }
    } else if (shift_amount > 0) {
        for (size_t i = 0; i < input_array->length; i++) {
            if (arrow::bit_util::GetBit(input_null_bitmap, i) &&
                arrow::bit_util::GetBit(output_null_bitmap, i)) {
                output_data[i] = input_data[i].IncreaseScaleBy(shift_amount);
            } else {
                arrow::bit_util::ClearBit(output_null_bitmap, i);
            }
        }
    } else {
        shift_amount = -shift_amount;
        for (size_t i = 0; i < input_array->length; i++) {
            if (arrow::bit_util::GetBit(input_null_bitmap, i) &&
                arrow::bit_util::GetBit(output_null_bitmap, i)) {
                output_data[i] =
                    input_data[i].ReduceScaleBy(shift_amount, true);
            } else {
                arrow::bit_util::ClearBit(output_null_bitmap, i);
            }
        }
    }
}

array_info* cast_decimal_to_decimal_array_unsafe_py_entry(
    array_info* arr, int64_t shift_amount) {
    try {
        std::shared_ptr<array_info> input_array =
            std::shared_ptr<array_info>(arr);
        std::shared_ptr<array_info> output_array =
            alloc_nullable_array_no_nulls(input_array->length,
                                          Bodo_CTypes::DECIMAL);
        shift_decimal_array(input_array, shift_amount, output_array);
        return new array_info(*output_array);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

array_info* cast_decimal_to_decimal_array_safe_py_entry(
    array_info* arr, int64_t shift_amount, int64_t max_power_of_ten,
    bool null_on_error) {
    try {
        std::shared_ptr<array_info> input_array =
            std::shared_ptr<array_info>(arr);
        std::shared_ptr<array_info> output_array =
            alloc_nullable_array_no_nulls(input_array->length,
                                          Bodo_CTypes::DECIMAL);
        check_cast_safe(input_array, max_power_of_ten, null_on_error,
                        output_array);
        shift_decimal_array(input_array, shift_amount, output_array);
        return new array_info(*output_array);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

double decimal_to_double_py_entry(decimal_value val, uint8_t scale) {
    auto high = static_cast<uint64_t>(val.high);
    auto low = static_cast<uint64_t>(val.low);
    return decimal_to_double((static_cast<__int128>(high) << 64) | low, scale);
}

/**
 * @brief Multiply two decimal scalars with the given precision and scale
 * and return the output. The output should have its scale truncated to
 * the provided output scale. If overflow is detected, then the overflow
 * need to be updated to true.
 *
 * @param v1 First decimal value
 * @param p1 Precision of first decimal value
 * @param s1 Scale of first decimal value
 * @param v2 Second decimal value
 * @param p2 Precision of second decimal value
 * @param s2 Scale of second decimal value
 * @param out_precision Output precision
 * @param out_scale Output scale
 * @param[out] overflow Overflow flag
 * @return arrow::Decimal128
 */
arrow::Decimal128 multiply_decimal_scalars_py_entry(
    arrow::Decimal128 v1, int64_t p1, int64_t s1, arrow::Decimal128 v2,
    int64_t p2, int64_t s2, int64_t out_precision, int64_t out_scale,
    bool* overflow) {
    try {
        arrow::Decimal128Scalar val1(v1, arrow::decimal128(p1, s1));
        arrow::Decimal128Scalar val2(v2, arrow::decimal128(p2, s2));
        return decimalops::Multiply(val1, val2, out_precision, out_scale,
                                    overflow);

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return arrow::Decimal128(0);
    }
}

/**
 * @brief Convert decimal value to int64 (unsafe cast)
 *
 * @param val input decimal value
 * @param precision input's precision
 * @param scale input's scale
 * @return int64_t input converted to int64
 */
int64_t decimal_to_int64_py_entry(decimal_value val, uint8_t precision,
                                  uint8_t scale) {
    try {
        // NOTE: using cast to allow unsafe cast (Rescale/ToInteger may throw
        // data loss error)
        arrow::Decimal128 arrow_decimal(val.high, val.low);
        arrow::Decimal128Scalar val_scalar(arrow_decimal,
                                           arrow::decimal128(precision, scale));

        auto res = arrow::compute::Cast(val_scalar, arrow::int64(),
                                        arrow::compute::CastOptions::Unsafe(),
                                        bodo::default_buffer_exec_context());

        if (!res.ok()) {
            std::string err_msg = "Arrow decimal to int64 failed for " +
                                  arrow_decimal.ToString(scale) + "\n";
            throw std::runtime_error(err_msg);
        }

        std::shared_ptr<arrow::Int64Scalar> out =
            std::static_pointer_cast<arrow::Int64Scalar>(
                res.ValueOrDie().scalar());

        return out->value;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}

/**
 * Copies the elements of an integer array into a decimal array, upcasting as
 * necessary.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
void copy_integer_arr_to_decimal_arr(std::shared_ptr<array_info>& int_arr,
                                     std::unique_ptr<array_info>& dec_arr) {
    using T = typename dtype_to_type<DType>::type;
    // Set the precision of the array type based on the integer dtype
    uint64_t siztype = numpy_item_size[DType];
    switch (siztype) {
        case 1: {
            dec_arr->precision = 3;
            break;
        }
        case 2: {
            dec_arr->precision = 5;
            break;
        }
        case 4: {
            dec_arr->precision = 9;
            break;
        }
        case 8: {
            dec_arr->precision = 19;
            break;
        }
        default: {
            dec_arr->precision = 38;
            break;
        }
    }
    // Copy the integer elements into the decimal buffer, upcasting
    // along the way.
    size_t rows = int_arr->length;
    __int128_t* res_buffer =
        dec_arr->data1<bodo_array_type::NULLABLE_INT_BOOL, __int128_t>();
    if constexpr (ArrType == bodo_array_type::NULLABLE_INT_BOOL) {
        // If the input array is nullable, copy over its null bitmask.
        // Otherwise, we don't need to do anything since the decimal array was
        // created with no nulls.
        int64_t n_bytes = arrow::bit_util::BytesForBits(rows);
        memcpy(dec_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               int_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               n_bytes);
    }
    for (size_t row = 0; row < rows; row++) {
        T val = int_arr->data1<ArrType, T>()[row];
        res_buffer[row] = (__int128_t)(val);
    }
}

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW(expr, msg)                                              \
    if (!(expr.ok())) {                                                     \
        std::string err_msg = std::string("Error in decimal utilities: ") + \
                              msg + " " + expr.ToString();                  \
        throw std::runtime_error(err_msg);                                  \
    }
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

arrow::Decimal128 cast_float_to_decimal_scalar_py_entry(double f,
                                                        int32_t precision,
                                                        int32_t scale,
                                                        bool* safe) {
    try {
        double max_value = std::pow(10.0, precision - scale);
        bool safe_cast = std::abs(f) < max_value;
        *safe = safe_cast;
        if (!safe_cast) {
            return arrow::Decimal128(0);
        } else {
            arrow::Decimal128 answer;
            auto result = arrow::Decimal128::FromReal(f, precision, scale);
            CHECK_ARROW_AND_ASSIGN(result, "failed to convert float to decimal",
                                   answer)
            return answer;
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return arrow::Decimal128(0);
    }
}

/**
 * Copies the elements of an float array into a decimal array.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool null_on_error>
void float_to_decimal_arr(std::shared_ptr<array_info>& arr_in,
                          std::unique_ptr<array_info>& arr_out,
                          int32_t precision, int32_t scale) {
    using T = typename dtype_to_type<DType>::type;
    T max_value = std::pow(10.0, precision - scale);
    size_t n_rows = arr_in->length;
    T* in_buffer = arr_in->data1<ArrType, T>();
    decimal_value* out_buffer =
        arr_out->data1<bodo_array_type::NULLABLE_INT_BOOL, decimal_value>();
    for (size_t row = 0; row < n_rows; row++) {
        T f = in_buffer[row];
        if (is_null_at<ArrType, T, DType>(*arr_in, row)) {
            arr_out->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(row, 0);
        } else if (isnan_alltype<T, DType>(f) || std::abs(f) >= max_value) {
            if constexpr (null_on_error) {
                arr_out->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(row,
                                                                          0);
            } else {
                throw std::runtime_error("Invalid float to decimal cast: " +
                                         std::to_string(f));
            }
        } else {
            arrow::Decimal128 answer;
            auto result = arrow::Decimal128::FromReal(f, precision, scale);
            CHECK_ARROW_AND_ASSIGN(result, "failed to convert float to decimal",
                                   answer);
            uint64_t low_bits = answer.low_bits();
            int64_t high_bits = answer.high_bits();
            decimal_value res = {static_cast<int64_t>(low_bits), high_bits};
            out_buffer[row] = res;
        }
    }
}

array_info* cast_float_to_decimal_array_py_entry(array_info* in_raw,
                                                 int32_t precision,
                                                 int32_t scale,
                                                 bool null_on_error) {
#define float_to_dec_case(ArrType, Dtype)                                      \
    if (null_on_error) {                                                       \
        float_to_decimal_arr<ArrType, Dtype, true>(in_arr, out_arr, precision, \
                                                   scale);                     \
    } else {                                                                   \
        float_to_decimal_arr<ArrType, Dtype, false>(in_arr, out_arr,           \
                                                    precision, scale);         \
    }
    try {
        auto in_arr = std::shared_ptr<array_info>(in_raw);
        size_t n_rows = in_arr->length;
        std::unique_ptr<array_info> out_arr =
            alloc_nullable_array_no_nulls(n_rows, Bodo_CTypes::DECIMAL);
        if (in_arr->arr_type == bodo_array_type::NUMPY) {
            if (in_arr->dtype == Bodo_CTypes::FLOAT32) {
                float_to_dec_case(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT32);
            } else if (in_arr->dtype == Bodo_CTypes::FLOAT64) {
                float_to_dec_case(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64);
            } else {
                throw std::runtime_error(
                    "Invalid dtype for cast_float_to_decimal_array_py_entry: " +
                    GetDtype_as_string(in_arr->dtype));
            }
        } else if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            if (in_arr->dtype == Bodo_CTypes::FLOAT32) {
                float_to_dec_case(bodo_array_type::NULLABLE_INT_BOOL,
                                  Bodo_CTypes::FLOAT32);
            } else if (in_arr->dtype == Bodo_CTypes::FLOAT64) {
                float_to_dec_case(bodo_array_type::NULLABLE_INT_BOOL,
                                  Bodo_CTypes::FLOAT64);
            } else {
                throw std::runtime_error(
                    "Invalid dtype for cast_float_to_decimal_array_py_entry: " +
                    GetDtype_as_string(in_arr->dtype));
            }
        } else {
            throw std::runtime_error(
                "Invalid array type for "
                "cast_float_to_decimal_array_py_entry: " +
                GetDtype_as_string(in_arr->arr_type));
        }
        return out_arr.release();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN

/**
 * Converts an integer array into a decimal array.
 * @param[in] arr_raw: the array info pointer of the input integer
 * nullable/numpy array.
 * @return an equivalent decimal array to arr_raw (with scale=0).
 */
array_info* int_to_decimal_array(array_info* arr_raw) {
    std::shared_ptr<array_info> arr = std::shared_ptr<array_info>(arr_raw);
    // Allocate a decimal array with the correct length (the precision will
    // be set later)
    size_t rows = arr->length;
    std::unique_ptr<array_info> res_arr =
        alloc_nullable_array_no_nulls(rows, Bodo_CTypes::DECIMAL);
    res_arr->scale = 0;
#define dtype_case(dtype)                                                   \
    case dtype: {                                                           \
        if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {          \
            copy_integer_arr_to_decimal_arr<                                \
                bodo_array_type::NULLABLE_INT_BOOL, dtype>(arr, res_arr);   \
        } else if (arr->arr_type == bodo_array_type::NUMPY) {               \
            copy_integer_arr_to_decimal_arr<bodo_array_type::NUMPY, dtype>( \
                arr, res_arr);                                              \
        } else {                                                            \
            throw std::runtime_error(                                       \
                "Invalid array type for int_to_decimal_array: " +           \
                GetArrType_as_string(arr->arr_type));                       \
        }                                                                   \
        break;                                                              \
    }
    // Map to the helper based on the correct dtype & array type
    switch (arr->dtype) {
        dtype_case(Bodo_CTypes::INT8);
        dtype_case(Bodo_CTypes::INT16);
        dtype_case(Bodo_CTypes::INT32);
        dtype_case(Bodo_CTypes::INT64);
        dtype_case(Bodo_CTypes::UINT8);
        dtype_case(Bodo_CTypes::UINT16);
        dtype_case(Bodo_CTypes::UINT32);
        dtype_case(Bodo_CTypes::UINT64);
        default: {
            throw std::runtime_error(
                "Invalid dtype type for int_to_decimal_array: " +
                GetDtype_as_string(arr->dtype));
        }
    }
#undef dtype_case
    return res_arr.release();
}

decimal_value int64_to_decimal(int64_t value) {
    arrow::Decimal128 dec(value);
    auto low_bits = dec.low_bits();
    auto high_bits = dec.high_bits();
    return {static_cast<int64_t>(low_bits), high_bits};
}

void unbox_decimal(PyObject* obj, uint8_t* data);

PyObject* box_decimal(decimal_value val, int8_t precision, int8_t scale) {
    // convert input to Arrow scalar
    arrow::Decimal128 arrow_decimal(val.high, val.low);
    std::shared_ptr<arrow::Decimal128Scalar> scalar =
        std::make_shared<arrow::Decimal128Scalar>(
            arrow_decimal, arrow::decimal128(precision, scale));

    // convert Arrow C++ to PyArrow
    return arrow::py::wrap_scalar(scalar);
}

void decimal_to_str(decimal_value val, NRT_MemInfo** meminfo_ptr,
                    int64_t* len_ptr, int scale);

arrow::Decimal128 str_to_decimal(const std::string& str_val);

/**
 * @brief Get the Arrow function name for comparison operator
 * See https://arrow.apache.org/docs/cpp/compute.html#comparisons
 *
 * @param op_enum operator enum value
 * @return std::string Arrow function name
 */
std::string get_arrow_function_name(int32_t op_enum) {
    switch (op_enum) {
        case CmpOp::LT:
            return "less";
        case CmpOp::LE:
            return "less_equal";
        case CmpOp::EQ:
            return "equal";
        case CmpOp::NE:
            return "not_equal";
        case CmpOp::GT:
            return "greater";
        case CmpOp::GE:
            return "greater_equal";
        default:
            throw std::runtime_error(
                fmt::format("get_arrow_function_name: invalid op {}", op_enum));
    }
}

/**
 * @brief Call Arrow compute function for comparison operator
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0 first argument for comparison
 * @param arg1 second argument for comparison
 * @param is_scalar_arg0 first argument is scalar (passed as a one element
 * array)
 * @param is_scalar_arg1 second argument is scalar (passed as a one element
 * array)
 * @return array_info* output of comparison which is a boolean array
 */
array_info* arrow_compute_cmp_py_entry(int32_t op_enum, array_info* arg0,
                                       array_info* arg1, bool is_scalar_arg0,
                                       bool is_scalar_arg1) {
    try {
        // convert inputs to Arrow and handle scalars
        std::shared_ptr<arrow::Array> arr0 =
            to_arrow(std::shared_ptr<array_info>(arg0));
        std::shared_ptr<arrow::Array> arr1 =
            to_arrow(std::shared_ptr<array_info>(arg1));

        arrow::Datum arg0_datum(arr0);
        arrow::Datum arg1_datum(arr1);

        // convert scalar arguments passed as one element arrays back to scalar
        if (is_scalar_arg0) {
            arrow::Result<std::shared_ptr<arrow::Scalar>> getitem_res =
                arr0->GetScalar(0);
            if (!getitem_res.ok()) [[unlikely]] {
                throw std::runtime_error(
                    fmt::format("arrow_compute_cmp_py_entry: Error in Arrow "
                                "getitem for first argument: {}",
                                getitem_res.status().message()));
            }
            arg0_datum = getitem_res.ValueOrDie();
        }

        if (is_scalar_arg1) {
            arrow::Result<std::shared_ptr<arrow::Scalar>> getitem_res =
                arr1->GetScalar(0);
            if (!getitem_res.ok()) [[unlikely]] {
                throw std::runtime_error(
                    fmt::format("arrow_compute_cmp_py_entry: Error in Arrow "
                                "getitem for second argument: {}",
                                getitem_res.status().message()));
            }
            arg1_datum = getitem_res.ValueOrDie();
        }

        // call Arrow compute function
        arrow::Result<arrow::Datum> cmp_res = arrow::compute::CallFunction(
            get_arrow_function_name(op_enum), {arg0_datum, arg1_datum});
        if (!cmp_res.ok()) [[unlikely]] {
            throw std::runtime_error(fmt::format(
                "arrow_compute_cmp_py_entry: Error in Arrow compute: {}",
                cmp_res.status().message()));
        }
        std::shared_ptr<arrow::Array> out_arrow_arr =
            cmp_res.ValueOrDie().make_array();

        // convert Arrow output to Bodo array
        std::shared_ptr<array_info> out_arr =
            arrow_array_to_bodo(out_arrow_arr);
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief call Arrow to compare scalars
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0_scalar first argument of comparison
 * @param arg1_scalar second argument of comparison
 * @return bool output of comparison
 */
template <class T0, class T1>
inline bool arrow_compute_cmp_scalar(int32_t op_enum, T0 arg0_scalar,
                                     T1 arg1_scalar) {
    // call Arrow compute function
    arrow::Result<arrow::Datum> cmp_res = arrow::compute::CallFunction(
        get_arrow_function_name(op_enum), {arg0_scalar, arg1_scalar});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            fmt::format("arrow_compute_cmp_scalar: Error in "
                        "Arrow compute: {}",
                        cmp_res.status().message()));
    }

    // return bool output
    std::shared_ptr<arrow::BooleanScalar> out =
        std::static_pointer_cast<arrow::BooleanScalar>(
            cmp_res.ValueOrDie().scalar());
    return out->value;
}

/**
 * @brief compare decimal scalar to integer scalar
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0 first argument of comparison (decimal)
 * @param precision decimal argument's precision
 * @param scale decimal argument's scale
 * @param arg1 second argument (int)
 * @return bool output of comparison
 */
bool arrow_compute_cmp_decimal_int_py_entry(int32_t op_enum, decimal_value arg0,
                                            int32_t precision, int32_t scale,
                                            int64_t arg1) {
    try {
        // convert input to Arrow scalars
        arrow::Decimal128 arrow_decimal(arg0.high, arg0.low);
        arrow::Decimal128Scalar arg0_scalar(
            arrow_decimal, arrow::decimal128(precision, scale));
        arrow::Int64Scalar arg1_scalar(arg1);
        return arrow_compute_cmp_scalar(op_enum, arg0_scalar, arg1_scalar);

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
    }
}

/**
 * @brief compare decimal scalar to float scalar
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0 first argument of comparison (decimal)
 * @param precision decimal argument's precision
 * @param scale decimal argument's scale
 * @param arg1 second argument (float)
 * @return bool output of comparison
 */
bool arrow_compute_cmp_decimal_float_py_entry(int32_t op_enum,
                                              decimal_value arg0,
                                              int32_t precision, int32_t scale,
                                              double arg1) {
    try {
        // convert input to Arrow scalars
        arrow::Decimal128 arrow_decimal(arg0.high, arg0.low);
        arrow::Decimal128Scalar arg0_scalar(
            arrow_decimal, arrow::decimal128(precision, scale));
        arrow::DoubleScalar arg1_scalar(arg1);
        return arrow_compute_cmp_scalar(op_enum, arg0_scalar, arg1_scalar);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
    }
}

/**
 * @brief compare decimal scalar to decimal scalar
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0 first argument of comparison (decimal)
 * @param precision0 first argument's precision
 * @param scale0 first argument's scale
 * @param arg1 second argument (decimal)
 * @param precision1 second argument's precision
 * @param scale1 second argument's scale
 * @return bool output of comparison
 */
bool arrow_compute_cmp_decimal_decimal_py_entry(
    int32_t op_enum, decimal_value arg0, int32_t precision0, int32_t scale0,
    int32_t precision1, int32_t scale1, decimal_value arg1) {
    try {
        // convert input to Arrow scalars
        arrow::Decimal128 arrow_decimal0(arg0.high, arg0.low);
        arrow::Decimal128Scalar arg0_scalar(
            arrow_decimal0, arrow::decimal128(precision0, scale0));
        arrow::Decimal128 arrow_decimal1(arg1.high, arg1.low);
        arrow::Decimal128Scalar arg1_scalar(
            arrow_decimal1, arrow::decimal128(precision1, scale1));
        return arrow_compute_cmp_scalar(op_enum, arg0_scalar, arg1_scalar);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
    }
}

PyMODINIT_FUNC PyInit_decimal_ext(void) {
    PyObject* m;
    MOD_DEF(m, "decimal_ext", "No docs", NULL);
    if (m == NULL)
        return NULL;

    // init numpy
    import_array();

    bodo_common_init();

    // decimal_value should be exactly 128 bits to match Python
    if (sizeof(decimal_value) != 16)
        std::cerr << "invalid decimal struct size" << std::endl;

    // These are all C functions, so they don't throw any exceptions.
    // We might still need to add better error handling in the future.
    SetAttrStringFromVoidPtr(m, unbox_decimal);
    SetAttrStringFromVoidPtr(m, box_decimal);

    SetAttrStringFromVoidPtr(m, decimal_to_str);
    SetAttrStringFromVoidPtr(m, str_to_decimal);
    SetAttrStringFromVoidPtr(m, decimal_to_double_py_entry);
    SetAttrStringFromVoidPtr(m, decimal_to_int64_py_entry);
    SetAttrStringFromVoidPtr(m, int_to_decimal_array);

    SetAttrStringFromVoidPtr(m, cast_float_to_decimal_array_py_entry);
    SetAttrStringFromVoidPtr(m, cast_float_to_decimal_scalar_py_entry);
    SetAttrStringFromVoidPtr(m, arrow_compute_cmp_py_entry);
    SetAttrStringFromVoidPtr(m, arrow_compute_cmp_decimal_int_py_entry);
    SetAttrStringFromVoidPtr(m, arrow_compute_cmp_decimal_float_py_entry);
    SetAttrStringFromVoidPtr(m, arrow_compute_cmp_decimal_decimal_py_entry);
    SetAttrStringFromVoidPtr(m, cast_decimal_to_decimal_scalar_safe_py_entry);
    SetAttrStringFromVoidPtr(m, cast_decimal_to_decimal_scalar_unsafe_py_entry);
    SetAttrStringFromVoidPtr(m, cast_decimal_to_decimal_array_unsafe_py_entry);
    SetAttrStringFromVoidPtr(m, cast_decimal_to_decimal_array_safe_py_entry);
    SetAttrStringFromVoidPtr(m, multiply_decimal_scalars_py_entry);

    return m;
}

/**
 * @brief unbox a PyArrow Decimal128Scalar object into a native Decimal128Type
 *
 * @param pa_decimal_obj PyArrow Decimal128Scalar input
 * @param data pointer to 128-bit data for Decimal128Type
 */
void unbox_decimal(PyObject* pa_decimal_obj, uint8_t* data_ptr) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return;                        \
    }
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK(res.status().ok(), msg)             \
    lhs = std::move(res).ValueOrDie();

    auto gilstate = PyGILState_Ensure();

    // unwrap pyarrow Decimal128Scalar to C++
    std::shared_ptr<arrow::Scalar> scalar;
    arrow::Result<std::shared_ptr<arrow::Scalar>> res =
        arrow::py::unwrap_scalar(pa_decimal_obj);
    CHECK_ARROW_AND_ASSIGN(res, "unwrap_scalar", scalar);

    arrow::Decimal128 decimal_scalar =
        std::static_pointer_cast<arrow::Decimal128Scalar>(scalar)->value;
    decimal_scalar.ToBytes(data_ptr);

    PyGILState_Release(gilstate);
}

/**
 * @brief convert decimal128 value to string and create a memptr pointer with
 * the data
 *
 * @param val decimal128 input value
 * @param meminfo_ptr output memptr pointer to set
 * @param len_ptr output length to set
 * @param scale scale parameter of decimal128
 */
void decimal_to_str(decimal_value val, NRT_MemInfo** meminfo_ptr,
                    int64_t* len_ptr, int scale) {
    // Creating the arrow_decimal value
    arrow::Decimal128 arrow_decimal(val.high, val.low);
    // Getting the string
    std::string str = decimal_to_std_string(arrow_decimal, scale);
    // Now doing the boxing.
    int64_t l = (int64_t)str.length();
    NRT_MemInfo* meminfo = alloc_meminfo(l + 1);
    memcpy(meminfo->data, str.c_str(), l);
    ((char*)meminfo->data)[l] = 0;
    *len_ptr = l;
    *meminfo_ptr = meminfo;
}

arrow::Decimal128 str_to_decimal(const std::string& str_val) {
    /*
        str_to_decimal ignores a scale and precision value from
        Python code because the user cannot specify them directly. Precision
        currently matches the arrow default value of 38 and scale is set in
        the global variable. If this should ever change this function will need
        to be adjusted.
    */
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return -1;                     \
    }
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK(res.status().ok(), msg)             \
    lhs = std::move(res).ValueOrDie();

    auto gilstate = PyGILState_Ensure();
    // Create the decimal array value from a string, precision, and scale
    arrow::Decimal128 decimal;
    int32_t precision;
    int32_t scale;
    arrow::Status status =
        arrow::Decimal128::FromString(str_val, &decimal, &precision, &scale);
    CHECK(status.ok(), "decimal rescale failed");
    // rescale decimal to 18 (default scale converting from Python)
    CHECK_ARROW_AND_ASSIGN(decimal.Rescale(scale, PY_DECIMAL_SCALE),
                           "decimal rescale error", decimal);
    return decimal;
#undef CHECK
}
