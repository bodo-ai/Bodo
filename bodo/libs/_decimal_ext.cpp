// C/C++ code for DecimalArray handling
#include <Python.h>
#include <arrow/util/basic_decimal.h>
#include <iostream>

#include <arrow/compute/cast.h>
#include <arrow/python/pyarrow.h>
#include <arrow/util/bit_util.h>
#include <arrow/util/decimal.h>
#include <fmt/format.h>
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_distributed.h"
#include "groupby/_groupby_common.h"
#include "groupby/_groupby_do_apply_to_column.h"
#include "groupby/_groupby_ftypes.h"
#include "vendored/_gandiva_decimal_copy.h"

#pragma pack(1)
struct decimal_value {
    int64_t low;
    int64_t high;
};

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

/**
 * Converts a Decimal128 value to a non-scientific notation string.
 * This implementation is needed as Arrow's Decimal128::ToString()
 * forces scientific notation on certain scales.
 * Implementation is based on Arrow's Decimal128::ToString() implementation:
 * https://github.com/apache/arrow/blob/main/cpp/src/arrow/util/decimal.cc
 */
std::string decimal_to_string_no_scientific(
    arrow::Decimal128 const& arrow_decimal, int32_t scale) {
    std::string str = arrow_decimal.ToIntegerString();
    if (scale == 0) {
        return str;
    }
    const bool is_negative = str.front() == '-';
    const auto is_negative_offset = static_cast<int32_t>(is_negative);
    const auto len = static_cast<int32_t>(str.size());
    const int32_t num_digits = len - is_negative_offset;

    if (num_digits > scale) {
        const auto n = static_cast<size_t>(len - scale);
        str.insert(str.begin() + n, '.');
        return str;
    }
    // Add leading zeroes if num_digits <= scale.
    str.insert(is_negative_offset, scale - num_digits + 2, '0');
    str.at(is_negative_offset + 1) = '.';
    return str;
}

/**
 * @brief Convert a Decimal128 value to a string.
 * Note this never returns scientific notation. Use arrow's built in
 * Decimal128::ToString() for scientific notation.
 *
 * @param remove_trailing_zeroes If true, remove trailing zeroes.
 * @param arrow_decimal The Decimal128 value to convert.
 * @param scale The scale of the Decimal128 value.
 * @return std::string The string representation of the Decimal128 value.
 */
template <bool remove_trailing_zeroes = true>
std::string decimal_to_std_string(arrow::Decimal128 const& arrow_decimal,
                                  int const& scale) {
    std::string str = decimal_to_string_no_scientific(arrow_decimal, scale);
    if constexpr (remove_trailing_zeroes) {
        if (str.find('.') == std::string::npos) {
            return str;
        }

        // We want to remove trailing zeroes after the decimal point.
        size_t last_char = str.length();
        while (true) {
            if (str[last_char - 1] != '0') {
                break;
            }
            last_char--;
        }
        // position reduce str to 0.45  or 4.
        if (str[last_char - 1] == '.') {
            last_char--;
        }

        // Slice String to New Range
        str = str.substr(0, last_char);
    }

    return str;
}

std::string int128_decimal_to_std_string(__int128_t const& val,
                                         int const& scale) {
    arrow::Decimal128 arrow_decimal((int64_t)(val >> 64), (int64_t)(val));
    return decimal_to_std_string(arrow_decimal, scale);
}

double decimal_to_double(__int128_t const& val, uint8_t scale) {
    // TODO: Zero-copy (cast __int128_t to int64[2] for Decimal128 constructor)
    // Can't figure out how to do this in C++
    arrow::Decimal128 dec((int64_t)(val >> 64), (int64_t)(val));
    return dec.ToDouble(scale);
}

arrow::Decimal256 shift_decimal_scalar(arrow::Decimal256 val,
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

void cast_decimal_to_decimal_scalar_unsafe_py_entry(uint64_t in_low,
                                                    int64_t in_high,
                                                    int64_t shift_amount,
                                                    uint64_t* out_low_ptr,
                                                    int64_t* out_high_ptr) {
    try {
        arrow::Decimal128 val = arrow::Decimal128(in_high, in_low);
        arrow::Decimal128 res = shift_decimal_scalar(val, shift_amount);
        *out_low_ptr = res.low_bits();
        *out_high_ptr = res.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

void cast_decimal_to_decimal_scalar_safe_py_entry(
    uint64_t in_low, int64_t in_high, int64_t shift_amount,
    int64_t max_power_of_ten, bool* safe, uint64_t* out_low_ptr,
    int64_t* out_high_ptr) {
    try {
        arrow::Decimal128 val = arrow::Decimal128(in_high, in_low);
        bool safe_cast = val.FitsInPrecision(max_power_of_ten);
        *safe = safe_cast;
        if (!safe_cast) {
            *out_low_ptr = in_low;
            *out_high_ptr = in_high;
        } else {
            arrow::Decimal128 res = shift_decimal_scalar(val, shift_amount);
            *out_low_ptr = res.low_bits();
            *out_high_ptr = res.high_bits();
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
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

double decimal_to_double_py_entry(uint64_t low, uint64_t high, uint8_t scale) {
    return decimal_to_double((static_cast<__int128_t>(high) << 64) | low,
                             scale);
}

/**
 * @brief Kernel for converting a decimal array to a double (FLOAT64) array.
 *
 * @param arr The input decimal array
 * @param scale Scale of the input decimal array
 * @return array_info pointer of type FLOAT64
 */
std::unique_ptr<array_info> decimal_arr_to_double(
    const std::unique_ptr<array_info>& arr, uint8_t scale) {
    size_t len = arr->length;
    std::unique_ptr<array_info> out_arr =
        alloc_nullable_array_no_nulls(len, Bodo_CTypes::FLOAT64);
    try {
        for (size_t i = 0; i < len; i++) {
            if (!arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
            } else {
                double* out_ptr =
                    out_arr
                        ->data1<bodo_array_type::NULLABLE_INT_BOOL, double>() +
                    i;
                const arrow::Decimal128& in_val =
                    *(arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                 arrow::Decimal128>() +
                      i);
                *out_ptr = in_val.ToDouble(scale);
            }
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return out_arr;
}

/**
 * @brief Python entry point for converting a decimal array to a double array.
 *
 * @param arr_ The input decimal array as an array_info struct.
 * @return array_info* of type FLOAT64.
 */
array_info* decimal_arr_to_double_py_entry(array_info* arr_) {
    try {
        std::unique_ptr<array_info> arr = std::unique_ptr<array_info>(arr_);
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr->dtype == Bodo_CTypes::DECIMAL);
        uint8_t scale = arr->scale;
        std::unique_ptr<array_info> out_arr;
        out_arr = decimal_arr_to_double(arr, scale);
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

array_info* decimal_array_sign_py_entry(array_info* arr_) {
    try {
        std::unique_ptr<array_info> arr = std::unique_ptr<array_info>(arr_);
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr->dtype == Bodo_CTypes::DECIMAL);
        size_t len = arr->length;
        std::unique_ptr<array_info> out_arr =
            alloc_nullable_array_no_nulls(len, Bodo_CTypes::INT8);
        for (size_t i = 0; i < len; i++) {
            if (!arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
            } else {
                int8_t* out_ptr =
                    out_arr
                        ->data1<bodo_array_type::NULLABLE_INT_BOOL, int8_t>() +
                    i;
                const arrow::Decimal128& in_val =
                    *(arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                 arrow::Decimal128>() +
                      i);
                *out_ptr = in_val == arrow::Decimal128(0) ? 0 : in_val.Sign();
            }
        }
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

int8_t decimal_scalar_sign_py_entry(uint64_t in_low, int64_t in_high) {
    arrow::Decimal128 val = arrow::Decimal128(in_high, in_low);
    return val == arrow::Decimal128(0) ? 0 : val.Sign();
}

/**
 * @brief Compute the sum of a decimal array.
 *
 * @param arr The input decimal array.
 * @param[out] is_null The pointer used to indicate whether the final answer is
 * null.
 * @param[in] parallel Do we need a parallel merge step.
 * @param[out] out_low_ptr Pointer to the low 64 bits of the result.
 * @param[out] out_high_ptr Pointer to the high 64 bits of the result.
 */
void sum_decimal_array_py_entry(array_info* arr_raw, bool* is_null,
                                bool parallel, uint64_t* out_low_ptr,
                                int64_t* out_high_ptr) noexcept {
    try {
        std::shared_ptr<array_info> arr = std::shared_ptr<array_info>(arr_raw);

        // Invoke the groupby codepath for the aggfunc, with every
        // row mapping to "group 0"
        grouping_info dummy_local_grp_info;
        dummy_local_grp_info.num_groups = 1;
        dummy_local_grp_info.row_to_group.resize(arr->length, 0);
        std::vector<std::shared_ptr<array_info>> dummy_aux_cols;

        std::shared_ptr<array_info> out_arr = alloc_array_top_level(
            1, -1, -1, bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::DECIMAL,
            -1, 0, 0, false, false, false);
        aggfunc_output_initialize(out_arr, Bodo_FTypes::sum, true);
        do_apply_to_column(arr, out_arr, dummy_aux_cols, dummy_local_grp_info,
                           Bodo_FTypes::sum);

        // If the code running in parallel, do a reduction on the local sum
        // to obtain the global sum
        if (parallel) {
            int myrank, num_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            std::shared_ptr<array_info> combined_arr =
                gather_array(out_arr, true, true, 0, num_ranks, myrank);
            // Repeat the pre-parallel procedure on the combined
            // results.
            grouping_info dummy_combine_grp_info;
            dummy_combine_grp_info.num_groups = 1;
            dummy_combine_grp_info.row_to_group.resize(num_ranks, 0);
            std::vector<std::shared_ptr<array_info>> dummy_aux_cols;
            aggfunc_output_initialize(out_arr, Bodo_FTypes::sum, true);
            do_apply_to_column(combined_arr, out_arr, dummy_aux_cols,
                               dummy_combine_grp_info, Bodo_FTypes::sum);
        }

        // Set the is_null pointer so the calling sight knows if the sum was
        // NULL, then return the decimal value stored in the singleton output
        // array.
        *is_null =
            !out_arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0);
        arrow::Decimal128 result =
            out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                           arrow::Decimal128>()[0];
        *out_low_ptr = result.low_bits();
        *out_high_ptr = result.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

template <bool fast_add, bool rescale_left, bool rescale_right,
          bool do_addition>
inline void add_or_subtract_decimal_scalars(const arrow::Decimal128& d1,
                                            const arrow::Decimal128& d2,
                                            int64_t s1, int64_t s2,
                                            int64_t out_scale, bool* overflow,
                                            arrow::Decimal128* result) {
    if constexpr (fast_add) {
        arrow::Decimal128 lhs = d1;
        arrow::Decimal128 rhs = d2;
        // Ensure the two decimals have the same scale
        if constexpr (rescale_left) {
            lhs = d1.Rescale(s1, out_scale).ValueOrDie();
        }
        if constexpr (rescale_right) {
            rhs = d2.Rescale(s2, out_scale).ValueOrDie();
        }
        if constexpr (do_addition) {
            *result = lhs + rhs;
        } else {
            *result = lhs - rhs;
        }
    } else {
        // Ensure the two decimals have the same scale (but are first upcasted
        // to 256)
        auto lhs = decimalops::ConvertToInt256(d1);
        auto rhs = decimalops::ConvertToInt256(d2);
        if constexpr (rescale_left) {
            lhs = decimalops::IncreaseScaleBy(lhs, out_scale - s1);
        }
        if constexpr (rescale_right) {
            rhs = decimalops::IncreaseScaleBy(rhs, out_scale - s2);
        }
        boost::multiprecision::int256_t combined_decimal;
        if constexpr (do_addition) {
            combined_decimal = lhs + rhs;
        } else {
            combined_decimal = lhs - rhs;
        }
        *result = decimalops::ConvertToDecimal128(combined_decimal, overflow);
    }
}

/**
 * @brief Add or subtract two decimal scalars with the given precision and scale
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
 * @param do_addition True if we are adding the two decimals, false if we are
 *                    subtracting them.
 * @param[out] overflow Overflow flag
 * @return arrow::Decimal128
 */
arrow::Decimal128 add_or_subtract_decimal_scalars_util(
    arrow::Decimal128 v1, int64_t p1, int64_t s1, arrow::Decimal128 v2,
    int64_t p2, int64_t s2, int64_t out_precision, int64_t out_scale,
    bool do_addition, bool* overflow) {
    bool fast_add = out_precision < decimalops::kMaxPrecision;
    arrow::Decimal128 result;
    if (do_addition) {
        if (fast_add) {
            if (s1 < s2) {
                add_or_subtract_decimal_scalars<true, true, false, true>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            } else if (s2 < s1) {
                add_or_subtract_decimal_scalars<true, false, true, true>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            } else {
                add_or_subtract_decimal_scalars<true, false, false, true>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            }
        } else {
            if (s1 < s2) {
                add_or_subtract_decimal_scalars<false, true, false, true>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            } else if (s2 < s1) {
                add_or_subtract_decimal_scalars<false, false, true, true>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            } else {
                add_or_subtract_decimal_scalars<false, false, false, true>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            }
        }
    } else {
        if (fast_add) {
            if (s1 < s2) {
                add_or_subtract_decimal_scalars<true, true, false, false>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            } else if (s2 < s1) {
                add_or_subtract_decimal_scalars<true, false, true, false>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            } else {
                add_or_subtract_decimal_scalars<true, false, false, false>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            }
        } else {
            if (s1 < s2) {
                add_or_subtract_decimal_scalars<false, true, false, false>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            } else if (s2 < s1) {
                add_or_subtract_decimal_scalars<false, false, true, false>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            } else {
                add_or_subtract_decimal_scalars<false, false, false, false>(
                    v1, v2, s1, s2, out_scale, overflow, &result);
            }
        }
    }

    return result;
}

/**
 * @brief Python entrypoint for addition and subtraction of decimal scalars.
 */
void add_or_subtract_decimal_scalars_py_entry(
    uint64_t v1_low, int64_t v1_high, int64_t p1, int64_t s1, uint64_t v2_low,
    int64_t v2_high, int64_t p2, int64_t s2, int64_t out_precision,
    int64_t out_scale, uint64_t* out_low_ptr, uint64_t* out_high_ptr,
    bool do_addition, bool* overflow) noexcept {
    try {
        arrow::Decimal128 v1(v1_high, v1_low);
        arrow::Decimal128 v2(v2_high, v2_low);
        arrow::Decimal128 res = add_or_subtract_decimal_scalars_util(
            v1, p1, s1, v2, p2, s2, out_precision, out_scale, do_addition,
            overflow);
        *out_low_ptr = res.low_bits();
        *out_high_ptr = res.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Helper for add_or_subtract_decimal_arrays that deals with the
 * templated loop body.
 */
template <bool is_scalar_1, bool is_scalar_2, bool fast_add, bool rescale_left,
          bool rescale_right, bool do_addition>
inline void add_or_subtract_decimal_arrays_loop_body(
    const std::unique_ptr<array_info>& arr1,
    const std::unique_ptr<array_info>& arr2,
    std::unique_ptr<array_info>& out_arr, int64_t s1, int64_t s2,
    int64_t out_scale, bool& out_overflow, size_t i) {
    bool is_null_1, is_null_2;
    if constexpr (is_scalar_1) {
        is_null_1 = false;
    } else {
        is_null_1 = !arr1->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i);
    }
    if constexpr (is_scalar_2) {
        is_null_2 = false;
    } else {
        is_null_2 = !arr2->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i);
    }
    if (is_null_1 || is_null_2) {
        out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
    } else {
        arrow::Decimal128* out_ptr =
            out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                           arrow::Decimal128>() +
            i;
        bool overflow_i = false;
        size_t idx1, idx2;
        if constexpr (is_scalar_1) {
            idx1 = 0;
        } else {
            idx1 = i;
        }
        if constexpr (is_scalar_2) {
            idx2 = 0;
        } else {
            idx2 = i;
        }
        const arrow::Decimal128& d1 =
            *(arr1->data1<bodo_array_type::NULLABLE_INT_BOOL,
                          arrow::Decimal128>() +
              idx1);
        const arrow::Decimal128& d2 =
            *(arr2->data1<bodo_array_type::NULLABLE_INT_BOOL,
                          arrow::Decimal128>() +
              idx2);
        if constexpr (!do_addition) {
            add_or_subtract_decimal_scalars<fast_add, rescale_left,
                                            rescale_right, do_addition>(
                d1, d2, s1, s2, out_scale, &overflow_i, out_ptr);
        } else {
            add_or_subtract_decimal_scalars<fast_add, rescale_left,
                                            rescale_right, do_addition>(
                d1, d2, s1, s2, out_scale, &overflow_i, out_ptr);
        }

        out_overflow |= overflow_i;
    }
}

/**
 * @brief Add two decimal arrays element-wise and return an output array
 * with the specified precision and scale. If overflow is detected during any
 * addition, then the 'overflow' flag is updated to true.
 *
 * @param arr1 First nullable decimal array.
 * @param arr2 Second nullable decimal array.
 * @param out_precision Output precision
 * @param out_scale Output scale
 * @param[out] overflow Overflow flag.
 * @return std::shared_ptr<array_info> Output nullable decimal array.
 */
template <bool is_scalar_1, bool is_scalar_2, bool do_addition>
std::unique_ptr<array_info> add_or_subtract_decimal_arrays(
    const std::unique_ptr<array_info>& arr1,
    const std::unique_ptr<array_info>& arr2, int64_t out_precision,
    int64_t out_scale, bool* const overflow) noexcept {
    int64_t s1 = arr1->scale;
    int64_t s2 = arr2->scale;
    // Allocate output array.
    size_t out_length = is_scalar_1 ? arr2->length : arr1->length;
    std::unique_ptr<array_info> out_arr =
        alloc_nullable_array_no_nulls(out_length, Bodo_CTypes::DECIMAL);
    try {
        out_arr->precision = out_precision;
        out_arr->scale = out_scale;
        bool fast_add = out_precision < decimalops::kMaxPrecision;
        bool rescale_left = s1 < s2;
        bool rescale_right = s2 < s1;
        bool out_overflow = false;
#ifndef DECIMAL_ARRAY_ADD
#define DECIMAL_ARRAY_ADD(FAST_ADD, RESCALE_LEFT, RESCALE_RIGHT)              \
    for (size_t i = 0; i < out_length; i++) {                                 \
        add_or_subtract_decimal_arrays_loop_body<is_scalar_1, is_scalar_2,    \
                                                 FAST_ADD, RESCALE_LEFT,      \
                                                 RESCALE_RIGHT, do_addition>( \
            arr1, arr2, out_arr, s1, s2, out_scale, out_overflow, i);         \
    }
#endif
        if (fast_add) {
            if (rescale_left) {
                DECIMAL_ARRAY_ADD(true, true, false)
            } else if (rescale_right) {
                DECIMAL_ARRAY_ADD(true, false, true)
            } else {
                DECIMAL_ARRAY_ADD(true, false, false)
            }
        } else {
            if (rescale_left) {
                DECIMAL_ARRAY_ADD(false, true, false)
            } else if (rescale_right) {
                DECIMAL_ARRAY_ADD(false, false, true)
            } else {
                DECIMAL_ARRAY_ADD(false, false, false)
            }
        }
        *overflow = out_overflow;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return out_arr;
}

#undef DECIMAL_ARRAY_ADD

/**
 * @brief Wrapper for add_or_subtract_decimal_arrays that handles the templating
 * of the is_scalar arguments.
 */
std::unique_ptr<array_info> add_or_subtract_decimal_arrays_wrapper(

    const std::unique_ptr<array_info>& arr1,
    const std::unique_ptr<array_info>& arr2, bool is_scalar_1, bool is_scalar_2,
    int64_t out_precision, int64_t out_scale, bool do_addition,
    bool* const overflow) {
    if (do_addition) {
        if (is_scalar_1) {
            return add_or_subtract_decimal_arrays<true, false, true>(
                arr1, arr2, out_precision, out_scale, overflow);
        } else if (is_scalar_2) {
            return add_or_subtract_decimal_arrays<false, true, true>(
                arr1, arr2, out_precision, out_scale, overflow);
        } else {
            return add_or_subtract_decimal_arrays<false, false, true>(
                arr1, arr2, out_precision, out_scale, overflow);
        }
    } else {
        if (is_scalar_1) {
            return add_or_subtract_decimal_arrays<true, false, false>(
                arr1, arr2, out_precision, out_scale, overflow);
        } else if (is_scalar_2) {
            return add_or_subtract_decimal_arrays<false, true, false>(
                arr1, arr2, out_precision, out_scale, overflow);
        } else {
            return add_or_subtract_decimal_arrays<false, false, false>(
                arr1, arr2, out_precision, out_scale, overflow);
        }
    }
}

array_info* add_or_subtract_decimal_arrays_py_entry(
    array_info* arr1_, array_info* arr2_, bool is_scalar_1, bool is_scalar_2,
    int64_t out_precision, int64_t out_scale, bool do_addition,
    bool* overflow) noexcept {
    try {
        std::unique_ptr<array_info> arr1 = std::unique_ptr<array_info>(arr1_);
        std::unique_ptr<array_info> arr2 = std::unique_ptr<array_info>(arr2_);
        assert(arr1->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr2->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr1->dtype == Bodo_CTypes::DECIMAL &&
               arr2->dtype == Bodo_CTypes::DECIMAL &&
               arr1->length == arr2->length);
        std::unique_ptr<array_info> out_arr =
            add_or_subtract_decimal_arrays_wrapper(
                arr1, arr2, is_scalar_1, is_scalar_2, out_precision, out_scale,
                do_addition, overflow);
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
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
arrow::Decimal128 multiply_decimal_scalars_util(
    arrow::Decimal128 v1, int64_t p1, int64_t s1, arrow::Decimal128 v2,
    int64_t p2, int64_t s2, int64_t out_precision, int64_t out_scale,
    bool* overflow) {
    arrow::Decimal128 result;
    bool fast_multiply = out_precision < decimalops::kMaxPrecision;
    int32_t delta_scale = s1 + s2 - out_scale;
    bool rescale = delta_scale != 0;
    if (fast_multiply && rescale) {
        decimalops::Multiply<true, true>(v1, v2, out_scale, delta_scale,
                                         overflow, &result);
    } else if (fast_multiply) {
        decimalops::Multiply<true, false>(v1, v2, out_scale, delta_scale,
                                          overflow, &result);
    } else if (rescale) {
        decimalops::Multiply<false, true>(v1, v2, out_scale, delta_scale,
                                          overflow, &result);
    } else {
        decimalops::Multiply<false, false>(v1, v2, out_scale, delta_scale,
                                           overflow, &result);
    }
    return result;
}

/**
 * @brief Python entry point to multiply two decimal scalars.
 */
void multiply_decimal_scalars_py_entry(uint64_t v1_low, int64_t v1_high,
                                       int64_t p1, int64_t s1, uint64_t v2_low,
                                       int64_t v2_high, int64_t p2, int64_t s2,
                                       int64_t out_precision, int64_t out_scale,
                                       uint64_t* out_low_ptr,
                                       uint64_t* out_high_ptr,
                                       bool* overflow) noexcept {
    try {
        arrow::Decimal128 v1(v1_high, v1_low);
        arrow::Decimal128 v2(v2_high, v2_low);
        arrow::Decimal128 res = multiply_decimal_scalars_util(
            v1, p1, s1, v2, p2, s2, out_precision, out_scale, overflow);
        *out_low_ptr = res.low_bits();
        *out_high_ptr = res.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Multiply two decimal arrays element-wise and return an output array
 * with the specified precision and scale. If overflow is detected during any
 * multiplication, then the 'overflow' flag is updated to true.
 *
 * @param arr1 First nullable decimal array.
 * @param arr2 Second nullable decimal array.
 * @param out_precision Output precision
 * @param out_scale Output scale
 * @param is_scalar_arg1 first argument is scalar (passed as a one element
 * array)
 * @param is_scalar_arg2 second argument is scalar (passed as a one element
 * array)
 * @param[out] overflow Overflow flag.
 * @return std::shared_ptr<array_info> Output nullable decimal array.
 */
std::unique_ptr<array_info> multiply_decimal_arrays(
    const std::unique_ptr<array_info>& arr1,
    const std::unique_ptr<array_info>& arr2, int64_t out_precision,
    int64_t out_scale, bool is_scalar_arg1, bool is_scalar_arg2,
    bool* const overflow) noexcept {
    int64_t s1 = arr1->scale;
    int64_t s2 = arr2->scale;
    int32_t delta_scale = s1 + s2 - out_scale;

    size_t len = is_scalar_arg1 ? arr2->length : arr1->length;

    // Allocate output array.
    std::unique_ptr<array_info> out_arr =
        alloc_nullable_array_no_nulls(len, Bodo_CTypes::DECIMAL);
    out_arr->precision = out_precision;
    out_arr->scale = out_scale;

    bool out_overflow = false;

    bool fast_multiply = out_precision < decimalops::kMaxPrecision;
    bool rescale = delta_scale != 0;

#ifndef DECIMAL_ARRAY_MULTIPLY
#define DECIMAL_ARRAY_MULTIPLY(FAST_MULTIPLY, RESCALE)                        \
    for (size_t i = 0; i < len; i++) {                                        \
        bool v1_null =                                                        \
            !is_scalar_arg1 &&                                                \
            !arr1->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i);       \
        bool v2_null =                                                        \
            !is_scalar_arg2 &&                                                \
            !arr2->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i);       \
        if (v1_null || v2_null) {                                             \
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i,      \
                                                                      false); \
        } else {                                                              \
            arrow::Decimal128* out_ptr =                                      \
                out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,            \
                               arrow::Decimal128>() +                         \
                i;                                                            \
            size_t d1_ind = is_scalar_arg1 ? 0 : i;                           \
            const arrow::Decimal128& d1 =                                     \
                *(arr1->data1<bodo_array_type::NULLABLE_INT_BOOL,             \
                              arrow::Decimal128>() +                          \
                  d1_ind);                                                    \
            size_t d2_ind = is_scalar_arg2 ? 0 : i;                           \
            const arrow::Decimal128& d2 =                                     \
                *(arr2->data1<bodo_array_type::NULLABLE_INT_BOOL,             \
                              arrow::Decimal128>() +                          \
                  d2_ind);                                                    \
            bool overflow_i;                                                  \
            decimalops::Multiply<FAST_MULTIPLY, RESCALE>(                     \
                d1, d2, out_scale, delta_scale, &overflow_i, out_ptr);        \
            out_overflow |= overflow_i;                                       \
        }                                                                     \
    }
#endif
    if (fast_multiply && rescale) {
        DECIMAL_ARRAY_MULTIPLY(true, true)
    } else if (fast_multiply) {
        DECIMAL_ARRAY_MULTIPLY(true, false)
    } else if (rescale) {
        DECIMAL_ARRAY_MULTIPLY(false, true)
    } else {
        DECIMAL_ARRAY_MULTIPLY(false, false)
    }
    *overflow = out_overflow;
    return out_arr;
}

#undef DECIMAL_ARRAY_MULTIPLY

array_info* multiply_decimal_arrays_py_entry(
    array_info* arr1_, array_info* arr2_, int64_t out_precision,
    int64_t out_scale, bool is_scalar_arg1, bool is_scalar_arg2,
    bool* overflow) noexcept {
    try {
        std::unique_ptr<array_info> arr1 = std::unique_ptr<array_info>(arr1_);
        std::unique_ptr<array_info> arr2 = std::unique_ptr<array_info>(arr2_);
        assert(arr1->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr2->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr1->dtype == Bodo_CTypes::DECIMAL &&
               arr2->dtype == Bodo_CTypes::DECIMAL);
        std::unique_ptr<array_info> out_arr =
            multiply_decimal_arrays(arr1, arr2, out_precision, out_scale,
                                    is_scalar_arg1, is_scalar_arg2, overflow);
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * Performs the modulo operation on two decimal scalars. This is
 * a valid operation because of the arrow implementation of the
 * '%' operator on Decimal128 values, linked here (see BasicDecimal128
 * operator%):
 * https://github.com/apache/arrow/blob/main/cpp/src/arrow/util/basic_decimal.cc
 *
 * @tparam fast_mod: true if we can avoid safety checks because
 * neither argument has to be rescaled to an unsafe extent.
 * @tparam rescale_left: true if the lhs of the modulo operation
 * must be rescaled to match the rhs.
 * @tparam rescale_right: true if the rhs of the modulo operation
 * must be rescaled to match the lhs.
 * @param d1 a reference to the lhs of the modulo operation.
 * @param d2 a reference to the rhs of the modulo operation.
 * @param s1 the scale of decimal d1.
 * @param s2 the scale of decimal d1.
 * @param out_scale the intended scale of the output decimal.
 * @param result the pointer where the result decimal is stored.
 */
template <bool fast_mod, bool rescale_left, bool rescale_right>
inline void modulo_decimal_scalars(const arrow::Decimal128& d1,
                                   const arrow::Decimal128& d2, int64_t s1,
                                   int64_t s2, int64_t out_scale,
                                   arrow::Decimal128* result) {
    arrow::Decimal128 lhs = d1;
    arrow::Decimal128 rhs = d2;
    if constexpr (fast_mod) {
        // Ensure the two decimals have the same scale
        if constexpr (rescale_left) {
            lhs = d1.Rescale(s1, out_scale).ValueOrDie();
        }
        if constexpr (rescale_right) {
            rhs = d2.Rescale(s2, out_scale).ValueOrDie();
        }
        if (rhs == 0) {
            throw std::runtime_error("Invalid modulo by zero");
        }
        *result = lhs % rhs;
    } else {
        // Ensure the two decimals have the same scale, but checking for invalid
        // values
        auto lhs = decimalops::ConvertToInt256(d1);
        auto rhs = decimalops::ConvertToInt256(d2);
        bool out_of_bounds = false;
        if constexpr (rescale_left) {
            lhs = decimalops::IncreaseScaleBy(lhs, out_scale - s1);
            // Verify that the new Decimal256 is in the valid range for a
            // Decimal128
            decimalops::ConvertToDecimal128(lhs, &out_of_bounds);
        }
        if constexpr (rescale_right) {
            rhs = decimalops::IncreaseScaleBy(rhs, out_scale - s2);
            // Verify that the new Decimal256 is in the valid range for a
            // Decimal128
            decimalops::ConvertToDecimal128(rhs, &out_of_bounds);
        }
        if (rhs == 0) {
            throw std::runtime_error("Invalid modulo by zero");
        }
        boost::multiprecision::int256_t combined_decimal = lhs % rhs;
        *result =
            decimalops::ConvertToDecimal128(combined_decimal, &out_of_bounds);
        if (out_of_bounds) {
            throw std::runtime_error("Invalid rescale during decimal modulo");
        }
    }
}

/**
 * @brief Mod two decimal scalars with the given precision and scale
 * and return the output. The output should have its scale truncated to
 * the provided output scale. If overflow is detected, then the overflow
 * need to be updated to true.
 *
 * @param[in] v1_low The low 64 bits of the first Decimal128 argument.
 * @param[in] v1_high The high 64 bits of the first Decimal128 argument.
 * @param p1 Precision of first decimal value
 * @param s1 Scale of first decimal value
 * @param[in] v2_low The low 64 bits of the second Decimal128 argument.
 * @param[in] v2_high The high 64 bits of the second Decimal128 argument.
 * @param p2 Precision of second decimal value
 * @param s2 Scale of second decimal value
 * @param out_precision Output precision
 * @param out_scale Output scale
 * @param[out] out_low_ptr Pointer to the low 64 bits of the result.
 * @param[out] out_high_ptr Pointer to the high 64 bits of the result.
 */
void modulo_decimal_scalars_py_entry(uint64_t v1_low, int64_t v1_high,
                                     int64_t p1, int64_t s1, uint64_t v2_low,
                                     int64_t v2_high, int64_t p2, int64_t s2,
                                     int64_t out_precision, int64_t out_scale,
                                     uint64_t* out_low_ptr,
                                     uint64_t* out_high_ptr) noexcept {
    try {
        // We only need to do safe checks if there is a chance that either side
        // could be rescaled into an invalid value.
        bool fast_mod = out_precision < decimalops::kMaxPrecision ||
                        (s1 == out_scale && s2 == out_scale);
        arrow::Decimal128 v1(v1_high, v1_low);
        arrow::Decimal128 v2(v2_high, v2_low);
        arrow::Decimal128 result;
        if (fast_mod) {
            if (s1 < s2) {
                modulo_decimal_scalars<true, true, false>(v1, v2, s1, s2,
                                                          out_scale, &result);
            } else if (s2 < s1) {
                modulo_decimal_scalars<true, false, true>(v1, v2, s1, s2,
                                                          out_scale, &result);
            } else {
                modulo_decimal_scalars<true, false, false>(v1, v2, s1, s2,
                                                           out_scale, &result);
            }
        } else {
            if (s1 < s2) {
                modulo_decimal_scalars<false, true, false>(v1, v2, s1, s2,
                                                           out_scale, &result);
            } else if (s2 < s1) {
                modulo_decimal_scalars<false, false, true>(v1, v2, s1, s2,
                                                           out_scale, &result);
            } else {
                modulo_decimal_scalars<false, false, false>(v1, v2, s1, s2,
                                                            out_scale, &result);
            }
        }
        *out_low_ptr = result.low_bits();
        *out_high_ptr = result.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Helper for modulo_decimal_arrays that deals with the
 * templated loop body.
 */
template <bool is_scalar_1, bool is_scalar_2, bool fast_mod, bool rescale_left,
          bool rescale_right>
inline void modulo_decimal_arrays_loop_body(
    const std::unique_ptr<array_info>& arr1,
    const std::unique_ptr<array_info>& arr2,
    std::unique_ptr<array_info>& out_arr, int64_t s1, int64_t s2,
    int64_t out_scale, size_t i) {
    bool is_null_1, is_null_2;
    if constexpr (is_scalar_1) {
        is_null_1 = false;
    } else {
        is_null_1 = !arr1->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i);
    }
    if constexpr (is_scalar_2) {
        is_null_2 = false;
    } else {
        is_null_2 = !arr2->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i);
    }
    if (is_null_1 || is_null_2) {
        out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
    } else {
        arrow::Decimal128* out_ptr =
            out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                           arrow::Decimal128>() +
            i;
        size_t idx1, idx2;
        if constexpr (is_scalar_1) {
            idx1 = 0;
        } else {
            idx1 = i;
        }
        if constexpr (is_scalar_2) {
            idx2 = 0;
        } else {
            idx2 = i;
        }
        const arrow::Decimal128& d1 =
            *(arr1->data1<bodo_array_type::NULLABLE_INT_BOOL,
                          arrow::Decimal128>() +
              idx1);
        const arrow::Decimal128& d2 =
            *(arr2->data1<bodo_array_type::NULLABLE_INT_BOOL,
                          arrow::Decimal128>() +
              idx2);
        modulo_decimal_scalars<fast_mod, rescale_left, rescale_right>(
            d1, d2, s1, s2, out_scale, out_ptr);
    }
}

/**
 * @brief Mod two decimal arrays element-wise and return an output array
 * with the specified precision and scale. If overflow is detected during any
 * modulo, then the 'overflow' flag is updated to true.
 *
 * @param arr1 First nullable decimal array.
 * @param arr2 Second nullable decimal array.
 * @param out_precision Output precision
 * @param out_scale Output scale
 * @param[out] overflow Overflow flag.
 * @return std::shared_ptr<array_info> Output nullable decimal array.
 */
template <bool is_scalar_1, bool is_scalar_2>
std::unique_ptr<array_info> modulo_decimal_arrays(
    const std::unique_ptr<array_info>& arr1,
    const std::unique_ptr<array_info>& arr2, int64_t out_precision,
    int64_t out_scale) noexcept {
    int64_t s1 = arr1->scale;
    int64_t s2 = arr2->scale;
    // Allocate output array.
    size_t out_length = is_scalar_1 ? arr2->length : arr1->length;
    std::unique_ptr<array_info> out_arr =
        alloc_nullable_array_no_nulls(out_length, Bodo_CTypes::DECIMAL);
    try {
        out_arr->precision = out_precision;
        out_arr->scale = out_scale;
        // We only need to do safe checks if there is a chance that either side
        // could be rescaled into an invalid value.
        bool fast_mod = out_precision < decimalops::kMaxPrecision ||
                        (s1 == out_scale && s2 == out_scale);
        bool rescale_left = s1 < s2;
        bool rescale_right = s2 < s1;
#ifndef DECIMAL_ARRAY_MOD
#define DECIMAL_ARRAY_MOD(FAST_MOD, RESCALE_LEFT, RESCALE_RIGHT)            \
    for (size_t i = 0; i < out_length; i++) {                               \
        modulo_decimal_arrays_loop_body<is_scalar_1, is_scalar_2, FAST_MOD, \
                                        RESCALE_LEFT, RESCALE_RIGHT>(       \
            arr1, arr2, out_arr, s1, s2, out_scale, i);                     \
    }
#endif
        if (fast_mod) {
            if (rescale_left) {
                DECIMAL_ARRAY_MOD(true, true, false)
            } else if (rescale_right) {
                DECIMAL_ARRAY_MOD(true, false, true)
            } else {
                DECIMAL_ARRAY_MOD(true, false, false)
            }
        } else {
            if (rescale_left) {
                DECIMAL_ARRAY_MOD(false, true, false)
            } else if (rescale_right) {
                DECIMAL_ARRAY_MOD(false, false, true)
            } else {
                DECIMAL_ARRAY_MOD(false, false, false)
            }
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return out_arr;
}

#undef DECIMAL_ARRAY_ADD

array_info* modulo_decimal_arrays_py_entry(array_info* arr1_, array_info* arr2_,
                                           bool is_scalar_1, bool is_scalar_2,
                                           int64_t out_precision,
                                           int64_t out_scale) noexcept {
    try {
        std::unique_ptr<array_info> arr1 = std::unique_ptr<array_info>(arr1_);
        std::unique_ptr<array_info> arr2 = std::unique_ptr<array_info>(arr2_);
        assert(arr1->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr2->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr1->dtype == Bodo_CTypes::DECIMAL &&
               arr2->dtype == Bodo_CTypes::DECIMAL &&
               arr1->length == arr2->length);
        std::unique_ptr<array_info> out_arr;
        if (is_scalar_1) {
            out_arr = modulo_decimal_arrays<true, false>(
                arr1, arr2, out_precision, out_scale);
        } else if (is_scalar_2) {
            out_arr = modulo_decimal_arrays<false, true>(
                arr1, arr2, out_precision, out_scale);
        } else {
            out_arr = modulo_decimal_arrays<false, false>(
                arr1, arr2, out_precision, out_scale);
        }
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Divide two decimal scalars with the given precision and scale
 * and return the output. The output should have its scale truncated to
 * the provided output scale. If overflow is detected, then the overflow
 * need to be updated to true.
 *
 * @param[in] v1_low The low 64 bits of the first Decimal128 argument.
 * @param[in] v1_high The high 64 bits of the first Decimal128 argument.
 * @param p1 Precision of first decimal value
 * @param s1 Scale of first decimal value
 * @param[in] v2_low The low 64 bits of the second Decimal128 argument.
 * @param[in] v2_high The high 64 bits of the second Decimal128 argument.
 * @param p2 Precision of second decimal value
 * @param s2 Scale of second decimal value
 * @param out_precision Output precision
 * @param out_scale Output scale
 * @param[out] out_low_ptr Pointer to the low 64 bits of the result.
 * @param[out] out_high_ptr Pointer to the high 64 bits of the result.
 * @param[out] overflow Overflow flag
 * @param do_div0 If true, return 0 if v2 is 0, otherwise return v1/v2
 */
void divide_decimal_scalars_py_entry(uint64_t v1_low, int64_t v1_high,
                                     int64_t p1, int64_t s1, uint64_t v2_low,
                                     int64_t v2_high, int64_t p2, int64_t s2,
                                     int64_t out_precision, int64_t out_scale,
                                     uint64_t* out_low_ptr,
                                     uint64_t* out_high_ptr, bool* overflow,
                                     bool do_div0 = false) noexcept {
    try {
        arrow::Decimal128 v1(v1_high, v1_low);
        arrow::Decimal128 v2(v2_high, v2_low);
        if (do_div0 && v2 == 0) {
            *out_low_ptr = 0;
            *out_high_ptr = 0;
            return;
        }
        int32_t delta_scale = out_scale + s2 - s1;
        arrow::Decimal128 res =
            decimalops::Divide(v1, v2, delta_scale, overflow);
        *out_low_ptr = res.low_bits();
        *out_high_ptr = res.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Divide two decimal arrays or a decimal array and a scalar element-wise
 * and return an output array with the specified precision and scale. If
 * overflow is detected during any division, then the 'overflow' flag is updated
 * to true.
 *
 * @param arr1 First nullable decimal array.
 * @param arr2 Second nullable decimal array.
 * @param out_precision Output precision
 * @param out_scale Output scale
 * @param is_scalar_arg1 first argument is scalar (passed as a one element
 * array)
 * @param is_scalar_arg2 second argument is scalar (passed as a one element
 * array)
 * @param[out] overflow Overflow flag.
 * @param do_div0 If true, return 0 if v2 is 0, otherwise return v1/v2
 * @return std::shared_ptr<array_info> Output nullable decimal array.
 */
template <bool do_div0 = false>
std::unique_ptr<array_info> divide_decimal_arrays(
    const std::unique_ptr<array_info>& arr1,
    const std::unique_ptr<array_info>& arr2, int64_t out_precision,
    int64_t out_scale, bool is_scalar_arg1, bool is_scalar_arg2,
    bool* const overflow) {
    int64_t s1 = arr1->scale;
    int64_t s2 = arr2->scale;
    int32_t delta_scale = out_scale + s2 - s1;

    size_t len = is_scalar_arg1 ? arr2->length : arr1->length;

    std::unique_ptr<array_info> out_arr =
        alloc_nullable_array_no_nulls(len, Bodo_CTypes::DECIMAL);
    out_arr->precision = out_precision;
    out_arr->scale = out_scale;

    bool out_overflow = false;

    for (size_t i = 0; i < len; i++) {
        bool v1_null =
            !is_scalar_arg1 &&
            !arr1->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i);
        bool v2_null =
            !is_scalar_arg2 &&
            !arr2->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i);

        if (v1_null || v2_null) {
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        } else {
            arrow::Decimal128* out_ptr =
                out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                               arrow::Decimal128>() +
                i;
            size_t d1_ind = is_scalar_arg1 ? 0 : i;
            const arrow::Decimal128& d1 =
                *(arr1->data1<bodo_array_type::NULLABLE_INT_BOOL,
                              arrow::Decimal128>() +
                  d1_ind);
            size_t d2_ind = is_scalar_arg2 ? 0 : i;
            const arrow::Decimal128& d2 =
                *(arr2->data1<bodo_array_type::NULLABLE_INT_BOOL,
                              arrow::Decimal128>() +
                  d2_ind);

            bool overflow_i;
            if constexpr (do_div0) {
                if (d2 == arrow::Decimal128(0)) {
                    *out_ptr = arrow::Decimal128(0);
                    overflow_i = false;
                } else {
                    *out_ptr =
                        decimalops::Divide(d1, d2, delta_scale, &overflow_i);
                }
            } else {
                *out_ptr = decimalops::Divide(d1, d2, delta_scale, &overflow_i);
            }
            out_overflow |= overflow_i;
        }
    }

    *overflow = out_overflow;
    return out_arr;
}

array_info* divide_decimal_arrays_py_entry(array_info* arr1_, array_info* arr2_,
                                           int64_t out_precision,
                                           int64_t out_scale,
                                           bool is_scalar_arg1,
                                           bool is_scalar_arg2, bool* overflow,
                                           bool do_div0 = false) noexcept {
    try {
        std::unique_ptr<array_info> arr1 = std::unique_ptr<array_info>(arr1_);
        std::unique_ptr<array_info> arr2 = std::unique_ptr<array_info>(arr2_);
        assert(arr1->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr2->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr1->dtype == Bodo_CTypes::DECIMAL &&
               arr2->dtype == Bodo_CTypes::DECIMAL);
        std::unique_ptr<array_info> out_arr;
        if (do_div0) {
            out_arr = divide_decimal_arrays<true>(arr1, arr2, out_precision,
                                                  out_scale, is_scalar_arg1,
                                                  is_scalar_arg2, overflow);
        } else {
            out_arr = divide_decimal_arrays<false>(arr1, arr2, out_precision,
                                                   out_scale, is_scalar_arg1,
                                                   is_scalar_arg2, overflow);
        }
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Rounds a given Decimal128 value to a specified scale and updates the
 * result.
 *
 * This function rounds the input Decimal128 value to a specified scale. The
 * rounding process involves shifting the decimal point of the input value by
 * the difference between its scale and the target scale,
 * multiplying the result by a power of 10 if the target scale is negative. The
 * function ensures that the result fits within the Decimal128 range, setting an
 * overflow flag if it does not.
 *
 * @param value The Decimal128 value to be rounded.
 * @param round_scale The target scale to round the value to. A negative scale
 * will multiply the value by 10^(-round_scale).
 * @param input_s The scale of the input value.
 * @param overflow Pointer to a boolean that will be set to true if the result
 * exceeds the Decimal128 range.
 * @param result Pointer to a Decimal128 where the rounded result will be
 * stored.
 */
template <bool negative_round_scale, bool fast_round>
inline void round_decimal_scalar(const arrow::Decimal128& value,
                                 int64_t round_scale, int64_t input_s,
                                 bool* overflow, arrow::Decimal128* result) {
    int64_t shift_amount = input_s - round_scale;
    arrow::Decimal128 rounded_value =
        shift_decimal_scalar(value, -shift_amount);
    if constexpr (negative_round_scale) {
        // For negative rounding scales, we need to multiply the shifted value
        // by 10^(-round_scale) to get the final value.

        // In the case we are certain there is no overflow, we can fast round.
        if constexpr (fast_round) {
            rounded_value = rounded_value.IncreaseScaleBy(-round_scale);
        } else {
            boost::multiprecision::int256_t rounded_value_int256 =
                decimalops::ConvertToInt256(rounded_value);
            rounded_value_int256 =
                decimalops::IncreaseScaleBy(rounded_value_int256, -round_scale);
            rounded_value =
                decimalops::ConvertToDecimal128(rounded_value_int256, overflow);
        }
    }
    *result = rounded_value;
}

/**
 * @brief Python entry point for rounding decimal scalars.
 *
 * This function is a Python entry point to round a Decimal128 value to a
 * specified scale by calling the `round_decimal_scalar` C++ function.
 *
 * @param[in] in_low The low 64 bits of the Decimal128 to be rounded.
 * @param[in] in_high The high 64 bits of the Decimal128 to be rounded.
 * @param[in] round_scale The scale to which the value should be rounded.
 * @param[in] input_s The scale of the input value.
 * @param[out] overflow Pointer to a boolean that indicates if an overflow
 * occurred.
 * @param[out] out_low_ptr Pointer to the low 64 bits of the result.
 * @param[out] out_high_ptr Pointer to the high 64 bits of the result.
 */
void round_decimal_scalar_py_entry(uint64_t in_low, int64_t in_high,
                                   int64_t round_scale, int64_t input_p,
                                   int64_t input_s, bool* overflow,
                                   uint64_t* out_low_ptr,
                                   uint64_t* out_high_ptr) noexcept {
    arrow::Decimal128 result;
    try {
        arrow::Decimal128 value = arrow::Decimal128(in_high, in_low);

        if (round_scale < 0) {
            if (input_p != 38) {
                round_decimal_scalar<true, true>(value, round_scale, input_s,
                                                 overflow, &result);
            } else {
                round_decimal_scalar<true, false>(value, round_scale, input_s,
                                                  overflow, &result);
            }
        } else {
            // fast_round only matters if the scale is negative
            round_decimal_scalar<false, false>(value, round_scale, input_s,
                                               overflow, &result);
        }
        *out_low_ptr = result.low_bits();
        *out_high_ptr = result.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Rounds each element in a Decimal128 array to a specified scale.
 *
 * This function takes an array of Decimal128 values and rounds each element to
 * the specified scale, creating a new array with the rounded values. It handles
 * null values appropriately, preserving their positions in the output array. If
 * any rounding operation causes an overflow, the overflow flag is set to true.
 *
 * @param arr A pointer to the input array of Decimal128 values.
 * @param round_scale The scale to which each decimal value should be rounded.
 * @param output_p The precision of the output array.
 * @param output_s The scale of the output array.
 * @param overflow Pointer to a boolean that indicates overflow.
 * @return A pointer to a new array containing the rounded Decimal128 values.
 */
template <bool negative_round_scale, bool fast_round>
std::unique_ptr<array_info> round_decimal_array(
    const std::unique_ptr<array_info>& arr, int64_t round_scale,
    int64_t output_p, int64_t output_s, bool* const overflow) {
    size_t len = arr->length;
    std::unique_ptr<array_info> out_arr =
        alloc_nullable_array_no_nulls(len, Bodo_CTypes::DECIMAL);
    try {
        out_arr->precision = output_p;
        out_arr->scale = output_s;
        for (size_t i = 0; i < len; i++) {
            if (!arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
            } else {
                arrow::Decimal128* out_ptr =
                    out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                   arrow::Decimal128>() +
                    i;
                const arrow::Decimal128& in_val =
                    *(arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                 arrow::Decimal128>() +
                      i);
                bool overflow_i = false;
                round_decimal_scalar<negative_round_scale, fast_round>(
                    in_val, round_scale, arr->scale, &overflow_i, out_ptr);
                *overflow |= overflow_i;
            }
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return out_arr;
}
/**
 * @brief Entry point for Python to round elements in a Decimal128 array to a
 * specified scale, by calling the `round_decimal_array` C++ function.
 *
 * @param arr_ Pointer to the input `array_info` structure.
 * @param round_scale The scale to which each decimal value should be rounded.
 * @param output_p The precision of the output array
 * @param output_s The scale of the output array.
 * @param overflow Pointer to a boolean that indicates overflow.
 * @return Pointer to a new `array_info` structure containing the rounded
 * Decimal128 values, or nullptr if an exception occurs.
 */
array_info* round_decimal_array_py_entry(array_info* arr_, int64_t round_scale,
                                         int64_t output_p, int64_t output_s,
                                         bool* overflow) noexcept {
    try {
        std::unique_ptr<array_info> arr = std::unique_ptr<array_info>(arr_);
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr->dtype == Bodo_CTypes::DECIMAL);
        std::unique_ptr<array_info> out_arr;
        if (round_scale < 0) {
            if (arr->precision != 38) {
                out_arr = round_decimal_array<true, true>(
                    arr, round_scale, output_p, output_s, overflow);
            } else {
                out_arr = round_decimal_array<true, false>(
                    arr, round_scale, output_p, output_s, overflow);
            }
        } else {
            // fast_round only matters if the scale is negative
            out_arr = round_decimal_array<false, false>(
                arr, round_scale, output_p, output_s, overflow);
        }
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Python entrypoint for taking the ceil or floor of a given Decimal128
 * value.
 *
 * @param[in] in_low The low 64 bits of the input Decimal128 value.
 * @param[in] in_high The high 64 bits of the input Decimal128 value.
 * @param[in] input_p The precision of the input decimal value.
 * @param[in] input_s The scale of the input decimal value.
 * @param[in] round_scale The scale to which the value should be rounded.
 * Negative scales indicate rounding to the left of the decimal point.
 * @param[in] is_ceil A boolean indicating whether to apply the ceiling (true)
 * or floor (false) operation.
 * @param[out] out_low_ptr A pointer to the low 64 bits of the result.
 * @param[out] out_high_ptr A pointer to the high 64 bits of the result.
 */
void ceil_floor_decimal_scalar_py_entry(uint64_t in_low, int64_t in_high,
                                        int32_t input_p, int32_t input_s,
                                        int32_t round_scale, bool is_ceil,
                                        uint64_t* out_low_ptr,
                                        int64_t* out_high_ptr) {
    arrow::Decimal128 result;
    try {
        bool overflow = false;
        arrow::Decimal128 value = arrow::Decimal128(in_high, in_low);

        if (is_ceil) {
            if (round_scale < 0) {
                result = decimalops::Ceil<true>(value, input_p, input_s,
                                                round_scale, &overflow);
            } else {
                result = decimalops::Ceil<false>(value, input_p, input_s,
                                                 round_scale, &overflow);
            }
        } else {
            if (round_scale < 0) {
                result = decimalops::Floor<true>(value, input_p, input_s,
                                                 round_scale, &overflow);
            } else {
                result = decimalops::Floor<false>(value, input_p, input_s,
                                                  round_scale, &overflow);
            }
        }
        if (overflow) {
            throw std::runtime_error("Number out of representable range");
        }
        *out_low_ptr = result.low_bits();
        *out_high_ptr = result.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Templated helper for applying the ceiling or floor operation to a
 * decimal array.
 *
 * @tparam is_ceil A boolean template parameter indicating whether to apply the
 * ceiling (true) or floor (false) operation.
 * @tparam negative_round A boolean template parameter indicating whether to
 * apply negative rounding.
 * @param arr Input array
 * @param output_p The precision of the output
 * @param output_s The scale of the output
 * @param round_scale The scale to which the values should be rounded. Negative
 * scales indicate rounding to the left of the decimal point.
 * @return Resulting array.
 */
template <bool is_ceil, bool negative_round>
std::unique_ptr<array_info> ceil_floor_decimal_array(
    const std::unique_ptr<array_info>& arr, int32_t output_p, int32_t output_s,
    int32_t round_scale) {
    bool overflow = false;
    size_t len = arr->length;
    std::unique_ptr<array_info> out_arr =
        alloc_nullable_array_no_nulls(len, Bodo_CTypes::DECIMAL);
    out_arr->precision = output_p;
    out_arr->scale = output_s;
    for (size_t i = 0; i < len; i++) {
        if (!arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        } else {
            arrow::Decimal128* out_ptr =
                out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                               arrow::Decimal128>() +
                i;
            const arrow::Decimal128& in_val =
                *(arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                             arrow::Decimal128>() +
                  i);
            bool overflow_i = false;
            if constexpr (is_ceil) {
                *out_ptr = decimalops::Ceil<negative_round>(
                    in_val, arr->precision, arr->scale, round_scale,
                    &overflow_i);
            } else {
                *out_ptr = decimalops::Floor<negative_round>(
                    in_val, arr->precision, arr->scale, round_scale,
                    &overflow_i);
            }
            overflow |= overflow_i;
        }
    }
    if (overflow) {
        throw std::runtime_error("Number out of representable range");
    }
    return out_arr;
}

/**
 * @brief Python entrypoint for applying ceil or floor to a decimal array.
 *
 * @param arr_ Input array
 * @param output_p The precision of the output
 * @param output_s The scale of the output
 * @param round_scale The scale to which the values should be rounded. Negative
 * scales indicate rounding to the left of the decimal point.
 * @param is_ceil A boolean indicating whether to apply the ceiling (true) or
 * floor (false) operation.
 * @return Resulting array_info object with the rounded values.
 */
array_info* ceil_floor_decimal_array_py_entry(array_info* arr_,
                                              int32_t output_p,
                                              int32_t output_s,
                                              int32_t round_scale,
                                              bool is_ceil) {
    try {
        std::unique_ptr<array_info> arr = std::unique_ptr<array_info>(arr_);
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr->dtype == Bodo_CTypes::DECIMAL);
        std::unique_ptr<array_info> out_arr;
        if (is_ceil) {
            if (round_scale < 0) {
                out_arr = ceil_floor_decimal_array<true, true>(
                    arr, output_p, output_s, round_scale);
            } else {
                out_arr = ceil_floor_decimal_array<true, false>(
                    arr, output_p, output_s, round_scale);
            }
        } else {
            if (round_scale < 0) {
                out_arr = ceil_floor_decimal_array<false, true>(
                    arr, output_p, output_s, round_scale);
            } else {
                out_arr = ceil_floor_decimal_array<false, false>(
                    arr, output_p, output_s, round_scale);
            }
        }
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Python entrypoint for truncating a given Decimal128 value.
 *
 * @param[in] in_low The low 64 bits of the the Decimal128 value to truncate.
 * @param[in] in_high The high 64 bits Decimal128 value.
 * @param[in] input_p The precision of the input decimal value.
 * @param[in] input_s The scale of the input decimal value.
 * @param[in] round_scale The scale to which the value should be truncated.
 * Negative scales indicate truncating to the left of the decimal point.
 * @param[out] overflow A pointer to a boolean indicating overflow.
 * @param[out] out_low_ptr A pointer to the low 64 bits of the result.
 * @param[out] out_high_ptr A pointer to the high 64 bits of the result.
 */
void trunc_decimal_scalar_py_entry(uint64_t in_low, int64_t in_high,
                                   int32_t input_p, int32_t input_s,
                                   int32_t output_p, int32_t output_s,
                                   int32_t round_scale, uint64_t* out_low_ptr,
                                   int64_t* out_high_ptr) {
    arrow::Decimal128 result;
    bool overflow = false;
    try {
        arrow::Decimal128 value = arrow::Decimal128(in_high, in_low);

        if (round_scale < 0) {
            result =
                decimalops::Truncate<true>(value, input_p, input_s, output_p,
                                           output_s, round_scale, &overflow);
        } else {
            result =
                decimalops::Truncate<false>(value, input_p, input_s, output_p,
                                            output_s, round_scale, &overflow);
        }
        if (overflow) {
            throw std::runtime_error("Number out of representable range");
        }
        *out_low_ptr = result.low_bits();
        *out_high_ptr = result.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Templated helper for truncating a decimal array to a given scale.
 *
 * @tparam negative_round A boolean template parameter indicating whether to
 * apply negative scale truncating.
 * @param arr Input array
 * @param output_p The precision of the output
 * @param output_s The scale of the output
 * @param round_scale The scale to which the values should be truncated.
 * Negative scales indicate truncating to the left of the decimal point.
 * @param overflow Boolean overflow pointer.
 * @return Resulting array.
 */
template <bool negative_round>
std::unique_ptr<array_info> trunc_decimal_array(
    const std::unique_ptr<array_info>& arr, int32_t output_p, int32_t output_s,
    int32_t round_scale) {
    bool overflow = false;
    size_t len = arr->length;
    std::unique_ptr<array_info> out_arr =
        alloc_nullable_array_no_nulls(len, Bodo_CTypes::DECIMAL);
    out_arr->precision = output_p;
    out_arr->scale = output_s;
    for (size_t i = 0; i < len; i++) {
        if (!arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        } else {
            arrow::Decimal128* out_ptr =
                out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                               arrow::Decimal128>() +
                i;
            const arrow::Decimal128& in_val =
                *(arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                             arrow::Decimal128>() +
                  i);
            bool overflow_i = false;
            *out_ptr = decimalops::Truncate<negative_round>(
                in_val, arr->precision, arr->scale, output_p, output_s,
                round_scale, &overflow_i);
            overflow |= overflow_i;
        }
    }
    if (overflow) {
        throw std::runtime_error("Number out of representable range");
    }
    return out_arr;
}

/**
 * @brief Python entrypoint for truncating a decimal array to a given scale.
 *
 * @param arr_ Input array
 * @param output_p The precision of the output
 * @param output_s The scale of the output
 * @param round_scale The scale to which the values should be truncated.
 * Negative scales indicate truncating to the left of the decimal point.
 * @param overflow A pointer to a boolean indicating overflow.
 * @return Resulting array_info object with the truncated values.
 */
array_info* trunc_decimal_array_py_entry(array_info* arr_, int32_t output_p,
                                         int32_t output_s,
                                         int32_t round_scale) {
    try {
        std::unique_ptr<array_info> arr = std::unique_ptr<array_info>(arr_);
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr->dtype == Bodo_CTypes::DECIMAL);
        std::unique_ptr<array_info> out_arr;
        if (round_scale < 0) {
            out_arr =
                trunc_decimal_array<true>(arr, output_p, output_s, round_scale);
        } else {
            out_arr = trunc_decimal_array<false>(arr, output_p, output_s,
                                                 round_scale);
        }
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * Computes the absolute value of a given decimal scalar.
 *
 * @param[in] in_low The low 64 bits of the decimal value for which the absolute
 * value is to be computed.
 * @param[in] in_high The high 64 bits of the decimal value for which the
 * absolute value is to be computed.
 * @param[out] out_low_ptr Pointer to the low 64 bits of the result.
 * @param[out] out_high_ptr Pointer to the high 64 bits of the result.
 */
void abs_decimal_scalar_py_entry(uint64_t in_low, int64_t in_high,
                                 uint64_t* out_low_ptr, int64_t* out_high_ptr) {
    try {
        arrow::Decimal128 val = arrow::Decimal128(in_high, in_low);
        arrow::Decimal128 result = val.Abs();
        *out_low_ptr = result.low_bits();
        *out_high_ptr = result.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * Computes the absolute values for all elements in a given decimal array.
 *
 * @param arr_ input array containing the decimal values.
 * @return Pointer to a new array_info object with the absolute values of the
 * input array elements. Returns nullptr if an exception occurs.
 */
array_info* abs_decimal_array_py_entry(array_info* arr_) {
    try {
        std::unique_ptr<array_info> arr = std::unique_ptr<array_info>(arr_);
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr->dtype == Bodo_CTypes::DECIMAL);
        size_t len = arr->length;
        std::unique_ptr<array_info> out_arr =
            alloc_nullable_array_no_nulls(len, Bodo_CTypes::DECIMAL);
        out_arr->precision = arr->precision;
        out_arr->scale = arr->scale;
        for (size_t i = 0; i < len; i++) {
            if (!arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
            } else {
                arrow::Decimal128* out_ptr =
                    out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                   arrow::Decimal128>() +
                    i;
                arrow::Decimal128& in_val =
                    *(arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                 arrow::Decimal128>() +
                      i);
                *out_ptr = in_val.Abs();
            }
        }
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Take the factorial of a decimal scalar.
 *
 * @param val Input decimal
 * @param input_s Scale of the input
 * @return arrow::Decimal128 Result of the factorial
 */
arrow::Decimal128 factorial_decimal_scalar(arrow::Decimal128 val,
                                           int64_t input_s) {
    // First, round the value to an integer
    arrow::Decimal128 rounded_val = arrow::Decimal128(0);
    bool overflow = false;
    round_decimal_scalar<false, false>(val, 0, input_s, &overflow,
                                       &rounded_val);
    // Check if within bounds
    if (overflow || rounded_val > arrow::Decimal128(33)) {
        std::string err_msg =
            "Factorial input " + val.ToString(input_s) + " is too large";
        throw std::runtime_error(err_msg);
    } else if (rounded_val < arrow::Decimal128(0)) {
        std::string err_msg =
            "Factorial input " + val.ToString(input_s) + " is negative";
        throw std::runtime_error(err_msg);
    }
    int rounded_val_int = rounded_val.ToInteger<int>().ValueOrDie();
    // Look up factorial value in table
    assert(rounded_val_int >= 0 && rounded_val_int <= 33);
    boost::multiprecision::int256_t factorial_val_int =
        decimalops::factorial_table[rounded_val_int];
    // Convert back to decimal
    arrow::Decimal128 factorial_val =
        decimalops::ConvertToDecimal128(factorial_val_int, nullptr);

    return factorial_val;
}

/**
 * @brief Python entry point for taking the factorial of a decimal scalar.
 *
 * @param[in] in_low Low 64 bits of the input decimal.
 * @param[in] in_high High 64 bits of the input decimal.
 * @param[in] input_s Scale of the input
 * @param[out] out_low_ptr Pointer to the low 64 bits of the result.
 * @param[out] out_high_ptr Pointer to the high 64 bits of the result.
 */
void factorial_decimal_scalar_py_entry(uint64_t in_low, int64_t in_high,
                                       int64_t input_s, uint64_t* out_low,
                                       int64_t* out_high) {
    try {
        arrow::Decimal128 val = arrow::Decimal128(in_high, in_low);
        arrow::Decimal128 out = factorial_decimal_scalar(val, input_s);
        *out_low = out.low_bits();
        *out_high = out.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Python entry point for taking the factorial of a decimal array.
 *
 * @param arr_ Input decimal array
 * @return array_info* of the result
 */
array_info* factorial_decimal_array_py_entry(array_info* arr_) {
    try {
        std::unique_ptr<array_info> arr = std::unique_ptr<array_info>(arr_);
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr->dtype == Bodo_CTypes::DECIMAL);
        size_t len = arr->length;
        std::unique_ptr<array_info> out_arr =
            alloc_nullable_array_no_nulls(len, Bodo_CTypes::DECIMAL);
        out_arr->precision = 37;
        out_arr->scale = 0;
        for (size_t i = 0; i < len; i++) {
            if (!arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
            } else {
                arrow::Decimal128* out_ptr =
                    out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                   arrow::Decimal128>() +
                    i;
                const arrow::Decimal128& in_val =
                    *(arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                 arrow::Decimal128>() +
                      i);
                *out_ptr = factorial_decimal_scalar(in_val, arr->scale);
            }
        }
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Convert decimal value to int64 (unsafe cast)
 *
 * @param in_low low 64 bits of the input decimal
 * @param in_high high 64 bits of the input decimal
 * @param precision input's precision
 * @param scale input's scale
 * @return int64_t input converted to int64
 */
int64_t decimal_to_int64_py_entry(uint64_t in_low, int64_t in_high,
                                  uint8_t precision, uint8_t scale) {
    try {
        // NOTE: using cast to allow unsafe cast (Rescale/ToInteger may throw
        // data loss error)
        arrow::Decimal128 arrow_decimal(in_high, in_low);
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

void cast_float_to_decimal_scalar_py_entry(double f, int32_t precision,
                                           int32_t scale, bool* safe,
                                           uint64_t* out_low_ptr,
                                           int64_t* out_high_ptr) {
    try {
        arrow::Decimal128 answer;
        double max_value = std::pow(10.0, precision - scale);
        bool safe_cast = std::abs(f) < max_value;
        *safe = safe_cast;
        if (!safe_cast) {
            answer = arrow::Decimal128(0);
        } else {
            auto result = arrow::Decimal128::FromReal(f, precision, scale);
            CHECK_ARROW_AND_ASSIGN(result, "failed to convert float to decimal",
                                   answer)
        }
        *out_low_ptr = answer.low_bits();
        *out_high_ptr = answer.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
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
            arr_out->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(row,
                                                                      false);
        } else if (isnan_alltype<T, DType>(f) || std::abs(f) >= max_value) {
            if constexpr (null_on_error) {
                arr_out->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    row, false);
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
            decimal_value res = {.low = static_cast<int64_t>(low_bits),
                                 .high = high_bits};
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
    return {.low = static_cast<int64_t>(low_bits), .high = high_bits};
}

void unbox_decimal(PyObject* obj, uint8_t* data);

PyObject* box_decimal(uint64_t low, int64_t high, int8_t precision,
                      int8_t scale) {
    // convert input to Arrow scalar
    arrow::Decimal128 arrow_decimal(high, low);
    std::shared_ptr<arrow::Decimal128Scalar> scalar =
        std::make_shared<arrow::Decimal128Scalar>(
            arrow_decimal, arrow::decimal128(precision, scale));

    // convert Arrow C++ to PyArrow
    return arrow::py::wrap_scalar(scalar);
}

void decimal_to_str(uint64_t in_low, int64_t in_high, NRT_MemInfo** meminfo_ptr,
                    int64_t* len_ptr, int scale, bool remove_trailing_zeroes);

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
            arrow_array_to_bodo(out_arrow_arr, nullptr);
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
 * @param arg0_low Low 64 bits of the input decimal.
 * @param arg0_high High 64 bits of the input decimal.
 * @param precision decimal argument's precision
 * @param scale decimal argument's scale
 * @param arg1 second argument (int)
 * @return bool output of comparison
 */
bool arrow_compute_cmp_decimal_int_py_entry(int32_t op_enum, uint64_t arg0_low,
                                            int64_t arg0_high,
                                            int32_t precision, int32_t scale,
                                            int64_t arg1) {
    try {
        // convert input to Arrow scalars
        arrow::Decimal128 arrow_decimal(arg0_high, arg0_low);
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
 * @param arg0_low Low 64 bits of the input decimal.
 * @param arg0_high High 64 bits of the input decimal.
 * @param precision decimal argument's precision
 * @param scale decimal argument's scale
 * @param arg1 second argument (float)
 * @return bool output of comparison
 */
bool arrow_compute_cmp_decimal_float_py_entry(int32_t op_enum,
                                              uint64_t arg0_low,
                                              int64_t arg0_high,
                                              int32_t precision, int32_t scale,
                                              double arg1) {
    try {
        // convert input to Arrow scalars
        arrow::Decimal128 arrow_decimal(arg0_high, arg0_low);
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
 * @param arg0_low Low 64 bits of the first input decimal.
 * @param arg0_high High 64 bits of the first input decimal.
 * @param precision0 first argument's precision
 * @param scale0 first argument's scale
 * @param arg1_low Low 64 bits of the second input decimal.
 * @param arg1_high High 64 bits of the second input decimal.
 * @param precision1 second argument's precision
 * @param scale1 second argument's scale
 * @return bool output of comparison
 */
bool arrow_compute_cmp_decimal_decimal_py_entry(
    int32_t op_enum, uint64_t arg0_low, int64_t arg0_high, int32_t precision0,
    int32_t scale0, int32_t precision1, int32_t scale1, uint64_t arg1_low,
    int64_t arg1_high) {
    try {
        // convert input to Arrow scalars
        arrow::Decimal128 arrow_decimal0(arg0_high, arg0_low);
        arrow::Decimal128Scalar arg0_scalar(
            arrow_decimal0, arrow::decimal128(precision0, scale0));
        arrow::Decimal128 arrow_decimal1(arg1_high, arg1_low);
        arrow::Decimal128Scalar arg1_scalar(
            arrow_decimal1, arrow::decimal128(precision1, scale1));
        return arrow_compute_cmp_scalar(op_enum, arg0_scalar, arg1_scalar);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
    }
}

/**
 * @brief Cast a string to a decimal scalar, rounding half up if the scale
 * doesn't fit.
 *
 * @param str_val Input string to cast
 * @param precision Output precision
 * @param scale Output scale
 * @param error[out] If an error occurs, set this to true.
 * @return arrow::Decimal128 output decimal scalar
 */
arrow::Decimal128 str_to_decimal_scalar(const std::string_view& str_val,
                                        int32_t precision, int32_t scale,
                                        bool* error) {
    int32_t final_leading_digits = precision - scale;
    // Create the decimal value from a string. Arrow provides
    // the parsed precision and scale.
    arrow::Decimal128 decimal;
    int32_t parsed_precision;
    int32_t parsed_scale;
    // Handle any leading and trailing spaces.
    // Note: Arrow handles any leading 0s.
    size_t start = 0;
    while (start < str_val.size() && isspace(str_val[start])) {
        start++;
    }
    size_t end = str_val.size() - 1;
    while (end > start && isspace(str_val[end])) {
        end--;
    }
    bool seen_exp = false;
    // Trailing E is equivalent to remainder of string * 1
    if (str_val[end] == 'E' || str_val[end] == 'e') {
        end--;
        seen_exp = true;
    }
    std::string_view used_val =
        std::string_view(str_val).substr(start, (end - start) + 1);
    if ((used_val.empty() || used_val == "-" || used_val == "+") && seen_exp) {
        // Just E is 0 and fits in any precision/scale.
        *error = false;
        return arrow::Decimal128(0);
    } else {
        arrow::Status status = arrow::Decimal128::FromString(
            used_val, &decimal, &parsed_precision, &parsed_scale);
        if (!status.ok()) {
            *error = true;
            return arrow::Decimal128(0);
        }
        int32_t parsed_leading_digits = parsed_precision - parsed_scale;
        if (parsed_leading_digits > final_leading_digits) {
            // If we have more leading digits than allowed we must thrown an
            // error.
            *error = true;
            return arrow::Decimal128(0);
        }
        if (parsed_precision > decimalops::kMaxPrecision) {
            // Extreme edge case where the value doesn't fit in the maximum
            // precision. Try parsing as a Decimal256 and then truncate.
            arrow::Decimal256 decimal256;
            int32_t parsed_precision256;
            int32_t parsed_scale256;
            status = arrow::Decimal256::FromString(
                used_val, &decimal256, &parsed_precision256, &parsed_scale256);
            if (!status.ok()) {
                *error = true;
                return arrow::Decimal128(0);
            }
            decimal256 = shift_decimal_scalar(decimal256, scale - parsed_scale);
            if (!decimal256.FitsInPrecision(decimalops::kMaxPrecision)) {
                *error = true;
                return arrow::Decimal128(0);
            }
            // Note: We add an explicit type because we want to make sure this
            // is type stable.
            std::array<uint64_t, 4> data_array =
                decimal256.little_endian_array();
            return arrow::Decimal128(arrow::BasicDecimal128{
                static_cast<int64_t>(data_array[1]), data_array[0]});
        } else {
            *error = false;
            // If the scale doesn't match then we need to shift the output.
            return shift_decimal_scalar(decimal, scale - parsed_scale);
        }
    }
}

/**
 * @brief Python entry point for converting a string value to decimal.
 *
 * @param str_val The string value to convert.
 * @param precision The output precision.
 * @param scale The output scale.
 * @param[out] out_low low 64 bits of output.
 * @param[out] out_high high 64 bits of output.
 * @param[out] error If an error occurs, set this to true.
 */
void str_to_decimal_scalar_py_entry(const std::string* str_val,
                                    int64_t precision, int64_t scale,
                                    uint64_t* out_low, int64_t* out_high,
                                    bool* error) {
    try {
        arrow::Decimal128 res =
            str_to_decimal_scalar(*str_val, precision, scale, error);
        *out_low = res.low_bits();
        *out_high = res.high_bits();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Python entry point for converting a string array to decimal.
 *
 * @param str_val The string value to convert.
 * @param precision The output precision.
 * @param scale The output scale.
 * @param null_on_error If an error occurs just set null.
 * @return Output decimal array.
 */
template <bool null_on_error>
std::unique_ptr<array_info> str_to_decimal_array(
    std::shared_ptr<array_info> in_arr, int64_t precision, int64_t scale) {
    // Note: This assumes we have a regular string array. Dictionary arrays
    // are handled in Python.
    assert(in_arr->arr_type == bodo_array_type::STRING);
    assert(in_arr->dtype == Bodo_CTypes::STRING);
    char* in_data = in_arr->data1<bodo_array_type::STRING, char>();
    offset_t* in_offsets = (offset_t*)in_arr->data2<bodo_array_type::STRING>();
    uint8_t* in_nulls =
        (uint8_t*)in_arr->null_bitmask<bodo_array_type::STRING>();
    size_t n_rows = in_arr->length;
    std::unique_ptr<array_info> out_arr =
        alloc_nullable_array_no_nulls(n_rows, Bodo_CTypes::DECIMAL);
    arrow::Decimal128* out_data =
        out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL, arrow::Decimal128>();
    for (size_t i = 0; i < n_rows; i++) {
        if (is_na(in_nulls, i)) {
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
            continue;
        }
        offset_t start = in_offsets[i];
        offset_t end = in_offsets[i + 1];
        const std::string_view str_val(in_data + start, end - start);
        bool error;
        out_data[i] = str_to_decimal_scalar(str_val, precision, scale, &error);
        if (error) {
            if (null_on_error) {
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
            } else {
                throw std::runtime_error(
                    "String value is out of range for decimal or doesn't parse "
                    "properly: " +
                    std::string(str_val));
            }
        }
    }
    return out_arr;
}

/**
 * @brief Python entry point for converting a string array to decimal.
 *
 * @param arr The string array to convert.
 * @param precision The output precision.
 * @param scale The output scale.
 * @param null_on_error If an error occurs just set null.
 * @return Output decimal array.
 */
array_info* str_to_decimal_array_py_entry(array_info* arr, int64_t precision,
                                          int64_t scale, bool null_on_error) {
    try {
        std::unique_ptr<array_info> out_arr;
        std::shared_ptr<array_info> in_arr = std::shared_ptr<array_info>(arr);
        if (null_on_error) {
            out_arr = str_to_decimal_array<true>(in_arr, precision, scale);
        } else {
            out_arr = str_to_decimal_array<false>(in_arr, precision, scale);
        }
        return out_arr.release();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Converts a Decimal128 array to a string array, preserving null values.
 *
 * This function takes an array of Decimal128 values and converts each element
 * to its string representation.
 *
 * @param arr A pointer to the input array of Decimal128 values.
 * @param pool Buffer pool. Defaults to the default buffer pool.
 * @param mm Memory manager. Defaults to the default memory manager.
 * @return A pointer to a new array_info with arr_type and dtype STRING.
 */
std::unique_ptr<array_info> decimal_array_to_str_array(
    const std::unique_ptr<array_info>& arr,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    size_t len = arr->length;
    int64_t scale = arr->scale;
    // We need a vector of strings and a null bitmask to create a new
    // string array.
    bodo::vector<std::string> strings(len, pool);
    // (len + 7) >> 3 is the minimum number of bytes
    // needed to store a bitmask for an array of length len.
    // Equivalent to ceil(len / 8).
    bodo::vector<uint8_t> nulls((len + 7) >> 3, 0, pool);
    try {
        for (size_t i = 0; i < len; i++) {
            if (!arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
                SetBitTo(nulls.data(), i, false);
            } else {
                SetBitTo(nulls.data(), i, true);
                const arrow::Decimal128& in_val =
                    *(arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                                 arrow::Decimal128>() +
                      i);
                std::string out_str =
                    decimal_to_std_string<false>(in_val, scale);
                strings[i] = out_str;
            }
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    std::unique_ptr<array_info> out_arr = create_string_array(
        Bodo_CTypes::STRING, nulls, strings, -1, pool, std::move(mm));
    return out_arr;
}

/**
 * Python entrypoint for converting a Decimal128 array to a string array.
 *
 * @param arr_ The Decimal128 array to convert.
 * @return A pointer to a new array_info with arr_type and dtype STRING.
 */
array_info* decimal_array_to_str_array_py_entry(array_info* arr_) {
    try {
        std::unique_ptr<array_info> arr = std::unique_ptr<array_info>(arr_);
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
               arr->dtype == Bodo_CTypes::DECIMAL);
        std::unique_ptr<array_info> out_arr = decimal_array_to_str_array(arr);
        assert(out_arr->arr_type == bodo_array_type::STRING &&
               out_arr->dtype == Bodo_CTypes::STRING);
        array_info* out_arr_info = new array_info(*out_arr);
        return out_arr_info;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyMODINIT_FUNC PyInit_decimal_ext(void) {
    PyObject* m;
    MOD_DEF(m, "decimal_ext", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    // decimal_value should be exactly 128 bits to match Python
    if (sizeof(decimal_value) != 16) {
        std::cerr << "invalid decimal struct size" << std::endl;
    }

    // These are all C functions, so they don't throw any exceptions.
    // We might still need to add better error handling in the future.
    SetAttrStringFromVoidPtr(m, unbox_decimal);
    SetAttrStringFromVoidPtr(m, box_decimal);

    SetAttrStringFromVoidPtr(m, decimal_to_str);
    SetAttrStringFromVoidPtr(m, str_to_decimal_scalar_py_entry);
    SetAttrStringFromVoidPtr(m, str_to_decimal_array_py_entry);
    SetAttrStringFromVoidPtr(m, decimal_to_double_py_entry);
    SetAttrStringFromVoidPtr(m, decimal_arr_to_double_py_entry);
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
    SetAttrStringFromVoidPtr(m, decimal_array_sign_py_entry);
    SetAttrStringFromVoidPtr(m, sum_decimal_array_py_entry);
    SetAttrStringFromVoidPtr(m, decimal_scalar_sign_py_entry);
    SetAttrStringFromVoidPtr(m, add_or_subtract_decimal_scalars_py_entry);
    SetAttrStringFromVoidPtr(m, add_or_subtract_decimal_arrays_py_entry);
    SetAttrStringFromVoidPtr(m, multiply_decimal_scalars_py_entry);
    SetAttrStringFromVoidPtr(m, multiply_decimal_arrays_py_entry);
    SetAttrStringFromVoidPtr(m, modulo_decimal_scalars_py_entry);
    SetAttrStringFromVoidPtr(m, modulo_decimal_arrays_py_entry);
    SetAttrStringFromVoidPtr(m, divide_decimal_scalars_py_entry);
    SetAttrStringFromVoidPtr(m, divide_decimal_arrays_py_entry);
    SetAttrStringFromVoidPtr(m, decimal_array_to_str_array_py_entry);
    SetAttrStringFromVoidPtr(m, round_decimal_array_py_entry);
    SetAttrStringFromVoidPtr(m, round_decimal_scalar_py_entry);
    SetAttrStringFromVoidPtr(m, ceil_floor_decimal_scalar_py_entry);
    SetAttrStringFromVoidPtr(m, ceil_floor_decimal_array_py_entry);
    SetAttrStringFromVoidPtr(m, trunc_decimal_scalar_py_entry);
    SetAttrStringFromVoidPtr(m, trunc_decimal_array_py_entry);
    SetAttrStringFromVoidPtr(m, abs_decimal_array_py_entry);
    SetAttrStringFromVoidPtr(m, abs_decimal_scalar_py_entry);
    SetAttrStringFromVoidPtr(m, factorial_decimal_scalar_py_entry);
    SetAttrStringFromVoidPtr(m, factorial_decimal_array_py_entry);

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
 * @param in_low low 64 bits of the input value
 * @param in_high high 64 bits of the input value
 * @param meminfo_ptr output memptr pointer to set
 * @param len_ptr output length to set
 * @param scale scale parameter of decimal128
 */
void decimal_to_str(uint64_t in_low, int64_t in_high, NRT_MemInfo** meminfo_ptr,
                    int64_t* len_ptr, int scale,
                    bool remove_trailing_zeroes = true) {
    // Creating the arrow_decimal value
    arrow::Decimal128 arrow_decimal(in_high, in_low);
    // Getting the string
    std::string str;
    if (remove_trailing_zeroes) {
        str = decimal_to_std_string<true>(arrow_decimal, scale);
    } else {
        str = decimal_to_std_string<false>(arrow_decimal, scale);
    }
    // Now doing the boxing.
    int64_t l = (int64_t)str.length();
    NRT_MemInfo* meminfo = alloc_meminfo(l + 1);
    memcpy(meminfo->data, str.c_str(), l);
    ((char*)meminfo->data)[l] = 0;
    *len_ptr = l;
    *meminfo_ptr = meminfo;
}
