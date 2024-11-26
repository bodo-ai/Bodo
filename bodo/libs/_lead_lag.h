#pragma once

#include "_array_utils.h"
#include "_bodo_common.h"

/**
 * @brief Sequential implementation of lead/lag, with support for default
 * values. The function will effectively shift all values of the array by
 * shift_amt, whether it is negative or positive. In the process, any values
 * that correspond to out of bounds indices in the input are set to the
 * default_fill_val, or null if it is not present.
 *
 * @tparam ArrType The arr_type enum value of the array.
 * @tparam DType The Bodo DType of the array's elements.
 * @tparam ignore_nulls Whether or not to ignore null values. If this is
 * true, instead of copying a corresponding null value, we will copy the *next*
 * non-null value, or the default value if there is no such value.
 * @tparam has_default Whether or not our std::optional default_fill_val has a
 * value.
 * @tparam T The C++ type equivalent of DType. Used for optional value and
 * operations on the array. Don't pass this manually--the default value is
 * required.
 *
 * @param in_col unique_ptr to the array_info that the operation will be
 * performed on.
 * @param shift_amt The amount to shift by. This can be negative or positive, or
 * even zero. For example, 3 corresponds to a value at index 3 in the input
 * being copied to index 0 in the output.
 * @param default_fill_val The value to fill out-of-bounds values with. This
 * type T must be of a type compatible with the array's dtype. If nullopt is
 * passed, these values are filled with NULL.
 * @param default_fill_val_len Length of default_fill_val if it is a string.
 * Zero, otherwise.
 *
 * @return A unique_ptr to a new array_info, on which the lead/lag operation has
 * been performed.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, bool has_default,
          typename T = dtype_to_type<DType>::type>
std::unique_ptr<array_info> lead_lag_seq(
    const std::shared_ptr<array_info> &in_col, int64_t shift_amt,
    const std::optional<T> &default_fill_val, int64_t default_fill_val_len = 0);

/**
 * @brief Wrapper to lead_lag_seq that does not template on ignore_nulls and
 * has_value.
 *
 * Performs runtime checks to calculate these values and call lead_lag_seq.
 *
 * @tparam ArrType The arr_type enum value of the array.
 * @tparam DType The Bodo DType of the array's elements.
 * @tparam T The C++ type equivalent of DType. Used for optional value and
 * operations on the array. Don't pass this manually--the default value is
 * required.
 *
 * @param in_col Passed to lead_lag_seq.
 * @param shift_amt Passed to lead_lag_seq.
 * @param default_fill_val Passed to lead_lag_seq.
 * @param ignore_nulls Converted to a tparam and passed to lead_lag_seq.
 * @param default_fill_val_len Length of default_fill_val if it is a string.
 * Zero, otherwise.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          typename T = dtype_to_type<DType>::type>
std::unique_ptr<array_info> lead_lag_seq_wrapper(
    const std::shared_ptr<array_info> &in_col, int64_t shift_amt,
    const std::optional<T> &default_fill_val, const bool ignore_nulls,
    int64_t default_fill_val_len = 0);
