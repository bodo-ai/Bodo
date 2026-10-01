#pragma once

#include <arrow/util/decimal.h>
#include <string>

#include "_bodo_common.h"

#define DECIMAL128_MAX_PRECISION 38

std::string int128_decimal_to_std_string(__int128_t const& value,
                                         int const& scale);

double decimal_to_double(__int128_t const& val, uint8_t scale = 18);

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
    bool do_addition, bool* overflow);

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
    bool* overflow);

/**
 * @brief Perform arithmetic operation on two Decimal arrow arrays of
 * equal length with the given precision and scale
 * and return the output. The output should have its scale truncated to
 * the provided output scale. If overflow is detected, then nullptr is
 * returned.
 *
 * @param left_arr First decimal array
 * @param left_precision Precision of first decimal array
 * @param left_scale Scale of first decimal array
 * @param right_arr First decimal array
 * @param right_precision Precision of first decimal array
 * @param right_scale Scale of first decimal array
 * @param length Length of both arrays
 * @param result_precision Output precision
 * @param result_scale Output scale
 * @param op Arithmetic operation either add, subtract, or multiply.
 * @return std::shared_ptr<arrow::Array>
 */
std::shared_ptr<arrow::Array> arrow_array_decimal_arithmetic_util(
    std::shared_ptr<arrow::Decimal128Array> left_arr, int left_precision,
    int left_scale, std::shared_ptr<arrow::Decimal128Array> right_arr,
    int right_precision, int right_scale, int length, int result_precision,
    int result_scale, const std::string& op);

/**
 * @brief Perform arithmetic operation on two Decimal arrow scalars
 * with the given precision and scale and return the output.
 * The output should have its scale truncated to
 * the provided output scale. If overflow is detected, then nullptr is
 * returned.
 *
 * @param left_arr First decimal scalar
 * @param left_precision Precision of first decimal scalar
 * @param left_scale Scale of first decimal scalar
 * @param right_arr First decimal scalar
 * @param right_precision Precision of first decimal scalar
 * @param right_scale Scale of first decimal scalar
 * @param length Length of both scalars
 * @param result_precision Output precision
 * @param result_scale Output scale
 * @param op Arithmetic operation either add, subtract, or multiply.
 * @return std::shared_ptr<arrow::Scalar>
 */
std::shared_ptr<arrow::Scalar> arrow_scalar_decimal_arithmetic_util(
    std::shared_ptr<arrow::Decimal128Scalar> left_val, int left_precision,
    int left_scale, std::shared_ptr<arrow::Decimal128Scalar> right_val,
    int right_precision, int right_scale, int result_precision,
    int result_scale, const std::string& op);

/**
 * @brief Whether the int64-backed decimal fast path is enabled. Can be
 * disabled by setting the BODO_DISABLE_DECIMAL_INT64_FASTPATH environment
 * variable (useful for A/B testing against the 128-bit path).
 */
bool decimal_int64_fastpath_enabled();

/**
 * @brief Int64-backed fast path for binary decimal operations (add,
 * subtract, multiply, and the six comparison operators).
 *
 * Operand values that fit in a signed 64-bit integer (checked per row for
 * decimal operands) are evaluated with int64 loads and __int128 arithmetic
 * instead of Arrow's 128-bit decimal kernels. Values that do not fit are
 * computed with the existing 128-bit scalar utilities, so results are
 * identical to the 128-bit path.
 *
 * @param left Left operand (decimal128 or integer, array or scalar)
 * @param left_precision Precision of the left operand
 * @param left_scale Scale of the left operand
 * @param right Right operand (decimal128 or integer, array or scalar)
 * @param right_precision Precision of the right operand
 * @param right_scale Scale of the right operand
 * @param result_precision Output precision
 * @param result_scale Output scale
 * @param op Operation name ("add", "subtract", "multiply", "equal",
 *            "not_equal", "less", "greater", "less_equal",
 *            "greater_equal")
 * @param check_max_precision If true, apply gandiva-style rounding on scale
 *                            reduction and check that the result fits in
 *                            decimal128 max precision (38), matching
 *                            decimal_arithmetic for result precision > 38.
 *                            If false, results are computed exactly with no
 *                            rounding (callers must ensure no scale
 *                            reduction is needed for multiply).
 * @return arrow::Result<arrow::Datum> Boolean (comparisons) or
 *         decimal128(result_precision, result_scale) datum, or an Invalid
 *         status on overflow.
 */
arrow::Result<arrow::Datum> decimal_int64_binary_op(
    const arrow::Datum& left, int left_precision, int left_scale,
    const arrow::Datum& right, int right_precision, int right_scale,
    int result_precision, int result_scale, const std::string& op,
    bool check_max_precision);

/**
 * @brief In-place array-level variant of decimal_int64_binary_op operating
 * directly on Bodo array_info buffers (no Arrow Datum conversion). The
 * output is allocated from Bodo's buffer pool. Returns nullptr if the
 * operation or operands are not supported (caller should fall back to the
 * generic path).
 *
 * Scalar operands are represented as one-element arrays with the
 * corresponding is_scalar flag set.
 *
 * @param left Left operand array_info
 * @param left_is_scalar Whether left is a scalar (one-element array)
 * @param right Right operand array_info
 * @param right_is_scalar Whether right is a scalar (one-element array)
 * @param result_precision Output precision
 * @param result_scale Output scale
 * @param op Operation name (same set as decimal_int64_binary_op)
 * @param check_max_precision See decimal_int64_binary_op
 * @return std::shared_ptr<array_info> Boolean (comparisons) or decimal
 *         output array, or nullptr if not supported.
 */
std::shared_ptr<array_info> decimal_int64_binary_op_arrays(
    const std::shared_ptr<array_info>& left, bool left_is_scalar,
    const std::shared_ptr<array_info>& right, bool right_is_scalar,
    int result_precision, int result_scale, const std::string& op,
    bool check_max_precision);
