#pragma once

#include <arrow/util/decimal.h>
#include <string>

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
 * @return arrow::Array
 */
std::shared_ptr<arrow::Array> arrow_array_decimal_arithmetic_util(
    std::shared_ptr<arrow::Decimal128Array> left_arr, int left_precision,
    int left_scale, std::shared_ptr<arrow::Decimal128Array> right_arr,
    int right_precision, int right_scale, int length, int result_precision,
    int result_scale, const std::string& op);
