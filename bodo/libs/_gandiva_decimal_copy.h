// Copyright (C) 2024 Bodo Inc. All rights reserved.
// https://github.com/apache/arrow/blob/299eb26e8c22b4aad4876c9e3b52f9adde699a5c/cpp/src/gandiva/precompiled/decimal_ops.cc#L331
// Code heavily copied from the gandiva precompiled code
// to avoid importing the library.
#pragma once

#include <arrow/util/decimal.h>
#include <boost/multiprecision/cpp_int.hpp>

namespace decimalops {

constexpr int32_t kMaxPrecision = 38;

static constexpr int32_t kMaxLargeScale = 2 * kMaxPrecision;

// Compute the scale multipliers once.
static std::array<boost::multiprecision::int256_t, kMaxLargeScale + 1>
    kLargeScaleMultipliers = ([]()
                                  -> std::array<boost::multiprecision::int256_t,
                                                kMaxLargeScale + 1> {
        std::array<boost::multiprecision::int256_t, kMaxLargeScale + 1> values;
        values[0] = 1;
        for (int32_t idx = 1; idx <= kMaxLargeScale; idx++) {
            values[idx] = values[idx - 1] * 10;
        }
        return values;
    })();

static boost::multiprecision::int256_t GetScaleMultiplier(int scale) {
    return kLargeScaleMultipliers[scale];
}

// Convert to 256-bit integer from 128-bit decimal.
static boost::multiprecision::int256_t ConvertToInt256(
    arrow::BasicDecimal128 in) {
    boost::multiprecision::int256_t v = in.high_bits();
    v <<= 64;
    v |= in.low_bits();
    return v;
}

// divide input by 10^reduce_by, and round up the fractional part.
static boost::multiprecision::int256_t ReduceScaleBy(
    boost::multiprecision::int256_t in, int32_t reduce_by) {
    if (reduce_by == 0) {
        // nothing to do.
        return in;
    }

    boost::multiprecision::int256_t divisor = GetScaleMultiplier(reduce_by);
    auto result = in / divisor;
    auto remainder = in % divisor;
    // round up (same as BasicDecimal128::ReduceScaleBy)
    if (abs(remainder) >= (divisor >> 1)) {
        result += (in > 0 ? 1 : -1);
    }
    return result;
}

// Convert to 128-bit decimal from 256-bit integer.
// If there is an overflow, the output is undefined.
static arrow::BasicDecimal128 ConvertToDecimal128(
    boost::multiprecision::int256_t in, bool* overflow) {
    arrow::BasicDecimal128 result;
    constexpr boost::multiprecision::int256_t UINT64_MASK =
        std::numeric_limits<uint64_t>::max();

    boost::multiprecision::int256_t in_abs = abs(in);
    bool is_negative = in < 0;

    uint64_t low = (in_abs & UINT64_MASK).convert_to<uint64_t>();
    in_abs >>= 64;
    uint64_t high = (in_abs & UINT64_MASK).convert_to<uint64_t>();
    in_abs >>= 64;

    if (in_abs > 0) {
        // we've shifted in by 128-bit, so nothing should be left.
        *overflow = true;
    } else if (high > INT64_MAX) {
        // the high-bit must not be set (signed 128-bit).
        *overflow = true;
    } else {
        result = arrow::BasicDecimal128(static_cast<int64_t>(high), low);
        if (result > arrow::BasicDecimal128::GetMaxValue()) {
            *overflow = true;
        }
    }
    return is_negative ? -result : result;
}

void gdv_xlarge_multiply_and_scale_down(int64_t x_high, uint64_t x_low,
                                        int64_t y_high, uint64_t y_low,
                                        int32_t reduce_scale_by,
                                        int64_t* out_high, uint64_t* out_low,
                                        bool* overflow) {
    arrow::BasicDecimal128 x{x_high, x_low};
    arrow::BasicDecimal128 y{y_high, y_low};
    auto intermediate_result = ConvertToInt256(x) * ConvertToInt256(y);
    intermediate_result = ReduceScaleBy(intermediate_result, reduce_scale_by);
    auto result = ConvertToDecimal128(intermediate_result, overflow);
    *out_high = result.high_bits();
    *out_low = result.low_bits();
}

// Multiply when the out_precision is 38, and there is no trimming of the scale
// i.e the intermediate value is the same as the final value.
static arrow::BasicDecimal128 MultiplyMaxPrecisionNoScaleDown(
    const arrow::Decimal128Scalar& x, const arrow::Decimal128Scalar& y,
    int32_t out_scale, bool* overflow) {
    arrow::BasicDecimal128 result;
    auto x_abs = arrow::BasicDecimal128::Abs(x.value);
    auto y_abs = arrow::BasicDecimal128::Abs(y.value);

    if (x_abs > arrow::BasicDecimal128::GetMaxValue() / y_abs) {
        *overflow = true;
    } else {
        // We've verified that the result will fit into 128 bits.
        *overflow = false;
        result = x.value * y.value;
    }
    return result;
}

// Multiply when the out_precision is 38, and there is trimming of the scale i.e
// the intermediate value could be larger than the final value.
static arrow::BasicDecimal128 MultiplyMaxPrecisionAndScaleDown(
    const arrow::Decimal128Scalar& x, const arrow::Decimal128Scalar& y,
    int32_t out_scale, bool* overflow) {
    std::shared_ptr<arrow::Decimal128Type> x_type =
        reinterpret_pointer_cast<arrow::Decimal128Type>(x.type);
    std::shared_ptr<arrow::Decimal128Type> y_type =
        reinterpret_pointer_cast<arrow::Decimal128Type>(y.type);
    auto delta_scale = x_type->scale() + y_type->scale() - out_scale;

    *overflow = false;
    arrow::BasicDecimal128 result;
    auto x_abs = arrow::BasicDecimal128::Abs(x.value);
    auto y_abs = arrow::BasicDecimal128::Abs(y.value);

    // It's possible that the intermediate value does not fit in 128-bits, but
    // the final value will (after scaling down).
    bool needs_int256 = false;
    int32_t total_leading_zeros =
        x_abs.CountLeadingBinaryZeros() + y_abs.CountLeadingBinaryZeros();
    // This check is quick, but conservative. In some cases it will indicate
    // that converting to 256 bits is necessary, when it's not actually the
    // case.
    needs_int256 = total_leading_zeros <= 128;
    if (ARROW_PREDICT_FALSE(needs_int256)) {
        int64_t result_high;
        uint64_t result_low;

        // This requires converting to 256-bit, and we use the boost library for
        // that. To avoid references to boost from the precompiled-to-ir code
        // (this causes issues with symbol resolution at runtime), we use a
        // wrapper exported from the CPP code.
        gdv_xlarge_multiply_and_scale_down(
            x.value.high_bits(), x.value.low_bits(), y.value.high_bits(),
            y.value.low_bits(), delta_scale, &result_high, &result_low,
            overflow);
        result = arrow::BasicDecimal128(result_high, result_low);
    } else {
        if (ARROW_PREDICT_TRUE(delta_scale <= 38)) {
            // The largest value that result can have here is (2^64 - 1) * (2^63
            // - 1), which is greater than BasicDecimal128::kMaxValue.
            result = x.value * y.value;
            // Since delta_scale is greater than zero, result can now be at most
            // ((2^64 - 1) * (2^63 - 1)) / 10, which is less than
            // BasicDecimal128::kMaxValue, so there cannot be any overflow.
            result = result.ReduceScaleBy(delta_scale);
        } else {
            // We are multiplying decimal(38, 38) by decimal(38, 38). The result
            // should be a decimal(38, 37), so delta scale = 38 + 38 - 37 = 39.
            // Since we are not in the 256 bit intermediate value case and we
            // are scaling down by 39, then we are guaranteed that the result is
            // 0 (even if we try to round). The largest possible intermediate
            // result is 38 "9"s. If we scale down by 39, the leftmost 9 is now
            // two digits to the right of the rightmost "visible" one. The
            // reason why we have to handle this case separately is because a
            // scale multiplier with a delta_scale 39 does not fit into 128 bit.
            result = 0;
        }
    }
    return result;
}

// Multiply when the out_precision is 38.
static arrow::BasicDecimal128 MultiplyMaxPrecision(
    const arrow::Decimal128Scalar& x, const arrow::Decimal128Scalar& y,
    int32_t out_scale, bool* overflow) {
    std::shared_ptr<arrow::Decimal128Type> x_type =
        reinterpret_pointer_cast<arrow::Decimal128Type>(x.type);
    std::shared_ptr<arrow::Decimal128Type> y_type =
        reinterpret_pointer_cast<arrow::Decimal128Type>(y.type);
    auto delta_scale = x_type->scale() + y_type->scale() - out_scale;
    if (delta_scale == 0) {
        return MultiplyMaxPrecisionNoScaleDown(x, y, out_scale, overflow);
    } else {
        return MultiplyMaxPrecisionAndScaleDown(x, y, out_scale, overflow);
    }
}

arrow::BasicDecimal128 Multiply(const arrow::Decimal128Scalar& x,
                                const arrow::Decimal128Scalar& y,
                                int32_t out_precision, int32_t out_scale,
                                bool* overflow) {
    arrow::BasicDecimal128 result;
    *overflow = false;
    if (out_precision < kMaxPrecision) {
        // fast-path multiply
        result = x.value * y.value;
    } else if (x.value == 0 || y.value == 0) {
        // Handle this separately to avoid divide-by-zero errors.
        result = arrow::BasicDecimal128(0, 0);
    } else {
        result = MultiplyMaxPrecision(x, y, out_scale, overflow);
    }
    return result;
}
}  // namespace decimalops