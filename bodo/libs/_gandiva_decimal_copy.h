// Copyright (C) 2024 Bodo Inc. All rights reserved.
// https://github.com/apache/arrow/blob/299eb26e8c22b4aad4876c9e3b52f9adde699a5c/cpp/src/gandiva/precompiled/decimal_ops.cc#L331
// Code heavily copied from the gandiva precompiled code
// to avoid importing the library.
// Usages of arrow::Decimal128Scalar have been replaced with
// arrow::Decimal128 to reduce allocation overheads.
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

static inline boost::multiprecision::int256_t GetScaleMultiplier(
    int scale) noexcept {
    return kLargeScaleMultipliers[scale];
}

// Convert to 256-bit integer from 128-bit decimal.
static inline boost::multiprecision::int256_t ConvertToInt256(
    arrow::BasicDecimal128 in) noexcept {
    boost::multiprecision::int256_t v = in.high_bits();
    v <<= 64;
    v |= in.low_bits();
    return v;
}

// divide input by 10^reduce_by, and round up the fractional part.
static inline boost::multiprecision::int256_t ReduceScaleBy(
    boost::multiprecision::int256_t in, int32_t reduce_by) noexcept {
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
static inline arrow::BasicDecimal128 ConvertToDecimal128(
    boost::multiprecision::int256_t in, bool* overflow) noexcept {
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

// Bodo change: Added 'inline'
inline void gdv_xlarge_multiply_and_scale_down(int64_t x_high, uint64_t x_low,
                                               int64_t y_high, uint64_t y_low,
                                               int32_t reduce_scale_by,
                                               int64_t* out_high,
                                               uint64_t* out_low,
                                               bool* overflow) noexcept {
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
static inline void MultiplyMaxPrecisionNoScaleDown(
    const arrow::Decimal128& x, const arrow::Decimal128& y, int32_t out_scale,
    bool* overflow, arrow::Decimal128* result) noexcept {
    auto x_abs = arrow::BasicDecimal128::Abs(x);
    auto y_abs = arrow::BasicDecimal128::Abs(y);

    if (x_abs > arrow::BasicDecimal128::GetMaxValue() / y_abs) {
        *overflow = true;
    } else {
        // We've verified that the result will fit into 128 bits.
        *overflow = false;
        *result = x * y;
    }
}

// Multiply when the out_precision is 38, and there is trimming of the scale i.e
// the intermediate value could be larger than the final value.
static inline void MultiplyMaxPrecisionAndScaleDown(
    const arrow::Decimal128& x, const arrow::Decimal128& y, int32_t delta_scale,
    bool* overflow, arrow::Decimal128* result) noexcept {
    *overflow = false;
    auto x_abs = arrow::BasicDecimal128::Abs(x);
    auto y_abs = arrow::BasicDecimal128::Abs(y);

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
            x.high_bits(), x.low_bits(), y.high_bits(), y.low_bits(),
            delta_scale, &result_high, &result_low, overflow);
        *result = arrow::BasicDecimal128(result_high, result_low);
    } else {
        // Note: The delta scale is always < 38, so this path is always safe.
        // The largest value that result can have here is (2^64 - 1) * (2^63
        // - 1), which is greater than BasicDecimal128::kMaxValue.
        *result = x * y;
        // Since delta_scale is greater than zero, result can now be at most
        // ((2^64 - 1) * (2^63 - 1)) / 10, which is less than
        // BasicDecimal128::kMaxValue, so there cannot be any overflow.
        *result = result->ReduceScaleBy(delta_scale);
    }
}

// Multiply when the out_precision is 38.
template <bool rescale>
inline void MultiplyMaxPrecision(const arrow::Decimal128& x,
                                 const arrow::Decimal128& y, int32_t out_scale,
                                 int32_t delta_scale, bool* overflow,
                                 arrow::Decimal128* result) noexcept {
    if (rescale) {
        MultiplyMaxPrecisionAndScaleDown(x, y, delta_scale, overflow, result);
    } else {
        MultiplyMaxPrecisionNoScaleDown(x, y, out_scale, overflow, result);
    }
}

/**
 * @brief Multiply two decimal values and place the result in result.
 *
 * @param x The first input.
 * @param y The second input.
 * @param x_scale The scale of x.
 * @param y_scale The scale of y.
 * @param out_precision The output precision.
 * @param out_scale The output scale.
 * @param overflow[out] Did overflow occur.
 * @param result[out] The result.
 */
template <bool fast_multiply, bool rescale>
inline void Multiply(const arrow::Decimal128& x, const arrow::Decimal128& y,
                     int32_t out_scale, int32_t delta_scale, bool* overflow,
                     arrow::Decimal128* result) noexcept {
    *overflow = false;
    if (fast_multiply) {
        // fast-path multiply
        *result = x * y;
    } else if (x == 0 || y == 0) {
        // Handle this separately to avoid divide-by-zero errors.
        *result = arrow::BasicDecimal128(0, 0);
    } else {
        MultiplyMaxPrecision<rescale>(x, y, out_scale, delta_scale, overflow,
                                      result);
    }
}

// Suppose we have a number that requires x bits to be represented and we scale
// it up by 10^scale_by. Let's say now y bits are required to represent it. This
// function returns the maximum possible y - x for a given 'scale_by'.
inline int32_t MaxBitsRequiredIncreaseAfterScaling(int32_t scale_by) {
    // We rely on the following formula:
    // bits_required(x * 10^y) <= bits_required(x) + floor(log2(10^y)) + 1
    // We precompute floor(log2(10^x)) + 1 for x = 0, 1, 2...75, 76
    assert(scale_by >= 0);
    assert(scale_by <= 76);
    static const int32_t floor_log2_plus_one[] = {
        0,   4,   7,   10,  14,  17,  20,  24,  27,  30,  34,  37,  40,
        44,  47,  50,  54,  57,  60,  64,  67,  70,  74,  77,  80,  84,
        87,  90,  94,  97,  100, 103, 107, 110, 113, 117, 120, 123, 127,
        130, 133, 137, 140, 143, 147, 150, 153, 157, 160, 163, 167, 170,
        173, 177, 180, 183, 187, 190, 193, 196, 200, 203, 206, 210, 213,
        216, 220, 223, 226, 230, 233, 236, 240, 243, 246, 250, 253};
    return floor_log2_plus_one[scale_by];
}

// Returns the maximum possible number of bits required to represent num *
// 10^scale_by.
inline int32_t MaxBitsRequiredAfterScaling(const arrow::Decimal128& value,
                                           int32_t scale_by) {
    arrow::BasicDecimal128 value_abs = arrow::BasicDecimal128::Abs(value);
    assert(scale_by >= 0);
    assert(scale_by <= 76);
    int32_t num_occupied = 128 - value_abs.CountLeadingBinaryZeros();
    return num_occupied + MaxBitsRequiredIncreaseAfterScaling(scale_by);
}

static inline arrow::BasicDecimal128 CheckAndIncreaseScale(
    const arrow::BasicDecimal128& in, int32_t delta) {
    return (delta <= 0) ? in : in.IncreaseScaleBy(delta);
}

// multiply input by 10^increase_by.
static inline boost::multiprecision::int256_t IncreaseScaleBy(
    boost::multiprecision::int256_t in, int32_t increase_by) {
    assert(increase_by >= 0);
    assert(increase_by <= 2 * kMaxPrecision);

    return in * GetScaleMultiplier(increase_by);
}

inline void gdv_xlarge_scale_up_and_divide(int64_t x_high, uint64_t x_low,
                                           int64_t y_high, uint64_t y_low,
                                           int32_t increase_scale_by,
                                           int64_t* out_high, uint64_t* out_low,
                                           bool* overflow) {
    arrow::BasicDecimal128 x{x_high, x_low};
    arrow::BasicDecimal128 y{y_high, y_low};

    boost::multiprecision::int256_t x_large = ConvertToInt256(x);
    boost::multiprecision::int256_t x_large_scaled_up =
        IncreaseScaleBy(x_large, increase_scale_by);
    boost::multiprecision::int256_t y_large = ConvertToInt256(y);
    boost::multiprecision::int256_t result_large = x_large_scaled_up / y_large;
    boost::multiprecision::int256_t remainder_large =
        x_large_scaled_up % y_large;

    // Since we are scaling up and then, scaling down, round-up the result (+1
    // for +ve, -1 for -ve), if the remainder is >= 2 * divisor.
    if (abs(2 * remainder_large) >= abs(y_large)) {
        // x +ve and y +ve, result is +ve =>   (1 ^ 1)  + 1 =  0 + 1 = +1
        // x +ve and y -ve, result is -ve =>  (-1 ^ 1)  + 1 = -2 + 1 = -1
        // x +ve and y -ve, result is -ve =>   (1 ^ -1) + 1 = -2 + 1 = -1
        // x -ve and y -ve, result is +ve =>  (-1 ^ -1) + 1 =  0 + 1 = +1
        result_large += (x.Sign() ^ y.Sign()) + 1;
    }
    auto result = ConvertToDecimal128(result_large, overflow);
    *out_high = result.high_bits();
    *out_low = result.low_bits();
}

inline arrow::Decimal128 Divide(const arrow::Decimal128& x,
                                const arrow::Decimal128& y, int32_t delta_scale,
                                bool* overflow) {
    if (y == 0) {
        throw std::runtime_error("Decimal division by zero error");
    }

    arrow::BasicDecimal128 result;
    int32_t num_bits_required_after_scaling =
        MaxBitsRequiredAfterScaling(x, delta_scale);
    if (num_bits_required_after_scaling <= 127) {
        // fast-path. The dividend fits in 128-bit after scaling too.
        *overflow = false;

        // do the division.
        arrow::BasicDecimal128 x_scaled = CheckAndIncreaseScale(x, delta_scale);
        arrow::BasicDecimal128 remainder;
        arrow::DecimalStatus status = x_scaled.Divide(y, &result, &remainder);
        if (status != arrow::DecimalStatus::kSuccess) {
            throw std::runtime_error("Decimal division failed");
        }

        // round-up
        if (arrow::BasicDecimal128::Abs(2 * remainder) >=
            arrow::BasicDecimal128::Abs(y)) {
            result += (x.Sign() ^ y.Sign()) + 1;
        }
    } else {
        // convert to 256-bit and do the divide.
        *overflow = delta_scale > 38 && num_bits_required_after_scaling > 255;
        if (!*overflow) {
            int64_t result_high;
            uint64_t result_low;

            gdv_xlarge_scale_up_and_divide(
                x.high_bits(), x.low_bits(), y.high_bits(), y.low_bits(),
                delta_scale, &result_high, &result_low, overflow);
            result = arrow::BasicDecimal128(result_high, result_low);
        }
    }
    return result;
}

}  // namespace decimalops
