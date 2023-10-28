// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

/**
 * This files contains the functions used for the eval step of
 * hashed based groupby. This is the a step that converts the
 * single row output from the reductions done across all ranks
 * into the final output format.
 */

/**
 * Final evaluation step for mean, which calculates the mean based on the
 * sum of observed values and the number of values.
 *
 * @param[in,out] sum of observed values, will be modified to contain the mean
 * @param count: number of observations
 */
inline void mean_eval(double& result, uint64_t count) { result /= count; }

/**
 * Perform final evaluation step for population std, which calculates the
 * standard deviation based on the count and m2 values. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param[in,out] stores the calculated std
 * @param count: number of observations
 * @param m2: sum of squares of differences from the current mean
 */
inline void std_pop_eval(double& result, uint64_t& count, double& m2) {
    if (count == 0) {
        result = std::numeric_limits<double>::quiet_NaN();
    } else {
        result = sqrt(m2 / count);
    }
}

/**
 * Perform final evaluation step for population variance, which calculates the
 * variance based on the count and m2 values. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param[in,out] stores the calculated variance
 * @param count: number of observations
 * @param m2: sum of squares of differences from the current mean
 */
inline void var_pop_eval(double& result, uint64_t& count, double& m2) {
    if (count == 0) {
        result = std::numeric_limits<double>::quiet_NaN();
    } else {
        result = m2 / count;
    }
}

/**
 * Perform final evaluation step for std, which calculates the standard
 * deviation based on the count and m2 values. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param[in,out] stores the calculated std
 * @param count: number of observations
 * @param m2: sum of squares of differences from the current mean
 */
inline void std_eval(double& result, uint64_t count, double m2) {
    if (count <= 1) {
        result = std::numeric_limits<double>::quiet_NaN();
    } else {
        result = sqrt(m2 / (count - 1));
    }
}

/**
 * Perform final evaluation step for variance, which calculates the variance
 * based on the count and m2 values. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param[in,out] stores the calculated variance
 * @param count: number of observations
 * @param m2: sum of squares of differences from the current mean
 */
inline void var_eval(double& result, uint64_t count, double m2) {
    if (count <= 1) {
        result = std::numeric_limits<double>::quiet_NaN();
    } else {
        result = m2 / (count - 1);
    }
}

/**
 * Perform final evaluation step for skew, which calculates the skew
 * based on the count and the precursor calculations to the first three moments.
 * See https://en.wikipedia.org/wiki/Skew
 * for more information. Precise formula taken from
 * https://github.com/pandas-dev/pandas/blob/7187e675002fe88e639b8c9c62a8625a1dd1235b/pandas/core/nanops.py
 *
 * @param[in,out] stores the calculated skew
 * @param count: number of observations
 * @param m1: the sum of elmements (precursor to the first moment)
 * @param m2: the sum of squares of elements (precursor to the second moment)
 * @param m3: the sum of cubes of elements (precursor to the third moment)
 */
inline void skew_eval(double& result, uint64_t count, double m1, double m2,
                      double m3) {
    if (count < 3) {
        result = std::numeric_limits<double>::quiet_NaN();
    } else {
        double sum = m1 / count;
        double numerator =
            m3 - 3.0 * m2 * sum + 2.0 * count * std::pow(sum, 3.0);
        double denominator = std::pow(m2 - sum * m1, 1.5);
        // If the numerator is zero, or the denominator is very close to zero,
        // then the skew should be set to zero to avoid floating point
        // arithmetic errors. These constants were derived from trial and error
        // to correctly flag values that should be zero without causing any of
        // the other tests to be falsely flagged just because they are near
        // zero.
        if (numerator == 0.0 || std::abs(denominator) < std::pow(10.0, -14.0) ||
            isnan(denominator) ||
            std::log2(std::abs(denominator)) - std::log2(std::abs(numerator)) <
                -20) {
            result = 0.0;
        } else {
            double s = ((count * std::pow((count - 1), 1.5) / (count - 2)) *
                        numerator / denominator);
            result = s / (count - 1);
        }
    }
}

/**
 * Perform final evaluation step for kurtosis, which calculates the kurtosis
 * based on the count and the precursor calculations to the first four moments.
 * See https://en.wikipedia.org/wiki/Kurtosis
 * for more information. Precise formula taken from
 * https://github.com/pandas-dev/pandas/blob/7187e675002fe88e639b8c9c62a8625a1dd1235b/pandas/core/nanops.py
 *
 * @param[in,out] stores the calculated kurtosis
 * @param count: number of observations
 * @param m1: the sum of elmements (precursor to the first moment)
 * @param m2: the sum of squares of elements (precursor to the second moment)
 * @param m3: the sum of cubes of elements (precursor to the third moment)
 * @param m4: the sum of fourths of elements (precursor to the fourth moment)
 */
inline void kurt_eval(double& result, uint64_t count, double m1, double m2,
                      double m3, double m4) {
    if (count < 4) {
        result = std::numeric_limits<double>::quiet_NaN();
    } else {
        double sum = m1 / count;
        double fm = (m4 - 4 * m3 * sum + 6 * m2 * std::pow(sum, 2.0) -
                     3 * count * std::pow(sum, 4.0));
        double sm = m2 - sum * m1;
        double numerator = count * (count + 1) * (count - 1) * fm;
        double denominator = (count - 2) * (count - 3) * std::pow(sm, 2.0);
        // If the numerator is zero, or the denominator is very close to zero,
        // then the kurtosis should be set to zero to avoid floating point
        // arithmetic errors. These constants were derived from trial and error
        // to correctly flag values that should be zero without causing any of
        // the other tests to be falsely flagged just because they are near
        // zero.
        if (numerator == 0.0 || std::abs(denominator) < std::pow(10.0, -14.0) ||
            isnan(denominator) ||
            std::log2(std::abs(denominator)) - std::log2(std::abs(numerator)) <
                -20) {
            result = 0.0;
        } else {
            double adj =
                3 * std::pow(count - 1, 2.0) / ((count - 2) * (count - 3));
            double s = (count - 1) * (numerator / denominator - adj);
            result = s / (count - 1);
        }
    }
}

/**
 * Perform final evaluation step for boolxor_agg, which returns outputs if
 * exactly one element is nonzero, or NULL if all of the elements are NULL
 *
 * @param[in, out] one: true if 1+ entries are non-zero, later becomes the
 * output column
 * @param[in] two: true if 2+ entries are non-zero
 * @param i: the index that the evaluation is being done on
 */
inline void boolxor_eval(const std::shared_ptr<array_info>& one,
                         const std::shared_ptr<array_info>& two, int64_t i) {
    bool one_bit = GetBit((uint8_t*)one->data1(), i);
    bool two_bit = GetBit((uint8_t*)two->data1(), i);
    SetBitTo((uint8_t*)one->data1(), i, one_bit && !two_bit);
}
