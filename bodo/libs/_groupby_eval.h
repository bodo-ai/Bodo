// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _GROUPBY_EVAL_H_INCLUDED
#define _GROUPBY_EVAL_H_INCLUDED

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
inline void mean_eval(double& result, uint64_t& count) { result /= count; }

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
inline void std_eval(double& result, uint64_t& count, double& m2) {
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
inline void var_eval(double& result, uint64_t& count, double& m2) {
    if (count <= 1) {
        result = std::numeric_limits<double>::quiet_NaN();
    } else {
        result = m2 / (count - 1);
    }
}

#endif  // _GROUPBY_EVAL_H_INCLUDED
