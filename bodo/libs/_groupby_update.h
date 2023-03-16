// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _GROUPBY_UPDATE_H_INCLUDED
#define _GROUPBY_UPDATE_H_INCLUDED

#include "_groupby.h"
#include "_groupby_ftypes.h"

/**
 * The file declares the aggregate functions that are used
 * for the update step of groupby but are too complicated to
 * be inlined.
 */

// Cumulative OPs

/**
 * @brief Perform the update computation for the cumulative operations (e.g.
 * cumsum, cumprod).
 *
 * @param[in] arr The input array.
 * @param[out] out_arr The output array.
 * @param[in] grp_info The grouping information.
 * @param[in] ftype The function type.
 * @param[in] skipna Whether to skip NA values.
 */
void cumulative_computation(array_info* arr, array_info* out_arr,
                            grouping_info const& grp_info, int32_t const& ftype,
                            bool const& skipna);

/**
 * @brief The head_computation function.
 * Copy rows identified by row_list from input to output column
 * @param[in] arr column on which we do the computation
 * @param[out] out_arr output column data
 * @param[in] row_list: row indices to copy
 */
void head_computation(array_info* arr, array_info* out_arr,
                      const std::vector<int64_t>& row_list);

// MEDIAN

/**
 * @brief The median_computation function. It uses the symbolic information to
 * compute the median results.
 *
 * @param[in] arr The input column on which we do the computation
 * @param[out] out_arr The output column'
 * @param[in] grp_info: The grouping information.
 * @param[in] skipna: Whether to skip NA values.
 * @param[in] use_sql_rules: Should allocation use SQL rules.
 */
void median_computation(array_info* arr, array_info* out_arr,
                        grouping_info const& grp_info, bool const& skipna,
                        bool const use_sql_rules);

// SHIFT

/**
 * @brief The shift_computation function.
 * Shift rows per group N times (up or down).
 * @param[in] arr column on which we do the computation
 * @param[out] out_arr column data after being shifted
 * @param[in] grp_info: grouping_info about groups and rows organization
 * @param[in] periods: Number of periods to shift
 */
void shift_computation(array_info* arr, array_info* out_arr,
                       grouping_info const& grp_info, int64_t const& periods);

#endif  // _GROUPBY_UPDATE_H_INCLUDED
