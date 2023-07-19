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

// Combine mapping

/**
 * @brief Get the combine function for a given update
 * function. Combine is used after we have already done
 * a local reduction.
 *
 * @param update_ftype The update function type.
 * @return The combine function type.
 */
int get_combine_func(int update_ftype);

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
void cumulative_computation(std::shared_ptr<array_info> arr,
                            std::shared_ptr<array_info> out_arr,
                            grouping_info const& grp_info, int32_t const& ftype,
                            bool const& skipna);

// HEAD

/**
 * @brief The head_computation function.
 * Copy rows identified by row_list from input to output column
 * @param[in] arr column on which we do the computation
 * @param[out] out_arr output column data
 * @param[in] row_list: row indices to copy
 */
void head_computation(std::shared_ptr<array_info> arr,
                      std::shared_ptr<array_info> out_arr,
                      const bodo::vector<int64_t>& row_list);

// NGROUP

/**
 * @brief ngroup assigns the same group number to each row in the group.
 * If data is replicated, start from 0 and the group number will be the output
 * value for all rows in that group. If data is distributed, we need to identify
 * starting group number on each rank. Then, row's output is: start group number
 * + row's local group number (igrp in current rank) This is done by summing
 * number of groups of ranks before current rank. This is achieved with
 * MPI_Exscan: partial reduction excluding current rank value.
 * @param[in] arr The input column on which we do the computation
 * @param[out] out_arr The output column which contains ngroup results
 * @param[in] grp_info grouping_info about groups and rows organization per rank
 * @param[in] is_parallel: true if data is distributed (used to indicate whether
 * we need to do cumsum on group numbers or not)
 */
void ngroup_computation(std::shared_ptr<array_info> arr,
                        std::shared_ptr<array_info> out_arr,
                        grouping_info const& grp_info, bool is_parallel);

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
void median_computation(std::shared_ptr<array_info> arr,
                        std::shared_ptr<array_info> out_arr,
                        grouping_info const& grp_info, bool const& skipna,
                        bool const use_sql_rules);

// MODE

/**
 * @brief The mode computation function. It uses the symbolic information to
 * compute the mode results.
 *
 * @param[in] arr The input column on which we do the computation.
 * @param[out] out_arr The output column.
 * @param[in] grp_info: The grouping information.
 * @param[in] is_parallel: Is the operation being done in parallel.
 * @param[in] use_sql_rules: Should allocation use SQL rules.
 */
void mode_computation(std::shared_ptr<array_info> arr,
                      std::shared_ptr<array_info> out_arr,
                      const grouping_info& grp_info, const bool is_parallel,
                      const bool use_sql_rules);

// SHIFT

/**
 * @brief The shift_computation function.
 * Shift rows per group N times (up or down).
 * @param[in] arr column on which we do the computation
 * @param[out] out_arr column data after being shifted
 * @param[in] grp_info: grouping_info about groups and rows organization
 * @param[in] periods: Number of periods to shift
 */
void shift_computation(std::shared_ptr<array_info> arr,
                       std::shared_ptr<array_info> out_arr,
                       grouping_info const& grp_info, int64_t const& periods);

// Skew

/**
 * @brief Compute the skew update function for combining the result
 * of local reductions on each rank.
 *
 * @param[in] count_col_in The count input column.
 * @param[in] m1_col_in The first moment input column.
 * @param[in] m2_col_in The second moment input column.
 * @param[in] m3_col_in The third moment input column.
 * @param[out] count_col_out The count output column.
 * @param[out] m1_col_out The first moment output column.
 * @param[out] m2_col_out The second moment output column.
 * @param[out] m3_col_out The second moment output column.
 * @param[in] grp_info The grouping information.
 */
void skew_combine(const std::shared_ptr<array_info>& count_col_in,
                  const std::shared_ptr<array_info>& m1_col_in,
                  const std::shared_ptr<array_info>& m2_col_in,
                  const std::shared_ptr<array_info>& m3_col_in,
                  const std::shared_ptr<array_info>& count_col_out,
                  const std::shared_ptr<array_info>& m1_col_out,
                  const std::shared_ptr<array_info>& m2_col_out,
                  const std::shared_ptr<array_info>& m3_col_out,
                  grouping_info const& grp_info);

// Kurtosis

/**
 * @brief Compute the kurtosis update function for combining the result
 * of local reductions on each rank.
 *
 * @param[in] count_col_in The count input column.
 * @param[in] m1_col_in The first moment input column.
 * @param[in] m2_col_in The second moment input column.
 * @param[in] m3_col_in The third moment input column.
 * @param[out] count_col_out The count output column.
 * @param[out] m1_col_out The first moment output column.
 * @param[out] m2_col_out The second moment output column.
 * @param[out] m3_col_out The second moment output column.
 * @param[in] grp_info The grouping information.
 */
void kurt_combine(const std::shared_ptr<array_info>& count_col_in,
                  const std::shared_ptr<array_info>& m1_col_in,
                  const std::shared_ptr<array_info>& m2_col_in,
                  const std::shared_ptr<array_info>& m3_col_in,
                  const std::shared_ptr<array_info>& m4_col_in,
                  const std::shared_ptr<array_info>& count_col_out,
                  const std::shared_ptr<array_info>& m1_col_out,
                  const std::shared_ptr<array_info>& m2_col_out,
                  const std::shared_ptr<array_info>& m3_col_out,
                  const std::shared_ptr<array_info>& m4_col_out,
                  grouping_info const& grp_info);

// Variance

/**
 * @brief Compute the variance update function for combining the result
 * of local reductions on each rank.
 *
 * @param[in] count_col_in The count input column.
 * @param[in] mean_col_in The mean input column.
 * @param[in] m2_col_in The mean^2 input column.
 * @param[out] count_col_out The count output column.
 * @param[out] mean_col_out The mean output column.
 * @param[out] m2_col_out The mean^2 output column.
 * @param[in] grp_info The grouping information.
 */
void var_combine(const std::shared_ptr<array_info>& count_col_in,
                 const std::shared_ptr<array_info>& mean_col_in,
                 const std::shared_ptr<array_info>& m2_col_in,
                 const std::shared_ptr<array_info>& count_col_out,
                 const std::shared_ptr<array_info>& mean_col_out,
                 const std::shared_ptr<array_info>& m2_col_out,
                 grouping_info const& grp_info);

// Boolxor

/**
 * @brief Compute the boolxor_agg update function for combining
 * the result of local reductions on each rank.
 *
 * @param[in] one_col_in The input column for if there is 1+ non-zero entries
 * @param[in] two_col_in The input column for if there are 2+ non-zero entries
 * @param[out] one_col_out The output column for if there is 1+ non-zero entries
 * @param[out] two_col_out The output column for if there are 2+ non-zero
 * entries
 * @param[in] grp_info The grouping information.
 */
void boolxor_combine(const std::shared_ptr<array_info>& one_col_in,
                     const std::shared_ptr<array_info>& two_col_in,
                     const std::shared_ptr<array_info>& one_col_out,
                     const std::shared_ptr<array_info>& two_col_out,
                     grouping_info const& grp_info);

// NUNIQUE

/**
 * The nunique_computation function. It uses the symbolic information to compute
 * the nunique results.
 *
 * @param arr The column on which we do the computation
 * @param out_arr[out] The column which contains nunique results
 * @param grp_info The array containing information on how the rows are
 * organized
 * @param dropna The boolean dropna indicating whether we drop or not the NaN
 * values from the nunique computation.
 * @param is_parallel: true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 */
void nunique_computation(std::shared_ptr<array_info> arr,
                         std::shared_ptr<array_info> out_arr,
                         grouping_info const& grp_info, bool const& dropna,
                         bool const& is_parallel);

#endif  // _GROUPBY_UPDATE_H_INCLUDED
