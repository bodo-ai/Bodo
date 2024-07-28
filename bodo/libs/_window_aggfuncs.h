// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_ftypes.h"

template <int ftype>
concept size = ftype == Bodo_FTypes::size;

template <int ftype>
concept count = ftype == Bodo_FTypes::count;

template <int ftype>
concept count_if = ftype == Bodo_FTypes::count_if;

template <int ftype>
concept count_fns = size<ftype> || count<ftype> || count_if<ftype>;

template <int ftype>
concept first = ftype == Bodo_FTypes::first;

template <int ftype>
concept last = ftype == Bodo_FTypes::last;

template <int ftype>
concept var = ftype == Bodo_FTypes::var;

template <int ftype>
concept var_pop = ftype == Bodo_FTypes::var_pop;

template <int ftype>
concept stddev = ftype == Bodo_FTypes::std;

template <int ftype>
concept stddev_pop = ftype == Bodo_FTypes::std_pop;

template <int ftype>
concept mean = ftype == Bodo_FTypes::mean;

template <int ftype>
concept ratio_to_report = ftype == Bodo_FTypes::ratio_to_report;

template <int ftype>
concept any_value = ftype == Bodo_FTypes::any_value;

template <int ftype>
concept row_number = ftype == Bodo_FTypes::row_number;

template <int ftype>
concept rank = ftype == Bodo_FTypes::rank;

template <int ftype>
concept dense_rank = ftype == Bodo_FTypes::dense_rank;

template <int ftype>
concept percent_rank = ftype == Bodo_FTypes::percent_rank;

template <int ftype>
concept cume_dist = ftype == Bodo_FTypes::cume_dist;

// Enums to specify which type of window frame is being computer, in case
// a window function should have different implementaitons for some of
// the formats (e.g. min/max can accumulate a running min/max for
// cumulative/no_frame, but needs to do a full slice aggregation
// on sliding frames)
enum window_frame_enum { CUMULATIVE, SLIDING, NO_FRAME };

/**
 * @brief Computes a window aggregation by casing on the array type
 * and forwarding  to the helper function with the array type templated.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_groups: the sorted array of group numbers indicating
 * which partition the current row belongs to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 * @param[in] frame_lo: Pointer to the lower bound value (nullptr if
 * the lower bound is UNBOUNDED PRECEDING)
 * @param[in] frame_hi: Pointer to the upper bound value (nullptr if
 * the upper bound is UNBOUNDED FOLLOWING)
 *
 * See window_frame_computation_bounds_handler for more details on the
 * interpretation of frame_lo / frame_hi
 */
void window_frame_computation(std::shared_ptr<array_info> in_arr,
                              std::shared_ptr<array_info> out_arr,
                              std::shared_ptr<array_info> sorted_groups,
                              std::shared_ptr<array_info> sorted_idx,
                              int64_t* frame_lo, int64_t* frame_hi, int ftype);
