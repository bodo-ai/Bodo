// Copyright (C) 2023 Bodo Inc. All rights reserved.

#ifndef _WINDOW_COMPUTE_H_INCLUDED
#define _WINDOW_COMPUTE_H_INCLUDED

#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_ftypes.h"

/**
 * @brief Handles the update step for the supported window functions.
 * These functions are not simple reductions and require additional
 * functionality to operate over a "window" of values (possibly a sort
 * or equivalent). The output size is always the same size as the original
 * input
 *
 * @param[in] orderby_arrs The arrays that is being "sorted" to determine
 * the groups. In some situations it may be possible to do a partial sort
 * or avoid sorting.
 * @param[in] window_func The name(s) of the window function(s) being computed.
 * @param[out] out_arr The output array being population.
 * @param[in] grp_info Struct containing information about the groups.
 * @param[in] asc Should the arrays be sorted in ascending order?
 * @param[in] na_pos Should NA's be placed at the end of the arrays?
 * @param[in] is_parallel Is the data distributed? This is used for tracing
 * @param[in] use_sql_rules Do we use SQL or Pandas Null rules?
 */
void window_computation(std::vector<std::shared_ptr<array_info>>& orderby_arrs,
                        std::vector<int64_t> window_funcs,
                        std::vector<std::shared_ptr<array_info>> out_arrs,
                        grouping_info const& grp_info,
                        std::vector<bool>& asc_vect,
                        std::vector<bool>& na_pos_vect,
                        std::vector<void*>& window_args, int n_input_cols,
                        bool is_parallel, bool use_sql_rules);

#endif  // _WINDOW_COMPUTE_H_INCLUDED
