// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_ftypes.h"

/**
 * @brief Helper function to determine the correct ftype and array-type for
 * index column to use during the update step of Min Row-Number Filter.
 *
 * @param n_orderby_arrs Number of order-by columns for the MRNF.
 * @param asc_vec Bitmask specifying the sort direction for the order-by
 * columns.
 * @param na_pos_vec Bitmask specifying whether nulls should be considered
 * 'last' in the order-by columns.
 * @return std::tuple<int64_t, bodo_array_type::arr_type_enum> Tuple of the
 * ftype and array-type for the index column.
 */
std::tuple<int64_t, bodo_array_type::arr_type_enum>
get_update_ftype_idx_arr_type_for_mrnf(size_t n_orderby_arrs,
                                       const std::vector<bool>& asc_vec,
                                       const std::vector<bool>& na_pos_vec);

/**
 * @brief Primary implementation of MRNF.
 * The function updates 'idx_col' in place and writes the index
 * of the output row corresponding to each group.
 * This is used by both the streaming MRNF implementation as well
 * as the non-streaming window implementation.
 * Note that this doesn't make any assumptions about the sorted-ness
 * of the data, i.e. it computes the minimum row per group based on
 * the order-by columns. If the data is known to be already sorted, use
 * the specialized 'min_row_number_filter_window_computation_already_sorted'
 * implementation instead.
 *
 * @param[in, out] idx_col Column with indices of the output rows. This will be
 * updated in place.
 * @param orderby_cols The columns used in the order by clause of the query.
 * @param grp_info Grouping information for the rows in the table.
 * @param asc Bitmask specifying the sort direction for the order-by
 * columns.
 * @param na_pos Bitmask specifying whether nulls should be considered
 * 'last' in the order-by columns.
 * @param update_ftype The ftype to use for update. This is the output
 * from 'get_update_ftype_idx_arr_type_for_mrnf'.
 * @param use_sql_rules Should initialization functions obey SQL semantics?
 * @param pool Memory pool to use for allocations during the execution of
 * this function.
 * @param mm Memory manager associated with the pool.
 */
void min_row_number_filter_no_sort(
    const std::shared_ptr<array_info>& idx_col,
    std::vector<std::shared_ptr<array_info>>& orderby_cols,
    grouping_info const& grp_info, const std::vector<bool>& asc,
    const std::vector<bool>& na_pos, int update_ftype, bool use_sql_rules,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

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
 * @param[out] out_arr The output array(s) being populated.
 * @param[in] grp_info Struct containing information about the groups.
 * @param[in] asc Should the arrays be sorted in ascending order?
 * @param[in] na_pos Should NA's be placed at the end of the arrays?
 * @param is_parallel Is the data distributed? This is used for tracing
 * @param use_sql_rules Do we use SQL or Pandas Null rules?
 * @param pool Memory pool to use for allocations during the execution of
 * this function.
 * @param mm Memory manager associated with the pool.
 */
void window_computation(
    std::vector<std::shared_ptr<array_info>>& orderby_arrs,
    std::vector<int64_t> window_funcs,
    std::vector<std::shared_ptr<array_info>> out_arrs,
    grouping_info const& grp_info, const std::vector<bool>& asc_vect,
    const std::vector<bool>& na_pos_vect, const std::vector<void*>& window_args,
    int n_input_cols, bool is_parallel, bool use_sql_rules,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Handles computing window functions based on the table that
 * has already been sorted. This code path uses sort based computation,
 * so we must pass the all the partition by values as we do not have grouping
 * information.
 *
 * If this computation is being done in parallel on distributed data, then there
 * may be an extra communication step where a rank needs to talk to its
 * neighbor(s) to update its value(s).
 *
 * @param[in] partition_by_arrs The arrays that hold the partition by values.
 * @param[in] order_by_arrs The arrays that hold the order by values.
 * @param[in] window_args The arrays that hold the window argument values.
 * @param[in] window_offset_indices The vector used to associate elements of
 * window_args with the corresponding function call.
 * @param[in] window_funcs The name(s) of the window function(s) being computed.
 * @param[out] out_arrs The output array(s) being populated.
 * @param[in] out_rows the number of rows the output should have
 * @param is_parallel Is the data distributed? This is used for communicating
 * with a neighboring rank for boundary groups.
 */
void sorted_window_computation(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& window_args,
    const std::vector<int32_t>& window_offset_indices,
    const std::vector<int32_t>& window_funcs,
    std::vector<std::shared_ptr<array_info>>& out_arrs, size_t out_rows,
    bool is_parallel);
