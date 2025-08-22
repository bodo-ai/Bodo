#pragma once

#include "../_bodo_common.h"
#include "../_dict_builder.h"
#include "../groupby/_groupby.h"

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
    std::vector<std::shared_ptr<array_info>>& out_arrs,
    std::vector<std::shared_ptr<DictionaryBuilder>>& out_dict_builders,
    grouping_info const& grp_info, const std::vector<bool>& asc_vect,
    const std::vector<bool>& na_pos_vect,
    const std::shared_ptr<table_info> window_args, int n_input_cols,
    bool is_parallel, bool use_sql_rules,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Handles computing a collection of window functions without
 * partition or orderby values. Just computes the single-row result for
 * each window function and relies on the call site to explode the results.
 *
 * @param[in] chunks The chunks of unpinned input data.
 * @param[in] window_funcs The name(s) of the window function(s) being computed.
 * @param[in] window_input_indices The indices of which input columns are inputs
 * to window functions.
 * @param[in] window_offset_indices The offsets of which indices in
 * window_input_indices match up to which window functions.
 * @param[out] out_arrs The collection of output arrays to populate with the
 * single-row answer for each window function.
 * @param is_parallel Is the data distributed?
 */
void global_window_computation(
    std::vector<std::shared_ptr<table_info>>& chunks,
    const std::vector<int32_t>& window_funcs,
    const std::vector<int32_t>& window_input_indices,
    const std::vector<int32_t>& window_offset_indices,
    std::vector<std::shared_ptr<array_info>>& out_arrs, bool is_parallel);
