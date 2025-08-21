#pragma once

#include "../_bodo_common.h"
#include "../_dict_builder.h"
#include "../groupby/_groupby.h"

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
