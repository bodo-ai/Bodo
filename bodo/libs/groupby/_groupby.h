#pragma once
#include "../_bodo_common.h"

/** Data structure used for the computation of groups.

    @data row_to_group       : This takes the index and returns the group
    @data group_to_first_row : This takes the group index and return the first
   row index.
    @data next_row_in_group  : for a row in the list returns the next row in the
   list if existent. if non-existent value is -1.
    @data list_missing       : list of rows which are missing and NaNs.

    This is only one data structure but it has two use cases.
    -- get_group_info computes only the entries row_to_group and
   group_to_first_row. This is the data structure used for groupby operations
   such as sum, mean, etc. for which the full group structure does not need to
   be known.
    -- get_group_info_iterate computes all the entries. This is needed for some
   operations such as nunique, median, and cumulative operations. The entry
   list_missing is computed only for cumulative operations and computed only if
   needed.
 */
struct grouping_info {
    // NOTE: row_to_group[i_row] == -1 means that i_row should be ignored (e.g.
    // due to null keys)
    bodo::vector<int64_t> row_to_group;
    bodo::vector<int64_t> group_to_first_row;
    bodo::vector<int64_t> next_row_in_group;
    bodo::vector<int64_t> list_missing;
    std::shared_ptr<table_info> dispatch_table = nullptr;
    std::shared_ptr<table_info> dispatch_info = nullptr;
    size_t num_groups;
    int mode;  // 1: for the update, 2: for the combine

    /**
     * @brief Construct a new grouping info object.
     *
     * @param pool The memory pool to use for allocating the
     * various vectors.
     */
    grouping_info(bodo::IBufferPool* pool = bodo::BufferPool::DefaultPtr())
        : row_to_group(pool),
          group_to_first_row(pool),
          next_row_in_group(pool),
          list_missing(pool) {}
};

/**
 * This operation groups rows in a distributed table based on keys, and applies
 * a function(s) to a set of columns in each group, producing one output column
 * for each (input column, function) pair. The general algorithm works as
 * follows:
 * a) Group and Update: Each process does the following with its local table:
 *   - Determine to which group each row in the input table belongs to by using
 *     a hash table on the key columns (obtaining a row to group mapping).
 *   - Allocate output table (one row per group -most of the time- or one row
 *     per input row for cumulative operations)
 *   - Initialize output columns (depends on aggregation function)
 *   - Update: apply function to input columns, write result to output (either
 *     directly to output data column or to temporary reduction variable
 *     columns). Uses the row_to_group mapping computed above.
 * b) Shuffle: If the table is distributed, do a parallel shuffle of the
 *    current output table to gather the rows that are part of the same group
 *    on the same process.
 * c) Group and Combine: after the shuffle, a process can end up with multiple
 *    rows belonging to the same group, so we repeat the grouping of a) with
 *    the new (shuffled) table, and apply a possibly different function
 *    ("combine").
 * d) Eval: for functions that required redvar columns, this computes the
 *    final result from the value in the redvar columns and writes it to the
 *    output data columns. This step is only needed for certain functions
 *    like mean, var, std and agg. Redvar columns are deleted afterwards.
 * NOTE: gb.head() is handled in a special case to preserve order in the output.
 * 1. It shuffles data at the beginning and sorts the final output
 *    (and no reverse shuffling).
 * 2. It maintains the same index but unlike cumulative operations, out_table
 *    could have different number of rows from in_table. So, we add index_col
 *    as part of col_sets and perform head operation on it.
 * 3. Per Pandas documentation, order of rows should be preserved. Hence, we
 *    added an extra column that has 0..nrows values and use it to sort
 * cur_table before generating the final out_table. (See GroupbyPipeline::run())
 * 4. If data is replicated, we don't add sort-key column or sort table at the
 * end.
 *
 * @param in_table: input table
 * @param num_keys: number of key columns in the table
 * @param ncols_per_func: number of columns used by each function
 * @param num_funcs: number of functions to apply
 * @param input_has_index: whether the input table has an index column
 * @param ftypes: functions to apply (see Bodo_FTypes::FTypeEnum)
 * @param func_offset: the functions to apply to input col i are in ftypes, in
 * range func_offsets[i] to func_offsets[i+1]
 * @param udf_nredvars[i] is the number of redvar columns needed by udf i
 * @param is_parallel: true if needs to run in parallel (distributed data on
 * multiple processes)
 * @param skip_na_data: whether to drop NaN values or not from the computation
 *                    (dropna for nunique and skipna for median/cumsum/cumprod)
 * @param periods: shift value to use with gb.shift operation.
 * @param transform_funcs: function number(s) to use with transform operation.
 * @param head_n: number of rows to return with gb.head operation.
 * @param return_key: whether to return the keys or not.
 * @param return_index: whether to return the index or not.
 * @param key_dropna: whether to  allow NA values in group keys or not.
 * @param update_cb: external 'update' function (a function pointer).
 *        For ftype=udf, the update step happens in external JIT-compiled code,
 *        which must initialize redvar columns and apply the update function.
 * @param combine_cb: external 'combine' function (a function pointer).
 *        For ftype=udf, external code does the combine step (apply combine
 *        function to current table)
 * @param eval_cb: external 'eval' function (a function pointer).
 *        For ftype=udf, external code does the eval step.
 * @param general_udfs_cb: external function (function pointer) for general
 * UDFs. Does the UDF for all input columns with ftype=gen_udf
 * @param udf_dummy_table: dummy table containing type info for output and
 * redvars columns for udfs
 * @param n_out_rows: the total number of rows in output table, necessary if
 * all the output columns are dead and only number of rows is used.
 * @param window_ascending: For groupby.window is each orderby column asc?
 * @param window_na_position: For groupby.window is each orderby column na last?
 * @param window_args: For groupby.window provides any extra arguments
 * @param n_window_args_per_func: For groupby.window provides offsets
 * to associate each window function call with its scalar arguments
 * @param n_window_args_per_func: For groupby.window provides offsets
 * to associate each window function call with its column arguments
 * @param maintain_input_size: Will the input df and output df have the same
 * length?
 * @param n_shuffle_keys: the number of keys to use when shuffling data across
 * ranks. For example n_shuffle_keys=2 and the keys are [0, 1, 3, 4] then the
 * shuffle_table is done on keys [0, 1].
 * @param use_sql_rules: whether to use SQL rules for group by or Pandas.
 */
table_info* groupby_and_aggregate_py_entry(
    table_info* in_table, int64_t num_keys, int8_t* ncols_per_func,
    int8_t* n_window_calls_per_func, int64_t num_funcs, bool input_has_index,
    int* ftypes, int* func_offsets, int* udf_nredvars, bool is_parallel,
    bool skip_na_data, int64_t periods, int64_t* transform_funcs,
    int64_t head_n, bool return_key, bool return_index, bool key_dropna,
    void* update_cb, void* combine_cb, void* eval_cb, void* general_udfs_cb,
    table_info* udf_dummy_table, int64_t* n_out_rows, bool* window_ascending,
    bool* window_na_position, table_info* window_args_,
    int8_t* n_window_args_per_func, int* n_input_cols_per_func,
    bool maintain_input_size, int64_t n_shuffle_keys, bool use_sql_rules);

/**
 * @brief Get total number of groups for input key arrays
 *
 * @param[in] table a table of all key arrays
 * @param[out] out_labels output array to fill
 * @param[out] sort_idx output array to fill
 * @param key_dropna: whether to include NA in key groups or not.
 * @param is_parallel: true if data is distributed
 * @return int64_t total number of groups
 */
int64_t get_groupby_labels(std::shared_ptr<table_info> table,
                           int64_t* out_labels, int64_t* sort_idx,
                           bool key_dropna, bool is_parallel);

// Python entry point for get_groupby_labels
int64_t get_groupby_labels_py_entry(table_info* table, int64_t* out_labels,
                                    int64_t* sort_idx, bool key_dropna,
                                    bool is_parallel);

/**
 * @brief Copy values from the tmp_col into the update_col. This is used
 * for the eval step of transform.
 *
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 * @param pool Memory pool to use for allocations during the execution of
 * this function.
 * @param mm Memory manager associated with the pool.
 *
 */
void copy_values_transform(
    std::shared_ptr<array_info> update_col, std::shared_ptr<array_info> tmp_col,
    const grouping_info& grp_info, bool is_parallel,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());
