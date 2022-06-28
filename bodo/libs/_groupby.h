// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _GROUPBY_H_INCLUDED
#define _GROUPBY_H_INCLUDED
#include "_bodo_common.h"

const int max_global_number_groups_exscan = 1000;

/// Initialize C++ groupby module
void groupby_init();

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
 * @param input_has_index:
 * @param ftypes: functions to apply (see Bodo_FTypes::FTypeEnum)
 * @param func_offset: the functions to apply to input col i are in ftypes, in
 * range func_offsets[i] to func_offsets[i+1]
 * @param udf_nredvars[i] is the number of redvar columns needed by udf i
 * @param is_parallel: true if needs to run in parallel (distributed data on
 * multiple processes)
 * @param skipdropna: whether to drop NaN values or not from the computation
 *                    (dropna for nunique and skipna for median/cumsum/cumprod)
 * @param periods: shift value to use with gb.shift operation.
 * @param transform_func: function number to use with transform operation.
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
 */
table_info* groupby_and_aggregate(
    table_info* in_table, int64_t num_keys, bool input_has_index, int* ftypes,
    int* func_offsets, int* udf_nredvars, bool is_parallel, bool skipdropna,
    int64_t periods, int64_t transform_func, int64_t head_n, bool return_key,
    bool return_index, bool key_dropna, void* update_cb, void* combine_cb,
    void* eval_cb, void* general_udfs_cb, table_info* udf_dummy_table);

table_info* pivot_groupby_and_aggregate(
    table_info* in_table, int64_t num_keys, table_info* dispatch_table,
    table_info* dispatch_info, bool input_has_index, int* ftypes,
    int* func_offsets, int* udf_nredvars, bool is_parallel, bool is_crosstab,
    bool skipdropna, bool return_key, bool return_index, void* update_cb,
    void* combine_cb, void* eval_cb, table_info* udf_dummy_table);

/**
 * @brief Get total number of groups for input key arrays
 *
 * @param table a table of all key arrays
 * @param out_labels output array to fill
 * @param sort_idx output array to fill
 * @param key_dropna: whether to include NA in key groups or not.
 * @param is_parallel: true if data is distributed
 * @return int64_t total number of groups
 */
int64_t get_groupby_labels(table_info* table, int64_t* out_labels,
                           int64_t* sort_idx, bool key_dropna,
                           bool is_parallel);

#endif  // _GROUPBY_H_INCLUDED
