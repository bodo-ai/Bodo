// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_bodo_common.h"

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
 *
 * @param input table
 * @param number of key columns in the table
 * @param functions to apply (see Bodo_FTypes::FTypeEnum)
 * @param the functions to apply to input col i are in ftypes, in range
 *        func_offsets[i] to func_offsets[i+1]
 * @param udf_nredvars[i] is the number of redvar columns needed by udf i
 * @param true if needs to run in parallel (distributed data on multiple
 *        processes)
 * @param skipdropna: whether to drop NaN values or not from the computation
 *                    (dropna for nunique and skipna for median/cumsum/cumprod)
 * @param return_key: whether to return the keys or not.
 * @param return_index: whether to return the index or not.
 * @param external 'update' function (a function pointer).
 *        For ftype=udf, the update step happens in external JIT-compiled code,
 *        which must initialize redvar columns and apply the update function.
 * @param external 'combine' function (a function pointer).
 *        For ftype=udf, external code does the combine step (apply combine
 *        function to current table)
 * @param external 'eval' function (a function pointer).
 *        For ftype=udf, external code does the eval step.
 * @param dummy table containing type info for output and redvars columns for
 *        udfs
 */
table_info* groupby_and_aggregate(table_info* in_table, int64_t num_keys,
                                  int* ftypes, int* func_offsets,
                                  int* udf_nredvars, bool is_parallel,
                                  bool skipdropna, bool return_key,
                                  bool return_index, void* update_cb,
                                  void* combine_cb, void* eval_cb,
                                  table_info* udf_dummy_table);
