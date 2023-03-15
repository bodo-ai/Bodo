// Copyright (C) 2023 Bodo Inc. All rights reserved.

#ifndef _GROUPBY_MPI_EXSCAN_H_INCLUDED
#define _GROUPBY_MPI_EXSCAN_H_INCLUDED

#include "_bodo_common.h"
/**
 * This file declares the functions that are used to determine and utilize
 * the MPI_Exscan strategy for groupby. This strategy is used when we have
 * only cumulative operations to avoid shuffling the data.
 */
const int max_global_number_groups_exscan = 1000;

/**
 * @brief Determine the strategy to be used for the computation of the groupby.
 * This computation is done whether a strategy is possible. But it also
   makes heuristic computation in order to reach a decision.
 *
 * @param in_table input table
 * @param num_keys number of keys
 * @param ftypes the type of operations used by groupby
 * @param func_offsets The offsets for the columns used in the functions.
 * @param input_has_index whether input table contains index col in last
   position
 * @return strategy to use :
 * ---0 will use the GroupbyPipeline based on hash partitioning
 * ---1 will use the MPI_Exscan strategy with CATEGORICAL column
 * ---2 will use the MPI_Exscan strategy with determination of the columns
 */
int determine_groupby_strategy(table_info* in_table, int64_t num_keys,
                               int* ftypes, int* func_offsets,
                               bool input_has_index);

/**
 * @brief This function is used to compute the groupby using the MPI_Exscan
 * operations.
 *
 * @param cat_column A categorical column used for tracking entries.
 * @param in_table input table
 * @param num_keys number of keys
 * @param ftypes the type of operations used by groupby
 * @param func_offsets The offsets for the columns used in the functions.
 * @param is_parallel Is the computation parallel or not?
 * @param skipdropna Should we skip dropna?
 * @param return_key Should we return the key?
 * @param return_index Should we return the index?
 * @param use_sql_rules Should we use SQL rules?
 * @return table_info* The table containing the result of the groupby.
 */
table_info* mpi_exscan_computation(array_info* cat_column, table_info* in_table,
                                   int64_t num_keys, int* ftypes,
                                   int* func_offsets, bool is_parallel,
                                   bool skipdropna, bool return_key,
                                   bool return_index, bool use_sql_rules);

/**
 * Create an array info that assigns an index to each unique category.
 * @param in_table : input table
 * @param num_keys : number of keys
 * @param is_parallel: whether we run in parallel or not.
 * @param key_dropna: whether we drop null keys or not.
 * @return key categorical array_info
 */
array_info* compute_categorical_index(table_info* in_table, int64_t num_keys,
                                      bool is_parallel, bool key_dropna = true);

#endif  // _GROUPBY_MPI_EXSCAN_H_INCLUDED
