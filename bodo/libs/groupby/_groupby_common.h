#pragma once

#include "../_bodo_common.h"
#include "_groupby.h"

/**
 * This file defines helper functions that are shared by multiple possible
 * groupby paths.
 */

/**
 * @brief Initialize the output column for the groupby operation
 * based on the type of the function and if we are using SQL rules.
 *
 * @param[in, out] out_col The array to initialize
 * @param[in] ftype The function type
 * @param[in] use_sql_rules Are we using SQL rules?
 */
void aggfunc_output_initialize(const std::shared_ptr<array_info>& out_col,
                               int ftype, bool use_sql_rules,
                               int64_t start_row = 0);

/**
 * @brief Returns the array type and dtype required for output columns based on
 * the aggregation function and input dtype.
 *
 * @param ftype Function type
 * @param array type (caller sets a default, this function only changes
 * in certain cases)
 * @param output dtype (caller sets a default, this function only
 * changes in certain cases)
 */
std::tuple<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>
get_groupby_output_dtype(int ftype, bodo_array_type::arr_type_enum array_type,
                         Bodo_CTypes::CTypeEnum dtype);

/**
 * Allocate and fill key columns, based on grouping info. It uses the
 * values of key columns from from_table to populate out_table.
 */
void alloc_init_keys(
    const std::vector<std::shared_ptr<table_info>>& from_tables,
    const std::shared_ptr<table_info>& out_table,
    const std::vector<grouping_info>& grp_infos, int64_t num_keys,
    size_t num_groups,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Sorts the local data for an aggregation that has shuffled
 * before aggregation first by the group and then by any orderby
 * columns.
 *
 * @param[in] group_info: The struct associating each row to its group.
 * @param[in] orderby_cols: The columns to sort by.
 * @param[in] extra_cols: Any additional columns to sort along with the ordering
 * columns.
 * @param[in] asc_vect: Which columns in orderby_cols should be ascending?
 * @param[in] na_pos_vect: Which columns in orderby_cols should be nulls
 * first/last?
 * @param[in] order_offset: What index in asc_vect/na_pos_vect corresponds to
 * the first entry in orderby_cols, in case there is an offset.
 * @param[in] is_parallel: is the operation happening in parallel
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 * @return the sorted table
 */
std::shared_ptr<table_info> grouped_sort(
    const grouping_info& grp_info,
    const std::vector<std::shared_ptr<array_info>>& orderby_cols,
    const std::vector<std::shared_ptr<array_info>>& extra_cols,
    const std::vector<bool>& asc_vect, const std::vector<bool>& na_pos_vect,
    int64_t order_offset, bool is_parallel,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());
