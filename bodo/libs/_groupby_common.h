// Copyright (C) 2023 Bodo Inc. All rights reserved.

#ifndef _GROUPBY_COMMON_H_INCLUDED
#define _GROUPBY_COMMON_H_INCLUDED

#include "_bodo_common.h"
#include "_groupby.h"

/**
 * This function defines helper functions that are shared by multiple possible
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
    size_t num_groups);

#endif  // _GROUPBY_COMMON_H_INCLUDED
