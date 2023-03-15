// Copyright (C) 2023 Bodo Inc. All rights reserved.

#ifndef _GROUPBY_COMMON_H_INCLUDED
#define _GROUPBY_COMMON_H_INCLUDED

#include "_bodo_common.h"

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
void aggfunc_output_initialize(array_info* out_col, int ftype,
                               bool use_sql_rules);

#endif  // _GROUPBY_COMMON_H_INCLUDED
