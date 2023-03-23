// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _GROUPBY_DO_APPLY_TO_COLUMN_H_INCLUDED
#define _GROUPBY_DO_APPLY_TO_COLUMN_H_INCLUDED

#include "_groupby.h"

/**
 * This file declares the functions that define the
 * general infrastructure used to apply most operations
 * to an individual column. This infrastructure is used
 * by update, combine, and eval.
 */

/**
 * Invokes the correct template instance of apply_to_column depending on
 * function (ftype) and dtype. See 'apply_to_column'
 *
 * @param[in] in_col column containing input values
 * @param[in, out] out_col output column
 * @param[in, out] aux_cols auxiliary input/output columns used for mean, var,
 * std
 * @param grp_info Grouping information relating rows to the proper group.
 * @param ftype The aggregate function to apply
 */
void do_apply_to_column(array_info* in_col, array_info* out_col,
                        std::vector<array_info*>& aux_cols,
                        const grouping_info& grp_info, int ftype);

#endif  // _GROUPBY_DO_APPLY_TO_COLUMN_H_INCLUDED
