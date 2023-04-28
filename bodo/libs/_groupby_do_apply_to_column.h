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
void do_apply_to_column(std::shared_ptr<array_info> in_col,
                        std::shared_ptr<array_info> out_col,
                        std::vector<std::shared_ptr<array_info>>& aux_cols,
                        const grouping_info& grp_info, int ftype);

/**
 * @brief Apply various idx*** operations to a set of N orderby
 * columns. Depending on the values of asc_vect and na_pos_vect
 * each column is performing one of the following operations:
 * - idxmin
 * - idxmax
 * - idxmin_na_first
 * - idxmax_na_first
 *
 * Note: This implementation requires update_before_shuffle=False.
 *
 * @param[out] out_arr The output index array. This is a UINT64 numpy array.
 * @param[in] orderby_arrs Vector of the N orderby columns. Smaller indices
 * have higher precedence and later columns are only used in the case of
 * ties.
 * @param[in] asc_vect Vector of booleans indicating whether the corresponding
 * orderby column is sorted in ascending order.
 * @param[in] na_pos_vect Vector of booleans indicating whether the
 * corresponding orderby column has NA values at the end in sort order.
 * @param[in] grp_info The grouping information.
 * @param ftype The function type. This is used for validating input.
 */
void idx_n_columns_apply(std::shared_ptr<array_info> out_arr,
                         std::vector<std::shared_ptr<array_info>>& orderby_arrs,
                         std::vector<bool>& asc_vect,
                         std::vector<bool>& na_pos_vect,
                         grouping_info const& grp_info, int ftype);

#endif  // _GROUPBY_DO_APPLY_TO_COLUMN_H_INCLUDED
