// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_groupby_ftypes.h"

/**
 * This file defines the various ftype helper functions that shouldn't
 * be exposed in a .h file.
 */

const char* Bodo_FTypes_names[] = {"no_op",
                                   "ngroup",
                                   "head",
                                   "transform",
                                   "size",
                                   "shift",
                                   "sum",
                                   "count",
                                   "nunique",
                                   "median",
                                   "cumsum",
                                   "cumprod",
                                   "cummin",
                                   "cummax",
                                   "mean",
                                   "min",
                                   "max",
                                   "prod",
                                   "first",
                                   "last",
                                   "idxmin",
                                   "idxmax",
                                   "var",
                                   "std",
                                   "boolor_agg",
                                   "count_if",
                                   "udf",
                                   "gen_udf",
                                   "window",
                                   "row_number",
                                   "min_row_number_filter",
                                   "num_funcs",
                                   "mean_eval",
                                   "var_eval",
                                   "std_eval",
                                   "idxmin_na_first",
                                   "idxmax_na_first",
                                   "idx_n_columns"};

const char* get_name_for_Bodo_FTypes(int enumVal) {
    return Bodo_FTypes_names[enumVal];
}
