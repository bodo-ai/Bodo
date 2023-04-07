// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _GROUPBY_FTYPES_H_INCLUDED
#define _GROUPBY_FTYPES_H_INCLUDED

/**
 * This file defines the various function types supported by groupby and shared
 * by the groupby infrastructure. In addition this defines some common helper
 * functions used for debugging the FTypes.
 *
 */

/**
 * Enum of aggregation, combine and eval functions used by groupby.
 * Some functions like sum can be used for multiple purposes, like aggregation
 * and combine. Some operations like sum don't need eval.
 */
struct Bodo_FTypes {
    // !!! IMPORTANT: this is supposed to match the positions in
    // supported_agg_funcs in aggregate.py
    // !!! IMPORTANT: this is supposed to match the defined
    // names in _groupby_ftypes.cpp
    enum FTypeEnum {
        no_op = 0,  // To make sure ftypes[0] isn't accidentally matched with
                    // any of the supported functions.
        ngroup = 1,
        head = 2,
        transform = 3,
        size = 4,
        shift = 5,
        sum = 6,
        count = 7,
        nunique = 8,
        median = 9,
        cumsum = 10,
        cumprod = 11,
        cummin = 12,
        cummax = 13,
        mean = 14,
        min = 15,
        max = 16,
        prod = 17,
        first = 18,
        last = 19,
        idxmin = 20,
        idxmax = 21,
        var = 22,
        std = 23,
        boolor_agg = 24,
        count_if = 25,
        udf = 26,
        gen_udf = 27,
        window = 28,
        row_number = 29,
        min_row_number_filter = 30,
        num_funcs = 31,  // num_funcs is used to know how many functions up to
                         // this point. Below this point are functions that are
                         // defined in the C++ code but not the Python enum.
        mean_eval = 32,
        var_eval = 33,
        std_eval = 34,
        // These are internal operators used by groupby.window
        // when the orderby clause has na values first.
        idxmin_na_first = 35,
        idxmax_na_first = 36,
        // This is the operator for when we are generating one
        // of the 4 idx functions to operate over N columns. Each
        // column may have a different function so we cannot define
        // more explicit ftypes. This is used only in the min_row_number_filter
        // window function path.
        idx_n_columns = 37
    };
};

const char* get_name_for_Bodo_FTypes(int enumVal);

#endif  // _GROUPBY_FTYPES_H_INCLUDED
