// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _GROUPBY_FTYPES_H_INCLUDED
#define _GROUPBY_FTYPES_H_INCLUDED

#include <iostream>
#include <string>

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
        var_pop = 22,
        std_pop = 23,
        var = 24,
        std = 25,
        kurtosis = 26,
        skew = 27,
        boolor_agg = 28,
        booland_agg = 29,
        boolxor_agg = 30,
        bitor_agg = 31,
        bitand_agg = 32,
        bitxor_agg = 33,
        count_if = 34,
        listagg = 35,
        mode = 36,
        percentile_cont = 37,
        percentile_disc = 38,
        udf = 39,
        gen_udf = 40,
        window = 41,
        row_number = 42,
        min_row_number_filter = 43,
        rank = 44,
        dense_rank = 45,
        percent_rank = 46,
        cume_dist = 47,
        ntile = 48,
        ratio_to_report = 49,
        conditional_true_event = 50,
        conditional_change_event = 51,
        any_value = 52,
        num_funcs = 53,  // num_funcs is used to know how many functions up to
                         // this point. Below this point are functions that are
                         // defined in the C++ code but not the Python enum.
        mean_eval = 54,
        var_pop_eval = 55,
        std_pop_eval = 56,
        var_eval = 57,
        std_eval = 58,
        kurt_eval = 59,
        skew_eval = 60,
        boolxor_eval = 61,
        // These are internal operators used by groupby.window
        // when the orderby clause has na values first.
        idxmin_na_first = 62,
        idxmax_na_first = 63,
        // This is the operator for when we are generating one
        // of the 4 idx functions to operate over N columns. Each
        // column may have a different function so we cannot defineF
        // more explicit ftypes. This is used only in the min_row_number_filter
        // window function path.
        idx_n_columns = 64,
    };
};

const std::string get_name_for_Bodo_FTypes(int enumVal);

#endif  // _GROUPBY_FTYPES_H_INCLUDED
