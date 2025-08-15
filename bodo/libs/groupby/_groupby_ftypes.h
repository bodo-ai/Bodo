#pragma once

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
        array_agg = 36,
        array_agg_distinct = 37,
        mode = 38,
        percentile_cont = 39,
        percentile_disc = 40,
        object_agg = 41,
        udf = 42,
        gen_udf = 43,
        window = 44,
        row_number = 45,
        min_row_number_filter = 46,
        rank = 47,
        dense_rank = 48,
        percent_rank = 49,
        cume_dist = 50,
        ntile = 51,
        ratio_to_report = 52,
        conditional_true_event = 53,
        conditional_change_event = 54,
        any_value = 55,
        grouping = 56,
        lead = 57,
        lag = 58,
        num_funcs = 59,  // num_funcs is used to know how many functions up to
                         // this point. Below this point are functions that are
                         // defined in the C++ code but not the Python enum.
        mean_eval = 60,
        var_pop_eval = 61,
        std_pop_eval = 62,
        var_eval = 63,
        std_eval = 64,
        kurt_eval = 65,
        skew_eval = 66,
        boolxor_eval = 67,
        // These are internal operators used by groupby.window
        // when the orderby clause has na values first.
        idxmin_na_first = 68,
        idxmax_na_first = 69,
        // This is the operator for when we are generating one
        // of the 4 idx functions to operate over N columns. Each
        // column may have a different function so we cannot defineF
        // more explicit ftypes. This is used only in the min_row_number_filter
        // window function path.
        idx_n_columns = 70,
        // Streaming UDFs for Bodo DataFrames groupby.agg
        stream_udf = 71,
        n_ftypes = 72,
    };
};

const std::string get_name_for_Bodo_FTypes(int enumVal);
