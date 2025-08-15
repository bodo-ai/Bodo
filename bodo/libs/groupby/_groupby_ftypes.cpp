#include "_groupby_ftypes.h"

#include <stdexcept>
#include <string>

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
                                   "var_pop",
                                   "std_pop",
                                   "var",
                                   "std",
                                   "kurtosis",
                                   "skew",
                                   "boolor_agg",
                                   "booland_agg",
                                   "boolxor_agg",
                                   "bitor_agg",
                                   "bitand_agg",
                                   "bitxor_agg",
                                   "count_if",
                                   "listagg",
                                   "array_agg",
                                   "array_agg_distinct",
                                   "mode",
                                   "percentile_cont",
                                   "percentile_disc",
                                   "object_agg",
                                   "udf",
                                   "gen_udf",
                                   "window",
                                   "row_number",
                                   "min_row_number_filter",
                                   "rank",
                                   "dense_rank",
                                   "percent_rank",
                                   "cume_dist",
                                   "ntile",
                                   "ratio_to_report",
                                   "conditional_true_event",
                                   "conditional_change_event",
                                   "any_value",
                                   "grouping",
                                   "lead",
                                   "lag",
                                   "num_funcs",
                                   "mean_eval",
                                   "var_pop_eval",
                                   "std_pop_eval",
                                   "var_eval",
                                   "std_eval",
                                   "skew_eval",
                                   "kurt_eval",
                                   "boolxor_eval",
                                   "idxmin_na_first",
                                   "idxmax_na_first",
                                   "idx_n_columns",
                                   "stream_udf",
                                   "n_ftypes"};

const std::string get_name_for_Bodo_FTypes(int enumVal) {
    if (enumVal < 0 || enumVal >= Bodo_FTypes::num_funcs) {
        throw std::runtime_error(std::string("Unknown function type: ") +
                                 std::to_string(enumVal));
    } else {
        return std::string(Bodo_FTypes_names[enumVal]);
    }
}
