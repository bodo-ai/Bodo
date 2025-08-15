#include <sstream>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "../libs/groupby/_groupby_ftypes.h"
#include "../libs/window/_window_compute.h"

#include "./table_generator.hpp"
#include "./test.hpp"

/**
 * Runs a parameterized test for a window function on all-null data.
 * @param[in] ftype the window function ftype to run.
 * @param[in] in_arr_type the type of the all-null array to generate for the
 * input(s).
 * @param[in] in_dtype the dtype of the all-null array to generate for the
 * input(s).
 * @param[in] out_arr_type the type of the array to generate for the expected
 * answer.
 * @param[in] out_dtype the dtype of the array to generate for the expected
 * answer.
 * @param[in] asc_vect the vector of booleans for ascending/descending orderby
 * cols.
 * @param[in] na_pos_vect the vector of booleans for na first/last orderby cols.
 * @param[in] window_args the vector of void* for any extra arguments to the
 * window fn.
 * @param[in] n_input_cols how many input columns to generate for the function
 * call.
 * @param[in] n_order_cols how many orderby columns to generate for the function
 * call.
 * @param[in] return_type enum indicating what type of output data to generate:
 *            - ZERO: all-zero
 *            - ONE: all-one
 *            - NULL_OUTPUT: all-null
 */
#define TEST_WINDOW_FN(ftype, in_arr_type, in_dtype, out_arr_type, out_dtype,  \
                       asc_vect, na_pos_vect, window_args, n_input_cols,       \
                       n_order_cols, return_type)                              \
    {                                                                          \
        std::vector<std::shared_ptr<array_info>> input_arrs;                   \
        std::vector<std::shared_ptr<array_info>> out_arrs;                     \
        grouping_info grp_info;                                                \
        grp_info.num_groups = 1;                                               \
        size_t n = 1;                                                          \
        for (size_t i = 0; i < n; i++) {                                       \
            grp_info.row_to_group.push_back(0);                                \
        }                                                                      \
        for (size_t i = 0; i < n_input_cols + n_order_cols; i++) {             \
            input_arrs.push_back(make_all_null_arr<in_arr_type, in_dtype>(n)); \
        }                                                                      \
        out_arrs.push_back(make_arr<out_arr_type, out_dtype>(n));              \
        std::shared_ptr<array_info> expected_out =                             \
            make_result_output<out_arr_type, out_dtype, return_type>(n);       \
        std::vector<std::shared_ptr<DictionaryBuilder>> out_dict_builders = {  \
            nullptr};                                                          \
        window_computation(input_arrs, {ftype}, out_arrs, out_dict_builders,   \
                           grp_info, asc_vect, na_pos_vect, window_args,       \
                           n_input_cols, false, true);                         \
        std::stringstream ss1;                                                 \
        std::stringstream ss2;                                                 \
        DEBUG_PrintColumn(ss1, out_arrs[0]);                                   \
        DEBUG_PrintColumn(ss2, expected_out);                                  \
        bodo::tests::check(ss1.str() == ss2.str());                            \
    }

static bodo::tests::suite tests([] {
    bodo::tests::test("ensure_all_window_ftypes_tested", [] {
        /* Every time a new ftype is added, we must add an all-null
         * test to this file for that ftype if it is a window function.
         * Once a test is added, the ftype is added to the set
         * tested_window_function_ftypes. If it is not a window function,
         * it is added to non_window_function_ftypes. Alternatively,
         * it can be added to the untested_window_function_ftypes if
         * there is a good reason to avoid testing it with all nulls.
         */
        std::set<size_t> tested_window_function_ftypes = {
            Bodo_FTypes::size,
            Bodo_FTypes::count,
            Bodo_FTypes::count_if,
            Bodo_FTypes::mean,
            Bodo_FTypes::var,
            Bodo_FTypes::var_pop,
            Bodo_FTypes::std,
            Bodo_FTypes::std_pop,
            Bodo_FTypes::rank,
            Bodo_FTypes::dense_rank,
            Bodo_FTypes::percent_rank,
            Bodo_FTypes::cume_dist,
            Bodo_FTypes::first,
            Bodo_FTypes::last,
            Bodo_FTypes::any_value,
            Bodo_FTypes::ratio_to_report,
            Bodo_FTypes::conditional_true_event,
            Bodo_FTypes::conditional_change_event,
            Bodo_FTypes::lead,
            Bodo_FTypes::lag};
        std::set<size_t> untested_window_function_ftypes = {
            // These functions do not have a permanent all-null C++ test since
            // the result is technically nondeterministic.
            Bodo_FTypes::row_number, Bodo_FTypes::min_row_number_filter,
            Bodo_FTypes::ntile, Bodo_FTypes::idxmin_na_first,
            Bodo_FTypes::idxmax_na_first, Bodo_FTypes::idx_n_columns,
            // stream_udf is used in groupy.agg and doesn't make sense to test
            // in
            // this setup.
            Bodo_FTypes::stream_udf};
        std::set<size_t> non_window_function_ftypes = {
            Bodo_FTypes::no_op,
            Bodo_FTypes::ngroup,
            Bodo_FTypes::head,
            Bodo_FTypes::transform,
            Bodo_FTypes::shift,
            Bodo_FTypes::sum,
            Bodo_FTypes::nunique,
            Bodo_FTypes::median,
            Bodo_FTypes::cumsum,
            Bodo_FTypes::cumprod,
            Bodo_FTypes::cummin,
            Bodo_FTypes::cummax,
            Bodo_FTypes::min,
            Bodo_FTypes::max,
            Bodo_FTypes::prod,
            Bodo_FTypes::idxmin,
            Bodo_FTypes::idxmax,
            Bodo_FTypes::kurtosis,
            Bodo_FTypes::skew,
            Bodo_FTypes::boolor_agg,
            Bodo_FTypes::booland_agg,
            Bodo_FTypes::boolxor_agg,
            Bodo_FTypes::bitor_agg,
            Bodo_FTypes::bitand_agg,
            Bodo_FTypes::bitxor_agg,
            Bodo_FTypes::listagg,
            Bodo_FTypes::array_agg,
            Bodo_FTypes::array_agg_distinct,
            Bodo_FTypes::mode,
            Bodo_FTypes::percentile_cont,
            Bodo_FTypes::percentile_disc,
            Bodo_FTypes::object_agg,
            Bodo_FTypes::udf,
            Bodo_FTypes::gen_udf,
            Bodo_FTypes::window,
            Bodo_FTypes::gen_udf,
            Bodo_FTypes::num_funcs,
            Bodo_FTypes::mean_eval,
            Bodo_FTypes::var_eval,
            Bodo_FTypes::var_pop_eval,
            Bodo_FTypes::std_eval,
            Bodo_FTypes::std_pop_eval,
            Bodo_FTypes::skew_eval,
            Bodo_FTypes::kurt_eval,
            Bodo_FTypes::boolxor_eval,
            Bodo_FTypes::grouping,
            Bodo_FTypes::n_ftypes,
        };
        for (size_t i = 0; i < Bodo_FTypes::n_ftypes; i++) {
            bool is_window_fn_tested =
                tested_window_function_ftypes.contains(i);
            bool is_window_fn_untested =
                untested_window_function_ftypes.contains(i);
            bool is_not_window_ftype = non_window_function_ftypes.contains(i);
            bodo::tests::check(is_window_fn_tested || is_window_fn_untested ||
                               is_not_window_ftype);
        }
    });

    bodo::tests::test("test_window_rank_fns", [] {
#define TEST_RANK_WINDOW_FNS(arr_type, dtype)                                  \
    TEST_WINDOW_FN(Bodo_FTypes::rank, arr_type, dtype, bodo_array_type::NUMPY, \
                   Bodo_CTypes::INT64, {true}, {true}, {}, 0, 1,               \
                   empty_return_enum::ONE);                                    \
    TEST_WINDOW_FN(Bodo_FTypes::dense_rank, arr_type, dtype,                   \
                   bodo_array_type::NUMPY, Bodo_CTypes::INT64, {true}, {true}, \
                   {}, 0, 1, empty_return_enum::ONE);                          \
    TEST_WINDOW_FN(Bodo_FTypes::percent_rank, arr_type, dtype,                 \
                   bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64, {true},       \
                   {true}, {}, 0, 1, empty_return_enum::ZERO);                 \
    TEST_WINDOW_FN(Bodo_FTypes::cume_dist, arr_type, dtype,                    \
                   bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64, {true},       \
                   {true}, {}, 0, 1, empty_return_enum::ONE);                  \
    TEST_WINDOW_FN(Bodo_FTypes::conditional_change_event, arr_type, dtype,     \
                   bodo_array_type::NUMPY, Bodo_CTypes::INT64, {true}, {true}, \
                   {}, 1, 1, empty_return_enum::ZERO);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::INT8);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::INT16);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::INT32);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::INT64);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::UINT8);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::UINT16);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::UINT32);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::UINT64);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::FLOAT32);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::FLOAT64);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::_BOOL);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::INT128);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::DECIMAL);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::DATE);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::TIME);
        TEST_RANK_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                             Bodo_CTypes::DATETIME);
        TEST_RANK_WINDOW_FNS(bodo_array_type::STRING, Bodo_CTypes::STRING);
        TEST_RANK_WINDOW_FNS(bodo_array_type::DICT, Bodo_CTypes::STRING);
        TEST_RANK_WINDOW_FNS(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::INT64);
        TEST_WINDOW_FN(Bodo_FTypes::conditional_true_event,
                       bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL,
                       bodo_array_type::NUMPY, Bodo_CTypes::INT64, {true},
                       {true}, {}, 1, 1, empty_return_enum::ZERO);
    });

    bodo::tests::test("test_window_select_fns", [] {
        int64_t idx0 = 3;
        int64_t idx1 = -3;
        std::shared_ptr<array_info> null_col =
            alloc_nullable_array_all_nulls(1, Bodo_CTypes::INT64);
        std::shared_ptr<array_info> idx0_col =
            bodo::tests::cppToBodoArr(std::vector<int64_t>{idx0});
        std::shared_ptr<array_info> idx1_col =
            bodo::tests::cppToBodoArr(std::vector<int64_t>{idx1});

        std::shared_ptr<table_info> empty_frame(
            new table_info({null_col, null_col}));
        std::shared_ptr<table_info> prefix_frame(
            new table_info({null_col, idx0_col}));
        std::shared_ptr<table_info> suffix_frame(
            new table_info({idx0_col, null_col}));
        std::shared_ptr<table_info> sliding_frame(
            new table_info({idx1_col, idx0_col}));

        std::shared_ptr<array_info> null_default;
        std::shared_ptr<table_info> window_args;
        std::shared_ptr<array_info> one_arr =
            alloc_numpy(1, Bodo_CTypes::INT64);
        getv<int64_t>(one_arr, 0) = 1;

#define TEST_SELECT_WINDOW_FNS(arr_type, dtype)                                \
    TEST_WINDOW_FN(Bodo_FTypes::any_value, arr_type, dtype, arr_type, dtype,   \
                   {}, {}, nullptr, 1, 0, empty_return_enum::NULL_OUTPUT);     \
    TEST_WINDOW_FN(Bodo_FTypes::first, arr_type, dtype, arr_type, dtype, {},   \
                   {}, empty_frame, 1, 0, empty_return_enum::NULL_OUTPUT);     \
    TEST_WINDOW_FN(Bodo_FTypes::first, arr_type, dtype, arr_type, dtype, {},   \
                   {}, prefix_frame, 1, 0, empty_return_enum::NULL_OUTPUT);    \
    TEST_WINDOW_FN(Bodo_FTypes::first, arr_type, dtype, arr_type, dtype, {},   \
                   {}, suffix_frame, 1, 0, empty_return_enum::NULL_OUTPUT);    \
    TEST_WINDOW_FN(Bodo_FTypes::first, arr_type, dtype, arr_type, dtype, {},   \
                   {}, sliding_frame, 1, 0, empty_return_enum::NULL_OUTPUT);   \
    TEST_WINDOW_FN(Bodo_FTypes::last, arr_type, dtype, arr_type, dtype, {},    \
                   {}, empty_frame, 1, 0, empty_return_enum::NULL_OUTPUT);     \
    TEST_WINDOW_FN(Bodo_FTypes::last, arr_type, dtype, arr_type, dtype, {},    \
                   {}, prefix_frame, 1, 0, empty_return_enum::NULL_OUTPUT);    \
    TEST_WINDOW_FN(Bodo_FTypes::last, arr_type, dtype, arr_type, dtype, {},    \
                   {}, suffix_frame, 1, 0, empty_return_enum::NULL_OUTPUT);    \
    TEST_WINDOW_FN(Bodo_FTypes::last, arr_type, dtype, arr_type, dtype, {},    \
                   {}, sliding_frame, 1, 0, empty_return_enum::NULL_OUTPUT);   \
    null_default = make_all_null_arr<arr_type, dtype>(1);                      \
    {                                                                          \
        std::shared_ptr<table_info> window_args_(                              \
            new table_info({one_arr, null_default}));                          \
        window_args = window_args_;                                            \
    }                                                                          \
    TEST_WINDOW_FN(Bodo_FTypes::lead, arr_type, dtype, arr_type, dtype, {},    \
                   {}, window_args, 1, 0, empty_return_enum::NULL_OUTPUT);     \
    TEST_WINDOW_FN(Bodo_FTypes::lag, arr_type, dtype, arr_type, dtype, {}, {}, \
                   window_args, 1, 0, empty_return_enum::NULL_OUTPUT);

        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT8);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT16);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT32);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT64);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT8);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT16);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT32);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT64);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::FLOAT32);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::FLOAT64);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::_BOOL);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT128);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::DECIMAL);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::DATE);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::TIME);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::DATETIME);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::STRING, Bodo_CTypes::STRING);
        TEST_SELECT_WINDOW_FNS(bodo_array_type::DICT, Bodo_CTypes::STRING);
    });

    bodo::tests::test("test_window_numeric_fns", [] {
        int64_t idx0 = 3;
        int64_t idx1 = -3;
        std::shared_ptr<array_info> null_col =
            alloc_nullable_array_all_nulls(1, Bodo_CTypes::INT64);
        std::shared_ptr<array_info> idx0_col =
            bodo::tests::cppToBodoArr(std::vector<int64_t>{idx0});
        std::shared_ptr<array_info> idx1_col =
            bodo::tests::cppToBodoArr(std::vector<int64_t>{idx1});

        std::shared_ptr<table_info> empty_frame(
            new table_info({null_col, null_col}));
        std::shared_ptr<table_info> prefix_frame(
            new table_info({null_col, idx0_col}));
        std::shared_ptr<table_info> suffix_frame(
            new table_info({idx0_col, null_col}));
        std::shared_ptr<table_info> sliding_frame(
            new table_info({idx1_col, idx0_col}));

#define TEST_NUMERIC_WINDOW_FNS(arr_type, dtype)                               \
    TEST_WINDOW_FN(Bodo_FTypes::ratio_to_report, arr_type, dtype,              \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, {}, 1, 0, empty_return_enum::NULL_OUTPUT);          \
    TEST_WINDOW_FN(Bodo_FTypes::mean, arr_type, dtype,                         \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, empty_frame, 1, 0, empty_return_enum::NULL_OUTPUT); \
    TEST_WINDOW_FN(Bodo_FTypes::mean, arr_type, dtype,                         \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, prefix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::mean, arr_type, dtype,                         \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, suffix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::mean, arr_type, dtype,                         \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, sliding_frame, 1, 0,                                \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::var, arr_type, dtype,                          \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, empty_frame, 1, 0, empty_return_enum::NULL_OUTPUT); \
    TEST_WINDOW_FN(Bodo_FTypes::var, arr_type, dtype,                          \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, prefix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::var, arr_type, dtype,                          \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, suffix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::var, arr_type, dtype,                          \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, sliding_frame, 1, 0,                                \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::std, arr_type, dtype,                          \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, empty_frame, 1, 0, empty_return_enum::NULL_OUTPUT); \
    TEST_WINDOW_FN(Bodo_FTypes::std, arr_type, dtype,                          \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, prefix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::std, arr_type, dtype,                          \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, suffix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::std, arr_type, dtype,                          \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, sliding_frame, 1, 0,                                \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::var_pop, arr_type, dtype,                      \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, empty_frame, 1, 0, empty_return_enum::NULL_OUTPUT); \
    TEST_WINDOW_FN(Bodo_FTypes::var_pop, arr_type, dtype,                      \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, prefix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::var_pop, arr_type, dtype,                      \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, suffix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::var_pop, arr_type, dtype,                      \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, sliding_frame, 1, 0,                                \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::std_pop, arr_type, dtype,                      \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, empty_frame, 1, 0, empty_return_enum::NULL_OUTPUT); \
    TEST_WINDOW_FN(Bodo_FTypes::std_pop, arr_type, dtype,                      \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, prefix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::std_pop, arr_type, dtype,                      \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, suffix_frame, 1, 0,                                 \
                   empty_return_enum::NULL_OUTPUT);                            \
    TEST_WINDOW_FN(Bodo_FTypes::std_pop, arr_type, dtype,                      \
                   bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::FLOAT64,   \
                   {}, {}, sliding_frame, 1, 0,                                \
                   empty_return_enum::NULL_OUTPUT);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT8);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT16);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT32);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT64);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT8);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT16);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT32);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT64);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::FLOAT32);
        TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::FLOAT64);
        // [BSE-2185] TODO: support numeric window functions on decimals
        // TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
        // Bodo_CTypes::INT128);
        // TEST_NUMERIC_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
        // Bodo_CTypes::DECIMAL);
    });

    bodo::tests::test("test_window_count_fns", [] {
        int64_t idx0 = 3;
        int64_t idx1 = -3;
        std::shared_ptr<array_info> null_col =
            alloc_nullable_array_all_nulls(1, Bodo_CTypes::INT64);
        std::shared_ptr<array_info> idx0_col =
            bodo::tests::cppToBodoArr(std::vector<int64_t>{idx0});
        std::shared_ptr<array_info> idx1_col =
            bodo::tests::cppToBodoArr(std::vector<int64_t>{idx1});

        std::shared_ptr<table_info> empty_frame(
            new table_info({null_col, null_col}));
        std::shared_ptr<table_info> prefix_frame(
            new table_info({null_col, idx0_col}));
        std::shared_ptr<table_info> suffix_frame(
            new table_info({idx1_col, null_col}));
        std::shared_ptr<table_info> sliding_frame(
            new table_info({idx1_col, idx0_col}));

#define TEST_COUNT_WINDOW_FNS(arr_type, dtype)                         \
    TEST_WINDOW_FN(Bodo_FTypes::count, arr_type, dtype,                \
                   bodo_array_type::NUMPY, Bodo_CTypes::INT64, {}, {}, \
                   empty_frame, 1, 0, empty_return_enum::ZERO);        \
    TEST_WINDOW_FN(Bodo_FTypes::count, arr_type, dtype,                \
                   bodo_array_type::NUMPY, Bodo_CTypes::INT64, {}, {}, \
                   prefix_frame, 1, 0, empty_return_enum::ZERO);       \
    TEST_WINDOW_FN(Bodo_FTypes::count, arr_type, dtype,                \
                   bodo_array_type::NUMPY, Bodo_CTypes::INT64, {}, {}, \
                   suffix_frame, 1, 0, empty_return_enum::ZERO);       \
    TEST_WINDOW_FN(Bodo_FTypes::count, arr_type, dtype,                \
                   bodo_array_type::NUMPY, Bodo_CTypes::INT64, {}, {}, \
                   sliding_frame, 1, 0, empty_return_enum::ZERO);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT8);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT16);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT32);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT64);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT8);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT16);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT32);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT64);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::FLOAT32);
        TEST_COUNT_WINDOW_FNS(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::FLOAT64);
        TEST_WINDOW_FN(Bodo_FTypes::size, bodo_array_type::NULLABLE_INT_BOOL,
                       Bodo_CTypes::INT64, bodo_array_type::NUMPY,
                       Bodo_CTypes::INT64, {}, {}, empty_frame, 0, 0,
                       empty_return_enum::ONE);
        TEST_WINDOW_FN(Bodo_FTypes::size, bodo_array_type::NULLABLE_INT_BOOL,
                       Bodo_CTypes::INT64, bodo_array_type::NUMPY,
                       Bodo_CTypes::INT64, {}, {}, prefix_frame, 0, 0,
                       empty_return_enum::ONE);
        TEST_WINDOW_FN(Bodo_FTypes::size, bodo_array_type::NULLABLE_INT_BOOL,
                       Bodo_CTypes::INT64, bodo_array_type::NUMPY,
                       Bodo_CTypes::INT64, {}, {}, suffix_frame, 0, 0,
                       empty_return_enum::ONE);
        TEST_WINDOW_FN(Bodo_FTypes::size, bodo_array_type::NULLABLE_INT_BOOL,
                       Bodo_CTypes::INT64, bodo_array_type::NUMPY,
                       Bodo_CTypes::INT64, {}, {}, sliding_frame, 0, 0,
                       empty_return_enum::ONE);
        TEST_WINDOW_FN(Bodo_FTypes::count_if,
                       bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL,
                       bodo_array_type::NUMPY, Bodo_CTypes::INT64, {}, {},
                       empty_frame, 1, 0, empty_return_enum::ZERO);
        TEST_WINDOW_FN(Bodo_FTypes::count_if,
                       bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL,
                       bodo_array_type::NUMPY, Bodo_CTypes::INT64, {}, {},
                       prefix_frame, 1, 0, empty_return_enum::ZERO);
        TEST_WINDOW_FN(Bodo_FTypes::count_if,
                       bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL,
                       bodo_array_type::NUMPY, Bodo_CTypes::INT64, {}, {},
                       suffix_frame, 1, 0, empty_return_enum::ZERO);
        TEST_WINDOW_FN(Bodo_FTypes::count_if,
                       bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL,
                       bodo_array_type::NUMPY, Bodo_CTypes::INT64, {}, {},
                       sliding_frame, 1, 0, empty_return_enum::ZERO);
    });
});
