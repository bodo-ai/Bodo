#include <iostream>
#include <sstream>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "../libs/_groupby_col_set.h"
#include "../libs/_window_compute.h"
#include "./test.hpp"

typedef enum {
    ZERO,
    ONE,
    NULL_OUTPUT,
    EMPTY_STRING,
    EMPTY_ARRAY,
    EMPTY_MAP,
} empty_return_enum;

// Create an array of the desired type without directly setting the null bit
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(nullable_array<ArrType>)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_nullable_array(n, DType);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(numpy_array<ArrType>)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_numpy(n, DType);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(string_array<ArrType>)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_string_array(DType, n, 0);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(dict_array<ArrType>)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_dict_string_array(n, 0, 0);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(ArrType == bodo_array_type::ARRAY_ITEM)
std::shared_ptr<array_info> make_arr(size_t n) {
    return alloc_array_item(n, alloc_numpy(0, DType));
}

// Create an array of the desired type with all null elements

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(nullable_array<ArrType>)
std::shared_ptr<array_info> make_all_null_arr(size_t n) {
    return alloc_nullable_array_all_nulls(n, DType, 0);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(numpy_array<ArrType>)
std::shared_ptr<array_info> make_all_null_arr(size_t n) {
    return alloc_numpy(n, DType);
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(string_array<ArrType> || dict_array<ArrType> ||
             array_item_array<ArrType>)
std::shared_ptr<array_info> make_all_null_arr(size_t n) {
    std::shared_ptr<array_info> arr = make_arr<ArrType, DType>(n);
    for (size_t i = 0; i < n; i++) {
        arr->set_null_bit(i, false);
    }
    return arr;
}

// Create an array of the desired type with all 0 elements. Only works
// on nullable/numpy arrays with a numeric dtype.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::ZERO)
std::shared_ptr<array_info> make_result_output(size_t n) {
    using T = typename dtype_to_type<DType>::type;
    std::shared_ptr<array_info> res = make_arr<ArrType, DType>(n);
    for (size_t i = 0; i < n; i++) {
        set_non_null<ArrType, T, DType>(*res, i);
        set_arr_item<ArrType, T, DType>(*res, i, static_cast<T>(0));
    }
    return res;
}

// Create an array of the desired type with all 1 elements. Only works
// on nullable/numpy arrays with a numeric dtype.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::ONE)
std::shared_ptr<array_info> make_result_output(size_t n) {
    using T = typename dtype_to_type<DType>::type;
    std::shared_ptr<array_info> res = make_arr<ArrType, DType>(n);
    for (size_t i = 0; i < n; i++) {
        set_non_null<ArrType, T, DType>(*res, i);
        set_arr_item<ArrType, T, DType>(*res, i, static_cast<T>(1));
    }
    return res;
}

// Create an array of the desired type with all null elements. Should work
// on all array types.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::NULL_OUTPUT)
std::shared_ptr<array_info> make_result_output(size_t n) {
    return make_all_null_arr<ArrType, DType>(n);
}

// Create an array of the desired type with all empty string elements. Only
// works when return type is a string array.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::EMPTY_STRING)
std::shared_ptr<array_info> make_result_output(size_t n) {
    bodo::vector<uint8_t> nulls((n + 7) >> 3, 0xff);
    bodo::vector<std::string> strings(n, "");
    return create_string_array(DType, nulls, strings);
}

// Create an array of the desired type with all empty array elements. Only
// works when return type is an array item array.
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::EMPTY_ARRAY)
std::shared_ptr<array_info> make_result_output(size_t n) {
    std::shared_ptr<array_info> inner_arr;
    if (DType == Bodo_CTypes::STRUCT) {
        inner_arr = alloc_struct(0, {});
    } else {
        inner_arr = alloc_numpy(0, DType);
    }
    std::shared_ptr<array_info> array_arr = alloc_array_item(n, inner_arr);
    offset_t *offsets =
        (offset_t *)(array_arr->buffers[0]->mutable_data() + array_arr->offset);
    for (size_t i = 0; i < n + 1; i++) {
        offsets[i] = 0;
    }
    return array_arr;
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          empty_return_enum RetType>
    requires(RetType == empty_return_enum::EMPTY_MAP)
std::shared_ptr<array_info> make_result_output(size_t n) {
    std::shared_ptr<array_info> inner_arr =
        make_result_output<bodo_array_type::ARRAY_ITEM, Bodo_CTypes::STRUCT,
                           empty_return_enum::EMPTY_ARRAY>(n);
    return alloc_map(n, inner_arr);
}

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
        window_computation(input_arrs, {ftype}, out_arrs, grp_info, asc_vect,  \
                           na_pos_vect, window_args, n_input_cols, false,      \
                           true);                                              \
        std::stringstream ss1;                                                 \
        std::stringstream ss2;                                                 \
        DEBUG_PrintColumn(ss1, out_arrs[0]);                                   \
        DEBUG_PrintColumn(ss2, expected_out);                                  \
        bodo::tests::check(ss1.str() == ss2.str());                            \
    }

/**
 * Runs a parameterized test for a window function on all-null data.
 * @param[in] col_set the groupby col set containing the all-null data.
 * @param[in] n the length of the data in col_set.
 * @param[in] out_arr_type the type of the array to generate for the expected
 * answer.
 * @param[in] out_dtype the dtype of the array to generate for the expected
 * answer.
 * @param[in] return_type enum indicating what type of output data to generate:
 *            - ZERO: all-zero
 *            - ONE: all-one
 *            - NULL_OUTPUT: all-null
 */
#define TEST_GROUPBY_FN(col_set, n, out_arr_type, out_dtype, return_type)   \
    {                                                                       \
        grouping_info grp_info;                                             \
        grp_info.num_groups = 1;                                            \
        grp_info.group_to_first_row.push_back(0);                           \
        for (size_t i = 0; i < n; i++) {                                    \
            grp_info.row_to_group.push_back(0);                             \
            grp_info.next_row_in_group.push_back((i == (n - 1)) ? -1        \
                                                                : (i + 1)); \
        }                                                                   \
        std::vector<std::shared_ptr<array_info>> list_arr;                  \
        col_set->alloc_update_columns(1, list_arr);                         \
        col_set->update({grp_info});                                        \
        col_set->eval(grp_info);                                            \
        const std::vector<std::shared_ptr<array_info>> out_arrs =           \
            col_set->getOutputColumns();                                    \
        std::shared_ptr<array_info> expected_out =                          \
            make_result_output<out_arr_type, out_dtype, return_type>(1);    \
        std::stringstream ss1;                                              \
        std::stringstream ss2;                                              \
        DEBUG_PrintColumn(ss1, out_arrs[0]);                                \
        DEBUG_PrintColumn(ss2, expected_out);                               \
        bodo::tests::check(ss1.str() == ss2.str());                         \
    }

static bodo::tests::suite tests([] {
    bodo::tests::test("ensure_all_window_ftypes tested", [] {
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
        };
        std::set<size_t> untested_window_function_ftypes = {
            // These functions do not have a permanent all-null C++ test since
            // the result is technically nondeterministic.
            Bodo_FTypes::row_number,      Bodo_FTypes::min_row_number_filter,
            Bodo_FTypes::ntile,           Bodo_FTypes::idxmin_na_first,
            Bodo_FTypes::idxmax_na_first, Bodo_FTypes::idx_n_columns,
        };
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
        std::vector<void *> empty_frame = {nullptr, nullptr};
        std::vector<void *> prefix_frame = {nullptr, &idx0};
        std::vector<void *> suffix_frame = {&idx0, nullptr};
        std::vector<void *> sliding_frame = {&idx1, &idx0};
#define TEST_SELECT_WINDOW_FNS(arr_type, dtype)                              \
    TEST_WINDOW_FN(Bodo_FTypes::any_value, arr_type, dtype, arr_type, dtype, \
                   {}, {}, {}, 1, 0, empty_return_enum::NULL_OUTPUT);        \
    TEST_WINDOW_FN(Bodo_FTypes::first, arr_type, dtype, arr_type, dtype, {}, \
                   {}, empty_frame, 1, 0, empty_return_enum::NULL_OUTPUT);   \
    TEST_WINDOW_FN(Bodo_FTypes::first, arr_type, dtype, arr_type, dtype, {}, \
                   {}, prefix_frame, 1, 0, empty_return_enum::NULL_OUTPUT);  \
    TEST_WINDOW_FN(Bodo_FTypes::first, arr_type, dtype, arr_type, dtype, {}, \
                   {}, suffix_frame, 1, 0, empty_return_enum::NULL_OUTPUT);  \
    TEST_WINDOW_FN(Bodo_FTypes::first, arr_type, dtype, arr_type, dtype, {}, \
                   {}, sliding_frame, 1, 0, empty_return_enum::NULL_OUTPUT); \
    TEST_WINDOW_FN(Bodo_FTypes::last, arr_type, dtype, arr_type, dtype, {},  \
                   {}, empty_frame, 1, 0, empty_return_enum::NULL_OUTPUT);   \
    TEST_WINDOW_FN(Bodo_FTypes::last, arr_type, dtype, arr_type, dtype, {},  \
                   {}, prefix_frame, 1, 0, empty_return_enum::NULL_OUTPUT);  \
    TEST_WINDOW_FN(Bodo_FTypes::last, arr_type, dtype, arr_type, dtype, {},  \
                   {}, suffix_frame, 1, 0, empty_return_enum::NULL_OUTPUT);  \
    TEST_WINDOW_FN(Bodo_FTypes::last, arr_type, dtype, arr_type, dtype, {},  \
                   {}, sliding_frame, 1, 0, empty_return_enum::NULL_OUTPUT);
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
        std::vector<void *> empty_frame = {nullptr, nullptr};
        std::vector<void *> prefix_frame = {nullptr, &idx0};
        std::vector<void *> suffix_frame = {&idx0, nullptr};
        std::vector<void *> sliding_frame = {&idx1, &idx0};
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
        std::vector<void *> no_frame = {nullptr, nullptr};
        std::vector<void *> prefix_frame = {nullptr, &idx0};
        std::vector<void *> suffix_frame = {&idx1, nullptr};
        std::vector<void *> sliding_frame = {&idx1, &idx0};
#define TEST_COUNT_WINDOW_FNS(arr_type, dtype)                         \
    TEST_WINDOW_FN(Bodo_FTypes::count, arr_type, dtype,                \
                   bodo_array_type::NUMPY, Bodo_CTypes::INT64, {}, {}, \
                   no_frame, 1, 0, empty_return_enum::ZERO);           \
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
                       Bodo_CTypes::INT64, {}, {}, no_frame, 0, 0,
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
                       no_frame, 1, 0, empty_return_enum::ZERO);
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

    bodo::tests::test("ensure_all_groupby_ftypes tested", [] {
        /* Every time a new ftype is added, we must add an all-null
         * test to this file for that ftype if it is a BodoSQL groupby
         * function. Once a test is added, the ftype is added to the set
         * tested_groupby_function_ftypes. If it is not a BodoSQL groupby
         * function, it is added to non_groupby_function_ftypes. Alternatively,
         * it can be added to the untested_groupby_function_ftypes if
         * there is a good reason to avoid testing it with all nulls.
         */
        std::set<size_t> tested_groupby_function_ftypes = {
            Bodo_FTypes::size,
            Bodo_FTypes::count,
            Bodo_FTypes::count_if,
            Bodo_FTypes::mean,
            Bodo_FTypes::var,
            Bodo_FTypes::var_pop,
            Bodo_FTypes::std,
            Bodo_FTypes::std_pop,
            Bodo_FTypes::sum,
            Bodo_FTypes::min,
            Bodo_FTypes::max,
            Bodo_FTypes::kurtosis,
            Bodo_FTypes::skew,
            Bodo_FTypes::first,
            Bodo_FTypes::boolor_agg,
            Bodo_FTypes::booland_agg,
            Bodo_FTypes::boolxor_agg,
            Bodo_FTypes::bitor_agg,
            Bodo_FTypes::bitand_agg,
            Bodo_FTypes::bitxor_agg,
            Bodo_FTypes::array_agg,
            Bodo_FTypes::array_agg_distinct,
            Bodo_FTypes::mode,
            Bodo_FTypes::median,
            Bodo_FTypes::percentile_cont,
            Bodo_FTypes::percentile_disc,
            Bodo_FTypes::object_agg,
        };
        std::set<size_t> untested_groupby_function_ftypes = {
            // These ftypes have bugs when run with all
            // null, so their tests are skipped until
            // followup issues fix the bugs.
            Bodo_FTypes::listagg,
            // Untested because it doesn't make sense to test
            // this function with this setup.
            Bodo_FTypes::nunique,
        };
        std::set<size_t> non_groupby_function_ftypes = {
            Bodo_FTypes::no_op,
            Bodo_FTypes::ngroup,
            Bodo_FTypes::head,
            Bodo_FTypes::transform,
            Bodo_FTypes::shift,
            Bodo_FTypes::cumsum,
            Bodo_FTypes::cumprod,
            Bodo_FTypes::cummin,
            Bodo_FTypes::cummax,
            Bodo_FTypes::prod,
            Bodo_FTypes::idxmin,
            Bodo_FTypes::idxmax,
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
            Bodo_FTypes::row_number,
            Bodo_FTypes::min_row_number_filter,
            Bodo_FTypes::ntile,
            Bodo_FTypes::idxmin_na_first,
            Bodo_FTypes::idxmax_na_first,
            Bodo_FTypes::idx_n_columns,
            Bodo_FTypes::n_ftypes,
            Bodo_FTypes::rank,
            Bodo_FTypes::dense_rank,
            Bodo_FTypes::percent_rank,
            Bodo_FTypes::cume_dist,
            Bodo_FTypes::last,
            Bodo_FTypes::any_value,
            Bodo_FTypes::ratio_to_report,
            Bodo_FTypes::conditional_true_event,
            Bodo_FTypes::conditional_change_event,
        };
        for (size_t i = 0; i < Bodo_FTypes::n_ftypes; i++) {
            bool is_groupby_fn_tested =
                tested_groupby_function_ftypes.contains(i);
            bool is_groupby_fn_untested =
                untested_groupby_function_ftypes.contains(i);
            bool is_not_groupby_ftype = non_groupby_function_ftypes.contains(i);
            bodo::tests::check(is_groupby_fn_tested || is_groupby_fn_untested ||
                               is_not_groupby_ftype);
        }
    });

    bodo::tests::test("test_groupby_moment_fns", [] {
#define TEST_MOMENT_GROUPBY_FN(arr_type, dtype)                                \
    {                                                                          \
        size_t n = 3;                                                          \
        std::shared_ptr<array_info> in_col =                                   \
            make_all_null_arr<arr_type, dtype>(n);                             \
        auto var_col_set =                                                     \
            new VarStdColSet(in_col, Bodo_FTypes::var, false, true);           \
        auto var_pop_col_set =                                                 \
            new VarStdColSet(in_col, Bodo_FTypes::var_pop, false, true);       \
        auto std_col_set =                                                     \
            new VarStdColSet(in_col, Bodo_FTypes::std, false, true);           \
        auto std_pop_col_set =                                                 \
            new VarStdColSet(in_col, Bodo_FTypes::std_pop, false, true);       \
        auto skew_col_set =                                                    \
            new SkewColSet(in_col, Bodo_FTypes::skew, false, true);            \
        auto kurt_col_set =                                                    \
            new KurtColSet(in_col, Bodo_FTypes::kurtosis, false, true);        \
        TEST_GROUPBY_FN(var_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,    \
                        Bodo_CTypes::FLOAT64, empty_return_enum::NULL_OUTPUT); \
        TEST_GROUPBY_FN(var_pop_col_set, n,                                    \
                        bodo_array_type::NULLABLE_INT_BOOL,                    \
                        Bodo_CTypes::FLOAT64, empty_return_enum::NULL_OUTPUT); \
        TEST_GROUPBY_FN(std_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,    \
                        Bodo_CTypes::FLOAT64, empty_return_enum::NULL_OUTPUT); \
        TEST_GROUPBY_FN(std_pop_col_set, n,                                    \
                        bodo_array_type::NULLABLE_INT_BOOL,                    \
                        Bodo_CTypes::FLOAT64, empty_return_enum::NULL_OUTPUT); \
        TEST_GROUPBY_FN(skew_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,   \
                        Bodo_CTypes::FLOAT64, empty_return_enum::NULL_OUTPUT); \
        TEST_GROUPBY_FN(kurt_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,   \
                        Bodo_CTypes::FLOAT64, empty_return_enum::NULL_OUTPUT); \
    }
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT8);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT16);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT32);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT64);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT8);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT16);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT32);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT64);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::DECIMAL);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::FLOAT32);
        TEST_MOMENT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::FLOAT64);
    });

    bodo::tests::test("test_groupby_count_sum_mode_first_fns", [] {
#define TEST_COUNT_GROUPBY_FN(arr_type, dtype)                               \
    {                                                                        \
        size_t n = 1;                                                        \
        std::shared_ptr<array_info> in_col =                                 \
            make_all_null_arr<arr_type, dtype>(n);                           \
        auto size_col_set =                                                  \
            new BasicColSet(in_col, Bodo_FTypes::size, false, true);         \
        auto count_col_set =                                                 \
            new BasicColSet(in_col, Bodo_FTypes::count, false, true);        \
        auto mode_col_set = new ModeColSet(in_col, true);                    \
        auto first_col_set = new FirstColSet(in_col, false, true);           \
        TEST_GROUPBY_FN(size_col_set, n, bodo_array_type::NUMPY,             \
                        Bodo_CTypes::INT64, empty_return_enum::ONE);         \
        TEST_GROUPBY_FN(count_col_set, n, bodo_array_type::NUMPY,            \
                        Bodo_CTypes::INT64, empty_return_enum::ZERO);        \
        TEST_GROUPBY_FN(mode_col_set, n, arr_type, dtype,                    \
                        empty_return_enum::NULL_OUTPUT);                     \
        TEST_GROUPBY_FN(first_col_set, n, arr_type, dtype,                   \
                        empty_return_enum::NULL_OUTPUT);                     \
        if (dtype == Bodo_CTypes::_BOOL) {                                   \
            auto count_if_col_set =                                          \
                new BasicColSet(in_col, Bodo_FTypes::count_if, false, true); \
            TEST_GROUPBY_FN(count_if_col_set, n, bodo_array_type::NUMPY,     \
                            Bodo_CTypes::INT64, empty_return_enum::ZERO);    \
        }                                                                    \
        if (float_dtype<dtype> || integer_dtype<dtype>) {                    \
            auto sum_col_set =                                               \
                new BasicColSet(in_col, Bodo_FTypes::sum, false, true);      \
            TEST_GROUPBY_FN(sum_col_set, n,                                  \
                            bodo_array_type::NULLABLE_INT_BOOL, dtype,       \
                            empty_return_enum::NULL_OUTPUT);                 \
        }                                                                    \
    }
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT8);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT16);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT32);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT64);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT8);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT16);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT32);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::UINT64);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::DECIMAL);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::FLOAT32);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::FLOAT64);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::DATE);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::TIME);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::DATETIME);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::_BOOL);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::STRING, Bodo_CTypes::STRING);
        TEST_COUNT_GROUPBY_FN(bodo_array_type::DICT, Bodo_CTypes::STRING);
    });

    bodo::tests::test("test_groupby_bit_bool_fns", [] {
#define TEST_BIT_BOOL_GROUPBY_FN(arr_type, dtype)                              \
    {                                                                          \
        size_t n = 1;                                                          \
        std::shared_ptr<array_info> in_col =                                   \
            make_all_null_arr<arr_type, dtype>(n);                             \
        auto bitor_col_set =                                                   \
            new BasicColSet(in_col, Bodo_FTypes::bitor_agg, false, true);      \
        auto bitand_col_set =                                                  \
            new BasicColSet(in_col, Bodo_FTypes::bitor_agg, false, true);      \
        auto bitxor_col_set =                                                  \
            new BasicColSet(in_col, Bodo_FTypes::bitor_agg, false, true);      \
        if (integer_dtype<dtype>) {                                            \
            TEST_GROUPBY_FN(bitor_col_set, n,                                  \
                            bodo_array_type::NULLABLE_INT_BOOL, dtype,         \
                            empty_return_enum::NULL_OUTPUT);                   \
            TEST_GROUPBY_FN(bitand_col_set, n,                                 \
                            bodo_array_type::NULLABLE_INT_BOOL, dtype,         \
                            empty_return_enum::NULL_OUTPUT);                   \
            TEST_GROUPBY_FN(bitxor_col_set, n,                                 \
                            bodo_array_type::NULLABLE_INT_BOOL, dtype,         \
                            empty_return_enum::NULL_OUTPUT);                   \
        } else if (dtype != Bodo_CTypes::DECIMAL &&                            \
                   dtype != Bodo_CTypes::_BOOL) {                              \
            TEST_GROUPBY_FN(                                                   \
                bitor_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,          \
                Bodo_CTypes::INT64, empty_return_enum::NULL_OUTPUT);           \
            TEST_GROUPBY_FN(                                                   \
                bitand_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,         \
                Bodo_CTypes::INT64, empty_return_enum::NULL_OUTPUT);           \
            TEST_GROUPBY_FN(                                                   \
                bitxor_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,         \
                Bodo_CTypes::INT64, empty_return_enum::NULL_OUTPUT);           \
        }                                                                      \
        if (dtype != Bodo_CTypes::STRING) {                                    \
            auto boolor_col_set =                                              \
                new BasicColSet(in_col, Bodo_FTypes::boolor_agg, false, true); \
            auto booland_col_set = new BasicColSet(                            \
                in_col, Bodo_FTypes::booland_agg, false, true);                \
            auto boolxor_col_set = new BoolXorColSet(                          \
                in_col, Bodo_FTypes::boolxor_agg, false, true);                \
            TEST_GROUPBY_FN(                                                   \
                boolor_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,         \
                Bodo_CTypes::_BOOL, empty_return_enum::NULL_OUTPUT);           \
            TEST_GROUPBY_FN(                                                   \
                booland_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,        \
                Bodo_CTypes::_BOOL, empty_return_enum::NULL_OUTPUT);           \
            TEST_GROUPBY_FN(                                                   \
                boolxor_col_set, n, bodo_array_type::NULLABLE_INT_BOOL,        \
                Bodo_CTypes::_BOOL, empty_return_enum::NULL_OUTPUT);           \
        }                                                                      \
    }
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT8);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT16);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT32);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT64);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT8);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT16);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT32);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT64);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::FLOAT32);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::FLOAT64);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::DECIMAL);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::_BOOL);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::STRING, Bodo_CTypes::STRING);
        TEST_BIT_BOOL_GROUPBY_FN(bodo_array_type::DICT, Bodo_CTypes::STRING);
    });

    bodo::tests::test("test_groupby_select_fns", [] {
#define TEST_SELECT_GROUPBY_FN(arr_type, dtype)                     \
    {                                                               \
        size_t n = 10;                                              \
        std::shared_ptr<array_info> in_col =                        \
            make_all_null_arr<arr_type, dtype>(n);                  \
        auto min_col_set =                                          \
            new BasicColSet(in_col, Bodo_FTypes::min, false, true); \
        auto max_col_set =                                          \
            new BasicColSet(in_col, Bodo_FTypes::max, false, true); \
        auto mode_col_set = new ModeColSet(in_col, false);          \
        TEST_GROUPBY_FN(min_col_set, n, arr_type, dtype,            \
                        empty_return_enum::NULL_OUTPUT);            \
        TEST_GROUPBY_FN(max_col_set, n, arr_type, dtype,            \
                        empty_return_enum::NULL_OUTPUT);            \
        TEST_GROUPBY_FN(mode_col_set, n, arr_type, dtype,           \
                        empty_return_enum::NULL_OUTPUT);            \
    }
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT8);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT16);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT32);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::INT64);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT8);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT16);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT32);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::UINT64);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::FLOAT32);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::FLOAT64);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::DECIMAL);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::_BOOL);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::DATE);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::TIME);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                               Bodo_CTypes::DATETIME);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::STRING, Bodo_CTypes::STRING);
        TEST_SELECT_GROUPBY_FN(bodo_array_type::DICT, Bodo_CTypes::STRING);
    });

    bodo::tests::test("test_groupby_ordinal_fns", [] {
#define TEST_ORDINAL_GROUPBY_FN(arr_type, dtype)                               \
    {                                                                          \
        size_t n = 10;                                                         \
        std::shared_ptr<array_info> in_col =                                   \
            make_all_null_arr<arr_type, dtype>(n);                             \
        std::shared_ptr<array_info> perc_col =                                 \
            make_result_output<bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64,   \
                               empty_return_enum::ONE>(n);                     \
        auto perc_disc_col_set =                                               \
            new PercentileColSet(in_col, perc_col, false, true);               \
        auto perc_cont_col_set =                                               \
            new PercentileColSet(in_col, perc_col, false, true);               \
        auto median_col_set = new MedianColSet(in_col, true, true);            \
        TEST_GROUPBY_FN(perc_disc_col_set, n,                                  \
                        bodo_array_type::NULLABLE_INT_BOOL,                    \
                        Bodo_CTypes::FLOAT64, empty_return_enum::NULL_OUTPUT); \
        TEST_GROUPBY_FN(perc_cont_col_set, n,                                  \
                        bodo_array_type::NULLABLE_INT_BOOL,                    \
                        Bodo_CTypes::FLOAT64, empty_return_enum::NULL_OUTPUT); \
        TEST_GROUPBY_FN(median_col_set, n, bodo_array_type::NULLABLE_INT_BOOL, \
                        Bodo_CTypes::FLOAT64, empty_return_enum::NULL_OUTPUT); \
    }
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT8);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT16);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT32);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT64);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT8);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT16);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT32);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::UINT64);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::FLOAT32);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::FLOAT64);
        TEST_ORDINAL_GROUPBY_FN(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::DECIMAL);
    });

    bodo::tests::test("test_groupby_agg_fns", [] {
        size_t n = 10;
        std::shared_ptr<array_info> str_col =
            make_all_null_arr<bodo_array_type::STRING, Bodo_CTypes::STRING>(n);
        std::shared_ptr<array_info> dict_col =
            make_all_null_arr<bodo_array_type::DICT, Bodo_CTypes::STRING>(n);
        std::shared_ptr<array_info> int_col =
            make_all_null_arr<bodo_array_type::NULLABLE_INT_BOOL,
                              Bodo_CTypes::INT64>(n);

        // [BSE-2182] Ensure LISTAGG with groupby returns "" instead of NULL on
        // all-null inputs. auto listagg_col_set = new ListAggColSet(str_col,
        // dict_col, {int_col}, {true}, {false});
        // TEST_GROUPBY_FN(listagg_col_set, n,
        //             bodo_array_type::STRING,
        //             Bodo_CTypes::STRING, empty_return_enum::EMPTY_STRING);

        auto array_agg_set = new ArrayAggColSet(
            int_col, {int_col}, {true}, {false}, Bodo_FTypes::array_agg, false);
        auto array_agg_distinct_set =
            new ArrayAggColSet(int_col, {int_col}, {true}, {false},
                               Bodo_FTypes::array_agg_distinct, false);
        TEST_GROUPBY_FN(array_agg_set, n, bodo_array_type::ARRAY_ITEM,
                        Bodo_CTypes::INT64, empty_return_enum::EMPTY_ARRAY);
        TEST_GROUPBY_FN(array_agg_distinct_set, n, bodo_array_type::ARRAY_ITEM,
                        Bodo_CTypes::INT64, empty_return_enum::EMPTY_ARRAY);

        auto object_agg_set = new ObjectAggColSet(str_col, int_col, false);
        TEST_GROUPBY_FN(object_agg_set, n, bodo_array_type::MAP,
                        Bodo_CTypes::MAP, empty_return_enum::EMPTY_MAP);
    });
});
