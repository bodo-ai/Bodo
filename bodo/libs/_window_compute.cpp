// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_window_compute.h"
#include <arrow/util/decimal.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_decimal_ext.h"
#include "_gandiva_decimal_copy.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_ftypes.h"
#include "_shuffle.h"
#include "_stream_shuffle.h"
#include "_table_builder.h"
#include "_table_builder_utils.h"
#include "_window_aggfuncs.h"
#include "mpi.h"

std::tuple<int64_t, bodo_array_type::arr_type_enum>
get_update_ftype_idx_arr_type_for_mrnf(size_t n_orderby_arrs,
                                       const std::vector<bool>& asc_vec,
                                       const std::vector<bool>& na_pos_vec) {
    assert(n_orderby_arrs > 0);
    int64_t update_ftype;
    bodo_array_type::arr_type_enum update_idx_arr_type;
    if (n_orderby_arrs == 1) {
        bool asc = asc_vec[0];
        bool na_pos = na_pos_vec[0];
        if (asc) {
            // The first value of an array in ascending order is the
            // min.
            if (na_pos) {
                update_ftype = Bodo_FTypes::idxmin;
                // We don't need null values for indices
                update_idx_arr_type = bodo_array_type::NUMPY;
            } else {
                update_ftype = Bodo_FTypes::idxmin_na_first;
                // We need null values to signal we found an NA
                // value.
                update_idx_arr_type = bodo_array_type::NULLABLE_INT_BOOL;
            }
        } else {
            // The first value of an array in descending order is the
            // max.
            if (na_pos) {
                update_ftype = Bodo_FTypes::idxmax;
                // We don't need null values for indices
                update_idx_arr_type = bodo_array_type::NUMPY;
            } else {
                update_ftype = Bodo_FTypes::idxmax_na_first;
                // We need null values to signal we found an NA
                // value.
                update_idx_arr_type = bodo_array_type::NULLABLE_INT_BOOL;
            }
        }
    } else {
        update_ftype = Bodo_FTypes::idx_n_columns;
        // We don't need null for indices
        update_idx_arr_type = bodo_array_type::NUMPY;
    }
    return std::make_tuple(update_ftype, update_idx_arr_type);
}

void min_row_number_filter_no_sort(
    const std::shared_ptr<array_info>& idx_col,
    std::vector<std::shared_ptr<array_info>>& orderby_cols,
    grouping_info const& grp_info, const std::vector<bool>& asc,
    const std::vector<bool>& na_pos, int update_ftype, bool use_sql_rules,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // To compute min_row_number_filter we want to find the
    // idxmin/idxmax based on the orderby columns. Then in the output
    // array those locations will have the value true. We have already
    // initialized all other locations to false.

    // We initialize the indices to the first row in group. There *must* be one
    // output per partition, so initializing with any row in the  group is
    // correct since we will eventually end up with the correct value once we go
    // over all the rows. This initialization also handles the all tie case,
    // i.e. the value in data_col never needs to get "updated". e.g. If we are
    // doing this on a boolean column and all boolean values for a group are the
    // same as the initial value, we'd never end up updating the row-id in
    // idx_col. If we had initialized all entries in idx-col with say 0, that
    // would give the incorrect output since we'd just end up with all 0s.
    // However, initializing all partitions with a valid row in the partition
    // leads to the correct result.
    if (idx_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // Set all entries to NOT NA since we will be setting all of them
        // to valid values.
        InitializeBitMask((uint8_t*)(idx_col->null_bitmask<
                                     bodo_array_type::NULLABLE_INT_BOOL>()),
                          idx_col->length, true);
    }
    for (size_t group_idx = 0; group_idx < idx_col->length; group_idx++) {
        getv<int64_t>(idx_col, group_idx) =
            grp_info.group_to_first_row[group_idx];
    }

    if (orderby_cols.size() == 0) {
        // Special case: if there are no columns to order by, use the
        // defaults where each group is mapped to an arbitrary row
        // number from that group.
        return;
    } else if (orderby_cols.size() == 1) {
        /// We generate an optimized and templated path for 1 column.
        // Create an array to store min/max value
        std::shared_ptr<array_info> orderby_arr = orderby_cols[0];
        std::shared_ptr<array_info> data_col = alloc_array_top_level(
            grp_info.num_groups, 1, 1, orderby_arr->arr_type,
            orderby_arr->dtype, -1, 0, 0, false, false, false, pool, mm);
        // Initialize the min/max column
        if (update_ftype == Bodo_FTypes::idxmax ||
            update_ftype == Bodo_FTypes::idxmax_na_first) {
            aggfunc_output_initialize(data_col, Bodo_FTypes::max,
                                      use_sql_rules);
        } else {
            aggfunc_output_initialize(data_col, Bodo_FTypes::min,
                                      use_sql_rules);
        }
        std::vector<std::shared_ptr<array_info>> aux_cols = {idx_col};
        // Compute the idxmin/idxmax
        do_apply_to_column(orderby_arr, data_col, aux_cols, grp_info,
                           update_ftype, pool, std::move(mm));
    } else {
        // Call the idx_n_columns function path.
        idx_n_columns_apply(idx_col, orderby_cols, asc, na_pos, grp_info,
                            update_ftype);
    }
}

/**
 * min_row_number_filter is used to evaluate the following type
 * of expression in BodoSQL:
 *
 * row_number() over (partition by ... order by ...) == 1.
 *
 * This function creates a boolean array of all-false, then finds the indices
 * corresponding to the idxmin/idxmax of the orderby columns and sets those
 * indices to true. This implementation does so without sorting the array since
 * if no other window functions being calculated require sorting, then we can
 * find the idxmin/idxmax without bothering to sort the whole table section.
 *
 * @param[in] orderby_arrs: the columns used in the order by clause of the query
 * @param[in,out] out_arr: output array where the true values will be placed
 * @param[in] grp_info: groupby information
 * @param[in] asc_vect: vector indicating which of the orderby columns are
 * ascending
 * @param[in] na_pos_vect: vector indicating which of the orderby columns are
 * null first/last
 * @param is_parallel: is the function being run in parallel?
 * @param use_sql_rules: should initialization functions obey SQL semantics?
 * @param pool Memory pool to use for allocations during the execution of
 * this function.
 * @param mm Memory manager associated with the pool.
 */
void min_row_number_filter_window_computation_no_sort(
    std::vector<std::shared_ptr<array_info>>& orderby_arrs,
    std::shared_ptr<array_info> out_arr, grouping_info const& grp_info,
    const std::vector<bool>& asc_vect, const std::vector<bool>& na_pos_vect,
    bool is_parallel, bool use_sql_rules, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    int ftype;
    bodo_array_type::arr_type_enum idx_arr_type;
    std::tie(ftype, idx_arr_type) = get_update_ftype_idx_arr_type_for_mrnf(
        orderby_arrs.size(), asc_vect, na_pos_vect);
    std::shared_ptr<array_info> idx_col = alloc_array_top_level(
        grp_info.num_groups, 1, 1, idx_arr_type, Bodo_CTypes::UINT64);
    min_row_number_filter_no_sort(idx_col, orderby_arrs, grp_info, asc_vect,
                                  na_pos_vect, ftype, use_sql_rules, pool, mm);
    // Now we have the idxmin/idxmax in the idx_col for each group.
    // We need to set the corresponding indices in the final array to true.
    uint8_t* out_arr_data1 = (uint8_t*)(out_arr->data1());
    for (size_t group_idx = 0; group_idx < idx_col->length; group_idx++) {
        int64_t row_one_idx = getv<int64_t>(idx_col, group_idx);
        SetBitTo(out_arr_data1, row_one_idx, true);
    }
}

/**
 * Alternative implementation for min_row_number_filter if another window
 * computation already requires sorting the table: iterates through
 * the sorted groups and sets the corresponding row to true if the
 * current row belongs to a different group than the previous row.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
void min_row_number_filter_window_computation_already_sorted(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx) {
    int64_t prev_group = -1;
    uint8_t* out_arr_data1 = (uint8_t*)out_arr->data1();
    for (uint64_t i = 0; i < out_arr->length; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        // If the current group is differne from the group of the previous row,
        // then this row is the row where the row number is 1
        if (curr_group != prev_group) {
            int64_t row_one_idx = getv<int64_t>(sorted_idx, i);
            SetBitTo(out_arr_data1, row_one_idx, true);
            prev_group = curr_group;
        }
    }
}

/**
 * Computes the BodoSQL window function ROW_NUMBER() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered togeher) and then by the columns in the
 * orderby clause.
 *
 * The computaiton works by having a counter that starts at 1, increases by
 * 1 each row, and resets to 1 whenever a new group is reached.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
void row_number_computation(std::shared_ptr<array_info> out_arr,
                            std::shared_ptr<array_info> sorted_groups,
                            std::shared_ptr<array_info> sorted_idx) {
    // Start the counter at 1, increase by 1 each time, and reset
    // the counter to 1 when we reach a new group
    int64_t prev_group = -1;
    int64_t row_num = 1;
    for (uint64_t i = 0; i < out_arr->length; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            row_num = 1;
        } else {
            row_num++;
        }
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = row_num;
        // Set the prev group
        prev_group = curr_group;
    }
}

/** Returns whether the current row of orderby keys is distinct from
 * the previous row when performing a rank computation. The templated
 * argument ArrType should only ever be non-unknown if all of the
 * arrays in sorted_orderbys have the same array type.
 *
 * @param[in] sorted_orderbys: the columns used to order the table
 * when performing a window computation.
 * @param i: the row that is being queried to see if it is distinct
 * from the previous row.
 */
template <bodo_array_type::arr_type_enum ArrType>
inline bool distinct_from_previous_row(
    const std::vector<std::shared_ptr<array_info>>& sorted_orderbys,
    int64_t i) {
    return distinct_from_other_row<ArrType>(sorted_orderbys, i, sorted_orderbys,
                                            i - 1);
}
/**
 * Perform the division step for a group once an entire group has had
 * its regular rank values calculated.
 *
 * @param[in,out] out_arr input value, and holds the result
 * @param[in] sorted_idx the array mapping sorted rows back to locaitons in
 * the original array
 * @param[in] rank_arr stores the regular rank values for each row
 * @param[in] group_start_idx the index in rank_arr where the current group
 * begins
 * @param[in] group_end_idx the index in rank_arr where the current group ends
 * @param[in] numerator_offset the amount to subtract from the regular rank
 * @param[in] denominator_offset the amount to subtract from the group size
 *
 * The formula for each row is:
 * out_arr[i] = (r - numerator_offset) / (n - denominator_offset)
 * where r is the rank and  n is the size of the group
 */
inline void rank_division_batch_update(std::shared_ptr<array_info> out_arr,
                                       std::shared_ptr<array_info> sorted_idx,
                                       std::shared_ptr<array_info> rank_arr,
                                       int64_t group_start_idx,
                                       int64_t group_end_idx,
                                       int64_t numerator_offset,
                                       int64_t denominator_offset) {
    // Special case: if the group has size below the offset, set the result to
    // zero
    int64_t group_size = group_end_idx - group_start_idx;
    if (group_size <= denominator_offset) {
        int64_t idx = getv<int64_t>(sorted_idx, group_start_idx);
        getv<double>(out_arr, idx) = 0.0;
    } else {
        // Otherwise, iterate through the entire group that just finished
        for (int64_t j = group_start_idx; j < group_end_idx; j++) {
            int64_t idx = getv<int64_t>(sorted_idx, j);
            getv<double>(out_arr, idx) =
                (static_cast<double>(getv<int64_t>(rank_arr, j)) -
                 numerator_offset) /
                (group_size - denominator_offset);
        }
    }
}

/**
 * Populate the rank values from tie_start_idx to tie_end_idx with
 * the current rank value using the rule all ties are brought upward.
 *
 * @param[in] rank_arr stores the regular rank values for each row
 * @param[in] group_start_idx the index in rank_arr where the current group
 * begins
 * @param[in] tie_start_idx the index in rank_arr where the tie group begins
 * @param[in] tie_end_idx the index in rank_arr where the tie group ends
 */
template <typename T = int64_t>
inline void rank_tie_upward_batch_update(std::shared_ptr<array_info> rank_arr,
                                         int64_t group_start_idx,
                                         int64_t tie_start_idx,
                                         int64_t tie_end_idx,
                                         int64_t extra_rows = 0) {
    assert(rank_arr->arr_type == bodo_array_type::NUMPY);
    T fill_value = static_cast<T>(tie_end_idx - group_start_idx);
    std::fill((T*)(rank_arr->data1<bodo_array_type::NUMPY>()) + tie_start_idx,
              (T*)(rank_arr->data1<bodo_array_type::NUMPY>()) + tie_end_idx,
              fill_value);
}

/**
 * Computes the BodoSQL window function RANK() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered togeher) and then by the columns in the
 * orderby clause.
 *
 * The computaiton works by having a counter that starts at 1, increases
 * whenever the orderby columns change values, and resets to 1 whenever a new
 * group is reached.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_orderbys the vector of sorted orderby columns, used to
 * keep track of when we have encountered a distinct row
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
template <bodo_array_type::arr_type_enum ArrType>
void rank_computation(std::shared_ptr<array_info> out_arr,
                      std::shared_ptr<array_info> sorted_groups,
                      std::vector<std::shared_ptr<array_info>> sorted_orderbys,
                      std::shared_ptr<array_info> sorted_idx) {
    // Start the counter at 1, snap upward each we encounter a row
    // that is distinct from the previous row, and reset the counter
    // to 1 when we reach a new group
    int64_t prev_group = -1;
    int64_t rank_val = 1;
    int64_t group_start_idx = 0;
    int64_t n = out_arr->length;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            rank_val = 1;
            group_start_idx = i;
            // Set the prev group
            prev_group = curr_group;
        } else if (distinct_from_previous_row<ArrType>(sorted_orderbys, i)) {
            rank_val = i - group_start_idx + 1;
        }
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = rank_val;
    }
};

/**
 * Computes the BodoSQL window function DENSE_RANK() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered together) and then by the columns in the
 * orderby clause.
 *
 * The computation works by having a counter that starts at 1, increases by 1
 * the orderby columns change values, and resets to 1 whenever a new group is
 * reached.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_orderbys the vector of sorted orderby columns, used to
 * keep track of when we have encountered a distinct row
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
template <bodo_array_type::arr_type_enum ArrType>
void dense_rank_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::vector<std::shared_ptr<array_info>> sorted_orderbys,
    std::shared_ptr<array_info> sorted_idx) {
    // Start the counter at 1, increase by 1 each we encounter a row
    // that is distinct from the previous row, and reset the counter
    // to 1 when we reach a new group
    int64_t prev_group = -1;
    int64_t rank_val = 1;
    int64_t n = out_arr->length;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            rank_val = 1;
            // Set the prev group
            prev_group = curr_group;
        } else if (distinct_from_previous_row<ArrType>(sorted_orderbys, i)) {
            rank_val++;
        }
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = rank_val;
    }
};

/**
 * Computes the BodoSQL window function PERCENT_RANK() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered together) and then by the columns in the
 * orderby clause.
 *
 * The computation works by calculating the regular rank for each row. Then,
 * after a group is finished, the percent rank for each row is calculated
 * via the formula (r-1)/(n-1) where r is the rank and n is the group size.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_orderbys the vector of sorted orderby columns, used to
 * keep track of when we have encountered a distinct row
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
template <bodo_array_type::arr_type_enum ArrType>
void percent_rank_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::vector<std::shared_ptr<array_info>> sorted_orderbys,
    std::shared_ptr<array_info> sorted_idx) {
    // Create an intermediary column to store the regular rank
    int64_t n = out_arr->length;
    std::shared_ptr<array_info> rank_arr = alloc_array_top_level(
        n, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::INT64);
    // Start the counter at 1, snap upward each we encounter a row
    // that is distinct from the previous row, and reset the counter
    // to 1 when we reach a new group. When a group ends, set
    // all rows in the output table to (r-1)/(n-1) where r is
    // the regular rank value and n is the group size (or 0 if n=1)
    int64_t prev_group = -1;
    int64_t rank_val = 1;
    int64_t group_start_idx = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if ((curr_group != prev_group)) {
            rank_val = 1;
            // Update the group that just finished by calculating
            // (r-1)/(n-1)
            rank_division_batch_update(out_arr, sorted_idx, rank_arr,
                                       group_start_idx, i, 1, 1);
            group_start_idx = i;
            // Set the prev group
            prev_group = curr_group;
        } else if (distinct_from_previous_row<ArrType>(sorted_orderbys, i)) {
            rank_val = i - group_start_idx + 1;
        }
        getv<int64_t>(rank_arr, i) = rank_val;
    }
    // Repeat the group ending procedure after the main loop finishes
    rank_division_batch_update(out_arr, sorted_idx, rank_arr, group_start_idx,
                               n, 1, 1);
};

/**
 * Computes the BodoSQL window function CUME_DIST() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered together) and then by the columns in the
 * orderby clause.
 *
 * The computation works by calculating the tie-upward rank for each row. Then,
 * after a group is finished, the percent rank for each row is calculated
 * via the formula r/n where r is the tie-upward rank and n is the group size.
 *
 * Suppose the sorted values in an array are as follows:
 * ["A", "B", "B", "B", "C", "C", "D", "E", "E", "E"]
 *
 * The regular rank would be as follows:
 * [1, 2, 2, 2, 5, 5, 7, 8, 8, 8]
 *
 * But the tie-upward rank is as follows:
 * [1, 4, 4, 4, 6, 6, 7, 10, 10, 10]
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_orderbys the vector of sorted orderby columns, used to
 * keep track of when we have encountered a distinct row
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
template <bodo_array_type::arr_type_enum ArrType>
void cume_dist_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::vector<std::shared_ptr<array_info>> sorted_orderbys,
    std::shared_ptr<array_info> sorted_idx) {
    // Create an intermediary column to store the tie-up rank
    int64_t n = out_arr->length;
    std::shared_ptr<array_info> rank_arr = alloc_array_top_level(
        n, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::INT64);
    // Start the counter at 1, snap upward each we encounter a row
    // that is distinct from the previous row, and reset the counter
    // to 1 when we reach a new group. When a group ends, set
    // all rows in the output table to r/n where r is
    // the tie-up rank value and n is the group size
    int64_t prev_group = -1;
    int64_t group_start_idx = 0;
    int64_t tie_start_idx = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if ((curr_group != prev_group)) {
            // Update the group of ties that just finished by setting
            // all of them to the rank value of the last position
            rank_tie_upward_batch_update(rank_arr, group_start_idx,
                                         tie_start_idx, i);
            // Update the group that just finished by calculating
            // r/n
            rank_division_batch_update(out_arr, sorted_idx, rank_arr,
                                       group_start_idx, i, 0, 0);
            group_start_idx = i;
            tie_start_idx = i;
            // Set the prev group
            prev_group = curr_group;
        } else if (distinct_from_previous_row<ArrType>(sorted_orderbys, i)) {
            // Update the group of ties that just finished by setting
            // all of them to the rank value of the last position
            rank_tie_upward_batch_update(rank_arr, group_start_idx,
                                         tie_start_idx, i);
            tie_start_idx = i;
        }
    }
    // Repeat the tie ending procedure after the final group finishes
    rank_tie_upward_batch_update(rank_arr, group_start_idx, tie_start_idx, n);
    // Repeat the group ending procedure after the final group finishes
    rank_division_batch_update(out_arr, sorted_idx, rank_arr, group_start_idx,
                               n, 0, 0);
}

// Wrapper function for the 4 rank-based window function computations.
template <Bodo_FTypes::FTypeEnum ftype>
void rank_fn_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::vector<std::shared_ptr<array_info>> sorted_orderbys,
    std::shared_ptr<array_info> sorted_idx) {
#define RANK_FN_ATYPE_CASE(ArrType, func)                                \
    case ArrType: {                                                      \
        bool can_use_arr_type = true;                                    \
        for (size_t order_col = 1; order_col < sorted_orderbys.size();   \
             order_col++) {                                              \
            if (sorted_orderbys[order_col]->arr_type != ArrType) {       \
                can_use_arr_type = false;                                \
                break;                                                   \
            }                                                            \
        }                                                                \
        if (can_use_arr_type) {                                          \
            func<ArrType>(out_arr, sorted_groups, sorted_orderbys,       \
                          sorted_idx);                                   \
        } else {                                                         \
            func<bodo_array_type::UNKNOWN>(out_arr, sorted_groups,       \
                                           sorted_orderbys, sorted_idx); \
        }                                                                \
        break;                                                           \
    }
#define RANK_FN_FTYPE_CASE(rank_ftype, func)                                  \
    case rank_ftype: {                                                        \
        if (sorted_orderbys.size() == 0) {                                    \
            func<bodo_array_type::UNKNOWN>(out_arr, sorted_groups,            \
                                           sorted_orderbys, sorted_idx);      \
        } else {                                                              \
            switch (sorted_orderbys[0]->arr_type) {                           \
                RANK_FN_ATYPE_CASE(bodo_array_type::NULLABLE_INT_BOOL, func); \
                RANK_FN_ATYPE_CASE(bodo_array_type::NUMPY, func);             \
                RANK_FN_ATYPE_CASE(bodo_array_type::STRING, func);            \
                RANK_FN_ATYPE_CASE(bodo_array_type::DICT, func);              \
                RANK_FN_ATYPE_CASE(bodo_array_type::TIMESTAMPTZ, func);       \
                RANK_FN_ATYPE_CASE(bodo_array_type::ARRAY_ITEM, func);        \
                RANK_FN_ATYPE_CASE(bodo_array_type::STRUCT, func);            \
                RANK_FN_ATYPE_CASE(bodo_array_type::MAP, func);               \
                default: {                                                    \
                    throw std::runtime_error(                                 \
                        "Unsupported ArrType for rank computation: " +        \
                        GetArrType_as_string(sorted_orderbys[0]->arr_type));  \
                }                                                             \
            }                                                                 \
        }                                                                     \
        break;                                                                \
    }
    switch (ftype) {
        RANK_FN_FTYPE_CASE(Bodo_FTypes::rank, rank_computation);
        RANK_FN_FTYPE_CASE(Bodo_FTypes::dense_rank, dense_rank_computation);
        RANK_FN_FTYPE_CASE(Bodo_FTypes::percent_rank, percent_rank_computation);
        RANK_FN_FTYPE_CASE(Bodo_FTypes::cume_dist, cume_dist_computation);
        default: {
            throw std::runtime_error(
                "Unsupported ftype for rank computation: " +
                get_name_for_Bodo_FTypes(ftype));
        }
    }
#undef rank_fn_atype_case
#undef define
}

void ntile_batch_update(std::shared_ptr<array_info> out_arr,
                        std::shared_ptr<array_info> sorted_idx,
                        int64_t group_start_idx, int64_t group_end_idx,
                        int num_bins) {
    // Calculate the number of items in the group, the number
    // of elements that will go into small vs large groups, and
    // the number of bins that will require a larger group
    int n = group_end_idx - group_start_idx;
    if (n == 0)
        return;
    int remainder = n % num_bins;
    int n_smaller = n / num_bins;
    int n_larger = n_smaller + (remainder ? 1 : 0);

    // Calculate the indices of bins that will use the large group
    // vs the small group
    int hi_cutoff = std::min(n, num_bins) + 1;
    int lo_cutoff = std::min(remainder, hi_cutoff) + 1;

    // For each bin from 1 to lo_cutoff, assign the next n_larger
    // indices to the current bin
    int bin_start_index = group_start_idx;
    for (int bin = 1; bin < lo_cutoff; bin++) {
        int bin_end_index = bin_start_index + n_larger;
        for (int i = bin_start_index; i < bin_end_index; i++) {
            // Get the index in the output array.
            int64_t idx = getv<int64_t>(sorted_idx, i);
            getv<int64_t>(out_arr, idx) = bin;
        }
        bin_start_index = bin_end_index;
    }

    // For each bin from lo_cutoff to hi_cutoff, assign the next n_smaller
    // indices to the current bin
    for (int64_t bin = lo_cutoff; bin < hi_cutoff; bin++) {
        int bin_end_index = bin_start_index + n_smaller;
        for (int i = bin_start_index; i < bin_end_index; i++) {
            // Get the index in the output array.
            int64_t idx = getv<int64_t>(sorted_idx, i);
            getv<int64_t>(out_arr, idx) = bin;
        }
        bin_start_index = bin_end_index;
    }
}

void ntile_computation(std::shared_ptr<array_info> out_arr,
                       std::shared_ptr<array_info> sorted_groups,
                       std::shared_ptr<array_info> sorted_idx,
                       int64_t num_bins) {
    // Each time we find the end of a group, invoke the ntile
    // procedure on all the rows between that index and
    // the index where the group started
    int64_t n = out_arr->length;
    int64_t prev_group = -1;
    int64_t group_start_idx = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            ntile_batch_update(out_arr, sorted_idx, group_start_idx, i,
                               num_bins);
            // Set the prev group
            prev_group = curr_group;
            group_start_idx = i;
        }
    }
    // Repeat the ntile batch procedure at the end on the final group
    ntile_batch_update(out_arr, sorted_idx, group_start_idx, n, num_bins);
}

/**
 * Computes the BodoSQL window function CONDITIONAL_TRUE_EVENT(A) on a
 * subset of a table containing complete partitions, where the rows are
 * sorted first by group (so each partition is clustered togeher) and then
 * by the columns in the orderby clause.
 *
 * The computaiton works by starting a counter at zero, resetting it to
 * zero each time a new group is reached, and otherwise only incrementing
 * the counter when the current row of the input column is set to true.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 * @param[in] input_col: the boolean array whose values are used to calculate
 * the conditional_true_event calculation
 */
void conditional_true_event_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx,
    std::shared_ptr<array_info> input_col) {
    int64_t n = out_arr->length;
    int64_t prev_group = -1;
    int64_t counter = 0;
    const uint8_t* input_col_data1 = (uint8_t*)input_col->data1();
    const uint8_t* input_col_null_bitmask =
        (uint8_t*)(input_col->null_bitmask());
    for (int64_t i = 0; i < n; i++) {
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        // If we have crossed the threshold into a new group,
        // reset the counter to zero
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            prev_group = curr_group;
            counter = 0;
        }
        // If the current row is true, increment the counter
        if (input_col->arr_type == bodo_array_type::NUMPY) {
            if (getv<uint8_t, bodo_array_type::NUMPY>(input_col, idx)) {
                counter++;
            }
        } else {
            if (GetBit(input_col_null_bitmask, idx) &&
                GetBit(input_col_data1, idx)) {
                counter++;
            }
        }
        getv<int64_t>(out_arr, idx) = counter;
    }
}

/**
 * Computes the BodoSQL window function CONDITIONAL_CHANGE_EVENT(A) on a
 * subset of a table containing complete partitions, where the rows are
 * sorted first by group (so each partition is clustered togeher) and then
 * by the columns in the orderby clause.
 *
 * The computaiton works by starting a counter at zero, resetting it to
 * zero each time a new group is reached, and otherwise only incrementing
 * the counter when the current row of the input column is a non-null
 * value that is distinct from the most recent non-null value.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 * @param[in] input_col: the array whose values are used to calculate
 * the conditional_true_event calculation
 */
void conditional_change_event_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx,
    std::shared_ptr<array_info> input_col) {
    int64_t n = out_arr->length;
    int64_t prev_group = -1;
    int64_t counter = 0;
    int64_t last_non_null = -1;
    for (int64_t i = 0; i < n; i++) {
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        // If we have crossed the threshold into a new group,
        // reset the counter to zero
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            prev_group = curr_group;
            counter = 0;
            last_non_null = -1;
        }
        // If the current row is non-null and does not equal
        // the most recent non-null row, increment the counter.
        // XXX TODO We need to template both the get_null_bit and the
        // TestEqualColumn calls!
        if (input_col->arr_type == bodo_array_type::NUMPY ||
            input_col->get_null_bit(idx)) {
            if (last_non_null != -1 &&
                !TestEqualColumn(input_col, idx, input_col, last_non_null,
                                 true)) {
                counter++;
            }
            last_non_null = idx;
        }
        getv<int64_t>(out_arr, idx) = counter;
    }
}

void window_computation(std::vector<std::shared_ptr<array_info>>& input_arrs,
                        std::vector<int64_t> window_funcs,
                        std::vector<std::shared_ptr<array_info>> out_arrs,
                        grouping_info const& grp_info,
                        const std::vector<bool>& asc_vect,
                        const std::vector<bool>& na_pos_vect,
                        const std::vector<void*>& window_args, int n_input_cols,
                        bool is_parallel, bool use_sql_rules,
                        bodo::IBufferPool* const pool,
                        std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t window_arg_offset = 0;
    int64_t window_col_offset = input_arrs.size() - n_input_cols;
    std::vector<std::shared_ptr<array_info>> orderby_arrs(
        input_arrs.begin(), input_arrs.begin() + window_col_offset);
    std::shared_ptr<table_info> iter_table = nullptr;
    // The sort table will be the same for every window function call that uses
    // it, so the table will be uninitialized until one of the calls specifies
    // that we do need to sort
    bool needs_sort = false;
    int64_t idx_col = 0;
    for (size_t i = 0; i < window_funcs.size(); i++) {
        if (window_funcs[i] != Bodo_FTypes::min_row_number_filter) {
            needs_sort = true;
            /* If this is the first function encountered that requires,
             * a sort, create a sorted table with the following columns:
             * - 1 column containing the group numbers so that when the
             *   table is sorted, each partition has its rows clustered
             * together
             * - 1 column for each of the orderby cols so that within each
             *   partition, the values are sorted as desired
             * - 1 extra column that is set to 0...n-1 so that when it is
             *   sorted, we have a way of converting rows in the sorted
             *   table back to rows in the original table.
             */
            size_t num_rows = grp_info.row_to_group.size();
            idx_col = orderby_arrs.size() + 1;
            std::shared_ptr<array_info> idx_arr =
                alloc_numpy(num_rows, Bodo_CTypes::INT64, pool, mm);
            for (size_t i = 0; i < num_rows; i++) {
                getv<int64_t>(idx_arr, i) = i;
            }
            iter_table =
                grouped_sort(grp_info, orderby_arrs, {idx_arr}, asc_vect,
                             na_pos_vect, 0, is_parallel, pool, mm);
            break;
        }
    }
    int64_t* frame_lo;
    int64_t* frame_hi;
    // For each window function call, compute the answer using the
    // sorted table to lookup the rows in the original ordering
    // that are to be modified
    for (size_t i = 0; i < window_funcs.size(); i++) {
        switch (window_funcs[i]) {
            // min_row_number_filter uses a sort-less implementation if no
            // other window functions being calculated require sorting. However,
            // if another window function in this computation requires sorting
            // the table, then we can just use the sorted groups instead.
            case Bodo_FTypes::min_row_number_filter: {
                if (needs_sort) {
                    min_row_number_filter_window_computation_already_sorted(
                        out_arrs[i], iter_table->columns[0],
                        iter_table->columns[idx_col]);
                } else {
                    min_row_number_filter_window_computation_no_sort(
                        orderby_arrs, out_arrs[i], grp_info, asc_vect,
                        na_pos_vect, is_parallel, use_sql_rules, pool, mm);
                }
                break;
            }
            case Bodo_FTypes::row_number: {
                row_number_computation(out_arrs[i], iter_table->columns[0],
                                       iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::rank: {
                rank_fn_computation<Bodo_FTypes::rank>(
                    out_arrs[i], iter_table->columns[0],
                    std::vector<std::shared_ptr<array_info>>(
                        iter_table->columns.begin() + 1,
                        iter_table->columns.begin() + idx_col),
                    iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::dense_rank: {
                rank_fn_computation<Bodo_FTypes::dense_rank>(
                    out_arrs[i], iter_table->columns[0],
                    std::vector<std::shared_ptr<array_info>>(
                        iter_table->columns.begin() + 1,
                        iter_table->columns.begin() + idx_col),
                    iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::percent_rank: {
                rank_fn_computation<Bodo_FTypes::percent_rank>(
                    out_arrs[i], iter_table->columns[0],
                    std::vector<std::shared_ptr<array_info>>(
                        iter_table->columns.begin() + 1,
                        iter_table->columns.begin() + idx_col),
                    iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::cume_dist: {
                rank_fn_computation<Bodo_FTypes::cume_dist>(
                    out_arrs[i], iter_table->columns[0],
                    std::vector<std::shared_ptr<array_info>>(
                        iter_table->columns.begin() + 1,
                        iter_table->columns.begin() + idx_col),
                    iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::ntile: {
                ntile_computation(out_arrs[i], iter_table->columns[0],
                                  iter_table->columns[idx_col],
                                  *((int64_t*)window_args[window_arg_offset]));
                window_arg_offset++;
                break;
            }
            case Bodo_FTypes::conditional_true_event: {
                conditional_true_event_computation(
                    out_arrs[i], iter_table->columns[0],
                    iter_table->columns[idx_col],
                    input_arrs[window_col_offset]);
                window_col_offset++;
                break;
            }
            case Bodo_FTypes::conditional_change_event: {
                conditional_change_event_computation(
                    out_arrs[i], iter_table->columns[0],
                    iter_table->columns[idx_col],
                    input_arrs[window_col_offset]);
                window_col_offset++;
                break;
            }
            // Size has no column argument so the output column
            // is used as a dummy value
            case Bodo_FTypes::size: {
                frame_lo = (int64_t*)window_args[window_arg_offset];
                window_arg_offset++;
                frame_hi = (int64_t*)window_args[window_arg_offset];
                window_arg_offset++;
                window_frame_computation(out_arrs[i], out_arrs[i],
                                         iter_table->columns[0],
                                         iter_table->columns[idx_col], frame_lo,
                                         frame_hi, Bodo_FTypes::size);
                break;
            }
            // Window functions that optionally allow a window frame
            case Bodo_FTypes::first:
            case Bodo_FTypes::last:
            case Bodo_FTypes::count:
            case Bodo_FTypes::count_if:
            case Bodo_FTypes::var:
            case Bodo_FTypes::var_pop:
            case Bodo_FTypes::std:
            case Bodo_FTypes::std_pop:
            case Bodo_FTypes::mean: {
                frame_lo = (int64_t*)window_args[window_arg_offset];
                window_arg_offset++;
                frame_hi = (int64_t*)window_args[window_arg_offset];
                window_arg_offset++;
                window_frame_computation(input_arrs[window_col_offset],
                                         out_arrs[i], iter_table->columns[0],
                                         iter_table->columns[idx_col], frame_lo,
                                         frame_hi, window_funcs[i]);
                window_col_offset++;
                break;
            }
            // Window functions that only support partition-wide aggregation
            case Bodo_FTypes::ratio_to_report:
            case Bodo_FTypes::any_value: {
                window_frame_computation(input_arrs[window_col_offset],
                                         out_arrs[i], iter_table->columns[0],
                                         iter_table->columns[idx_col], nullptr,
                                         nullptr, window_funcs[i]);
                window_col_offset++;
                break;
            }
            default:
                throw std::runtime_error(
                    "Invalid window function: " +
                    std::string(get_name_for_Bodo_FTypes(window_funcs[i])));
        }
    }
}

/**
 * @brief Extracts a consecutive range of columns from a table.
 *
 * @param table The input table.
 * @param start_idx The idx of the first column to extract from the table (i.e
 * table->columns[start_idx]).
 * @param length The number of columns to extract.
 * @return std::vector<std::shared_ptr<array_info>> a vector of columns.
 */
std::vector<std::shared_ptr<array_info>> extract_columns(
    std::shared_ptr<table_info> table, size_t start_idx, size_t length) {
    // TODO move to _array.cpp ?
    assert(start_idx + length <= table->ncols());
    std::vector<std::shared_ptr<array_info>> out_cols(length);
    for (size_t i = 0; i < length; i++) {
        out_cols[i] = table->columns[i + start_idx];
    }
    return out_cols;
}

/**
 * @brief Implement a local implementation of dense
 * rank on an already sorted table. The partition by and order
 * by arrays are used to determine equality.
 * @tparam PartitionByArrType A single array type for the partition
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @param[in] partition_by_arrs The arrays used in partition by.
 * @param[in] order_by_arrs The arrays used in order by.
 * @param[out] out_arr The pre-allocated output array to update.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(dense_rank<window_func>)
void local_sorted_window_fn(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& out_arr) {
    uint64_t rank_val = 1;
    int64_t n = out_arr->length;
    for (int64_t i = 0; i < n; i++) {
        if (distinct_from_previous_row<PartitionByArrType>(partition_by_arrs,
                                                           i)) {
            rank_val = 1;
        } else if (distinct_from_previous_row<OrderByArrType>(order_by_arrs,
                                                              i)) {
            rank_val++;
        }
        getv<uint64_t, bodo_array_type::NUMPY>(out_arr, i) = rank_val;
    }
}

template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(row_number<window_func>)
void local_sorted_window_fn(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& out_arr) {
    uint64_t rank_val = 1;
    int64_t n = out_arr->length;
    for (int64_t i = 0; i < n; i++) {
        if (distinct_from_previous_row<PartitionByArrType>(partition_by_arrs,
                                                           i)) {
            rank_val = 1;
        } else {
            rank_val++;
        }
        getv<uint64_t, bodo_array_type::NUMPY>(out_arr, i) = rank_val;
    }
}

template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(rank<window_func>)
uint64_t local_sorted_window_fn(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& out_arr) {
    uint64_t rank_val = 1;
    uint64_t row_number = 1;
    int64_t n = out_arr->length;
    for (int64_t i = 0; i < n; i++) {
        row_number++;
        if (distinct_from_previous_row<PartitionByArrType>(partition_by_arrs,
                                                           i)) {
            rank_val = 1;
            row_number = 1;
        } else if (distinct_from_previous_row<OrderByArrType>(order_by_arrs,
                                                              i)) {
            rank_val = row_number;
        }
        getv<uint64_t, bodo_array_type::NUMPY>(out_arr, i) = rank_val;
    }
    return row_number;
}

/**
 * @brief Helper for subtracting 1.0 from a range of consecutive ranks and
 * dividing them by group size - 1.0.
 *
 * @param[in, out] out_arr The array of ranks.
 * @param[in] group_start_idx The starting index of the group.
 * @param[in] group_end_idx The index of the start of the next group.
 */
void sorted_rank_division_batch_update(std::shared_ptr<array_info> out_arr,
                                       int64_t group_start_idx,
                                       int64_t group_end_idx,
                                       int64_t extra_rows = 0,
                                       double rank_offset = 0.0) {
    int64_t group_size = (group_end_idx - group_start_idx) + extra_rows;
    if (group_size <= 1) {
        getv<double, bodo_array_type::NUMPY>(out_arr, group_start_idx) = 0.0;
        return;
    }
    for (int64_t i = group_start_idx; i < group_end_idx; i++) {
        double rank_val = getv<double, bodo_array_type::NUMPY>(out_arr, i);
        getv<double, bodo_array_type::NUMPY>(out_arr, i) = static_cast<double>(
            (rank_val + rank_offset - 1.0) / (group_size - 1.0));
    }
}

/**
 * @brief Creates a table containing information from the local computation
 * phase of percent rank that needs to be sent to neighbors
 *
 * @param partition_by_arrs The partition by columns.
 * @param order_by_arrs The order by columns.
 * @param out_arr The output of the local percent rank computation.
 * @param first_group_size The size of the first group.
 * @param last_group_size The size of the last group.
 * @param last_rank The rank of the last element in the output array.
 *
 * @returns A table containing partition by, order by columns, the maximum rank,
 * and group size information.
 */
template <uint32_t window_func>
    requires(percent_rank<window_func>)
std::shared_ptr<table_info> create_boundary_table(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& out_arr, int64_t first_group_size,
    int64_t last_group_size, double last_rank) {
    // output table will look like :
    // row 1: first row of partition cols, orderby cols (unused), first rank
    // (unused), first group size
    // row 2: last row of partition cols , last row of orderby cols
    // , last rank , last group size.
    std::vector<std::shared_ptr<array_info>> data_arrs;
    for (auto arr : partition_by_arrs) {
        data_arrs.push_back(arr);
    }
    for (auto arr : order_by_arrs) {
        data_arrs.push_back(arr);
    }
    data_arrs.push_back(out_arr);

    std::vector<int64_t> indices = {0,
                                    static_cast<int64_t>(out_arr->length - 1)};
    std::shared_ptr<table_info> data_table =
        std::make_shared<table_info>(data_arrs);
    std::shared_ptr<table_info> output_table =
        RetrieveTable(data_table, indices);

    // create a column for group sizes
    std::shared_ptr<array_info> group_size_arr =
        alloc_numpy(2, Bodo_CTypes::INT64);
    getv<int64_t>(group_size_arr, 0) = first_group_size;
    getv<int64_t>(group_size_arr, 1) = last_group_size;
    output_table->columns.push_back(group_size_arr);

    // convert back to rank (simplifies calculations the communication phase)
    getv<double>(output_table->columns[output_table->ncols() - 2], 1) =
        last_rank;

    return output_table;
}

/**
 * @brief Creates table of boundary information to send to previous/next
 * neighbors for cume dist.
 *
 * @param partition_by_arrs The partition by columns.
 * @param order_by_arrs The order by columns.
 * @param out_arr The output of the window function.
 * @param first_group_size Size of the first group.
 * @param last_group_size Size of the last group.
 * @param num_ties Number of elements that tie with the first order by of the
 * first group.
 * @return std::shared_ptr<table_info> Table containing the partition by, order
 * by, number of ties with the first order by, and group size for the first/last
 * rows.
 */
template <uint32_t window_func>
    requires(cume_dist<window_func>)
std::shared_ptr<table_info> create_boundary_table(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& out_arr, int64_t first_group_size,
    int64_t last_group_size, int64_t num_ties) {
    std::shared_ptr<table_info> output_table =
        create_boundary_table<Bodo_FTypes::percent_rank>(
            partition_by_arrs, order_by_arrs, out_arr, first_group_size,
            last_group_size, 0.0);

    getv<double>(output_table->columns[output_table->ncols() - 2], 0) =
        static_cast<double>(num_ties);

    return output_table;
}

/**
 * @brief Computes percent rank on local data using the formula (rank - 1) / (n
 * - 1). If the is_parallel flag is set, returns a table containing information
 * to send to the neighboring ranks.
 *
 * @tparam PartitionByArrType A single array type for the partition
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @param[in] partition_by_arrs The partition by columns.
 * @param[in] order_by_arrs The order by columns.
 * @param[out] out_arr The window output column to populate.
 * @param[in] is_parallel Whether to return information to send to neighboring
 * ranks.
 * @return std::shared_ptr<table_info> Information to send to neighboring ranks
 * if is_parallel flag is set, nullptr otherwise.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(percent_rank<window_func>)
std::shared_ptr<table_info> local_sorted_window_fn(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& out_arr, bool is_parallel = false) {
    // idxs for computing the size of the first/last group (useful for
    // communication)
    int64_t first_group_ends_idx = 0;
    int64_t curr_group_start_idx = 0;

    int64_t rank_val = 1;

    int64_t n = out_arr->length;

    for (int64_t i = 0; i < n; i++) {
        if (distinct_from_previous_row<PartitionByArrType>(partition_by_arrs,
                                                           i)) {
            if (first_group_ends_idx == 0)
                first_group_ends_idx = i;
            // Update the group that just finished by calculating
            // (r-1)/(n-1)
            sorted_rank_division_batch_update(out_arr, curr_group_start_idx, i);
            curr_group_start_idx = i;
            rank_val = 1;
        } else if (distinct_from_previous_row<OrderByArrType>(order_by_arrs,
                                                              i)) {
            rank_val = i - curr_group_start_idx + 1;
        }
        getv<double, bodo_array_type::NUMPY>(out_arr, i) =
            static_cast<double>(rank_val);
    }
    // Update the last group
    sorted_rank_division_batch_update(out_arr, curr_group_start_idx, n);

    if (!is_parallel)
        return nullptr;

    // if the first group ends idx is still at 0, then there is only one group
    if (first_group_ends_idx == 0) {
        first_group_ends_idx = n;
    }

    // construct output table to send to other ranks
    return create_boundary_table<window_func>(
        partition_by_arrs, order_by_arrs, out_arr, first_group_ends_idx,
        n - curr_group_start_idx, static_cast<double>(rank_val));
}

template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(cume_dist<window_func>)
std::shared_ptr<table_info> local_sorted_window_fn(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& out_arr, bool is_parallel = false) {
    int64_t n = out_arr->length;
    int64_t curr_group_start_idx = 0;
    int64_t first_group_ends_idx = 0;

    // tracking ties for rank update
    int64_t curr_tie_start_idx = 0;
    int64_t first_tie_ends_idx = 0;

    for (int64_t i = 0; i < n; i++) {
        if (distinct_from_previous_row<PartitionByArrType>(partition_by_arrs,
                                                           i)) {
            if (first_group_ends_idx == 0)
                first_group_ends_idx = i;

            rank_tie_upward_batch_update<double>(out_arr, curr_group_start_idx,
                                                 curr_tie_start_idx, i);
            // Update the group that just finished by calculating
            // r / n
            sorted_rank_division_batch_update(out_arr, curr_group_start_idx, i,
                                              1, 1.0);
            curr_tie_start_idx = i;
            curr_group_start_idx = i;
        } else if (distinct_from_previous_row<OrderByArrType>(order_by_arrs,
                                                              i)) {
            if (first_tie_ends_idx == 0)
                first_tie_ends_idx = i;

            rank_tie_upward_batch_update<double>(out_arr, curr_group_start_idx,
                                                 curr_tie_start_idx, i);

            curr_tie_start_idx = i;
        }
    }

    // if the first group ends idx is still at 0, then there is only one group
    if (first_group_ends_idx == 0)
        first_group_ends_idx = n;

    if (first_tie_ends_idx == 0)
        first_tie_ends_idx = n;

    rank_tie_upward_batch_update<double>(out_arr, curr_group_start_idx,
                                         curr_tie_start_idx, n);
    sorted_rank_division_batch_update(out_arr, curr_group_start_idx, n, 1, 1.0);

    if (!is_parallel)
        return nullptr;

    return create_boundary_table<Bodo_FTypes::cume_dist>(
        partition_by_arrs, order_by_arrs, out_arr, first_group_ends_idx,
        n - curr_group_start_idx, first_tie_ends_idx);
}
/**
 * @brief Receive data from a neighboring rank if it exists
 * using sorted window functions.
 *
 * @param[in] empty_table An empty table with the same schema as
 * the one that should be received.
 * @param[out] partition_by_arrs The partition by columns, transposed with the
 * updated dict builders
 * @param[out] order_by_arrs The order by columns, transposed with the updated
 * dict builders
 * @param[in] recv_from The rank we are receiving data from.
 * @return std::shared_ptr<table_info> The received table info
 * or an empty table if there is no data to receive.
 */
std::shared_ptr<table_info> recv_sorted_window_data(
    const std::shared_ptr<table_info>& empty_table,
    std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    std::vector<std::shared_ptr<DictionaryBuilder>> builders, int recv_from) {
    int my_rank;
    int n_pes;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    if (recv_from < 0 || recv_from >= n_pes) {
        // recv_from is out of bounds.
        return empty_table;
    } else {
        // First recv the data length.
        int recv_data_size = -1;
        MPI_Recv(&recv_data_size, 1, MPI_INT, recv_from, 0, MPI_COMM_WORLD,
                 MPI_STATUSES_IGNORE);
        if (recv_data_size == 0) {
            return empty_table;
        } else {
            std::vector<AsyncShuffleRecvState> recv_states;
            while (recv_states.size() == 0) {
                // Buffer until we receive the first message
                shuffle_irecv(empty_table, MPI_COMM_WORLD, recv_states);
            }
            std::unique_ptr<bodo::Schema> schema = empty_table->schema();
            // Note: assuming that the only dictionary encoded columns come
            // from the partition/orderby columns, not the function columns.
            std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders(
                empty_table->columns.size());
            for (size_t i = 0;
                 i < std::min(dict_builders.size(), builders.size()); i++) {
                dict_builders[i] = builders[i];
            }
            for (size_t i = builders.size(); i < dict_builders.size(); i++) {
                dict_builders[i] = create_dict_builder_for_array(
                    schema->column_types[i]->copy(), false);
            }

            TableBuildBuffer result_table(std::move(schema), dict_builders);
            IncrementalShuffleMetrics metrics;
            while (recv_states.size() != 0) {
                consume_completed_recvs(recv_states, dict_builders, metrics,
                                        result_table);
            }

            // Transpose the partition by and order by columns with the updated
            // dictionary builders
            for (size_t i = 0; i < partition_by_arrs.size(); ++i) {
                if (builders[i] != nullptr) {
                    partition_by_arrs[i] =
                        builders[i]->TransposeExisting(partition_by_arrs[i]);
                }
            }
            for (size_t i = 0; i < order_by_arrs.size(); ++i) {
                if (builders[i + partition_by_arrs.size()] != nullptr) {
                    order_by_arrs[i] =
                        builders[i + partition_by_arrs.size()]
                            ->TransposeExisting(order_by_arrs[i]);
                }
            }

            return result_table.data_table;
        }
    }
}

/**
 * @brief Combines local data with boundary data in the case where we have a
 * single group that matches with it's neighbor(s)
 *
 * @tparam PartitionByArrType A single array type for the partition
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @param[in] partition_by_arrs The partition by columns.
 * @param[in] order_by_arrs The order by columns.
 * @param[in,out] output_table The combined result of recv_boundary_data and
 * local data has the form (partition_by_cols, orderby_cols, max_rank, num_rows)
 * @param[in] last_group_size_arr The array containing the locally computed
 * size of the last group.
 * @param[in] recv_boundary_data A table of data we received from our
 * neighboring rank. (Has the same columns as output_table)
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(dense_rank<window_func>)
void combine_boundary_info(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<table_info>& recv_boundary_data,
    std::shared_ptr<table_info> output_table) {
    // offset to increment our local rank being sent by
    uint64_t rank_offset = getv<uint64_t, bodo_array_type::NUMPY>(
        recv_boundary_data->columns[recv_boundary_data->ncols() - 2], 0);

    // compare order by with neighbor, if they match we want to subtract
    // 1 from the offset
    std::vector<std::shared_ptr<array_info>> other_order_by_arrs =
        extract_columns(recv_boundary_data, partition_by_arrs.size(),
                        order_by_arrs.size());

    if (!distinct_from_other_row<OrderByArrType>(order_by_arrs, 0,
                                                 other_order_by_arrs, 0)) {
        rank_offset -= 1;
    }

    // update rank
    getv<uint64_t, bodo_array_type::NUMPY>(
        output_table->columns[output_table->ncols() - 2], 0) += rank_offset;
}

template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(row_number<window_func>)
void combine_boundary_info(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<table_info>& recv_boundary_data,
    std::shared_ptr<table_info> output_table) {
    // the max row number from our neighbor
    uint64_t row_offset = getv<uint64_t, bodo_array_type::NUMPY>(
        recv_boundary_data->columns[recv_boundary_data->ncols() - 2], 0);
    // update row number
    getv<uint64_t, bodo_array_type::NUMPY>(
        output_table->columns[output_table->ncols() - 2], 0) += row_offset;
}

template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(rank<window_func>)
void combine_boundary_info(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<table_info>& recv_boundary_data,
    std::shared_ptr<table_info> output_table) {
    // offset to increment our local rank being sent by
    uint64_t max_rank_neighbor = getv<uint64_t, bodo_array_type::NUMPY>(
        recv_boundary_data->columns[recv_boundary_data->ncols() - 2], 0);
    // group size from neighbor
    uint64_t row_offset = getv<uint64_t, bodo_array_type::NUMPY>(
        recv_boundary_data->columns[recv_boundary_data->ncols() - 1], 0);

    std::vector<std::shared_ptr<array_info>> other_order_by_arrs =
        extract_columns(recv_boundary_data, partition_by_arrs.size(),
                        order_by_arrs.size());

    bool dist_order_by_from_neighbor = distinct_from_other_row<OrderByArrType>(
        order_by_arrs, 0, other_order_by_arrs, 0);
    bool single_order = !distinct_from_other_row<OrderByArrType>(
        order_by_arrs, 0, order_by_arrs, order_by_arrs[0]->length - 1);

    // update rank, if we match on the orderby with our neighbor and
    // only contain a single orderby val, then we forward the neighbor's max
    // rank otherwise we update our own
    if (single_order && !dist_order_by_from_neighbor) {
        getv<uint64_t, bodo_array_type::NUMPY>(
            output_table->columns[output_table->ncols() - 2], 0) =
            max_rank_neighbor;
    } else {
        getv<uint64_t, bodo_array_type::NUMPY>(
            output_table->columns[output_table->ncols() - 2], 0) += row_offset;
    }

    // update group size
    getv<uint64_t, bodo_array_type::NUMPY>(
        output_table->columns[output_table->ncols() - 1], 0) += row_offset;
}

/**
 * @brief Combines local data with boundary data for percent rank computation in
 * the case where we have a single group that matches with it's neighbor(s).
 *
 * @tparam PartitionByArrType A single array type for the partition
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @param[in] partition_by_arrs The partition by columns.
 * @param[in] order_by_arrs The order by columns.
 * @param[in] recv_boundary_data The data we are receiving from our neighbor.
 * Contains partition by, order by, rank, and group size information as columns.
 * @param[in,out] output_table The combined result of recv_boundary_data and
 * local data has the form (partition_by_cols, orderby_cols, max_rank, num_rows)
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(percent_rank<window_func>)
void combine_boundary_info(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<table_info>& recv_boundary_data,
    std::shared_ptr<table_info> output_table) {
    std::vector<std::shared_ptr<array_info>> other_order_by_arrs =
        extract_columns(recv_boundary_data, partition_by_arrs.size(),
                        order_by_arrs.size());

    size_t ncols = recv_boundary_data->ncols();
    int64_t group_size_acc =
        getv<int64_t>(recv_boundary_data->columns[ncols - 1], 0);

    getv<int64_t>(output_table->columns[ncols - 1], 0) += group_size_acc;

    // this part is the same logic as rank.
    bool is_single_order = !distinct_from_other_row<OrderByArrType>(
        order_by_arrs, 0, order_by_arrs, order_by_arrs[0]->length - 1);
    bool is_dist_order_by = distinct_from_other_row<OrderByArrType>(
        other_order_by_arrs, 0, order_by_arrs, 0);
    if (is_single_order && !is_dist_order_by) {
        getv<double>(output_table->columns[ncols - 2], 0) =
            getv<double>(recv_boundary_data->columns[ncols - 2], 0);
    } else {
        getv<double>(output_table->columns[ncols - 2], 0) +=
            static_cast<double>(
                getv<int64_t>(recv_boundary_data->columns[ncols - 1], 0));
    }
}

template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, uint32_t window_func>
    requires(cume_dist<window_func>)
void combine_boundary_info(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<table_info>& recv_boundary_data,
    std::shared_ptr<table_info> output_table) {
    std::vector<std::shared_ptr<array_info>> other_order_by_arrs =
        extract_columns(recv_boundary_data, partition_by_arrs.size(),
                        order_by_arrs.size());

    size_t ncols = recv_boundary_data->ncols();

    // accumulate group size.
    int64_t group_size_acc =
        getv<int64_t>(recv_boundary_data->columns[ncols - 1], 0);

    getv<int64_t>(output_table->columns[ncols - 1], 0) += group_size_acc;

    // update outgoing num_ties
    bool is_single_order = !distinct_from_other_row<OrderByArrType>(
        order_by_arrs, 0, order_by_arrs, order_by_arrs[0]->length - 1);
    bool is_dist_order_by = distinct_from_other_row<OrderByArrType>(
        other_order_by_arrs, 0, order_by_arrs, 0);
    // we want to update the number of ties to send to the previous rank in the
    // case where our rank contains a single orderby value and it matches with
    // the first order by on the next rank.
    if (is_single_order && !is_dist_order_by) {
        getv<double>(output_table->columns[ncols - 2], 0) +=
            getv<double>(recv_boundary_data->columns[ncols - 2], 0);
    }
}

/**
 * @brief Compute the table to send to the nearest neighbor. If the data on the
 * rank is a single group then we may need to "update" the data we already
 * received from our neighbor to get an updated dense_rank value for the last
 * element.
 *
 * @param partition_by_arrs The partition by columns.
 * @param order_by_arrs The order by columns.
 * @param window_out_arr The output intermediate result for computing sorted
 * window fn.
 * @param last_group_size_arr The array containing the locally computed
 * size of the last group
 * @param recv_boundary_data A table of data we received from our neighboring
 * rank. This is only needed if is_single_group=True or we don't have any data.
 * If any other case exists it may just be an empty table.
 * @param is_single_group Is all the data on this rank a single group?
 * @return std::shared_ptr<table_info> A table with either 1 row or 0 rows to
 * send to our neighboring rank.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(dense_rank<window_func> || rank<window_func> ||
             row_number<window_func>)
std::shared_ptr<table_info> compute_sorted_window_send_data(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& window_out_arr,
    std::shared_ptr<array_info>& last_group_size_arr,
    const std::shared_ptr<table_info>& recv_boundary_data,
    bool is_single_group) {
    // TODO refactor to combine with other compute_sorted_window_send_data for
    // percent rank, this will probably involve send boundary computations to
    // take place outside of this function and/or be a templated helper
    if (window_out_arr->length == 0) {
        // If this rank is empty just forward data from our neighbor
        // in case there are holes.
        return recv_boundary_data;
    } else {
        // create output table from partition/order by columns, window data
        std::vector<std::shared_ptr<array_info>> data_arrs;
        for (auto& arr : partition_by_arrs) {
            data_arrs.push_back(arr);
        }
        for (auto& arr : order_by_arrs) {
            data_arrs.push_back(arr);
        }
        data_arrs.push_back(window_out_arr);
        std::unique_ptr<table_info> data_table =
            std::make_unique<table_info>(data_arrs);
        std::vector<int64_t> indices = {
            static_cast<int64_t>(window_out_arr->length - 1)};
        std::shared_ptr<table_info> output_table =
            RetrieveTable(std::move(data_table), indices);

        // append the group size info for rank()
        output_table->columns.push_back(last_group_size_arr);

        // Check if we need to combine with recv_boundary_data
        if (is_single_group && recv_boundary_data->nrows() > 0) {
            std::vector<std::shared_ptr<array_info>> other_partition_by_arrs =
                extract_columns(recv_boundary_data, 0,
                                partition_by_arrs.size());

            bool matching_group = !distinct_from_other_row<PartitionByArrType>(
                partition_by_arrs, window_out_arr->length - 1,
                other_partition_by_arrs, 0);
            if (matching_group) {
                assert(window_func == Bodo_FTypes::dense_rank ||
                       window_func == Bodo_FTypes::row_number ||
                       window_func == Bodo_FTypes::rank);
                combine_boundary_info<PartitionByArrType, OrderByArrType,
                                      window_func>(
                    partition_by_arrs, order_by_arrs, recv_boundary_data,
                    output_table);
            }
        }
        return output_table;
    }
}

/**
 * @brief Send data to the "next" rank for the last value in the current
 * data based on the send data. Since MPI won't send "empty" tables, we
 * also send the length to ensure the receiver knows if there is actual
 * data to accept.
 *
 * @param[in] send_data The table to send.
 * @param[out] send_states A vector to hold any issend MPI requests.
 * @param[in, out] length_ptr A buffer used to set and then send the length
 * of the table being transmitted. This buffer is allocated by a parent and is
 * required to be kept alive until the send is received.
 * @param[in] send_to The rank to send data to.
 * @return MPI_Request The MPI request used to communicate the length.
 */
MPI_Request send_sorted_window_data(
    const std::shared_ptr<table_info>& send_data,
    std::vector<AsyncShuffleSendState>& send_states,
    const std::unique_ptr<int>& length_ptr, int send_to) {
    int my_rank, n_pes;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    if (send_to < n_pes && send_to >= 0) {
        // We always send at most 1 row of data to our neighboring rank.
        uint32_t dest_hash = static_cast<uint32_t>(send_to);
        // Communicate how much data we will send first. This is necessary
        // because if we have 0 data to send we can't send a message.
        *length_ptr = static_cast<int>(send_data->nrows());
        MPI_Request final_send;
        MPI_Issend(length_ptr.get(), 1, MPI_INT, dest_hash, 0, MPI_COMM_WORLD,
                   &final_send);
        // This if statement is to make it obvious that no data is sent with an
        // empty table.
        if (send_data->nrows() != 0) {
            // change n=1 to n rows to send multiple rows
            std::shared_ptr<uint32_t[]> hashes =
                std::make_shared<uint32_t[]>(1);
            hashes[0] = dest_hash;
            send_states.push_back(
                shuffle_issend(send_data, hashes, nullptr, MPI_COMM_WORLD));
        }
        return final_send;
    } else {
        // Return a dummy value. We can always check we didn't send data.
        return MPI_REQUEST_NULL;
    }
}

/**
 * @brief Receive boundary information from rank `i - 1` for rank `i`
 * and output information to rank `i + 1`. This could should eventually
 * be generic enough to handle all functions, as all window functions may
 * need to transmit different information. In the future this could be
 * executed once for all functions.
 *
 * @tparam PartitionByArrType A single array type for the partition
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @param[in] partition_by_arrs The arrays used in partition by.
 * @param[in] order_by_arrs The arrays used in order by.
 * @param[out] window_out_arr The output array with the result of the local
 * computation. For window functions there will generally be a result that
 * can be converted to the final output and this array contains that result.
 * @param[in, out] send_states A vector to hold the MPI_Requests associated
 * with sending boundary information.
 * @param[in, out] length_ptr A buffer to use for sending the number of rows
 * that will be communicated to the neighbor. This is done to ensure the message
 * stays alive until the data is received.
 * @return std::tuple<std::shared_ptr<table_info>, MPI_Request> A pair of values
 * consisting of the boundary values received from the previous rank (or an
 * empty table if there is no data) and the MPI_Request send to communicate the
 * length.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(row_number<window_func> || rank<window_func> ||
             dense_rank<window_func>)
std::tuple<std::shared_ptr<table_info>, MPI_Request>
sorted_window_boundary_communication(
    std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& window_out_arr,
    std::vector<AsyncShuffleSendState>& send_states,
    const std::unique_ptr<int>& length_ptr, uint64_t last_group_size,
    std::vector<std::shared_ptr<DictionaryBuilder>> builders,
    std::shared_ptr<table_info> boundary_send_data) {
    // TODO refactor to make this interface more generic
    std::vector<std::shared_ptr<array_info>> arrs;

    for (auto& arr : partition_by_arrs) {
        arrs.push_back(arr);
    }
    for (auto& arr : order_by_arrs) {
        arrs.push_back(arr);
    }
    arrs.push_back(window_out_arr);

    // make column for extra row number data for computing rank
    std::shared_ptr<array_info> last_group_size_arr =
        alloc_numpy(1, Bodo_CTypes::UINT64);
    getv<uint64_t, bodo_array_type::NUMPY>(last_group_size_arr, 0) =
        last_group_size;
    // group size column for rank
    arrs.push_back(last_group_size_arr);

    std::unique_ptr<table_info> table = make_unique<table_info>(arrs);
    std::shared_ptr<table_info> empty_table =
        alloc_table_like(std::move(table));
    // Supports dense_rank, row_number and rank.
    assert(window_func == Bodo_FTypes::dense_rank ||
           window_func == Bodo_FTypes::row_number ||
           window_func == Bodo_FTypes::rank);
    bool empty_group = window_out_arr->length == 0;
    bool single_group = !distinct_from_other_row<PartitionByArrType>(
        partition_by_arrs, 0, partition_by_arrs, window_out_arr->length - 1);
    std::shared_ptr<table_info> recv_boundary;
    MPI_Request send_request;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // NOTE: Rank 0 will always send first since the recv in the empty_group ||
    // single_group case will not have any MPI calls, meaning in the worst case
    // we can always have the ordering send 0 -> recv 1 -> send 1 -> recv 2 ...
    if (empty_group || single_group) {
        // If the current rank doesn't have any data or is a single group we
        // need to receive data from our neighbor first to send to the next
        // neighbor.
        // recv from previous neighbor myrank - 1
        recv_boundary =
            recv_sorted_window_data(empty_table, partition_by_arrs,
                                    order_by_arrs, builders, my_rank - 1);

        // send data looks slightly different in this case (do not need to check
        // if rows the same, just the groups)
        std::shared_ptr<table_info> send_boundary =
            compute_sorted_window_send_data<PartitionByArrType, OrderByArrType,
                                            window_func>(
                partition_by_arrs, order_by_arrs, window_out_arr,
                last_group_size_arr, recv_boundary, single_group);
        // send to next neighbor (myrank + 1)
        send_request = send_sorted_window_data(send_boundary, send_states,
                                               length_ptr, my_rank + 1);
    } else {
        std::shared_ptr<table_info> send_boundary =
            compute_sorted_window_send_data<PartitionByArrType, OrderByArrType,
                                            window_func>(
                partition_by_arrs, order_by_arrs, window_out_arr,
                last_group_size_arr, empty_table, false);
        // send to next neighbor (myrank + 1)
        send_request = send_sorted_window_data(send_boundary, send_states,
                                               length_ptr, my_rank + 1);
        // recv from previous neighbor (myrank -1)
        recv_boundary =
            recv_sorted_window_data(empty_table, partition_by_arrs,
                                    order_by_arrs, builders, my_rank - 1);
    }
    return std::tuple(recv_boundary, send_request);
}

/**
 * @brief Computes the data to send to the "next" neighbor which will either be
 * rank `i - 1` or rank `i + 1` for rank `i`.
 *
 * @tparam PartitionByArrType A single array type for the partition
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @param[in] partition_by_arrs The arrays used in partition by.
 * @param[in] order_by_arrs The arrays used in order by.
 * @param[out] window_out_arr The output array with the result of the local
 * @param[in,out] send_boundary_data The local data. We will also use this table
 * to accumulate data from neighboring ranks in the case where the partition by
 * columns match and we have a single group.
 * @param[in,out] recv_boundary_data The data we are receiving from our
 * neighbor, will also be the return value in the case our rank has empty data.
 * @param is_single_group Whether the current rank contains
 * @return std::shared_ptr<table_info> The data to send to out to our (other)
 * neighbor.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(percent_rank<window_func> || cume_dist<window_func>)
std::shared_ptr<table_info> compute_sorted_window_send_data(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& window_out_arr,
    std::shared_ptr<table_info>& send_boundary_data,
    const std::shared_ptr<table_info>& recv_boundary_data,
    bool is_single_group) {
    if (window_out_arr->length == 0) {
        // If this rank is empty just forward data from our neighbor
        // in case there are holes.
        return recv_boundary_data;
    }
    if (is_single_group && recv_boundary_data->nrows() > 0) {
        // check to see if the neighboring rank contains the same partition. We
        // will need to accumulate the data to send in this case
        std::vector<std::shared_ptr<array_info>> other_partition_by_arrs =
            extract_columns(recv_boundary_data, 0, partition_by_arrs.size());

        bool matching_group = !distinct_from_other_row<PartitionByArrType>(
            partition_by_arrs, window_out_arr->length - 1,
            other_partition_by_arrs, 0);
        if (matching_group) {
            // same implementation as rank for now, we technically are wasting a
            // bit because some values we are checking/updating are not used in
            // one of the communication directions (i.e. sending to the previous
            // neighbor we only need to send group size and not rank)
            combine_boundary_info<PartitionByArrType, OrderByArrType,
                                  window_func>(partition_by_arrs, order_by_arrs,
                                               recv_boundary_data,
                                               send_boundary_data);
        }
    }
    return send_boundary_data;
}

/**
 * @brief Receive boundary information from rank `i - 1` and `i + 1` for rank
 * `i` and output information to rank `i + 1` and `i - 1`.
 *
 * @tparam PartitionByArrType A single array type for the partition
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @param[in] partition_by_arrs The arrays used in partition by.
 * @param[in] order_by_arrs The arrays used in order by.
 * @param[out] window_out_arr The output array with the result of the local
 * computation. For window functions there will generally be a result that
 * can be converted to the final output and this array contains that result.
 * @param[in, out] send_states A vector to hold the MPI_Requests associated
 * with sending boundary information.
 * @param[in, out] length_ptr A buffer to use for sending the number of rows
 * that will be communicated to the neighbor. This is done to ensure the message
 * stays alive until the data is received.
 * @param[in] boundary_send_data Table consisting of two rows the first row is
 * the data we would like to send to rank `i - 1` and the second row is the
 * data we would like to send to rank `i + 1`
 * @return std::tuple<std::shared_ptr<table_info>, MPI_Request> A pair of values
 * consisting of the boundary values received from the previous rank (if no data
 * is sent then the table will have zero values which will allow for correctness
 * in the final result
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(percent_rank<window_func> || cume_dist<window_func>)
std::tuple<std::shared_ptr<table_info>, MPI_Request>
sorted_window_boundary_communication(
    std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& window_out_arr,
    std::vector<AsyncShuffleSendState>& send_states,
    const std::unique_ptr<int>& length_ptr, uint64_t last_group_size,
    std::vector<std::shared_ptr<DictionaryBuilder>> builders,
    std::shared_ptr<table_info> boundary_send_data) {
    // TODO refactor to make this interface more generic
    std::shared_ptr<table_info> empty_table =
        alloc_table_like(boundary_send_data);

    bool empty_group = window_out_arr->length == 0;
    bool single_group = !distinct_from_other_row<PartitionByArrType>(
        partition_by_arrs, 0, partition_by_arrs, window_out_arr->length - 1);
    MPI_Request send_request;
    // combined info from both neighbors
    std::shared_ptr<table_info> recv_boundary;
    // information we are recv'ing from our next neighbor
    std::shared_ptr<table_info> recv_boundary_upper;
    // info we are recv'ing from our previous neighbor
    std::shared_ptr<table_info> recv_boundary_lower;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // NOTE: Rank 0 will always send first since the recv in the empty_group ||
    // single_group case will not have any MPI calls, meaning in the worst case
    // we can always have the ordering send 0 -> recv 1 -> send 1 -> recv 2 ...
    if (empty_group || single_group) {
        // If the current rank doesn't have any data or is a single group we
        // need to receive data from our neighbor first to send to the next
        // neighbor.
        // recv from previous neighbor myrank - 1
        recv_boundary_lower =
            recv_sorted_window_data(empty_table, partition_by_arrs,
                                    order_by_arrs, builders, my_rank - 1);

        // first we want to propagate information about the last groupby/orderby
        // vals to our next neighbor
        std::shared_ptr<table_info> send_boundary_upper =
            RetrieveTable(boundary_send_data, std::vector<int64_t>({1}));

        send_boundary_upper =
            compute_sorted_window_send_data<PartitionByArrType, OrderByArrType,
                                            window_func>(
                partition_by_arrs, order_by_arrs, window_out_arr,
                send_boundary_upper, recv_boundary_lower, single_group);
        // send to next neighbor (myrank+1)
        send_request = send_sorted_window_data(send_boundary_upper, send_states,
                                               length_ptr, my_rank + 1);
        // ensure all send/recvs are done before doing the next round of sends
        MPI_Barrier(MPI_COMM_WORLD);

        // NOTE: here we rely on the fact that the last rank will return without
        // any MPI calls
        recv_boundary_upper =
            recv_sorted_window_data(empty_table, partition_by_arrs,
                                    order_by_arrs, builders, my_rank + 1);

        // then we propagate information about the first groupby/orderby vals to
        // our previous neighbor
        std::shared_ptr<table_info> send_boundary_lower =
            RetrieveTable(boundary_send_data, std::vector<int64_t>({0}));

        send_boundary_lower =
            compute_sorted_window_send_data<PartitionByArrType, OrderByArrType,
                                            window_func>(
                partition_by_arrs, order_by_arrs, window_out_arr,
                send_boundary_lower, recv_boundary_upper, single_group);

        send_request = send_sorted_window_data(send_boundary_lower, send_states,
                                               length_ptr, my_rank - 1);
    } else {
        // send to my_rank + 1
        std::shared_ptr<table_info> send_boundary_upper =
            RetrieveTable(boundary_send_data, std::vector<int64_t>({1}));

        send_request = send_sorted_window_data(send_boundary_upper, send_states,
                                               length_ptr, my_rank + 1);
        // recv from my_rank - 1
        recv_boundary_lower =
            recv_sorted_window_data(empty_table, partition_by_arrs,
                                    order_by_arrs, builders, my_rank - 1);

        // ensure all send/recvs are done before doing the next round of sends
        MPI_Barrier(MPI_COMM_WORLD);

        // send to my_rank - 1
        std::shared_ptr<table_info> send_boundary_lower =
            RetrieveTable(boundary_send_data, std::vector<int64_t>({0}));

        send_request = send_sorted_window_data(send_boundary_lower, send_states,
                                               length_ptr, my_rank - 1);

        // recv from my_rank + 1
        recv_boundary_upper =
            recv_sorted_window_data(empty_table, partition_by_arrs,
                                    order_by_arrs, builders, my_rank + 1);
    }
    // if we recv'd empty data from either of our neighbors, set data values to
    // zero to ensure the final update step can be performed correctly
    if (recv_boundary_lower->nrows() == 0 && window_out_arr->length > 0) {
        recv_boundary_lower =
            RetrieveTable(boundary_send_data, std::vector<int64_t>({0}));
        getv<double>(
            recv_boundary_lower->columns[recv_boundary_lower->ncols() - 2], 0) =
            0.0;
        getv<int64_t>(
            recv_boundary_lower->columns[recv_boundary_lower->ncols() - 1], 0) =
            0;
    }
    if (recv_boundary_upper->nrows() == 0 && window_out_arr->length > 0) {
        recv_boundary_upper =
            RetrieveTable(boundary_send_data, std::vector<int64_t>({1}));
        getv<double>(
            recv_boundary_upper->columns[recv_boundary_upper->ncols() - 2], 0) =
            0.0;
        getv<int64_t>(
            recv_boundary_upper->columns[recv_boundary_upper->ncols() - 1], 0) =
            0;
    }

    // combine info we've seen for the boundaries
    recv_boundary = concat_tables({recv_boundary_lower, recv_boundary_upper});

    return std::tuple(recv_boundary, send_request);
}

/**
 * @brief Update all window output values in the first partition group with the
 * provided offset. The caller should have already checked that
 * boundary_partition_by_arrs has a single row and that the offset is non-zero.
 *
 * @tparam PartitionByArrType The shared type of all partition by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @param[in] partition_by_arrs The partition by columns to check for the first
 * group.
 * @param[in] boundary_partition_by_arrs The partition by columns of the
 * boundary data.
 * @param[out] window_out_arr The output array that needs to be updated.
 * @param offset The offset for how much to increment each value in the first
 * group.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          int32_t window_func>
    requires(row_number<window_func> || dense_rank<window_func>)
void sorted_window_update_offset(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& boundary_partition_by_arrs,
    const std::shared_ptr<array_info>& window_out_arr, uint64_t offset) {
    for (size_t i = 0; i < window_out_arr->length &&
                       !distinct_from_other_row<PartitionByArrType>(
                           partition_by_arrs, i, boundary_partition_by_arrs, 0);
         i++) {
        getv<uint64_t, bodo_array_type::NUMPY>(window_out_arr, i) += offset;
    }
}

/**
 * @brief Update all window output values that match with the boundary
 * order by data by the max rank_offset and update values in the first partition
 * with the row_offset.
 *
 * @tparam PartitionByArrType The shared type of all partition by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @param[in] partition_by_arrs The partition by columns to check for the first
 * group.
 * @param[in] boundary_partition_by_arrs The partition by columns of the
 * boundary data.
 * @param[out] window_out_arr The output array that needs to be updated.
 * @param rank_offset The offset for how much to increment values that match
 * the recv'd boundary data order by columns
 * @param row_offset The offset for how much to increment each value in the
 * first group.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(rank<window_func>)
void sorted_window_update_offset(
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& boundary_order_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& boundary_partition_by_arrs,
    const std::shared_ptr<array_info>& window_out_arr, uint64_t rank_offset,
    uint64_t row_offset) {
    size_t same_order_offset = -1;
    // update all elements with the same rank
    for (size_t i = 0; i < window_out_arr->length &&
                       !distinct_from_other_row<OrderByArrType>(
                           order_by_arrs, i, boundary_order_by_arrs, 0);
         i++) {
        getv<uint64_t, bodo_array_type::NUMPY>(window_out_arr, i) +=
            rank_offset;
        same_order_offset = i;
    }
    // rest of the elements in the partition update with row_offset i.e.
    // max_rank + num_ties - 1
    for (size_t i = same_order_offset + 1;
         i < window_out_arr->length &&
         !distinct_from_other_row<PartitionByArrType>(
             partition_by_arrs, i, boundary_partition_by_arrs, 0);
         i++) {
        getv<uint64_t, bodo_array_type::NUMPY>(window_out_arr, i) += row_offset;
    }
}

/**
 * @brief For the first group on our rank, we want to combine information
 * about group size, max rank from our previous neighbor and do the rank
 * update and batch division. Similarly for the last group, we accumulate
 * the group size and do a batch division update (we do not need to update)
 * the ranks. At the end of this step all values in the output will be
 * (r-1) / (group_size -1)
 *
 * @tparam PartitionByArrType The shared type of all partition by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @tparam OrderByArrType The shared type of all order by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @tparam window_func The window function we are currently computing
 * @param[in] partition_by_arrs The partition by columns to check for the first
 * group.
 * @param[out] window_out_arr The output array that needs to be updated.
 * @param[in] recv_boundary_data The data that we have recv'd from neighboring
 * ranks, the first row contains data from `i - 1` and the second row contains
 * data from `i + 1`. Note that if they are both empty then the data is
 * set to zero to ensure the batch division updates still happen
 * @param[in] send_boundary_data The data we computed in the local step that
 * contains info about our group size which will be used in the final rank
 * division batch update step.
 * @param row_offset The offset for how much to increment each value in the
 * first group.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(percent_rank<window_func>)
void update_sorted_window_output(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& window_out_arr,
    const std::shared_ptr<table_info>& recv_boundary_data,
    const std::shared_ptr<table_info>& send_boundary_data) {
    int64_t ncols = recv_boundary_data->ncols();
    int64_t nrows = window_out_arr->length;

    // information about the group size accumulated from our neighbors
    int64_t recv_first_group_size =
        getv<int64_t>(recv_boundary_data->columns[ncols - 1], 0);
    int64_t recv_last_group_size =
        getv<int64_t>(recv_boundary_data->columns[ncols - 1], 1);

    // our locally computed group sizes
    int64_t local_first_group_size =
        getv<int64_t>(send_boundary_data->columns[ncols - 1], 0);
    int64_t local_last_group_size =
        getv<int64_t>(send_boundary_data->columns[ncols - 1], 1);

    // rank offset from our neighbor
    double first_rank_offset =
        getv<double>(recv_boundary_data->columns[ncols - 2], 0);

    // partition by, order by arrs we recv'd from our neighbors
    std::vector<std::shared_ptr<array_info>> other_partition_by_arrs =
        extract_columns(recv_boundary_data, 0, partition_by_arrs.size());

    std::vector<std::shared_ptr<array_info>> other_order_by_arrs =
        extract_columns(recv_boundary_data, partition_by_arrs.size(),
                        order_by_arrs.size());

    // examine our received data to see which values we need to update
    bool is_single_group = !distinct_from_other_row<PartitionByArrType>(
        partition_by_arrs, 0, partition_by_arrs, nrows - 1);

    // data coming from other neighbors could be empty but we insert dummy rows
    // into recv_boundary_data for consistency.
    bool dist_partition_by_lower =
        recv_first_group_size == 0 ||
        distinct_from_other_row<PartitionByArrType>(other_partition_by_arrs, 0,
                                                    partition_by_arrs, 0);

    bool dist_partition_by_upper =
        recv_last_group_size == 0 ||
        distinct_from_other_row<PartitionByArrType>(
            other_partition_by_arrs, 1, partition_by_arrs, nrows - 1);

    // update the first group when the neighbor before matches with our first
    // group
    bool do_update_first_group = !dist_partition_by_lower;
    // only update last last group in the case where the partitions match AND we
    // have not already updated it in the first loop.
    bool do_update_last_group =
        !dist_partition_by_upper && !(is_single_group && do_update_first_group);

    // calculate sizes of the first and last group using received data
    int64_t total_first_group_size =
        local_first_group_size + recv_first_group_size;
    // case where there is only one group and it matches both neighbors
    if (is_single_group && !dist_partition_by_upper) {
        total_first_group_size += recv_last_group_size;
    }
    int64_t total_last_group_size =
        local_last_group_size + recv_last_group_size;

    // update the first group
    if (do_update_first_group) {
        for (int64_t i = 0; i < local_first_group_size; i++) {
            // Undo batch division from local step
            getv<double>(window_out_arr, i) *= (local_first_group_size - 1.0);
            if (!distinct_from_other_row<OrderByArrType>(other_order_by_arrs, 0,
                                                         order_by_arrs, i)) {
                // all rows that match with the orderby value of the previous
                // neighbor's last row will get incremented by the max rank on
                // our previous neighbor
                getv<double>(window_out_arr, i) += first_rank_offset - 1.0;
            } else {
                // otherwise, increment the rank by the number of rows in that
                // group on on the previous neighbor (same logic as rank)
                getv<double>(window_out_arr, i) +=
                    static_cast<double>(recv_first_group_size);
            }
            // lastly, divide by group size to get the percent rank
            getv<double>(window_out_arr, i) /=
                static_cast<double>(total_first_group_size) - 1.0;
        }
    }

    // do update for the last group, we do not need to update the rank in the
    // case of pct_rank
    if (do_update_last_group) {
        for (int64_t i = nrows - local_last_group_size; i < nrows; i++) {
            // update the denominator for the last group
            getv<double>(window_out_arr, i) = getv<double>(window_out_arr, i) *
                                              (local_last_group_size - 1.0) /
                                              (total_last_group_size - 1.0);
        }
    }
}

/**
 * @brief Updates window output for cume dist. Same as percent rank except we
 * want to do "tie upward" on the last group (which could also just be a single
 * group)
 *
 * @tparam PartitionByArrType The shared type of all partition by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @tparam OrderByArrType The shared type of all order by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @param[in] partition_by_arrs The partition by columns.
 * @param[in] order_by_arrs The order by columns.
 * @param[in,out] window_out_arr The output of the window function
 * @param[in] recv_boundary_data The data we are receiving from our neighbors.
 * @param[in] send_boundary_data Contains the locally computed data of group
 * sizes for first/last group.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(cume_dist<window_func>)
void update_sorted_window_output(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& window_out_arr,
    const std::shared_ptr<table_info>& recv_boundary_data,
    const std::shared_ptr<table_info>& send_boundary_data) {
    int64_t nrows = window_out_arr->length;
    int64_t ncols = recv_boundary_data->ncols();

    // information about the group size accumulated from our neighbors
    // maybe move this into a helper that extracts this information from the
    // table and returns it as a tuple
    // information about the group size accumulated from our neighbors
    int64_t recv_first_group_size =
        getv<int64_t>(recv_boundary_data->columns[ncols - 1], 0);
    int64_t recv_last_group_size =
        getv<int64_t>(recv_boundary_data->columns[ncols - 1], 1);

    // our locally computed group sizes
    int64_t local_first_group_size =
        getv<int64_t>(send_boundary_data->columns[ncols - 1], 0);
    int64_t local_last_group_size =
        getv<int64_t>(send_boundary_data->columns[ncols - 1], 1);

    // partition by, order by arrs we recv'd from our neighbors
    std::vector<std::shared_ptr<array_info>> other_partition_by_arrs =
        extract_columns(recv_boundary_data, 0, partition_by_arrs.size());
    std::vector<std::shared_ptr<array_info>> other_order_by_arrs =
        extract_columns(recv_boundary_data, partition_by_arrs.size(),
                        order_by_arrs.size());

    // tie value comes from neighbor to the right / myrank + 1
    double tie_upward_offset = getv<double>(
        recv_boundary_data->columns[recv_boundary_data->ncols() - 2], 1);

    // check neighboring data to determine which groups we will need to update
    bool is_single_group = !distinct_from_other_row<PartitionByArrType>(
        partition_by_arrs, 0, partition_by_arrs, nrows - 1);

    bool dist_partition_by_lower =
        recv_first_group_size == 0 ||
        distinct_from_other_row<PartitionByArrType>(other_partition_by_arrs, 0,
                                                    partition_by_arrs, 0);

    bool dist_partition_by_upper =
        recv_last_group_size == 0 ||
        distinct_from_other_row<PartitionByArrType>(
            other_partition_by_arrs, 1, partition_by_arrs, nrows - 1);

    bool do_update_first_group = !dist_partition_by_lower;
    // we want to update the last group if we match with our neighbor on rank `i
    // + 1` and we have not already updated the values in the first loop.
    bool do_update_last_group =
        !dist_partition_by_upper && !(is_single_group && do_update_first_group);
    // check whether we need to do the tie upwards step as part of the first
    // loop to ensure each value that needs to gets updates exactly once
    bool do_tie_upwards_first =
        do_update_first_group && (is_single_group && !dist_partition_by_upper);

    // calculate the size of the first group
    int64_t total_first_group_size =
        local_first_group_size + recv_first_group_size;
    // case where we match with neighbors on either side
    if (do_tie_upwards_first)
        total_first_group_size += recv_last_group_size;

    int64_t total_last_group_size =
        local_last_group_size + recv_last_group_size;

    if (do_update_first_group) {
        for (int64_t i = 0; i < local_first_group_size; i++) {
            // if we are in the single group case and our partition matches with
            // the last neighbor, we want to make sure that we tie upwards for
            // all orderby values that match our previous neighbor
            getv<double>(window_out_arr, i) *= local_first_group_size;
            if (do_tie_upwards_first &&
                !distinct_from_other_row<OrderByArrType>(other_order_by_arrs, 1,
                                                         order_by_arrs, i)) {
                getv<double>(window_out_arr, i) +=
                    static_cast<double>(tie_upward_offset);
            }
            getv<double>(window_out_arr, i) =
                (getv<double>(window_out_arr, i) + recv_first_group_size) /
                static_cast<double>(total_first_group_size);
        }
    }

    if (do_update_last_group) {
        for (int64_t i = nrows - local_last_group_size; i < nrows; i++) {
            getv<double>(window_out_arr, i) *= local_last_group_size;
            if (!distinct_from_other_row<OrderByArrType>(other_order_by_arrs, 1,
                                                         order_by_arrs, i)) {
                getv<double>(window_out_arr, i) += tie_upward_offset;
            }
            getv<double>(window_out_arr, i) =
                getv<double>(window_out_arr, i) /
                static_cast<double>(total_last_group_size);
        }
    }
}

/**
 * @brief Updates the window function output based on the value of
 * the output column(s) in the recv_boundary_data. If or how
 * much we update the window_out_arr by depends on where the
 * partition_by arrays match and if the order_by arrays match
 * on the boundary.
 *
 * @tparam PartitionByArrType A single array type for the partition
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @tparam OrderByArrType A single array type for the order
 * by columns or bodo_array_type::UNKNOWN if mixed types.
 * @param[in] partition_by_arrs The partition by arrays on the local
 * output data.
 * @param[in] order_by_arrs The order by arrays on the local output
 * data.
 * @param[in, out] window_out_arr The output array for computing the
 * rank result.
 * @param[in] recv_boundary_data The table received from the neighbor
 * array)
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
    requires(rank<window_func> || dense_rank<window_func> ||
             row_number<window_func>)
void update_sorted_window_output(
    const std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::shared_ptr<array_info>& window_out_arr,
    const std::shared_ptr<table_info>& recv_boundary_data,
    const std::shared_ptr<table_info>& send_boundary_data) {
    if (recv_boundary_data->nrows() == 0) {
        return;
    }
    int64_t rank_offset = getv<uint64_t, bodo_array_type::NUMPY>(
        recv_boundary_data->columns[recv_boundary_data->ncols() - 2], 0);
    int64_t last_group_size = getv<uint64_t, bodo_array_type::NUMPY>(
        recv_boundary_data->columns[recv_boundary_data->ncols() - 1], 0);

    // Determine if the order by values might be tied. We don't need to check
    // partition by because this is checked in the loop.
    std::vector<std::shared_ptr<array_info>> other_order_by_arrs =
        extract_columns(recv_boundary_data, partition_by_arrs.size(),
                        order_by_arrs.size());

    std::vector<std::shared_ptr<array_info>> other_partition_by_arrs =
        extract_columns(recv_boundary_data, 0, partition_by_arrs.size());

    assert(window_func == Bodo_FTypes::dense_rank ||
           window_func == Bodo_FTypes::row_number ||
           window_func == Bodo_FTypes::rank);
    bool dist_order_by = distinct_from_other_row<OrderByArrType>(
        order_by_arrs, 0, other_order_by_arrs, 0);
    switch (window_func) {
        case Bodo_FTypes::dense_rank: {
            // If the order by values are the same then we
            // subtract 1 from the increment value because all ranks have
            // the same value in case of ties for dense rank.
            rank_offset -= (dist_order_by ? 0 : 1);
            if (rank_offset <= 0) {
                return;
            }
            sorted_window_update_offset<PartitionByArrType,
                                        Bodo_FTypes::dense_rank>(
                partition_by_arrs, other_partition_by_arrs, window_out_arr,
                rank_offset);
            break;
        }
        case Bodo_FTypes::rank: {
            // If the order by values are the same then we
            // subtract 1 from the increment value because all ranks
            // have the same value in case of ties for rank.
            rank_offset -= (dist_order_by ? 0 : 1);
            sorted_window_update_offset<PartitionByArrType, OrderByArrType,
                                        Bodo_FTypes::rank>(
                order_by_arrs, other_order_by_arrs, partition_by_arrs,
                other_partition_by_arrs, window_out_arr, rank_offset,
                last_group_size);
            break;
        }
        case Bodo_FTypes::row_number: {
            sorted_window_update_offset<PartitionByArrType,
                                        Bodo_FTypes::row_number>(
                partition_by_arrs, other_partition_by_arrs, window_out_arr,
                rank_offset);
            break;
        }
        default:
            break;
    }
}

/**
 * @brief Performs the parallel step of sorted window computation
 * which involves communication with neighboring ranks to propagate
 * information that might be on the boundary and updating the output array.
 *
 * @tparam PartitionByArrType The shared type of all partition by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @tparam OrderByArrType The shared type of all order by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @tparam window_func The window function we are currently computing
 * @param[in] partition_by_arrs The partition by arrays on the local
 * output data.
 * @param[in] order_by_arrs The order by arrays on the local output
 * data.
 * @param[in, out] out_arr The output array for computing the
 * @param last_group_size the locally computed size of the last group
 * rank result.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType, int32_t window_func>
void sorted_window_parallel_step(
    std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    std::shared_ptr<array_info> out_arr,
    std::vector<std::shared_ptr<DictionaryBuilder>> builders,
    int64_t last_group_size, std::shared_ptr<table_info> boundary_send_data) {
    std::vector<AsyncShuffleSendState> send_states;
    std::unique_ptr<int> length_ptr = std::make_unique<int>(0);
    auto [boundary_info, send_request] =
        sorted_window_boundary_communication<PartitionByArrType, OrderByArrType,
                                             window_func>(
            partition_by_arrs, order_by_arrs, out_arr, send_states, length_ptr,
            last_group_size, builders, boundary_send_data);
    update_sorted_window_output<PartitionByArrType, OrderByArrType,
                                window_func>(partition_by_arrs, order_by_arrs,
                                             out_arr, boundary_info,
                                             boundary_send_data);
    bool sent_data = send_request != MPI_REQUEST_NULL;
    if (sent_data) {
        MPI_Wait(&send_request, MPI_STATUSES_IGNORE);
        while (send_states.size() > 0) {
            std::erase_if(send_states, [&](AsyncShuffleSendState& s) {
                return s.sendDone();
            });
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // resets added here so they do not get garage collected before loop
    // finishes
    boundary_info.reset();
    length_ptr.reset();
}

/**
 * @brief Templated function for doing computation on window function(s)
 *
 * @tparam PartitionByArrType The shared type of all partition by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @tparam OrderByArrType The shared type of all order by columns for
 * faster comparisons or bodo_array_type::UNKNOWN if the types differ.
 * @param[in] partition_by_arrs The arrays that hold the partition by values.
 * @param[in] window_funcs The name(s) of the window function(s) being computed.
 * @param[in] order_by_arrs The arrays that hold the order by values.
 * @param[out] out_arrs The output array(s) being populated.
 * @param is_parallel Is the data distributed? This is used for communicating
 * with a neighboring rank for boundary groups.
 */
template <bodo_array_type::arr_type_enum PartitionByArrType,
          bodo_array_type::arr_type_enum OrderByArrType>
void _sorted_window_computation(
    std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::vector<int32_t>& window_funcs,
    std::vector<std::shared_ptr<array_info>> out_arrs,
    std::vector<std::shared_ptr<DictionaryBuilder>> builders,
    bool is_parallel) {
    // rank
    uint64_t last_group_size = 0;
    std::shared_ptr<table_info> boundary_send_data;
    for (size_t i = 0; i < window_funcs.size(); i++) {
        switch (window_funcs[i]) {
            // TODO all these cases are basically the same... add a macro
            case Bodo_FTypes::dense_rank: {
                local_sorted_window_fn<PartitionByArrType, OrderByArrType,
                                       Bodo_FTypes::dense_rank>(
                    partition_by_arrs, order_by_arrs, out_arrs[i]);
                break;
            }
            case Bodo_FTypes::row_number: {
                local_sorted_window_fn<PartitionByArrType, OrderByArrType,
                                       Bodo_FTypes::row_number>(
                    partition_by_arrs, order_by_arrs, out_arrs[i]);
                break;
            }
            case Bodo_FTypes::rank: {
                last_group_size =
                    local_sorted_window_fn<PartitionByArrType, OrderByArrType,
                                           Bodo_FTypes::rank>(
                        partition_by_arrs, order_by_arrs, out_arrs[i]);
                break;
            }
            case Bodo_FTypes::percent_rank: {
                boundary_send_data =
                    local_sorted_window_fn<PartitionByArrType, OrderByArrType,
                                           Bodo_FTypes::percent_rank>(
                        partition_by_arrs, order_by_arrs, out_arrs[i],
                        is_parallel);
                break;
            }
            case Bodo_FTypes::cume_dist: {
                boundary_send_data =
                    local_sorted_window_fn<PartitionByArrType, OrderByArrType,
                                           Bodo_FTypes::cume_dist>(
                        partition_by_arrs, order_by_arrs, out_arrs[i], true);
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported window function for the sort implementation");
        }
    }
    // Handle any neighbor communication for window functions.
    if (is_parallel) {
        for (size_t i = 0; i < window_funcs.size(); i++) {
            switch (window_funcs[i]) {
                // TODO add macro for cases
                case Bodo_FTypes::dense_rank: {
                    sorted_window_parallel_step<PartitionByArrType,
                                                OrderByArrType,
                                                Bodo_FTypes::dense_rank>(
                        partition_by_arrs, order_by_arrs, out_arrs[i], builders,
                        last_group_size, boundary_send_data);
                    break;
                }
                case Bodo_FTypes::row_number: {
                    sorted_window_parallel_step<PartitionByArrType,
                                                OrderByArrType,
                                                Bodo_FTypes::row_number>(
                        partition_by_arrs, order_by_arrs, out_arrs[i], builders,
                        last_group_size, boundary_send_data);
                    break;
                }
                case Bodo_FTypes::rank: {
                    sorted_window_parallel_step<
                        PartitionByArrType, OrderByArrType, Bodo_FTypes::rank>(
                        partition_by_arrs, order_by_arrs, out_arrs[i], builders,
                        last_group_size, boundary_send_data);
                    break;
                }
                case Bodo_FTypes::percent_rank: {
                    sorted_window_parallel_step<PartitionByArrType,
                                                OrderByArrType,
                                                Bodo_FTypes::percent_rank>(
                        partition_by_arrs, order_by_arrs, out_arrs[i], builders,
                        last_group_size, boundary_send_data);
                    break;
                }
                case Bodo_FTypes::cume_dist: {
                    sorted_window_parallel_step<PartitionByArrType,
                                                OrderByArrType,
                                                Bodo_FTypes::cume_dist>(
                        partition_by_arrs, order_by_arrs, out_arrs[i], builders,
                        last_group_size, boundary_send_data);
                    break;
                }
                default:
                    throw std::runtime_error(
                        "Unsupported window function for the sort "
                        "implementation parallel path");
            }
        }
    }
}

/**
 * @brief Converts a sum to double so it can be used to compute an average
 * Assumes the ::sum operation upcasts ints to int64/uint64
 *
 * @param sum_arr The singleton array containing the sum in it's first element
 * @returns double the sum casted to double
 */
double avg_sum_to_double(std::shared_ptr<array_info> sum_arr) {
    double sum_total;

#define SUM_TYPE_CASE(dtype)                                                   \
    case dtype: {                                                              \
        sum_total =                                                            \
            static_cast<double>(getv<dtype_to_type<dtype>::type>(sum_arr, 0)); \
        break;                                                                 \
    }

    // TODO check if list of sum output types covers all relevant cases
    switch (sum_arr->dtype) {
        SUM_TYPE_CASE(Bodo_CTypes::FLOAT64)
        SUM_TYPE_CASE(Bodo_CTypes::FLOAT32)
        SUM_TYPE_CASE(Bodo_CTypes::INT64)
        SUM_TYPE_CASE(Bodo_CTypes::UINT64)
        default: {
            throw std::runtime_error("Unsupported dtype for mean");
        }
    }
#undef SUM_TYPE_CASE

    return sum_total;
}

/**
 * @brief Computes a single, global result after applying window_func to a
 * column.
 *
 * @tparam window_func The window func to compute. Assumes the window function
 * does not require any auxillary columns.
 * @param in_col The input column
 * @param out_arr The array to store
 * @param is_parallel Whether to do parallel communication step.
 */
template <Bodo_FTypes::FTypeEnum window_func>
    requires(no_aux_col<window_func>)
void single_global_window_computation(const std::shared_ptr<array_info>& in_col,
                                      std::shared_ptr<array_info> out_arr,
                                      bool is_parallel) {
    grouping_info dummy_local_grp_info;
    dummy_local_grp_info.num_groups = 1;
    dummy_local_grp_info.row_to_group.resize(in_col->length, 0);

    std::vector<std::shared_ptr<array_info>> dummy_aux_cols;

    do_apply_to_column(in_col, out_arr, dummy_aux_cols, dummy_local_grp_info,
                       window_func);

    if (is_parallel) {
        int myrank, num_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        grouping_info dummy_combine_grp_info;
        dummy_combine_grp_info.num_groups = 1;
        dummy_combine_grp_info.row_to_group.resize(num_ranks, 0);
        std::shared_ptr<array_info> combined_arr =
            gather_array(out_arr, true, true, 0, num_ranks, myrank);
        switch (window_func) {
            case Bodo_FTypes::sum:
            case Bodo_FTypes::min:
            case Bodo_FTypes::max: {
                // Repeat the pre-parallel procedure on the combined
                // results.
                std::vector<std::shared_ptr<array_info>> dummy_aux_cols;
                aggfunc_output_initialize(out_arr, window_func, true);
                do_apply_to_column(combined_arr, out_arr, dummy_aux_cols,
                                   dummy_combine_grp_info, window_func);
                break;
            }
            case Bodo_FTypes::count: {
                // For count, combine the results by adding them.
                std::vector<std::shared_ptr<array_info>> dummy_aux_cols;
                aggfunc_output_initialize(out_arr, Bodo_FTypes::sum, true);
                do_apply_to_column(combined_arr, out_arr, dummy_aux_cols,
                                   dummy_combine_grp_info, Bodo_FTypes::sum);
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported window function for the global "
                    "implementation");
        }
    }
}

/**
 * @brief Generates a single global average using sum / count
 *
 * @param in_col The input column
 * @param out_arr The array to store the ouput value
 * @param is_parallel Whether to do parallel communication step
 */
template <Bodo_FTypes::FTypeEnum window_func>
    requires(mean<window_func>)
void single_global_window_computation(const std::shared_ptr<array_info>& in_col,
                                      std::shared_ptr<array_info> out_arr,
                                      bool is_parallel) {
    grouping_info dummy_local_grp_info;
    dummy_local_grp_info.num_groups = 1;
    dummy_local_grp_info.row_to_group.resize(in_col->length, 0);
    std::vector<std::shared_ptr<array_info>> dummy_aux_cols;

    // create an intermediate column to hold the result of the sum
    auto [sum_arr_type, sum_dtype] = get_groupby_output_dtype(
        Bodo_FTypes::sum, in_col->arr_type, in_col->dtype);
    std::shared_ptr<array_info> sum_arr =
        alloc_array_top_level(1, -1, -1, sum_arr_type, sum_dtype);
    aggfunc_output_initialize(sum_arr, Bodo_FTypes::sum, true);

    // create an intermediate column to hold the result of the counts
    std::shared_ptr<array_info> count_arr = alloc_numpy(1, Bodo_CTypes::INT64);
    aggfunc_output_initialize(count_arr, Bodo_FTypes::count, true);

    // Implements avg by calculating a single sum and count value for the column
    // and then dividing. This requires looping over the column twice. Could be
    // done in one loop i.e. with do_apply_column and ::mean but would need to
    // be extended to work for Decimals.
    single_global_window_computation<Bodo_FTypes::sum>(in_col, sum_arr,
                                                       is_parallel);
    single_global_window_computation<Bodo_FTypes::count>(in_col, count_arr,
                                                         is_parallel);

    int64_t total_rows = getv<int64_t, bodo_array_type::NUMPY>(count_arr, 0);

    if (total_rows == 0)
        return;

    // Do the division
    if (out_arr->dtype == Bodo_CTypes::FLOAT64) {
        getv<double>(out_arr, 0) =
            avg_sum_to_double(sum_arr) / static_cast<double>(total_rows);
        out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0, true);
    } else {
        bool overflow;

        const arrow::Decimal128& total_rows_asdecimal =
            (arrow::Decimal128)total_rows;
        const arrow::Decimal128& sum_rows =
            *(sum_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                             arrow::Decimal128>());

        arrow::Decimal128* out_ptr =
            out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL,
                           arrow::Decimal128>();

        // For decimal division, we add 6 to the scale (but don't go beyond 12)
        // Note: This needs to be kept consistent with the scale/precision rule
        // for DIVIDE in decimal_arr_ext.py
        int32_t new_scale =
            std::max(in_col->scale, std::min(in_col->scale + 6, 12));
        out_arr->precision = DECIMAL128_MAX_PRECISION;
        out_arr->scale = new_scale;
        *out_ptr = decimalops::Divide(sum_rows, total_rows_asdecimal,
                                      new_scale - in_col->scale, &overflow);

        out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0, true);

        if (overflow) {
            throw std::runtime_error(
                "Overflow detected when trying to compute column average.");
        }
    }
}

/**
 * @brief Special case of single_global_window_computation where we have no
 * input columns The only supported function in this path is count (e.g.
 * count(*) over ()) so the out_arr is populated with row count.
 *
 * @param output_rows The number of rows locally.
 * @param out_arr The singleton array containing the globally computed row count
 * @param is_parallel Whether to sum the row counts across ranks.
 */
template <Bodo_FTypes::FTypeEnum window_func>
    requires(size<window_func>)
void single_global_window_computation(size_t output_rows,
                                      std::shared_ptr<array_info> out_arr,
                                      bool is_parallel) {
    size_t total_rows = output_rows;
    if (is_parallel) {
        MPI_Allreduce(MPI_IN_PLACE, &total_rows, 1, MPI_UINT64_T, MPI_SUM,
                      MPI_COMM_WORLD);
    }
    getv<uint64_t>(out_arr, 0) = total_rows;
}

void global_window_computation(
    const std::vector<int32_t>& window_funcs,
    const std::vector<std::shared_ptr<array_info>>& window_args,
    const std::vector<int32_t>& window_offset_indices,
    std::vector<std::shared_ptr<array_info>>& out_arrs, size_t output_rows,
    bool is_parallel) {
    for (size_t i = 0; i < window_funcs.size(); i++) {
        switch (window_funcs[i]) {
            case Bodo_FTypes::size: {
                // count(*) case the "input column" is essentially just the
                // row count
                single_global_window_computation<Bodo_FTypes::size>(
                    output_rows, out_arrs[i], is_parallel);

                break;
            }
            case Bodo_FTypes::count: {
                int32_t offset = window_offset_indices[i];
                const std::shared_ptr<array_info>& in_col = window_args[offset];
                single_global_window_computation<Bodo_FTypes::count>(
                    in_col, out_arrs[i], is_parallel);
                break;
            }
            case Bodo_FTypes::sum: {
                int32_t offset = window_offset_indices[i];
                const std::shared_ptr<array_info>& in_col = window_args[offset];
                single_global_window_computation<Bodo_FTypes::sum>(
                    in_col, out_arrs[i], is_parallel);
                break;
            }
            case Bodo_FTypes::max: {
                int32_t offset = window_offset_indices[i];
                const std::shared_ptr<array_info>& in_col = window_args[offset];
                single_global_window_computation<Bodo_FTypes::max>(
                    in_col, out_arrs[i], is_parallel);
                break;
            }
            case Bodo_FTypes::min: {
                int32_t offset = window_offset_indices[i];
                const std::shared_ptr<array_info>& in_col = window_args[offset];
                single_global_window_computation<Bodo_FTypes::min>(
                    in_col, out_arrs[i], is_parallel);
                break;
            }
            case Bodo_FTypes::mean: {
                int32_t offset = window_offset_indices[i];
                const std::shared_ptr<array_info>& in_col = window_args[offset];
                single_global_window_computation<Bodo_FTypes::mean>(
                    in_col, out_arrs[i], is_parallel);
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported window function for the global "
                    "implementation");
        }
    }

    // For each output, explode the singleton output row into the
    // complete answer.
    std::vector<int64_t> idxs(output_rows, 0);
    for (size_t i = 0; i < window_funcs.size(); i++) {
        out_arrs[i] = RetrieveArray_SingleColumn(out_arrs[i], idxs);
    }
}

/**
 * @brief Wrapper around sorted window computation to template the comparisons
 * on the order by and partition by columns.
 *
 * @param[in] partition_by_arrs The arrays that hold the partition by values.
 * @param[in] order_by_arrs The arrays that hold the order by values.
 * @param[in] window_args The arrays that hold the window argument values.
 * @param[in] window_offset_indices The vector used to associate elements of
 * window_args with the corresponding function call.
 * @param[in] window_funcs The name(s) of the window function(s) being computed.
 * @param[out] out_arrs The output array(s) being populated.
 * @param is_parallel Is the data distributed? This is used for communicating
 * with a neighboring rank for boundary groups.
 */
void sorted_window_computation(
    std::vector<std::shared_ptr<array_info>>& partition_by_arrs,
    std::vector<std::shared_ptr<array_info>>& order_by_arrs,
    const std::vector<std::shared_ptr<array_info>>& window_args,
    const std::vector<int32_t>& window_offset_indices,
    const std::vector<int32_t>& window_funcs,
    std::vector<std::shared_ptr<array_info>>& out_arrs, size_t out_rows,
    std::vector<std::shared_ptr<DictionaryBuilder>> builders,
    bool is_parallel) {
    if (partition_by_arrs.size() == 0 && order_by_arrs.size() == 0) {
        // If there is no partition/order column, re-route to the global
        // computation.
        global_window_computation(window_funcs, window_args,
                                  window_offset_indices, out_arrs, out_rows,
                                  is_parallel);
        return;
    }

    bool single_part_arr_type = partition_by_arrs.size() > 0;
    for (size_t i = 1; i < partition_by_arrs.size(); i++) {
        if (partition_by_arrs[i]->arr_type != partition_by_arrs[0]->arr_type) {
            single_part_arr_type = false;
            break;
        }
    }
    bool single_order_arr_type = order_by_arrs.size() > 0;
    for (size_t i = 1; i < order_by_arrs.size(); i++) {
        if (order_by_arrs[i]->arr_type != order_by_arrs[0]->arr_type) {
            single_order_arr_type = false;
            break;
        }
    }
    if (!single_part_arr_type && !single_order_arr_type) {
        _sorted_window_computation<bodo_array_type::UNKNOWN,
                                   bodo_array_type::UNKNOWN>(
            partition_by_arrs, order_by_arrs, window_funcs, out_arrs, builders,
            is_parallel);
    } else if (!single_order_arr_type) {
#define SORTED_WINDOW_PART_ATYPE_CASE(PartitionByArrType)             \
    case PartitionByArrType: {                                        \
        _sorted_window_computation<PartitionByArrType,                \
                                   bodo_array_type::UNKNOWN>(         \
            partition_by_arrs, order_by_arrs, window_funcs, out_arrs, \
            builders, is_parallel);                                   \
        break;                                                        \
    }

        switch (partition_by_arrs[0]->arr_type) {
            SORTED_WINDOW_PART_ATYPE_CASE(bodo_array_type::NULLABLE_INT_BOOL);
            SORTED_WINDOW_PART_ATYPE_CASE(bodo_array_type::NUMPY);
            SORTED_WINDOW_PART_ATYPE_CASE(bodo_array_type::STRING);
            SORTED_WINDOW_PART_ATYPE_CASE(bodo_array_type::DICT);
            SORTED_WINDOW_PART_ATYPE_CASE(bodo_array_type::ARRAY_ITEM);
            SORTED_WINDOW_PART_ATYPE_CASE(bodo_array_type::MAP);
            SORTED_WINDOW_PART_ATYPE_CASE(bodo_array_type::STRUCT);
            SORTED_WINDOW_PART_ATYPE_CASE(bodo_array_type::TIMESTAMPTZ);
            default: {
                throw std::runtime_error(
                    "Unsupported partition by array type for the sorted "
                    "implementation of window function" +
                    GetArrType_as_string(partition_by_arrs[0]->arr_type));
            }
        }
#undef SORTED_WINDOW_PART_ATYPE_CASE

    } else if (!single_part_arr_type) {
#define SORTED_WINDOW_ORDER_ATYPE_CASE(OrderByArrType)                        \
    case OrderByArrType: {                                                    \
        _sorted_window_computation<bodo_array_type::UNKNOWN, OrderByArrType>( \
            partition_by_arrs, order_by_arrs, window_funcs, out_arrs,         \
            builders, is_parallel);                                           \
        break;                                                                \
    }
        switch (order_by_arrs[0]->arr_type) {
            SORTED_WINDOW_ORDER_ATYPE_CASE(bodo_array_type::NULLABLE_INT_BOOL);
            SORTED_WINDOW_ORDER_ATYPE_CASE(bodo_array_type::NUMPY);
            SORTED_WINDOW_ORDER_ATYPE_CASE(bodo_array_type::STRING);
            SORTED_WINDOW_ORDER_ATYPE_CASE(bodo_array_type::DICT);
            SORTED_WINDOW_ORDER_ATYPE_CASE(bodo_array_type::ARRAY_ITEM);
            SORTED_WINDOW_ORDER_ATYPE_CASE(bodo_array_type::MAP);
            SORTED_WINDOW_ORDER_ATYPE_CASE(bodo_array_type::STRUCT);
            SORTED_WINDOW_ORDER_ATYPE_CASE(bodo_array_type::TIMESTAMPTZ);
            default: {
                throw std::runtime_error(
                    "Unsupported partition by array type for the sorted "
                    "implementation of window function" +
                    GetArrType_as_string(partition_by_arrs[0]->arr_type));
            }
        }
#undef SORTED_WINDOW_ORDER_ATYPE_CASE
    } else {
#define SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE(PartitionByArrType,       \
                                              OrderByArrType)           \
    case OrderByArrType: {                                              \
        _sorted_window_computation<PartitionByArrType, OrderByArrType>( \
            partition_by_arrs, order_by_arrs, window_funcs, out_arrs,   \
            builders, is_parallel);                                     \
        break;                                                          \
    }

#define SORTED_WINDOW_NESTED_PART_ATYPE_CASE(PartitionByArrType)            \
    case PartitionByArrType: {                                              \
        switch (order_by_arrs[0]->arr_type) {                               \
            SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE(                          \
                PartitionByArrType, bodo_array_type::NULLABLE_INT_BOOL);    \
            SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE(PartitionByArrType,       \
                                                  bodo_array_type::NUMPY);  \
            SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE(PartitionByArrType,       \
                                                  bodo_array_type::STRING); \
            SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE(PartitionByArrType,       \
                                                  bodo_array_type::DICT);   \
            SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE(PartitionByArrType,       \
                                                  bodo_array_type::MAP);    \
            SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE(PartitionByArrType,       \
                                                  bodo_array_type::STRUCT); \
            SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE(                          \
                PartitionByArrType, bodo_array_type::ARRAY_ITEM);           \
            SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE(                          \
                PartitionByArrType, bodo_array_type::TIMESTAMPTZ);          \
            default: {                                                      \
                throw std::runtime_error(                                   \
                    "Unsupported partition by array type for the sorted "   \
                    "implementation of window function" +                   \
                    GetArrType_as_string(partition_by_arrs[0]->arr_type));  \
            }                                                               \
        }                                                                   \
        break;                                                              \
    }

        switch (partition_by_arrs[0]->arr_type) {
            SORTED_WINDOW_NESTED_PART_ATYPE_CASE(
                bodo_array_type::NULLABLE_INT_BOOL);
            SORTED_WINDOW_NESTED_PART_ATYPE_CASE(bodo_array_type::NUMPY);
            SORTED_WINDOW_NESTED_PART_ATYPE_CASE(bodo_array_type::STRING);
            SORTED_WINDOW_NESTED_PART_ATYPE_CASE(bodo_array_type::DICT);
            SORTED_WINDOW_NESTED_PART_ATYPE_CASE(bodo_array_type::ARRAY_ITEM);
            SORTED_WINDOW_NESTED_PART_ATYPE_CASE(bodo_array_type::MAP);
            SORTED_WINDOW_NESTED_PART_ATYPE_CASE(bodo_array_type::STRUCT);
            SORTED_WINDOW_NESTED_PART_ATYPE_CASE(bodo_array_type::TIMESTAMPTZ);
            default: {
                throw std::runtime_error(
                    "Unsupported partition by array type for the sorted "
                    "implementation of window function" +
                    GetArrType_as_string(partition_by_arrs[0]->arr_type));
            }
        }

#undef SORTED_WINDOW_NESTED_ORDER_ATYPE_CASE
#undef SORTED_WINDOW_NESTED_PART_ATYPE_CASE
    }
}
