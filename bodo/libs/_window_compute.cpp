// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_ftypes.h"
#include "_window_aggfuncs.h"

/**
 * min_row_number_filter is used to evaluate the following type
 * of expression in BodoSQL:
 *
 * row_number() over (partition by ... order by ...) == 1.
 *
 * This function creates a boolean array of all-false, then finds the indices
 * corresponding to the idxmin/idxmax of the orderby columns and sets those
 * indices to true. This implementaiton does so without sorting the array since
 * if no other window functions being calculated require sorting, then we can
 * find the idxmin/idxmax without bothering to sort the whole table seciton.
 *
 * @param[in] orderby_arrs: the columns used in the order by clause of the query
 * @param[in,out] out_arr: output array where the true values will be placed
 * @param[in] grp_info: groupby information
 * @param[in] asc_vect: vector indicating which of the orderby columns are
 * ascending
 * @param[in] na_pos_vect: vector indicating which of the orderby columns are
 * null first/last
 * @param[in] is_parallel: is the function being run in parallel?
 * @param[in] use_sql_rules: should initialization functions obey SQL semantics?
 */
void min_row_number_filter_window_computation_no_sort(
    std::vector<std::shared_ptr<array_info>>& orderby_arrs,
    std::shared_ptr<array_info> out_arr, grouping_info const& grp_info,
    const std::vector<bool>& asc_vect, const std::vector<bool>& na_pos_vect,
    bool is_parallel, bool use_sql_rules) {
    // To compute min_row_number_filter we want to find the
    // idxmin/idxmax based on the orderby columns. Then in the output
    // array those locations will have the value true. We have already
    // initialized all other locations to false.
    size_t num_groups = grp_info.num_groups;
    int64_t ftype;
    std::shared_ptr<array_info> idx_col;
    if (orderby_arrs.size() == 1) {
        // We generate an optimized and templated path for 1 column.
        std::shared_ptr<array_info> orderby_arr = orderby_arrs[0];
        bool asc = asc_vect[0];
        bool na_pos = na_pos_vect[0];
        bodo_array_type::arr_type_enum idx_arr_type;
        if (asc) {
            // The first value of an array in ascending order is the
            // min.
            if (na_pos) {
                ftype = Bodo_FTypes::idxmin;
                // We don't need null values for indices
                idx_arr_type = bodo_array_type::NUMPY;
            } else {
                ftype = Bodo_FTypes::idxmin_na_first;
                // We need null values to signal we found an NA
                // value.
                idx_arr_type = bodo_array_type::NULLABLE_INT_BOOL;
            }
        } else {
            // The first value of an array in descending order is the
            // max.
            if (na_pos) {
                ftype = Bodo_FTypes::idxmax;
                // We don't need null values for indices
                idx_arr_type = bodo_array_type::NUMPY;
            } else {
                ftype = Bodo_FTypes::idxmax_na_first;
                // We need null values to signal we found an NA
                // value.
                idx_arr_type = bodo_array_type::NULLABLE_INT_BOOL;
            }
        }
        // Allocate intermediate buffer to find the true element for
        // each group.
        idx_col = alloc_array_top_level(num_groups, 1, 1, idx_arr_type,
                                        Bodo_CTypes::UINT64);
        // create array to store min/max value
        std::shared_ptr<array_info> data_col = alloc_array_top_level(
            num_groups, 1, 1, orderby_arr->arr_type, orderby_arr->dtype);
        // Initialize the index column.
        if (ftype == Bodo_FTypes::idxmin || ftype == Bodo_FTypes::idxmax) {
            // Initialize indices to first row in group to handle all NA case in
            // idxmin/idxmax.
            for (size_t group_idx = 0; group_idx < idx_col->length;
                 group_idx++) {
                getv<int64_t>(idx_col, group_idx) =
                    grp_info.group_to_first_row[group_idx];
            }
        } else {
            // This is 0 initialized and will not initialize the null values.
            // idxmin_na_first/idxmax_na_first handle the all NA case during
            // computation.
            aggfunc_output_initialize(idx_col, Bodo_FTypes::count,
                                      use_sql_rules);
        }
        std::vector<std::shared_ptr<array_info>> aux_cols = {idx_col};
        // Initialize the min/max column
        if (ftype == Bodo_FTypes::idxmax ||
            ftype == Bodo_FTypes::idxmax_na_first) {
            aggfunc_output_initialize(data_col, Bodo_FTypes::max,
                                      use_sql_rules);
        } else {
            aggfunc_output_initialize(data_col, Bodo_FTypes::min,
                                      use_sql_rules);
        }
        // Compute the idxmin/idxmax
        do_apply_to_column(orderby_arr, data_col, aux_cols, grp_info, ftype);
    } else {
        ftype = Bodo_FTypes::idx_n_columns;
        // We don't need null for indices
        // We only allocate an index column.
        idx_col = alloc_array_top_level(
            num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
        aggfunc_output_initialize(idx_col, Bodo_FTypes::count, use_sql_rules);
        // Call the idx_n_columns function path.
        idx_n_columns_apply(idx_col, orderby_arrs, asc_vect, na_pos_vect,
                            grp_info, ftype);
    }
    // Now we have the idxmin/idxmax in the idx_col for each group.
    // We need to set the corresponding indices in the final array to true.
    for (size_t group_idx = 0; group_idx < idx_col->length; group_idx++) {
        int64_t row_one_idx = getv<int64_t>(idx_col, group_idx);
        SetBitTo((uint8_t*)out_arr->data1(), row_one_idx, true);
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
    for (uint64_t i = 0; i < out_arr->length; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        // If the current group is differne from the group of the previous row,
        // then this row is the row where the row number is 1
        if (curr_group != prev_group) {
            int64_t row_one_idx = getv<int64_t>(sorted_idx, i);
            SetBitTo((uint8_t*)out_arr->data1(), row_one_idx, true);
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
 * the previous row when performing a rank computation.
 *
 * @param[in] sorted_orderbys: the columns used to order the table
 * when performing a window computation.
 * @param[in] i: the row that is being queried to see if it is distinct
 * from the previous row.
 */
inline bool distinct_from_previous_row(
    std::vector<std::shared_ptr<array_info>> sorted_orderbys, int64_t i) {
    if (i == 0) {
        return true;
    }
    for (auto arr : sorted_orderbys) {
        if (!TestEqualColumn(arr, i, arr, i - 1, true)) {
            return true;
        }
    }
    return false;
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
inline void rank_tie_upward_batch_update(std::shared_ptr<array_info> rank_arr,
                                         int64_t group_start_idx,
                                         int64_t tie_start_idx,
                                         int64_t tie_end_idx) {
    int64_t fill_value = tie_end_idx - group_start_idx;
    std::fill((int64_t*)(rank_arr->data1()) + tie_start_idx,
              (int64_t*)(rank_arr->data1()) + tie_end_idx, fill_value);
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
        } else if (distinct_from_previous_row(sorted_orderbys, i)) {
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
 * (so each partition is clustered togeher) and then by the columns in the
 * orderby clause.
 *
 * The computaiton works by having a counter that starts at 1, increases by 1
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
        } else if (distinct_from_previous_row(sorted_orderbys, i)) {
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
 * (so each partition is clustered togeher) and then by the columns in the
 * orderby clause.
 *
 * The computaiton works by calculating the regular rank for each row. Then,
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
        } else if (distinct_from_previous_row(sorted_orderbys, i)) {
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
        } else if (distinct_from_previous_row(sorted_orderbys, i)) {
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
            if (getv<uint8_t>(input_col, idx))
                counter++;
        } else {
            if (input_col->get_null_bit(idx) &&
                GetBit((uint8_t*)input_col->data1(), idx))
                counter++;
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
        // the most recent non-null row, increment the counter
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

/**
 * Computes a batch of window functions for BodoSQL on a subset of a table
 * containing complete partitions. All of the window functions in the
 * batch use the same partitioning and orderby scheme. If any of the window
 * functions require the table to be sorted in order for the result to be
 * calculated, performs a sort on the table first by group and then by
 * the orderby columns.
 *
 * @param[in] input_arrs: the columns being used to order the rows of
 * the table when computing a window function, as well as any additional
 * columns that are being aggregated by the window functions.
 * @param[in] window_funcs: the vector of window functions being computed
 * @param[in,out] out_arrs: the arrays where the final answer for each window
 * function computed will be stored
 * @param[in] asc_vect: vector indicating which of the orderby columns are
 * to be sorted in ascending vs descending order
 * @param[in] na_pos_vect: vector indicating which of the orderby columns are
 * to place nulls first vs last
 * @param[in] window_args: vector of any scalar arguments for the window
 * functions being calculated. It is the responsibility of each function to know
 * how many arguments it is expected and to cast them to the correct type.
 * @param[in] n_input_cols: the number of arrays from input_arrs that correspond
 * to inputs to the window functions. If there are any, they are at the end
 * of the vector in the same order as the functions in window_funcs.
 * @param[in] is_parallel: is the function being run in parallel?
 * @param[in] use_sql_rules: should initialization functions obey SQL semantics?
 */
void window_computation(std::vector<std::shared_ptr<array_info>>& input_arrs,
                        std::vector<int64_t> window_funcs,
                        std::vector<std::shared_ptr<array_info>> out_arrs,
                        grouping_info const& grp_info,
                        const std::vector<bool>& asc_vect,
                        const std::vector<bool>& na_pos_vect,
                        const std::vector<void*>& window_args, int n_input_cols,
                        bool is_parallel, bool use_sql_rules) {
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
                alloc_numpy(num_rows, Bodo_CTypes::INT64);
            for (size_t i = 0; i < num_rows; i++) {
                getv<int64_t>(idx_arr, i) = i;
            }
            iter_table = grouped_sort(grp_info, orderby_arrs, {idx_arr},
                                      asc_vect, na_pos_vect, 0, is_parallel);
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
            // min_row_number_filter uses a sort-less implementaiton if no
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
                        na_pos_vect, is_parallel, use_sql_rules);
                }
                break;
            }
            case Bodo_FTypes::row_number: {
                row_number_computation(out_arrs[i], iter_table->columns[0],
                                       iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::rank: {
                rank_computation(out_arrs[i], iter_table->columns[0],
                                 std::vector<std::shared_ptr<array_info>>(
                                     iter_table->columns.begin() + 1,
                                     iter_table->columns.begin() + idx_col),
                                 iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::dense_rank: {
                dense_rank_computation(
                    out_arrs[i], iter_table->columns[0],
                    std::vector<std::shared_ptr<array_info>>(
                        iter_table->columns.begin() + 1,
                        iter_table->columns.begin() + idx_col),
                    iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::percent_rank: {
                percent_rank_computation(
                    out_arrs[i], iter_table->columns[0],
                    std::vector<std::shared_ptr<array_info>>(
                        iter_table->columns.begin() + 1,
                        iter_table->columns.begin() + idx_col),
                    iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::cume_dist: {
                cume_dist_computation(
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
