#include "_window_compute.h"
#include <arrow/util/decimal.h>
#include <mpi.h>
#include <memory>
#include <stdexcept>

#include "../_array_utils.h"
#include "../_bodo_common.h"
#include "../_decimal_ext.h"
#include "../_dict_builder.h"
#include "../_distributed.h"
#include "../groupby/_groupby_col_set.h"
#include "../groupby/_groupby_common.h"
#include "../groupby/_groupby_do_apply_to_column.h"
#include "../groupby/_groupby_ftypes.h"
#include "../vendored/_gandiva_decimal_copy.h"
#include "_window_aggfuncs.h"

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
    if (n == 0) {
        return;
    }
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

/**
 * @brief Helper function for lead/lag that translates an array of sorted
 * indices and group info to an list of unsorted indices
 *
 * Loops over sorted indices
 * Translate sortedIdx -> sortedIdx + shift (do bounds check)
 * IdxArray[unsort(sortedIdx)] = unsort(sortedIdx + shift)
 *
 * @param sorted_groups
 * @param sorted_idx
 * @param shifted_idxs
 * @param shift_amt
 * @tparam ignore_nulls
 */
template <bool ignore_nulls>
    requires(!ignore_nulls)
void translate_indices(std::shared_ptr<array_info> sorted_groups,
                       std::shared_ptr<array_info> sorted_idx,
                       std::vector<int64_t>& shifted_idxs, int64_t shift_amt) {
    const int64_t n_rows = sorted_groups->length;
    const int64_t out_of_bounds_val = -1;
    // loop over sorted array of indexes and perform the shift.
    // Check if the shifted index is out of bounds (for the group or the entire
    // column)
    for (int64_t i = 0; i < n_rows; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        int64_t shifted_idx = i + shift_amt;
        // the index to populate in the output column
        int64_t unsorted_idx = getv<int64_t>(sorted_idx, i);
        // bounds check
        if (shifted_idx < 0 || shifted_idx >= n_rows ||
            getv<int64_t>(sorted_groups, shifted_idx) != curr_group) {
            shifted_idxs[unsorted_idx] = out_of_bounds_val;
        } else {
            int64_t unsorted_shifted_idx =
                getv<int64_t>(sorted_idx, shifted_idx);
            shifted_idxs[unsorted_idx] = unsorted_shifted_idx;
        }
    }
}

template <bool ignore_nulls>
void lead_lag_computation(std::shared_ptr<array_info>& out_arr,
                          std::shared_ptr<array_info> sorted_groups,
                          std::shared_ptr<array_info> sorted_idx,
                          std::shared_ptr<array_info> in_col,
                          std::shared_ptr<array_info> default_value,
                          int64_t shift_amt, bodo::IBufferPool* const pool,
                          std::shared_ptr<::arrow::MemoryManager> mm) {
    size_t n_rows = out_arr->length;

    std::vector<int64_t> shifted_idxs(n_rows, 0);

    // TODO refactor retrieve array two column so that this only needs to be a
    // single row
    std::vector<int64_t> default_idxs(n_rows, 0);

    translate_indices<ignore_nulls>(sorted_groups, sorted_idx, shifted_idxs,
                                    shift_amt);

    out_arr = RetrieveArray_TwoColumns(in_col, default_value, shifted_idxs,
                                       default_idxs, pool, mm);
}

/**
 * @brief Get the window frame args from window args table.
 *
 * @param window_args The table of scalar window args containing the frame
 * bounds.
 * @param window_arg_offset The current index into the window scalar arguments
 * table.
 * @return std::tuple<int64_t*, int64_t*> The lo and hi bounds for the frame.
 */
std::tuple<int64_t*, int64_t*> get_window_frame_args(
    std::shared_ptr<table_info> window_args, int64_t& window_arg_offset) {
    int64_t* frame_lo;
    int64_t* frame_hi;

    // if either of the bounds is "None" the null bit will be set and we will
    // pass in a nullptr
    std::shared_ptr<array_info> frame_lo_col =
        window_args->columns[window_arg_offset];
    if (frame_lo_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        bool frame_lo_null =
            frame_lo_col->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0);
        frame_lo = frame_lo_null ? &(getv<int64_t>(frame_lo_col, 0)) : nullptr;
    } else {
        // arr_type == NUMPY
        frame_lo = &(getv<int64_t>(frame_lo_col, 0));
    }
    window_arg_offset = window_arg_offset + 1;

    std::shared_ptr<array_info> frame_hi_col =
        window_args->columns[window_arg_offset];
    if (frame_hi_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        bool frame_hi_null =
            frame_hi_col->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0);
        frame_hi = frame_hi_null ? &(getv<int64_t>(frame_hi_col, 0)) : nullptr;
    } else {
        // arr_type == NUMPY
        frame_hi = &(getv<int64_t>(frame_hi_col, 0));
    }

    window_arg_offset = window_arg_offset + 1;

    return std::make_tuple(frame_lo, frame_hi);
}

void window_computation(
    std::vector<std::shared_ptr<array_info>>& input_arrs,
    std::vector<int64_t> window_funcs,
    std::vector<std::shared_ptr<array_info>>& out_arrs,
    std::vector<std::shared_ptr<DictionaryBuilder>>& out_dict_builders,
    grouping_info const& grp_info, const std::vector<bool>& asc_vect,
    const std::vector<bool>& na_pos_vect,
    const std::shared_ptr<table_info> window_args, int n_input_cols,
    bool is_parallel, bool use_sql_rules, bodo::IBufferPool* const pool,
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
    for (long long window_func : window_funcs) {
        if (window_func != Bodo_FTypes::min_row_number_filter) {
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
                ntile_computation(
                    out_arrs[i], iter_table->columns[0],
                    iter_table->columns[idx_col],
                    getv<int64_t>(window_args->columns[window_arg_offset], 0));
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
                auto [frame_lo, frame_hi] =
                    get_window_frame_args(window_args, window_arg_offset);
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
                auto [frame_lo, frame_hi] =
                    get_window_frame_args(window_args, window_arg_offset);

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
            case Bodo_FTypes::lead:
            case Bodo_FTypes::lag: {
                std::shared_ptr<array_info> input_col =
                    input_arrs[window_col_offset];
                // shift amount is standardized so that we can do one function
                // call
                int64_t shift_amt =
                    getv<int64_t>(window_args->columns[window_arg_offset], 0);
                shift_amt = (window_funcs[i] == Bodo_FTypes::lag)
                                ? -1 * shift_amt
                                : shift_amt;
                // default value could be any type or NULL
                std::shared_ptr<array_info> default_value =
                    window_args->columns[window_arg_offset + 1];

                // if there is a dictionary builder for the input/output
                // we need to unify with the default value.
                int64_t dict_builder_offset =
                    (out_dict_builders.size() - window_funcs.size()) + i;
                if (out_dict_builders[dict_builder_offset] != nullptr) {
                    default_value = out_dict_builders[dict_builder_offset]
                                        ->UnifyDictionaryArray(default_value);
                    // technically we shouldn't need to do this but
                    // RetrieveArray_TwoColumn was complaining...
                    set_array_dict_from_builder(
                        input_col, out_dict_builders[dict_builder_offset]);
                }

                // TODO add ignore nulls case
                lead_lag_computation<false>(out_arrs[i], iter_table->columns[0],
                                            iter_table->columns[idx_col],
                                            input_col, default_value, shift_amt,
                                            pool, mm);
                window_col_offset++;
                window_arg_offset = window_arg_offset + 2;
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
 * @param chunks The chunks of input data.
 * @param input_idx The input column index within chunks.
 * @param out_arr The array to store the single-row result in.
 * @param is_parallel Whether to do parallel communication step.
 */
template <Bodo_FTypes::FTypeEnum window_func>
    requires(no_aux_col<window_func>)
void single_global_window_computation(
    const std::vector<std::shared_ptr<table_info>>& chunks, size_t input_idx,
    std::shared_ptr<array_info>& out_arr, bool is_parallel) {
    for (auto& it : chunks) {
        std::shared_ptr<array_info> in_col = it->columns[input_idx];
        in_col->pin();
        grouping_info dummy_local_grp_info;
        dummy_local_grp_info.num_groups = 1;
        dummy_local_grp_info.row_to_group.resize(in_col->length, 0);

        std::vector<std::shared_ptr<array_info>> dummy_aux_cols;

        do_apply_to_column(in_col, out_arr, dummy_aux_cols,
                           dummy_local_grp_info, window_func);
        in_col->unpin();
    }

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
 * @tparam window_func The window func to compute. Assumes the window function
 * does not require any auxillary columns.
 * @param chunks The chunks of input data.
 * @param input_idx The input column index within chunks.
 * @param out_arr The array to store the single-row result in.
 * @param is_parallel Whether to do parallel communication step.
 */
template <Bodo_FTypes::FTypeEnum window_func>
    requires(mean<window_func>)
void single_global_window_computation(
    const std::vector<std::shared_ptr<table_info>>& chunks, size_t input_idx,
    std::shared_ptr<array_info>& out_arr, bool is_parallel) {
    // create an intermediate column to hold the result of the sum
    std::shared_ptr<array_info>& in_col = chunks[0]->columns[input_idx];
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
    single_global_window_computation<Bodo_FTypes::sum>(chunks, input_idx,
                                                       sum_arr, is_parallel);
    single_global_window_computation<Bodo_FTypes::count>(
        chunks, input_idx, count_arr, is_parallel);

    int64_t total_rows = getv<int64_t, bodo_array_type::NUMPY>(count_arr, 0);

    if (total_rows == 0) {
        return;
    }

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
 * @param chunks The chunks of input data.
 * @param out_arr The singleton array containing the globally computed row
 * count.
 * @param is_parallel Whether to sum the row counts across ranks.
 */
template <Bodo_FTypes::FTypeEnum window_func>
    requires(size<window_func>)
void single_global_window_computation(
    std::vector<std::shared_ptr<table_info>>& chunks,
    std::shared_ptr<array_info> out_arr, bool is_parallel) {
    size_t total_rows = 0;
    for (auto& it : chunks) {
        total_rows += it->nrows();
    }
    if (is_parallel) {
        CHECK_MPI(
            MPI_Allreduce(MPI_IN_PLACE, &total_rows, 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD),
            "single_global_window_computation: MPI error on MPI_Allreduce:");
    }
    getv<uint64_t>(out_arr, 0) = total_rows;
}

void global_window_computation(
    std::vector<std::shared_ptr<table_info>>& chunks,
    const std::vector<int32_t>& window_funcs,
    const std::vector<int32_t>& window_input_indices,
    const std::vector<int32_t>& window_offset_indices,
    std::vector<std::shared_ptr<array_info>>& out_arrs, bool is_parallel) {
    for (size_t func_idx = 0; func_idx < window_funcs.size(); func_idx++) {
        switch (window_funcs[func_idx]) {
            case Bodo_FTypes::size: {
                // count(*) case the "input column" is essentially just the
                // row count
                single_global_window_computation<Bodo_FTypes::size>(
                    chunks, out_arrs[func_idx], is_parallel);

                break;
            }
            case Bodo_FTypes::count: {
                int32_t offset = window_offset_indices[func_idx];
                int32_t input_idx = window_input_indices[offset];
                single_global_window_computation<Bodo_FTypes::count>(
                    chunks, input_idx, out_arrs[func_idx], is_parallel);
                break;
            }
            case Bodo_FTypes::sum: {
                int32_t offset = window_offset_indices[func_idx];
                int32_t input_idx = window_input_indices[offset];
                single_global_window_computation<Bodo_FTypes::sum>(
                    chunks, input_idx, out_arrs[func_idx], is_parallel);
                break;
            }
            case Bodo_FTypes::max: {
                int32_t offset = window_offset_indices[func_idx];
                int32_t input_idx = window_input_indices[offset];
                single_global_window_computation<Bodo_FTypes::max>(
                    chunks, input_idx, out_arrs[func_idx], is_parallel);
                break;
            }
            case Bodo_FTypes::min: {
                int32_t offset = window_offset_indices[func_idx];
                int32_t input_idx = window_input_indices[offset];
                single_global_window_computation<Bodo_FTypes::min>(
                    chunks, input_idx, out_arrs[func_idx], is_parallel);
                break;
            }
            case Bodo_FTypes::mean: {
                int32_t offset = window_offset_indices[func_idx];
                int32_t input_idx = window_input_indices[offset];
                single_global_window_computation<Bodo_FTypes::mean>(
                    chunks, input_idx, out_arrs[func_idx], is_parallel);
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported window function for the global "
                    "implementation");
        }
    }
}
