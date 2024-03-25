
// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_join.h"
#include "_shuffle.h"

/**
 * @brief Check if an array contains a NA value according
 * to SQL rules at a given index.
 *
 * TODO: Template on array_type + move to a helper function if/when its used
 * in more places.
 *
 */
inline bool isna_sql(const std::shared_ptr<array_info>& arr,
                     const size_t& idx) {
    if (arr->null_bitmask() != nullptr && !arr->get_null_bit(idx)) {
        return true;
    }
    // Datetime still uses a NaT value. TODO: Remove.
    if (arr->dtype == Bodo_CTypes::DATETIME) {
        assert(arr->arr_type == bodo_array_type::NUMPY);
        int64_t* dt64_arr = (int64_t*)arr->data1<bodo_array_type::NUMPY>();
        return dt64_arr[idx] == std::numeric_limits<int64_t>::min();
    }
    return false;
}

inline bool is_point_right_of_interval_start(
    const std::shared_ptr<array_info>& left_interval_col, const size_t& int_idx,
    const std::shared_ptr<array_info>& point_col, const size_t& point_idx,
    bool strictly_right) {
    if (isna_sql(point_col, point_idx)) {
        return false;
    }

    auto comp = KeyComparisonAsPython_Column(true, left_interval_col, int_idx,
                                             point_col, point_idx);
    return strictly_right ? comp > 0 : comp >= 0;
}

inline bool is_point_left_of_interval_end(
    const std::shared_ptr<array_info>& right_interval_col,
    const size_t& int_idx, const std::shared_ptr<array_info>& point_col,
    const size_t& point_idx, bool strictly_left) {
    // Note: No need to check if Point is NA because we will have already
    // skipped it in is_point_right_of_interval_start.

    auto comp = KeyComparisonAsPython_Column(true, right_interval_col, int_idx,
                                             point_col, point_idx);
    return strictly_left ? comp < 0 : comp <= 0;
}

/**
 * @brief Perform the merge step of sort-merge join for a point-in-interval join
 *
 * This algorithm assumes that both tables are sorted. Note that the interval
 * table will be scanned once but we can backtrack on the point table since
 * intervals on the interval table can overlap, and a point can be in multiple
 * intervals.
 *
 * @param interval_table Table containing the 2 interval columns
 * @param point_table Table containing the point column that needs to be between
 * the interval values
 * @param cond_func Full condition function to determine if point is in interval
 * @param interval_start_col_id Index of the left interval column in
 * interval_table
 * @param interval_end_col_id Index of the right interval column in
 * interval_table
 * @param point_col_id Index of the point column in point_table
 * @param curr_rank Current MPI rank
 * @param n_pes Total number of MPI ranks running in program
 * @param interval_parallel True if interval_table if data distributed, false if
 * replicated
 * @param point_parallel True if point_table if data distributed, false if
 * replicated
 * @param is_point_outer Is the join an outer join on the point table? If so,
 * include any point rows without matching interval rows in the final output
 * @param is_strict_contained In the join condition, is the point required to be
 * strictly between the interval endpoints? Only false when condition is l <= p
 * <= r
 * @param is_strict_start_cond In the join condition, is the point required to
 * be strictly right of the left interval? True when l < p, false when l <= p
 * @param is_strict_end_cond In the join condition, is the point required to
 * be strictly left of the right interval? True when p < r, false when p <= r
 * @return A pair of vectors of indexes to the left and right table
 * representing the output
 */
std::pair<bodo::vector<int64_t>, bodo::vector<int64_t>> interval_merge(
    std::shared_ptr<table_info> interval_table,
    std::shared_ptr<table_info> point_table, uint64_t interval_start_col_id,
    uint64_t interval_end_col_id, uint64_t point_col_id, int curr_rank,
    int n_pes, bool interval_parallel, bool point_parallel, bool is_point_outer,
    bool is_strict_contained, bool is_strict_start_cond,
    bool is_strict_end_cond) {
    tracing::Event ev("interval_merge", interval_parallel || point_parallel);

    // When the point side is empty, the output will be empty regardless of
    // whether it's an inner join or a point-outer join.
    // Note that in the case that the interval side is empty, but the point
    // side is not, the output won't be empty in case of a point-outer join
    // (it'll be the point table plus nulls for all the columns from the
    // interval side).
    if (point_table->nrows() == 0) {
        ev.add_attribute("out_num_rows", 0);
        ev.add_attribute("out_num_inner_rows", 0);
        ev.add_attribute("out_num_outer_rows", 0);
        ev.finalize();
        return std::pair(bodo::vector<int64_t>(), bodo::vector<int64_t>());
    }

    auto [interval_arr_infos, interval_col_data, interval_col_null] =
        get_gen_cond_data_ptrs(interval_table);
    auto [point_arr_infos, point_col_data, point_col_null] =
        get_gen_cond_data_ptrs(point_table);

    // Start Col of Interval Table, End Col of Interval Table and Point Col
    std::shared_ptr<array_info> left_inter_col =
        interval_table->columns[interval_start_col_id];
    std::shared_ptr<array_info> right_inter_col =
        interval_table->columns[interval_end_col_id];
    std::shared_ptr<array_info> point_col = point_table->columns[point_col_id];

    // Rows of the Output Joined Table
    bodo::vector<int64_t> joined_interval_idxs;
    bodo::vector<int64_t> joined_point_idxs;

    // Bitmask indicating all matched rows in left table
    size_t n_bytes_point = is_point_outer ? (point_table->nrows() + 7) >> 3 : 0;
    bodo::vector<uint8_t> point_matched_rows(n_bytes_point, 0);

    // Set 500K batch size to make sure batch data of all cores fits in L3
    // cache.
    int64_t batch_size_bytes = DEFAULT_BLOCK_SIZE_BYTES;
    char* batch_size = std::getenv("BODO_INTERVAL_JOIN_BATCH_SIZE");
    if (batch_size) {
        batch_size_bytes = std::stoi(batch_size);
    }
    if (batch_size_bytes <= 0) {
        throw std::runtime_error("interval_join_table: batch_size_bytes <= 0");
    }

    uint64_t point_pos = 0;
    // Keep track of the previous end for values with the same start
    // interval. We sort ties by the end interval in Ascending order, so
    // any entries with the same start value can skip some additional checks.
    // prev_point_pos_end is the first not matched value.
    uint64_t prev_point_pos_end = 0;
    for (uint64_t interval_pos = 0; interval_pos < interval_table->nrows();
         interval_pos++) {
        // Skip intervals that contain NA values, resetting prev_point_pos_end.
        // XXX TODO The isna_sql checks are expensive because they're not
        // templated for the array types, i.e. it will go through a switch
        // statement for every row multiple times. Needs to be fixed.
        if (isna_sql(left_inter_col, interval_pos) ||
            isna_sql(right_inter_col, interval_pos)) {
            // Skip intervals that contain NA values, resetting
            // prev_point_pos_end.
            prev_point_pos_end = point_pos;
            continue;
        }

        // Find first row in the point table thats in the interval
        while (point_pos < point_table->nrows() &&
               !is_point_right_of_interval_start(left_inter_col, interval_pos,
                                                 point_col, point_pos,
                                                 is_strict_start_cond)) {
            point_pos++;
        }
        if (point_pos >= point_table->nrows())
            break;

        // Starting location for the current interval. If the
        // start is the same as the previous interval we can skip
        // ahead.
        uint64_t start_point_pos = point_pos;
        // Check if the intervals start at the same point and the previous
        // interval had any matches.
        if (interval_pos != 0 && prev_point_pos_end > point_pos) {
            // Note: We don't need NA to match because NA should never have
            // matched any entries.
            bool start_equal =
                TestEqualColumn(left_inter_col, interval_pos, left_inter_col,
                                interval_pos - 1, false);
            if (start_equal) {
                for (uint64_t i = point_pos; i < prev_point_pos_end; i++) {
                    joined_interval_idxs.push_back(interval_pos);
                    joined_point_idxs.push_back(i);
                    // Note we don't need to update point_matched_rows because
                    // these values have already matched.
                }
                start_point_pos = prev_point_pos_end;
            }
        }

        // Because tables are sorted, a consecutive range of rows in point table
        // will fit in the interval.
        // Thus, we loop and match all until outside of interval. Then reset.
        for (uint64_t curr_point = start_point_pos;
             curr_point < point_table->nrows(); curr_point++) {
            bool match = is_point_left_of_interval_end(
                right_inter_col, interval_pos, point_col, curr_point,
                is_strict_end_cond);
            if (match) {
                joined_interval_idxs.push_back(interval_pos);
                joined_point_idxs.push_back(curr_point);
                if (is_point_outer) {
                    SetBitTo(point_matched_rows.data(), curr_point, true);
                }
            } else {
                // Update where we had our first !match to skip checks on
                // future iterations.
                prev_point_pos_end = curr_point;
                break;
            }
        }
    }

    // Interval table is always on the inner side, so number of matched rows
    // will always be the size of the matches on the interval table.
    auto num_matched_rows = joined_interval_idxs.size();
    if (is_point_outer) {
        add_unmatched_rows(point_matched_rows, point_table->nrows(),
                           joined_point_idxs, joined_interval_idxs,
                           !point_parallel && interval_parallel);
    }

    ev.add_attribute("out_num_rows", joined_interval_idxs.size());
    ev.add_attribute("out_num_inner_rows", num_matched_rows);
    ev.add_attribute("out_num_outer_rows",
                     joined_interval_idxs.size() - num_matched_rows);
    return std::pair(std::move(joined_interval_idxs),
                     std::move(joined_point_idxs));
}

table_info* interval_join_table(
    table_info* in_left_table, table_info* in_right_table, bool left_parallel,
    bool right_parallel, bool is_left, bool is_right, bool is_left_point,
    bool strict_start, bool strict_end, uint64_t point_col_id,
    uint64_t interval_start_col_id, uint64_t interval_end_col_id,
    bool* key_in_output, int64_t* use_nullable_arr_type,
    bool rebalance_if_skewed, uint64_t* num_rows_ptr) {
    try {
        std::shared_ptr<table_info> left_table =
            std::shared_ptr<table_info>(in_left_table);
        std::shared_ptr<table_info> right_table =
            std::shared_ptr<table_info>(in_right_table);

        // TODO: Make this an assertion
        if ((is_left_point && is_right) || (!is_left_point && is_left)) {
            throw std::runtime_error(
                "Point-In-Interval Join should only support Inner or Left "
                "Joins");
        }

        bool strict_contained = strict_start || strict_end;
        bool parallel_trace = (left_parallel || right_parallel);
        tracing::Event ev("interval_join_table", parallel_trace);
        ev.add_attribute("left_table_len", left_table->nrows());
        ev.add_attribute("right_table_len", right_table->nrows());
        ev.add_attribute("is_left", is_left);
        ev.add_attribute("is_right", is_right);

        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        tracing::Event ev_sort("interval_join_table_sort", parallel_trace);

        bool left_table_bcast = false;
        bool right_table_bcast = false;

        // use broadcast join if left or right table is small (allgather the
        // small table)
        if (left_parallel && right_parallel) {
            int bcast_join_threshold = get_bcast_join_threshold();
            int64_t left_total_memory = table_global_memory_size(left_table);
            int64_t right_total_memory = table_global_memory_size(right_table);
            if (left_total_memory < right_total_memory &&
                left_total_memory < bcast_join_threshold) {
                // Broadcast the left table
                left_table =
                    gather_table(std::move(left_table), -1, true, true);
                left_parallel = false;
                left_table_bcast = true;
            } else if (right_total_memory <= left_total_memory &&
                       right_total_memory < bcast_join_threshold) {
                // Broadcast the right table
                right_table =
                    gather_table(std::move(right_table), -1, true, true);
                right_parallel = false;
                right_table_bcast = true;
            }
        }
        ev.add_attribute("left_table_bcast", left_table_bcast);
        ev.add_attribute("right_table_bcast", right_table_bcast);

        std::shared_ptr<table_info> point_table;
        std::shared_ptr<table_info> interval_table;
        bool point_table_parallel;
        bool interval_table_parallel;
        bool is_outer_point;
        if (is_left_point) {
            point_table = std::move(left_table);
            interval_table = std::move(right_table);
            point_table_parallel = left_parallel;
            interval_table_parallel = right_parallel;
            is_outer_point = is_left;
        } else {
            point_table = std::move(right_table);
            interval_table = std::move(left_table);
            point_table_parallel = right_parallel;
            interval_table_parallel = left_parallel;
            is_outer_point = is_right;
        }

        // Rearrange point table for sort
        std::unordered_map<uint64_t, uint64_t> point_table_restore_map =
            move_cols_to_front(point_table,
                               std::vector<uint64_t>{point_col_id});

        // Rearrange interval table for sort
        std::unordered_map<uint64_t, uint64_t> interval_table_restore_map =
            move_cols_to_front(interval_table,
                               std::vector<uint64_t>{interval_start_col_id,
                                                     interval_end_col_id});

        // Sort
        // Note: Bad intervals (i.e. start > end) will be filtered out of the
        // interval table during the sort.
        auto [sorted_point_table, sorted_interval_table] =
            sort_tables_for_point_in_interval_join(
                std::move(point_table), std::move(interval_table),
                point_table_parallel, interval_table_parallel,
                strict_contained);

        // Put point column back in the right location
        restore_col_order(sorted_point_table, point_table_restore_map);

        // Put interval columns back in the right location
        restore_col_order(sorted_interval_table, interval_table_restore_map);

        ev_sort.add_attribute("sorted_point_table_len",
                              sorted_point_table->nrows());
        ev_sort.add_attribute("sorted_interval_table_len",
                              sorted_interval_table->nrows());
        ev_sort.finalize();

        auto [interval_idxs, point_idxs] = interval_merge(
            sorted_interval_table, sorted_point_table, interval_start_col_id,
            interval_end_col_id, point_col_id, myrank, n_pes,
            interval_table_parallel, point_table_parallel, is_outer_point,
            strict_contained, strict_start, strict_end);

        std::shared_ptr<table_info> sorted_left_table, sorted_right_table;
        bodo::vector<int64_t> left_idxs, right_idxs;
        if (is_left_point) {
            sorted_left_table = std::move(sorted_point_table);
            sorted_right_table = std::move(sorted_interval_table);
            left_idxs = point_idxs;
            right_idxs = interval_idxs;
        } else {
            sorted_left_table = std::move(sorted_interval_table);
            sorted_right_table = std::move(sorted_point_table);
            left_idxs = interval_idxs;
            right_idxs = point_idxs;
        }
        std::shared_ptr<table_info> out_table = create_out_table(
            std::move(sorted_left_table), std::move(sorted_right_table),
            left_idxs, right_idxs, key_in_output, use_nullable_arr_type,
            nullptr, 0, nullptr, 0);

        // Check for skew if BodoSQL suggested we should
        if (rebalance_if_skewed && (left_parallel || right_parallel)) {
            out_table = rebalance_join_output(std::move(out_table));
        }

        // number of local output rows is passed to Python in case all output
        // columns are dead.
        *num_rows_ptr = out_table->nrows();
        ev.add_attribute("out_table_nrows", *num_rows_ptr);
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
