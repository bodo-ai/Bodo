// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_nested_loop_join.h"

#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_nested_loop_join_impl.h"
#include "_shuffle.h"

// design overview:
// https://bodo.atlassian.net/l/cp/Av2ijf9A
table_info* nested_loop_join_table(
    table_info* in_left_table, table_info* in_right_table, bool left_parallel,
    bool right_parallel, bool is_left_outer, bool is_right_outer,
    bool* key_in_output, int64_t* use_nullable_arr_type,
    bool rebalance_if_skewed, cond_expr_fn_batch_t cond_func,
    uint64_t* cond_func_left_columns, uint64_t cond_func_left_column_len,
    uint64_t* cond_func_right_columns, uint64_t cond_func_right_column_len,
    uint64_t* num_rows_ptr) {
    try {
        std::shared_ptr<table_info> left_table =
            std::shared_ptr<table_info>(in_left_table);
        std::shared_ptr<table_info> right_table =
            std::shared_ptr<table_info>(in_right_table);

        nested_loop_join_handle_dict_encoded(left_table, right_table,
                                             left_parallel, right_parallel);
        bool parallel_trace = (left_parallel || right_parallel);
        tracing::Event ev("nested_loop_join_table", parallel_trace);
        std::shared_ptr<table_info> out_table;
        bool left_bcast_join = false;
        bool right_bcast_join = false;

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
                left_bcast_join = true;
            } else if (right_total_memory <= left_total_memory &&
                       right_total_memory < bcast_join_threshold) {
                // Broadcast the right table
                right_table =
                    gather_table(std::move(right_table), -1, true, true);
                right_parallel = false;
                right_bcast_join = true;
            }
        }
        ev.add_attribute("left_bcast_join", left_bcast_join);
        ev.add_attribute("right_bcast_join", right_bcast_join);

        // handle parallel nested loop join by broadcasting one side's table
        // chunks of from every rank (loop over all ranks). For outer join
        // handling, the broadcast side's unmatched rows need to be added right
        // after each iteration since joining on the broadcast table chunk is
        // fully done. This needs a reduction of the bitmap to find all
        // potential matches. Handling outer join for the non-broadcast table
        // should be done after all iterations are done to find all potential
        // matches. No need for reduction of bitmap since chunks are independent
        // and not replicated.
        if (left_parallel && right_parallel) {
            int n_pes, myrank;
            MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            std::vector<std::shared_ptr<table_info>> out_table_chunks;
            out_table_chunks.reserve(n_pes);

            size_t n_rows_left = left_table->nrows();
            size_t n_rows_right = right_table->nrows();

            // broadcast the smaller table to reduce overall communication
            int64_t left_table_size = table_global_memory_size(left_table);
            int64_t right_table_size = table_global_memory_size(right_table);
            bool left_table_bcast = left_table_size < right_table_size;
            std::shared_ptr<table_info> bcast_table = left_table;
            std::shared_ptr<table_info> other_table = right_table;
            size_t n_bytes_other = is_right_outer ? (n_rows_right + 7) >> 3 : 0;
            if (!left_table_bcast) {
                bcast_table = right_table;
                other_table = left_table;
                n_bytes_other = is_left_outer ? (n_rows_left + 7) >> 3 : 0;
            }

            // bcast_row_is_matched is reset in each iteration, but
            // other_row_is_matched is updated across iterations
            bodo::vector<uint8_t> bcast_row_is_matched;
            bodo::vector<uint8_t> other_row_is_matched(n_bytes_other, 0);

#ifndef JOIN_TABLE_LOCAL_DISTRIBUTED
#define JOIN_TABLE_LOCAL_DISTRIBUTED(left_table_outer, right_table_outer,      \
                                     left_table_outer_exp,                     \
                                     right_table_outer_exp)                    \
    if (left_table_outer == left_table_outer_exp &&                            \
        right_table_outer == right_table_outer_exp) {                          \
        for (int p = 0; p < n_pes; p++) {                                      \
            std::shared_ptr<table_info> bcast_table_chunk =                    \
                broadcast_table(bcast_table, bcast_table, nullptr,             \
                                bcast_table->ncols(), parallel_trace, p);      \
            bool is_bcast_outer = (is_left_outer && left_table_bcast) ||       \
                                  (is_right_outer && !left_table_bcast);       \
            size_t n_bytes_bcast =                                             \
                is_bcast_outer ? (bcast_table_chunk->nrows() + 7) >> 3 : 0;    \
            bcast_row_is_matched.resize(n_bytes_bcast);                        \
            std::ranges::fill(bcast_row_is_matched, 0);                        \
                                                                               \
            bodo::vector<int64_t> left_idxs;                                   \
            bodo::vector<int64_t> right_idxs;                                  \
            if (left_table_bcast) {                                            \
                nested_loop_join_table_local<left_table_outer_exp,             \
                                             right_table_outer_exp>(           \
                    bcast_table_chunk, other_table, cond_func, parallel_trace, \
                    left_idxs, right_idxs, bcast_row_is_matched,               \
                    other_row_is_matched);                                     \
            } else {                                                           \
                nested_loop_join_table_local<left_table_outer_exp,             \
                                             right_table_outer_exp>(           \
                    other_table, bcast_table_chunk, cond_func, parallel_trace, \
                    left_idxs, right_idxs, other_row_is_matched,               \
                    bcast_row_is_matched);                                     \
            }                                                                  \
                                                                               \
            /** handle bcast table's unmatched rows if outer **/               \
            if (is_left_outer && left_table_bcast) {                           \
                add_unmatched_rows(bcast_row_is_matched,                       \
                                   bcast_table_chunk->nrows(), left_idxs,      \
                                   right_idxs, true);                          \
            }                                                                  \
            if (is_right_outer && !left_table_bcast) {                         \
                add_unmatched_rows(bcast_row_is_matched,                       \
                                   bcast_table_chunk->nrows(), right_idxs,     \
                                   left_idxs, true);                           \
            }                                                                  \
            std::shared_ptr<table_info> left_in_chunk = bcast_table_chunk;     \
            std::shared_ptr<table_info> right_in_chunk = other_table;          \
            if (!left_table_bcast) {                                           \
                left_in_chunk = other_table;                                   \
                right_in_chunk = bcast_table_chunk;                            \
            }                                                                  \
            std::shared_ptr<table_info> out_table_chunk = create_out_table(    \
                left_in_chunk, right_in_chunk, left_idxs, right_idxs,          \
                key_in_output, use_nullable_arr_type, cond_func_left_columns,  \
                cond_func_left_column_len, cond_func_right_columns,            \
                cond_func_right_column_len);                                   \
            out_table_chunks.emplace_back(out_table_chunk);                    \
        }                                                                      \
    }
#endif
            JOIN_TABLE_LOCAL_DISTRIBUTED(is_left_outer, is_right_outer, true,
                                         true);
            JOIN_TABLE_LOCAL_DISTRIBUTED(is_left_outer, is_right_outer, true,
                                         false);
            JOIN_TABLE_LOCAL_DISTRIBUTED(is_left_outer, is_right_outer, false,
                                         true);
            JOIN_TABLE_LOCAL_DISTRIBUTED(is_left_outer, is_right_outer, false,
                                         false);
#undef JOIN_TABLE_LOCAL_DISTRIBUTED

            // handle non-bcast table's unmatched rows of if outer
            if (is_left_outer && !left_table_bcast) {
                bodo::vector<int64_t> left_idxs;
                bodo::vector<int64_t> right_idxs;
                add_unmatched_rows(other_row_is_matched, left_table->nrows(),
                                   left_idxs, right_idxs, false);
                std::shared_ptr<table_info> out_table_chunk = create_out_table(
                    std::move(left_table), std::move(right_table), left_idxs,
                    right_idxs, key_in_output, use_nullable_arr_type,
                    cond_func_left_columns, cond_func_left_column_len,
                    cond_func_right_columns, cond_func_right_column_len);
                out_table_chunks.emplace_back(out_table_chunk);
            } else if (is_right_outer && left_table_bcast) {
                bodo::vector<int64_t> left_idxs;
                bodo::vector<int64_t> right_idxs;
                add_unmatched_rows(other_row_is_matched, right_table->nrows(),
                                   right_idxs, left_idxs, false);
                std::shared_ptr<table_info> out_table_chunk = create_out_table(
                    std::move(left_table), std::move(right_table), left_idxs,
                    right_idxs, key_in_output, use_nullable_arr_type,
                    cond_func_left_columns, cond_func_left_column_len,
                    cond_func_right_columns, cond_func_right_column_len);
                out_table_chunks.emplace_back(out_table_chunk);
            }

            out_table = concat_tables(out_table_chunks);
            out_table_chunks.clear();
        }
        // If either table is already replicated then broadcasting
        // isn't necessary (output's distribution will match the other input as
        // intended)
        else {
            bodo::vector<int64_t> left_idxs;
            bodo::vector<int64_t> right_idxs;

            size_t n_bytes_left =
                is_left_outer ? (left_table->nrows() + 7) >> 3 : 0;
            size_t n_bytes_right =
                is_right_outer ? (right_table->nrows() + 7) >> 3 : 0;
            bodo::vector<uint8_t> left_row_is_matched(n_bytes_left, 0);
            bodo::vector<uint8_t> right_row_is_matched(n_bytes_right, 0);

#ifndef JOIN_TABLE_LOCAL_REPLICATED
#define JOIN_TABLE_LOCAL_REPLICATED(left_table_outer, right_table_outer,   \
                                    left_table_outer_exp,                  \
                                    right_table_outer_exp)                 \
    if (left_table_outer == left_table_outer_exp &&                        \
        right_table_outer == right_table_outer_exp) {                      \
        nested_loop_join_table_local<left_table_outer_exp,                 \
                                     right_table_outer_exp>(               \
            left_table, right_table, cond_func, parallel_trace, left_idxs, \
            right_idxs, left_row_is_matched, right_row_is_matched);        \
    }
#endif
            JOIN_TABLE_LOCAL_REPLICATED(is_left_outer, is_right_outer, true,
                                        true);
            JOIN_TABLE_LOCAL_REPLICATED(is_left_outer, is_right_outer, true,
                                        false);
            JOIN_TABLE_LOCAL_REPLICATED(is_left_outer, is_right_outer, false,
                                        true);
            JOIN_TABLE_LOCAL_REPLICATED(is_left_outer, is_right_outer, false,
                                        false);
#undef JOIN_TABLE_LOCAL_REPLICATED

            // handle unmatched rows of if outer
            if (is_left_outer) {
                bool needs_reduction = !left_parallel && right_parallel;
                add_unmatched_rows(left_row_is_matched, left_table->nrows(),
                                   left_idxs, right_idxs, needs_reduction);
            }
            if (is_right_outer) {
                bool needs_reduction = !right_parallel && left_parallel;
                add_unmatched_rows(right_row_is_matched, right_table->nrows(),
                                   right_idxs, left_idxs, needs_reduction);
            }

            out_table = create_out_table(
                std::move(left_table), std::move(right_table), left_idxs,
                right_idxs, key_in_output, use_nullable_arr_type,
                cond_func_left_columns, cond_func_left_column_len,
                cond_func_right_columns, cond_func_right_column_len);
        }
        // Check for skew if BodoSQL suggested we should
        if (rebalance_if_skewed && (left_parallel || right_parallel)) {
            out_table = rebalance_join_output(std::move(out_table));
        }

        // NOTE: no need to delete table pointers since done in generated Python
        // code in join.py

        // number of local output rows is passed to Python in case all output
        // columns are dead.
        *num_rows_ptr = out_table->nrows();
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Create output table for join given the row indices.
 * Removes dead condition function columns from the output.
 *
 * @param left_table left input table
 * @param right_table right input table
 * @param left_idxs indices of left table rows in output
 * @param right_idxs indices of right table rows in output
 * @param key_in_output : a vector of booleans specifying if cond
 * func columns are included in the output table. The booleans first contain
 * all cond columns of the left table and then the right table.
 * @param use_nullable_arr_type : a vector specifying whether a column's type
 * needs to be changed to nullable or not. Only applicable to Numpy
 * integer/float columns currently.
 * @param cond_func_left_columns: Array of column numbers in the left table
 * used by cond_func.
 * @param cond_func_left_column_len: Length of cond_func_left_columns.
 * @param cond_func_right_columns: Array of column numbers in the right table
 * used by cond_func.
 * @param cond_func_right_column_len: Length of cond_func_right_columns.
 * @return std::shared_ptr<table_info> output table of join
 */
std::shared_ptr<table_info> create_out_table(
    std::shared_ptr<table_info> left_table,
    std::shared_ptr<table_info> right_table, bodo::vector<int64_t>& left_idxs,
    bodo::vector<int64_t>& right_idxs, bool* key_in_output,
    int64_t* use_nullable_arr_type, uint64_t* cond_func_left_columns,
    uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    uint64_t cond_func_right_column_len) {
    // Create sets for cond func columns. These columns act
    // like key columns and are contained inside of key_in_output,
    // so we need an efficient lookup.
    bodo::unord_set_container<int64_t> left_cond_func_cols_set;
    left_cond_func_cols_set.reserve(cond_func_left_column_len);

    bodo::unord_set_container<int64_t> right_cond_func_cols_set;
    right_cond_func_cols_set.reserve(cond_func_right_column_len);

    for (size_t i = 0; i < cond_func_left_column_len; i++) {
        uint64_t col_num = cond_func_left_columns[i];
        left_cond_func_cols_set.insert(col_num);
    }
    for (size_t i = 0; i < cond_func_right_column_len; i++) {
        uint64_t col_num = cond_func_right_columns[i];
        right_cond_func_cols_set.insert(col_num);
    }

    // create output columns
    std::vector<std::shared_ptr<array_info>> out_arrs;
    int idx = 0;
    offset_t key_in_output_idx = 0;
    // add left columns to output
    for (size_t i = 0; i < left_table->ncols(); i++) {
        std::shared_ptr<array_info> in_arr = left_table->columns[i];

        // cond columns may be dead
        if (!left_cond_func_cols_set.contains(i) ||
            key_in_output[key_in_output_idx++]) {
            bool use_nullable_arr = use_nullable_arr_type[idx];
            out_arrs.emplace_back(RetrieveArray_SingleColumn(in_arr, left_idxs,
                                                             use_nullable_arr));
            idx++;
        }
        // Release reference (and potentially memory) early if possible.
        reset_col_if_last_table_ref(left_table, i);
    }
    left_table.reset();

    // XXX Drop left indices sooner (using shrink_to_fit) or wrap in shared_ptr?

    // add right columns to output
    for (size_t i = 0; i < right_table->ncols(); i++) {
        std::shared_ptr<array_info> in_arr = right_table->columns[i];

        // cond columns may be dead
        if (!right_cond_func_cols_set.contains(i) ||
            key_in_output[key_in_output_idx++]) {
            bool use_nullable_arr = use_nullable_arr_type[idx];
            out_arrs.emplace_back(RetrieveArray_SingleColumn(in_arr, right_idxs,
                                                             use_nullable_arr));
            idx++;
        }
        // Release reference (and potentially memory) early if possible.
        reset_col_if_last_table_ref(right_table, i);
    }
    right_table.reset();

    return std::make_shared<table_info>(out_arrs);
}
