#pragma once

#include "_join.h"
#include "_nested_loop_join_impl.h"

/**
 * @brief nested loop join two tables locally with a simple nested loop join.
 * steals references (decrefs) both inputs since it calls RetrieveTable.
 *
 * @param left_table left input table
 * @param right_table right input table
 * @param is_left_outer : whether we do an inner or outer merge on the left.
 * @param is_right_outer : whether we do an inner or outer merge on the right.
 * @param cond_func function generated in Python to evaluate general join
 * conditions. It takes data pointers for left/right tables and row indices.
 * @param parallel_trace parallel flag to pass to tracing calls
 * @param left_idxs row indices of left table to fill for creating output table
 * @param right_idxs row indices of right table to fill for creating output
 * table
 * @param left_row_is_matched bitmap of matched left table rows to fill (left
 * join only)
 * @param right_row_is_matched bitmap of matched right table rows to fill (right
 * join only)
 * @param right_offset the number of bits already used from the start of the
 * right_row_is_matched. Default is 0
 */
template <bool is_left_outer, bool is_right_outer, typename Allocator>
void nested_loop_join_table_local(
    std::shared_ptr<table_info> left_table,
    std::shared_ptr<table_info> right_table, cond_expr_fn_batch_t cond_func,
    bool parallel_trace, bodo::vector<int64_t>& left_idxs,
    bodo::vector<int64_t>& right_idxs,
    bodo::vector<uint8_t>& left_row_is_matched,
    bodo::vector<uint8_t, Allocator>& right_row_is_matched,
    int64_t right_offset = 0) {
    tracing::Event ev("nested_loop_join_table_local", parallel_trace);
    size_t n_rows_left = left_table->nrows();
    size_t n_rows_right = right_table->nrows();

    // if either table is empty, return early
    // this avoids a division by zero in the block_n_rows calculation
    if (n_rows_left == 0 || n_rows_right == 0) {
        return;
    }

    auto [left_table_infos, col_ptrs_left, null_bitmap_left] =
        get_gen_cond_data_ptrs(left_table);
    auto [right_table_infos, col_ptrs_right, null_bitmap_right] =
        get_gen_cond_data_ptrs(right_table);

    // set 500K block size to make sure block data of all cores fits in L3 cache
    int64_t block_size_bytes = DEFAULT_BLOCK_SIZE_BYTES;
    char* block_size = std::getenv("BODO_CROSS_JOIN_BLOCK_SIZE");
    if (block_size) {
        block_size_bytes = std::stoi(block_size);
    }
    if (block_size_bytes < 0) {
        throw std::runtime_error(
            "nested_loop_join_table_local: block_size_bytes < 0");
    }

    int64_t n_left_blocks = (int64_t)std::ceil(
        table_local_memory_size(left_table, false) / (double)block_size_bytes);
    int64_t n_right_blocks = (int64_t)std::ceil(
        table_local_memory_size(right_table, false) / (double)block_size_bytes);

    int64_t left_block_n_rows =
        (int64_t)std::ceil(n_rows_left / (double)n_left_blocks);
    int64_t right_block_n_rows =
        (int64_t)std::ceil(n_rows_right / (double)n_right_blocks);

    // bitmap for output of condition function for each block
    uint8_t* match_arr = nullptr;
    if (cond_func != nullptr) {
        int64_t n_bytes_match =
            ((left_block_n_rows * right_block_n_rows) + 7) >> 3;
        match_arr = new uint8_t[n_bytes_match];
    }
    for (int64_t b_right = 0; b_right < n_right_blocks; b_right++) {
        for (int64_t b_left = 0; b_left < n_left_blocks; b_left++) {
            int64_t left_block_start = b_left * left_block_n_rows;
            int64_t right_block_start = b_right * right_block_n_rows;
            int64_t left_block_end = std::min(
                left_block_start + left_block_n_rows, (int64_t)n_rows_left);
            int64_t right_block_end = std::min(
                right_block_start + right_block_n_rows, (int64_t)n_rows_right);
            // call condition function on the input block which sets match
            // results in match_arr bitmap
            if (cond_func != nullptr) {
                cond_func(left_table_infos.data(), right_table_infos.data(),
                          col_ptrs_left.data(), col_ptrs_right.data(),
                          null_bitmap_left.data(), null_bitmap_right.data(),
                          match_arr, left_block_start, left_block_end,
                          right_block_start, right_block_end);
            }

            int64_t match_ind = 0;
            for (int64_t j = right_block_start; j < right_block_end; j++) {
                for (int64_t i = left_block_start; i < left_block_end; i++) {
                    bool match = (match_arr == nullptr) ||
                                 GetBit(match_arr, match_ind++);
                    if (match) {
                        left_idxs.emplace_back(i);
                        right_idxs.emplace_back(j);
                        if (is_left_outer) {
                            SetBitTo(left_row_is_matched.data(), i, true);
                        }
                        if (is_right_outer) {
                            SetBitTo(right_row_is_matched.data(),
                                     j + right_offset, true);
                        }
                    }
                }
            }
        }
    }

    if (match_arr != nullptr) {
        delete[] match_arr;
    }
};
