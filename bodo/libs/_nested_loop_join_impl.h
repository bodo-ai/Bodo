#pragma once
#include "_bodo_common.h"
#include "_join.h"
#include "_memory.h"

/**
 * @brief Create data structures for column data to match the format
 * expected by cond_func. We create three vectors:
 * the array_infos (which handle general types), and data1/nullbitmap pointers
 * as a fast path for accessing numeric data. These include both keys
 * and data columns as either can be used in the cond_func.
 *
 * @param table Input table
 * @return std::tuple<std::vector<array_info*>, std::vector<void*>,
 * std::vector<void*>> Vectors of array info, data1, and null bitmap pointers
 */
std::tuple<std::vector<array_info*>, std::vector<void*>, std::vector<void*>>
get_gen_cond_data_ptrs(std::shared_ptr<table_info> table);

/**
 * @brief Populate existing data structures with column data to match the
 * format expected by cond_func. We append pointers to three vectors:
 * array_infos (which handle general types), and data1/nullbitmap pointers
 * as a fast path for accessing numeric data. These include both keys
 * and data columns as either can be used in the cond_func.
 *
 * @param table Input table
 * @param table_infos Pointer to output vector of array infos
 * @param col_ptrs Pointer to output vector of data1 pointers
 * @param null_bitmaps Pointer to output vector of null_bitmap pointers
 */
void get_gen_cond_data_ptrs(std::shared_ptr<table_info> table,
                            std::vector<array_info*>* array_infos,
                            std::vector<void*>* col_ptrs,
                            std::vector<void*>* null_bitmaps);

std::shared_ptr<table_info> create_out_table(
    std::shared_ptr<table_info> left_table,
    std::shared_ptr<table_info> right_table, bodo::vector<int64_t>& left_idxs,
    bodo::vector<int64_t>& right_idxs, bool* key_in_output,
    int64_t* use_nullable_arr_type, uint64_t* cond_func_left_columns,
    uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    uint64_t cond_func_right_column_len);

/**
 * @brief Find unmatched outer join rows (using reduction over bit map if
 * necessary) and add them to list of output row indices.
 *
 * @param bit_map bitmap of matched rows
 * @param n_rows number of rows in input table
 * @param table_idxs indices in input table used for output generation
 * @param other_table_idxs indices in the other table used for output generation
 * @param needs_reduction : whether the bitmap needs a reduction (the
 * corresponding table is replicated, but the other table is distributed).
 * @param offset number of bits from the start of bit_map that belongs to
 * previous chunks. Default is 0
 */
void add_unmatched_rows(std::span<uint8_t> bit_map, size_t n_rows,
                        bodo::vector<int64_t>& table_idxs,
                        bodo::vector<int64_t>& other_table_idxs,
                        bool needs_reduction, int64_t offset = 0);
