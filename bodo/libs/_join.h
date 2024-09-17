#pragma once

#include "_bodo_common.h"

// Default block size to 500K to make sure block data fits in L3 cache
const int64_t DEFAULT_BLOCK_SIZE_BYTES = 500 * 1024;

/**
 * @brief Function type for general join condition functions generated in
 * Python. Takes data pointers for left and right tables and row indices, and
 * evaluates the condition. NOTE: Only works for numerical columns and does not
 * handle NAs yet.
 * @param left_table array of array_info pointers for all columns of left table
 * @param right_table array of array_info pointers for all columns of right
 * table
 * @param left_data1 array of data1 pointers for the left table. This is used as
 *      a fast path for numeric columns.
 * @param right_data1  array of data1 pointers for the right table. This is used
 * as a fast path for numeric columns.
 * @param left_null_bitmap array of null bitmaps for the left table.
 * @param right_null_bitmap array of null bitmaps for the right table.
 * @param l_ind index in left table
 * @param r_ind index in right table
 */
typedef bool (*cond_expr_fn_t)(array_info** left_table,
                               array_info** right_table, void** left_data1,
                               void** right_data1, void** left_null_bitmap,
                               void** right_null_bitmap, int64_t l_ind,
                               int64_t r_ind);

/**
 * @brief Same as previous function type, but processes data in batches and sets
 * bits in a bitmap.
 *
 * @param left_table array of array_info pointers for all columns of left table
 * @param right_table array of array_info pointers for all columns of right
 * table
 * @param left_data1 array of data1 pointers for the left table. This is used as
 *      a fast path for numeric columns.
 * @param right_data1  array of data1 pointers for the right table. This is used
 * as a fast path for numeric columns.
 * @param left_null_bitmap array of null bitmaps for the left table.
 * @param right_null_bitmap array of null bitmaps for the right table.
 * @param match_arr array of null bitmaps to set output
 * @param left_block_start start index for left table loop
 * @param left_block_end end index for left table loop
 * @param right_block_start start index for right table loop
 * @param right_block_end end index for right table loop
 */
typedef void (*cond_expr_fn_batch_t)(
    array_info** left_table, array_info** right_table, void** left_data1,
    void** right_data1, void** left_null_bitmap, void** right_null_bitmap,
    uint8_t* match_arr, int64_t left_block_start, int64_t left_block_end,
    int64_t right_block_start, int64_t right_block_end);

/** This function does the joining of the table and returns the joined
 * table
 * Using raw pointers since called from Python.
 *
 * This implementation follows the Shared partition procedure.
 * The data is partitioned and shuffled with the _gen_par_shuffle.
 *
 * The first stage is the partitioning of the data by using hashes array
 * and unordered map array.
 *
 * Afterwards, secondary partitioning is done if the hashes match.
 * Then the pairs of left/right origins are created for subsequent
 * work. If a left key has no matching on the right, then value -1
 * is put (thus the std::ptrdiff_t type is used).
 *
 *
 * We need to merge all of the arrays in input because we cannot
 * have empty arrays.
 *
 * is_left and is_right correspond
 *   "inner" : is_left = T, is_right = T
 *   "outer" : is_left = F, is_right = F
 *   "left"  : is_left = T, is_right = F
 *   "right" : is_left = F, is_right = T
 *
 * @param left_table : the left table
 * @param right_table : the right table
 * @param left_parallel : whether the left table is parallel or not
 * @param right_parallel : whether the right table is parallel or not
 * @param n_keys : the number of columns of keys on input
 * @param n_data_left_t : the number of columns of data on the left
 * @param n_data_right_t : the number of columns of data on the right
 * @param vect_same_key : a vector of integers specifying if a key has the same
 * name on the left and on the right.
 * @param key_in_output : a vector of booleans specifying if the key and cond
 * func columns are included in the output table. The booleans first contain
 * all of the left table (keys then cond columns in order of the table) and
 * then the right table. Shared output keys are only stored for the left table.
 * @param use_nullable_arr_type : a vector specifying whether a column needs to
 * be changed or not. This usage is due to the need to support categorical
 * array.
 * @param is_left_outer : whether we do an inner or outer merge on the left.
 * @param is_right_outer : whether we do an inner or outer merge on the right.
 * @param is_join : whether the call is a join in Pandas or not (as opposed to
 * merge).
 * @param extra_data_col : When doing a merge on column and index, the key
 *    is put also in output, so we need one additional column in that case.
 * @param indicator: When doing a merge, if indicator=True outputs an additional
 *    Categorical column with name _merge that says if the data source is from
 * left_only, right_only, or both.
 * @param is_na_equal: When doing a merge, are NA values considered equal?
 * @param rebalance_if_skewed: After the merge should we check the data for
 *   skew and rebalance if needed?
 * @param cond_func function generated in Python to evaluate general join
 * conditions. It takes data pointers for left/right tables and row indices.
 * @param cond_func_left_columns: Array of column numbers in the left table
 *     used by cond_func and not found in keys. This is used for assigning
 groups.
 * @param cond_func_left_column_len: Length of cond_func_left_columns.
 * @param cond_func_right_columns: Array of column numbers in the right table
 *     used by cond_func and not found in keys. This is used for assigning
 groups.
 * @param cond_func_right_column_len: Length of cond_func_right_columns.
 * @param num_rows_ptr: Pointer used to store the number of rows in the
        output to return to Python. This enables marking all columns as
        dead.

 * @return the returned table used in the code.
 */
table_info* hash_join_table(
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, int64_t n_keys, int64_t n_data_left_t,
    int64_t n_data_right_t, int64_t* vect_same_key, bool* key_in_output,
    int64_t* use_nullable_arr_type, bool is_left_outer, bool is_right_outer,
    bool is_join, bool extra_data_col, bool indicator, bool is_na_equal,
    bool rebalance_if_skewed, cond_expr_fn_t cond_func,
    uint64_t* cond_func_left_columns, uint64_t cond_func_left_column_len,
    uint64_t* cond_func_right_columns, uint64_t cond_func_right_column_len,
    uint64_t* num_rows_ptr);

/**
 * @brief nested loop join two tables (parallel if any input is parallel)
 * Using raw pointers since called from Python.
 *
 * @param left_table left input table
 * @param right_table right input table
 * @param left_parallel whether the left table is parallel or not
 * @param right_parallel whether the right table is parallel or not
 * @param is_left : whether we do an inner or outer merge on the left.
 * @param is_right : whether we do an inner or outer merge on the right.
 * @param key_in_output : a vector of booleans specifying if cond
 * func columns are included in the output table. The booleans first contain
 * all cond columns of the left table and then the right table.
 * @param vect_need_typechange : a vector specifying whether a column's type
 * needs to be changed to nullable or not. Only application to Numpy
 * integer/float columns currently.
 * @param rebalance_if_skewed: After the merge should we check the data for
 *   skew and rebalance if needed?
 * @param cond_func function generated in Python to evaluate general join
 * conditions. It takes data pointers for left/right tables and row indices.
 * @param cond_func_left_columns: Array of column numbers in the left table
 * used by cond_func.
 * @param cond_func_left_column_len: Length of cond_func_left_columns.
 * @param cond_func_right_columns: Array of column numbers in the right table
 * used by cond_func.
 * @param cond_func_right_column_len: Length of cond_func_right_columns.
 * @param num_rows_ptr Pointer used to store the number of rows in the
        output to return to Python. This enables marking all columns as
        dead.
 * @return table_info* nested loop join output table
 */
table_info* nested_loop_join_table(
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, bool is_left, bool is_right, bool* key_in_output,
    int64_t* vect_need_typechange, bool rebalance_if_skewed,
    cond_expr_fn_batch_t cond_func, uint64_t* cond_func_left_columns,
    uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    uint64_t cond_func_right_column_len, uint64_t* num_rows_ptr);

/**
 * @brief Point-in-interval or interval-overlap join of two tables (parallel if
 * any input is parallel).
 * Design doc: https://bodo.atlassian.net/l/cp/1JCnntP1
 *
 * Using raw pointers since called from Python.
 *
 * @param left_table left input table
 * @param right_table right input table
 * @param left_parallel whether the left table is parallel or not
 * @param right_parallel whether the right table is parallel or not
 * @param is_left whether we do an inner or outer merge on the left. Can only be
 * outer in case of point-in-interval join where the point side is on the left.
 * @param is_right whether we do an inner or outer merge on the right. Can only
 * be outer in case of point-in-interval join where the point side is on the
 * right.
 * @param is_left_point Is the point side on the left side. Only applicable if
 * point-in-interval join.
 * @param strict_start Does the point need to be strictly right of the interval
 * start
 * @param strict_end Does the point need to strictly left of the interval end
 * @param point_col_id Column id of the point column. Only applicable if
 * point-in-interval join.
 * @param interval_start_col_id Column id of the interval start column. Only
 * applicable if point-in-interval join.
 * @param interval_end_col_id Column id of the interval end column. Only
 * applicable if point-in-interval join.
 * @param key_in_output a vector of booleans specifying if cond
 * func columns are included in the output table. The booleans first contain
 * all cond columns of the left table and then the right table.
 * @param use_nullable_arr_type a vector specifying whether a column's type
 * needs to be changed to nullable or not. Only application to Numpy integer
 * columns currently.
 * @param rebalance_if_skewed: After the merge should we check the data for
 *   skew and rebalance if needed?
 * @param num_rows_ptr Pointer used to store the number of rows in the
 * output to return to Python. This enables marking all columns as dead.
 * @return table_info* interval join output table
 */
table_info* interval_join_table(
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, bool is_left, bool is_right, bool is_left_point,
    bool strict_start, bool strict_end, uint64_t point_col_id,
    uint64_t interval_start_col_id, uint64_t interval_end_col_id,
    bool* key_in_output, int64_t* use_nullable_arr_type,
    bool rebalance_if_skewed, uint64_t* num_rows_ptr);

// Helper function declarations
void nested_loop_join_handle_dict_encoded(
    std::shared_ptr<table_info> left_table,
    std::shared_ptr<table_info> right_table, bool left_parallel,
    bool right_parallel);
int get_bcast_join_threshold();

std::shared_ptr<table_info> rebalance_join_output(
    std::shared_ptr<table_info> original_output);

template <bool is_left_outer, bool is_right_outer, bool non_equi_condition,
          typename Allocator>
void nested_loop_join_table_local(
    std::shared_ptr<table_info> left_table,
    std::shared_ptr<table_info> right_table, cond_expr_fn_batch_t cond_func,
    bool parallel_trace, bodo::vector<int64_t>& left_idxs,
    bodo::vector<int64_t>& right_idxs,
    bodo::vector<uint8_t>& left_row_is_matched,
    bodo::vector<uint8_t, Allocator>& right_row_is_matched,
    int64_t right_offset = 0);

template <typename BitMapAllocator>
void add_unmatched_rows(bodo::vector<uint8_t, BitMapAllocator>& bit_map,
                        size_t n_rows, bodo::vector<int64_t>& table_idxs,
                        bodo::vector<int64_t>& other_table_idxs,
                        bool needs_reduction, int64_t offset = 0);

std::shared_ptr<table_info> create_out_table(
    std::shared_ptr<table_info> left_table,
    std::shared_ptr<table_info> right_table, bodo::vector<int64_t>& left_idxs,
    bodo::vector<int64_t>& right_idxs, bool* key_in_output,
    int64_t* use_nullable_arr_type, uint64_t* cond_func_left_columns,
    uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    uint64_t cond_func_right_column_len);

/**
 * @brief Create data structures for column data to match the format
 * expected by cond_func. We create three vectors:
 * the array_infos (which handle general types), and data1/nullbitmap pointers
 * as a fast path for accessing numeric data. These include both keys
 * and data columns as either can be used in the cond_func.
 * Defined in _nested_loop_join.cpp
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
 * Defined in _nested_loop_join.cpp
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
