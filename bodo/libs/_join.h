#ifndef _JOIN_H_INCLUDED
#define _JOIN_H_INCLUDED

#include "_bodo_common.h"

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

/** This function does the joining of the table and returns the joined
 * table
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
 * @param n_key_t : the number of columns of keys on input
 * @param n_data_left_t : the number of columns of data on the left
 * @param n_data_right_t : the number of columns of data on the right
 * @param vect_same_key : a vector of integers specifying if a key has the same
 * name on the left and on the right.
 * @param key_in_output : a vector of booleans specifying if the key and cond
 * func columns are included in the output table. The booleans first contain
 * all of the left table (keys then cond columns in order of the table) and
 * then the right table. Shared output keys are only stored for the left table.
 * @param vect_need_typechange : a vector specifying whether a column needs to
 * be changed or not. This usage is due to the need to support categorical
 * array.
 * @param is_left : whether we do an inner or outer merge on the left.
 * @param is_right : whether we do an inner or outer merge on the right.
 * @param is_join : whether the call is a join in Pandas or not (as opposed to
 * merge).
 * @param optional_col : When doing a merge on column and index, the key
 *    is put also in output, so we need one additional column in that case.
 * @param indicator: When doing a merge, if indicator=True outputs an additional
 *    Categorical column with name _merge that says if the data source is from
 * left_only, right_only, or both.
 * @param is_na_equal: When doing a merge, are NA values considered equal?
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

 * @return the returned table used in the code.
 */
table_info* hash_join_table(
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, int64_t n_key_t, int64_t n_data_left_t,
    int64_t n_data_right_t, int64_t* vect_same_key, bool* key_in_output,
    int64_t* vect_need_typechange, bool is_left, bool is_right, bool is_join,
    bool optional_col, bool indicator, bool is_na_equal,
    cond_expr_fn_t cond_func, int64_t* cond_func_left_columns,
    int64_t cond_func_left_column_len, int64_t* cond_func_right_columns,
    int64_t cond_func_right_column_len);
#endif  // _JOIN_H_INCLUDED
