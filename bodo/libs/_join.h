#ifndef _JOIN_H_INCLUDED
#define _JOIN_H_INCLUDED

#include "_bodo_common.h"

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
 * @param is_na_equal: When doing a merge, are NA values considered equal.
 * @return the returned table used in the code.
 */
table_info* hash_join_table(table_info* left_table, table_info* right_table,
                            bool left_parallel, bool right_parallel,
                            int64_t n_key_t, int64_t n_data_left_t,
                            int64_t n_data_right_t, int64_t* vect_same_key,
                            int64_t* vect_need_typechange, bool is_left,
                            bool is_right, bool is_join, bool optional_col,
                            bool indicator, bool is_na_equal);
#endif  // _JOIN_H_INCLUDED
