#include "_bodo_common.h"

/**
 * Compute the boolean array on output corresponds to the "isin" function in
 * matlab. each group, writes the result to a new output table containing one
 * row per group.
 *
 * @param out_arr the boolean array on output.
 * @param in_arr the list of values on input
 * @param in_values the list of values that we need to check with
 * @param is_parallel, whether the computation is parallel or not.
 */
void array_isin(array_info* out_arr, array_info* in_arr, array_info* in_values,
                bool is_parallel);

/**
 * Implementation of the sort_values functionality in C++
 * Notes:
 * - We depend on the timsort code from https://github.com/timsort/cpp-TimSort
 *   which provides stable sort taking care of already sorted parts.
 * - We use lambda for the call functions.
 * - The KeyComparisonAsPython is used for having the comparison as in Python.
 *
 * @param input table
 * @param number of key columns in the table used for the comparison
 * @param ascending, whether to sort ascending or not
 * @param na_position, true corresponds to last, false to first
   @param parallel, true in case of parallel computation, false otherwise.
 */
table_info* sort_values_table(table_info* in_table, int64_t n_key_t,
                              int64_t* vect_ascending, bool na_position,
                              bool parallel);

/** This function is the function for the dropping of duplicated rows.
 * This C++ code should provide following functionality of pandas
 * drop_duplicates:
 * ---possibility of selecting columns for the identification
 * ---possibility of keeping first, last or removing all entries with duplicate
 * inplace operation for keeping the data in the same place is another problem.
 *
 * @param in_table : the input table
 * @param is_parallel: the boolean specifying if the computation is parallel or
 * not.
 * @param num_keys: the number of keys used for the computation
 * @param keep: integer specifying the expected behavior.
 *        keep = 0 corresponds to the case of keep="first" keep first entry
 *        keep = 1 corresponds to the case of keep="last" keep last entry
 *        keep = 2 corresponds to the case of keep=False : remove all duplicates
 * @return the vector of pointers to be used.
 */
table_info* drop_duplicates_table(table_info* in_table, bool is_parallel,
                                  int64_t num_keys, int64_t keep);
