#pragma once

#include "_bodo_common.h"

// Forward Declaration from _stream_dict_encoding
class DictEncodingState;

/**
 * Compute the boolean array on output corresponds to the "isin" function in
 * matlab. each group, writes the result to a new output table containing one
 * row per group.
 * Using raw pointers since called from Python.
 *
 * @param out_arr the boolean array on output.
 * @param in_arr the list of values on input
 * @param in_values the list of values that we need to check with
 * @param is_parallel, whether the computation is parallel or not.
 */
void array_isin_py_entry(array_info* out_arr, array_info* in_arr,
                         array_info* in_values, bool is_parallel);

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
 * @param na_position, true corresponds to last, false to first. 1 value per
 column
 * @param dead_keys, array with 0/1 for each key indicating dead keys (1 is
 dead)
 * @param out_n_rows, single-element array to store the number of output rows,
 can be nullptr if not needed.
 * @param bounds, single-array table that provides parallel chunk boundaries for
 data redistribution during parallel sort (optional). Currently only used for
 Iceberg MERGE INTO.
 * @param parallel, true in case of parallel computation, false otherwise.
 */
std::shared_ptr<table_info> sort_values_table(
    std::shared_ptr<table_info> in_table, int64_t n_key_t,
    int64_t* vect_ascending, int64_t* na_position, int64_t* dead_keys,
    int64_t* out_n_rows, std::shared_ptr<table_info> bounds, bool parallel);

/**
 * Python entry point and wrapper for sort_values_table()
 * Converts table/array raw pointers to smart pointers, and
 * sets Python error in case of C++ exception.
 *
 * @param in_table input table
 * @param number of key columns in the table used for the comparison
 * @param ascending, whether to sort ascending or not
 * @param na_position, true corresponds to last, false to first. 1 value per
 column
 * @param dead_keys, array with 0/1 for each key indicating dead keys (1 is
 dead)
 * @param out_n_rows, single-element array to store the number of output rows,
 can be nullptr if not needed.
 * @param bounds, single-array table that provides parallel chunk boundaries for
 data redistribution during parallel sort (optional). Currently only used for
 Iceberg MERGE INTO.
 * @param parallel, true in case of parallel computation, false otherwise.
 */
table_info* sort_values_table_py_entry(table_info* in_table, int64_t n_key_t,
                                       int64_t* vect_ascending,
                                       int64_t* na_position, int64_t* dead_keys,
                                       int64_t* out_n_rows, table_info* bounds,
                                       bool parallel);

/**
 * @brief Sort local values within a continguous range from start_offset to
 * n_rows and return the indices from the input tables that would produce the
 * sorted data.
 *
 * For example, if we called this function sorting on the following column:
 *   ["c", "d", "b", "a"]
 * The output would be:
 *   [3, 2, 0, 1]
 * The actual sorted table can be obtained by calling RetriveTable, or by using
 * the sort_values_table_local function which internally does just that.
 *
 * @param in_table input table to sort
 * @param n_key_t number of columns to be considered as part of the sort key
 * @param vect_ascending whether a column should be sorted as ascending or
 * descending
 * @param na_position controls where nulls should be sent per column (beginning
 * or end)
 * @param is_parallel
 * @param start_offset row to start sorting from
 * @param n_rows number of rows to sort
 * @param pool
 * @param mm
 */
bodo::vector<int64_t> sort_values_table_local_get_indices(
    std::shared_ptr<table_info> in_table, int64_t n_key_t,
    const int64_t* vect_ascending, const int64_t* na_position, bool is_parallel,
    size_t start_offset, size_t n_rows,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * See sort_values_table_local_get_indices for parameter descriptions. Calls
 * RetriveTable on the sorted indices to return a sorted table. This will
 * effectively do a full copy of the input.
 */
std::shared_ptr<table_info> sort_values_table_local(
    std::shared_ptr<table_info> in_table, int64_t n_key_t,
    const int64_t* vect_ascending, const int64_t* na_position,
    const int64_t* dead_keys, bool is_parallel,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * Helper function to sort contents of array_info.
 * Note that the provided array_info is deleted and a new one
 * is returned.
 * This is implemented through a wrapper around sort_values_table_local
 *
 * @param in_arr array_info to sort
 * @param is_parallel true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 * @param ascending, whether to sort ascending or not
 * @param na_position, true corresponds to last, false to first.
 * @return std::shared_ptr<array_info> sorted array_info
 */
std::shared_ptr<array_info> sort_values_array_local(
    std::shared_ptr<array_info> in_arr, bool is_parallel, int64_t ascending,
    int64_t na_position);

/**
 * @brief Python entrypoint for sort_table_for_interval_join.
 *
 * When !is_table_point_side, the output table can have rows duplicated across
 * consecutive ranks.
 *
 * @param table Table to sort.
 * @param bounds_arr Bounds to use for distributing the data. Should have length
 * (#ranks - 1).
 * @param is_table_point_side Whether this is a point side table. If false, it's
 * an interval side table where the first two columns are the start and end
 * columns respectively.
 * @param parallel Whether the table is distributed.
 * @return table_info* Sorted table (after redistribution).
 */
table_info* sort_table_for_interval_join_py_entrypoint(table_info* table,
                                                       array_info* bounds_arr,
                                                       bool is_table_point_side,
                                                       bool parallel);

/**
 * @brief Sort the tables involved in a point in interval join.
 * We assume that the first column in the point table is the point column and
 * the first two columns in the interval table are the interval start and end
 * columns, respectively.
 * It's a convenience wrapper around sort_both_tables_for_interval_join.
 *
 * NOTE: All arrays in table_point and table_interval will
 * be decref-ed, same as sort-values.
 *
 * @param table_point Point side table
 * @param table_interval Interval side table
 * @param table_point_parallel Is the point side table distributed
 * @param table_interval_parallel Is the interval side table distributed
 * @param strict Only filter strict bad intervals (where A > B instead of A >=
 * B)
 * @return std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>>
 * sorted point table, sorted interval table.
 */
std::pair<std::shared_ptr<table_info>, std::shared_ptr<table_info>>
sort_tables_for_point_in_interval_join(
    std::shared_ptr<table_info> table_point,
    std::shared_ptr<table_info> table_interval, bool table_point_parallel,
    bool table_interval_parallel, bool strict);

/**
 * @brief Sort the tables involved in an interval overlap join.
 * We assume that the first two columns in both tables are the start and end of
 * the intervals respectively.
 * It's a convenience wrapper around sort_both_tables_for_interval_join.
 *
 * NOTE: All arrays in table_1 and table_2 will be
 * decref-ed, same as sort-values.
 *
 * @param table_1 First interval table.
 * @param table_2 Second interval table.
 * @param table_1_parallel True when the first interval table is distributed.
 * @param table_2_parallel True when the second interval table is distributed.
 * @return std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>,
 * std::shared_ptr<array_info>> Sorted first table, sorted second table and
 * bounds array.
 */
std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>,
           std::shared_ptr<array_info>>
sort_tables_for_interval_overlap_join(std::shared_ptr<table_info> table_1,
                                      std::shared_ptr<table_info> table_2,
                                      bool table_1_parallel,
                                      bool table_2_parallel);

/** This function is the function for the dropping of duplicated rows.
 * This C++ code should provide following functionality of pandas
 * drop_duplicates:
 * ---possibility of selecting columns for the identification
 * ---possibility of keeping first, last or removing all entries with
 * duplicate inplace operation for keeping the data in the same place is
 * another problem.
 *
 * @param in_table : the input table
 * @param is_parallel: the boolean specifying if the computation is parallel
 * or not.
 * @param num_keys: number of columns to use identifying duplicates
 * @param keep: integer specifying the expected behavior.
 *        keep = 0 corresponds to the case of keep="first" keep first entry
 *        keep = 1 corresponds to the case of keep="last" keep last entry
 *        keep = 2 corresponds to the case of keep=False : remove all
 * duplicates
 * @param dropna: Should NA be included in the final table
 * @param drop_local_first: Whether to drop duplicates in local data before
 * shuffling
 * @return the unicized table
 */
std::shared_ptr<table_info> drop_duplicates_table(
    std::shared_ptr<table_info> in_table, bool is_parallel, int64_t num_keys,
    int64_t keep, bool dropna = false, bool drop_local_first = true,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

bodo::vector<int64_t> drop_duplicates_table_helper(
    std::shared_ptr<table_info> in_table, int64_t num_keys, int64_t keep,
    int step, bool is_parallel, bool dropna, bool drop_duplicates_dict,
    std::shared_ptr<uint32_t[]> hashes,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::shared_ptr<table_info> drop_duplicates_table_inner(
    std::shared_ptr<table_info> in_table, int64_t num_keys, int64_t keep,
    int step, bool is_parallel, bool dropna, bool drop_duplicates_dict,
    std::shared_ptr<uint32_t[]> hashes = std::shared_ptr<uint32_t[]>(nullptr),
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

table_info* drop_duplicates_table_py_entry(table_info* in_table,
                                           bool is_parallel, int64_t num_keys,
                                           int64_t keep, bool dropna = false,
                                           bool drop_local_first = true);

/**
 * @brief Performs a SQL UNION operation on the input tables.
 * This operations concatenates all of the input tables and optionally removes
 * any duplicate rows.
 * Uses raw pointers since called from Python.
 *
 * @param in_table Array of input tables each with the same schema.
 * @param num_tables The number of tables in the input array.
 * @param drop_duplicates Should duplicate rows be removed? This is used for the
 * SQL UNION DISTINCT operation.
 * @param is_parallel Are the tables distributed or replicated? If one table is
 * replicated all will be replicated.
 * @return table_info* The output table.
 */
table_info* union_tables(table_info** in_table, int64_t num_tables,
                         bool drop_duplicates, bool is_parallel);

/**
 * @brief Python entry point for concat_tables
 * @param in_table Array of input tables each with the same schema.
 * @param num_tables The number of tables in the input array.
 * @return table_info* The output table.
 */
table_info* concat_tables_py_entry(table_info** in_table, int64_t num_tables);

/** This function is the function for the dropping of duplicated keys:
 * ---only the keys are returned
 * ---non-null entries are removed from the output.
 *
 * @param in_table     : the input table
 * @param num_keys     : the number of keys used for the computation
 * @param is_parallel  : the boolean specifying if the computation is parallel
 * @param dropna       : whether we drop null keys or not.
 * or not.
 * @return the table in output
 */
std::shared_ptr<table_info> drop_duplicates_keys(
    std::shared_ptr<table_info> in_table, int64_t num_keys, bool is_parallel,
    bool dropna = true);

/**
 * @brief C++ implementation of df.sample. This is implemented using rejection
 * sampling for small sample sizes, and whole-array permutation for large
 * sample sizes. In the parallel case, each rank samples from local data only.
 * @param table_info: the input table
 * @param n_samp: the number of rows in output
 * @param frac: the fraction of rows selected
 * @param replace: whether we allow replaced entries or not
 * @param random_state: seed for random number generator
 * @param parallel: if true the array is distributed, if false it is replicated
 * @return the sampled entries in the table (same distribution as input table)
 */
table_info* sample_table_py_entry(table_info* in_table, int64_t n, double frac,
                                  bool replace, int64_t random_state,
                                  bool parallel);

void get_search_regex(std::shared_ptr<array_info> in_arr,
                      const bool case_sensitive, const bool match_beginning,
                      char const* const pat,
                      std::shared_ptr<array_info> out_arr);

// Python wrapper for get_search_regex
void get_search_regex_py_entry(array_info* in_arr, const bool case_sensitive,
                               const bool match_beginning,
                               char const* const pat, array_info* out_arr);

/**
 * @brief C++ implementation of re.sub to replace each element in in_arr with
 * the replacement string based on the regex pattern. This is supported for both
 * dictionary encoded array and regular string arrays. This is implemented via
 * boost::xpressive::regex_replace
 *
 * @param in_arr The input array of string needed replacement.
 * @param pat A utf-8 encoded regex pattern.
 * @param replacement A utf-8 encoded replacement string to insert based on the
 * pattern.
 * @return array_info* A output array with the replaced values.
 * This is either dictionary encoded or regular string array depending on the
 * input array.
 */
array_info* get_replace_regex_py_entry(array_info* p_in_arr,
                                       char const* const pat,
                                       char const* replacement);

/**
 * @brief Equivalent to get_replace_regex_py_entry, but contains extra
 * information to enable caching dictionary arrays for use in streaming.
 *
 * @param in_arr The input array of string needed replacement. This must be a
 * dictionary encoded array.
 * @param pat A utf-8 encoded regex pattern.
 * @param replacement A utf-8 encoded replacement string to insert based on the
 * pattern.
 * @param state Dictionary encoding state used for caching the output.
 * @param func_id The id used as the key for the function.
 * @return array_info* A output array with the replaced values.
 * This is either dictionary encoded or regular string array depending on the
 * input array.
 */
array_info* get_replace_regex_dict_state_py_entry(array_info* p_in_arr,
                                                  char const* const pat,
                                                  char const* replacement,
                                                  DictEncodingState* state,
                                                  int64_t func_id);

//
// Sampling Utilities
//

/**
 * @brief Get the number of samples based on the size of the local table.
 *
 * @param n_pes Number of MPI ranks.
 * @param n_total Total number of rows in the table.
 * @param n_local Number of rows in the local table.
 */
int64_t get_num_samples_from_local_table(int n_pes, int64_t n_total,
                                         int64_t n_local);

/**
 * @brief Get a vector of random samples from a locally sorted table.
 *
 * @param n_local Number of rows in the local table.
 * @param n_loc_sample Number of samples to get.
 */
bodo::vector<int64_t> get_sample_selection_vector(int64_t n_local,
                                                  int64_t n_loc_sample);

/**
 * @brief Compute bounds for the ranks based on the collected samples.
 * All samples are assumed to be on rank 0. Tables on the rest of the
 * ranks are assumed to be empty.
 * The samples are first sorted, and then the bounds are computed
 * by picking the elements at the appropriate location for the rank.
 *
 * @param all_samples Table with all samples (gathered on rank 0). It is assumed
 * to be unsorted.
 * @param ref_table Reference table to use for the broadcast step. This is
 * mainly needed for dict encoded string arrays. In those cases, it is important
 * for the dictionary in this reference table to be same as the dictionary of
 * the actual array.
 * @param n_key_t Number of key columns.
 * @param vect_ascending Vector of booleans (one for each key column) describing
 * whether to sort in ascending order on the key columns.
 * @param na_position Vector of booleans (one for each key column) describing
 * where to put the NAs (last or first) in the key columns.
 * @param myrank MPI rank of the calling process.
 * @param n_pes Total number of MPI ranks.
 * @param parallel Is the process parallel.
 * @return std::shared_ptr<table_info> Bounds table with n_pes-1 rows. A full
 * bounds table is computed on rank 0, broadcasted to all ranks and returned.
 */
std::shared_ptr<table_info> compute_bounds_from_samples(
    std::shared_ptr<table_info> all_samples,
    std::shared_ptr<table_info> ref_table, int64_t n_key_t,
    int64_t* vect_ascending, int64_t* na_position, int myrank, int n_pes,
    bool parallel);
