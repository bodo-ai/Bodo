// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _SHUFFLE_H_INCLUDED
#define _SHUFFLE_H_INCLUDED

#include <mpi.h>
#include "_bodo_common.h"

#if 0

#define PRIME 109345121  // prime number larger than n_pes
// random integers from [0, PRIME-1], with SCALE > 0
#define SCALE 30457      // 1 + random.randrange(PRIME - 1)
#define SHIFT 84577466   // random.randrange(PRIME)

static inline int hash_to_rank(uint32_t hash, int n_pes) {
    return (((size_t)hash * SCALE + SHIFT) % PRIME) % (size_t)n_pes;
}

#undef PRIME
#undef SCALE
#undef SHIFT

#else

static inline int hash_to_rank(uint32_t hash, int n_pes) {
    return (size_t)hash % (size_t)n_pes;
}

#endif

/**
 * @brief shuffle information used to reverse shuffle later
 *
 */
struct shuffle_info {
    std::shared_ptr<mpi_comm_info> comm_info;
    std::shared_ptr<uint32_t[]> hashes;
    explicit shuffle_info(std::shared_ptr<mpi_comm_info> _comm_info,
                          std::shared_ptr<uint32_t[]> _hashes)
        : comm_info(_comm_info), hashes(_hashes) {}
};

/** Shuffle a table so that same keys are on the same process.
 *  Note: Steals a reference from the input table.
 *
 * @param in_table : the input table.
 * @param n_keys   : the number of keys for comparison.
 * @param is_parallel: true because this function is called with distributed
 * data only.
 * @param keep_comm_info : specifies if shuffle information should be kept in
 * output table, to be used for reverse shuffle later (e.g. in groupby apply).
 * @param hashes : provide precalculated hashes
 * @return the new table after shuffling
 */
std::shared_ptr<table_info> shuffle_table(
    std::shared_ptr<table_info> in_table, int64_t n_keys,
    bool is_parallel = true, int32_t keep_comm_info = 0,
    std::shared_ptr<uint32_t[]> hashes = std::shared_ptr<uint32_t[]>(nullptr));

table_info* shuffle_table_py_entrypt(table_info* in_table, int64_t n_keys,
                                     bool is_parallel = true,
                                     int32_t keep_comm_info = 0);

/**
 * @brief get shuffle info from table struct
 * Using raw pointers since called from Python.
 *
 * @param table input table
 * @return shuffle_info* shuffle info of input table
 */
shuffle_info* get_shuffle_info(table_info* table);

/**
 * @brief free allocated data of shuffle info
 * Called from Python.
 *
 * @param sh_info input shuffle info
 */
void delete_shuffle_info(shuffle_info* sh_info);

// Note: Steals a reference from the input table.
/**
 * @brief reverse a previous shuffle of input table
 * Using raw pointers since called from Python.
 *
 * @param in_table input table
 * @param sh_info shuffle info
 * @return table_info* reverse shuffled output table
 */
table_info* reverse_shuffle_table(table_info* in_table, shuffle_info* sh_info);

/** Shuffling a table from all nodes to all the other nodes.
 *  obtained by hashes. For different tables, hashes have to be coherent
 *
 * @param in_table : the input table.
 * @param ref_table : the other table with which we need coherent hashes
 * @param n_keys : the number of keys for comparison.
 * @param hashes : provide precomputed hashes, otherwise this function
 * will compute the hashes
 * @param filter : filter to discard rows from shuffle. If no filter is
 * provided then no filtering will happen.
 * @param null_bitmask : In case of a SQL join where nulls are not considered
 equal, a null bitmask can be provided that indicates whether any of the keys
 columns in the table are null (and hence cannot match with any other row).
 * @param keep_nulls_and_filter_misses : In case a Bloom filter is provided and
 * a key is not present in the bloom filter, should we keep the value on this
 * rank (i.e. not discard it altogether). Similarly, in cases where a
 * null-bitmask is provided, this parameter determines what should be done with
 * rows with nulls in keys. If set to true, we keep these rows on this same
 * rank, and if set to false, we drop these rows altogether. This is useful in
 * the outer join cases.
 *
 * @return the new table after the shuffling-
 */
std::shared_ptr<table_info> coherent_shuffle_table(
    std::shared_ptr<table_info> in_table, std::shared_ptr<table_info> ref_table,
    int64_t n_keys,
    std::shared_ptr<uint32_t[]> hashes = std::shared_ptr<uint32_t[]>(nullptr),
    SimdBlockFilterFixed<::hashing::SimpleMixSplit>* filter = nullptr,
    const uint8_t* null_bitmask = nullptr,
    const bool keep_nulls_and_filter_misses = true);

/** Shuffling a table from all nodes to all the other nodes.
 *
 * @param in_table     : the input table.
 * @param hashes       : the array containing the values to be
 *                       i_node = hash_to_rank(i_row, n_pes)
 *                       for the nodes of rank not equal to 0.
 * @param comm_info    : the array for the communication.
 * @param is_parallel: Used to indicate whether tracing should be parallel or
 * not
 * @return the new table after the shuffling.
 */
std::shared_ptr<table_info> shuffle_table_kernel(
    std::shared_ptr<table_info> in_table, std::shared_ptr<uint32_t[]>& hashes,
    mpi_comm_info const& comm_info, bool is_parallel = true);

/** Reverse shuffling a table from all nodes to all the other nodes.
 *
 * @param in_table  : the input table.
 * @param hashes    : the hash of the reversed table
 * @param comm_info : the array for the communication.
 * @return the new table after the shuffling-
 */
std::shared_ptr<table_info> reverse_shuffle_table_kernel(
    std::shared_ptr<table_info> in_table, std::shared_ptr<uint32_t[]>& hashes,
    mpi_comm_info const& comm_info);

/** Broadcasting a table.
 * The table in the nodes of rank 0 is broadcast to all nodes.
 *
 * @param ref_table : the reference table used for the types
 * @param in_table  : the input table. It can be a NULL
 *                    for the nodes of rank not equal to 0 since it is not
 *                    read for those nodes.
 * @param n_cols    : the number of columns of the keys.
 * @param is_parallel: Used to indicate whether tracing should be parallel or
 * not
 * @param mpi_root: root rank for broadcast (where data is broadcast from)
 * @return the table equal to in_table but available on all the nodes.
 */
std::shared_ptr<table_info> broadcast_table(
    std::shared_ptr<table_info> ref_table, std::shared_ptr<table_info> in_table,
    size_t n_cols, bool is_parallel, int mpi_root);

/** Gather a table.
 *
 * @param in_table     : the input table.
 * @param n_cols       : the number of columns of the keys.
 *     If -1 then all columns are used. Otherwise, the first n_cols_i columns
 * are gather.
 * @param all_gather   : Whether to do all_gather or not.
 * @param is_parallel: Used to indicate whether tracing should be parallel or
 * not
 * @return the table obtained by concatenating the tables
 *         on the node 0.
 */
std::shared_ptr<table_info> gather_table(std::shared_ptr<table_info> in_table,
                                         int64_t n_cols_i, bool all_gather,
                                         bool is_parallel);

/** Compute whether we need to do a reshuffling or not for performance reasons.
    The dilemma is following:
    ---If the workload is not well partitioned then the code becomes serialized
   and slower.
    ---Reshuffling itself is an expensive operation.

    The heuristic idea is following:
    ---What slows down the running is if one or 2 processors have a much higher
   load than other because it serializes the computation.
    ---If 1 or 2 processors have little load then that is not so bad. It just
   decreases the number of effective processors used.
    ---Thus the metric to consider or not a reshuffling is
        (max nb_row) / (avg nb_row)
    ---If the value is larger than 2 then reshuffling is interesting

    @param in_table : the input partitioned table
    @param crit_fraction : the critical fraction
    @return the boolean saying whether or not we need to do reshuffling.
*/
bool need_reshuffling(std::shared_ptr<table_info> in_table,
                      double crit_fraction);

/* Apply a renormalization shuffling
   After the operation, all nodes will have a standard size.

   @param in_table : the input partitioned table
   @param random : if random > 0 also do a random shuffle of the data
   @param random_seed : seed to use for random shuffling if random=2
   @param parallel : whether data is distributed or not. This is a nop if
   parallel=false
   @return the reshuffled table
 */
std::shared_ptr<table_info> shuffle_renormalization(
    std::shared_ptr<table_info> in_table, int random, int64_t random_seed,
    bool parallel);

table_info* shuffle_renormalization_py_entrypt(table_info* in_table, int random,
                                               int64_t random_seed,
                                               bool parallel);

/* Apply a renormalization shuffling getting data from all the ranks
   and sending to only a given subset of ranks.
   After the operation, all specified destination ranks will have
   same or similar data size.

   @param in_table : the input partitioned table
   @param random : if random > 0 also do a random shuffle of the data
   @param random_seed : seed to use for random shuffling if random=2
   @param parallel : whether data is distributed or not. This is a nop if
   parallel=false
   @param n_dest_ranks: number of destination ranks
   @param dest_ranks: array of destination ranks
   @return the reshuffled table
 */
std::shared_ptr<table_info> shuffle_renormalization_group(
    std::shared_ptr<table_info> in_table, int random, int64_t random_seed,
    bool parallel, int64_t n_dest_ranks, int* dest_ranks);

table_info* shuffle_renormalization_group_py_entrypt(
    table_info* in_table, int random, int64_t random_seed, bool parallel,
    int64_t n_dest_ranks, int* dest_ranks);

/* This function is used for the reverse shuffling of numpy data.
 *
 * It takes the rows after the MPI_alltoall and put them at their right position
 */
template <class T>
inline void fill_recv_data_inner(T* recv_buff, T* data,
                                 std::shared_ptr<uint32_t[]>& hashes,
                                 std::vector<int64_t> const& send_disp,
                                 int n_pes, size_t n_rows) {
    std::vector<int64_t> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = static_cast<size_t>(hash_to_rank(hashes[i], n_pes));
        int64_t ind = tmp_offset[node];
        data[i] = recv_buff[ind];
        tmp_offset[node]++;
    }
}

/**
 * @brief Update dictionary encoded array to drop any duplicates in its
 * local copy of the dictionary. If the dictionary is already global then
 * this maintains the global dictionary because the operations are
 * deterministic.
 *
 * @param dict_array The dictionary array whose dictionary needs updating.
 * @param sort_dictionary_if_modified Should the dictionary be sorted if we
 * need to gather the data? Note: The output should not assume the data is
 * sorted.
 */
void drop_duplicates_local_dictionary(std::shared_ptr<array_info> dict_array,
                                      bool sort_dictionary_if_modified = false);

/**
 * @brief Python wrapper for drop_duplicates_local_dictionary
 *
 * @param dict_array The dictionary array whose dictionary needs updating.
 * @param sort_dictionary_if_modified Should the dictionary be sorted if we
 * need to gather the data? Note: The output should not assume the data is
 * sorted.
 * @return updated dictionary array (same as input since updated inplace, just a
 * new reference for Python)
 */
array_info* drop_duplicates_local_dictionary_py_entry(
    array_info* dict_array, bool sort_dictionary_if_modified = false);

/**
 * @brief Update a dictionary encoded array to gather all dictionary values onto
 * each rank. If the dictionary is updated then we also drop duplicates and may
 * sort the dictionary.
 *
 * @param dict_array The dictionary array whose dictionary needs updating.
 * @param is_parallel Is the input distributed? If so we must gather the
 * dictionary from all ranks. If not we just mark the dictionary as global.
 * @param sort_dictionary_if_modified Should the dictionary be sorted if we
 * modify the dictionary? Note: The output should not assume the data is sorted.
 */
void convert_local_dictionary_to_global(
    std::shared_ptr<array_info> dict_array, bool is_parallel,
    bool sort_dictionary_if_modified = false);

/**
 * @brief Update a dictionary encoded array to gather all dictionary values onto
 * each rank and then drop any duplicates. If the dictionary is updated then we
 * may optionally sort the dictionary.
 *
 * @param dict_array
 * @param is_parallel
 * @param sort_dictionary_if_modified
 */
void make_dictionary_global_and_unique(
    std::shared_ptr<array_info> dict_array, bool is_parallel,
    bool sort_dictionary_if_modified = false);

/**
 * @brief Perform a reverse shuffle for data1 of either a
 * Numpy, Categorical, or Nullable array.
 *
 * @param[in] in_arr The input array to reverse-shuffle.
 * @param[out] out_arr The output array to fill.
 * @param[in] hashes The hashes to each rank of the reversed array.
 * @param[in] comm_info The communication information from the original shuffle.
 */
void reverse_shuffle_preallocated_data_array(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> out_arr,
    std::shared_ptr<uint32_t[]>& hashes, mpi_comm_info const& comm_info);

/**
 * @brief Determine if we should shuffle this iteration.
 * This is true if we are on the last iteration, every sync_iters
 * the max size of every ranks shuffle buffer is checked and if it is
 * greater than shuffle_at_nrows then we shuffle.
 * @param parallel : is the table to be shuffled parallel
 * @param is_last : is this the last iteration
 * @param shuffle_table: the table to test for shuffling
 * @param iter : the current iteration
 * @param sync_iter : the number of iterations between syncs
 */
bool shuffle_this_iter(const bool parallel, const bool is_last,
                       const std::shared_ptr<table_info>& shuffle_table,
                       const uint64_t iter, const uint64_t sync_iter);

#endif  // _SHUFFLE_H_INCLUDED
