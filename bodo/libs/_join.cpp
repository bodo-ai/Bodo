// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_join.h"
#include "_array_hash.h"
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_join_hashing.h"
#include "_shuffle.h"

/**
 * @brief Validate the input to the equi_join_table function.
 *
 * @param left_table The left input table.
 * @param right_table The right input table.
 * @param n_key The number of key columns.
 * @param extra_data_col Is there an extra data column generated to handle an
 * extra key?
 */
void validate_equi_join_input(table_info* left_table, table_info* right_table,
                              size_t n_key, bool extra_data_col) {
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        // Check that all of the key pairs have matching types.
        CheckEqualityArrayType(left_table->columns[iKey],
                               right_table->columns[iKey]);
    }
    // in the case of merging on index and one column, it can only be one
    // column
    if (n_key > 1 && extra_data_col) {
        throw std::runtime_error(
            "Error in join.cpp::hash_join_table: if extra_data_col=true "
            "then "
            "we must have n_key=1.");
    }
}

/**
 * @brief handle dict-encoded columns of input tables to keys
 * of equijoin. Dictionaries need to be global and unified since
 * we use the indices to hash.
 *
 * @param left_table left input table to cross join
 * @param right_table right input table to cross join
 * @param left_parallel left table is parallel
 * @param right_parallel right table is parallel
 * @param n_keys number of keys
 */
void equi_join_keys_handle_dict_encoded(table_info* left_table,
                                        table_info* right_table,
                                        bool left_parallel, bool right_parallel,
                                        size_t n_key) {
    // Unify dictionaries of DICT key columns (required for key comparison)
    // IMPORTANT: need to do this before computing the hashes
    //
    // This implementation of hashing dictionary encoded data is based on
    // the values in the indices array. To do this, we make and enforce
    // a few assumptions
    //
    // 1. Both arrays are dictionary encoded. This is enforced in join.py
    // where determine_table_cast_map requires either both inputs to be
    // dictionary encoded or neither.
    //
    // 2. Both arrays share the exact same dictionary. This occurs in
    // unify_dictionaries.
    //
    // 3. The dictionary does not contain any duplicate values. This is
    // enforced by drop_duplicates_local_dictionary. In particular,
    // drop_duplicates_local_dictionary contains a drop duplicates step
    // that ensures all values are unique. If the dictionary is not global
    // then convert_local_dictionary_to_global will also drop duplicate and
    // drop_duplicates_local_dictionary will be a no-op.
    for (size_t i = 0; i < n_key; i++) {
        array_info* arr1 = left_table->columns[i];
        array_info* arr2 = right_table->columns[i];
        if ((arr1->arr_type == bodo_array_type::DICT) &&
            (arr2->arr_type == bodo_array_type::DICT)) {
            make_dictionary_global_and_unique(arr1, left_parallel);
            make_dictionary_global_and_unique(arr2, right_parallel);
            unify_dictionaries(arr1, arr2, left_parallel, right_parallel);
        }
    }
}

/**
 * @brief Insert column numbers that are used in the non-equality C funcs but
 * not the equality function into the set of column numbers. Also, if the column
 * is a dictionary, it makes the dictionary global.
 *
 * @param set The set of column numbers that are used in the non-equality C
 * funcs.
 * @param table The table that the column numbers are from.
 * @param non_equi_func_col_nums The indices of the columns that are used in the
 * non-equality C funcs.
 * @param len_non_equi The length of non_equi_func_col_nums.
 * @param is_parallel Is the table distributed?
 * @param n_key The number of key columns. Indices below this will not be added
 * to the set.
 */
void insert_non_equi_func_set(UNORD_SET_CONTAINER<int64_t>* set,
                              table_info* table,
                              uint64_t* non_equi_func_col_nums,
                              uint64_t len_non_equi, bool is_parallel,
                              size_t n_key) {
    // Non-keys used in non-equality functions need global + unique dictionaries
    // for consistent hashing, but no unifying is necessary.
    for (size_t i = 0; i < len_non_equi; i++) {
        uint64_t col_num = non_equi_func_col_nums[i];
        // If a column is in both the non-equality C funcs and the regular
        // equality check we don't include it in the set.
        if (col_num >= n_key) {
            array_info* arr = table->columns[col_num];
            if (arr->arr_type == bodo_array_type::DICT) {
                make_dictionary_global_and_unique(arr, is_parallel);
            }
            set->insert(col_num);
        }
    }
}

/**
 * @brief Converts the arrays of unique column numbers that are used in the
 * non-equality C funcs to sets. Also, if the column is a dictionary, it makes
 * the dictionary global.
 *
 * @param left_table The left table.
 * @param right_table The right table.
 * @param left_non_equi_func_col_nums The array of unique column numbers that
 * are used in the non-equality C funcs from the left table.
 * @param len_left_non_equi The length of left_non_equi_func_col_nums.
 * @param right_non_equi_func_col_nums The array of unique column numbers that
 * are used in the non-equality C funcs from the right table.
 * @param len_right_non_equi The length of right_non_equi_func_col_nums.
 * @param left_parallel Is the left table distributed.
 * @param right_parallel Is the right table distributed.
 * @param n_key The number of key columns. These will not be processed in the
 * non-equality functions.
 * @return std::tuple<UNORD_SET_CONTAINER<int64_t>*,
 * UNORD_SET_CONTAINER<int64_t>*> A tuple of pointers to the sets of column
 * numbers that are used in the non-equality C funcs.
 */
std::tuple<UNORD_SET_CONTAINER<int64_t>*, UNORD_SET_CONTAINER<int64_t>*>
create_non_equi_func_sets(table_info* left_table, table_info* right_table,
                          uint64_t* left_non_equi_func_col_nums,
                          uint64_t len_left_non_equi,
                          uint64_t* right_non_equi_func_col_nums,
                          uint64_t len_right_non_equi, bool left_parallel,
                          bool right_parallel, size_t n_key) {
    // Convert the left table.
    UNORD_SET_CONTAINER<int64_t>* left_non_equi_func_col_num_set =
        new UNORD_SET_CONTAINER<int64_t>();
    left_non_equi_func_col_num_set->reserve(len_left_non_equi);
    insert_non_equi_func_set(left_non_equi_func_col_num_set, left_table,
                             left_non_equi_func_col_nums, len_left_non_equi,
                             left_parallel, n_key);

    // Convert the right table.
    UNORD_SET_CONTAINER<int64_t>* right_non_equi_func_col_num_set =
        new UNORD_SET_CONTAINER<int64_t>();
    right_non_equi_func_col_num_set->reserve(len_right_non_equi);
    insert_non_equi_func_set(right_non_equi_func_col_num_set, right_table,
                             right_non_equi_func_col_nums, len_right_non_equi,
                             right_parallel, n_key);

    return std::make_tuple(left_non_equi_func_col_num_set,
                           right_non_equi_func_col_num_set);
}

/**
 * @brief Insert the rows from the build table into the hash map in the case
 * where there is a non-equality function.
 *
 * @param key_rows_map Hashmap mapping the relevant keys to the list of rows
 * with the same key.
 * @param second_level_hash_maps Hashmap mapping the relevant non-equality keys
 * to the rows with the same key.
 * @param second_level_hash_fct The hash function for the second level hashmap.
 * @param second_level_equal_fct The equality function for the second level
 * hashmap.
 * @param groups A vector of rows considered equivalent. This is used to
 * compress rows with the same columns used in the non-equality condition and
 * the same keys.
 * @param build_table_rows THe number of rows in the build table.
 */
template <typename Map>
void insert_build_table_equi_join_some_non_equality(
    Map* key_rows_map,
    std::vector<UNORD_MAP_CONTAINER<
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>*
        second_level_hash_maps,
    joinHashFcts::SecondLevelHashHashJoinTable second_level_hash_fct,
    joinHashFcts::SecondLevelKeyEqualHashJoinTable second_level_equal_fct,
    std::vector<std::vector<size_t>*>* groups, size_t build_table_rows) {
    // If 'uses_cond_func' we have a separate insertion process. We
    // place the condition before the loop to avoid overhead.
    for (size_t i_build = 0; i_build < build_table_rows; i_build++) {
        // Check if the group already exists, if it doesn't this
        // will insert a value.
        size_t& first_level_group_id = (*key_rows_map)[i_build];
        // first_level_group_id==0 means the equality condition
        // doesn't have a match
        if (first_level_group_id == 0) {
            // Update the value of first_level_group_id stored in
            // the hash map as well since its pass by reference.
            first_level_group_id = second_level_hash_maps->size() + 1;
            UNORD_MAP_CONTAINER<
                size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
                joinHashFcts::SecondLevelKeyEqualHashJoinTable>* group_map =
                new UNORD_MAP_CONTAINER<
                    size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
                    joinHashFcts::SecondLevelKeyEqualHashJoinTable>(
                    {}, second_level_hash_fct, second_level_equal_fct);
            second_level_hash_maps->emplace_back(group_map);
        }
        auto group_map = (*second_level_hash_maps)[first_level_group_id - 1];
        // Check if the group already exists, if it doesn't this
        // will insert a value.
        size_t& second_level_group_id = (*group_map)[i_build];
        if (second_level_group_id == 0) {
            // Update the value of group_id stored in the hash map
            // as well since its pass by reference.
            second_level_group_id = groups->size() + 1;
            groups->emplace_back(new std::vector<size_t>());
        }
        (*groups)[second_level_group_id - 1]->emplace_back(i_build);
    }
}

/**
 * @brief Insert the rows from the build table into the hash map in the case
 * where there is only an equality condition.
 *
 * @param key_rows_map Map from the set of keys to rows that share that key.
 * @param groups The vector of groups of equivalent rows. This is done to
 * compress rows with common keys.
 * @param build_table_rows The number of rows in the build table.
 */
template <typename Map>
void insert_build_table_equi_join_all_equality(
    Map* key_rows_map, std::vector<std::vector<size_t>*>* groups,
    size_t build_table_rows) {
    for (size_t i_build = 0; i_build < build_table_rows; i_build++) {
        // Check if the group already exists, if it doesn't this
        // will insert a value.
        size_t& group_id = (*key_rows_map)[i_build];
        // group_id==0 means key doesn't exist in map
        if (group_id == 0) {
            // Update the value of group_id stored in the hash map
            // as well since its pass by reference.
            group_id = groups->size() + 1;
            groups->emplace_back(new std::vector<size_t>());
        }
        (*groups)[group_id - 1]->emplace_back(i_build);
    }
}

/**
 * @brief Get the table size threshold (in bytes) for broadcast join
 *
 * @return int threshold value
 */
int get_bcast_join_threshold() {
    // We default to 10MB, which matches Spark, unless to user specifies a
    // threshold manually.
    int bcast_join_threshold = 10 * 1024 * 1024;  // in bytes
    char* bcast_threshold = std::getenv("BODO_BCAST_JOIN_THRESHOLD");
    if (bcast_threshold) {
        bcast_join_threshold = std::stoi(bcast_threshold);
    }
    if (bcast_join_threshold < 0) {
        throw std::runtime_error("hash_join: bcast_join_threshold < 0");
    }
    return bcast_join_threshold;
}

/**
 * @brief Handles any required shuffle steps for the left and right table to
 * generate local tables for computing the join. This code can either shuffle
 * both tables if we need to do a hash join, broadcast one table if there is a
 * small table to do a broadcast join or do nothing if there is already a
 * replicated table (and do a broadcast join without requiring the broadcast).
 * Optionally if we choose to do a hash join and at least
 * one side of the table is an inner join we may use bloom filters to reduce the
 * amount of data being shuffled.
 *
 * @param left_table The left input table.
 * @param right_table The right input table.
 * @param is_left_outer Is the join an outer join on the left side?
 * @param is_right_outer Is the join an outer join on the right side?
 * @param left_parallel Is the left table parallel?
 * @param right_parallel Is the right table parallel?
 * @param n_pes The total number of processes.
 * @param n_key The total number of key columns in the equality function.
 * @param is_na_equal When doing a merge, are NA values considered equal?
 * @return std::tuple<table_info*, table_info *, bool, bool> A tuple of the new
 * left and right tables after any shuffling and whether or not the left and
 * right tables are replicated. A table will become replicated if we broadcast
 * it.
 */
std::tuple<table_info*, table_info*, bool, bool> equi_join_shuffle(
    table_info* left_table, table_info* right_table, bool is_left_outer,
    bool is_right_outer, bool left_parallel, bool right_parallel, int64_t n_pes,
    size_t n_key, const bool is_na_equal) {
    // Create a tracing event for the shuffle.
    tracing::Event ev("equi_join_table_shuffle",
                      left_parallel || right_parallel);

    // By default the work tables are the inputs.
    table_info *work_left_table = left_table, *work_right_table = right_table;
    // Default replicated values are opposite of parallel. These
    // are updated if we broadcast a table.
    bool left_replicated = !left_parallel, right_replicated = !right_parallel;

    // If either table is replicated we can just use work_left_table and
    // work_right_table as a broadcast join.
    if (left_parallel && right_parallel) {
        // If both tables are parallel then we need to decide between
        // shuffle and broadcast join.

        // Determine the memory size of each table
        int64_t left_total_memory = table_global_memory_size(left_table);
        int64_t right_total_memory = table_global_memory_size(right_table);
        int bcast_join_threshold = get_bcast_join_threshold();
        if (ev.is_tracing()) {
            ev.add_attribute("g_left_total_memory", left_total_memory);
            ev.add_attribute("g_right_total_memory", right_total_memory);
            ev.add_attribute("g_bloom_filter_supported",
                             bloom_filter_supported());
            ev.add_attribute("bcast_join_threshold", bcast_join_threshold);
        }
        bool all_gather = true;
        // Broadcast the smaller table if its replicated size is below a
        // size limit (bcast_join_threshold) and not similar or larger than the
        // size of the local "large" table (the latter is to avoid
        // detrimental impact on parallelization/scaling, note that bloom
        // filter approach also reduces cost of shuffle and does not have
        // sequential bottleneck)
        // We should tune this doing a more extensive study, see:
        // https://bodo.atlassian.net/browse/BE-1030
        double global_build_to_local_probe_ratio_limit;
        if (bloom_filter_supported()) {
            global_build_to_local_probe_ratio_limit = 0.8;
        } else {
            // Since we can't rely on bloom filters to reduce the shuffle
            // cost we give some more headroom to use broadcast join
            global_build_to_local_probe_ratio_limit = 2.0;
        }
        if (left_total_memory < right_total_memory &&
            left_total_memory < bcast_join_threshold &&
            left_total_memory < (right_total_memory / double(n_pes) *
                                 global_build_to_local_probe_ratio_limit)) {
            // Broadcast the left table
            work_left_table = gather_table(left_table, -1, all_gather, true);
            left_replicated = true;
            // Delete the left_table as it is no longer used.
            delete_table(left_table);
        } else if (right_total_memory <= left_total_memory &&
                   right_total_memory < bcast_join_threshold &&
                   right_total_memory <
                       (left_total_memory / double(n_pes) *
                        global_build_to_local_probe_ratio_limit)) {
            // Broadcast the right table
            work_right_table = gather_table(right_table, -1, all_gather, true);
            right_replicated = true;
            // Delete the right_table as it is no longer used.
            delete_table(right_table);
        } else {
            using BloomFilter = SimdBlockFilterFixed<::hashing::SimpleMixSplit>;
            // If the smaller table is larger than the threshold
            // we do a shuffle-join. To shuffle the tables we build
            // a hash table (ensuring that comparable
            // types hash to the same values).

            // only do filters for the inner side of a join for now. Note
            // is we have a left join we will generate the bloom filter
            // potentially for the left side so we can filter the right side
            // (which is the inner join).
            BloomFilter* bloom_left = nullptr;
            BloomFilter* bloom_right = nullptr;
            std::shared_ptr<uint32_t[]> hashes_left =
                std::shared_ptr<uint32_t[]>(nullptr);
            std::shared_ptr<uint32_t[]> hashes_right =
                std::shared_ptr<uint32_t[]>(nullptr);
            uint8_t* null_bitmask_keys_left = nullptr;
            uint8_t* null_bitmask_keys_right = nullptr;

            // If NAs are not considered equal, then we can use that
            // information to identify rows with nulls in any of the key
            // columns. These rows cannot match with any other rows, so in case
            // they are on an inner side of a join, they can be dropped
            // altogether, and in case they are on an outer side of the join, we
            // can keep these rows on this same rank (i.e. avoid shuffle and
            // skew generated by sending all nulls to the same rank).
            if (!is_na_equal) {
                std::vector<array_info*> key_arrs_left(
                    left_table->columns.begin(),
                    left_table->columns.begin() + n_key);
                null_bitmask_keys_left =
                    bitwise_and_null_bitmasks(key_arrs_left, true);

                std::vector<array_info*> key_arrs_right(
                    right_table->columns.begin(),
                    right_table->columns.begin() + n_key);
                null_bitmask_keys_right =
                    bitwise_and_null_bitmasks(key_arrs_right, true);
            }

            if (bloom_filter_supported()) {
                hashes_left = coherent_hash_keys_table(
                    left_table, right_table, n_key, SEED_HASH_PARTITION, true);
                hashes_right = coherent_hash_keys_table(
                    right_table, left_table, n_key, SEED_HASH_PARTITION, true);
                const int64_t left_table_nrows = left_table->nrows();
                const int64_t right_table_nrows = right_table->nrows();
                int64_t left_table_global_nrows;
                int64_t right_table_global_nrows;
                // TODO do this in a single reduction?
                MPI_Allreduce(&left_table_nrows, &left_table_global_nrows, 1,
                              MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&right_table_nrows, &right_table_global_nrows, 1,
                              MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
                static constexpr size_t MAX_BLOOM_SIZE = 100 * 1024 * 1024;
                size_t left_cardinality = 0;
                size_t right_cardinality = 0;
                // Filter built from table A is used to filter table
                // B. If A is much larger than B we don't make a filter
                // from A because the cost of making the bloom filter will
                // probably be larger than what we save from filtering B
                // Also see https://bodo.atlassian.net/browse/BE-1321
                // regarding tuning these values.
                // Note that we create a bloom filter (if appropriate), even in
                // the outer (left/right/full) join case since the filter is
                // still useful.
                bool make_bloom_left =
                    left_table_global_nrows <= (right_table_global_nrows * 10);
                bool make_bloom_right =
                    right_table_global_nrows <= (left_table_global_nrows * 10);
                // We are going to build global bloom filters, and we need
                // to know how many unique elements we are inserting into
                // each to set a bloom filter size that has good false
                // positive probability. We only get global cardinality
                // estimate if there is a reasonable chance that the bloom
                // filter will be smaller than MAX_BLOOM_SIZE.
                // Also see https://bodo.atlassian.net/browse/BE-1321
                // regarding tuning these values.
                make_bloom_left =
                    make_bloom_left &&
                    (bloom_size_bytes(double(left_table_global_nrows) * 0.3) <
                     MAX_BLOOM_SIZE);
                make_bloom_right =
                    make_bloom_right &&
                    (bloom_size_bytes(double(right_table_global_nrows) * 0.3) <
                     MAX_BLOOM_SIZE);
                // Additionally we don't make a bloom filter if it is larger
                // than MAX_BLOOM_SIZE or much larger than the data that
                // it's supposed to filter (see previous reasoning)
                // Also see https://bodo.atlassian.net/browse/BE-1321
                // regarding tuning these values
                if (make_bloom_left) {
                    left_cardinality = std::get<1>(get_nunique_hashes_global(
                        hashes_left, left_table_nrows, true));
                    size_t bloom_left_bytes =
                        bloom_size_bytes(left_cardinality);
                    make_bloom_left = (bloom_left_bytes <= MAX_BLOOM_SIZE) &&
                                      (bloom_left_bytes / (right_total_memory /
                                                           double(n_pes)) <=
                                       2.0);
                }
                if (make_bloom_right) {
                    right_cardinality = std::get<1>(get_nunique_hashes_global(
                        hashes_right, right_table_nrows, true));
                    size_t bloom_right_bytes =
                        bloom_size_bytes(right_cardinality);
                    make_bloom_right = (bloom_right_bytes <= MAX_BLOOM_SIZE) &&
                                       (bloom_right_bytes / (left_total_memory /
                                                             double(n_pes)) <=
                                        2.0);
                }
                ev.add_attribute("g_make_bloom_left", make_bloom_left);
                ev.add_attribute("g_make_bloom_right", make_bloom_right);

                if (make_bloom_left) {
                    tracing::Event ev_bloom("make_bloom", true);
                    // A bloom filter is a set, and we need to know how
                    // many unique elements we are inserting so that the
                    // implementation chooses a buffer size that guarantees
                    // low false positive probability. We are inserting
                    // the hashes, and this is going to be a global set
                    // (based on hashes from all the ranks) so we use the
                    // estimated global cardinality of the hashes
                    bloom_left = new BloomFilter(left_cardinality);
                    bloom_left->AddAll(hashes_left, 0, left_table_nrows);
                    // Do a union of the filters from all ranks (performs a
                    // bitwise OR allreduce of the bloom filter's buffer)
                    bloom_left->union_reduction();
                    ev_bloom.add_attribute("input_len", left_table_nrows);
                    ev_bloom.add_attribute("g_size", left_cardinality);
                    ev_bloom.add_attribute(
                        "g_size_bytes",
                        static_cast<size_t>(bloom_left->SizeInBytes()));
                    ev_bloom.add_attribute("which", "left");
                }
                if (make_bloom_right) {
                    tracing::Event ev_bloom("make_bloom", true);
                    bloom_right = new BloomFilter(right_cardinality);
                    bloom_right->AddAll(hashes_right, 0, right_table_nrows);
                    bloom_right->union_reduction();
                    ev_bloom.add_attribute("input_len", right_table_nrows);
                    ev_bloom.add_attribute("g_size", right_cardinality);
                    ev_bloom.add_attribute(
                        "g_size_bytes",
                        static_cast<size_t>(bloom_right->SizeInBytes()));
                    ev_bloom.add_attribute("which", "right");
                }
            }

            // Shuffle both the left and right tables.
            // If it's a left outer join or full outer join, we want to keep
            // the misses in the left table (based on the bloom filter) on this
            // rank (i.e. not shuffle them to other ranks). If they don't appear
            // in the bloom filter (which can only give false positives and not
            // false negatives), we can be sure that they don't exist at all in
            // the right table and therefore won't match with anything.
            // Vice-versa for the right table.
            // Similarly, in case !is_na_equal, rows with nulls in any of the
            // keys cannot match with any other rows. In that case, we can drop
            // nulls entirely if a table is on an inner side of the join. If a
            // table is on an outer side of the join, we can keep these rows
            // locally and reduce inter-rank communication cost.
            // XXX In the future, it might be worth storing the rows with nulls
            // and the misses based on the bloom filter in a separate table, so
            // that we can avoid the join steps (building hashmap from one table
            // and probing from the other) on them altogether and simply append
            // the rows to the join output (with null padding for the rest of
            // the columns).
            work_left_table = coherent_shuffle_table(
                left_table, right_table, n_key, hashes_left, bloom_right,
                null_bitmask_keys_left,
                /*keep_nulls_and_filter_misses=*/is_left_outer);
            work_right_table = coherent_shuffle_table(
                right_table, left_table, n_key, hashes_right, bloom_left,
                null_bitmask_keys_right,
                /*keep_nulls_and_filter_misses=*/is_right_outer);
            // Delete left_table and right_table as they are no longer used.
            delete_table(left_table);
            delete_table(right_table);
            if (bloom_left != nullptr) {
                delete bloom_left;
            }
            if (bloom_right != nullptr) {
                delete bloom_right;
            }
            if (null_bitmask_keys_left != nullptr) {
                delete[] null_bitmask_keys_left;
            }
            if (null_bitmask_keys_right != nullptr) {
                delete[] null_bitmask_keys_right;
            }
        }
    }
    ev.finalize();
    return std::make_tuple(work_left_table, work_right_table, left_replicated,
                           right_replicated);
}

/**
 * @brief Select which table should be the "build" table, i.e. the table
 * over which we build the hash map. This function uses heuristics based
 * on table size and outer join to decide which table should be the build table.
 *
 * @param n_rows_left How many rows are in the left table locally?
 * @param n_rows_right How many rows are in the right table locally?
 * @param is_left_outer Is the left table an outer join?
 * @param is_right_outer Is the right table an outer join?
 * @param ev The event to add attributes to for tracing.
 * @return If the left table is the build table.
 */
bool select_build_table(int64_t n_rows_left, int64_t n_rows_right,
                        bool is_left_outer, bool is_right_outer,
                        tracing::Event& ev) {
    // In the computation of the join we are building some hash map on one
    // of the tables. Which one we choose influences the running time of
    // course. The parameters to weigh are:
    //
    // --- Selecting the smaller table. The smaller the table,
    // the smaller the hash map, so this is usually a good heuristic
    // --- Which tables require an outer merge. Outer merges require
    // tracking which rows found matches in the other table. This
    // complicates the data structure as we now need to track if
    // a row is used.
    //
    // To make this decision, we compare
    bool use_size_method;  // true: Populate the hash map with the table with
                           // fewer rows. false: Populate the hash map with the
                           // table that uniquely
                           //    requires an outer join (i.e.
                           //    is_left_outer/is_right_outer).
    if (is_left_outer == is_right_outer) {
        // In that case we are doing either inner or full outer merge
        // the only relevant metric is table size.
        use_size_method = true;
    } else {
        // In the case of is_left_outer <> is_right_outer we must decide if the
        // size of the tables or the outer join property are more important. To
        // do this, we set a threshhold K (CritQuotientNrows) and compare the
        // size of the table. If either table has K times more rows we will
        // always make the smaller table the table that populates the hash map,
        // regardless of if that table needs an outer join. If the ratio of rows
        // is less than K, then populate the hash map with the table that
        // doesn't require an outer join.
        //
        // TODO: Justify our threshhold of 1 table being 6x larger than
        // the other.
        double CritQuotientNrows = 6.0;
        // Compute the ratio of the table sizes on the current rank.
        // This is used when determining which table populates the
        // hash map.
        double quot1 = double(n_rows_left) / double(n_rows_right);
        double quot2 = double(n_rows_right) / double(n_rows_left);
        if (quot2 < CritQuotientNrows && quot1 < CritQuotientNrows) {
            // In that case the large table is not so large comparable to
            // the build one This means that we can use the is_left_outer /
            // is_right_outer for making the choice
            use_size_method = false;
        } else {
            // In that case one table is much larger than the other,
            // therefore the choice by the number of rows is the best here.
            use_size_method = true;
        }
    }
    // We have chosen our criteria for selecting our table to populate
    // the hash map, now we map this to a variable selecting the table.
    // true: left table
    // false: right table
    int build_table_is_left;
    if (use_size_method) {
        // We choose by the number of rows.
        build_table_is_left = n_rows_left < n_rows_right;
    } else {
        // When is_left_outer <> is_right_outer
        // and the tables are similarly sized
        // we take the table without the outer join.
        // to avoid tracking if a row has a match in the hash table.
        build_table_is_left = is_right_outer;
    }
    ev.add_attribute("use_size_method", use_size_method);
    ev.add_attribute("build_table_is_left", build_table_is_left);
    return build_table_is_left;
}

/**
 * @brief Template used to handle the case where a build table hit is found.
 * If we have an outer join we need to update that the row found a match.
 *
 * @tparam build_table_outer Is the join an outer join?
 */
template <bool build_table_outer>
struct handle_build_table_hit;

/**
 * @brief Specialization of handle_build_table_hit for the case where
 * we have an outer join.
 *
 */
template <>
struct handle_build_table_hit<true> {
    /**
     * @brief Mark the current group as having found a match in the join.
     *
     * @param V_build_map: Bitmap for the build table with an outer join.
     * @param pos Current group number for the build table.
     */
    static inline void apply(std::vector<uint8_t>& V_build_map, size_t pos) {
        SetBitTo(V_build_map.data(), pos, true);
    }
};

/**
 * @brief Specialization of handle_build_table_hit for the case where we
 * have an inner join.
 *
 */
template <>
struct handle_build_table_hit<false> {
    /**
     * @brief This function does nothing.
     *
     * @param V_build_map: Bitmap for the build table with an outer join.
     * @param pos Current group number for the build table.
     */
    static inline void apply(std::vector<uint8_t>& V_build_map, size_t pos) {
        // This function does nothing.
    }
};

/**
 * @brief Template used to handle the case where a probe table miss is found.
 * Different behavior is selected depending on if we have an outer join and
 * also if we did a broadcast join.
 *
 * @tparam probe_table_outer Is the join an outer join?
 * @tparam is_outer_broadcast Is the outer join a broadcast join?
 */
template <bool probe_table_outer, bool is_outer_broadcast>
struct handle_probe_table_miss;

/**
 * @brief Specialization of handle_probe_table_miss for the case where we
 * do not have an outer join.
 *
 */
template <>
struct handle_probe_table_miss<false, false> {
    /**
     * @brief This function does nothing because we do not have an outer join.
     *
     * @param build_write_idxs The indices of the build table for an output.
     * This is a parameter we will modify.
     * @param probe_write_idxs The indices of the probe table for an output.
     * This is a parameter we will modify.
     * @param V_probe_map The bitmap holding if we found a match for the rows
     * in the probe table.
     * @param pos The current row number.
     */
    static inline void apply(std::vector<int64_t>& build_write_idxs,
                             std::vector<int64_t>& probe_write_idxs,
                             std::vector<uint8_t>& V_probe_map, size_t pos) {
        // This function does nothing.
    }
};

/**
 * @brief Specialization of handle_probe_table_miss for the case where we
 * have an outer join but not a broadcast join.
 *
 */
template <>
struct handle_probe_table_miss<true, false> {
    /**
     * @brief Mark the current row as not having found a match in the join
     * by indices in the build and probe table outputs.
     *
     * @param build_write_idxs The indices of the build table for an output.
     * This is a parameter we will modify.
     * @param probe_write_idxs The indices of the probe table for an output.
     * This is a parameter we will modify.
     * @param V_probe_map The bitmap holding if we found a match for the rows
     * in the probe table.
     * @param pos The current row number.
     */
    static inline void apply(std::vector<int64_t>& build_write_idxs,
                             std::vector<int64_t>& probe_write_idxs,
                             std::vector<uint8_t>& V_probe_map, size_t pos) {
        build_write_idxs.emplace_back(-1);
        probe_write_idxs.emplace_back(pos);
    }
};

/**
 * @brief Specialization of handle_probe_table_miss for the case where we
 * have an outer join and a broadcast join.
 *
 */
template <>
struct handle_probe_table_miss<true, true> {
    /**
     * @brief Mark the current row as not having found a match in the join
     * by updating the bitmap.
     *
     * @param build_write_idxs The indices of the build table for an output.
     * @param probe_write_idxs The indices of the probe table for an output.
     * @param V_probe_map The bitmap holding if we found a match for the rows
     * in the probe table. This is the parameter we will modify.
     * @param pos The current row number.
     */
    static inline void apply(std::vector<int64_t>& build_write_idxs,
                             std::vector<int64_t>& probe_write_idxs,
                             std::vector<uint8_t>& V_probe_map, size_t pos) {
        SetBitTo(V_probe_map.data(), pos, false);
    }
};

/**
 * @brief Insert the probe table rows into the hash maps for
 * joins that have additional non-equality conditions.
 *
 * @tparam build_table_outer Is the build table output an outer join?
 * @tparam probe_table_outer Is the probe table output an outer join?
 * @tparam is_outer_broadcast Is the probe table an outer join and a broadcast
 * join?
 * @param[in] key_rows_map The hash map that maps keys to a hashmap of common
 * non-key columns used in the non-equality function.
 * @param[in] second_level_hash_maps The hash map that groups rows with the same
 * keys and non-key columns in the non-equality function.
 * @param[in] groups Vector of row groups that have the same keys in the build
 * table.
 * @param[in] build_table_rows The number of rows in the build table.
 * @param[in] probe_table_rows The number of rows in the probe table.
 * @param[in] V_build_map The bitmap that indicates which rows in the build
 * table found matches.
 * @param[in] V_probe_map The bitmap that indicates which rows in the probe
 * table found matches.
 * @param[out] build_write_idxs The vector of rows for generating the output for
 * the build table.
 * @param[out] probe_write_idxs The vector of rows for generating the output for
 * the probe table.
 * @param[in] build_is_left Is the build table the left table in the join?
 * @param[in] col_ptrs_left The column pointers for the left table.
 * @param[in] col_ptrs_right The column pointers for the right table.
 * @param[in] null_bitmap_left The null bitmaps for the left table.
 * @param[in] null_bitmap_right The null bitmaps for the right table.
 * @param[in] cond_func The non-equality function checked for correctness.
 */
template <bool build_table_outer, bool probe_table_outer,
          bool is_outer_broadcast, typename Map>
void insert_probe_table_equi_join_some_non_equality(
    Map* key_rows_map,
    std::vector<UNORD_MAP_CONTAINER<
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>*
        second_level_hash_maps,
    std::vector<std::vector<size_t>*>* groups, size_t build_table_rows,
    size_t probe_table_rows, std::vector<uint8_t>& V_build_map,
    std::vector<uint8_t>& V_probe_map, std::vector<int64_t>& build_write_idxs,
    std::vector<int64_t>& probe_write_idxs, bool build_is_left,
    std::vector<array_info*>& left_table_infos,
    std::vector<array_info*>& right_table_infos,
    std::vector<void*>& col_ptrs_left, std::vector<void*>& col_ptrs_right,
    std::vector<void*>& null_bitmap_left, std::vector<void*>& null_bitmap_right,
    cond_expr_fn_t cond_func) {
    for (size_t i_probe = 0; i_probe < probe_table_rows; i_probe++) {
        size_t i_probe_shift = i_probe + build_table_rows;
        auto iter = key_rows_map->find(i_probe_shift);
        if (iter == key_rows_map->end()) {
            handle_probe_table_miss<probe_table_outer,
                                    is_outer_broadcast>::apply(build_write_idxs,
                                                               probe_write_idxs,
                                                               V_probe_map,
                                                               i_probe);
        } else {
            // If the first level matches, check each second level
            // hash.
            UNORD_MAP_CONTAINER<
                size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
                joinHashFcts::SecondLevelKeyEqualHashJoinTable>* group_map =
                (*second_level_hash_maps)[iter->second - 1];

            bool has_match = false;
            // Iterate over all of the keys and compare each group.
            // TODO [BE-1300]: Explore tsl:sparse_map
            for (auto& item : *group_map) {
                size_t pos = item.second - 1;
                std::vector<size_t>* group = (*groups)[pos];
                // Select a single member
                size_t cmp_row = (*group)[0];
                size_t left_ind = 0;
                size_t right_ind = 0;
                if (build_is_left) {
                    left_ind = cmp_row;
                    right_ind = i_probe;
                } else {
                    left_ind = i_probe;
                    right_ind = cmp_row;
                }
                bool match =
                    cond_func(left_table_infos.data(), right_table_infos.data(),
                              col_ptrs_left.data(), col_ptrs_right.data(),
                              null_bitmap_left.data(), null_bitmap_right.data(),
                              left_ind, right_ind);
                if (match) {
                    // If our group matches, add every row and
                    // update the bitmap
                    handle_build_table_hit<build_table_outer>::apply(
                        V_build_map, pos);
                    has_match = true;
                    for (size_t idx = 0; idx < group->size(); idx++) {
                        size_t j_build = (*group)[idx];
                        build_write_idxs.emplace_back(j_build);
                        probe_write_idxs.emplace_back(i_probe);
                    }
                }
            }
            if (!has_match) {
                handle_probe_table_miss<probe_table_outer, is_outer_broadcast>::
                    apply(build_write_idxs, probe_write_idxs, V_probe_map,
                          i_probe);
            }
        }
    }
}

/**
 * @brief Insert the probe table rows into the hash maps for
 * joins that only have equality conditions.
 *
 * @tparam build_table_outer Is the build table output an outer join?
 * @tparam probe_table_outer Is the probe table output an outer join?
 * @tparam is_outer_broadcast Is the probe table an outer join and a broadcast
 * join?
 * @param[in] key_rows_map The hash map that maps keys to rows in the build
 * table.
 * @param[in] groups Vector of row groups that have the same keys in the build
 * table.
 * @param[in] build_table_rows The number of rows in the build table.
 * @param[in] probe_table_rows The number of rows in the probe table.
 * @param[in] V_build_map The bitmap that indicates which rows in the build
 * table found matches.
 * @param[in] V_probe_map The bitmap that indicates which rows in the probe
 * table found matches.
 * @param[out] build_write_idxs The vector of rows for generating the output for
 * the build table.
 * @param[out] probe_write_idxs The vector of rows for generating the output for
 * the probe table.
 */
template <bool build_table_outer, bool probe_table_outer,
          bool is_outer_broadcast, typename Map>
void insert_probe_table_equi_join_all_equality(
    Map* key_rows_map, std::vector<std::vector<size_t>*>* groups,
    size_t build_table_rows, size_t probe_table_rows,
    std::vector<uint8_t>& V_build_map, std::vector<uint8_t>& V_probe_map,
    std::vector<int64_t>& build_write_idxs,
    std::vector<int64_t>& probe_write_idxs) {
    for (size_t i_probe = 0; i_probe < probe_table_rows; i_probe++) {
        size_t i_probe_shift = i_probe + build_table_rows;
        auto iter = key_rows_map->find(i_probe_shift);
        if (iter == key_rows_map->end()) {
            handle_probe_table_miss<probe_table_outer,
                                    is_outer_broadcast>::apply(build_write_idxs,
                                                               probe_write_idxs,
                                                               V_probe_map,
                                                               i_probe);
        } else {
            // If the build table entry is present in output as
            // well, then we need to keep track whether they are
            // used or not by the probe table.
            std::vector<size_t>* group = (*groups)[iter->second - 1];
            size_t pos = iter->second - 1;
            handle_build_table_hit<build_table_outer>::apply(V_build_map, pos);
            for (size_t idx = 0; idx < group->size(); idx++) {
                size_t j_build = (*group)[idx];
                build_write_idxs.emplace_back(j_build);
                probe_write_idxs.emplace_back(i_probe);
            }
        }
    }
}

/**
 * @brief Template to handle the update to build_write_idxs and probe_write_idxs
 * when a build table doesn't have a match and we are in an outer join.
 *
 * @tparam build_miss_needs_reduction Is a reduction step required to update the
 * build table. This is means the build table is broadcast but the output is
 * distributed.
 */
template <bool build_miss_needs_reduction>
struct insert_build_table_miss;

template <>
struct insert_build_table_miss<true> {
    /**
     * @brief Update the build_write_idxs and probe_write_idxs when a short
     * table doesn't have a match and we are in an outer join. This is the case
     * where the build table is broadcast, so we divide the misses across ranks.
     *
     * @param pos The position of the build table row.
     * @param build_write_idxs The build table write indices.
     * @param probe_write_idxs The probe table write indices.
     * @param myrank The current rank.
     * @param n_pes The total number of processes.
     * @param pos_build_disp The total number of missing rows in the build table
     * so far.
     */
    static inline void apply(size_t pos, std::vector<int64_t>& build_write_idxs,
                             std::vector<int64_t>& probe_write_idxs,
                             int64_t myrank, int64_t n_pes,
                             int64_t pos_build_disp) {
        int node = pos_build_disp % n_pes;
        // For build_miss_needs_reduction=True, the output table
        // is distributed. Since the table in input is
        // replicated, we dispatch it by rank.
        if (node == myrank) {
            build_write_idxs.emplace_back(pos);
            probe_write_idxs.emplace_back(-1);
        }
    }
};

template <>
struct insert_build_table_miss<false> {
    /**
     * @brief Update the build_write_idxs and probe_write_idxs when a build
     * table doesn't have a match and we are in an outer join. This is the case
     * where the build table is distributed so we just write to the output.
     *
     * @param pos The position of the build table row.
     * @param build_write_idxs The build table write indices.
     * @param probe_write_idxs The probe table write indices.
     * @param myrank The current rank.
     * @param n_pes The total number of processes.
     * @param pos_build_disp The total number of missing rows in the build table
     * so far.
     */
    static inline void apply(size_t pos, std::vector<int64_t>& build_write_idxs,
                             std::vector<int64_t>& probe_write_idxs,
                             int64_t myrank, int64_t n_pes,
                             int64_t pos_build_disp) {
        build_write_idxs.emplace_back(pos);
        probe_write_idxs.emplace_back(-1);
    }
};

/**
 * @brief Template to handle the update build_write_idxs and probe_write_idxs
 * with any missing rows from the build table.
 *
 * @tparam build_miss_needs_reduction Is a reduction step required to update the
 * build table. This means the build table is broadcast but the output is
 * distributed.
 *
 * @param V_build_map Bitmap vector of hits/misses in the build table's groups.
 * @param groups Vector of vector of rows in the build table.
 * @param build_write_idxs The build table write indices.
 * @param probe_write_idxs The probe table write indices.
 * @param myrank The current rank.
 * @param n_pes The total number of processes.
 */
template <bool build_miss_needs_reduction>
void insert_build_table_misses(std::vector<uint8_t>& V_build_map,
                               std::vector<std::vector<size_t>*>* groups,
                               std::vector<int64_t>& build_write_idxs,
                               std::vector<int64_t>& probe_write_idxs,
                               int64_t myrank, int64_t n_pes) {
    if (build_miss_needs_reduction) {
        // Perform the reduction on build table misses if necessary
        MPI_Allreduce_bool_or(V_build_map);
    }
    int64_t pos_build_disp = 0;
    // Add missing rows for outer joins when there are no matching build
    // table groups.
    for (size_t pos = 0; pos < groups->size(); pos++) {
        std::vector<size_t>* group = (*groups)[pos];
        bool bit = GetBit(V_build_map.data(), pos);
        if (!bit) {
            for (size_t idx = 0; idx < group->size(); idx++) {
                size_t j_build = (*group)[idx];
                insert_build_table_miss<build_miss_needs_reduction>::apply(
                    j_build, build_write_idxs, probe_write_idxs, myrank, n_pes,
                    pos_build_disp);
                pos_build_disp++;
            }
        }
    }
}

/**
 * @brief Update the build_write_idxs and probe_write_idxs when a probe table is
 * broadcast but the output is distributed. For each miss, we divide the misses
 * across ranks.
 *
 * @param V_probe_map The bitmap vector of hits/misses in the probe table's
 * groups.
 * @param build_write_idxs The build table write indices.
 * @param probe_write_idxs The probe table write indices.
 * @param probe_table_rows The number of rows in the probe table.
 * @param myrank The current rank.
 * @param n_pes The total number of processes.
 */
void insert_probe_table_broadcast_misses(std::vector<uint8_t>& V_probe_map,
                                         std::vector<int64_t>& build_write_idxs,
                                         std::vector<int64_t>& probe_write_idxs,
                                         size_t probe_table_rows,
                                         int64_t myrank, int64_t n_pes) {
    MPI_Allreduce_bool_or(V_probe_map);
    int pos = 0;
    for (size_t i_probe = 0; i_probe < probe_table_rows; i_probe++) {
        bool bit = GetBit(V_probe_map.data(), i_probe);
        // The replicated input table is dispatched over the rows in the
        // distributed output.
        if (!bit) {
            int node = pos % n_pes;
            if (node == myrank) {
                build_write_idxs.emplace_back(-1);
                probe_write_idxs.emplace_back(i_probe);
            }
            pos++;
        }
    }
}

/**
 * @brief Compute the information about the last use of each column in the work
 * tables and free any columns that are not needed for generating the outputs.
 * There are many cases to cover, so we preprocess last_col_use_left and
 * last_col_use_right first to determine last usage before creating the return
 * table. last_col_use_left/right can be assigned many different values.
 * Here are there meanings.
 *  0 - Default value. This means the column is not used in the output at
 *  all. 1 - Column is the output of converting an index to the output
 *  data columns. 2 - Column is a key column with the same name in the
 *  left and right tables. 3 - Column is in the left table and is not a
 * matching key with the right table. 4 - Column is in the right table,
 * is not a matching key with the left table, and the operation is not a
 * join in Pandas.
 *
 * This function also frees any columns in last_col_use_left/right that have a
 * value of 0.
 *
 * @param[out] last_col_use_left The vector of last use information for the left
 * table.
 * @param[out] last_col_use_right The vector of last use information for the
 * right table.
 * @param[in] work_left_table The left table.
 * @param[in] work_right_table The right table.
 * @param[in] n_tot_left The number of left table columns.
 * @param[in] n_tot_right The number of right table columns.
 * @param[in] n_key The number of key columns.
 * @param[in] vect_same_key Boolean array tracking if the same key becomes 1
 * output column.
 * @param[in] key_in_output Boolean array tracking which key columns are live in
 * the output.
 * @param[in] left_cond_func_cols_set Set of left table columns used in the
 * condition function.
 * @param[in] right_cond_func_cols_set Set of right table columns used in the
 * condition function.
 * @param[in] extra_data_col Does the output have an extra data column that is
 * generated from converting an index column to a data column?
 * @param[in] is_join Is this operation generated via DataFrame.join? This seems
 * to impact which right table columns are live in the output, although this may
 * be a bug.
 */
void generate_col_last_use_info(
    std::vector<uint8_t>& last_col_use_left,
    std::vector<uint8_t>& last_col_use_right, table_info* work_left_table,
    table_info* work_right_table, size_t n_tot_left, size_t n_tot_right,
    size_t n_key, int64_t* vect_same_key, bool* key_in_output,
    UNORD_SET_CONTAINER<int64_t>* left_cond_func_cols_set,
    UNORD_SET_CONTAINER<int64_t>* right_cond_func_cols_set, bool extra_data_col,
    bool is_join) {
    offset_t key_in_output_idx = 0;

    if (extra_data_col) {
        size_t i = 0;
        last_col_use_left[i] = 1;
        last_col_use_right[i] = 1;
    }
    // Iterate through the left table columns. If its a key column, check if it
    // is shared with the right table and if so update the right key column as
    // being used when the left key is.
    for (size_t i = 0; i < n_tot_left; i++) {
        if (i < n_key && vect_same_key[i] == 1) {
            if (key_in_output[key_in_output_idx++]) {
                // Key is shared by both tables and kept in the output.
                last_col_use_left[i] = 2;
                last_col_use_right[i] = 2;
            }
        } else {
            // For deletion purposes the key is either a regular key column or
            // a data column used in a general join condition.
            bool is_key_col = i < n_key || left_cond_func_cols_set->contains(i);
            if (!is_key_col || key_in_output[key_in_output_idx++]) {
                // Column is only in the left table.
                last_col_use_left[i] = 3;
            }
        }
    }
    // Iterate through the right table columns. Here we check for any columns
    // in the right table that are live in the output. There are two main cases
    // in which we mark a column as being used when inserting into the right
    // table.
    // -- It is a right key column with different name from the left and
    // !is_join?
    //    is_join indicates that we are doing a DataFrame.join, which right now
    //    always uses the right table's index as a key column. This may be a bug
    //    and we likely need to handle the more general case of merging on
    //    indices.
    // -- It is a right data column. If a right data column is used in the
    // general
    //    merge condition then we have to check if it is kept in the output as
    //    it is used like a key.
    for (size_t i = 0; i < n_tot_right; i++) {
        bool is_normal_key = i < n_key && vect_same_key[i] == 0 && !is_join;
        if (i >= n_key || is_normal_key) {
            // Is the column a key column via the non-equality function?
            bool is_non_equal_key =
                i >= n_key && right_cond_func_cols_set->contains(i);
            if (!(is_normal_key || is_non_equal_key) ||
                key_in_output[key_in_output_idx++]) {
                // Column is only in the right table.
                last_col_use_right[i] = 4;
            }
        }
    }
    // If the arrays aren't used in the output decref immediately.
    for (size_t i = 0; i < n_tot_left; i++) {
        if (last_col_use_left[i] == 0) {
            array_info* left_arr = work_left_table->columns[i];
            decref_array(left_arr);
        }
    }
    for (size_t i = 0; i < n_tot_right; i++) {
        if (last_col_use_right[i] == 0) {
            array_info* right_arr = work_right_table->columns[i];
            decref_array(right_arr);
        }
    }
}

/**
 * @brief Helper function for the compute tuple step of hash join.
 *
 * All arguments are passed from the main hash join function.
 *
 */
template <typename Map>
void hash_join_compute_tuples_helper(
    /*const*/ table_info* work_left_table,
    /*const*/ table_info* work_right_table, const size_t n_tot_left,
    const size_t n_tot_right, const bool uses_cond_func,
    const bool build_is_left, uint64_t* cond_func_left_columns,
    const uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    const uint64_t cond_func_right_column_len, const bool parallel_trace,
    const size_t build_table_rows, const table_info* build_table,
    const size_t probe_table_rows, const bool probe_miss_needs_reduction,
    Map* key_rows_map, std::vector<std::vector<size_t>*>* groups,
    const bool build_table_outer, const bool probe_table_outer,
    cond_expr_fn_t& cond_func, tracing::Event& ev_alloc_map,
    std::vector<UNORD_MAP_CONTAINER<
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>*
        second_level_hash_maps,
    std::shared_ptr<uint32_t[]>& build_nonequal_key_hashes,
    std::vector<uint8_t>& V_build_map, std::vector<int64_t>& build_write_idxs,
    std::vector<uint8_t>& V_probe_map, std::vector<int64_t>& probe_write_idxs) {
    // Create a data structure containing the columns to match the format
    // expected by cond_func. We create two pairs of vectors, one with
    // the array_infos, which handle general types, and one with just data1
    // as a fast path for accessing numeric data. These include both keys
    // and data columns as either can be used in the cond_func.
    std::vector<array_info*>& left_table_infos = work_left_table->columns;
    std::vector<array_info*>& right_table_infos = work_right_table->columns;
    std::vector<void*> col_ptrs_left(n_tot_left);
    std::vector<void*> col_ptrs_right(n_tot_right);
    // Vectors for null bitmaps for fast null checking from the cfunc
    std::vector<void*> null_bitmap_left(n_tot_left);
    std::vector<void*> null_bitmap_right(n_tot_right);
    for (size_t i = 0; i < n_tot_left; i++) {
        col_ptrs_left[i] =
            static_cast<void*>(work_left_table->columns[i]->data1);
        null_bitmap_left[i] =
            static_cast<void*>(work_left_table->columns[i]->null_bitmask);
    }
    for (size_t i = 0; i < n_tot_right; i++) {
        col_ptrs_right[i] =
            static_cast<void*>(work_right_table->columns[i]->data1);
        null_bitmap_right[i] =
            static_cast<void*>(work_right_table->columns[i]->null_bitmask);
    }

    // Keep track of which table to use to populate the second level hash
    // table if 'uses_cond_func'
    uint64_t* build_data_key_cols = nullptr;
    uint64_t build_data_key_n_cols = 0;

    if (uses_cond_func) {
        if (build_is_left) {
            build_data_key_cols = cond_func_left_columns;
            build_data_key_n_cols = cond_func_left_column_len;
        } else {
            build_data_key_cols = cond_func_right_columns;
            build_data_key_n_cols = cond_func_right_column_len;
        }
        build_nonequal_key_hashes = hash_data_cols_table(
            build_table->columns, build_data_key_cols, build_data_key_n_cols,
            SEED_HASH_JOIN, parallel_trace);
        // [BE-1078]: Should this use statistics to influence how much we
        // reserve?
        second_level_hash_maps->reserve(build_table_rows);
    }
    joinHashFcts::SecondLevelHashHashJoinTable second_level_hash_fct{
        build_nonequal_key_hashes};
    joinHashFcts::SecondLevelKeyEqualHashJoinTable second_level_equal_fct{
        build_table, build_data_key_cols, build_data_key_n_cols};

    // [BE-1078]: how much should we reserve?
    build_write_idxs.reserve(probe_table_rows);
    probe_write_idxs.reserve(probe_table_rows);
    ev_alloc_map.finalize();

    // If we will need to perform a reduction on probe table matches,
    // specify an allocation size for the vector.
    size_t n_bytes_probe = 0;
    if (probe_miss_needs_reduction) {
        n_bytes_probe = (probe_table_rows + 7) >> 3;
    }
    V_probe_map.resize(n_bytes_probe, 255);

    tracing::Event ev_groups("calc_groups", parallel_trace);
    // Loop over the build table.
    if (uses_cond_func) {
        insert_build_table_equi_join_some_non_equality(
            key_rows_map, second_level_hash_maps, second_level_hash_fct,
            second_level_equal_fct, groups, build_table_rows);
    } else {
        insert_build_table_equi_join_all_equality(key_rows_map, groups,
                                                  build_table_rows);
    }
    ev_groups.finalize();

    // Resize V_build_map based on the groups calculation.
    size_t n_bytes_build = 0;
    if (build_table_outer) {
        n_bytes_build = (groups->size() + 7) >> 3;
    }
    V_build_map.resize(n_bytes_build, 0);

    // We now iterate over all the entries of the long table in order to
    // get the entries in the build_write_idxs and probe_write_idxs. We add
    // different paths to allow templated code since the miss handling depends
    // on the type of join or if we have an outer join.
    if (uses_cond_func) {
        if (build_table_outer) {
            if (probe_table_outer) {
                if (probe_miss_needs_reduction) {
                    insert_probe_table_equi_join_some_non_equality<true, true,
                                                                   true, Map>(
                        key_rows_map, second_level_hash_maps, groups,
                        build_table_rows, probe_table_rows, V_build_map,
                        V_probe_map, build_write_idxs, probe_write_idxs,
                        build_is_left, left_table_infos, right_table_infos,
                        col_ptrs_left, col_ptrs_right, null_bitmap_left,
                        null_bitmap_right, cond_func);
                } else {
                    insert_probe_table_equi_join_some_non_equality<true, true,
                                                                   false, Map>(
                        key_rows_map, second_level_hash_maps, groups,
                        build_table_rows, probe_table_rows, V_build_map,
                        V_probe_map, build_write_idxs, probe_write_idxs,
                        build_is_left, left_table_infos, right_table_infos,
                        col_ptrs_left, col_ptrs_right, null_bitmap_left,
                        null_bitmap_right, cond_func);
                }
            } else {
                insert_probe_table_equi_join_some_non_equality<true, false,
                                                               false, Map>(
                    key_rows_map, second_level_hash_maps, groups,
                    build_table_rows, probe_table_rows, V_build_map,
                    V_probe_map, build_write_idxs, probe_write_idxs,
                    build_is_left, left_table_infos, right_table_infos,
                    col_ptrs_left, col_ptrs_right, null_bitmap_left,
                    null_bitmap_right, cond_func);
            }
        } else {
            if (probe_table_outer) {
                if (probe_miss_needs_reduction) {
                    insert_probe_table_equi_join_some_non_equality<false, true,
                                                                   true, Map>(
                        key_rows_map, second_level_hash_maps, groups,
                        build_table_rows, probe_table_rows, V_build_map,
                        V_probe_map, build_write_idxs, probe_write_idxs,
                        build_is_left, left_table_infos, right_table_infos,
                        col_ptrs_left, col_ptrs_right, null_bitmap_left,
                        null_bitmap_right, cond_func);
                } else {
                    insert_probe_table_equi_join_some_non_equality<false, true,
                                                                   false, Map>(
                        key_rows_map, second_level_hash_maps, groups,
                        build_table_rows, probe_table_rows, V_build_map,
                        V_probe_map, build_write_idxs, probe_write_idxs,
                        build_is_left, left_table_infos, right_table_infos,
                        col_ptrs_left, col_ptrs_right, null_bitmap_left,
                        null_bitmap_right, cond_func);
                }
            } else {
                insert_probe_table_equi_join_some_non_equality<false, false,
                                                               false, Map>(
                    key_rows_map, second_level_hash_maps, groups,
                    build_table_rows, probe_table_rows, V_build_map,
                    V_probe_map, build_write_idxs, probe_write_idxs,
                    build_is_left, left_table_infos, right_table_infos,
                    col_ptrs_left, col_ptrs_right, null_bitmap_left,
                    null_bitmap_right, cond_func);
            }
        }
    } else {
        if (build_table_outer) {
            if (probe_table_outer) {
                if (probe_miss_needs_reduction) {
                    insert_probe_table_equi_join_all_equality<true, true, true,
                                                              Map>(
                        key_rows_map, groups, build_table_rows,
                        probe_table_rows, V_build_map, V_probe_map,
                        build_write_idxs, probe_write_idxs);
                } else {
                    insert_probe_table_equi_join_all_equality<true, true, false,
                                                              Map>(
                        key_rows_map, groups, build_table_rows,
                        probe_table_rows, V_build_map, V_probe_map,
                        build_write_idxs, probe_write_idxs);
                }
            } else {
                insert_probe_table_equi_join_all_equality<true, false, false,
                                                          Map>(
                    key_rows_map, groups, build_table_rows, probe_table_rows,
                    V_build_map, V_probe_map, build_write_idxs,
                    probe_write_idxs);
            }
        } else {
            if (probe_table_outer) {
                if (probe_miss_needs_reduction) {
                    insert_probe_table_equi_join_all_equality<false, true, true,
                                                              Map>(
                        key_rows_map, groups, build_table_rows,
                        probe_table_rows, V_build_map, V_probe_map,
                        build_write_idxs, probe_write_idxs);
                } else {
                    insert_probe_table_equi_join_all_equality<false, true,
                                                              false, Map>(
                        key_rows_map, groups, build_table_rows,
                        probe_table_rows, V_build_map, V_probe_map,
                        build_write_idxs, probe_write_idxs);
                }
            } else {
                insert_probe_table_equi_join_all_equality<false, false, false,
                                                          Map>(
                    key_rows_map, groups, build_table_rows, probe_table_rows,
                    V_build_map, V_probe_map, build_write_idxs,
                    probe_write_idxs);
            }
        }
    }
}

// An overview of the join design can be found on Confluence:
// https://bodo.atlassian.net/wiki/spaces/B/pages/821624833/Join+Code+Design

// There are several heuristic in this code:
// ---We can use shuffle-join or broadcast-join. We have one constant for the
// maximal
//    size of the broadcasted table. It is set to 10 MB by default (same as
//    Spark). Variable name "bcast_join_threshold".
// ---For the join, we need to construct one hash map for the keys. If we take
// the left
//    table and is_left_outer=T then we need to build a more complicated
//    hash-map. Thus the size of the table is just one parameter in the choice.
//    We put a factor of 6.0 in this choice. Variable is CritQuotientNrows.

table_info* hash_join_table_inner(
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, int64_t n_key_t, int64_t n_data_left_t,
    int64_t n_data_right_t, int64_t* vect_same_key, bool* key_in_output,
    int64_t* use_nullable_arr_type, bool is_left_outer, bool is_right_outer,
    bool is_join, bool extra_data_col, bool indicator, bool is_na_equal,
    cond_expr_fn_t cond_func, uint64_t* cond_func_left_columns,
    uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    uint64_t cond_func_right_column_len, uint64_t* num_rows_ptr) {
    // Does this join need an additional cond_func
    const bool uses_cond_func = cond_func != nullptr;
    const bool parallel_trace = (left_parallel || right_parallel);
    tracing::Event ev("hash_join_table", parallel_trace);
    // Reading the MPI settings
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Doing checks and basic assignments.
    size_t n_key = size_t(n_key_t);
    size_t n_data_left = size_t(n_data_left_t);
    size_t n_data_right = size_t(n_data_right_t);
    size_t n_tot_left = n_key + n_data_left;
    size_t n_tot_right = n_key + n_data_right;
    // Check that the input is valid.
    // This ensures that the array type and dtype of key columns are the same
    // in both tables. This is an assumption we will use later in the code.
    validate_equi_join_input(left_table, right_table, n_key, extra_data_col);

    if (ev.is_tracing()) {
        ev.add_attribute("in_left_table_nrows",
                         static_cast<size_t>(left_table->nrows()));
        ev.add_attribute("in_right_table_nrows",
                         static_cast<size_t>(right_table->nrows()));
        ev.add_attribute("g_left_parallel", left_parallel);
        ev.add_attribute("g_right_parallel", right_parallel);
        ev.add_attribute("g_n_key", n_key_t);
        ev.add_attribute("g_n_data_cols_left", n_data_left_t);
        ev.add_attribute("g_n_data_cols_right", n_data_right_t);
        ev.add_attribute("g_is_left", is_left_outer);
        ev.add_attribute("g_is_right", is_right_outer);
        ev.add_attribute("g_extra_data_col", extra_data_col);
    }
    // Handle dict encoding
    equi_join_keys_handle_dict_encoded(left_table, right_table, left_parallel,
                                       right_parallel, n_key);

    auto [left_cond_func_cols_set, right_cond_func_cols_set] =
        create_non_equi_func_sets(
            left_table, right_table, cond_func_left_columns,
            cond_func_left_column_len, cond_func_right_columns,
            cond_func_right_column_len, left_parallel, right_parallel, n_key);

    // Now deciding how we handle the parallelization of the tables
    // and doing the allgather/shuffle exchanges as needed.
    // There are two possible algorithms:
    // --- shuffle join where we shuffle both tables and then do the join.
    // --- broadcast join where one table is small enough to be gathered and
    //     broadcast to all nodes.
    // Note that Spark has other join algorithms, e.g. sort-shuffle join.
    //
    // Relative complexity of each algorithms:
    // ---Shuffle join requires shuffling which implies the use of
    // MPI_alltoallv
    //    and uses a lot of memory.
    // ---Broadcast join requires one table to be gathered which may be
    // advantageous
    //    if one table is much smaller. Still making a table replicated
    //    should give pause to users.
    //
    // Both algorithms require the construction of a hash map for the keys.
    //
    auto [work_left_table, work_right_table, left_replicated,
          right_replicated] =
        equi_join_shuffle(left_table, right_table, is_left_outer,
                          is_right_outer, left_parallel, right_parallel, n_pes,
                          n_key, is_na_equal);
    ev.add_attribute("g_left_replicated", left_replicated);
    ev.add_attribute("g_right_replicated", right_replicated);

    // At this point we always refer to the work tables, which abstracts
    // away the original left and right tables. We cannot garbage
    // collect either table because they may be reused in the Python code,
    // so we must store both the original table and the shuffled/broadcast
    // table.

    size_t n_rows_left = work_left_table->nrows();
    size_t n_rows_right = work_right_table->nrows();

    // Now computing the hashes that will be used in the hash map
    // or compared to the hash map.
    //
    std::shared_ptr<uint32_t[]> hashes_left =
        hash_keys_table(work_left_table, n_key, SEED_HASH_JOIN, parallel_trace);
    std::shared_ptr<uint32_t[]> hashes_right = hash_keys_table(
        work_right_table, n_key, SEED_HASH_JOIN, parallel_trace);

    bool build_is_left = select_build_table(n_rows_left, n_rows_right,
                                            is_left_outer, is_right_outer, ev);

    // Select the "build" and "probe" table based upon select_build_table
    // output. For the build table we construct a hash map, for
    // the probe table, we simply iterate over the rows and see if the keys
    // are in the hash map.
    size_t build_table_rows, probe_table_rows;  // the number of rows
    std::shared_ptr<uint32_t[]> build_table_hashes;
    std::shared_ptr<uint32_t[]> probe_table_hashes;
    // This corresponds to is_left_outer/is_right_outer and determines
    // if the build/probe tables are outer joins.
    bool build_table_outer, probe_table_outer;
    table_info *build_table, *probe_table;
    bool build_replicated, probe_replicated;
    // Flag set to true if left table is assigned as build table,
    // used in equal_fct below
    if (build_is_left) {
        // build = left and probe = right
        build_table_outer = is_left_outer;
        probe_table_outer = is_right_outer;
        build_table = work_left_table;
        probe_table = work_right_table;
        build_table_rows = n_rows_left;
        probe_table_rows = n_rows_right;
        build_table_hashes = hashes_left;
        probe_table_hashes = hashes_right;
        build_replicated = left_replicated;
        probe_replicated = right_replicated;
    } else {
        // build = right and probe = left
        build_table_outer = is_right_outer;
        probe_table_outer = is_left_outer;
        build_table = work_right_table;
        probe_table = work_left_table;
        build_table_rows = n_rows_right;
        probe_table_rows = n_rows_left;
        build_table_hashes = hashes_right;
        probe_table_hashes = hashes_left;
        build_replicated = right_replicated;
        probe_replicated = left_replicated;
    }
    // If exactly one table is replicated and that table uses an outer
    // join (i.e. is_left_outer), we do not have enough information in one
    // rank to determine that a row has no matches. As a result,
    // these variables track if we will need to perform a reduction
    // or if a row has a match.
    const bool build_miss_needs_reduction =
        build_table_outer && build_replicated && !probe_replicated;
    const bool probe_miss_needs_reduction =
        probe_table_outer && probe_replicated && !build_replicated;

    tracing::Event ev_tuples("compute_tuples", parallel_trace);
    if (ev_tuples.is_tracing()) {
        ev_tuples.add_attribute("build_table_rows", build_table_rows);
        ev_tuples.add_attribute("probe_table_rows", probe_table_rows);
        ev_tuples.add_attribute("build_table_outer", build_table_outer);
        ev_tuples.add_attribute("probe_table_outer", probe_table_outer);
    }

    tracing::Event ev_alloc_map("alloc_hashmap", parallel_trace);
    joinHashFcts::HashHashJoinTable hash_fct{
        build_table_rows, build_table_hashes, probe_table_hashes};

    std::vector<std::vector<size_t>*>* groups = nullptr;
    std::shared_ptr<uint32_t[]> build_nonequal_key_hashes =
        std::shared_ptr<uint32_t[]>(nullptr);
    std::vector<UNORD_MAP_CONTAINER<
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>*
        second_level_hash_maps = nullptr;

    // build_write_idxs and probe_write_idxs are used for the output.
    // It precises the index used for the writing of the output table
    // from the build and probe table.
    std::vector<int64_t> build_write_idxs, probe_write_idxs;

    // Allocate the vector for any build misses.
    // Start off as empty since it will be resized after the groups are
    // calculated.
    std::vector<uint8_t> V_build_map(0);

    // V_probe_map and V_build_map takes similar roles.
    // They indicate if an entry in the build or probe table
    // has been matched to the other side.
    // This is needed only if said table is replicated
    // (i.e probe_miss_needs_reduction/build_miss_needs_reduction).
    // Start off as empty, since it will be re-sized later.
    std::vector<uint8_t> V_probe_map(0);

    // The 'key_rows_map' contains the identical keys with the corresponding
    // rows. We address the entry by the row index. We store all the rows which
    // are identical in a "group". The hashmap stores the group number
    // and the groups are stored in the `groups` std::vector.
    //
    // The groups are stored in the hash map from values 1 to n,
    // although the groups indices are still 0 to n - 1.
    // This is because 0 is a reserved value that we use to indicate
    // that a value is not found in the hash map when inserting the build
    // table.
    //
    // NOTE: we don't store the group vectors in the map because it makes
    // map (re)allocs and deallocation very expensive, and possibly also
    // makes insertion and/or lookups slower.
    // StoreHash=true speeds up join code overall in TPC-H.
    // If we also need to do build_table_outer then we store an index in the
    // first position of the std::vector

    // Defining the common part of EQ_JOIN_IMPL_ELSE_CASE, EQ_JOIN_1_KEY_IMPL
    // and EQ_JOIN_2_KEYS_IMPL macros to avoid repetition.
#ifndef EQ_JOIN_IMPL_COMMON
#define EQ_JOIN_IMPL_COMMON(JOIN_KEY_TYPE)                                     \
    using unordered_map_t =                                                    \
        UNORD_MAP_CONTAINER<size_t, size_t, joinHashFcts::HashHashJoinTable,   \
                            JOIN_KEY_TYPE>;                                    \
    unordered_map_t* key_rows_map = new UNORD_MAP_CONTAINER<                   \
        size_t, size_t, joinHashFcts::HashHashJoinTable, JOIN_KEY_TYPE>(       \
        {}, hash_fct, equal_fct);                                              \
    /* reserving space is very important to avoid expensive reallocations      \
     * (at the cost of using more memory)                                      \
     * [BE-1078]: Should this use statistics to influence how much we          \
     * reserve?                                                                \
     */                                                                        \
    key_rows_map->reserve(build_table_rows);                                   \
    groups = new std::vector<std::vector<size_t>*>();                          \
    groups->reserve(build_table_rows);                                         \
    /* Define additional information needed for non-equality conditions,       \
     * determined by 'uses_cond_func'. We need a vector of hash maps for the   \
     * build table groups. In addition, we also need hashes for the build      \
     * table on all columns that are not since they will insert into the       \
     * hash map.                                                               \
     */                                                                        \
    second_level_hash_maps = new std::vector<UNORD_MAP_CONTAINER<              \
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,            \
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>();                   \
    hash_join_compute_tuples_helper(                                           \
        work_left_table, work_right_table, n_tot_left, n_tot_right,            \
        uses_cond_func, build_is_left, cond_func_left_columns,                 \
        cond_func_left_column_len, cond_func_right_columns,                    \
        cond_func_right_column_len, parallel_trace, build_table_rows,          \
        build_table, probe_table_rows, probe_miss_needs_reduction,             \
        key_rows_map, groups, build_table_outer, probe_table_outer, cond_func, \
        ev_alloc_map, second_level_hash_maps, build_nonequal_key_hashes,       \
        V_build_map, build_write_idxs, V_probe_map, probe_write_idxs);         \
    tracing::Event ev_clear_map("dealloc_key_rows_map", parallel_trace);       \
    /* Data structures used during computation of the tuples can become        \
     * quite large and their deallocation can take a non-negligible amount     \
     * of time. We dealloc them here to free memory for the next stage and     \
     * also to trace dealloc time.                                             \
     */                                                                        \
    delete key_rows_map;                                                       \
    ev_clear_map.finalize();
#endif

    // General implementation with generic key comparator class.
    // Used for non-specialized cases.
#ifndef EQ_JOIN_IMPL_ELSE_CASE
#define EQ_JOIN_IMPL_ELSE_CASE                                               \
    using JoinKeyType = joinHashFcts::KeyEqualHashJoinTable;                 \
    JoinKeyType equal_fct{build_table_rows, n_key, build_table, probe_table, \
                          is_na_equal};                                      \
    EQ_JOIN_IMPL_COMMON(JoinKeyType);
#endif

    // Use faster specialized implementation for common 1 key cases.
    if (n_key == 1) {
        array_info* build_arr = build_table->columns[0];
        bodo_array_type::arr_type_enum build_arr_type = build_arr->arr_type;
        Bodo_CTypes::CTypeEnum build_dtype = build_arr->dtype;
        array_info* probe_arr = probe_table->columns[0];
        bodo_array_type::arr_type_enum probe_arr_type = probe_arr->arr_type;
        Bodo_CTypes::CTypeEnum probe_dtype = probe_arr->dtype;

        // Macro to reduce code duplication for the 1-key specialization.
#ifndef EQ_JOIN_1_KEY_IMPL
#define EQ_JOIN_1_KEY_IMPL(ARRAY_TYPE, DTYPE, IS_NA_EQUAL, FOUND_MATCH_VAR) \
    if (build_arr_type == ARRAY_TYPE && build_dtype == DTYPE &&             \
        is_na_equal == IS_NA_EQUAL && probe_arr_type == ARRAY_TYPE &&       \
        probe_dtype == DTYPE) {                                             \
        FOUND_MATCH_VAR = true;                                             \
        using JoinKeyType =                                                 \
            JoinKeysEqualComparatorOneKey<ARRAY_TYPE, DTYPE, IS_NA_EQUAL>;  \
        JoinKeyType equal_fct{build_arr, probe_arr};                        \
        EQ_JOIN_IMPL_COMMON(JoinKeyType);                                   \
    }
#endif
        bool found_match = false;
        EQ_JOIN_1_KEY_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                           Bodo_CTypes::INT32, true, found_match);
        EQ_JOIN_1_KEY_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                           Bodo_CTypes::INT32, false, found_match);
        EQ_JOIN_1_KEY_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                           Bodo_CTypes::INT64, true, found_match);
        EQ_JOIN_1_KEY_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                           Bodo_CTypes::INT64, false, found_match);
        EQ_JOIN_1_KEY_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME, true,
                           found_match);
        EQ_JOIN_1_KEY_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME, false,
                           found_match);
        EQ_JOIN_1_KEY_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING, true,
                           found_match);
        EQ_JOIN_1_KEY_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING, false,
                           found_match);
        if (!found_match) {
            EQ_JOIN_IMPL_ELSE_CASE;
        }
    } else if (n_key == 2) {
        array_info* build_arr1 = build_table->columns[0];
        array_info* build_arr2 = build_table->columns[1];
        bodo_array_type::arr_type_enum build_arr_type1 = build_arr1->arr_type;
        Bodo_CTypes::CTypeEnum build_dtype1 = build_arr1->dtype;
        bodo_array_type::arr_type_enum build_arr_type2 = build_arr2->arr_type;
        Bodo_CTypes::CTypeEnum build_dtype2 = build_arr2->dtype;

        array_info* probe_arr1 = probe_table->columns[0];
        array_info* probe_arr2 = probe_table->columns[1];
        bodo_array_type::arr_type_enum probe_arr_type1 = probe_arr1->arr_type;
        Bodo_CTypes::CTypeEnum probe_dtype1 = probe_arr1->dtype;
        bodo_array_type::arr_type_enum probe_arr_type2 = probe_arr2->arr_type;
        Bodo_CTypes::CTypeEnum probe_dtype2 = probe_arr2->dtype;

        // Macro to reduce code duplication for the 2-keys specialization.
#ifndef EQ_JOIN_2_KEYS_IMPL
#define EQ_JOIN_2_KEYS_IMPL(ARRAY_TYPE1, DTYPE1, ARRAY_TYPE2, DTYPE2,          \
                            IS_NA_EQUAL, FOUND_MATCH_VAR)                      \
    if (build_arr_type1 == ARRAY_TYPE1 && build_dtype1 == DTYPE1 &&            \
        build_arr_type2 == ARRAY_TYPE2 && build_dtype2 == DTYPE2 &&            \
        is_na_equal == IS_NA_EQUAL && probe_arr_type1 == ARRAY_TYPE1 &&        \
        probe_dtype1 == DTYPE1 && probe_arr_type2 == ARRAY_TYPE2 &&            \
        probe_dtype2 == DTYPE2) {                                              \
        FOUND_MATCH_VAR = true;                                                \
        using JoinKeyType =                                                    \
            JoinKeysEqualComparatorTwoKeys<ARRAY_TYPE1, DTYPE1, ARRAY_TYPE2,   \
                                           DTYPE2, IS_NA_EQUAL>;               \
        JoinKeyType equal_fct{build_arr1, build_arr2, probe_arr1, probe_arr2}; \
        EQ_JOIN_IMPL_COMMON(JoinKeyType);                                      \
    }
#endif
        bool found_match = false;
        // int32 / (int32, int64, datetime, dict-encoded)
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, bodo_array_type::NUMPY,
                            Bodo_CTypes::DATETIME, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, bodo_array_type::NUMPY,
                            Bodo_CTypes::DATETIME, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, bodo_array_type::DICT,
                            Bodo_CTypes::STRING, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, bodo_array_type::DICT,
                            Bodo_CTypes::STRING, false, found_match);
        // int64 / (int32, int64, datetime, dict-encoded)
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, bodo_array_type::NUMPY,
                            Bodo_CTypes::DATETIME, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, bodo_array_type::NUMPY,
                            Bodo_CTypes::DATETIME, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, bodo_array_type::DICT,
                            Bodo_CTypes::STRING, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, bodo_array_type::DICT,
                            Bodo_CTypes::STRING, false, found_match);
        // datetime / (int32, int64, datetime, dict-encoded)
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            bodo_array_type::NUMPY, Bodo_CTypes::DATETIME, true,
                            found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            bodo_array_type::DICT, Bodo_CTypes::STRING, true,
                            found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            bodo_array_type::DICT, Bodo_CTypes::STRING, false,
                            found_match);
        // dict-encoded / (int32, int64, datetime, dict-encoded)
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, true, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING,
                            bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64, false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING,
                            bodo_array_type::NUMPY, Bodo_CTypes::DATETIME, true,
                            found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING,
                            bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                            false, found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING,
                            bodo_array_type::DICT, Bodo_CTypes::STRING, true,
                            found_match);
        EQ_JOIN_2_KEYS_IMPL(bodo_array_type::DICT, Bodo_CTypes::STRING,
                            bodo_array_type::DICT, Bodo_CTypes::STRING, false,
                            found_match);
        if (!found_match) {
            EQ_JOIN_IMPL_ELSE_CASE;
        }

    } else {
        EQ_JOIN_IMPL_ELSE_CASE;
    }

    tracing::Event ev_clear_map("dealloc_second_level_hashmaps",
                                parallel_trace);
    // Delete all the hash maps in the non-equality function case.
    for (auto map : *second_level_hash_maps) {
        delete map;
    }
    delete second_level_hash_maps;

    // Delete the hashes
    hashes_left.reset();
    hashes_right.reset();
    if (build_nonequal_key_hashes) {
        build_nonequal_key_hashes.reset();
    }
    ev_clear_map.finalize();

    // Handle updating the indices for any misses in the short table.
    if (build_table_outer) {
        if (build_miss_needs_reduction) {
            insert_build_table_misses<true>(V_build_map, groups,
                                            build_write_idxs, probe_write_idxs,
                                            myrank, n_pes);
        } else {
            insert_build_table_misses<false>(V_build_map, groups,
                                             build_write_idxs, probe_write_idxs,
                                             myrank, n_pes);
        }
    }

    tracing::Event ev_clear_groups("dealloc_groups", parallel_trace);
    // Delete the groups now that they are no longer needed.
    for (auto group : *groups) {
        delete group;
    }
    delete groups;
    ev_clear_groups.finalize();

    // In replicated case, we put the long rows in distributed output
    if (probe_miss_needs_reduction) {
        insert_probe_table_broadcast_misses(V_probe_map, build_write_idxs,
                                            probe_write_idxs, probe_table_rows,
                                            myrank, n_pes);
    }
    ev_tuples.add_attribute("output_nrows", probe_write_idxs.size());
    ev_tuples.finalize();
    // TODO if we are tight on memory for the next phase and
    // build_write_idxs/probe_write_idxs capacity is much greater than its
    // size, we could do a realloc+resize here
    std::vector<array_info*> out_arrs;
    // Computing the last time at which a column is used.
    // This is for the call to decref_array.
    // There are many cases to cover, so it is better to preprocess
    // first to determine last usage before creating the return
    // table.
    // last_col_use_left/right can be assigned many different values.
    // Here are there meanings.
    // 0 - Default value. This means the column is not used in the output at
    // all. 1 - Column is the output of converting an index to the output
    // data columns. 2 - Column is a key column with the same name in the
    // left and right tables. 3 - Column is in the left table and is not a
    // matching key with the right table. 4 - Column is in the right table,
    // is not a matching key with the left table,
    //        and the operation is not a join in Pandas.
    std::vector<uint8_t> last_col_use_left(n_tot_left, 0);
    std::vector<uint8_t> last_col_use_right(n_tot_right, 0);

    generate_col_last_use_info(
        last_col_use_left, last_col_use_right, work_left_table,
        work_right_table, n_tot_left, n_tot_right, n_key, vect_same_key,
        key_in_output, left_cond_func_cols_set, right_cond_func_cols_set,
        extra_data_col, is_join);

    // Determine the number of rows in your local chunk of the output.
    // This is passed to Python in case all columns are dead.
    uint64_t num_rows = probe_write_idxs.size();
    *num_rows_ptr = num_rows;

    // Construct the output tables. This merges the results in the left and
    // right tables. We resume using work_left_table and work_right_table to
    // ensure we match the expected column order (as opposed to build/probe).

    // Inserting the optional column in the case of merging on column and
    // an index.
    if (extra_data_col) {
        tracing::Event ev_fill_optional("fill_extra_data_col", parallel_trace);
        size_t i = 0;
        array_info* left_arr = work_left_table->columns[i];
        array_info* right_arr = work_right_table->columns[i];
        if (build_is_left) {
            out_arrs.push_back(RetrieveArray_TwoColumns(
                left_arr, right_arr, build_write_idxs, probe_write_idxs));
        } else {
            out_arrs.push_back(RetrieveArray_TwoColumns(
                right_arr, left_arr, build_write_idxs, probe_write_idxs));
        }
        // After adding the optional collect decref the original values
        if (last_col_use_left[i] == 1) {
            decref_array(left_arr);
        }
        if (last_col_use_right[i] == 1) {
            decref_array(right_arr);
        }
    }

    // Where to check in key_in_output if we have a key or
    // a general merge condition.
    offset_t key_in_output_idx = 0;

    // Inserting the Left side of the table
    tracing::Event ev_fill_left("fill_left", parallel_trace);
    int idx = 0;

    // Insert the key columns
    for (size_t i = 0; i < n_key; i++) {
        // Check if this key column is included.
        if (key_in_output[key_in_output_idx++]) {
            if (vect_same_key[i] == 1) {
                // If we are merging the same in two table then
                // additional NaNs cannot happen.
                array_info* left_arr = work_left_table->columns[i];
                array_info* right_arr = work_right_table->columns[i];
                if (build_is_left) {
                    out_arrs.emplace_back(RetrieveArray_TwoColumns(
                        left_arr, right_arr, build_write_idxs,
                        probe_write_idxs));
                } else {
                    out_arrs.emplace_back(RetrieveArray_TwoColumns(
                        right_arr, left_arr, build_write_idxs,
                        probe_write_idxs));
                }
                // Decref columns that are no longer used
                if (last_col_use_left[i] == 2) {
                    decref_array(left_arr);
                }
                if (last_col_use_right[i] == 2) {
                    decref_array(right_arr);
                }
            } else {
                // We are just inserting the keys so converting to a nullable
                // array depends on the type of join.
                array_info* left_arr = work_left_table->columns[i];
                bool use_nullable_arr = use_nullable_arr_type[idx];
                if (build_is_left) {
                    out_arrs.push_back(RetrieveArray_SingleColumn(
                        left_arr, build_write_idxs, use_nullable_arr));
                } else {
                    out_arrs.push_back(RetrieveArray_SingleColumn(
                        left_arr, probe_write_idxs, use_nullable_arr));
                }
                // Decref columns that are no longer used
                if (last_col_use_left[i] == 3) {
                    decref_array(left_arr);
                }
            }
            // update the output indices
            idx++;
        }
    }
    // Insert the data columns
    for (size_t i = n_key; i < n_tot_left; i++) {
        if (left_cond_func_cols_set->contains(i) &&
            !key_in_output[key_in_output_idx++]) {
            // If this data column is used in the non equality
            // condition we have to check if its in the output.
            continue;
        }
        array_info* left_arr = work_left_table->columns[i];
        bool use_nullable_arr = use_nullable_arr_type[idx];
        if (build_is_left) {
            out_arrs.push_back(RetrieveArray_SingleColumn(
                left_arr, build_write_idxs, use_nullable_arr));
        } else {
            out_arrs.push_back(RetrieveArray_SingleColumn(
                left_arr, probe_write_idxs, use_nullable_arr));
        }
        // update the output indices
        idx++;
        // Decref columns that are no longer used
        if (last_col_use_left[i] == 3) {
            decref_array(left_arr);
        }
    }
    // Delete to free memory
    delete left_cond_func_cols_set;
    ev_fill_left.finalize();

    // Inserting the right side of the table.
    tracing::Event ev_fill_right("fill_right", parallel_trace);

    // Insert right keys
    for (size_t i = 0; i < n_key; i++) {
        // vect_same_key[i] == 0 means this key doesn't become a natural
        // join and get clobbered. The !is_join is unclear but seems to be
        // because DataFrame.join always merged with the index of the right
        // table, so this could be a bug in handling index cases.
        if (vect_same_key[i] == 0 && !is_join) {
            // If we might insert a key we have to verify it is in the output.
            if (key_in_output[key_in_output_idx++]) {
                array_info* right_arr = work_right_table->columns[i];
                bool use_nullable_arr = use_nullable_arr_type[idx];
                if (build_is_left) {
                    out_arrs.push_back(RetrieveArray_SingleColumn(
                        right_arr, probe_write_idxs, use_nullable_arr));
                } else {
                    out_arrs.push_back(RetrieveArray_SingleColumn(
                        right_arr, build_write_idxs, use_nullable_arr));
                }
                // update the output indices
                idx++;
                // Decref columns that are no longer used
                if (last_col_use_right[i] == 4) {
                    decref_array(right_arr);
                }
            }
        }
    }
    // Insert the data columns
    for (size_t i = n_key; i < n_tot_right; i++) {
        if (right_cond_func_cols_set->contains(i) &&
            !key_in_output[key_in_output_idx++]) {
            // If this data column is used in the non equality
            // condition we have to check if its in the output.
            continue;
        }
        array_info* right_arr = work_right_table->columns[i];
        bool use_nullable_arr = use_nullable_arr_type[idx];
        if (build_is_left) {
            out_arrs.push_back(RetrieveArray_SingleColumn(
                right_arr, probe_write_idxs, use_nullable_arr));
        } else {
            out_arrs.push_back(RetrieveArray_SingleColumn(
                right_arr, build_write_idxs, use_nullable_arr));
        }
        // update the output indices
        idx++;
        // Decref columns that are no longer used
        if (last_col_use_right[i] == 4) {
            decref_array(right_arr);
        }
    }

    // Delete to free memory
    delete right_cond_func_cols_set;
    ev_fill_right.finalize();

    // Create indicator column if indicator=True
    if (indicator) {
        tracing::Event ev_indicator("create_indicator", parallel_trace);
        array_info* indicator_col =
            alloc_array(num_rows, -1, -1, bodo_array_type::CATEGORICAL,
                        Bodo_CTypes::INT8, 0, 3);
        for (size_t rownum = 0; rownum < num_rows; rownum++) {
            // Determine the source of each row. At most 1 value can be -1.
            // Whichever value is -1, other table is the source of the row.
            std::vector<int64_t>* left_write_idxs;
            std::vector<int64_t>* right_write_idxs;
            if (build_is_left) {
                left_write_idxs = &build_write_idxs;
                right_write_idxs = &probe_write_idxs;
            } else {
                left_write_idxs = &probe_write_idxs;
                right_write_idxs = &build_write_idxs;
            }
            // Left is null
            if ((*left_write_idxs)[rownum] == -1) {
                indicator_col->data1[rownum] = 1;
                // Right is null
            } else if ((*right_write_idxs)[rownum] == -1) {
                indicator_col->data1[rownum] = 0;
                // Neither is null
            } else {
                indicator_col->data1[rownum] = 2;
            }
        }
        out_arrs.emplace_back(indicator_col);
        ev_indicator.finalize();
    }

    // Delete the inputs
    delete_table(work_left_table);
    delete_table(work_right_table);

    // Only return a table if there is at least 1
    // output column.
    if (out_arrs.size() > 0) {
        return new table_info(out_arrs);
    } else {
        return nullptr;
    }
}

table_info* hash_join_table(
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, int64_t n_key_t, int64_t n_data_left_t,
    int64_t n_data_right_t, int64_t* vect_same_key, bool* key_in_output,
    int64_t* use_nullable_arr_type, bool is_left_outer, bool is_right_outer,
    bool is_join, bool extra_data_col, bool indicator, bool is_na_equal,
    cond_expr_fn_t cond_func, uint64_t* cond_func_left_columns,
    uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    uint64_t cond_func_right_column_len, uint64_t* num_rows_ptr) {
    try {
        return hash_join_table_inner(
            left_table, right_table, left_parallel, right_parallel, n_key_t,
            n_data_left_t, n_data_right_t, vect_same_key, key_in_output,
            use_nullable_arr_type, is_left_outer, is_right_outer, is_join,
            extra_data_col, indicator, is_na_equal, cond_func,
            cond_func_left_columns, cond_func_left_column_len,
            cond_func_right_columns, cond_func_right_column_len, num_rows_ptr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

/**
 * @brief handle dict-encoded columns of input tables to cross join.
 * Dictionaries need to be global since we use broadcast.
 *
 * @param left_table left input table to cross join
 * @param right_table right input table to cross join
 * @param left_parallel left table is parallel
 * @param right_parallel right table is parallel
 */
void cross_join_handle_dict_encoded(table_info* left_table,
                                    table_info* right_table, bool left_parallel,
                                    bool right_parallel) {
    // make all dictionaries global (necessary for broadcast and potentially
    // other operations)
    for (array_info* a : left_table->columns) {
        if (a->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(a, left_parallel);
        }
    }
    for (array_info* a : right_table->columns) {
        if (a->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(a, right_parallel);
        }
    }
}

/**
 * @brief Create data structures for column data to match the format
 * expected by cond_func. We create three vectors:
 * the array_infos (which handle general types), and data1/nullbitmap pointers
 * as a fast path for accessing numeric data. These include both keys
 * and data columns as either can be used in the cond_func.
 *
 * @param table input table
 * @return std::tuple<std::vector<array_info*>, std::vector<void*>,
 * std::vector<void*>> vectors of array infos, data1 pointers, and null bitmap
 * pointers
 */
std::tuple<std::vector<array_info*>, std::vector<void*>, std::vector<void*>>
get_gen_cond_data_ptrs(table_info* table) {
    std::vector<array_info*> table_infos = table->columns;
    std::vector<void*> col_ptrs(table->ncols());
    std::vector<void*> null_bitmaps(table->ncols());

    for (size_t i = 0; i < table->ncols(); i++) {
        col_ptrs[i] = static_cast<void*>(table->columns[i]->data1);
        null_bitmaps[i] = static_cast<void*>(table->columns[i]->null_bitmask);
    }
    return std::make_tuple(table_infos, col_ptrs, null_bitmaps);
}

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
 */
void add_unmatched_rows(std::vector<uint8_t>& bit_map, size_t n_rows,
                        std::vector<int64_t>& table_idxs,
                        std::vector<int64_t>& other_table_idxs,
                        bool needs_reduction) {
    if (needs_reduction) {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        MPI_Allreduce_bool_or(bit_map);
        int pos = 0;
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(bit_map.data(), i);
            // distribute the replicated input table rows across ranks
            // to load balance the output
            if (!bit) {
                int node = pos % n_pes;
                if (node == myrank) {
                    table_idxs.emplace_back(i);
                    other_table_idxs.emplace_back(-1);
                }
                pos++;
            }
        }
    } else {
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(bit_map.data(), i);
            if (!bit) {
                table_idxs.emplace_back(i);
                other_table_idxs.emplace_back(-1);
            }
        }
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
 * @param decref_arrs whether input arrays should be decrefed after use.
 * @return table_info* output table of join
 */
table_info* create_out_table(
    table_info* left_table, table_info* right_table,
    std::vector<int64_t>& left_idxs, std::vector<int64_t>& right_idxs,
    bool* key_in_output, int64_t* use_nullable_arr_type,
    uint64_t* cond_func_left_columns, uint64_t cond_func_left_column_len,
    uint64_t* cond_func_right_columns, uint64_t cond_func_right_column_len,
    bool decref_arrs = true) {
    // Create sets for cond func columns. These columns act
    // like key columns and are contained inside of key_in_output,
    // so we need an efficient lookup.
    UNORD_SET_CONTAINER<int64_t> left_cond_func_cols_set;
    left_cond_func_cols_set.reserve(cond_func_left_column_len);

    UNORD_SET_CONTAINER<int64_t> right_cond_func_cols_set;
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
    std::vector<array_info*> out_arrs;
    int idx = 0;
    offset_t key_in_output_idx = 0;
    // add left columns to output
    for (size_t i = 0; i < left_table->ncols(); i++) {
        array_info* in_arr = left_table->columns[i];

        // cond columns may be dead
        if (!left_cond_func_cols_set.contains(i) ||
            key_in_output[key_in_output_idx++]) {
            bool use_nullable_arr = use_nullable_arr_type[idx];
            out_arrs.emplace_back(RetrieveArray_SingleColumn(in_arr, left_idxs,
                                                             use_nullable_arr));
            idx++;
        }
        if (decref_arrs) {
            decref_array(in_arr);
        }
    }

    // add right columns to output
    for (size_t i = 0; i < right_table->ncols(); i++) {
        array_info* in_arr = right_table->columns[i];

        // cond columns may be dead
        if (!right_cond_func_cols_set.contains(i) ||
            key_in_output[key_in_output_idx++]) {
            bool use_nullable_arr = use_nullable_arr_type[idx];
            out_arrs.emplace_back(RetrieveArray_SingleColumn(in_arr, right_idxs,
                                                             use_nullable_arr));
            idx++;
        }
        if (decref_arrs) {
            decref_array(in_arr);
        }
    }

    return new table_info(out_arrs);
}

/**
 * @brief cross join two tables locally with a simple nested loop join.
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
 * @return table_info* cross join output table
 */
void cross_join_table_local(table_info* left_table, table_info* right_table,
                            bool is_left_outer, bool is_right_outer,
                            cond_expr_fn_batch_t cond_func, bool parallel_trace,
                            std::vector<int64_t>& left_idxs,
                            std::vector<int64_t>& right_idxs,
                            std::vector<uint8_t>& left_row_is_matched,
                            std::vector<uint8_t>& right_row_is_matched) {
    tracing::Event ev("cross_join_table_local", parallel_trace);
    size_t n_rows_left = left_table->nrows();
    size_t n_rows_right = right_table->nrows();

    auto [left_table_infos, col_ptrs_left, null_bitmap_left] =
        get_gen_cond_data_ptrs(left_table);
    auto [right_table_infos, col_ptrs_right, null_bitmap_right] =
        get_gen_cond_data_ptrs(right_table);

    // set 500K block size to make sure block data of all cores fits in L3 cache
    int64_t block_size_bytes = 500 * 1024;
    char* block_size = std::getenv("BODO_CROSS_JOIN_BLOCK_SIZE");
    if (block_size) {
        block_size_bytes = std::stoi(block_size);
    }
    if (block_size_bytes < 0) {
        throw std::runtime_error(
            "cross_join_table_local: block_size_bytes < 0");
    }

    int64_t n_left_blocks = (int64_t)std::ceil(
        table_local_memory_size(left_table) / (double)block_size_bytes);
    int64_t n_right_blocks = (int64_t)std::ceil(
        table_local_memory_size(right_table) / (double)block_size_bytes);

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

    for (int64_t b_left = 0; b_left < n_left_blocks; b_left++) {
        for (int64_t b_right = 0; b_right < n_right_blocks; b_right++) {
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
            for (int64_t i = left_block_start; i < left_block_end; i++) {
                for (int64_t j = right_block_start; j < right_block_end; j++) {
                    bool match = (match_arr == nullptr) ||
                                 GetBit(match_arr, match_ind++);
                    if (match) {
                        left_idxs.emplace_back(i);
                        right_idxs.emplace_back(j);
                        if (is_left_outer) {
                            SetBitTo(left_row_is_matched.data(), i, true);
                        }
                        if (is_right_outer) {
                            SetBitTo(right_row_is_matched.data(), j, true);
                        }
                    }
                }
            }
        }
    }

    if (match_arr != nullptr) {
        delete[] match_arr;
    }
}

// design overview:
// https://bodo.atlassian.net/l/cp/Av2ijf9A
table_info* cross_join_table(
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, bool is_left_outer, bool is_right_outer,
    bool* key_in_output, int64_t* use_nullable_arr_type,
    cond_expr_fn_batch_t cond_func, uint64_t* cond_func_left_columns,
    uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    uint64_t cond_func_right_column_len, uint64_t* num_rows_ptr) {
    try {
        cross_join_handle_dict_encoded(left_table, right_table, left_parallel,
                                       right_parallel);
        bool parallel_trace = (left_parallel || right_parallel);
        tracing::Event ev("cross_join_table", parallel_trace);
        table_info* out_table;

        // use broadcast join if left or right table is small (allgather the
        // small table)
        if (left_parallel && right_parallel) {
            int bcast_join_threshold = get_bcast_join_threshold();
            int64_t left_total_memory = table_global_memory_size(left_table);
            int64_t right_total_memory = table_global_memory_size(right_table);
            if (left_total_memory < right_total_memory &&
                left_total_memory < bcast_join_threshold) {
                // Broadcast the left table
                table_info* work_left_table =
                    gather_table(left_table, -1, true, true);
                left_parallel = false;
                delete_table(left_table);
                left_table = work_left_table;
            } else if (right_total_memory <= left_total_memory &&
                       right_total_memory < bcast_join_threshold) {
                // Broadcast the right table
                table_info* work_right_table =
                    gather_table(right_table, -1, true, true);
                right_parallel = false;
                delete_table(right_table);
                right_table = work_right_table;
            }
        }

        // handle parallel cross join by broadcasting one side's table chunks of
        // from every rank (loop over all ranks). For outer join handling, the
        // broadcast side's unmatched rows need to be added right after each
        // iteration since joining on the broadcast table chunk is fully done.
        // This needs a reduction of the bitmap to find all potential matches.
        // Handling outer join for the non-broadcast table should be done
        // after all iterations are done to find all potential matches.
        // No need for reduction of bitmap since chunks are independent and not
        // replicated.
        if (left_parallel && right_parallel) {
            int n_pes, myrank;
            MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            std::vector<table_info*> out_table_chunks;
            out_table_chunks.reserve(n_pes);

            size_t n_rows_left = left_table->nrows();
            size_t n_rows_right = right_table->nrows();

            // broadcast the smaller table to reduce overall communication
            int64_t left_table_size = table_global_memory_size(left_table);
            int64_t right_table_size = table_global_memory_size(right_table);
            bool left_table_bcast = left_table_size < right_table_size;
            table_info* bcast_table = left_table;
            table_info* other_table = right_table;
            size_t n_bytes_other = is_right_outer ? (n_rows_right + 7) >> 3 : 0;
            if (!left_table_bcast) {
                bcast_table = right_table;
                other_table = left_table;
                n_bytes_other = is_left_outer ? (n_rows_left + 7) >> 3 : 0;
            }

            // bcast_row_is_matched is reset in each iteration, but
            // other_row_is_matched is updated across iterations
            std::vector<uint8_t> bcast_row_is_matched;
            std::vector<uint8_t> other_row_is_matched(n_bytes_other, 0);

            for (int p = 0; p < n_pes; p++) {
                // NOTE: broadcast_table steals a reference from bcast_table on
                // root rank. We need to keep bcast_table alive throughout this
                // loop since dictionary values are necessary as part of the
                // "reference" table on all ranks.
                if (myrank == p) {
                    incref_table_arrays(bcast_table);
                }
                table_info* bcast_table_chunk =
                    broadcast_table(bcast_table, bcast_table,
                                    bcast_table->ncols(), parallel_trace, p);
                bool is_bcast_outer = (is_left_outer && left_table_bcast) ||
                                      (is_right_outer && !left_table_bcast);
                size_t n_bytes_bcast =
                    is_bcast_outer ? (bcast_table_chunk->nrows() + 7) >> 3 : 0;
                bcast_row_is_matched.resize(n_bytes_bcast);
                std::fill(bcast_row_is_matched.begin(),
                          bcast_row_is_matched.end(), 0);

                std::vector<int64_t> left_idxs;
                std::vector<int64_t> right_idxs;

                // incref other table since needed in next iterations and
                // cross_join_table_local decrefs
                if (left_table_bcast) {
                    cross_join_table_local(
                        bcast_table_chunk, other_table, is_left_outer,
                        is_right_outer, cond_func, parallel_trace, left_idxs,
                        right_idxs, bcast_row_is_matched, other_row_is_matched);
                } else {
                    cross_join_table_local(
                        other_table, bcast_table_chunk, is_left_outer,
                        is_right_outer, cond_func, parallel_trace, left_idxs,
                        right_idxs, other_row_is_matched, bcast_row_is_matched);
                }
                // handle bcast table's unmatched rows if outer
                if (is_left_outer && left_table_bcast) {
                    add_unmatched_rows(bcast_row_is_matched,
                                       bcast_table_chunk->nrows(), left_idxs,
                                       right_idxs, true);
                }
                if (is_right_outer && !left_table_bcast) {
                    add_unmatched_rows(bcast_row_is_matched,
                                       bcast_table_chunk->nrows(), right_idxs,
                                       left_idxs, true);
                }
                // incref since create_out_table() decrefs all arrays
                incref_table_arrays(other_table);
                table_info* left_in_chunk = bcast_table_chunk;
                table_info* right_in_chunk = other_table;
                if (!left_table_bcast) {
                    left_in_chunk = other_table;
                    right_in_chunk = bcast_table_chunk;
                }
                table_info* out_table_chunk = create_out_table(
                    left_in_chunk, right_in_chunk, left_idxs, right_idxs,
                    key_in_output, use_nullable_arr_type,
                    cond_func_left_columns, cond_func_left_column_len,
                    cond_func_right_columns, cond_func_right_column_len);
                out_table_chunks.emplace_back(out_table_chunk);
                delete bcast_table_chunk;
            }

            // handle non-bcast table's unmatched rows of if outer
            if (is_left_outer && !left_table_bcast) {
                std::vector<int64_t> left_idxs;
                std::vector<int64_t> right_idxs;
                add_unmatched_rows(other_row_is_matched, left_table->nrows(),
                                   left_idxs, right_idxs, false);
                table_info* out_table_chunk = create_out_table(
                    left_table, right_table, left_idxs, right_idxs,
                    key_in_output, use_nullable_arr_type,
                    cond_func_left_columns, cond_func_left_column_len,
                    cond_func_right_columns, cond_func_right_column_len, false);
                out_table_chunks.emplace_back(out_table_chunk);
            }
            if (is_right_outer && left_table_bcast) {
                std::vector<int64_t> left_idxs;
                std::vector<int64_t> right_idxs;
                add_unmatched_rows(other_row_is_matched, right_table->nrows(),
                                   right_idxs, left_idxs, false);
                table_info* out_table_chunk = create_out_table(
                    left_table, right_table, left_idxs, right_idxs,
                    key_in_output, use_nullable_arr_type,
                    cond_func_left_columns, cond_func_left_column_len,
                    cond_func_right_columns, cond_func_right_column_len, false);
                out_table_chunks.emplace_back(out_table_chunk);
            }

            decref_table_arrays(left_table);
            decref_table_arrays(right_table);

            out_table = concat_tables(out_table_chunks);
        }
        // If either table is already replicated then broadcasting
        // isn't necessary (output's distribution will match the other input as
        // intended)
        else {
            std::vector<int64_t> left_idxs;
            std::vector<int64_t> right_idxs;

            size_t n_bytes_left =
                is_left_outer ? (left_table->nrows() + 7) >> 3 : 0;
            size_t n_bytes_right =
                is_right_outer ? (right_table->nrows() + 7) >> 3 : 0;
            std::vector<uint8_t> left_row_is_matched(n_bytes_left, 0);
            std::vector<uint8_t> right_row_is_matched(n_bytes_right, 0);

            cross_join_table_local(left_table, right_table, is_left_outer,
                                   is_right_outer, cond_func, parallel_trace,
                                   left_idxs, right_idxs, left_row_is_matched,
                                   right_row_is_matched);

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
                left_table, right_table, left_idxs, right_idxs, key_in_output,
                use_nullable_arr_type, cond_func_left_columns,
                cond_func_left_column_len, cond_func_right_columns,
                cond_func_right_column_len);
        }
        // NOTE: no need to delete table pointers since done in generated Python
        // code in join.py

        // number of local output rows is passed to Python in case all output
        // columns are dead.
        *num_rows_ptr = out_table->nrows();
        return out_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

bool is_point_right_of_interval_start(array_info* left_interval_col,
                                      const size_t& int_idx,
                                      array_info* point_col,
                                      const size_t& point_idx,
                                      bool strictly_right) {
    if (point_col->null_bitmask != nullptr &&
        !point_col->get_null_bit((size_t)point_idx)) {
        return false;
    }

    auto comp = KeyComparisonAsPython_Column(true, left_interval_col, int_idx,
                                             point_col, point_idx);
    return strictly_right ? comp > 0 : comp >= 0;
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
 * @param point_left Is the point table the left table? Required when passing
 * idxs to cond_func
 * @return A pair of vectors of indexes to the left and right table
 * repesenting the output
 */
std::pair<std::vector<int64_t>, std::vector<int64_t>> interval_merge(
    table_info* interval_table, table_info* point_table,
    cond_expr_fn_batch_t cond_func, uint64_t interval_start_col_id,
    uint64_t interval_end_col_id, uint64_t point_col_id, int curr_rank,
    int n_pes, bool interval_parallel, bool point_parallel, bool is_point_outer,
    bool is_strict_contained, bool is_strict_start_cond, bool point_left) {
    tracing::Event ev("interval_merge", interval_parallel || point_parallel);

    // When the point side is empty, the output will be empty regardless of
    // whether it's an inner join or a point-outer join.
    // Note that in the case that the interval side is empty, but the point
    // side is not, the output won't be empty in case of a point-outer join
    // (it'll be the point table plus nulls for all the columns from the
    // interval side).
    // This was added to avoid problems with batch_n_rows computation
    // (undefined behavior)
    if (point_table->nrows() == 0) {
        ev.add_attribute("out_num_rows", 0);
        ev.add_attribute("out_num_inner_rows", 0);
        ev.add_attribute("out_num_outer_rows", 0);
        ev.finalize();
        return std::pair(std::vector<int64_t>(), std::vector<int64_t>());
    }

    auto [interval_arr_infos, interval_col_data, interval_col_null] =
        get_gen_cond_data_ptrs(interval_table);
    auto [point_arr_infos, point_col_data, point_col_null] =
        get_gen_cond_data_ptrs(point_table);

    auto interval_arr_infos_ptr = interval_arr_infos.data();
    auto interval_col_data_ptr = interval_col_data.data();
    auto interval_col_null_ptr = interval_col_null.data();
    auto point_arr_infos_ptr = point_arr_infos.data();
    auto point_col_data_ptr = point_col_data.data();
    auto point_col_null_ptr = point_col_null.data();

    // Prebuild the condition function by partial application
    // Makes actual join loop simpler to read
    // TODO: Does this impact performance? Assuming C++ compiler
    //       can recognize and undo this during compilation
    std::function<void(int64_t, int64_t, int64_t, uint8_t*)> inner_cond_func;
    if (point_left) {
        inner_cond_func = [&cond_func, interval_arr_infos_ptr,
                           point_arr_infos_ptr, interval_col_data_ptr,
                           point_col_data_ptr, interval_col_null_ptr,
                           point_col_null_ptr](
                              int64_t interval_idx, int64_t point_start_idx,
                              int64_t point_end_idx, uint8_t* match_arr) {
            cond_func(point_arr_infos_ptr, interval_arr_infos_ptr,
                      point_col_data_ptr, interval_col_data_ptr,
                      point_col_null_ptr, interval_col_null_ptr, match_arr,
                      point_start_idx, point_end_idx, interval_idx,
                      interval_idx + 1 /*+1 so that it's not an empty loop*/);
        };
    } else {
        inner_cond_func = [&cond_func, interval_arr_infos_ptr,
                           point_arr_infos_ptr, interval_col_data_ptr,
                           point_col_data_ptr, interval_col_null_ptr,
                           point_col_null_ptr](
                              int64_t interval_idx, int64_t point_start_idx,
                              int64_t point_end_idx, uint8_t* match_arr) {
            cond_func(interval_arr_infos_ptr, point_arr_infos_ptr,
                      interval_col_data_ptr, point_col_data_ptr,
                      interval_col_null_ptr, point_col_null_ptr, match_arr,
                      interval_idx,
                      interval_idx + 1 /*+1 so that it's not an empty loop*/,
                      point_start_idx, point_end_idx);
        };
    }

    // Start Col of Interval Table and Point Col
    array_info* left_inter_col = interval_table->columns[interval_start_col_id];
    array_info* point_col = point_table->columns[point_col_id];

    // Rows of the Output Joined Table
    std::vector<int64_t> joined_interval_idxs;
    std::vector<int64_t> joined_point_idxs;

    // Bitmask indicating all matched rows in left table
    size_t n_bytes_point = is_point_outer ? (point_table->nrows() + 7) >> 3 : 0;
    std::vector<uint8_t> point_matched_rows(n_bytes_point, 0);

    // Set 500K batch size to make sure batch data of all cores fits in L3
    // cache.
    int64_t batch_size_bytes = 500 * 1024;
    char* batch_size = std::getenv("BODO_INTERVAL_JOIN_BATCH_SIZE");
    if (batch_size) {
        batch_size_bytes = std::stoi(batch_size);
    }
    if (batch_size_bytes <= 0) {
        throw std::runtime_error("interval_join_table: batch_size_bytes <= 0");
    }

    // Since we iterate on the point side (with the interval side constant),
    // we use the point table size for batch size calculation.
    // XXX We can technically do it based on the one column instead of
    // the whole table since we're guaranteed that only one column is involved
    // in the general join condition.
    uint64_t n_batches = (uint64_t)std::ceil(
        table_local_memory_size(point_table) / (double)batch_size_bytes);
    uint64_t batch_n_rows =
        (uint64_t)std::ceil(point_table->nrows() / (double)n_batches);
    uint64_t n_bytes_match = (batch_n_rows + 7) >> 3;
    uint8_t* match_arr = new uint8_t[n_bytes_match];

    uint64_t point_pos = 0;
    for (uint64_t interval_pos = 0; interval_pos < interval_table->nrows();
         interval_pos++) {
        // Find first row in the point table thats in the interval
        while (point_pos < point_table->nrows() &&
               !is_point_right_of_interval_start(left_inter_col, interval_pos,
                                                 point_col, point_pos,
                                                 is_strict_start_cond)) {
            point_pos++;
        }
        if (point_pos >= point_table->nrows()) break;

        // Because tables are sorted, a consecutive range of rows in point table
        // will fit in the interval.
        // Thus, we loop and match all until outside of interval. Then reset.
        // For best efficiency, we do this in batches. If during a batch, we
        // encounter any non-matches, we break out of the loop.
        for (uint64_t point_batch_start = point_pos;
             point_batch_start < point_table->nrows();
             point_batch_start += batch_n_rows) {
            uint64_t point_batch_end = std::min(
                point_batch_start + batch_n_rows, point_table->nrows());

            inner_cond_func(interval_pos, point_batch_start, point_batch_end,
                            match_arr);
            // Whether or not to break out of the loop. We can break as soon
            // as we see a not-matching point.
            bool found_not_match = false;
            int64_t match_ind = 0;
            for (uint64_t i = point_batch_start; i < point_batch_end; i++) {
                bool match = GetBit(match_arr, match_ind++);
                if (match) {
                    joined_interval_idxs.push_back(interval_pos);
                    joined_point_idxs.push_back(i);
                    if (is_point_outer) {
                        SetBitTo(point_matched_rows.data(), i, true);
                    }
                } else {
                    found_not_match = true;
                    break;
                }
            }
            if (found_not_match) {
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
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, bool is_left, bool is_right, bool is_left_point,
    bool strict_start, bool strict_end, uint64_t point_col_id,
    uint64_t interval_start_col_id, uint64_t interval_end_col_id,
    bool* key_in_output, int64_t* use_nullable_arr_type,
    cond_expr_fn_batch_t cond_func, uint64_t* cond_func_left_columns,
    uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    uint64_t cond_func_right_column_len, uint64_t* num_rows_ptr) {
    try {
        // TODO: Make this an assertion
        if ((is_left_point && is_right) || (!is_left_point && is_left)) {
            throw std::runtime_error(
                "Point-In-Interval Join should only support Inner or Left "
                "Joins");
        }

        // TODO Use broadcast join if one of the tables is very small (allgather
        // the small table)

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

        table_info* point_table = left_table;
        table_info* interval_table = right_table;
        bool point_table_parallel = left_parallel;
        bool interval_table_parallel = right_parallel;
        bool is_outer_point = is_left;
        if (!is_left_point) {
            point_table = right_table;
            interval_table = left_table;
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
                point_table, interval_table, point_table_parallel,
                interval_table_parallel, strict_contained);

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
            sorted_interval_table, sorted_point_table, cond_func,
            interval_start_col_id, interval_end_col_id, point_col_id, myrank,
            n_pes, interval_table_parallel, point_table_parallel,
            is_outer_point, strict_contained, strict_start, is_left_point);

        table_info *sorted_left_table, *sorted_right_table;
        std::vector<int64_t> left_idxs, right_idxs;
        if (is_left_point) {
            sorted_left_table = sorted_point_table;
            sorted_right_table = sorted_interval_table;
            left_idxs = point_idxs;
            right_idxs = interval_idxs;
        } else {
            sorted_left_table = sorted_interval_table;
            sorted_right_table = sorted_point_table;
            left_idxs = interval_idxs;
            right_idxs = point_idxs;
        }
        table_info* out_table = create_out_table(
            sorted_left_table, sorted_right_table, left_idxs, right_idxs,
            key_in_output, use_nullable_arr_type, cond_func_left_columns,
            cond_func_left_column_len, cond_func_right_columns,
            cond_func_right_column_len);

        // number of local output rows is passed to Python in case all output
        // columns are dead.
        *num_rows_ptr = out_table->nrows();
        ev.add_attribute("out_table_nrows", *num_rows_ptr);
        return out_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
