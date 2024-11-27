#include <algorithm>

#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_dict_builder.h"
#include "_distributed.h"
#include "_join.h"
#include "_join_hashing.h"
#include "_shuffle.h"

/**
 * @brief Compute the skew of a population where
 * each rank contains a single value. To do this we use the
 * Adjusted Fisher-Pearson Standardized Moment Coefficient
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1287290894/Investigation+Dealing+with+Skews+in+Joins#How-to-measure-skew?
 *
 * @param val The input integer value for this rank.
 * @return double The skew of the population.
 */
double compute_population_skew(int64_t val) {
    int64_t nranks = dist_get_size();
    std::vector<int64_t> vals_each_rank(nranks);
    // Gather the values on all ranks.
    c_gather_scalar(&val, vals_each_rank.data(), Bodo_CTypes::INT64, true, 0);
    // population skew G = sum( ((x_i - x_avg) / x_stddev)^3 )
    // Compute the components of the skew.
    // Compute the mean
    double sum =
        std::accumulate(vals_each_rank.begin(), vals_each_rank.end(), 0.0);
    double mean = sum / nranks;
    // Compute Xi - mu
    std::vector<double> diff(nranks);
    std::ranges::transform(vals_each_rank, diff.begin(),
                           [mean](double x) { return x - mean; });
    // Compute sum(xi^2)
    double sq_sum =
        std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    if (sq_sum == 0.0) {
        // If sq_sum is 0 stddev will be 0 so we have 0 skew.
        return 0.0;
    }
    // compute the standard deviation
    double stddev = std::sqrt(sq_sum / nranks);
    std::vector<double> elems(nranks);
    // Compute (Xi - mu) / stdev
    std::ranges::transform(diff, elems.begin(),
                           [stddev](double x) { return x / stddev; });
    // Compute elem^3
    std::vector<double> cubes(nranks);
    std::ranges::transform(elems, cubes.begin(),
                           [](double x) { return x * x * x; });
    // Compute skew = sum(cubes)
    double skew = std::accumulate(cubes.begin(), cubes.end(), 0.0);
    return skew;
}

/**
 * @brief Rebalance the output of a join if there is significant
 * skew across ranks. This is only called once the join has been
 * given a hint and the output is distributed. The original output
 * is freed by this function.
 *
 * @param original_output The original output of the join. If
 * there is not significant skew this will be returned.
 * @return std::shared_ptr<table_info> The table output after rebalancing the
 * ranks.
 */
std::shared_ptr<table_info> rebalance_join_output(
    std::shared_ptr<table_info> original_output) {
    tracing::Event ev("rebalance_join_output", true);
    // Communicate the number of rows on each rank.
    int64_t nrows = original_output->nrows();
    double skew = compute_population_skew(nrows);
    // Skew can be positive of negative. Here we only care about
    // the magnitude of the skew.
    skew = std::abs(skew);
    ev.add_attribute("g_skew", skew);
    double skew_threshold = 3.0;
    char* env_skew_threshold =
        std::getenv("BODO_JOIN_REBALANCE_SKEW_THRESHOLD");
    if (env_skew_threshold) {
        skew_threshold = std::stoi(env_skew_threshold);
    }
    if (skew > skew_threshold) {
        // 1.0 is a highly skewed population according to the internet,
        // but our relevant example with only 2 outlier has a skew of nearly
        // 2600. As a result we use 3.0 as the threshold for rebalancing to be
        // slightly more conservative until we have more data/have done further
        // testing.
        std::shared_ptr<table_info> out_table =
            shuffle_renormalization(std::move(original_output), 0, 0, true);
        return out_table;
    } else {
        return original_output;
    }
}

/**
 * @brief Validate the input to the equi_join_table function.
 *
 * @param left_table The left input table.
 * @param right_table The right input table.
 * @param n_key The number of key columns.
 * @param extra_data_col Is there an extra data column generated to handle an
 * extra key?
 */
void validate_equi_join_input(std::shared_ptr<table_info> left_table,
                              std::shared_ptr<table_info> right_table,
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
void equi_join_keys_handle_dict_encoded(std::shared_ptr<table_info> left_table,
                                        std::shared_ptr<table_info> right_table,
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
        std::shared_ptr<array_info> arr1 = left_table->columns[i];
        std::shared_ptr<array_info> arr2 = right_table->columns[i];
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
void insert_non_equi_func_set(bodo::unord_set_container<int64_t>* set,
                              std::shared_ptr<table_info> table,
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
            std::shared_ptr<array_info> arr = table->columns[col_num];
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
 * @return std::tuple<bodo::unord_set_container<int64_t>*,
 * bodo::unord_set_container<int64_t>*> A tuple of pointers to the sets of
 * column numbers that are used in the non-equality C funcs.
 */
std::tuple<bodo::unord_set_container<int64_t>*,
           bodo::unord_set_container<int64_t>*>
create_non_equi_func_sets(std::shared_ptr<table_info> left_table,
                          std::shared_ptr<table_info> right_table,
                          uint64_t* left_non_equi_func_col_nums,
                          uint64_t len_left_non_equi,
                          uint64_t* right_non_equi_func_col_nums,
                          uint64_t len_right_non_equi, bool left_parallel,
                          bool right_parallel, size_t n_key) {
    // Convert the left table.
    bodo::unord_set_container<int64_t>* left_non_equi_func_col_num_set =
        new bodo::unord_set_container<int64_t>();
    left_non_equi_func_col_num_set->reserve(len_left_non_equi);
    insert_non_equi_func_set(left_non_equi_func_col_num_set, left_table,
                             left_non_equi_func_col_nums, len_left_non_equi,
                             left_parallel, n_key);

    // Convert the right table.
    bodo::unord_set_container<int64_t>* right_non_equi_func_col_num_set =
        new bodo::unord_set_container<int64_t>();
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
 * @param groups A vector of row-ids considered equivalent. This is
 * used to compress rows with the same columns used in the non-equality
 * condition and the same keys. It's a single contiguous buffer where the
 * different groups can be identified using the offsets buffer. This function
 * will resize and fill this buffer.
 * @param groups_offsets A vector with offsets for the groups buffer.
 * It's size is num_groups + 1. This function will resize and fill these
 * offsets.
 * @param build_table_rows THe number of rows in the build table.
 */
template <typename Map>
void insert_build_table_equi_join_some_non_equality(
    Map* key_rows_map,
    bodo::vector<bodo::unord_map_container<
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>*
        second_level_hash_maps,
    joinHashFcts::SecondLevelHashHashJoinTable second_level_hash_fct,
    joinHashFcts::SecondLevelKeyEqualHashJoinTable second_level_equal_fct,
    bodo::vector<size_t>& groups, bodo::vector<size_t>& groups_offsets,
    size_t build_table_rows) {
    // Vector to store the size of the groups as we iterate over the build table
    // rows. We reserve build_table_rows to avoid expensive re-allocations.
    bodo::vector<uint64_t> num_rows_in_group;
    num_rows_in_group.reserve(build_table_rows);

    // We iterate over the build-table twice: first to calculate the number of
    // groups and their sizes, and then a second time to actually fill the
    // groups buffer. We do this to avoid expensive vector re-allocations.

    // If 'uses_cond_func' we have a separate insertion process. We
    // place the condition before the loop to avoid overhead.
    // Calculate the number of groups and their sizes and store them in
    // num_rows_in_group by iterating over the build table first:
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
            auto* group_map = new bodo::unord_map_container<
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
            second_level_group_id = num_rows_in_group.size() + 1;
            // Initialize group size to 0.
            num_rows_in_group.emplace_back(0);
        }
        // Increment the group row count:
        num_rows_in_group[second_level_group_id - 1]++;
    }

    // Number of groups is the same as the size of num_rows_in_group.
    size_t num_groups = num_rows_in_group.size();

    // Resize offsets vector based on the number of groups
    groups_offsets.resize(num_groups + 1);
    // First element should always be 0
    groups_offsets[0] = 0;
    // Do a cumulative sum and fill the rest:
    if (num_groups > 0) {
        std::partial_sum(num_rows_in_group.cbegin(), num_rows_in_group.cend(),
                         groups_offsets.begin() + 1);
    }

    // Resize based on how many total elements in all groups.
    groups.resize(groups_offsets[groups_offsets.size() - 1]);

    // Fill the groups vector. The hashmap(s) are already populated from the
    // first iteration, so we don't need to check if the row-ids exist.
    // Store counters for each group, so we can add the row-id in the correct
    // location.
    bodo::vector<size_t> group_fill_counter(num_groups, 0);
    for (size_t i_build = 0; i_build < build_table_rows; i_build++) {
        const size_t& first_level_group_id = (*key_rows_map)[i_build];
        auto group_map = (*second_level_hash_maps)[first_level_group_id - 1];
        const size_t& second_level_group_id = (*group_map)[i_build];
        size_t groups_idx = groups_offsets[second_level_group_id - 1] +
                            group_fill_counter[second_level_group_id - 1];
        groups[groups_idx] = i_build;
        group_fill_counter[second_level_group_id - 1]++;
    }
}

/**
 * @brief Insert the rows from the build table into the hash map in the case
 * where there is only an equality condition.
 *
 * @param key_rows_map Map from the set of keys to rows that share that key.
 * @param groups A vector of row-ids considered equivalent. This is
 * done to compress rows with common keys. It's a single contiguous buffer where
 * the different groups can be identified using the offsets buffer. This
 * function will resize and fill this buffer.
 * @param groups_offsets A vector with offsets for the groups buffer.
 * It's size is num_groups + 1. This function will resize and fill these
 * offsets.
 * @param build_table_rows The number of rows in the build table.
 */
template <typename Map>
void insert_build_table_equi_join_all_equality(
    Map* key_rows_map, bodo::vector<size_t>& groups,
    bodo::vector<size_t>& groups_offsets, size_t build_table_rows) {
    // Vector to store the size of the groups as we iterate over the build table
    // rows. We reserve build_table_rows to avoid expensive re-allocations.
    bodo::vector<uint64_t> num_rows_in_group;
    num_rows_in_group.reserve(build_table_rows);

    // We iterate over the build-table twice: first to calculate the number of
    // groups and their sizes, and then a second time to actually fill the
    // groups buffer. We do this to avoid expensive vector re-allocations.

    // Calculate the number of groups and their sizes and store them in
    // num_rows_in_group by iterating over the build table first:
    for (size_t i_build = 0; i_build < build_table_rows; i_build++) {
        // Check if the group already exists, if it doesn't this
        // will insert a value.
        size_t& group_id = (*key_rows_map)[i_build];
        // group_id==0 means key doesn't exist in map
        if (group_id == 0) {
            // Update the value of group_id stored in the hash map
            // as well since its pass by reference.
            group_id = num_rows_in_group.size() + 1;
            // Initialize group size to 0.
            num_rows_in_group.emplace_back(0);
        }
        // Increment count for the group
        num_rows_in_group[group_id - 1]++;
    }

    // Number of groups is the same as the size of num_rows_in_group.
    size_t num_groups = num_rows_in_group.size();

    // Resize offsets vector based on the number of groups
    groups_offsets.resize(num_groups + 1);
    // First element should always be 0
    groups_offsets[0] = 0;
    // Do a cumulative sum and fill the rest:
    if (num_groups > 0) {
        std::partial_sum(num_rows_in_group.cbegin(), num_rows_in_group.cend(),
                         groups_offsets.begin() + 1);
    }

    // Resize based on how many total elements in all groups.
    groups.resize(groups_offsets[groups_offsets.size() - 1]);

    // Fill the groups vector. The hashmap(s) are already populated from the
    // first iteration, so we don't need to check if the row-ids exist.
    // Store counters for each group, so we can add the row-id in the correct
    // location.
    bodo::vector<size_t> group_fill_counter(num_groups, 0);
    for (size_t i_build = 0; i_build < build_table_rows; i_build++) {
        // Guaranteed to find a match here:
        const size_t& group_id = (*key_rows_map)[i_build];
        size_t groups_idx =
            groups_offsets[group_id - 1] + group_fill_counter[group_id - 1];
        groups[groups_idx] = i_build;
        group_fill_counter[group_id - 1]++;
    }
}

/**
 * @brief Get the table size threshold (in bytes) for broadcast join
 *
 * @return int threshold value
 */
int get_bcast_join_threshold() {
    // We default to 10MB, which matches Spark, unless the user specifies a
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
 * @return std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>
 * , bool, bool> A tuple of the new left and right tables after any shuffling
 * and whether or not the left and right tables are replicated. A table will
 * become replicated if we broadcast it.
 */
std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>, bool, bool>
equi_join_shuffle(std::shared_ptr<table_info> left_table,
                  std::shared_ptr<table_info> right_table, bool is_left_outer,
                  bool is_right_outer, bool left_parallel, bool right_parallel,
                  int64_t n_pes, size_t n_key, const bool is_na_equal) {
    // Create a tracing event for the shuffle.
    tracing::Event ev("equi_join_table_shuffle",
                      left_parallel || right_parallel);

    // By default the work tables are the inputs.
    std::shared_ptr<table_info> work_left_table = left_table;
    std::shared_ptr<table_info> work_right_table = right_table;
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
        } else if (right_total_memory <= left_total_memory &&
                   right_total_memory < bcast_join_threshold &&
                   right_total_memory <
                       (left_total_memory / double(n_pes) *
                        global_build_to_local_probe_ratio_limit)) {
            // Broadcast the right table
            work_right_table = gather_table(right_table, -1, all_gather, true);
            right_replicated = true;
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
                std::vector<std::shared_ptr<array_info>> key_arrs_left(
                    left_table->columns.begin(),
                    left_table->columns.begin() + n_key);
                null_bitmask_keys_left =
                    bitwise_and_null_bitmasks(key_arrs_left, true);

                std::vector<std::shared_ptr<array_info>> key_arrs_right(
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
                CHECK_MPI(
                    MPI_Allreduce(&left_table_nrows, &left_table_global_nrows,
                                  1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD),
                    "equi_join_shuffle: MPI error on MPI_Allreduce [left "
                    "table]:");
                CHECK_MPI(
                    MPI_Allreduce(&right_table_nrows, &right_table_global_nrows,
                                  1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD),
                    "equi_join_shuffle: MPI error on MPI_Allreduce [right "
                    "table]:");
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
        // do this, we set a threshold K (CritQuotientNrows) and compare the
        // size of the table. If either table has K times more rows we will
        // always make the smaller table the table that populates the hash map,
        // regardless of if that table needs an outer join. If the ratio of rows
        // is less than K, then populate the hash map with the table that
        // doesn't require an outer join.
        //
        // TODO: Justify our threshold of 1 table being 6x larger than
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
    static inline void apply(bodo::vector<uint8_t>& V_build_map, size_t pos) {
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
    static inline void apply(bodo::vector<uint8_t>& V_build_map, size_t pos) {
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
    static inline void apply(bodo::vector<int64_t>& build_write_idxs,
                             bodo::vector<int64_t>& probe_write_idxs,
                             bodo::vector<uint8_t>& V_probe_map, size_t pos) {
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
    static inline void apply(bodo::vector<int64_t>& build_write_idxs,
                             bodo::vector<int64_t>& probe_write_idxs,
                             bodo::vector<uint8_t>& V_probe_map, size_t pos) {
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
    static inline void apply(bodo::vector<int64_t>& build_write_idxs,
                             bodo::vector<int64_t>& probe_write_idxs,
                             bodo::vector<uint8_t>& V_probe_map, size_t pos) {
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
 * @param[in] groups Vector of row-ids that have the same keys in the build
 * table. The groups are laid out contiguously in a single buffer, and can be
 * indexed using the 'groups_offsets' array.
 * @param[in] groups_offsets  A vector with offsets for the groups buffer.
 * It's size is num_groups + 1.
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
    bodo::vector<bodo::unord_map_container<
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>*
        second_level_hash_maps,
    const bodo::vector<size_t>& groups,
    const bodo::vector<size_t>& groups_offsets, size_t build_table_rows,
    size_t probe_table_rows, bodo::vector<uint8_t>& V_build_map,
    bodo::vector<uint8_t>& V_probe_map, bodo::vector<int64_t>& build_write_idxs,
    bodo::vector<int64_t>& probe_write_idxs, bool build_is_left,
    std::vector<std::shared_ptr<array_info>>& left_table_infos,
    std::vector<std::shared_ptr<array_info>>& right_table_infos,
    std::vector<void*>& col_ptrs_left, std::vector<void*>& col_ptrs_right,
    std::vector<void*>& null_bitmap_left, std::vector<void*>& null_bitmap_right,
    cond_expr_fn_t cond_func) {
    // get raw array_info pointers for cond_func
    std::vector<array_info*> left_table_info_ptrs, right_table_info_ptrs;
    for (std::shared_ptr<array_info> arr : left_table_infos) {
        left_table_info_ptrs.push_back(arr.get());
    }
    for (std::shared_ptr<array_info> arr : right_table_infos) {
        right_table_info_ptrs.push_back(arr.get());
    }

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
            bodo::unord_map_container<
                size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
                joinHashFcts::SecondLevelKeyEqualHashJoinTable>* group_map =
                (*second_level_hash_maps)[iter->second - 1];

            bool has_match = false;
            // Iterate over all of the keys and compare each group.
            // TODO [BE-1300]: Explore tsl:sparse_map
            for (auto& item : *group_map) {
                size_t pos = item.second - 1;
                // Find the start and end indices for this group in the groups
                // buffer using groups_offsets.
                size_t group_start_idx = groups_offsets[pos];
                size_t group_end_idx = groups_offsets[pos + 1];
                // Select a single member
                size_t cmp_row = groups[group_start_idx];
                size_t left_ind = 0;
                size_t right_ind = 0;
                if (build_is_left) {
                    left_ind = cmp_row;
                    right_ind = i_probe;
                } else {
                    left_ind = i_probe;
                    right_ind = cmp_row;
                }
                bool match = cond_func(
                    left_table_info_ptrs.data(), right_table_info_ptrs.data(),
                    col_ptrs_left.data(), col_ptrs_right.data(),
                    null_bitmap_left.data(), null_bitmap_right.data(), left_ind,
                    right_ind);
                if (match) {
                    // If our group matches, add every row and
                    // update the bitmap
                    handle_build_table_hit<build_table_outer>::apply(
                        V_build_map, pos);
                    has_match = true;
                    for (size_t idx = group_start_idx; idx < group_end_idx;
                         idx++) {
                        size_t j_build = groups[idx];
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
 * @param[in] groups Vector of row-ids that have the same keys in the build
 * table. The groups are laid out contiguously in a single buffer, and can be
 * indexed using the 'groups_offsets' array.
 * @param[in] groups_offsets  A vector with offsets for the groups buffer. It's
 * size is num_groups + 1.
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
    Map* key_rows_map, const bodo::vector<size_t>& groups,
    const bodo::vector<size_t>& groups_offsets, size_t build_table_rows,
    size_t probe_table_rows, bodo::vector<uint8_t>& V_build_map,
    bodo::vector<uint8_t>& V_probe_map, bodo::vector<int64_t>& build_write_idxs,
    bodo::vector<int64_t>& probe_write_idxs) {
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
            // Find the start and end indices for this group in the groups
            // buffer using groups_offsets
            size_t group_start_idx = groups_offsets[iter->second - 1];
            size_t group_end_idx = groups_offsets[iter->second - 1 + 1];
            size_t pos = iter->second - 1;
            handle_build_table_hit<build_table_outer>::apply(V_build_map, pos);
            for (size_t idx = group_start_idx; idx < group_end_idx; idx++) {
                size_t j_build = groups[idx];
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
    static inline void apply(size_t pos,
                             bodo::vector<int64_t>& build_write_idxs,
                             bodo::vector<int64_t>& probe_write_idxs,
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
    static inline void apply(size_t pos,
                             bodo::vector<int64_t>& build_write_idxs,
                             bodo::vector<int64_t>& probe_write_idxs,
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
 * @param[in] groups Vector rows in the build table, organized by groups in a
 * single contiguous buffer. The groups can be indexed using the
 * 'groups_offsets' array.
 * @param[in] groups_offsets  A vector with offsets for the groups buffer. It's
 * size is num_groups + 1. This function will resize and fill these offsets.
 * @param build_write_idxs The build table write indices.
 * @param probe_write_idxs The probe table write indices.
 * @param myrank The current rank.
 * @param n_pes The total number of processes.
 */
template <bool build_miss_needs_reduction>
void insert_build_table_misses(bodo::vector<uint8_t>& V_build_map,
                               const bodo::vector<size_t>& groups,
                               const bodo::vector<size_t>& groups_offsets,
                               bodo::vector<int64_t>& build_write_idxs,
                               bodo::vector<int64_t>& probe_write_idxs,
                               int64_t myrank, int64_t n_pes) {
    if (build_miss_needs_reduction) {
        // Perform the reduction on build table misses if necessary
        MPI_Allreduce_bool_or(V_build_map);
    }
    int64_t pos_build_disp = 0;
    // Add missing rows for outer joins when there are no matching build
    // table groups.
    // (Number of groups is groups_offsets->size() - 1)
    for (size_t pos = 0; pos < groups_offsets.size() - 1; pos++) {
        size_t group_start_idx = groups_offsets[pos];
        size_t group_end_idx = groups_offsets[pos + 1];
        bool bit = GetBit(V_build_map.data(), pos);
        if (!bit) {
            for (size_t idx = group_start_idx; idx < group_end_idx; idx++) {
                size_t j_build = groups[idx];
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
void insert_probe_table_broadcast_misses(
    bodo::vector<uint8_t>& V_probe_map, bodo::vector<int64_t>& build_write_idxs,
    bodo::vector<int64_t>& probe_write_idxs, size_t probe_table_rows,
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
    std::vector<uint8_t>& last_col_use_right,
    std::shared_ptr<table_info> work_left_table,
    std::shared_ptr<table_info> work_right_table, size_t n_tot_left,
    size_t n_tot_right, size_t n_key, int64_t* vect_same_key,
    bool* key_in_output,
    bodo::unord_set_container<int64_t>* left_cond_func_cols_set,
    bodo::unord_set_container<int64_t>* right_cond_func_cols_set,
    bool extra_data_col, bool is_join) {
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
            work_left_table->columns[i].reset();
        }
    }
    for (size_t i = 0; i < n_tot_right; i++) {
        if (last_col_use_right[i] == 0) {
            work_right_table->columns[i].reset();
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
    /*const*/ std::shared_ptr<table_info> work_left_table,
    /*const*/ std::shared_ptr<table_info> work_right_table,
    const size_t n_tot_left, const size_t n_tot_right,
    const bool uses_cond_func, const bool build_is_left,
    uint64_t* cond_func_left_columns, const uint64_t cond_func_left_column_len,
    uint64_t* cond_func_right_columns,
    const uint64_t cond_func_right_column_len, const bool parallel_trace,
    const size_t build_table_rows,
    const std::shared_ptr<table_info> build_table,
    const size_t probe_table_rows, const bool probe_miss_needs_reduction,
    Map* key_rows_map, bodo::vector<size_t>& groups,
    bodo::vector<size_t>& groups_offsets, const bool build_table_outer,
    const bool probe_table_outer, cond_expr_fn_t& cond_func,
    tracing::Event& ev_alloc_map,
    bodo::vector<bodo::unord_map_container<
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>*
        second_level_hash_maps,
    std::shared_ptr<uint32_t[]>& build_nonequal_key_hashes,
    bodo::vector<uint8_t>& V_build_map, bodo::vector<int64_t>& build_write_idxs,
    bodo::vector<uint8_t>& V_probe_map,
    bodo::vector<int64_t>& probe_write_idxs) {
    // Create a data structure containing the columns to match the format
    // expected by cond_func. We create two pairs of vectors, one with
    // the array_infos, which handle general types, and one with just data1
    // as a fast path for accessing numeric data. These include both keys
    // and data columns as either can be used in the cond_func.
    std::vector<std::shared_ptr<array_info>>& left_table_infos =
        work_left_table->columns;
    std::vector<std::shared_ptr<array_info>>& right_table_infos =
        work_right_table->columns;
    std::vector<void*> col_ptrs_left(n_tot_left);
    std::vector<void*> col_ptrs_right(n_tot_right);
    // Vectors for null bitmaps for fast null checking from the cfunc
    std::vector<void*> null_bitmap_left(n_tot_left);
    std::vector<void*> null_bitmap_right(n_tot_right);
    for (size_t i = 0; i < n_tot_left; i++) {
        col_ptrs_left[i] =
            static_cast<void*>(work_left_table->columns[i]->data1());
        null_bitmap_left[i] =
            static_cast<void*>(work_left_table->columns[i]->null_bitmask());
    }
    for (size_t i = 0; i < n_tot_right; i++) {
        col_ptrs_right[i] =
            static_cast<void*>(work_right_table->columns[i]->data1());
        null_bitmap_right[i] =
            static_cast<void*>(work_right_table->columns[i]->null_bitmask());
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
        .build_table = build_table,
        .build_data_key_cols = build_data_key_cols,
        .build_data_key_n_cols = build_data_key_n_cols};

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
            second_level_equal_fct, groups, groups_offsets, build_table_rows);
    } else {
        insert_build_table_equi_join_all_equality(
            key_rows_map, groups, groups_offsets, build_table_rows);
    }
    ev_groups.finalize();

    // Resize V_build_map based on the groups calculation.
    size_t n_bytes_build = 0;
    if (build_table_outer) {
        // Number of groups is groups_offsets->size() - 1
        n_bytes_build = (groups_offsets.size() - 1 + 7) >> 3;
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
                        groups_offsets, build_table_rows, probe_table_rows,
                        V_build_map, V_probe_map, build_write_idxs,
                        probe_write_idxs, build_is_left, left_table_infos,
                        right_table_infos, col_ptrs_left, col_ptrs_right,
                        null_bitmap_left, null_bitmap_right, cond_func);
                } else {
                    insert_probe_table_equi_join_some_non_equality<true, true,
                                                                   false, Map>(
                        key_rows_map, second_level_hash_maps, groups,
                        groups_offsets, build_table_rows, probe_table_rows,
                        V_build_map, V_probe_map, build_write_idxs,
                        probe_write_idxs, build_is_left, left_table_infos,
                        right_table_infos, col_ptrs_left, col_ptrs_right,
                        null_bitmap_left, null_bitmap_right, cond_func);
                }
            } else {
                insert_probe_table_equi_join_some_non_equality<true, false,
                                                               false, Map>(
                    key_rows_map, second_level_hash_maps, groups,
                    groups_offsets, build_table_rows, probe_table_rows,
                    V_build_map, V_probe_map, build_write_idxs,
                    probe_write_idxs, build_is_left, left_table_infos,
                    right_table_infos, col_ptrs_left, col_ptrs_right,
                    null_bitmap_left, null_bitmap_right, cond_func);
            }
        } else {
            if (probe_table_outer) {
                if (probe_miss_needs_reduction) {
                    insert_probe_table_equi_join_some_non_equality<false, true,
                                                                   true, Map>(
                        key_rows_map, second_level_hash_maps, groups,
                        groups_offsets, build_table_rows, probe_table_rows,
                        V_build_map, V_probe_map, build_write_idxs,
                        probe_write_idxs, build_is_left, left_table_infos,
                        right_table_infos, col_ptrs_left, col_ptrs_right,
                        null_bitmap_left, null_bitmap_right, cond_func);
                } else {
                    insert_probe_table_equi_join_some_non_equality<false, true,
                                                                   false, Map>(
                        key_rows_map, second_level_hash_maps, groups,
                        groups_offsets, build_table_rows, probe_table_rows,
                        V_build_map, V_probe_map, build_write_idxs,
                        probe_write_idxs, build_is_left, left_table_infos,
                        right_table_infos, col_ptrs_left, col_ptrs_right,
                        null_bitmap_left, null_bitmap_right, cond_func);
                }
            } else {
                insert_probe_table_equi_join_some_non_equality<false, false,
                                                               false, Map>(
                    key_rows_map, second_level_hash_maps, groups,
                    groups_offsets, build_table_rows, probe_table_rows,
                    V_build_map, V_probe_map, build_write_idxs,
                    probe_write_idxs, build_is_left, left_table_infos,
                    right_table_infos, col_ptrs_left, col_ptrs_right,
                    null_bitmap_left, null_bitmap_right, cond_func);
            }
        }
    } else {
        if (build_table_outer) {
            if (probe_table_outer) {
                if (probe_miss_needs_reduction) {
                    insert_probe_table_equi_join_all_equality<true, true, true,
                                                              Map>(
                        key_rows_map, groups, groups_offsets, build_table_rows,
                        probe_table_rows, V_build_map, V_probe_map,
                        build_write_idxs, probe_write_idxs);
                } else {
                    insert_probe_table_equi_join_all_equality<true, true, false,
                                                              Map>(
                        key_rows_map, groups, groups_offsets, build_table_rows,
                        probe_table_rows, V_build_map, V_probe_map,
                        build_write_idxs, probe_write_idxs);
                }
            } else {
                insert_probe_table_equi_join_all_equality<true, false, false,
                                                          Map>(
                    key_rows_map, groups, groups_offsets, build_table_rows,
                    probe_table_rows, V_build_map, V_probe_map,
                    build_write_idxs, probe_write_idxs);
            }
        } else {
            if (probe_table_outer) {
                if (probe_miss_needs_reduction) {
                    insert_probe_table_equi_join_all_equality<false, true, true,
                                                              Map>(
                        key_rows_map, groups, groups_offsets, build_table_rows,
                        probe_table_rows, V_build_map, V_probe_map,
                        build_write_idxs, probe_write_idxs);
                } else {
                    insert_probe_table_equi_join_all_equality<false, true,
                                                              false, Map>(
                        key_rows_map, groups, groups_offsets, build_table_rows,
                        probe_table_rows, V_build_map, V_probe_map,
                        build_write_idxs, probe_write_idxs);
                }
            } else {
                insert_probe_table_equi_join_all_equality<false, false, false,
                                                          Map>(
                    key_rows_map, groups, groups_offsets, build_table_rows,
                    probe_table_rows, V_build_map, V_probe_map,
                    build_write_idxs, probe_write_idxs);
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

std::shared_ptr<table_info> hash_join_table_inner(
    std::shared_ptr<table_info> left_table,
    std::shared_ptr<table_info> right_table, bool left_parallel,
    bool right_parallel, size_t n_keys, int64_t n_data_left_t,
    int64_t n_data_right_t, int64_t* vect_same_key, bool* key_in_output,
    int64_t* use_nullable_arr_type, bool is_left_outer, bool is_right_outer,
    bool is_join, bool extra_data_col, bool indicator, bool is_na_equal,
    bool rebalance_if_skewed, cond_expr_fn_t cond_func,
    uint64_t* cond_func_left_columns, uint64_t cond_func_left_column_len,
    uint64_t* cond_func_right_columns, uint64_t cond_func_right_column_len,
    uint64_t* num_rows_ptr) {
    // Does this join need an additional cond_func
    const bool uses_cond_func = cond_func != nullptr;
    const bool parallel_trace = (left_parallel || right_parallel);
    tracing::Event ev("hash_join_table", parallel_trace);
    // Reading the MPI settings
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Doing checks and basic assignments.
    size_t n_data_left = size_t(n_data_left_t);
    size_t n_data_right = size_t(n_data_right_t);
    size_t n_tot_left = n_keys + n_data_left;
    size_t n_tot_right = n_keys + n_data_right;
    // Check that the input is valid.
    // This ensures that the array type and dtype of key columns are the same
    // in both tables. This is an assumption we will use later in the code.
    validate_equi_join_input(left_table, right_table, n_keys, extra_data_col);

    if (ev.is_tracing()) {
        ev.add_attribute("in_left_table_nrows",
                         static_cast<size_t>(left_table->nrows()));
        ev.add_attribute("in_right_table_nrows",
                         static_cast<size_t>(right_table->nrows()));
        ev.add_attribute("g_left_parallel", left_parallel);
        ev.add_attribute("g_right_parallel", right_parallel);
        ev.add_attribute("g_n_key", n_keys);
        ev.add_attribute("g_n_data_cols_left", n_data_left_t);
        ev.add_attribute("g_n_data_cols_right", n_data_right_t);
        ev.add_attribute("g_is_left", is_left_outer);
        ev.add_attribute("g_is_right", is_right_outer);
        ev.add_attribute("g_extra_data_col", extra_data_col);
        ev.add_attribute("g_rebalance_if_skewed", rebalance_if_skewed);
    }
    // Handle dict encoding
    equi_join_keys_handle_dict_encoded(left_table, right_table, left_parallel,
                                       right_parallel, n_keys);

    auto [left_cond_func_cols_set, right_cond_func_cols_set] =
        create_non_equi_func_sets(
            left_table, right_table, cond_func_left_columns,
            cond_func_left_column_len, cond_func_right_columns,
            cond_func_right_column_len, left_parallel, right_parallel, n_keys);

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
                          n_keys, is_na_equal);
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
    std::shared_ptr<uint32_t[]> hashes_left = hash_keys_table(
        work_left_table, n_keys, SEED_HASH_JOIN, parallel_trace);
    std::shared_ptr<uint32_t[]> hashes_right = hash_keys_table(
        work_right_table, n_keys, SEED_HASH_JOIN, parallel_trace);

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
    std::shared_ptr<table_info> build_table, probe_table;
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
        .short_table_rows = build_table_rows,
        .short_table_hashes = build_table_hashes,
        .long_table_hashes = probe_table_hashes};

    // 'groups' is single contiguous buffer of row ids arranged by groups.
    // 'group_offsets' store the offsets for the individual groups within the
    // 'groups' buffer (similar to how we store strings in array_info). We will
    // resize these to the exact required sizes when inserting the build table
    // into the hashmap.
    std::unique_ptr<bodo::vector<size_t>> groups =
        std::make_unique<bodo::vector<size_t>>();
    std::unique_ptr<bodo::vector<size_t>> groups_offsets =
        std::make_unique<bodo::vector<size_t>>();
    std::shared_ptr<uint32_t[]> build_nonequal_key_hashes =
        std::shared_ptr<uint32_t[]>(nullptr);
    bodo::vector<bodo::unord_map_container<
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>*
        second_level_hash_maps = nullptr;

    // build_write_idxs and probe_write_idxs are used for the output.
    // It precises the index used for the writing of the output table
    // from the build and probe table.
    bodo::vector<int64_t> build_write_idxs, probe_write_idxs;

    // Allocate the vector for any build misses.
    // Start off as empty since it will be resized after the groups are
    // calculated.
    bodo::vector<uint8_t> V_build_map(0);

    // V_probe_map and V_build_map takes similar roles.
    // They indicate if an entry in the build or probe table
    // has been matched to the other side.
    // This is needed only if said table is replicated
    // (i.e probe_miss_needs_reduction/build_miss_needs_reduction).
    // Start off as empty, since it will be re-sized later.
    bodo::vector<uint8_t> V_probe_map(0);

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
    using unordered_map_t = bodo::unord_map_container<                         \
        size_t, size_t, joinHashFcts::HashHashJoinTable, JOIN_KEY_TYPE>;       \
    unordered_map_t* key_rows_map = new bodo::unord_map_container<             \
        size_t, size_t, joinHashFcts::HashHashJoinTable, JOIN_KEY_TYPE>(       \
        {}, hash_fct, equal_fct);                                              \
    /* reserving space is very important to avoid expensive reallocations      \
     * (at the cost of using more memory)                                      \
     * [BE-1078]: Should this use statistics to influence how much we          \
     * reserve?                                                                \
     */                                                                        \
    key_rows_map->reserve(build_table_rows);                                   \
    /* Define additional information needed for non-equality conditions,       \
     * determined by 'uses_cond_func'. We need a vector of hash maps for the   \
     * build table groups. In addition, we also need hashes for the build      \
     * table on all columns that are not since they will insert into the       \
     * hash map.                                                               \
     */                                                                        \
    second_level_hash_maps = new bodo::vector<bodo::unord_map_container<       \
        size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,            \
        joinHashFcts::SecondLevelKeyEqualHashJoinTable>*>();                   \
    hash_join_compute_tuples_helper(                                           \
        work_left_table, work_right_table, n_tot_left, n_tot_right,            \
        uses_cond_func, build_is_left, cond_func_left_columns,                 \
        cond_func_left_column_len, cond_func_right_columns,                    \
        cond_func_right_column_len, parallel_trace, build_table_rows,          \
        build_table, probe_table_rows, probe_miss_needs_reduction,             \
        key_rows_map, *groups, *groups_offsets, build_table_outer,             \
        probe_table_outer, cond_func, ev_alloc_map, second_level_hash_maps,    \
        build_nonequal_key_hashes, V_build_map, build_write_idxs, V_probe_map, \
        probe_write_idxs);                                                     \
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
#define EQ_JOIN_IMPL_ELSE_CASE                                                \
    using JoinKeyType = joinHashFcts::KeyEqualHashJoinTable;                  \
    JoinKeyType equal_fct{build_table_rows, n_keys, build_table, probe_table, \
                          is_na_equal};                                       \
    EQ_JOIN_IMPL_COMMON(JoinKeyType);
#endif

    // Use faster specialized implementation for common 1 key cases.
    if (n_keys == 1) {
        std::shared_ptr<array_info> build_arr = build_table->columns[0];
        bodo_array_type::arr_type_enum build_arr_type = build_arr->arr_type;
        Bodo_CTypes::CTypeEnum build_dtype = build_arr->dtype;
        std::shared_ptr<array_info> probe_arr = probe_table->columns[0];
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
    } else if (n_keys == 2) {
        std::shared_ptr<array_info> build_arr1 = build_table->columns[0];
        std::shared_ptr<array_info> build_arr2 = build_table->columns[1];
        bodo_array_type::arr_type_enum build_arr_type1 = build_arr1->arr_type;
        Bodo_CTypes::CTypeEnum build_dtype1 = build_arr1->dtype;
        bodo_array_type::arr_type_enum build_arr_type2 = build_arr2->arr_type;
        Bodo_CTypes::CTypeEnum build_dtype2 = build_arr2->dtype;

        std::shared_ptr<array_info> probe_arr1 = probe_table->columns[0];
        std::shared_ptr<array_info> probe_arr2 = probe_table->columns[1];
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
            insert_build_table_misses<true>(V_build_map, *groups,
                                            *groups_offsets, build_write_idxs,
                                            probe_write_idxs, myrank, n_pes);
        } else {
            insert_build_table_misses<false>(V_build_map, *groups,
                                             *groups_offsets, build_write_idxs,
                                             probe_write_idxs, myrank, n_pes);
        }
    }

    tracing::Event ev_clear_groups("dealloc_groups", parallel_trace);
    // Free the groups buffer now that they are no longer needed.
    groups.reset();
    groups_offsets.reset();

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
    std::vector<std::shared_ptr<array_info>> out_arrs;
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
        work_right_table, n_tot_left, n_tot_right, n_keys, vect_same_key,
        key_in_output, left_cond_func_cols_set, right_cond_func_cols_set,
        extra_data_col, is_join);

    // Determine the number of rows in your local chunk of the output.
    // This is passed to Python in case all columns are dead.
    uint64_t num_rows = probe_write_idxs.size();

    // Construct the output tables. This merges the results in the left and
    // right tables. We resume using work_left_table and work_right_table to
    // ensure we match the expected column order (as opposed to build/probe).

    // Inserting the optional column in the case of merging on column and
    // an index.
    if (extra_data_col) {
        tracing::Event ev_fill_optional("fill_extra_data_col", parallel_trace);
        size_t i = 0;
        std::shared_ptr<array_info> left_arr = work_left_table->columns[i];
        std::shared_ptr<array_info> right_arr = work_right_table->columns[i];
        if (build_is_left) {
            out_arrs.push_back(RetrieveArray_TwoColumns(
                left_arr, right_arr, build_write_idxs, probe_write_idxs));
        } else {
            out_arrs.push_back(RetrieveArray_TwoColumns(
                right_arr, left_arr, build_write_idxs, probe_write_idxs));
        }
        // After adding the optional collect decref the original values
        if (last_col_use_left[i] == 1) {
            work_left_table->columns[i].reset();
        }
        if (last_col_use_right[i] == 1) {
            work_right_table->columns[i].reset();
        }
    }

    // Where to check in key_in_output if we have a key or
    // a general merge condition.
    offset_t key_in_output_idx = 0;

    // Inserting the Left side of the table
    tracing::Event ev_fill_left("fill_left", parallel_trace);
    int idx = 0;

    // Insert the key columns
    for (size_t i = 0; i < n_keys; i++) {
        // Check if this key column is included.
        if (key_in_output[key_in_output_idx++]) {
            if (vect_same_key[i] == 1) {
                // If we are merging the same in two table then
                // additional NaNs cannot happen.
                std::shared_ptr<array_info> left_arr =
                    work_left_table->columns[i];
                std::shared_ptr<array_info> right_arr =
                    work_right_table->columns[i];
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
                    work_left_table->columns[i].reset();
                }
                if (last_col_use_right[i] == 2) {
                    work_right_table->columns[i].reset();
                }
            } else {
                // We are just inserting the keys so converting to a nullable
                // array depends on the type of join.
                std::shared_ptr<array_info> left_arr =
                    work_left_table->columns[i];
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
                    work_left_table->columns[i].reset();
                }
            }
            // update the output indices
            idx++;
        }
    }
    // Insert the data columns
    for (size_t i = n_keys; i < n_tot_left; i++) {
        if (left_cond_func_cols_set->contains(i) &&
            !key_in_output[key_in_output_idx++]) {
            // If this data column is used in the non equality
            // condition we have to check if its in the output.
            continue;
        }
        std::shared_ptr<array_info> left_arr = work_left_table->columns[i];
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
            work_left_table->columns[i].reset();
        }
    }
    // Delete to free memory
    delete left_cond_func_cols_set;
    ev_fill_left.finalize();

    // Inserting the right side of the table.
    tracing::Event ev_fill_right("fill_right", parallel_trace);

    // Insert right keys
    for (size_t i = 0; i < n_keys; i++) {
        // vect_same_key[i] == 0 means this key doesn't become a natural
        // join and get clobbered. The !is_join is unclear but seems to be
        // because DataFrame.join always merged with the index of the right
        // table, so this could be a bug in handling index cases.
        if (vect_same_key[i] == 0 && !is_join) {
            // If we might insert a key we have to verify it is in the output.
            if (key_in_output[key_in_output_idx++]) {
                std::shared_ptr<array_info> right_arr =
                    work_right_table->columns[i];
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
                    work_right_table->columns[i].reset();
                }
            }
        }
    }
    // Insert the data columns
    for (size_t i = n_keys; i < n_tot_right; i++) {
        if (right_cond_func_cols_set->contains(i) &&
            !key_in_output[key_in_output_idx++]) {
            // If this data column is used in the non equality
            // condition we have to check if its in the output.
            continue;
        }
        std::shared_ptr<array_info> right_arr = work_right_table->columns[i];
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
            work_right_table->columns[i].reset();
        }
    }

    // Delete to free memory
    delete right_cond_func_cols_set;
    ev_fill_right.finalize();

    // Create indicator column if indicator=True
    if (indicator) {
        tracing::Event ev_indicator("create_indicator", parallel_trace);
        std::shared_ptr<array_info> indicator_col = alloc_array_top_level(
            num_rows, -1, -1, bodo_array_type::CATEGORICAL, Bodo_CTypes::INT8,
            -1, 0, 3);
        for (size_t rownum = 0; rownum < num_rows; rownum++) {
            // Determine the source of each row. At most 1 value can be -1.
            // Whichever value is -1, other table is the source of the row.
            bodo::vector<int64_t>* left_write_idxs;
            bodo::vector<int64_t>* right_write_idxs;
            if (build_is_left) {
                left_write_idxs = &build_write_idxs;
                right_write_idxs = &probe_write_idxs;
            } else {
                left_write_idxs = &probe_write_idxs;
                right_write_idxs = &build_write_idxs;
            }
            // Left is null
            if ((*left_write_idxs)[rownum] == -1) {
                indicator_col->data1<bodo_array_type::CATEGORICAL>()[rownum] =
                    1;
                // Right is null
            } else if ((*right_write_idxs)[rownum] == -1) {
                indicator_col->data1<bodo_array_type::CATEGORICAL>()[rownum] =
                    0;
                // Neither is null
            } else {
                indicator_col->data1<bodo_array_type::CATEGORICAL>()[rownum] =
                    2;
            }
        }
        out_arrs.emplace_back(indicator_col);
        ev_indicator.finalize();
    }

    // Only return a table if there is at least 1
    // output column.
    if (out_arrs.size() == 0) {
        // Update the length in case its needed.
        *num_rows_ptr = num_rows;
        return nullptr;
    }

    std::shared_ptr<table_info> out_table =
        std::make_shared<table_info>(out_arrs);

    // Check for skew if BodoSQL suggested we should
    if (rebalance_if_skewed && (left_parallel || right_parallel)) {
        out_table = rebalance_join_output(out_table);
    }

    *num_rows_ptr = out_table->nrows();

    return out_table;
}

table_info* hash_join_table(
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, int64_t n_keys, int64_t n_data_left_t,
    int64_t n_data_right_t, int64_t* vect_same_key, bool* key_in_output,
    int64_t* use_nullable_arr_type, bool is_left_outer, bool is_right_outer,
    bool is_join, bool extra_data_col, bool indicator, bool is_na_equal,
    bool rebalance_if_skewed, cond_expr_fn_t cond_func,
    uint64_t* cond_func_left_columns, uint64_t cond_func_left_column_len,
    uint64_t* cond_func_right_columns, uint64_t cond_func_right_column_len,
    uint64_t* num_rows_ptr) {
    try {
        std::shared_ptr<table_info> out_table = hash_join_table_inner(
            std::shared_ptr<table_info>(left_table),
            std::shared_ptr<table_info>(right_table), left_parallel,
            right_parallel, n_keys, n_data_left_t, n_data_right_t,
            vect_same_key, key_in_output, use_nullable_arr_type, is_left_outer,
            is_right_outer, is_join, extra_data_col, indicator, is_na_equal,
            rebalance_if_skewed, cond_func, cond_func_left_columns,
            cond_func_left_column_len, cond_func_right_columns,
            cond_func_right_column_len, num_rows_ptr);
        // join returns nullptr if output table is empty
        if (out_table == nullptr) {
            return nullptr;
        }
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
