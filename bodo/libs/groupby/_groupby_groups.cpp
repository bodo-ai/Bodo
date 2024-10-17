// Copyright (C) 2023 Bodo Inc. All rights reserved.

#include "_groupby_groups.h"
#include "../_bodo_common.h"
#include "../_dict_builder.h"
#include "_groupby_hashing.h"

/**
 * This file contains the functions that are responsible for organizing
 * the rows into groups or providing information about accessing the groups.
 *
 */

// Main Groupby Code

/**
 * The main get_group_info loop which populates a grouping_info structure
 * (map rows from input to their group number, and store the first input row
 * for each group).
 *
 * @tparam Map The hashmap type to use for tracking groups.
 * @param[in,out] key_to_group: The hash map used to populate the grouping_info
 * structure, maps row index from input data to group numbers (the group numbers
 * in the map are the real group number + 1)
 * @param[in] key_cols: key columns
 * @param[in,out] grp_info: The grouping_info structure that we are populating
 * @param key_drop_nulls : whether to drop null keys
 * @param nrows : number of input rows
 * @param is_parallel: true if data is distributed
 */
template <typename T>
static void get_group_info_loop(
    T& key_to_group, const std::vector<std::shared_ptr<array_info>>& key_cols,
    grouping_info& grp_info, const bool key_drop_nulls, const int64_t nrows,
    bool is_parallel) {
    tracing::Event ev("get_group_info_loop", is_parallel);
    bodo::vector<int64_t>& group_to_first_row = grp_info.group_to_first_row;
    bodo::vector<int64_t>& row_to_group = grp_info.row_to_group;
    // Start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values recorded in grp_info go from
    // 0 to num_groups - 1)
    int64_t next_group = 1;
    // There are two versions of the loop because a `if (key_drop_nulls)`
    // branch inside the loop has a bit of a performance hit and
    // `get_group_info_loop` is one of the most expensive computations.
    // To not duplicate code, we put the common portion of the loop in
    // MAIN_LOOP_BODY macro

#define MAIN_LOOP_BODY                                                      \
    int64_t& group = key_to_group[i]; /* this inserts 0 into the map if key \
                                         doesn't exist */                   \
    if (group == 0) {                                                       \
        group = next_group++; /* this updates the value in the map without  \
                                 another lookup */                          \
        group_to_first_row.emplace_back(i);                                 \
    }                                                                       \
    row_to_group[i] = group - 1

    if (!key_drop_nulls) {
        for (int64_t i = 0; i < nrows; i++) {
            MAIN_LOOP_BODY;
        }
    } else {
        for (int64_t i = 0; i < nrows; i++) {
            if (does_row_has_nulls(key_cols, i)) {
                row_to_group[i] = -1;
                continue;
            }
            MAIN_LOOP_BODY;
        }
    }
}
#undef MAIN_LOOP_BODY

/**
 * @brief Get the group information for each row in
 * the input table.
 *
 * @tparam Map The hashmap type to use for tracking groups.
 * @param[in, out] key_to_group The map to fill that maps each
 * key set to a group number.
 * @param ev Tracing event for tracking relevant attributes.
 * @param[out] grp_info Grouping object that contains the key results.
 * @param[in] table The input table.
 * @param[in] key_cols Vector of key columns.
 * @param[in, out] hashes The hashes of the groups. This may be modified
 * in this function to save memory.
 * @param nunique_hashes Estimated number of unique hashes. Used for
 * allocations.
 * @param check_for_null_keys Should we check if the keys contain null values?
 * @param key_dropna Do we need to drop null entries?
 * @param load_factor Load factor used for the hashmap.
 * @param is_parallel Is the implementation parallel?
 */
template <typename Map>
void get_group_info_impl(
    Map& key_to_group, tracing::Event& ev, grouping_info& grp_info,
    std::shared_ptr<table_info> const table,
    const std::vector<std::shared_ptr<array_info>>& key_cols,
    std::shared_ptr<uint32_t[]>& hashes, const size_t nunique_hashes,
    const bool check_for_null_keys, const bool key_dropna,
    const double load_factor, bool is_parallel) {
    tracing::Event ev_alloc("get_group_info_alloc", is_parallel);
    key_to_group.max_load_factor(load_factor);
    key_to_group.reserve(nunique_hashes);
    ev_alloc.add_attribute("map bucket_count", key_to_group.bucket_count());
    ev_alloc.add_attribute("map max_load_factor",
                           key_to_group.max_load_factor());

    // XXX Don't want to initialize data (just reserve and set size) but
    // std::vector doesn't support this. Since we know the size of the
    // vector, doing this is better than reserving and then doing
    // emplace_backs in get_group_info_loop, which are slower than []
    // operator
    grp_info.row_to_group.resize(table->nrows());
    grp_info.group_to_first_row.reserve(nunique_hashes * 1.1);
    ev_alloc.finalize();

    const bool key_is_nullable =
        check_for_null_keys ? does_keys_have_nulls(key_cols) : false;
    const bool key_drop_nulls = key_is_nullable && key_dropna;
    ev.add_attribute("g_key_is_nullable", key_is_nullable);
    ev.add_attribute("g_key_dropna", key_dropna);
    ev.add_attribute("g_key_drop_nulls", key_drop_nulls);

    get_group_info_loop<Map>(key_to_group, key_cols, grp_info, key_drop_nulls,
                             table->nrows(), is_parallel);
    grp_info.num_groups = grp_info.group_to_first_row.size();
    ev.add_attribute("num_groups", static_cast<size_t>(grp_info.num_groups));
    do_map_dealloc<true>(hashes, key_to_group, is_parallel);
}

void get_group_info(std::vector<std::shared_ptr<table_info>>& tables,
                    std::shared_ptr<uint32_t[]>& hashes, size_t nunique_hashes,
                    std::vector<grouping_info>& grp_infos, const int64_t n_keys,
                    bool check_for_null_keys, bool key_dropna, bool is_parallel,
                    bodo::IBufferPool* const pool) {
    tracing::Event ev("get_group_info", is_parallel);
    if (tables.size() != 1) {
        throw std::runtime_error("get_group_info: expected 1 table input");
    }
    std::shared_ptr<table_info> table = tables[0];
    ev.add_attribute("input_table_nrows", static_cast<size_t>(table->nrows()));
    std::vector<std::shared_ptr<array_info>> key_cols =
        std::vector<std::shared_ptr<array_info>>(
            table->columns.begin(), table->columns.begin() + n_keys);

    if (!hashes) {
        hashes = bodo::make_shared_arr<uint32_t>(table->nrows(), pool);
        hash_keys(hashes.get(), key_cols, SEED_HASH_GROUPBY_SHUFFLE,
                  is_parallel);
        nunique_hashes =
            get_nunique_hashes(hashes, table->nrows(), is_parallel);
    }
    ev.add_attribute("nunique_hashes_est", nunique_hashes);
    grp_infos.emplace_back(pool);
    grouping_info& grp_info = grp_infos.back();

    HashLookupIn32bitTable hash_fct{hashes};

    // use faster specialized implementation for common 1 key cases
    if (n_keys == 1) {
        std::shared_ptr<array_info> arr = table->columns[0];

        bodo_array_type::arr_type_enum arr_type = arr->arr_type;
        Bodo_CTypes::CTypeEnum dtype = arr->dtype;

        // macro to reduce code duplication
#ifndef GROUPBY_INFO_IMPL_1_KEY
#define GROUPBY_INFO_IMPL_1_KEY(ARRAY_TYPE, DTYPE)                       \
    if (arr_type == ARRAY_TYPE && dtype == DTYPE) {                      \
        using KeyType = KeysEqualComparatorOneKey<ARRAY_TYPE, DTYPE,     \
                                                  /*is_na_equal=*/true>; \
        KeyType equal_fct{arr};                                          \
        using rh_flat_t =                                                \
            bodo::unord_map_container<int64_t, int64_t,                  \
                                      HashLookupIn32bitTable, KeyType>;  \
        rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct, pool);   \
        get_group_info_impl(key_to_group_rh_flat, ev, grp_info, table,   \
                            key_cols, hashes, nunique_hashes,            \
                            check_for_null_keys, key_dropna,             \
                            UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel); \
        return;                                                          \
    }
#endif
        GROUPBY_INFO_IMPL_1_KEY(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT32);
        GROUPBY_INFO_IMPL_1_KEY(bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT64);
        GROUPBY_INFO_IMPL_1_KEY(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME);
        GROUPBY_INFO_IMPL_1_KEY(bodo_array_type::DICT, Bodo_CTypes::STRING);
    }

    // use faster specialized implementation for common 2 key cases
    if (n_keys == 2) {
        std::shared_ptr<array_info> arr1 = table->columns[0];
        std::shared_ptr<array_info> arr2 = table->columns[1];
        bodo_array_type::arr_type_enum arr_type1 = arr1->arr_type;
        Bodo_CTypes::CTypeEnum dtype1 = arr1->dtype;
        bodo_array_type::arr_type_enum arr_type2 = arr2->arr_type;
        Bodo_CTypes::CTypeEnum dtype2 = arr2->dtype;

        // macro to reduce code duplication
#ifndef GROUPBY_INFO_IMPL_2_KEYS
#define GROUPBY_INFO_IMPL_2_KEYS(ARRAY_TYPE1, DTYPE1, ARRAY_TYPE2, DTYPE2) \
    if (arr_type1 == ARRAY_TYPE1 && dtype1 == DTYPE1 &&                    \
        arr_type2 == ARRAY_TYPE2 && dtype2 == DTYPE2) {                    \
        using KeyType =                                                    \
            KeysEqualComparatorTwoKeys<ARRAY_TYPE1, DTYPE1, ARRAY_TYPE2,   \
                                       DTYPE2, /*is_na_equal=*/true>;      \
        KeyType equal_fct{arr1, arr2};                                     \
        using rh_flat_t =                                                  \
            bodo::unord_map_container<int64_t, int64_t,                    \
                                      HashLookupIn32bitTable, KeyType>;    \
        rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct, pool);     \
        get_group_info_impl(key_to_group_rh_flat, ev, grp_info, table,     \
                            key_cols, hashes, nunique_hashes,              \
                            check_for_null_keys, key_dropna,               \
                            UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);   \
        return;                                                            \
    }
#endif

        // int32/(int32, int64, datetime, dict-encoded)
        GROUPBY_INFO_IMPL_2_KEYS(
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT32,
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT32);
        GROUPBY_INFO_IMPL_2_KEYS(
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT32,
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT64);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT32, bodo_array_type::NUMPY,
                                 Bodo_CTypes::DATETIME);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT32, bodo_array_type::DICT,
                                 Bodo_CTypes::STRING);
        // int64/(int32, int64, datetime, dict-encoded)
        GROUPBY_INFO_IMPL_2_KEYS(
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT64,
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT32);
        GROUPBY_INFO_IMPL_2_KEYS(
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT64,
            bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT64);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT64, bodo_array_type::NUMPY,
                                 Bodo_CTypes::DATETIME);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT64, bodo_array_type::DICT,
                                 Bodo_CTypes::STRING);
        // datetime/(int32, int64, datetime, dict-encoded)
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                                 bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT32);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                                 bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT64);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                                 bodo_array_type::NUMPY, Bodo_CTypes::DATETIME);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME,
                                 bodo_array_type::DICT, Bodo_CTypes::STRING);
        // dict-encoded/(int32, int64, datetime, dict-encoded)
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::DICT, Bodo_CTypes::STRING,
                                 bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT32);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::DICT, Bodo_CTypes::STRING,
                                 bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT64);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::DICT, Bodo_CTypes::STRING,
                                 bodo_array_type::NUMPY, Bodo_CTypes::DATETIME);
        GROUPBY_INFO_IMPL_2_KEYS(bodo_array_type::DICT, Bodo_CTypes::STRING,
                                 bodo_array_type::DICT, Bodo_CTypes::STRING);
#undef GROUPBY_INFO_IMPL_2_KEYS
    }

    // Use a specialized implementation for the case where all keys have the
    // same array and data types. We set `check_hash_first` to true if there are
    // 4 or more keys.
    // Tested using 'bodo/tests/test_groupby.py::test_many_same_type_keys'.
    bool all_keys_same_type = true;
    bodo_array_type::arr_type_enum arr_type = table->columns[0]->arr_type;
    Bodo_CTypes::CTypeEnum dtype = table->columns[0]->dtype;
    for (int64_t i = 1; i < n_keys; i++) {
        all_keys_same_type &= ((table->columns[i]->arr_type == arr_type) &&
                               (table->columns[i]->dtype == dtype));
    }

    if (all_keys_same_type) {
        std::vector<std::shared_ptr<array_info>> key_arrs(n_keys);
        for (int64_t i = 0; i < n_keys; i++) {
            key_arrs[i] = table->columns[i];
        }

#ifndef GROUPBY_INFO_IMPL_ALL_SAME_KEY_TYPES
#define GROUPBY_INFO_IMPL_ALL_SAME_KEY_TYPES(ARRAY_TYPE, DTYPE)             \
    if (arr_type == ARRAY_TYPE && dtype == DTYPE && n_keys >= 4) {          \
        using KeyType =                                                     \
            KeysEqualComparatorAllSameTypeKeys<ARRAY_TYPE, DTYPE,           \
                                               /*is_na_equal=*/true,        \
                                               /*check_hash_first=*/true>;  \
        KeyType equal_fct{key_arrs, hashes};                                \
        using rh_flat_t =                                                   \
            bodo::unord_map_container<int64_t, int64_t,                     \
                                      HashLookupIn32bitTable, KeyType>;     \
        rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct, pool);      \
        get_group_info_impl(key_to_group_rh_flat, ev, grp_info, table,      \
                            key_cols, hashes, nunique_hashes,               \
                            check_for_null_keys, key_dropna,                \
                            UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);    \
        return;                                                             \
    }                                                                       \
    if (arr_type == ARRAY_TYPE && dtype == DTYPE) {                         \
        using KeyType =                                                     \
            KeysEqualComparatorAllSameTypeKeys<ARRAY_TYPE, DTYPE,           \
                                               /*is_na_equal=*/true,        \
                                               /*check_hash_first=*/false>; \
        KeyType equal_fct{key_arrs, hashes};                                \
        using rh_flat_t =                                                   \
            bodo::unord_map_container<int64_t, int64_t,                     \
                                      HashLookupIn32bitTable, KeyType>;     \
        rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct, pool);      \
        get_group_info_impl(key_to_group_rh_flat, ev, grp_info, table,      \
                            key_cols, hashes, nunique_hashes,               \
                            check_for_null_keys, key_dropna,                \
                            UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);    \
        return;                                                             \
    }
#endif

        GROUPBY_INFO_IMPL_ALL_SAME_KEY_TYPES(bodo_array_type::NULLABLE_INT_BOOL,
                                             Bodo_CTypes::INT32);
        GROUPBY_INFO_IMPL_ALL_SAME_KEY_TYPES(bodo_array_type::NULLABLE_INT_BOOL,
                                             Bodo_CTypes::INT64);
        GROUPBY_INFO_IMPL_ALL_SAME_KEY_TYPES(bodo_array_type::NUMPY,
                                             Bodo_CTypes::INT32);
        GROUPBY_INFO_IMPL_ALL_SAME_KEY_TYPES(bodo_array_type::NUMPY,
                                             Bodo_CTypes::INT64);
        GROUPBY_INFO_IMPL_ALL_SAME_KEY_TYPES(bodo_array_type::NUMPY,
                                             Bodo_CTypes::DATETIME);
        GROUPBY_INFO_IMPL_ALL_SAME_KEY_TYPES(bodo_array_type::DICT,
                                             Bodo_CTypes::STRING);
        GROUPBY_INFO_IMPL_ALL_SAME_KEY_TYPES(bodo_array_type::STRING,
                                             Bodo_CTypes::STRING);
    }

    // general implementation with generic key comparator class
    KeysEqualComparator equal_fct{n_keys, table, /*is_na_equal=*/true};

    using rh_flat_t =
        bodo::unord_map_container<int64_t, int64_t, HashLookupIn32bitTable,
                                  KeysEqualComparator>;
    rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct, pool);

    get_group_info_impl(key_to_group_rh_flat, ev, grp_info, table, key_cols,
                        hashes, nunique_hashes, check_for_null_keys, key_dropna,
                        UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);
}

void get_group_info_iterate(std::vector<std::shared_ptr<table_info>>& tables,
                            std::shared_ptr<uint32_t[]>& hashes,
                            size_t nunique_hashes,
                            std::vector<grouping_info>& grp_infos,
                            const int64_t n_keys, const bool consider_missing,
                            bool key_dropna, bool is_parallel,
                            bodo::IBufferPool* const pool) {
    tracing::Event ev("get_group_info_iterate", is_parallel);
    if (tables.size() == 0) {
        throw std::runtime_error("get_group_info: tables is empty");
    }
    std::shared_ptr<table_info> table = tables[0];
    std::vector<std::shared_ptr<array_info>> key_cols =
        std::vector<std::shared_ptr<array_info>>(
            table->columns.begin(), table->columns.begin() + n_keys);
    // TODO: if |tables| > 1 then we probably need to use hashes from all the
    // tables to get an accurate nunique_hashes estimate. We can do it, but
    // it would mean calculating all hashes in advance
    // if |tables| > 1 means nunique is used in agg/aggregate with other
    // operations. In this case, recalculate hashes since hashes arg. passed is
    // computed with a different seed which leads to extra fake number of
    // groups.
    if (tables.size() > 1 || !hashes) {
        hashes = bodo::make_shared_arr<uint32_t>(table->nrows(), pool);
        hash_keys(hashes.get(), key_cols, SEED_HASH_GROUPBY_SHUFFLE,
                  is_parallel);
        nunique_hashes =
            get_nunique_hashes(hashes, table->nrows(), is_parallel);
    }
    grp_infos.emplace_back(pool);
    grouping_info& grp_info = grp_infos.back();

    uint64_t max_rows = 0;
    for (std::shared_ptr<table_info> table : tables) {
        max_rows = std::max(max_rows, table->nrows());
    }
    grp_info.row_to_group.reserve(max_rows);
    grp_info.row_to_group.resize(table->nrows());
    grp_info.next_row_in_group.reserve(max_rows);
    grp_info.next_row_in_group.resize(table->nrows(), -1);
    grp_info.group_to_first_row.reserve(nunique_hashes * 1.1);
    bodo::vector<int64_t> active_group_repr(pool);
    active_group_repr.reserve(nunique_hashes * 1.1);

    // TODO Incorporate or adapt other optimizations from `get_group_info`

    bool key_is_nullable = does_keys_have_nulls(key_cols);
    bool key_drop_nulls = key_is_nullable && key_dropna;

    // Start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values recorded in grp_info go from
    // 0 to num_groups - 1)
    int64_t next_group = 1;
    bodo::unord_map_container<multi_col_key, int64_t, multi_col_key_hash>
        key_to_group(pool);
    key_to_group.reserve(nunique_hashes);
    for (uint64_t i = 0; i < table->nrows(); i++) {
        if (key_drop_nulls) {
            if (does_row_has_nulls(key_cols, i)) {
                grp_info.row_to_group[i] = -1;
                if (consider_missing) {
                    grp_info.list_missing.push_back(i);
                }
                continue;
            }
        }
        multi_col_key key(hashes[i], table, i, n_keys);
        int64_t& group = key_to_group[key];  // this inserts 0 into the map if
                                             // key doesn't exist
        if (group == 0) {
            group = next_group++;  // this updates the value in the map without
                                   // another lookup
            grp_info.group_to_first_row.emplace_back(i);
            active_group_repr.emplace_back(i);
        } else {
            int64_t prev_elt = active_group_repr[group - 1];
            grp_info.next_row_in_group[prev_elt] = i;
            active_group_repr[group - 1] = i;
        }
        grp_info.row_to_group[i] = group - 1;
    }
    hashes.reset();
    grp_info.num_groups = grp_info.group_to_first_row.size();

    for (size_t j = 1; j < tables.size(); j++) {
        int64_t num_groups = next_group - 1;
        // IMPORTANT: Assuming all the tables have the same number and type of
        // key columns (but not the same values in key columns)
        table = tables[j];
        key_cols = std::vector<std::shared_ptr<array_info>>(
            table->columns.begin(), table->columns.begin() + n_keys);
        hashes = bodo::make_shared_arr<uint32_t>(table->nrows(), pool);
        hash_keys(hashes.get(), key_cols, SEED_HASH_GROUPBY_SHUFFLE,
                  is_parallel);
        grp_infos.emplace_back(pool);
        grouping_info& grp_info = grp_infos.back();
        grp_info.row_to_group.resize(table->nrows());
        grp_info.next_row_in_group.resize(table->nrows(), -1);
        grp_info.group_to_first_row.resize(num_groups, -1);
        active_group_repr.resize(num_groups);

        for (uint64_t i = 0; i < table->nrows(); i++) {
            if (key_drop_nulls) {
                if (does_row_has_nulls(key_cols, i)) {
                    grp_info.row_to_group[i] = -1;
                    if (consider_missing) {
                        grp_info.list_missing.push_back(i);
                    }
                    continue;
                }
            }
            multi_col_key key(hashes[i], table, i, n_keys);
            int64_t& group = key_to_group[key];  // this inserts 0 into the map
                                                 // if key doesn't exist
            if ((group == 0) ||
                (grp_info.group_to_first_row[group - 1] == -1)) {
                if (group == 0) {
                    group = next_group++;  // this updates the value in the map
                                           // without another lookup
                    grp_info.group_to_first_row.emplace_back(i);
                    active_group_repr.emplace_back(i);
                } else {
                    grp_info.group_to_first_row[group - 1] = i;
                    active_group_repr[group - 1] = i;
                }
            } else {
                int64_t prev_elt = active_group_repr[group - 1];
                grp_info.next_row_in_group[prev_elt] = i;
                active_group_repr[group - 1] = i;
            }
            grp_info.row_to_group[i] = group - 1;
        }
        hashes.reset();
        grp_info.num_groups = grp_info.group_to_first_row.size();
    }

    // set same num_groups in every group_info
    int64_t num_groups = next_group - 1;
    for (auto& grp_info : grp_infos) {
        grp_info.group_to_first_row.resize(num_groups, -1);
        grp_info.num_groups = num_groups;
    }
    ev.add_attribute("num_groups", static_cast<size_t>(num_groups));
}

// Groupby Apply

/**
 * @brief The main performance sensitive lop for computing
 * get_groupby_labels.
 *
 * @tparam Map The hashmap type to use
 * @param[in, out] key_to_group A hashmap mapping each set
 * of keys to a unique group number. It is initially empty and
 * filled throughout execution, although the output is not
 * needed after this function.
 * @param ev The tracing event to update with additional attributes.
 * @param[in] key_cols A vector containing the key columns.
 * @param[out] row_to_group Output array of labels to fill with each group
 * number.
 * @param[out] sort_idx Output array to fill with sort information for
 * performing a reverse shuffle.
 * @param key_drop_null Do we drop NA keys?
 * @param nrows How many rows in the input table?
 * @param is_parallel Is this executed in parallel.
 * @return int64_t The actual number of groups.
 */
template <typename T>
static int64_t get_groupby_labels_loop(
    T& key_to_group, std::vector<std::shared_ptr<array_info>>& key_cols,
    int64_t* row_to_group, int64_t* sort_idx, const bool key_drop_nulls,
    const int64_t nrows, bool is_parallel) {
    tracing::Event ev("get_groupby_labels_loop", is_parallel);
    // Start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values recorded in grp_info go from
    // 0 to num_groups - 1)
    int64_t next_group = 1;
    // There are two versions of the loop because a `if (key_drop_nulls)`
    // branch inside the loop has a bit of a performance hit and
    // this loop is one of the most expensive computations.
    // To not duplicate code, we put the common portion of the loop in
    // MAIN_LOOP_BODY macro

    bodo::vector<bodo::vector<int64_t>> group_rows;

#define MAIN_LOOP_BODY                                                      \
    int64_t& group = key_to_group[i]; /* this inserts 0 into the map if key \
                                         doesn't exist */                   \
    if (group == 0) {                                                       \
        group = next_group++; /* this updates the value in the map without  \
                                 another lookup */                          \
        group_rows.emplace_back();                                          \
    }                                                                       \
    group_rows[group - 1].push_back(i);                                     \
    row_to_group[i] = group - 1

    // keep track of how many NA values in the column
    int64_t na_pos = 0;
    if (!key_drop_nulls) {
        for (int64_t i = 0; i < nrows; i++) {
            MAIN_LOOP_BODY;
        }
    } else {
        for (int64_t i = 0; i < nrows; i++) {
            if (does_row_has_nulls(key_cols, i)) {
                row_to_group[i] = -1;
                // We need to keep position of all values in the sort_idx
                // regardless of being dropped or not Indexes of all NA values
                // are first in sort_idx since their group number is -1
                sort_idx[na_pos] = i;
                na_pos++;
                continue;
            }
            MAIN_LOOP_BODY;
        }
    }
    int64_t pos = 0 + na_pos;
    for (size_t i = 0; i < group_rows.size(); i++) {
        memcpy(sort_idx + pos, group_rows[i].data(),
               group_rows[i].size() * sizeof(int64_t));
        pos += group_rows[i].size();
    }
    return next_group - 1;
}
#undef MAIN_LOOP_BODY

/**
 * @brief Calculate the number of groups
 * for a given a map implementation.
 *
 * @tparam Map The hashmap type to use
 * @param[in, out] key_to_group A hashmap mapping each set
 * of keys to a unique group number. It is initially empty and
 * filled throughout execution, although the output is not
 * needed after this function.
 * @param ev The tracing event to update with additional attributes.
 * @param[out] out_labels Output array of labels to fill with each group number.
 * @param[out] sort_idx Output array to fill with sort information for
 * performing a reverse shuffle.
 * @param[in] table The input table.
 * @param[in] key_cols A vector containing the key columns.
 * @param[in] hashes Vector containing the hashes
 * @param nunique_hashes Number of unique hashes estimate. Used for estimating
 * the number of groups.
 * @param check_for_null_keys Do we check if keys are null?
 * @param key_dropna Do we drop NA keys?
 * @param load_factor What is the load factor for the hashmap?
 * @param is_parallel Is this executed in parallel.
 * @return int64_t The actual number of groups.
 */
template <typename Map>
int64_t get_groupby_labels_impl(
    Map& key_to_group, tracing::Event& ev, int64_t* out_labels,
    int64_t* sort_idx, std::shared_ptr<table_info> const table,
    std::vector<std::shared_ptr<array_info>>& key_cols,
    std::shared_ptr<uint32_t[]>& hashes, const size_t nunique_hashes,
    const bool check_for_null_keys, const bool key_dropna,
    const double load_factor, bool is_parallel) {
    tracing::Event ev_alloc("get_groupby_labels_alloc", is_parallel);
    key_to_group.max_load_factor(load_factor);
    key_to_group.reserve(nunique_hashes);
    ev_alloc.add_attribute("map bucket_count", key_to_group.bucket_count());
    ev_alloc.add_attribute("map max_load_factor",
                           key_to_group.max_load_factor());
    ev_alloc.finalize();

    const bool key_is_nullable =
        check_for_null_keys ? does_keys_have_nulls(key_cols) : false;
    const bool key_drop_nulls = key_is_nullable && key_dropna;
    ev.add_attribute("g_key_is_nullable", key_is_nullable);
    ev.add_attribute("g_key_dropna", key_dropna);
    ev.add_attribute("g_key_drop_nulls", key_drop_nulls);

    const int64_t num_groups = get_groupby_labels_loop<Map>(
        key_to_group, key_cols, out_labels, sort_idx, key_drop_nulls,
        table->nrows(), is_parallel);
    ev.add_attribute("num_groups", num_groups);
    do_map_dealloc<false>(hashes, key_to_group, is_parallel);
    return num_groups;
}

int64_t get_groupby_labels(std::shared_ptr<table_info> table,
                           int64_t* out_labels, int64_t* sort_idx,
                           bool key_dropna, bool is_parallel) {
    tracing::Event ev("get_groupby_labels", is_parallel);
    ev.add_attribute("input_table_nrows", static_cast<size_t>(table->nrows()));
    // TODO(ehsan): refactor to avoid code duplication with get_group_info
    // This function is similar to get_group_info. See that function for
    // more comments
    const int64_t n_keys = table->columns.size();
    std::vector<std::shared_ptr<array_info>> key_cols = table->columns;
    uint32_t seed = SEED_HASH_GROUPBY_SHUFFLE;
    for (auto a : key_cols) {
        if (a->arr_type == bodo_array_type::DICT) {
            // We need dictionaries to be global and unique for hashing.
            make_dictionary_global_and_unique(a, is_parallel);
        }
    }
    std::shared_ptr<uint32_t[]> hashes = hash_keys(key_cols, seed, is_parallel);

    size_t nunique_hashes =
        get_nunique_hashes(hashes, table->nrows(), is_parallel);
    ev.add_attribute("nunique_hashes_est", nunique_hashes);

    HashLookupIn32bitTable hash_fct{hashes};
    KeyEqualLookupIn32bitTable equal_fct{n_keys, table};

    const bool check_for_null_keys = true;
    using rh_flat_t =
        bodo::unord_map_container<int64_t, int64_t, HashLookupIn32bitTable,
                                  KeyEqualLookupIn32bitTable>;
    rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct);
    return get_groupby_labels_impl(
        key_to_group_rh_flat, ev, out_labels, sort_idx, table, key_cols, hashes,
        nunique_hashes, check_for_null_keys, key_dropna,
        UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);
}

int64_t get_groupby_labels_py_entry(table_info* table, int64_t* out_labels,
                                    int64_t* sort_idx, bool key_dropna,
                                    bool is_parallel) {
    try {
        return get_groupby_labels(std::shared_ptr<table_info>(table),
                                  out_labels, sort_idx, key_dropna,
                                  is_parallel);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}
