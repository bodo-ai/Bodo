// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_groupby.h"
#include <functional>
#include <limits>
#include <map>
#include "_array_hash.h"
#include "_array_operations.h"
#include "_array_utils.h"
#include "_decimal_ext.h"
#include "_distributed.h"
#include "_groupby_agg_funcs.h"
#include "_groupby_common.h"
#include "_groupby_eval.h"
#include "_groupby_ftypes.h"
#include "_groupby_hashing.h"
#include "_groupby_mpi_exscan.h"
#include "_groupby_udf.h"
#include "_groupby_update.h"
#include "_murmurhash3.h"
#include "_shuffle.h"

// this mapping is used by BasicColSet operations to know what combine (i.e.
// step (c)) function to use for a given aggregation function
static UNORD_MAP_CONTAINER<int, int> combine_funcs = {
    {Bodo_FTypes::size, Bodo_FTypes::sum},
    {Bodo_FTypes::sum, Bodo_FTypes::sum},
    {Bodo_FTypes::count, Bodo_FTypes::sum},
    {Bodo_FTypes::mean, Bodo_FTypes::sum},  // sum totals and counts
    {Bodo_FTypes::min, Bodo_FTypes::min},
    {Bodo_FTypes::max, Bodo_FTypes::max},
    {Bodo_FTypes::prod, Bodo_FTypes::prod},
    {Bodo_FTypes::first, Bodo_FTypes::first},
    {Bodo_FTypes::last, Bodo_FTypes::last},
    {Bodo_FTypes::nunique, Bodo_FTypes::sum},  // used in nunique_mode = 2
    {Bodo_FTypes::boolor_agg, Bodo_FTypes::boolor_agg}};

/**
 * Perform combine operation for variance, which for a set of rows belonging
 * to the same group with count (# observations), mean and m2, reduces
 * the values to one row. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param input array of counts (can include multiple values per group and
 * multiple groups)
 * @param input array of means (can include multiple values per group and
 * multiple groups)
 * @param input array of m2 (can include multiple values per group and multiple
 * groups)
 * @param output array of counts (one row per group)
 * @param output array of means (one row per group)
 * @param output array of m2 (one row per group)
 * @param maps row numbers in input arrays to row number in output array
 */
template <typename F>
void var_combine_F(array_info* count_col_in, array_info* mean_col_in,
                   array_info* m2_col_in, array_info* count_col_out,
                   array_info* mean_col_out, array_info* m2_col_out, F f) {
    for (size_t i = 0; i < count_col_in->length; i++) {
        // Var always has null compute columns even
        // if there is an original numpy input. All arrays
        // will have the same null bit value so just check 1.
        if (count_col_in->get_null_bit(i)) {
            int64_t j = f(i);
            uint64_t& count_a = getv<uint64_t>(count_col_out, j);
            uint64_t& count_b = getv<uint64_t>(count_col_in, i);
            // TODO: Can I delete this comment + condition
            // in the pivot case we can receive dummy values from other ranks
            // when combining the results (in this case with count == 0). This
            // is because the pivot case groups on index and creates n_pivot
            // columns, and so for each of its index values a rank will have
            // n_pivot fields, even if a rank does not have a particular (index,
            // pivot_value) pair
            if (count_b == 0) continue;
            double& mean_a = getv<double>(mean_col_out, j);
            double& mean_b = getv<double>(mean_col_in, i);
            double& m2_a = getv<double>(m2_col_out, j);
            double& m2_b = getv<double>(m2_col_in, i);
            uint64_t count = count_a + count_b;
            double delta = mean_b - mean_a;
            mean_a = (count_a * mean_a + count_b * mean_b) / count;
            m2_a = m2_a + m2_b + delta * delta * count_a * count_b / count;
            count_a = count;
            // Set all the null bits to true.
            count_col_out->set_null_bit(i, true);
            mean_col_out->set_null_bit(i, true);
            m2_col_out->set_null_bit(i, true);
        }
    }
}

void var_combine(array_info* count_col_in, array_info* mean_col_in,
                 array_info* m2_col_in, array_info* count_col_out,
                 array_info* mean_col_out, array_info* m2_col_out,
                 grouping_info const& grp_info) {
    auto f = [&](int64_t const& i_row) -> int64_t {
        return grp_info.row_to_group[i_row];
    };
    return var_combine_F<decltype(f)>(count_col_in, mean_col_in, m2_col_in,
                                      count_col_out, mean_col_out, m2_col_out,
                                      f);
}

/**
 * The main get_group_info loop which populates a grouping_info structure
 * (map rows from input to their group number, and store the first input row
 * for each group).
 *
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
static void get_group_info_loop(T& key_to_group,
                                std::vector<array_info*>& key_cols,
                                grouping_info& grp_info,
                                const bool key_drop_nulls, const int64_t nrows,
                                bool is_parallel) {
    tracing::Event ev("get_group_info_loop", is_parallel);
    std::vector<int64_t>& group_to_first_row = grp_info.group_to_first_row;
    std::vector<int64_t>& row_to_group = grp_info.row_to_group;
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

template <typename Map>
void get_group_info_impl(Map& key_to_group, tracing::Event& ev,
                         grouping_info& grp_info, table_info* const table,
                         std::vector<array_info*>& key_cols, uint32_t*& hashes,
                         const size_t nunique_hashes,
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

/**
 * Given a set of tables with n key columns, this function calculates the row to
 * group mapping for every row based on its key. For every row in the tables,
 * this only does *one* lookup in the hash map.
 *
 * @param[in] tables the tables
 * @param[in] hashes hashes if they have already been calculated. nullptr
 * otherwise
 * @param[in] nunique_hashes estimated number of unique hashes if hashes are
 * provided
 * @param[out] grp_infos is grouping_info structures that map row numbers to
 * group numbers
 * @param[in] check_for_null_keys whether to check for null keys. If a key is
 * null and key_dropna=True that row will not be mapped to any group
 * @param[in] key_dropna whether to allow NA values in group keys or not.
 * @param[in] is_parallel: true if data is distributed
 */
void get_group_info(std::vector<table_info*>& tables, uint32_t*& hashes,
                    size_t nunique_hashes,
                    std::vector<grouping_info>& grp_infos,
                    bool check_for_null_keys, bool key_dropna,
                    bool is_parallel) {
    tracing::Event ev("get_group_info", is_parallel);
    if (tables.size() != 1) {
        throw std::runtime_error("get_group_info: expected 1 table input");
    }
    table_info* table = tables[0];
    ev.add_attribute("input_table_nrows", static_cast<size_t>(table->nrows()));
    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table->columns.begin(), table->columns.begin() + table->num_keys);
    if (hashes == nullptr) {
        hashes = hash_keys(key_cols, SEED_HASH_GROUPBY_SHUFFLE, is_parallel);
        nunique_hashes =
            get_nunique_hashes(hashes, table->nrows(), is_parallel);
    }
    ev.add_attribute("nunique_hashes_est", nunique_hashes);
    grp_infos.emplace_back();
    grouping_info& grp_info = grp_infos.back();
    const int64_t n_keys = table->num_keys;

    HashLookupIn32bitTable hash_fct{hashes};

    // use faster specialized implementation for common 1 key cases
    if (n_keys == 1) {
        array_info* arr = table->columns[0];
        bodo_array_type::arr_type_enum arr_type = arr->arr_type;
        Bodo_CTypes::CTypeEnum dtype = arr->dtype;

        // macro to reduce code duplication
#ifndef GROUPBY_INFO_IMPL_1_KEY
#define GROUPBY_INFO_IMPL_1_KEY(ARRAY_TYPE, DTYPE)                        \
    if (arr_type == ARRAY_TYPE && dtype == DTYPE) {                       \
        using KeyType = KeysEqualComparatorOneKey<ARRAY_TYPE, DTYPE,      \
                                                  /*is_na_equal=*/true>;  \
        KeyType equal_fct{arr};                                           \
        using rh_flat_t =                                                 \
            UNORD_MAP_CONTAINER<int64_t, int64_t, HashLookupIn32bitTable, \
                                KeyType>;                                 \
        rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct);          \
        get_group_info_impl(key_to_group_rh_flat, ev, grp_info, table,    \
                            key_cols, hashes, nunique_hashes,             \
                            check_for_null_keys, key_dropna,              \
                            UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);  \
        return;                                                           \
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
        array_info* arr1 = table->columns[0];
        array_info* arr2 = table->columns[1];
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
            UNORD_MAP_CONTAINER<int64_t, int64_t, HashLookupIn32bitTable,  \
                                KeyType>;                                  \
        rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct);           \
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

    // general implementation with generic key comparator class
    KeysEqualComparator equal_fct{n_keys, table, /*is_na_equal=*/true};

    using rh_flat_t =
        UNORD_MAP_CONTAINER<int64_t, int64_t, HashLookupIn32bitTable,
                            KeysEqualComparator>;
    rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct);
    get_group_info_impl(key_to_group_rh_flat, ev, grp_info, table, key_cols,
                        hashes, nunique_hashes, check_for_null_keys, key_dropna,
                        UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);
}

template <typename T>
static int64_t get_groupby_labels_loop(T& key_to_group,
                                       std::vector<array_info*>& key_cols,
                                       int64_t* row_to_group, int64_t* sort_idx,
                                       const bool key_drop_nulls,
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

    std::vector<std::vector<int64_t>> group_rows;

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

namespace {
template <typename Map>
int64_t get_groupby_labels_impl(Map& key_to_group, tracing::Event& ev,
                                int64_t* out_labels, int64_t* sort_idx,
                                table_info* const table,
                                std::vector<array_info*>& key_cols,
                                uint32_t*& hashes, const size_t nunique_hashes,
                                const bool check_for_null_keys,
                                const bool key_dropna, const double load_factor,
                                bool is_parallel) {
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
}  // namespace

/**
 * @brief Get total number of groups for input key arrays
 *
 * @param table a table of all key arrays
 * @param[out] out_labels output array to fill
 * @param[out] sort_idx sorted group indices
 * @param[in] key_dropna whether to allow NA values in group keys or not.
 * @param is_parallel: true if data is distributed
 * @return int64_t total number of groups
 */
int64_t get_groupby_labels(table_info* table, int64_t* out_labels,
                           int64_t* sort_idx, bool key_dropna,
                           bool is_parallel) {
    tracing::Event ev("get_groupby_labels", is_parallel);
    ev.add_attribute("input_table_nrows", static_cast<size_t>(table->nrows()));
    // TODO(ehsan): refactor to avoid code duplication with get_group_info
    // This function is similar to get_group_info. See that function for
    // more comments
    table->num_keys = table->columns.size();
    std::vector<array_info*> key_cols = table->columns;
    uint32_t seed = SEED_HASH_GROUPBY_SHUFFLE;
    for (auto a : key_cols) {
        if (a->arr_type == bodo_array_type::DICT) {
            // We need dictionaries to be global and unique for hashing.
            make_dictionary_global_and_unique(a, is_parallel);
        }
    }
    uint32_t* hashes = hash_keys(key_cols, seed, is_parallel);

    size_t nunique_hashes =
        get_nunique_hashes(hashes, table->nrows(), is_parallel);
    ev.add_attribute("nunique_hashes_est", nunique_hashes);
    const int64_t n_keys = table->num_keys;

    HashLookupIn32bitTable hash_fct{hashes};
    KeyEqualLookupIn32bitTable equal_fct{n_keys, table};

    const bool check_for_null_keys = true;
    using rh_flat_t =
        UNORD_MAP_CONTAINER<int64_t, int64_t, HashLookupIn32bitTable,
                            KeyEqualLookupIn32bitTable>;
    rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct);
    return get_groupby_labels_impl(
        key_to_group_rh_flat, ev, out_labels, sort_idx, table, key_cols, hashes,
        nunique_hashes, check_for_null_keys, key_dropna,
        UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);
}

/**
 * Given a set of tables with n key columns, this function calculates the row to
 * group mapping for every row based on its key. For every row in the tables,
 * this only does *one* lookup in the hash map.
 *
 * @param           tables: the tables
 * @param[in] hashes hashes for first table in tables, if they have already
 * been calculated. nullptr otherwise
 * @param[in] nunique_hashes estimated number of unique hashes if hashes are
 * provided (for first table)
 * @param[out]      grouping_info structures that map row numbers to group
 * numbers
 * @param[in] consider_missing: whether to return the list of missing rows or
 * not
 * @param[in] key_dropna whether to allow NA values in group keys or not.
 * @param[in] is_parallel: true if data is distributed
 */
void get_group_info_iterate(std::vector<table_info*>& tables, uint32_t*& hashes,
                            size_t nunique_hashes,
                            std::vector<grouping_info>& grp_infos,
                            const bool consider_missing, bool key_dropna,
                            bool is_parallel) {
    tracing::Event ev("get_group_info_iterate", is_parallel);
    if (tables.size() == 0) {
        throw std::runtime_error("get_group_info: tables is empty");
    }
    table_info* table = tables[0];
    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table->columns.begin(), table->columns.begin() + table->num_keys);
    // TODO: if |tables| > 1 then we probably need to use hashes from all the
    // tables to get an accurate nunique_hashes estimate. We can do it, but
    // it would mean calculating all hashes in advance
    // if |tables| > 1 means nunique is used in agg/aggregate with other
    // operations. In this case, recalculate hashes since hashes arg. passed is
    // computed with a different seed which leads to extra fake number of
    // groups.
    if (tables.size() > 1 || hashes == nullptr) {
        hashes = hash_keys(key_cols, SEED_HASH_GROUPBY_SHUFFLE, is_parallel);
        nunique_hashes =
            get_nunique_hashes(hashes, table->nrows(), is_parallel);
    }
    grp_infos.emplace_back();
    grouping_info& grp_info = grp_infos.back();

    uint64_t max_rows = 0;
    for (table_info* table : tables) {
        max_rows = std::max(max_rows, table->nrows());
    }
    grp_info.row_to_group.reserve(max_rows);
    grp_info.row_to_group.resize(table->nrows());
    grp_info.next_row_in_group.reserve(max_rows);
    grp_info.next_row_in_group.resize(table->nrows(), -1);
    grp_info.group_to_first_row.reserve(nunique_hashes * 1.1);
    std::vector<int64_t> active_group_repr;
    active_group_repr.reserve(nunique_hashes * 1.1);

    // TODO Incorporate or adapt other optimizations from `get_group_info`

    bool key_is_nullable = does_keys_have_nulls(key_cols);
    bool key_drop_nulls = key_is_nullable && key_dropna;

    // Start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values recorded in grp_info go from
    // 0 to num_groups - 1)
    int64_t next_group = 1;
    UNORD_MAP_CONTAINER<multi_col_key, int64_t, multi_col_key_hash>
        key_to_group;
    key_to_group.reserve(nunique_hashes);
    for (uint64_t i = 0; i < table->nrows(); i++) {
        if (key_drop_nulls) {
            if (does_row_has_nulls(key_cols, i)) {
                grp_info.row_to_group[i] = -1;
                if (consider_missing) grp_info.list_missing.push_back(i);
                continue;
            }
        }
        multi_col_key key(hashes[i], table, i, is_parallel);
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
    delete[] hashes;
    hashes = nullptr;
    grp_info.num_groups = grp_info.group_to_first_row.size();

    for (size_t j = 1; j < tables.size(); j++) {
        int64_t num_groups = next_group - 1;
        // IMPORTANT: Assuming all the tables have the same number and type of
        // key columns (but not the same values in key columns)
        table = tables[j];
        key_cols = std::vector<array_info*>(
            table->columns.begin(), table->columns.begin() + table->num_keys);
        hashes = hash_keys(key_cols, SEED_HASH_GROUPBY_SHUFFLE, is_parallel);
        grp_infos.emplace_back();
        grouping_info& grp_info = grp_infos.back();
        grp_info.row_to_group.resize(table->nrows());
        grp_info.next_row_in_group.resize(table->nrows(), -1);
        grp_info.group_to_first_row.resize(num_groups, -1);
        active_group_repr.resize(num_groups);

        for (uint64_t i = 0; i < table->nrows(); i++) {
            if (key_drop_nulls) {
                if (does_row_has_nulls(key_cols, i)) {
                    grp_info.row_to_group[i] = -1;
                    if (consider_missing) grp_info.list_missing.push_back(i);
                    continue;
                }
            }
            multi_col_key key(hashes[i], table, i, is_parallel);
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
        delete[] hashes;
        hashes = nullptr;
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

/**
 * @brief ngroup assigns the same group number to each row in the group.
 * If data is replicated, start from 0 and the group number will be the output
 * value for all rows in that group. If data is distributed, we need to identify
 * starting group number on each rank. Then, row's output is: start group number
 * + row's local group number (igrp in current rank) This is done by summing
 * number of groups of ranks before current rank. This is achieved with
 * MPI_Exscan: partial reduction excluding current rank value.
 * @param[in] arr The input column on which we do the computation
 * @param[out] out_arr The output column which contains ngroup results
 * @param[in] grp_info grouping_info about groups and rows organization per rank
 * @param[in] is_parallel: true if data is distributed (used to indicate whether
 * we need to do cumsum on group numbers or not)
 */
void ngroup_computation(array_info* arr, array_info* out_arr,
                        grouping_info const& grp_info, bool is_parallel) {
    //
    size_t num_group = grp_info.group_to_first_row.size();
    int64_t start_ngroup = 0;
    if (is_parallel) {
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::INT64);
        MPI_Exscan(&num_group, &start_ngroup, 1, mpi_typ, MPI_SUM,
                   MPI_COMM_WORLD);
    }
    for (size_t i = 0; i < arr->length; i++) {
        int64_t i_grp = grp_info.row_to_group[i];
        if (i_grp != -1) {
            int64_t val = i_grp + start_ngroup;
            ngroup_agg<int64_t, Bodo_CTypes::INT64>::apply(
                getv<int64_t>(out_arr, i), val);
        }
    }
}

namespace {
/**
 * Compute hash for numpy or nullable_int_bool bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashNuniqueComputationNumpyOrNullableIntBool {
    uint32_t operator()(const int64_t i) const {
        char* ptr = arr->data1 + i * siztype;
        uint32_t retval = 0;
        hash_string_32(ptr, siztype, seed, &retval);
        return retval;
    }
    array_info* arr;
    size_t siztype;
    uint32_t seed;
};

/**
 * Key comparison for numpy or nullable_int_bool bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualNuniqueComputationNumpyOrNullableIntBool {
    bool operator()(const int64_t i1, const int64_t i2) const {
        char* ptr1 = arr->data1 + i1 * siztype;
        char* ptr2 = arr->data1 + i2 * siztype;
        return memcmp(ptr1, ptr2, siztype) == 0;
    }

    array_info* arr;
    size_t siztype;
};

/**
 * Compute hash for list string bodo types.
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashNuniqueComputationListString {
    size_t operator()(const int64_t i) const {
        // We do not put the lengths and bitmask in the hash
        // computation. after all, it is just a hash
        char* val_chars = arr->data1 + in_data_offsets[in_index_offsets[i]];
        int len = in_data_offsets[in_index_offsets[i + 1]] -
                  in_data_offsets[in_index_offsets[i]];
        uint32_t val;
        hash_string_32(val_chars, len, seed, &val);
        return static_cast<size_t>(val);
    }
    array_info* arr;
    offset_t* in_index_offsets;
    offset_t* in_data_offsets;
    uint32_t seed;
};

/**
 * Key comparison for list string bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualNuniqueComputationListString {
    bool operator()(const int64_t i1, const int64_t i2) const {
        bool bit1 = arr->get_null_bit(i1);
        bool bit2 = arr->get_null_bit(i2);
        if (bit1 != bit2)
            return false;  // That first case, might not be necessary.
        size_t len1 = in_index_offsets[i1 + 1] - in_index_offsets[i1];
        size_t len2 = in_index_offsets[i2 + 1] - in_index_offsets[i2];
        if (len1 != len2) return false;
        for (size_t u = 0; u < len1; u++) {
            offset_t len_str1 = in_data_offsets[in_index_offsets[i1] + 1] -
                                in_data_offsets[in_index_offsets[i1]];
            offset_t len_str2 = in_data_offsets[in_index_offsets[i2] + 1] -
                                in_data_offsets[in_index_offsets[i2]];
            if (len_str1 != len_str2) return false;
            bool bit1 = GetBit(sub_null_bitmask, in_index_offsets[i1]);
            bool bit2 = GetBit(sub_null_bitmask, in_index_offsets[i2]);
            if (bit1 != bit2) return false;
        }
        offset_t nb_char1 = in_data_offsets[in_index_offsets[i1 + 1]] -
                            in_data_offsets[in_index_offsets[i1]];
        offset_t nb_char2 = in_data_offsets[in_index_offsets[i2 + 1]] -
                            in_data_offsets[in_index_offsets[i2]];
        if (nb_char1 != nb_char2) return false;
        char* ptr1 = arr->data1 +
                     sizeof(offset_t) * in_data_offsets[in_index_offsets[i1]];
        char* ptr2 = arr->data1 +
                     sizeof(offset_t) * in_data_offsets[in_index_offsets[i2]];
        return memcmp(ptr1, ptr2, len1) == 0;
    }

    array_info* arr;
    offset_t* in_index_offsets;
    offset_t* in_data_offsets;
    uint8_t* sub_null_bitmask;
    uint32_t seed;
};

/**
 * Compute hash for string bodo types.
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashNuniqueComputationString {
    size_t operator()(const int64_t i) const {
        char* val_chars = arr->data1 + in_offsets[i];
        size_t len = in_offsets[i + 1] - in_offsets[i];
        uint32_t val;
        hash_string_32(val_chars, len, seed, &val);
        return size_t(val);
    }
    array_info* arr;
    offset_t* in_offsets;
    uint32_t seed;
};

/**
 * Key comparison for string bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualNuniqueComputationString {
    bool operator()(const int64_t i1, const int64_t i2) const {
        size_t len1 = in_offsets[i1 + 1] - in_offsets[i1];
        size_t len2 = in_offsets[i2 + 1] - in_offsets[i2];
        if (len1 != len2) {
            return false;
        }
        char* ptr1 = arr->data1 + in_offsets[i1];
        char* ptr2 = arr->data1 + in_offsets[i2];
        return memcmp(ptr1, ptr2, len1) == 0;
    }

    array_info* arr;
    offset_t* in_offsets;
};
}  // namespace

/**
 * The nunique_computation function. It uses the symbolic information to compute
 * the nunique results.
 *
 * @param arr The column on which we do the computation
 * @param out_arr[out] The column which contains nunique results
 * @param grp_info The array containing information on how the rows are
 * organized
 * @param dropna The boolean dropna indicating whether we drop or not the NaN
 * values from the nunique computation.
 * @param is_parallel: true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 */
void nunique_computation(array_info* arr, array_info* out_arr,
                         grouping_info const& grp_info, bool const& dropna,
                         bool const& is_parallel) {
    tracing::Event ev("nunique_computation", is_parallel);
    size_t num_group = grp_info.group_to_first_row.size();
    if (num_group == 0) {
        return;
    }
    // Note: Dictionary encoded is supported because we just
    // call nunique on the indices. See update that converts
    // the dict array to its indices. This is tested with
    // test_nunique_dict.
    if (arr->arr_type == bodo_array_type::NUMPY ||
        arr->arr_type == bodo_array_type::CATEGORICAL) {
        /**
         * Check if a pointer points to a NaN or not
         *
         * @param the char* pointer
         * @param the type of the data in input
         */
        auto isnan_entry = [&](char* ptr) -> bool {
            if (arr->dtype == Bodo_CTypes::FLOAT32) {
                float* ptr_f = (float*)ptr;
                return isnan(*ptr_f);
            }
            if (arr->dtype == Bodo_CTypes::FLOAT64) {
                double* ptr_d = (double*)ptr;
                return isnan(*ptr_d);
            }
            if (arr->dtype == Bodo_CTypes::DATETIME ||
                arr->dtype == Bodo_CTypes::TIMEDELTA) {
                int64_t* ptr_i = (int64_t*)ptr;
                return *ptr_i == std::numeric_limits<int64_t>::min();
            }
            if (arr->arr_type == bodo_array_type::CATEGORICAL) {
                return isnan_categorical_ptr(arr->dtype, ptr);
            }
            return false;
        };
        const size_t siztype = numpy_item_size[arr->dtype];
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationNumpyOrNullableIntBool hash_fct{arr, siztype,
                                                              seed};
        KeyEqualNuniqueComputationNumpyOrNullableIntBool equal_fct{arr,
                                                                   siztype};
        UNORD_SET_CONTAINER<int64_t,
                            HashNuniqueComputationNumpyOrNullableIntBool,
                            KeyEqualNuniqueComputationNumpyOrNullableIntBool>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) {
                continue;
            }
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                char* ptr = arr->data1 + (i * siztype);
                if (!isnan_entry(ptr)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    } else if (arr->arr_type == bodo_array_type::LIST_STRING) {
        offset_t* in_index_offsets = (offset_t*)arr->data3;
        offset_t* in_data_offsets = (offset_t*)arr->data2;
        uint8_t* sub_null_bitmask = (uint8_t*)arr->sub_null_bitmask;
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationListString hash_fct{arr, in_index_offsets,
                                                  in_data_offsets, seed};
        KeyEqualNuniqueComputationListString equal_fct{
            arr, in_index_offsets, in_data_offsets, sub_null_bitmask, seed};
        UNORD_SET_CONTAINER<int64_t, HashNuniqueComputationListString,
                            KeyEqualNuniqueComputationListString>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) {
                continue;
            }
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                if (arr->get_null_bit(i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    } else if (arr->arr_type == bodo_array_type::STRING) {
        offset_t* in_offsets = (offset_t*)arr->data2;
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationString hash_fct{arr, in_offsets, seed};
        KeyEqualNuniqueComputationString equal_fct{arr, in_offsets};
        UNORD_SET_CONTAINER<int64_t, HashNuniqueComputationString,
                            KeyEqualNuniqueComputationString>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) {
                continue;
            }
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                if (arr->get_null_bit(i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    } else if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        const size_t siztype = numpy_item_size[arr->dtype];
        HashNuniqueComputationNumpyOrNullableIntBool hash_fct{arr, siztype};
        KeyEqualNuniqueComputationNumpyOrNullableIntBool equal_fct{arr,
                                                                   siztype};
        UNORD_SET_CONTAINER<int64_t,
                            HashNuniqueComputationNumpyOrNullableIntBool,
                            KeyEqualNuniqueComputationNumpyOrNullableIntBool>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) {
                continue;
            }
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                if (arr->get_null_bit(i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    } else {
        throw std::runtime_error(
            "Unsupported array type encountered with nunique. Found type: " +
            GetArrType_as_string(arr->arr_type));
    }
}

template <typename F, int ftype>
array_info* apply_to_column_list_string(array_info* in_col, array_info* out_col,
                                        const grouping_info& grp_info, F f) {
    size_t num_groups = grp_info.num_groups;
    std::vector<std::vector<std::pair<std::string, bool>>> ListListPair(
        num_groups);
    char* data_i = in_col->data1;
    offset_t* index_offsets_i = (offset_t*)in_col->data3;
    offset_t* data_offsets_i = (offset_t*)in_col->data2;
    uint8_t* sub_null_bitmask_i = (uint8_t*)in_col->sub_null_bitmask;
    // Computing the strings used in output.
    uint64_t n_bytes = (num_groups + 7) >> 3;
    std::vector<uint8_t> Vmask(n_bytes, 0);
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            bool out_bit_set = out_col->get_null_bit(i_grp);
            if (ftype == Bodo_FTypes::first && out_bit_set) continue;
            offset_t start_offset = index_offsets_i[i];
            offset_t end_offset = index_offsets_i[i + 1];
            offset_t len = end_offset - start_offset;
            std::vector<std::pair<std::string, bool>> LStrB(len);
            for (offset_t i = 0; i < len; i++) {
                offset_t len_str = data_offsets_i[start_offset + i + 1] -
                                   data_offsets_i[start_offset + i];
                offset_t pos_start = data_offsets_i[start_offset + i];
                std::string val(&data_i[pos_start], len_str);
                bool str_bit = GetBit(sub_null_bitmask_i, start_offset + i);
                LStrB[i] = {val, str_bit};
            }
            if (out_bit_set) {
                aggliststring<ftype>::apply(ListListPair[i_grp], LStrB);
            } else {
                ListListPair[i_grp] = LStrB;
                out_col->set_null_bit(i_grp, true);
            }
        }
    }
    return create_list_string_array(Vmask, ListListPair);
}

template <typename F, int ftype>
array_info* apply_to_column_string(array_info* in_col, array_info* out_col,
                                   std::vector<array_info*>& aux_cols,
                                   const grouping_info& grp_info, F f) {
    size_t num_groups = grp_info.num_groups;
    size_t n_bytes = (num_groups + 7) >> 3;
    std::vector<uint8_t> V(n_bytes, 0);
    std::vector<std::string> ListString(num_groups);
    char* data_i = in_col->data1;
    offset_t* offsets_i = (offset_t*)in_col->data2;
    switch (ftype) {
        case Bodo_FTypes::idxmax: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    offset_t start_offset = offsets_i[i];
                    offset_t end_offset = offsets_i[i + 1];
                    offset_t len = end_offset - start_offset;
                    std::string val(&data_i[start_offset], len);
                    if (out_bit_set) {
                        idxmax_string(ListString[i_grp], val,
                                      getv<uint64_t>(index_pos, i_grp), i);
                    } else {
                        ListString[i_grp] = val;
                        getv<uint64_t>(index_pos, i_grp) = i;
                        SetBitTo(V.data(), i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmin: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    offset_t start_offset = offsets_i[i];
                    offset_t end_offset = offsets_i[i + 1];
                    offset_t len = end_offset - start_offset;
                    std::string val(&data_i[start_offset], len);
                    if (out_bit_set) {
                        idxmin_string(ListString[i_grp], val,
                                      getv<uint64_t>(index_pos, i_grp), i);
                    } else {
                        ListString[i_grp] = val;
                        getv<uint64_t>(index_pos, i_grp) = i;
                        SetBitTo(V.data(), i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmax_na_first: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                // If we are putting NA values first then we stop
                // visiting a group once we see a NA value. We
                // initialize the output values to all be non-NA values
                // to ensure this.
                if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                    if (in_col->get_null_bit(i)) {
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        offset_t start_offset = offsets_i[i];
                        offset_t end_offset = offsets_i[i + 1];
                        offset_t len = end_offset - start_offset;
                        std::string val(&data_i[start_offset], len);
                        if (out_bit_set) {
                            idxmax_string(ListString[i_grp], val,
                                          getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            ListString[i_grp] = val;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                        }
                    } else {
                        // If we have an NA value mark this group as
                        // done and update the index
                        getv<uint64_t>(index_pos, i_grp) = i;
                        // set the null bit for the count so we know
                        // to stop visiting the group. The data is still
                        // valid.
                        index_pos->set_null_bit(i_grp, false);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmin_na_first: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                // If we are putting NA values first then we stop
                // visiting a group once we see a NA value. We
                // initialize the output values to all be non-NA values
                // to ensure this.
                if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                    if (in_col->get_null_bit(i)) {
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        offset_t start_offset = offsets_i[i];
                        offset_t end_offset = offsets_i[i + 1];
                        offset_t len = end_offset - start_offset;
                        std::string val(&data_i[start_offset], len);
                        if (out_bit_set) {
                            idxmin_string(ListString[i_grp], val,
                                          getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            ListString[i_grp] = val;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                        }
                    } else {
                        // If we have an NA value mark this group as
                        // done and update the index
                        getv<uint64_t>(index_pos, i_grp) = i;
                        // set the null bit for the count so we know
                        // to stop visiting the group. The data is still
                        // valid.
                        index_pos->set_null_bit(i_grp, false);
                    }
                }
            }
            break;
        }
        default:
            // Computing the strings used in output.
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    if (ftype == Bodo_FTypes::first && out_bit_set) continue;
                    offset_t start_offset = offsets_i[i];
                    offset_t end_offset = offsets_i[i + 1];
                    offset_t len = end_offset - start_offset;
                    std::string val(&data_i[start_offset], len);
                    if (out_bit_set) {
                        aggstring<ftype>::apply(ListString[i_grp], val);
                    } else {
                        ListString[i_grp] = val;
                        SetBitTo(V.data(), i_grp, true);
                    }
                }
            }
    }
    // TODO: Avoid creating the output array for
    // idxmin/idxmax/idxmin_na_first/idxmax_na_first
    // Determining the number of characters in output.
    return create_string_array(V, ListString);
}

template <typename F, int ftype>
array_info* apply_sum_to_column_string(array_info* in_col, array_info* out_col,
                                       const grouping_info& grp_info, F f) {
    // allocate output array (length is number of groups, number of chars same
    // as input)
    size_t num_groups = grp_info.num_groups;
    int64_t n_chars = in_col->n_sub_elems;
    array_info* out_arr = alloc_string_array(num_groups, n_chars, 0);
    size_t n_bytes = (num_groups + 7) >> 3;
    memset(out_arr->null_bitmask, 0xff, n_bytes);  // null not possible

    // find offsets for each output string
    std::vector<offset_t> str_offsets(num_groups + 1, 0);
    char* data_i = in_col->data1;
    offset_t* offsets_i = (offset_t*)in_col->data2;
    char* data_o = out_arr->data1;
    offset_t* offsets_o = (offset_t*)out_arr->data2;

    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            offset_t len = offsets_i[i + 1] - offsets_i[i];
            str_offsets[i_grp + 1] += len;
        }
    }
    std::partial_sum(str_offsets.begin(), str_offsets.end(),
                     str_offsets.begin());
    memcpy(offsets_o, str_offsets.data(), (num_groups + 1) * sizeof(offset_t));

    // copy characters to output
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            offset_t len = offsets_i[i + 1] - offsets_i[i];
            memcpy(&data_o[str_offsets[i_grp]], data_i + offsets_i[i], len);
            str_offsets[i_grp] += len;
        }
    }
    return out_arr;
}

/**
 * @brief Applies max/min/first/last to dictionary encoded string column
 *
 * @tparam F
 * @tparam ftype
 * @param in_col: the input dictionary encoded string column
 * @param out_col: the output dictionary encoded string column
 * @param grp_info: groupby information
 * @param f: a function that returns group index given row index
 * @return array_info*
 */
template <typename F, int ftype>
array_info* apply_to_column_dict(array_info* in_col, array_info* out_col,
                                 std::vector<array_info*>& aux_cols,
                                 const grouping_info& grp_info, F f) {
    size_t num_groups = grp_info.num_groups;
    size_t n_bytes = (num_groups + 7) >> 3;
    array_info* indices_arr =
        alloc_nullable_array(num_groups, Bodo_CTypes::INT32, 0);
    std::vector<uint8_t> V(n_bytes,
                           0);  // bitmask to mark if group's been updated
    char* data_i = in_col->info1->data1;
    offset_t* offsets_i = (offset_t*)in_col->info1->data2;
    switch (ftype) {
        case Bodo_FTypes::idxmax: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
                    int32_t& org_ind = getv<int32_t>(indices_arr, i_grp);
                    if (out_bit_set) {
                        // Get the address and length of the new value to be
                        // compared
                        offset_t start_offset = offsets_i[dict_ind];
                        offset_t end_offset = offsets_i[dict_ind + 1];
                        offset_t len = end_offset - start_offset;
                        std::string s2(&data_i[start_offset], len);
                        // Get the address and length of the cumulative result
                        offset_t start_offset_org = offsets_i[org_ind];
                        offset_t end_offset_org = offsets_i[org_ind + 1];
                        offset_t len_org = end_offset_org - start_offset_org;
                        std::string s1(&data_i[start_offset_org], len_org);
                        idxmax_dict(org_ind, dict_ind, s1, s2,
                                    getv<uint64_t>(index_pos, i_grp), i);
                    } else {
                        org_ind = dict_ind;
                        getv<uint64_t>(index_pos, i_grp) = i;
                        SetBitTo(V.data(), i_grp, true);
                        indices_arr->set_null_bit(i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmin: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
                    int32_t& org_ind = getv<int32_t>(indices_arr, i_grp);
                    if (out_bit_set) {
                        // Get the address and length of the new value to be
                        // compared
                        offset_t start_offset = offsets_i[dict_ind];
                        offset_t end_offset = offsets_i[dict_ind + 1];
                        offset_t len = end_offset - start_offset;
                        std::string s2(&data_i[start_offset], len);
                        // Get the address and length of the cumulative result
                        offset_t start_offset_org = offsets_i[org_ind];
                        offset_t end_offset_org = offsets_i[org_ind + 1];
                        offset_t len_org = end_offset_org - start_offset_org;
                        std::string s1(&data_i[start_offset_org], len_org);
                        idxmin_dict(org_ind, dict_ind, s1, s2,
                                    getv<uint64_t>(index_pos, i_grp), i);
                    } else {
                        org_ind = dict_ind;
                        getv<uint64_t>(index_pos, i_grp) = i;
                        SetBitTo(V.data(), i_grp, true);
                        indices_arr->set_null_bit(i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmax_na_first: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                // If we are putting NA values first then we stop
                // visiting a group once we see a NA value. We
                // initialize the output values to all be non-NA values
                // to ensure this.
                if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                    if (in_col->get_null_bit(i)) {
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
                        int32_t& org_ind = getv<int32_t>(indices_arr, i_grp);
                        if (out_bit_set) {
                            // Get the address and length of the new value to be
                            // compared
                            offset_t start_offset = offsets_i[dict_ind];
                            offset_t end_offset = offsets_i[dict_ind + 1];
                            offset_t len = end_offset - start_offset;
                            std::string s2(&data_i[start_offset], len);
                            // Get the address and length of the cumulative
                            // result
                            offset_t start_offset_org = offsets_i[org_ind];
                            offset_t end_offset_org = offsets_i[org_ind + 1];
                            offset_t len_org =
                                end_offset_org - start_offset_org;
                            std::string s1(&data_i[start_offset_org], len_org);
                            idxmax_dict(org_ind, dict_ind, s1, s2,
                                        getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            org_ind = dict_ind;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                            indices_arr->set_null_bit(i_grp, true);
                        }
                    } else {
                        // If we have an NA value mark this group as
                        // done and update the index
                        getv<uint64_t>(index_pos, i_grp) = i;
                        // set the null bit for the count so we know
                        // to stop visiting the group. The data is still
                        // valid.
                        index_pos->set_null_bit(i_grp, false);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmin_na_first: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                // If we are putting NA values first then we stop
                // visiting a group once we see a NA value. We
                // initialize the output values to all be non-NA values
                // to ensure this.
                if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                    if (in_col->get_null_bit(i)) {
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
                        int32_t& org_ind = getv<int32_t>(indices_arr, i_grp);
                        if (out_bit_set) {
                            // Get the address and length of the new value to be
                            // compared
                            offset_t start_offset = offsets_i[dict_ind];
                            offset_t end_offset = offsets_i[dict_ind + 1];
                            offset_t len = end_offset - start_offset;
                            std::string s2(&data_i[start_offset], len);
                            // Get the address and length of the cumulative
                            // result
                            offset_t start_offset_org = offsets_i[org_ind];
                            offset_t end_offset_org = offsets_i[org_ind + 1];
                            offset_t len_org =
                                end_offset_org - start_offset_org;
                            std::string s1(&data_i[start_offset_org], len_org);
                            idxmin_dict(org_ind, dict_ind, s1, s2,
                                        getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            org_ind = dict_ind;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                            indices_arr->set_null_bit(i_grp, true);
                        }
                    } else {
                        // If we have an NA value mark this group as
                        // done and update the index
                        getv<uint64_t>(index_pos, i_grp) = i;
                        // set the null bit for the count so we know
                        // to stop visiting the group. The data is still
                        // valid.
                        index_pos->set_null_bit(i_grp, false);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::last: {
            // Define a specialized implementation of last
            // so we avoid allocating for the underlying strings.
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->info2->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
                    int32_t& org_ind = getv<int32_t>(indices_arr, i_grp);
                    if (out_bit_set) {
                        aggfunc<int32_t, Bodo_CTypes::STRING,
                                Bodo_FTypes::last>::apply(org_ind, dict_ind);
                    } else {
                        org_ind = dict_ind;
                        SetBitTo(V.data(), i_grp, true);
                        indices_arr->set_null_bit(i_grp, true);
                    }
                }
            }
            break;
        }
        default:
            // Populate the new indices array.
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->info2->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    if (ftype == Bodo_FTypes::first && out_bit_set) continue;
                    int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
                    int32_t& org_ind = getv<int32_t>(indices_arr, i_grp);
                    if (out_bit_set) {
                        // Get the address and length of the new value to be
                        // compared
                        offset_t start_offset = offsets_i[dict_ind];
                        offset_t end_offset = offsets_i[dict_ind + 1];
                        offset_t len = end_offset - start_offset;
                        std::string s2(&data_i[start_offset], len);
                        // Get the address and length of the cumulative result
                        offset_t start_offset_org = offsets_i[org_ind];
                        offset_t end_offset_org = offsets_i[org_ind + 1];
                        offset_t len_org = end_offset_org - start_offset_org;
                        std::string s1(&data_i[start_offset_org], len_org);
                        aggdict<ftype>::apply(org_ind, dict_ind, s1, s2);
                    } else {
                        org_ind = dict_ind;
                        SetBitTo(V.data(), i_grp, true);
                        indices_arr->set_null_bit(i_grp, true);
                    }
                }
            }
    }
    // TODO: Avoid creating the output dict array if have
    // idxmin/idxmax/idxmin_na_first/idxmax_na_first
    // Start at 1 since 0 is is returned by the hashmap if data needs to be
    // inserted.
    int32_t k = 1;
    UNORD_MAP_CONTAINER<int32_t, int32_t>
        old_to_new;  // Maps old index to new index
    old_to_new.reserve(num_groups);
    for (size_t i = 0; i < num_groups; i++) {
        // check if the value for the group is NaN
        if (!indices_arr->get_null_bit(i)) {
            continue;
        }
        // Insert 0 into the map if key is not in it.
        int32_t& old_ind = getv<int32_t>(indices_arr, i);
        int32_t& new_ind = old_to_new[old_ind];
        if (new_ind == 0) {
            new_ind =
                k++;  // Updates the value in the map without another lookup
        }
        old_ind =
            old_to_new[old_ind] - 1;  // map back from 1-indexing to 0-indexing
    }
    // Create new dict string array from map
    size_t n_dict = old_to_new.size();
    n_bytes = (n_dict + 7) >> 3;
    std::vector<uint8_t> bitmask_vec(n_bytes, 0);
    std::vector<std::string> ListString(n_dict);
    for (auto& it : old_to_new) {
        offset_t start_offset = offsets_i[it.first];
        offset_t end_offset = offsets_i[it.first + 1];
        offset_t len = end_offset - start_offset;
        std::string val(&data_i[start_offset], len);
        ListString[it.second - 1] = val;  // -1 to account for the 1 offset
        SetBitTo(bitmask_vec.data(), it.second - 1, true);
    }
    array_info* dict_arr = create_string_array(bitmask_vec, ListString);
    return create_dict_string_array(dict_arr, indices_arr, num_groups);
}

/**
 * @brief Apply sum operation on a dictionary-encoded string column. The
 * partial_sum trick used for regular string columns to calculate the offsets is
 * also used here. Returns a regulare string array instead of a
 * dictionary-encoded string array.
 *
 * @tparam F
 * @tparam ftype
 * @param in_col: the input dictionary-encoded string array
 * @param out_col: the output string array
 * @param grp_info: groupby information
 * @param f: a function that returns the group index given the row index
 * @return array_info*
 */
template <typename F, int ftype>
array_info* apply_sum_to_column_dict(array_info* in_col, array_info* out_col,
                                     const grouping_info& grp_info, F f) {
    size_t num_groups = grp_info.num_groups;
    int64_t n_chars = 0;
    size_t n_bytes = (num_groups + 7) >> 3;

    // calculate the total number of characters in the dict-encoded array
    // and find offsets for each output string
    // every string has a start and end offset so len(offsets) == (len(data) +
    // 1)
    std::vector<offset_t> str_offsets(num_groups + 1, 0);
    char* data_i = in_col->info1->data1;
    offset_t* offsets_i = (offset_t*)in_col->info1->data2;
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->info2->get_null_bit(i)) {
            int32_t dict_ind = getv<int32_t>(in_col->info2, i);
            offset_t len = offsets_i[dict_ind + 1] - offsets_i[dict_ind];
            str_offsets[i_grp + 1] += len;
            n_chars += len;
        }
    }

    array_info* out_arr = alloc_string_array(num_groups, n_chars, 0);
    memset(out_arr->null_bitmask, 0xff, n_bytes);  // null not possible
    char* data_o = out_arr->data1;
    offset_t* offsets_o = (offset_t*)out_arr->data2;

    std::partial_sum(str_offsets.begin(), str_offsets.end(),
                     str_offsets.begin());
    memcpy(offsets_o, str_offsets.data(), (num_groups + 1) * sizeof(offset_t));

    // copy characters to output
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->info2->get_null_bit(i)) {
            int32_t dict_ind = getv<int32_t>(in_col->info2, i);
            offset_t len = offsets_i[dict_ind + 1] - offsets_i[dict_ind];
            memcpy(&data_o[str_offsets[i_grp]], data_i + offsets_i[dict_ind],
                   len);
            str_offsets[i_grp] += len;
        }
    }
    return out_arr;
}

/**
 * Apply a function to a column(s), save result to (possibly reduced) output
 * column(s) Semantics of this function right now vary depending on function
 * type (ftype).
 *
 * @param column containing input values
 * @param output column
 * @param auxiliary input/output columns used for mean, var, std
 * @param maps row numbers in input columns to group numbers (for reduction
 * operations)
 */
template <typename F, typename T, int ftype, int dtype>
void apply_to_column_F(array_info* in_col, array_info* out_col,
                       std::vector<array_info*>& aux_cols,
                       const grouping_info& grp_info, F f, bool use_sql_rules) {
    switch (in_col->arr_type) {
        case bodo_array_type::CATEGORICAL:
            if (ftype == Bodo_FTypes::count) {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        T& val = getv<T>(in_col, i);
                        if (!isnan_categorical<T, dtype>(val)) {
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp), val);
                        }
                    }
                }
                return;
            } else if (ftype == Bodo_FTypes::min ||
                       ftype == Bodo_FTypes::last) {
                // NOTE: Bodo_FTypes::max is handled for categorical type
                // since NA is -1.
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        T& val = getv<T>(in_col, i);
                        if (!isnan_categorical<T, dtype>(val)) {
                            aggfunc<T, dtype, ftype>::apply(
                                getv<T>(out_col, i_grp), val);
                        }
                    }
                }
                // aggfunc_output_initialize_kernel, min defaults
                // to num_categories if all entries are NA
                // this needs to be replaced with -1
                if (ftype == Bodo_FTypes::min) {
                    for (size_t i = 0; i < out_col->length; i++) {
                        T& val = getv<T>(out_col, i);
                        set_na_if_num_categories<T, dtype>(
                            val, out_col->num_categories);
                    }
                }
                return;
            } else if (ftype == Bodo_FTypes::first) {
                int64_t n_bytes = ((out_col->length + 7) >> 3);
                std::vector<uint8_t> bitmask_vec(n_bytes, 0);
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    T val = getv<T>(in_col, i);
                    if ((i_grp != -1) && !GetBit(bitmask_vec.data(), i_grp) &&
                        !isnan_categorical<T, dtype>(val)) {
                        getv<T>(out_col, i_grp) = val;
                        SetBitTo(bitmask_vec.data(), i_grp, true);
                    }
                }
            }
        case bodo_array_type::NUMPY:
            if (ftype == Bodo_FTypes::mean) {
                array_info* count_col = aux_cols[0];
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        mean_agg<T, dtype>::apply(
                            getv<double>(out_col, i_grp), getv<T>(in_col, i),
                            getv<uint64_t>(count_col, i_grp));
                        // Mean always has a nullable output even
                        // if there is a numpy input.
                        out_col->set_null_bit(i_grp, true);
                        count_col->set_null_bit(i_grp, true);
                    }
                }
            } else if (ftype == Bodo_FTypes::mean_eval) {
                for (size_t i = 0; i < in_col->length; i++) {
                    mean_eval(getv<double>(out_col, i),
                              getv<uint64_t>(in_col, i));
                }
            } else if (ftype == Bodo_FTypes::var) {
                array_info* count_col = aux_cols[0];
                array_info* mean_col = aux_cols[1];
                array_info* m2_col = aux_cols[2];
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        var_agg<T, dtype>::apply(
                            getv<T>(in_col, i),
                            getv<uint64_t>(count_col, i_grp),
                            getv<double>(mean_col, i_grp),
                            getv<double>(m2_col, i_grp));
                        // Var always has a nullable output even
                        // if there is a numpy input
                        out_col->set_null_bit(i_grp, true);
                        count_col->set_null_bit(i_grp, true);
                        mean_col->set_null_bit(i_grp, true);
                        m2_col->set_null_bit(i_grp, true);
                    }
                }
            } else if (ftype == Bodo_FTypes::var_eval) {
                array_info* count_col = aux_cols[0];
                array_info* m2_col = aux_cols[2];
                for (size_t i = 0; i < in_col->length; i++)
                    var_eval(getv<double>(out_col, i),
                             getv<uint64_t>(count_col, i),
                             getv<double>(m2_col, i));
            } else if (ftype == Bodo_FTypes::std_eval) {
                array_info* count_col = aux_cols[0];
                array_info* m2_col = aux_cols[2];
                for (size_t i = 0; i < in_col->length; i++)
                    std_eval(getv<double>(out_col, i),
                             getv<uint64_t>(count_col, i),
                             getv<double>(m2_col, i));
            } else if (ftype == Bodo_FTypes::count) {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1)
                        count_agg<T, dtype>::apply(
                            getv<int64_t>(out_col, i_grp), getv<T>(in_col, i));
                }
            } else if (ftype == Bodo_FTypes::first) {
                // create a temporary bitmask to know if we have set a
                // value for each row/group
                int64_t n_bytes = ((out_col->length + 7) >> 3);
                std::vector<uint8_t> bitmask_vec(n_bytes, 0);
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    T val = getv<T>(in_col, i);
                    if ((i_grp != -1) && !GetBit(bitmask_vec.data(), i_grp) &&
                        !isnan_alltype<T, dtype>(val)) {
                        getv<T>(out_col, i_grp) = val;
                        SetBitTo(bitmask_vec.data(), i_grp, true);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmax) {
                array_info* index_pos = aux_cols[0];
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        idxmax_agg<T, dtype>::apply(
                            getv<T>(out_col, i_grp), getv<T>(in_col, i),
                            getv<uint64_t>(index_pos, i_grp), i);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmin) {
                array_info* index_pos = aux_cols[0];
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        idxmin_agg<T, dtype>::apply(
                            getv<T>(out_col, i_grp), getv<T>(in_col, i),
                            getv<uint64_t>(index_pos, i_grp), i);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmax_na_first) {
                // Datetime64 and Timedelta64 represent NA values in the array.
                // We need to handle these the same as the nullable case. For
                // all other NA like values (e.g. floats) the relative value of
                // NaN should be based upon wherever they would be sorted. This
                // may need to be handled to match SQL, but is a separate issue.
                array_info* index_pos = aux_cols[0];
                if (dtype == Bodo_CTypes::DATETIME ||
                    dtype == Bodo_CTypes::TIMEDELTA) {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        // If we are putting NA values first then we stop
                        // visiting a group once we see a NA value. We
                        // initialize the output values to all be non-NA values
                        // to ensure this.
                        if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                            // If we see NA/NaN mark this as the match.
                            T input_val = getv<T>(in_col, i);
                            if (!isnan_alltype<T, dtype>(input_val)) {
                                idxmax_agg<T, dtype>::apply(
                                    getv<T>(out_col, i_grp), input_val,
                                    getv<uint64_t>(index_pos, i_grp), i);
                            } else {
                                // If we have an NA value mark this group as
                                // done and update the index
                                getv<uint64_t>(index_pos, i_grp) = i;
                                // set the null bit for the count so we know
                                // to stop visiting the group. The data is still
                                // valid.
                                index_pos->set_null_bit(i_grp, false);
                            }
                        }
                    }
                } else {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if (i_grp != -1) {
                            idxmax_agg<T, dtype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                getv<uint64_t>(index_pos, i_grp), i);
                        }
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmin_na_first) {
                // Datetime64 and Timedelta64 represent NA values in the array.
                // We need to handle these the same as the nullable case. For
                // all other NA like values (e.g. floats) the relative value of
                // NaN should be based upon wherever they would be sorted. This
                // may need to be handled to match SQL, but is a separate issue.
                array_info* index_pos = aux_cols[0];
                if (dtype == Bodo_CTypes::DATETIME ||
                    dtype == Bodo_CTypes::TIMEDELTA) {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        // If we are putting NA values first then we stop
                        // visiting a group once we see a NA value. We
                        // initialize the output values to all be non-NA values
                        // to ensure this.
                        if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                            // If we see NA/NaN mark this as the match.
                            T input_val = getv<T>(in_col, i);
                            if (!isnan_alltype<T, dtype>(input_val)) {
                                idxmin_agg<T, dtype>::apply(
                                    getv<T>(out_col, i_grp), input_val,
                                    getv<uint64_t>(index_pos, i_grp), i);
                            } else {
                                // If we have an NA value mark this group as
                                // done and update the index
                                getv<uint64_t>(index_pos, i_grp) = i;
                                // set the null bit for the count so we know
                                // to stop visiting the group. The data is still
                                // valid.
                                index_pos->set_null_bit(i_grp, false);
                            }
                        }
                    }
                } else {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if (i_grp != -1) {
                            idxmin_agg<T, dtype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                getv<uint64_t>(index_pos, i_grp), i);
                        }
                    }
                }
            } else if (ftype == Bodo_FTypes::boolor_agg) {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if ((i_grp != -1)) {
                        T val2 = getv<T>(in_col, i);
                        // Skip NA values
                        if (!isnan_alltype<T, dtype>(val2)) {
                            bool_aggfunc<T, dtype, ftype>::apply(
                                getv<bool>(out_col, i_grp), val2);
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        aggfunc<T, dtype, ftype>::apply(getv<T>(out_col, i_grp),
                                                        getv<T>(in_col, i));
                    }
                }
            }
            return;
        // For DICT(dictionary-encoded string) we support count
        case bodo_array_type::DICT:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if (i_grp != -1 && in_col->info2->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp),
                                getv<T>(in_col, i));
                    }
                    return;
                }
                // optimized groupby sum for strings (concatenation)
                case Bodo_FTypes::sum: {
                    array_info* new_out_col =
                        apply_sum_to_column_dict<F, ftype>(in_col, out_col,
                                                           grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
                }
                default:
                    array_info* new_out_col = apply_to_column_dict<F, ftype>(
                        in_col, out_col, aux_cols, grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
            }
        // for list strings, we are supporting count, sum, max, min, first, last
        case bodo_array_type::LIST_STRING:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp),
                                getv<T>(in_col, i));
                    }
                    return;
                }
                default:
                    array_info* new_out_col =
                        apply_to_column_list_string<F, ftype>(in_col, out_col,
                                                              grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
            }

        // For the STRING we compute the count, sum, max, min, first, last,
        // idxmin, idxmax, idxmin_na_first, idxmax_na_first
        case bodo_array_type::STRING:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp),
                                getv<T>(in_col, i));
                    }
                    return;
                }
                // optimized groupby sum for strings (concatenation)
                case Bodo_FTypes::sum: {
                    array_info* new_out_col =
                        apply_sum_to_column_string<F, ftype>(in_col, out_col,
                                                             grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
                }
                default:
                    array_info* new_out_col = apply_to_column_string<F, ftype>(
                        in_col, out_col, aux_cols, grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
            }
        case bodo_array_type::NULLABLE_INT_BOOL:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp),
                                getv<T>(in_col, i));
                    }
                    return;
                }
                case Bodo_FTypes::mean: {
                    array_info* count_col = aux_cols[0];
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            mean_agg<T, dtype>::apply(
                                getv<double>(out_col, i_grp),
                                getv<T>(in_col, i),
                                getv<uint64_t>(count_col, i_grp));
                            out_col->set_null_bit(i_grp, true);
                            count_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::mean_eval: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        if (in_col->get_null_bit(i) &&
                            getv<uint64_t>(in_col, i) > 0) {
                            mean_eval(getv<double>(out_col, i),
                                      getv<uint64_t>(in_col, i));
                            out_col->set_null_bit(i, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::var: {
                    array_info* count_col = aux_cols[0];
                    array_info* mean_col = aux_cols[1];
                    array_info* m2_col = aux_cols[2];
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            var_agg<T, dtype>::apply(
                                getv<T>(in_col, i),
                                getv<uint64_t>(count_col, i_grp),
                                getv<double>(mean_col, i_grp),
                                getv<double>(m2_col, i_grp));
                            out_col->set_null_bit(i_grp, true);
                            count_col->set_null_bit(i_grp, true);
                            mean_col->set_null_bit(i_grp, true);
                            m2_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::var_eval: {
                    array_info* count_col = aux_cols[0];
                    array_info* m2_col = aux_cols[2];
                    for (size_t i = 0; i < in_col->length; i++) {
                        // TODO: Template on use_sql_rules
                        if (count_col->get_null_bit(i) &&
                            getv<uint64_t>(count_col, i) > 1) {
                            var_eval(getv<double>(out_col, i),
                                     getv<uint64_t>(count_col, i),
                                     getv<double>(m2_col, i));
                            out_col->set_null_bit(i, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::std_eval: {
                    array_info* count_col = aux_cols[0];
                    array_info* m2_col = aux_cols[2];
                    for (size_t i = 0; i < in_col->length; i++) {
                        // TODO: Template on use_sql_rules
                        if (count_col->get_null_bit(i) &&
                            getv<uint64_t>(count_col, i) > 1) {
                            std_eval(getv<double>(out_col, i),
                                     getv<uint64_t>(count_col, i),
                                     getv<double>(m2_col, i));
                            out_col->set_null_bit(i, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::first:
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && !out_col->get_null_bit(i_grp) &&
                            in_col->get_null_bit(i)) {
                            getv<T>(out_col, i_grp) = getv<T>(in_col, i);
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                case Bodo_FTypes::idxmax:
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            idxmax_agg<T, dtype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                getv<uint64_t>(aux_cols[0], i_grp), i);
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                case Bodo_FTypes::idxmax_na_first: {
                    array_info* index_pos = aux_cols[0];
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        // If we are putting NA values first then we stop
                        // visiting a group once we see a NA value. We
                        // initialize the output values to all be non-NA values
                        // to ensure this.
                        if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                            if (in_col->get_null_bit(i)) {
                                idxmax_agg<T, dtype>::apply(
                                    getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                    getv<uint64_t>(index_pos, i_grp), i);
                            } else {
                                // If we have an NA value mark this group as
                                // done and update the index
                                getv<uint64_t>(index_pos, i_grp) = i;
                                // set the null bit for the count so we know
                                // to stop visiting the group. The data is still
                                // valid.
                                index_pos->set_null_bit(i_grp, false);
                            }
                        }
                    }
                    return;
                }
                case Bodo_FTypes::idxmin:
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            idxmin_agg<T, dtype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                getv<uint64_t>(aux_cols[0], i_grp), i);
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                case Bodo_FTypes::idxmin_na_first: {
                    array_info* index_pos = aux_cols[0];
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        // If we are putting NA values first then we stop
                        // visiting a group once we see a NA value. We
                        // initialize the output values to all be non-NA values
                        // to ensure this.
                        if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                            if (in_col->get_null_bit(i)) {
                                idxmin_agg<T, dtype>::apply(
                                    getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                    getv<uint64_t>(index_pos, i_grp), i);
                            } else {
                                // If we have an NA value mark this group as
                                // done and update the index
                                getv<uint64_t>(index_pos, i_grp) = i;
                                // set the null bit for the count so we know
                                // to stop visiting the group. The data is still
                                // valid.
                                index_pos->set_null_bit(i_grp, false);
                            }
                        }
                    }
                    return;
                }
                case Bodo_FTypes::boolor_agg:
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            bool_aggfunc<T, dtype, ftype>::apply(
                                getv<bool>(out_col, i_grp), getv<T>(in_col, i));
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                default: {
                    if (ftype == Bodo_FTypes::sum &&
                        dtype == Bodo_CTypes::_BOOL) {
                        for (size_t i = 0; i < in_col->length; i++) {
                            int64_t i_grp = f(i);
                            if ((i_grp != -1) && in_col->get_null_bit(i)) {
                                bool_sum<bool, dtype>::apply(
                                    getv<int64_t>(out_col, i_grp),
                                    getv<bool>(in_col, i));
                            }
                            // The output is never null for count_if, which is
                            // implemented by bool_sum.
                            // TODO: Replace with an explicit count_if function
                            // to avoid NULL issues with SUM(BOOLEAN) column.
                            out_col->set_null_bit(i_grp, true);
                        }
                        return;
                    }
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            aggfunc<T, dtype, ftype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i));
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                }
            }
        default:
            Bodo_PyErr_SetString(PyExc_RuntimeError,
                                 "apply_to_column: incorrect array type");
            return;
    }
}

template <typename T, int ftype, int dtype>
void apply_to_column(array_info* in_col, array_info* out_col,
                     std::vector<array_info*>& aux_cols,
                     const grouping_info& grp_info, bool use_sql_rules) {
    auto f = [&](int64_t const& i_row) -> int64_t {
        return grp_info.row_to_group[i_row];
    };
    return apply_to_column_F<decltype(f), T, ftype, dtype>(
        in_col, out_col, aux_cols, grp_info, f, use_sql_rules);
}

/**
 * Invokes the correct template instance of apply_to_column depending on
 * function (ftype) and dtype. See 'apply_to_column'
 *
 * @param column containing input values
 * @param output column
 * @param auxiliary input/output columns used for mean, var, std
 * @param maps row numbers in input columns to group numbers (for reduction
 * operations)
 * @param function to apply
 */
void do_apply_to_column(array_info* in_col, array_info* out_col,
                        std::vector<array_info*>& aux_cols,
                        const grouping_info& grp_info, int ftype,
                        bool use_sql_rules) {
    // size operation is the same regardless of type of data.
    // Hence, just compute number of rows per group here.
    if (ftype == Bodo_FTypes::size) {
        for (size_t i = 0; i < in_col->length; i++) {
            int64_t i_grp = grp_info.row_to_group[i];
            if (i_grp != -1)
                size_agg<int64_t, Bodo_CTypes::INT64>::apply(
                    getv<int64_t>(out_col, i_grp), getv<int64_t>(in_col, i));
        }
        return;
    }
    if (in_col->arr_type == bodo_array_type::STRING ||
        in_col->arr_type == bodo_array_type::LIST_STRING ||
        in_col->arr_type == bodo_array_type::DICT) {
        switch (ftype) {
            // NOTE: The int template argument is not used in this call to
            // apply_to_column
            case Bodo_FTypes::sum:
                return apply_to_column<int, Bodo_FTypes::sum,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::min:
                return apply_to_column<int, Bodo_FTypes::min,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::max:
                return apply_to_column<int, Bodo_FTypes::max,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::first:
                return apply_to_column<int, Bodo_FTypes::first,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::last:
                return apply_to_column<int, Bodo_FTypes::last,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::idxmin:
                // idxmin handles the na_last case
                return apply_to_column<int, Bodo_FTypes::idxmin,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::idxmax:
                // idxmax handles the na_last case
                return apply_to_column<int, Bodo_FTypes::idxmax,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::idxmin_na_first:
                return apply_to_column<int, Bodo_FTypes::idxmin_na_first,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::idxmax_na_first:
                return apply_to_column<int, Bodo_FTypes::idxmax_na_first,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
        }
    }
    if (ftype == Bodo_FTypes::count) {
        switch (in_col->dtype) {
            case Bodo_CTypes::FLOAT32:
                // data will only be used to check for nans
                return apply_to_column<float, Bodo_FTypes::count,
                                       Bodo_CTypes::FLOAT32>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_CTypes::FLOAT64:
                // data will only be used to check for nans
                return apply_to_column<double, Bodo_FTypes::count,
                                       Bodo_CTypes::FLOAT64>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_CTypes::DATETIME:
            case Bodo_CTypes::TIMEDELTA:
                // data will only be used to check for NATs
                return apply_to_column<int64_t, Bodo_FTypes::count,
                                       Bodo_CTypes::DATETIME>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            default:
                // data will be ignored in this case, so type doesn't matter
                return apply_to_column<int8_t, Bodo_FTypes::count,
                                       Bodo_CTypes::INT8>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
        }
    }

    switch (in_col->dtype) {
        case Bodo_CTypes::_BOOL:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<bool, Bodo_FTypes::sum,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<bool, Bodo_FTypes::min,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<bool, Bodo_FTypes::max,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<bool, Bodo_FTypes::prod,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<bool, Bodo_FTypes::first,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<bool, Bodo_FTypes::last,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<bool, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<bool, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<bool, Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<bool, Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<bool, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                default:
                    Bodo_PyErr_SetString(
                        PyExc_RuntimeError,
                        "unsupported aggregation for boolean type column");
                    return;
            }
        case Bodo_CTypes::INT8:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int8_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int8_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int8_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int8_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int8_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int8_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int8_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int8_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<int8_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::UINT8:
            switch (ftype) {
                case Bodo_FTypes::sum: {
                    return apply_to_column<uint8_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                }
                case Bodo_FTypes::min:
                    return apply_to_column<uint8_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<uint8_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint8_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<uint8_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<uint8_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint8_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint8_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint8_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint8_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<uint8_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<uint8_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<uint8_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::INT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int16_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int16_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int16_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int16_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int16_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int16_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int16_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int16_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int16_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int16_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int16_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int16_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<int16_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::UINT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint16_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<uint16_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<uint16_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint16_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<uint16_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<uint16_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint16_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint16_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint16_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint16_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<uint16_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<uint16_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<uint16_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::INT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int32_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int32_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int32_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int32_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int32_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int32_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int32_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int32_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int32_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int32_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int32_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int32_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<int32_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::UINT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint32_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<uint32_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<uint32_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint32_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<uint32_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<uint32_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint32_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint32_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint32_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint32_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<uint32_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<uint32_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<uint32_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::INT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<int64_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::UINT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<uint64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<uint64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<uint64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<uint64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<uint64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<uint64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<uint64_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::DATE:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        // TODO: [BE-4106] Split Time into Time32 and Time64
        case Bodo_CTypes::TIME:
            // NOTE: only min/max/first/last are supported.
            // Others, the column will be dropped on Python side.
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::DATETIME:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::TIMEDELTA:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::FLOAT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<float, Bodo_FTypes::sum,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<float, Bodo_FTypes::min,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<float, Bodo_FTypes::max,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<float, Bodo_FTypes::prod,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<float, Bodo_FTypes::first,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<float, Bodo_FTypes::last,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<float, Bodo_FTypes::mean,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<float, Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<float, Bodo_FTypes::var,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<float, Bodo_FTypes::var_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<float, Bodo_FTypes::std_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<float, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<float, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<float, Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<float, Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<float, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::FLOAT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<double, Bodo_FTypes::sum,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<double, Bodo_FTypes::min,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<double, Bodo_FTypes::max,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<double, Bodo_FTypes::prod,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<double, Bodo_FTypes::first,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<double, Bodo_FTypes::last,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<double, Bodo_FTypes::mean,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<double, Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<double, Bodo_FTypes::var,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<double, Bodo_FTypes::var_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<double, Bodo_FTypes::std_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<double, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<double, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<double, Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<double, Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<double, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::DECIMAL:
            switch (ftype) {
                case Bodo_FTypes::first:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::last,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::min,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::max,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::mean,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::var,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::var_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::std_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        default:
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                (std::string("do_apply_to_column: unsuported array dtype: ") +
                 std::string(GetDtype_as_string(in_col->dtype)))
                    .c_str());
            return;
    }
}

/**
 * Returns the array type and dtype required for output columns based on the
 * aggregation function and input dtype.
 *
 * @param function identifier
 * @param[in,out] array type (caller sets a default, this function only changes
 * in certain cases)
 * @param[in,out] output dtype (caller sets a default, this function only
 * changes in certain cases)
 * @param true if column is key column (in this case ignore because output type
 * will be the same)
 */
void get_groupby_output_dtype(int ftype,
                              bodo_array_type::arr_type_enum& array_type,
                              Bodo_CTypes::CTypeEnum& dtype, bool is_key,
                              bool is_combine) {
    if (is_combine) ftype = combine_funcs[ftype];
    if (is_key) return;
    switch (ftype) {
        case Bodo_FTypes::nunique:
        case Bodo_FTypes::count:
        case Bodo_FTypes::size:
        case Bodo_FTypes::ngroup:
            array_type = bodo_array_type::NUMPY;
            dtype = Bodo_CTypes::INT64;
            return;
        case Bodo_FTypes::median:
        case Bodo_FTypes::mean:
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
            array_type = bodo_array_type::NULLABLE_INT_BOOL;
            dtype = Bodo_CTypes::FLOAT64;
            return;
        case Bodo_FTypes::cumsum:
        case Bodo_FTypes::sum:
            // This is safe even for cumsum because a boolean cumsum is not yet
            // supported on the Python side, so an error will be raised there
            if (dtype == Bodo_CTypes::_BOOL) {
                array_type = bodo_array_type::NULLABLE_INT_BOOL;
                dtype = Bodo_CTypes::INT64;
            }
            if (dtype == Bodo_CTypes::STRING) {
                array_type = bodo_array_type::STRING;
            }
            return;
        case Bodo_FTypes::boolor_agg:
            array_type = bodo_array_type::NULLABLE_INT_BOOL;
            dtype = Bodo_CTypes::_BOOL;
            return;
        case Bodo_FTypes::row_number:
            array_type = bodo_array_type::NUMPY;
            dtype = Bodo_CTypes::UINT64;
            return;
        case Bodo_FTypes::min_row_number_filter:
            array_type = bodo_array_type::NUMPY;
            dtype = Bodo_CTypes::_BOOL;
            return;
        default:
            return;
    }
}

/*
 An instance of GroupbyPipeline class manages a groupby operation. In a
 groupby operation, an arbitrary number of functions can be applied to each
 input column. The functions can vary between input columns. Each combination
 of (input column, function) is an operation that produces a column in the
 output table. The computation of each (input column, function) pair is
 encapsulated in what is called a "column set" (for lack of a better name).
 There are different column sets for different types of operations (e.g. var,
 mean, median, udfs, basic operations...). Each column set creates,
 initializes, operates on and manages the arrays needed to perform its
 computation. Different column set types may require different number of
 columns and dtypes. The main control flow of groupby is in
 GroupbyPipeline::run(). It invokes update, shuffle, combine and eval steps
 (as needed), and these steps iterate through the column sets and invoke
 their operations.
*/

/*
 * This is the base column set class which is used by most operations (like
 * sum, prod, count, etc.). Several subclasses also rely on some of the methods
 * of this base class.
 */
class BasicColSet {
   public:
    /**
     * Construct column set corresponding to function of type ftype applied to
     * the input column in_col
     * @param in_col input column of groupby associated with this column set
     * @param ftype function associated with this column set
     * @param combine_step tells the column set whether GroupbyPipeline is going
     * to perform a combine operation or not. If false, this means that either
     *        shuffling is not necessary or that it will be done at the
     *        beginning of the pipeline.
     * @param use_sql_rules tells the column set whether to use SQL or Pandas
     * rules
     */
    BasicColSet(array_info* in_col, int ftype, bool combine_step,
                bool use_sql_rules)
        : in_col(in_col),
          ftype(ftype),
          combine_step(combine_step),
          use_sql_rules(use_sql_rules) {}
    virtual ~BasicColSet() {}

    /**
     * Allocate my columns for update step.
     * @param number of groups found in the input table
     * @param[in,out] vector of columns of update table. This method adds
     *                columns to this vector.
     */
    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        bodo_array_type::arr_type_enum arr_type = in_col->arr_type;
        Bodo_CTypes::CTypeEnum dtype = in_col->dtype;
        int64_t num_categories = in_col->num_categories;
        // calling this modifies arr_type and dtype
        bool is_combine = false;
        get_groupby_output_dtype(ftype, arr_type, dtype, false, is_combine);
        out_cols.push_back(
            alloc_array(num_groups, 1, 1, arr_type, dtype, 0, num_categories));
        update_cols.push_back(out_cols.back());
    }

    /**
     * Perform update step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void update(const std::vector<grouping_info>& grp_infos) {
        std::vector<array_info*> aux_cols;
        aggfunc_output_initialize(update_cols[0], ftype, use_sql_rules);
        do_apply_to_column(in_col, update_cols[0], aux_cols, grp_infos[0],
                           ftype, use_sql_rules);
    }

    /**
     * When GroupbyPipeline shuffles the table after update, the column set
     * needs to be updated with the columns from the new shuffled table. This
     * method is called by GroupbyPipeline with an iterator pointing to my
     * first column. The column set will update its columns and return an
     * iterator pointing to the next set of columns.
     * @param iterator pointing to the first column in this column set
     */
    virtual typename std::vector<array_info*>::iterator update_after_shuffle(
        typename std::vector<array_info*>::iterator& it) {
        for (size_t i_col = 0; i_col < update_cols.size(); i_col++) {
            update_cols[i_col] = *(it++);
        }
        return it;
    }

    /**
     * Allocate my columns for combine step.
     * @param number of groups found in the input table (which is the update
     * table)
     * @param[in,out] vector of columns of combine table. This method adds
     *                columns to this vector.
     */
    virtual void alloc_combine_columns(size_t num_groups,
                                       std::vector<array_info*>& out_cols) {
        for (auto col : update_cols) {
            bodo_array_type::arr_type_enum arr_type = col->arr_type;
            Bodo_CTypes::CTypeEnum dtype = col->dtype;
            int64_t num_categories = col->num_categories;
            // calling this modifies arr_type and dtype
            bool is_combine = true;
            get_groupby_output_dtype(ftype, arr_type, dtype, false, is_combine);
            out_cols.push_back(alloc_array(num_groups, 1, 1, arr_type, dtype, 0,
                                           num_categories));
            combine_cols.push_back(out_cols.back());
        }
    }

    /**
     * Perform combine step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void combine(const grouping_info& grp_info) {
        Bodo_FTypes::FTypeEnum combine_ftype =
            static_cast<Bodo_FTypes::FTypeEnum>(combine_funcs[ftype]);
        std::vector<array_info*> aux_cols(combine_cols.begin() + 1,
                                          combine_cols.end());
        for (auto col : combine_cols)
            aggfunc_output_initialize(col, combine_ftype, use_sql_rules);
        do_apply_to_column(update_cols[0], combine_cols[0], aux_cols, grp_info,
                           combine_ftype, use_sql_rules);
    }

    /**
     * Perform eval step for this column set. This will fill the output column
     * with the final result of the aggregation operation corresponding to this
     * column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void eval(const grouping_info& grp_info) {}

    /**
     * Obtain the final output column resulting from the groupby operation on
     * this column set. This will free all other intermediate or auxiliary
     * columns (if any) used by the column set (like reduction variables).
     */
    virtual array_info* getOutputColumn() {
        std::vector<array_info*>* mycols;
        if (combine_step)
            mycols = &combine_cols;
        else
            mycols = &update_cols;
        array_info* out_col = mycols->at(0);
        for (auto it = mycols->begin() + 1; it != mycols->end(); it++) {
            array_info* a = *it;
            delete_info_decref_array(a);
        }
        return out_col;
    }

   protected:
    array_info* in_col;  // the input column (from groupby input table) to which
                         // this column set corresponds to
    int ftype;
    bool combine_step;   // GroupbyPipeline is going to perform a combine
                         // operation or not
    bool use_sql_rules;  // Use SQL rules for aggregation or Pandas?
    std::vector<array_info*> update_cols;   // columns for update step
    std::vector<array_info*> combine_cols;  // columns for combine step
};

class MeanColSet : public BasicColSet {
   public:
    MeanColSet(array_info* in_col, bool combine_step, bool use_sql_rules)
        : BasicColSet(in_col, Bodo_FTypes::mean, combine_step, use_sql_rules) {}
    virtual ~MeanColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        array_info* c1 =
            alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::FLOAT64, 0,
                        0);  // for sum and result
        array_info* c2 =
            alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::UINT64, 0,
                        0);  // for counts
        out_cols.push_back(c1);
        out_cols.push_back(c2);
        this->update_cols.push_back(c1);
        this->update_cols.push_back(c2);
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        std::vector<array_info*> aux_cols = {this->update_cols[1]};
        aggfunc_output_initialize(this->update_cols[0], this->ftype,
                                  use_sql_rules);
        aggfunc_output_initialize(this->update_cols[1], this->ftype,
                                  use_sql_rules);
        do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                           grp_infos[0], this->ftype, use_sql_rules);
    }

    virtual void combine(const grouping_info& grp_info) {
        std::vector<array_info*> aux_cols;
        aggfunc_output_initialize(this->combine_cols[0], this->ftype,
                                  use_sql_rules);
        // Initialize the output as mean to match the nullable behavior.
        aggfunc_output_initialize(this->combine_cols[1], this->ftype,
                                  use_sql_rules);
        do_apply_to_column(this->update_cols[0], this->combine_cols[0],
                           aux_cols, grp_info, Bodo_FTypes::sum, use_sql_rules);
        do_apply_to_column(this->update_cols[1], this->combine_cols[1],
                           aux_cols, grp_info, Bodo_FTypes::sum, use_sql_rules);
    }

    virtual void eval(const grouping_info& grp_info) {
        std::vector<array_info*> aux_cols;
        if (this->combine_step) {
            do_apply_to_column(this->combine_cols[1], this->combine_cols[0],
                               aux_cols, grp_info, Bodo_FTypes::mean_eval,
                               use_sql_rules);
        } else {
            do_apply_to_column(this->update_cols[1], this->update_cols[0],
                               aux_cols, grp_info, Bodo_FTypes::mean_eval,
                               use_sql_rules);
        }
    }
};

class IdxMinMaxColSet : public BasicColSet {
   public:
    IdxMinMaxColSet(array_info* in_col, array_info* _index_col, int ftype,
                    bool combine_step, bool use_sql_rules)
        : BasicColSet(in_col, ftype, combine_step, use_sql_rules),
          index_col(_index_col) {}
    virtual ~IdxMinMaxColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        // output column containing index values. dummy for now. will be
        // assigned the real data at the end of update()
        array_info* out_col = alloc_array(num_groups, 1, 1, index_col->arr_type,
                                          index_col->dtype, 0, 0);
        // create array to store min/max value
        array_info* max_col = alloc_array(
            num_groups, 1, 1, this->in_col->arr_type, this->in_col->dtype, 0,
            0);  // for min/max
        // create array to store index position of min/max value
        array_info* index_pos_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::UINT64, 0, 0);
        out_cols.push_back(out_col);
        out_cols.push_back(max_col);
        this->update_cols.push_back(out_col);
        this->update_cols.push_back(max_col);
        this->update_cols.push_back(index_pos_col);
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        array_info* index_pos_col = this->update_cols[2];
        std::vector<array_info*> aux_cols = {index_pos_col};
        if (this->ftype == Bodo_FTypes::idxmax)
            aggfunc_output_initialize(this->update_cols[1], Bodo_FTypes::max,
                                      use_sql_rules);
        if (this->ftype == Bodo_FTypes::idxmin)
            aggfunc_output_initialize(this->update_cols[1], Bodo_FTypes::min,
                                      use_sql_rules);
        aggfunc_output_initialize(index_pos_col, Bodo_FTypes::count,
                                  use_sql_rules);  // zero init
        do_apply_to_column(this->in_col, this->update_cols[1], aux_cols,
                           grp_infos[0], this->ftype, use_sql_rules);

        array_info* real_out_col =
            RetrieveArray_SingleColumn_arr(index_col, index_pos_col);
        array_info* out_col = this->update_cols[0];
        *out_col = std::move(*real_out_col);
        delete real_out_col;
        delete_info_decref_array(index_pos_col);
        this->update_cols.pop_back();
    }

    virtual void alloc_combine_columns(size_t num_groups,
                                       std::vector<array_info*>& out_cols) {
        // output column containing index values. dummy for now. will be
        // assigned the real data at the end of combine()
        array_info* out_col = alloc_array(num_groups, 1, 1, index_col->arr_type,
                                          index_col->dtype, 0, 0);
        // create array to store min/max value
        array_info* max_col = alloc_array(
            num_groups, 1, 1, this->in_col->arr_type, this->in_col->dtype, 0,
            0);  // for min/max
        // create array to store index position of min/max value
        array_info* index_pos_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::UINT64, 0, 0);
        out_cols.push_back(out_col);
        out_cols.push_back(max_col);
        this->combine_cols.push_back(out_col);
        this->combine_cols.push_back(max_col);
        this->combine_cols.push_back(index_pos_col);
    }

    virtual void combine(const grouping_info& grp_info) {
        array_info* index_pos_col = this->combine_cols[2];
        std::vector<array_info*> aux_cols = {index_pos_col};
        if (this->ftype == Bodo_FTypes::idxmax)
            aggfunc_output_initialize(this->combine_cols[1], Bodo_FTypes::max,
                                      use_sql_rules);
        if (this->ftype == Bodo_FTypes::idxmin)
            aggfunc_output_initialize(this->combine_cols[1], Bodo_FTypes::min,
                                      use_sql_rules);
        aggfunc_output_initialize(index_pos_col, Bodo_FTypes::count,
                                  use_sql_rules);  // zero init
        do_apply_to_column(this->update_cols[1], this->combine_cols[1],
                           aux_cols, grp_info, this->ftype, use_sql_rules);

        array_info* real_out_col =
            RetrieveArray_SingleColumn_arr(this->update_cols[0], index_pos_col);
        array_info* out_col = this->combine_cols[0];
        *out_col = std::move(*real_out_col);
        delete real_out_col;
        delete_info_decref_array(index_pos_col);
        this->combine_cols.pop_back();
    }

   private:
    array_info* index_col;
};

class VarStdColSet : public BasicColSet {
   public:
    VarStdColSet(array_info* in_col, int ftype, bool combine_step,
                 bool use_sql_rules)
        : BasicColSet(in_col, ftype, combine_step, use_sql_rules) {}
    virtual ~VarStdColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        if (!this->combine_step) {
            // need to create output column now
            array_info* col = alloc_array(
                num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                Bodo_CTypes::FLOAT64, 0, 0);  // for result
            // Initialize as ftype to match nullable behavior
            aggfunc_output_initialize(col, this->ftype,
                                      use_sql_rules);  // zero initialize
            out_cols.push_back(col);
            this->update_cols.push_back(col);
        }
        array_info* count_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::UINT64, 0, 0);
        array_info* mean_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::FLOAT64, 0, 0);
        array_info* m2_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::FLOAT64, 0, 0);
        aggfunc_output_initialize(count_col, Bodo_FTypes::count,
                                  use_sql_rules);  // zero initialize
        aggfunc_output_initialize(mean_col, Bodo_FTypes::count,
                                  use_sql_rules);  // zero initialize
        aggfunc_output_initialize(m2_col, Bodo_FTypes::count,
                                  use_sql_rules);  // zero initialize
        out_cols.push_back(count_col);
        out_cols.push_back(mean_col);
        out_cols.push_back(m2_col);
        this->update_cols.push_back(count_col);
        this->update_cols.push_back(mean_col);
        this->update_cols.push_back(m2_col);
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        if (!this->combine_step) {
            std::vector<array_info*> aux_cols = {this->update_cols[1],
                                                 this->update_cols[2],
                                                 this->update_cols[3]};
            do_apply_to_column(this->in_col, this->update_cols[1], aux_cols,
                               grp_infos[0], this->ftype, use_sql_rules);
        } else {
            std::vector<array_info*> aux_cols = {this->update_cols[0],
                                                 this->update_cols[1],
                                                 this->update_cols[2]};
            do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                               grp_infos[0], this->ftype, use_sql_rules);
        }
    }

    virtual void alloc_combine_columns(size_t num_groups,
                                       std::vector<array_info*>& out_cols) {
        array_info* col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NULLABLE_INT_BOOL,
                        Bodo_CTypes::FLOAT64, 0,
                        0);  // for result
        // Initialize as ftype to match nullable behavior
        aggfunc_output_initialize(col, this->ftype,
                                  use_sql_rules);  // zero initialize
        out_cols.push_back(col);
        this->combine_cols.push_back(col);
        BasicColSet::alloc_combine_columns(num_groups, out_cols);
    }

    virtual void combine(const grouping_info& grp_info) {
        array_info* count_col_in = this->update_cols[0];
        array_info* mean_col_in = this->update_cols[1];
        array_info* m2_col_in = this->update_cols[2];
        array_info* count_col_out = this->combine_cols[1];
        array_info* mean_col_out = this->combine_cols[2];
        array_info* m2_col_out = this->combine_cols[3];
        aggfunc_output_initialize(count_col_out, Bodo_FTypes::count,
                                  use_sql_rules);
        aggfunc_output_initialize(mean_col_out, Bodo_FTypes::count,
                                  use_sql_rules);
        aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count,
                                  use_sql_rules);
        var_combine(count_col_in, mean_col_in, m2_col_in, count_col_out,
                    mean_col_out, m2_col_out, grp_info);
    }

    virtual void eval(const grouping_info& grp_info) {
        std::vector<array_info*>* mycols;
        if (this->combine_step) {
            mycols = &this->combine_cols;
        } else {
            mycols = &this->update_cols;
        }

        std::vector<array_info*> aux_cols = {mycols->at(1), mycols->at(2),
                                             mycols->at(3)};
        if (this->ftype == Bodo_FTypes::var) {
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::var_eval, use_sql_rules);
        } else {
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::std_eval, use_sql_rules);
        }
    }
};

class UdfColSet : public BasicColSet {
   public:
    UdfColSet(array_info* in_col, bool combine_step, table_info* udf_table,
              int udf_table_idx, int n_redvars, bool use_sql_rules)
        : BasicColSet(in_col, Bodo_FTypes::udf, combine_step, use_sql_rules),
          udf_table(udf_table),
          udf_table_idx(udf_table_idx),
          n_redvars(n_redvars) {}
    virtual ~UdfColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        int offset = 0;
        if (this->combine_step) offset = 1;
        // for update table we only need redvars (skip first column which is
        // output column)
        for (int i = udf_table_idx + offset; i < udf_table_idx + 1 + n_redvars;
             i++) {
            // we get the type from the udf dummy table that was passed to C++
            // library
            bodo_array_type::arr_type_enum arr_type =
                udf_table->columns[i]->arr_type;
            Bodo_CTypes::CTypeEnum dtype = udf_table->columns[i]->dtype;
            int64_t num_categories = udf_table->columns[i]->num_categories;
            out_cols.push_back(alloc_array(num_groups, 1, 1, arr_type, dtype, 0,
                                           num_categories));
            if (!this->combine_step)
                this->update_cols.push_back(out_cols.back());
        }
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

    virtual typename std::vector<array_info*>::iterator update_after_shuffle(
        typename std::vector<array_info*>::iterator& it) {
        // UdfColSet doesn't keep the update cols, return the updated iterator
        return it + n_redvars;
    }

    virtual void alloc_combine_columns(size_t num_groups,
                                       std::vector<array_info*>& out_cols) {
        for (int i = udf_table_idx; i < udf_table_idx + 1 + n_redvars; i++) {
            // we get the type from the udf dummy table that was passed to C++
            // library
            bodo_array_type::arr_type_enum arr_type =
                udf_table->columns[i]->arr_type;
            Bodo_CTypes::CTypeEnum dtype = udf_table->columns[i]->dtype;
            int64_t num_categories = udf_table->columns[i]->num_categories;
            out_cols.push_back(alloc_array(num_groups, 1, 1, arr_type, dtype, 0,
                                           num_categories));
            this->combine_cols.push_back(out_cols.back());
        }
    }

    virtual void combine(const grouping_info& grp_info) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

    virtual void eval(const grouping_info& grp_info) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

   private:
    table_info* udf_table;  // the table containing type info for UDF columns
    int udf_table_idx;      // index to my information in the udf table
    int n_redvars;          // number of redvar columns this UDF uses
};

class GeneralUdfColSet : public UdfColSet {
   public:
    GeneralUdfColSet(array_info* in_col, table_info* udf_table,
                     int udf_table_idx, bool use_sql_rules)
        : UdfColSet(in_col, false, udf_table, udf_table_idx, 0, use_sql_rules) {
    }
    virtual ~GeneralUdfColSet() {}

    /**
     * Fill in the input table for general UDF cfunc. See udf_general_fn
     * and aggregate.py::gen_general_udf_cb for more information.
     */
    void fill_in_columns(table_info* general_in_table,
                         const grouping_info& grp_info) const {
        array_info* in_col = this->in_col;
        std::vector<std::vector<int64_t>> group_rows(grp_info.num_groups);
        // get the rows in each group
        for (size_t i = 0; i < in_col->length; i++) {
            int64_t i_grp = grp_info.row_to_group[i];
            group_rows[i_grp].push_back(i);
        }
        // retrieve one column per group from the input column, add it to the
        // general UDF input table
        for (size_t i = 0; i < grp_info.num_groups; i++) {
            array_info* col = RetrieveArray_SingleColumn(in_col, group_rows[i]);
            general_in_table->columns.push_back(col);
        }
    }
};

class MedianColSet : public BasicColSet {
   public:
    MedianColSet(array_info* in_col, bool _skipna, bool use_sql_rules)
        : BasicColSet(in_col, Bodo_FTypes::median, false, use_sql_rules),
          skipna(_skipna) {}
    virtual ~MedianColSet() {}

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        median_computation(this->in_col, this->update_cols[0], grp_infos[0],
                           this->skipna, use_sql_rules);
    }

   private:
    bool skipna;
};

class NUniqueColSet : public BasicColSet {
   public:
    NUniqueColSet(array_info* in_col, bool _dropna, table_info* nunique_table,
                  bool do_combine, bool _is_parallel, bool use_sql_rules)
        : BasicColSet(in_col, Bodo_FTypes::nunique, do_combine, use_sql_rules),
          dropna(_dropna),
          my_nunique_table(nunique_table),
          is_parallel(_is_parallel) {}

    virtual ~NUniqueColSet() {
        if (my_nunique_table != nullptr)
            delete_table_decref_arrays(my_nunique_table);
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        // to support nunique for dictionary-encoded arrays we only need to
        // perform the nunqiue operation on the indices array(info2), which is a
        // int32_t numpy array.
        array_info* input_col = this->in_col->arr_type == bodo_array_type::DICT
                                    ? this->in_col->info2
                                    : this->in_col;
        // TODO: check nunique with pivot_table operation
        if (my_nunique_table != nullptr) {
            // use the grouping_info that corresponds to my nunique table
            aggfunc_output_initialize(this->update_cols[0], Bodo_FTypes::sum,
                                      use_sql_rules);  // zero initialize
            nunique_computation(input_col, this->update_cols[0],
                                grp_infos[my_nunique_table->id], dropna,
                                is_parallel);
        } else {
            // use default grouping_info
            nunique_computation(input_col, this->update_cols[0], grp_infos[0],
                                dropna, is_parallel);
        }
    }

   private:
    bool dropna;
    table_info* my_nunique_table = nullptr;
    bool is_parallel;
};

class CumOpColSet : public BasicColSet {
   public:
    CumOpColSet(array_info* in_col, int ftype, bool _skipna, bool use_sql_rules)
        : BasicColSet(in_col, ftype, false, use_sql_rules), skipna(_skipna) {}
    virtual ~CumOpColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        // NOTE: output size of cum ops is the same as input size
        //       (NOT the number of groups)
        bodo_array_type::arr_type_enum out_type = this->in_col->arr_type;
        if (out_type == bodo_array_type::DICT) {
            // for dictionary-encoded input the arrtype of the output is regular
            // string
            out_type = bodo_array_type::STRING;
        }
        out_cols.push_back(alloc_array(this->in_col->length, 1, 1, out_type,
                                       this->in_col->dtype, 0,
                                       this->in_col->num_categories));
        this->update_cols.push_back(out_cols.back());
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        cumulative_computation(this->in_col, this->update_cols[0], grp_infos[0],
                               this->ftype, this->skipna);
    }

   private:
    bool skipna;
};

class ShiftColSet : public BasicColSet {
   public:
    ShiftColSet(array_info* in_col, int ftype, int64_t _periods,
                bool use_sql_rules)
        : BasicColSet(in_col, ftype, false, use_sql_rules), periods(_periods) {}
    virtual ~ShiftColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        // NOTE: output size of shift is the same as input size
        //       (NOT the number of groups)
        out_cols.push_back(
            alloc_array(this->in_col->length, 1, 1, this->in_col->arr_type,
                        this->in_col->dtype, 0, this->in_col->num_categories));
        this->update_cols.push_back(out_cols.back());
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        shift_computation(this->in_col, this->update_cols[0], grp_infos[0],
                          this->periods);
    }

   private:
    int64_t periods;
};

/**
 * Copy nullable value from tmp_col to all the rows in the
 * corresponding group update_col.
 * @param update_col[out] output column
 * @param tmp_col[in] input column (one value per group)
 * @param grouping_info[in] structures used to get rows for each group
 *
 */
template <typename T>
void copy_nullable_values_transform(array_info* update_col, array_info* tmp_col,
                                    const grouping_info& grp_info) {
    int64_t nrows = update_col->length;
    bool bit = false;
    for (int64_t iRow = 0; iRow < nrows; iRow++) {
        int64_t igrp = grp_info.row_to_group[iRow];
        bit = tmp_col->get_null_bit(igrp);
        T val = tmp_col->template at<T>(igrp);
        update_col->set_null_bit(iRow, bit);
        update_col->template at<T>(iRow) = val;
    }
}
/**
 * Propagate value from the row in the tmp_col to all the rows in the
 * group update_col.
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 *
 * */
void copy_string_values_transform(array_info* update_col, array_info* tmp_col,
                                  const grouping_info& grp_info) {
    int64_t num_groups = grp_info.num_groups;
    array_info* out_arr = NULL;
    // first we have to deal with offsets first so we
    // need one first loop to determine the needed length. In the second
    // loop, the assignation is made. If the entries are missing then the
    // bitmask is set to false.
    bodo_array_type::arr_type_enum arr_type = tmp_col->arr_type;
    Bodo_CTypes::CTypeEnum dtype = tmp_col->dtype;
    int64_t n_chars = 0;
    int64_t nRowOut = update_col->length;
    // Store size of data per row
    std::vector<offset_t> ListSizes(nRowOut);
    offset_t* in_offsets = (offset_t*)tmp_col->data2;
    char* in_data1 = tmp_col->data1;
    // 1. Determine needed length (total number of characters)
    // and number of characters per element/row
    // All rows in same group gets same data
    for (int64_t igrp = 0; igrp < num_groups; igrp++) {
        offset_t size = 0;
        offset_t start_offset = in_offsets[igrp];
        offset_t end_offset = in_offsets[igrp + 1];
        size = end_offset - start_offset;
        int64_t idx = grp_info.group_to_first_row[igrp];
        while (true) {
            if (idx == -1) break;
            ListSizes[idx] = size;
            n_chars += size;
            idx = grp_info.next_row_in_group[idx];
        }
    }
    out_arr = alloc_array(nRowOut, n_chars, -1, arr_type, dtype, 0, 0);
    offset_t* out_offsets = (offset_t*)out_arr->data2;
    char* out_data1 = out_arr->data1;
    // keep track of output array position
    offset_t pos = 0;
    // 2. Copy data from tmp_col to corresponding rows in out_arr
    bool bit = false;
    for (int64_t iRow = 0; iRow < nRowOut; iRow++) {
        offset_t size = ListSizes[iRow];
        int64_t igrp = grp_info.row_to_group[iRow];
        offset_t start_offset = in_offsets[igrp];
        char* in_ptr = in_data1 + start_offset;
        char* out_ptr = out_data1 + pos;
        out_offsets[iRow] = pos;
        memcpy(out_ptr, in_ptr, size);
        pos += size;
        bit = tmp_col->get_null_bit(igrp);
        out_arr->set_null_bit(iRow, bit);
    }
    out_offsets[nRowOut] = pos;
    *update_col = std::move(*out_arr);
    delete out_arr;
}

/**
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 * Propagate value from the row in the tmp_col to all the rows in the
 * group update_col.
 *
 * */
template <typename T>
void copy_values(array_info* update_col, array_info* tmp_col,
                 const grouping_info& grp_info) {
    if (tmp_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        copy_nullable_values_transform<T>(update_col, tmp_col, grp_info);
        return;
    }
    // Copy result from tmp_col to corressponding group rows in
    // update_col.
    int64_t nrows = update_col->length;
    for (int64_t iRow = 0; iRow < nrows; iRow++) {
        int64_t igrp = grp_info.row_to_group[iRow];
        T& val = getv<T>(tmp_col, igrp);
        T& val2 = getv<T>(update_col, iRow);
        val2 = val;
    }
}

/**
 * Propagate value from the row in the tmp_col to all the rows in the
 * group update_col. Both tmp_col and update_col are dictionary encoded
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 *
 * */
void copy_dict_string_values_transform(array_info* update_col,
                                       array_info* tmp_col,
                                       const grouping_info& grp_info) {
    int64_t length = update_col->length;
    array_info* indices_arr =
        alloc_nullable_array(length, Bodo_CTypes::INT32, 0);
    copy_values<int32_t>(indices_arr, tmp_col->info2, grp_info);
    incref_array(tmp_col->info1);  // increase reference because we reuse the
                                   // underlying data array.
    array_info* out_col =
        create_dict_string_array(tmp_col->info1, indices_arr, length);
    *update_col = std::move(*out_col);
    // reverse_shuffle_table needs the dictionary to be global
    // copy_dict_string_values_transform is only called on distributed data
    // Does this implementation require the dictionary is sorted.
    // Similarly does it require that there are no duplicates.
    make_dictionary_global_and_unique(update_col, true);
    delete out_col;
}

// Add function declaration before usage in TransformColSet
/**
 * Construct and return a column set based on the ftype.
 * @param groupby input column associated with this column set.
 * @param ftype function type associated with this column set.
 * @param do_combine whether GroupbyPipeline will perform combine operation
 *        or not.
 * @param skipna option used for nunique, cumsum, cumprod, cummin, cummax
 * @param periods option used for shift
 * @param transform_func option used for identifying transform function
 *        (currently groupby operation that are already supported)
 */
BasicColSet* makeColSet(array_info* in_col, array_info* index_col, int ftype,
                        bool do_combine, bool skipna, int64_t periods,
                        int64_t transform_func, int n_udf, bool is_parallel,
                        bool window_ascending, bool window_na_position,
                        int* udf_n_redvars = nullptr,
                        table_info* udf_table = nullptr, int udf_table_idx = 0,
                        table_info* nunique_table = nullptr,
                        bool use_sql_rules = false);

class TransformColSet : public BasicColSet {
   public:
    TransformColSet(array_info* in_col, int ftype, int _func_num,
                    bool do_combine, bool is_parallel, bool use_sql_rules)
        : BasicColSet(in_col, ftype, false, use_sql_rules),
          transform_func(_func_num) {
        transform_op_col =
            makeColSet(in_col, nullptr, transform_func, do_combine, false, 0,
                       transform_func, 0, is_parallel, false, false, nullptr,
                       nullptr, 0, nullptr, use_sql_rules);
    }
    virtual ~TransformColSet() {
        if (transform_op_col != nullptr) delete transform_op_col;
    }

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        // Allocate child column that does the actual computation
        std::vector<array_info*> list_arr;
        transform_op_col->alloc_update_columns(num_groups, list_arr);

        // Get output column type based on transform_func and its in_col
        // datatype
        auto arr_type = this->in_col->arr_type;
        auto dtype = this->in_col->dtype;
        int64_t num_categories = this->in_col->num_categories;
        bool is_combine = false;
        get_groupby_output_dtype(transform_func, arr_type, dtype, false,
                                 is_combine);
        // NOTE: output size of transform is the same as input size
        //       (NOT the number of groups)
        out_cols.push_back(alloc_array(this->in_col->length, 1, 1, arr_type,
                                       dtype, 0, num_categories));
        this->update_cols.push_back(out_cols.back());
    }

    // Call corresponding groupby function operation to compute
    // transform_op_col column.
    virtual void update(const std::vector<grouping_info>& grp_infos) {
        transform_op_col->update(grp_infos);
        aggfunc_output_initialize(this->update_cols[0], transform_func,
                                  use_sql_rules);
    }
    // Fill the output column by copying values from the transform_op_col column
    virtual void eval(const grouping_info& grp_info) {
        // Needed to get final result for transform operation on
        // transform_op_col
        transform_op_col->eval(grp_info);
        // copy_values need to know type of the data it'll copy.
        // Hence we use switch case on the column dtype
        array_info* child_out_col = this->transform_op_col->getOutputColumn();
        switch (child_out_col->dtype) {
            case Bodo_CTypes::_BOOL:
                copy_values<bool>(this->update_cols[0], child_out_col,
                                  grp_info);
                break;
            case Bodo_CTypes::INT8:
                copy_values<int8_t>(this->update_cols[0], child_out_col,
                                    grp_info);
                break;
            case Bodo_CTypes::UINT8:
                copy_values<uint8_t>(this->update_cols[0], child_out_col,
                                     grp_info);
                break;
            case Bodo_CTypes::INT16:
                copy_values<int16_t>(this->update_cols[0], child_out_col,
                                     grp_info);
                break;
            case Bodo_CTypes::UINT16:
                copy_values<uint16_t>(this->update_cols[0], child_out_col,
                                      grp_info);
                break;
            case Bodo_CTypes::INT32:
                copy_values<int32_t>(this->update_cols[0], child_out_col,
                                     grp_info);
                break;
            case Bodo_CTypes::UINT32:
                copy_values<uint32_t>(this->update_cols[0], child_out_col,
                                      grp_info);
                break;
            case Bodo_CTypes::DATE:
            case Bodo_CTypes::DATETIME:
            case Bodo_CTypes::TIMEDELTA:
            case Bodo_CTypes::INT64:
            // TODO: [BE-4106] Split Time into Time32 and Time64
            case Bodo_CTypes::TIME:
                copy_values<int64_t>(this->update_cols[0], child_out_col,
                                     grp_info);
                break;
            case Bodo_CTypes::UINT64:
                copy_values<uint64_t>(this->update_cols[0], child_out_col,
                                      grp_info);
                break;
            case Bodo_CTypes::FLOAT32:
                copy_values<float>(this->update_cols[0], child_out_col,
                                   grp_info);
                break;
            case Bodo_CTypes::FLOAT64:
                copy_values<double>(this->update_cols[0], child_out_col,
                                    grp_info);
                break;
            case Bodo_CTypes::STRING:
                if (child_out_col->arr_type == bodo_array_type::DICT) {
                    copy_dict_string_values_transform(this->update_cols[0],
                                                      child_out_col, grp_info);
                } else {
                    copy_string_values_transform(this->update_cols[0],
                                                 child_out_col, grp_info);
                }
                break;
            default:
                Bodo_PyErr_SetString(
                    PyExc_RuntimeError,
                    (std::string("unsuported data type for eval: ") +
                     std::string(GetDtype_as_string(child_out_col->dtype)))
                        .c_str());
                return;
        }
        delete_info_decref_array(child_out_col);
    }

   private:
    int64_t transform_func;
    BasicColSet* transform_op_col = nullptr;
};
class HeadColSet : public BasicColSet {
   public:
    HeadColSet(array_info* in_col, int ftype, bool use_sql_rules)
        : BasicColSet(in_col, ftype, false, use_sql_rules) {}
    virtual ~HeadColSet() {}

    virtual void alloc_update_columns(size_t update_col_len,
                                      std::vector<array_info*>& out_cols) {
        // NOTE: output size of head is dependent on number of rows to
        // get from each group. This is computed in GroupbyPipeline::update().
        out_cols.push_back(
            alloc_array(update_col_len, 1, 1, this->in_col->arr_type,
                        this->in_col->dtype, 0, this->in_col->num_categories));
        this->update_cols.push_back(out_cols.back());
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        head_computation(this->in_col, this->update_cols[0], head_row_list);
    }
    void set_head_row_list(std::vector<int64_t> row_list) {
        head_row_list = row_list;
    }

   private:
    std::vector<int64_t> head_row_list;
};

/**
 * @brief Handles the update step for the supported window functions.
 * These functions are not simple reductions and require additional
 * functionality to operate over a "window" of values (possibly a sort
 * or equivalent). The output size is always the same size as the original
 * input
 *
 * @param[in] orderby_arr The array that is being "sorted" to determine
 * the groups. In some situations it may be possible to do a partial sort
 * or avoid sorting.
 * @param[in] window_func The name of the window function being computed.
 * Currently we only support row_number.
 * @param[out] out_arr The output array being population.
 * @param[in] grp_info Struct containing information about the groups.
 * @param[in] asc Should the array be sorted in ascending order?
 * @param[in] na_pos Should NA's be placed at the end of the array?
 * @param[in] is_parallel Is the data distributed? This is used for tracing
 * @param[in] use_sql_rules Do we use SQL or Pandas Null rules
 * informate.
 */
void window_computation(array_info* orderby_arr, int64_t window_func,
                        array_info* out_arr, grouping_info const& grp_info,
                        bool asc, bool na_pos, bool is_parallel,
                        bool use_sql_rules) {
    switch (window_func) {
        case Bodo_FTypes::row_number: {
            const std::vector<int64_t>& row_to_group = grp_info.row_to_group;
            int64_t num_rows = row_to_group.size();
            // Wrap the row_to_group in an array info so we can use it to sort.
            array_info* group_arr = alloc_numpy(num_rows, Bodo_CTypes::INT64);
            // TODO: Reuse the row_to_group buffer
            for (int64_t i = 0; i < num_rows; i++) {
                getv<int64_t>(group_arr, i) = row_to_group[i];
            }
            // Create a new table. We want to sort the table first by
            // the groups and second by the orderby_arr.
            table_info* sort_table = new table_info();
            // sort_values_table_local steals a reference
            incref_array(orderby_arr);
            sort_table->columns.push_back(group_arr);
            sort_table->columns.push_back(orderby_arr);
            // Append an index column so we can find the original
            // index in the out array.
            array_info* idx_arr = alloc_numpy(num_rows, Bodo_CTypes::INT64);
            for (int64_t i = 0; i < num_rows; i++) {
                getv<int64_t>(idx_arr, i) = i;
            }
            sort_table->columns.push_back(idx_arr);
            int64_t vect_ascending[2] = {asc, asc};
            int64_t na_position[2] = {na_pos, na_pos};

            // Sort the table
            // XXX: We don't need the entire chunk of data sorted,
            // just the groups. We could do a partial sort to avoid
            // the overhead of sorting the data and in the future
            // we may be want to explore if we can use hashing
            // instead to avoid sort overhead.
            table_info* iter_table = sort_values_table_local(
                sort_table, 2, vect_ascending, na_position, nullptr,
                is_parallel /* This is just used for tracing */);
            array_info* sorted_groups = iter_table->columns[0];
            array_info* sorted_idx = iter_table->columns[2];
            // sort_values_table_local steals a reference so
            // we don't need to decref
            delete sort_table;

            int64_t prev_group = -1;
            int64_t row_num = 1;
            for (int64_t i = 0; i < num_rows; i++) {
                int64_t curr_group = getv<int64_t>(sorted_groups, i);
                if (curr_group != prev_group) {
                    row_num = 1;
                } else {
                    row_num++;
                }
                // Get the index in the output array.
                int64_t idx = getv<int64_t>(sorted_idx, i);
                getv<int64_t>(out_arr, idx) = row_num;
                // Set the prev group
                prev_group = curr_group;
            }
            // Delete the sorted table.
            delete_table_decref_arrays(iter_table);
            break;
        }
        case Bodo_FTypes::min_row_number_filter: {
            // To compute min_row_number_filter we want to find the
            // idxmin/idxmax based on the orderby column. Then in the output
            // array those locations will have the value true. We have already
            // initialized all other locations to false.
            int64_t ftype;
            bodo_array_type::arr_type_enum idx_arr_type;
            if (asc) {
                // The first value of an array in ascending order is the min.
                if (na_pos) {
                    ftype = Bodo_FTypes::idxmin;
                    // We don't need null values for indices
                    idx_arr_type = bodo_array_type::NUMPY;
                } else {
                    ftype = Bodo_FTypes::idxmin_na_first;
                    // We need null values to signal we found an NA
                    // value.
                    idx_arr_type = bodo_array_type::NULLABLE_INT_BOOL;
                }
            } else {
                // The first value of an array in descending order is the max.
                if (na_pos) {
                    ftype = Bodo_FTypes::idxmax;
                    // We don't need null values for indices
                    idx_arr_type = bodo_array_type::NUMPY;
                } else {
                    ftype = Bodo_FTypes::idxmax_na_first;
                    // We need null values to signal we found an NA
                    // value.
                    idx_arr_type = bodo_array_type::NULLABLE_INT_BOOL;
                }
            }
            // Allocate intermediate buffer to find the true element for each
            // group. Indices
            size_t num_groups = grp_info.num_groups;
            array_info* idx_col = alloc_array(num_groups, 1, 1, idx_arr_type,
                                              Bodo_CTypes::UINT64, 0, 0);
            // create array to store min/max value
            array_info* data_col =
                alloc_array(num_groups, 1, 1, orderby_arr->arr_type,
                            orderby_arr->dtype, 0, 0);
            // Initialize the index column. This is 0 intialized and will
            // not initial the null values.
            aggfunc_output_initialize(idx_col, Bodo_FTypes::count,
                                      use_sql_rules);
            std::vector<array_info*> aux_cols = {idx_col};
            // Initialize the max column
            if (ftype == Bodo_FTypes::idxmax ||
                ftype == Bodo_FTypes::idxmax_na_first) {
                aggfunc_output_initialize(data_col, Bodo_FTypes::max,
                                          use_sql_rules);
            } else {
                aggfunc_output_initialize(data_col, Bodo_FTypes::min,
                                          use_sql_rules);
            }
            // Compute the idxmin/idxmax
            do_apply_to_column(orderby_arr, data_col, aux_cols, grp_info, ftype,
                               use_sql_rules);
            // Delete the max/min result
            delete_info_decref_array(data_col);
            // Now we have the idxmin/idxmax in the idx_col. We need to set the
            // indices to true.
            for (size_t i = 0; i < idx_col->length; i++) {
                int64_t idx = getv<int64_t>(idx_col, i);
                getv<bool>(out_arr, idx) = true;
            }
            // Delete the idx_col
            delete_info_decref_array(idx_col);
            break;
        }
        default:
            throw new std::runtime_error("Invalid window function");
    }
}

/**
 * @brief WindowColSet column set for window operations.
 *
 */
class WindowColSet : public BasicColSet {
   public:
    /**
     * Construct Window column set
     * @param in_col input column of groupby associated with this column set.
     * This is a column that we will sort on.
     * @param _window_func: What function are we computing.
     * @param _asc: Is the sort ascending on the input column.
     * @param _na_pos: Are NAs last in the sort
     * @param _is_parallel: flag to identify whether data is distributed
     * @param use_sql_rules: Do we use SQL or Pandas null handling rules.
     *
     */
    WindowColSet(array_info* in_col, int64_t _window_func, bool _asc,
                 bool _na_pos, bool _is_parallel, bool use_sql_rules)
        : BasicColSet(in_col, Bodo_FTypes::window, false, use_sql_rules),
          window_func(_window_func),
          asc(_asc),
          na_pos(_na_pos),
          is_parallel(_is_parallel) {}
    virtual ~WindowColSet() {}

    /**
     * Allocate column for update step.
     * @param num_groups: number of groups found in the input table
     * @param[in,out] out_cols: vector of columns of update table. This method
     * adds columns to this vector.
     * NOTE: the added column is an integer array with same length as
     * input column regardless of input column types (i.e num_groups is not used
     * in this case)
     */
    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        bodo_array_type::arr_type_enum arr_type = this->in_col->arr_type;
        Bodo_CTypes::CTypeEnum dtype = this->in_col->dtype;
        int64_t num_categories = this->in_col->num_categories;
        bool is_combine = false;
        // calling this modifies arr_type and dtype
        // Output dtype is based on the window function.
        get_groupby_output_dtype(window_func, arr_type, dtype, false,
                                 is_combine);
        // NOTE: output size of window is the same as input size
        //       (NOT the number of groups)
        array_info* c = alloc_array(this->in_col->length, 1, 1, arr_type, dtype,
                                    0, num_categories);
        aggfunc_output_initialize(c, window_func, use_sql_rules);
        out_cols.push_back(c);
        this->update_cols.push_back(c);
    }

    /**
     * Perform update step for this column set. This first shuffles
     * the data based on the orderby condition + group columns and
     * then computes the window function. If this is a parallel operations
     * then we must update the shuffle info so the reverse shuffle will
     * be correct. If this is a serial operation then we need to execute
     * a local reverse shuffle.
     * @param grp_infos: grouping info calculated by GroupbyPipeline
     */
    virtual void update(const std::vector<grouping_info>& grp_infos) {
        window_computation(this->in_col, window_func, this->update_cols[0],
                           grp_infos[0], asc, na_pos, is_parallel,
                           use_sql_rules);
    }

   private:
    int64_t window_func;
    bool asc;
    bool na_pos;
    bool is_parallel;  // whether input column data is distributed or
                       // replicated
};

/**
 * @brief NgroupColSet column set for ngroup operation
 */
class NgroupColSet : public BasicColSet {
   public:
    /**
     * Construct Ngroup column set
     * @param in_col input column of groupby associated with this column set
     * @param _is_parallel: flag to identify whether data is distributed or
     * replicated across ranks
     */
    NgroupColSet(array_info* in_col, bool _is_parallel, bool use_sql_rules)
        : BasicColSet(in_col, Bodo_FTypes::ngroup, false, use_sql_rules),
          is_parallel(_is_parallel) {}
    virtual ~NgroupColSet() {}

    /**
     * Allocate column for update step.
     * @param num_groups: number of groups found in the input table
     * @param[in,out] out_cols: vector of columns of update table. This method
     * adds columns to this vector.
     * NOTE: the added column is an integer array with same length as
     * input column regardless of input column types (i.e num_groups is not used
     * in this case)
     */
    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        bodo_array_type::arr_type_enum arr_type = this->in_col->arr_type;
        Bodo_CTypes::CTypeEnum dtype = this->in_col->dtype;
        int64_t num_categories = this->in_col->num_categories;
        // calling this modifies arr_type and dtype
        bool is_combine = false;
        get_groupby_output_dtype(this->ftype, arr_type, dtype, false,
                                 is_combine);
        // NOTE: output size of ngroup is the same as input size
        //       (NOT the number of groups)
        out_cols.push_back(alloc_array(this->in_col->length, 1, 1, arr_type,
                                       dtype, 0, num_categories));
        this->update_cols.push_back(out_cols.back());
    }

    /**
     * Perform update step for this column set. compute and fill my columns with
     * the result of the ngroup operation.
     * @param grp_infos: grouping info calculated by GroupbyPipeline
     */
    virtual void update(const std::vector<grouping_info>& grp_infos) {
        ngroup_computation(this->in_col, this->update_cols[0], grp_infos[0],
                           is_parallel);
    }

   private:
    bool is_parallel;  // whether input column data is distributed or
                       // replicated.
};

BasicColSet* makeColSet(array_info* in_col, array_info* index_col, int ftype,
                        bool do_combine, bool skipna, int64_t periods,
                        int64_t transform_func, int n_udf, bool is_parallel,
                        bool window_ascending, bool window_na_position,
                        int* udf_n_redvars, table_info* udf_table,
                        int udf_table_idx, table_info* nunique_table,
                        bool use_sql_rules) {
    switch (ftype) {
        case Bodo_FTypes::udf:
            return new UdfColSet(in_col, do_combine, udf_table, udf_table_idx,
                                 udf_n_redvars[n_udf], use_sql_rules);
        case Bodo_FTypes::gen_udf:
            return new GeneralUdfColSet(in_col, udf_table, udf_table_idx,
                                        use_sql_rules);
        case Bodo_FTypes::median:
            return new MedianColSet(in_col, skipna, use_sql_rules);
        case Bodo_FTypes::nunique:
            return new NUniqueColSet(in_col, skipna, nunique_table, do_combine,
                                     is_parallel, use_sql_rules);
        case Bodo_FTypes::cumsum:
        case Bodo_FTypes::cummin:
        case Bodo_FTypes::cummax:
        case Bodo_FTypes::cumprod:
            return new CumOpColSet(in_col, ftype, skipna, use_sql_rules);
        case Bodo_FTypes::mean:
            return new MeanColSet(in_col, do_combine, use_sql_rules);
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
            return new VarStdColSet(in_col, ftype, do_combine, use_sql_rules);
        case Bodo_FTypes::idxmin:
        case Bodo_FTypes::idxmax:
            return new IdxMinMaxColSet(in_col, index_col, ftype, do_combine,
                                       use_sql_rules);
        case Bodo_FTypes::shift:
            return new ShiftColSet(in_col, ftype, periods, use_sql_rules);
        case Bodo_FTypes::transform:
            return new TransformColSet(in_col, ftype, transform_func,
                                       do_combine, is_parallel, use_sql_rules);
        case Bodo_FTypes::head:
            return new HeadColSet(in_col, ftype, use_sql_rules);
        case Bodo_FTypes::ngroup:
            return new NgroupColSet(in_col, is_parallel, use_sql_rules);
        case Bodo_FTypes::window:
            return new WindowColSet(in_col, transform_func, window_ascending,
                                    window_na_position, is_parallel,
                                    use_sql_rules);
        default:
            return new BasicColSet(in_col, ftype, do_combine, use_sql_rules);
    }
}

class GroupbyPipeline {
   public:
    GroupbyPipeline(table_info* _in_table, int64_t _num_keys,
                    table_info* _dispatch_table, table_info* _dispatch_info,
                    bool input_has_index, bool _is_parallel, int* ftypes,
                    int* func_offsets, int* _udf_nredvars,
                    table_info* _udf_table, udf_table_op_fn update_cb,
                    udf_table_op_fn combine_cb, udf_eval_fn eval_cb,
                    udf_general_fn general_udfs_cb, bool skipna,
                    int64_t periods, int64_t transform_func, int64_t _head_n,
                    bool _return_key, bool _return_index, bool _key_dropna,
                    bool window_ascending, bool window_na_position,
                    bool _maintain_input_size, int64_t _n_shuffle_keys,
                    bool _use_sql_rules)
        : orig_in_table(_in_table),
          in_table(_in_table),
          num_keys(_num_keys),
          dispatch_table(_dispatch_table),
          dispatch_info(_dispatch_info),
          is_parallel(_is_parallel),
          return_key(_return_key),
          return_index(_return_index),
          key_dropna(_key_dropna),
          udf_table(_udf_table),
          udf_n_redvars(_udf_nredvars),
          head_n(_head_n),
          maintain_input_size(_maintain_input_size),
          n_shuffle_keys(_n_shuffle_keys),
          use_sql_rules(_use_sql_rules) {
        tracing::Event ev("GroupbyPipeline()", is_parallel);
        udf_info = {udf_table, update_cb, combine_cb, eval_cb, general_udfs_cb};
        // if true, the last column is the index on input and output.
        // this is relevant only to cumulative operations like cumsum
        // and transform.
        int index_i = int(input_has_index);
        // NOTE cumulative operations (cumsum, cumprod, etc.) cannot be mixed
        // with non cumulative ops. This is checked at compile time in
        // aggregate.py

        bool has_udf = false;
        nunique_op = false;
        int nunique_count = 0;
        const int num_funcs =
            func_offsets[in_table->ncols() - num_keys - index_i];
        for (int i = 0; i < num_funcs; i++) {
            int ftype = ftypes[i];
            if (ftype == Bodo_FTypes::gen_udf && is_parallel)
                shuffle_before_update = true;
            if (ftype == Bodo_FTypes::udf) has_udf = true;
            if (ftype == Bodo_FTypes::head) {
                head_op = true;
                if (is_parallel) shuffle_before_update = true;
                break;
            }
            if (ftype == Bodo_FTypes::nunique) {
                nunique_op = true;
                req_extended_group_info = true;
                nunique_count++;
            } else if (ftype == Bodo_FTypes::median ||
                       ftype == Bodo_FTypes::cumsum ||
                       ftype == Bodo_FTypes::cumprod ||
                       ftype == Bodo_FTypes::cummin ||
                       ftype == Bodo_FTypes::cummax ||
                       ftype == Bodo_FTypes::shift ||
                       ftype == Bodo_FTypes::transform ||
                       ftype == Bodo_FTypes::ngroup ||
                       ftype == Bodo_FTypes::window) {
                // these operations first require shuffling the data to
                // gather all rows with the same key in the same process
                if (is_parallel) {
                    shuffle_before_update = true;
                }
                // these operations require extended group info
                req_extended_group_info = true;
                if (ftype == Bodo_FTypes::cumsum ||
                    ftype == Bodo_FTypes::cummin ||
                    ftype == Bodo_FTypes::cumprod ||
                    ftype == Bodo_FTypes::cummax) {
                    cumulative_op = true;
                } else if (ftype == Bodo_FTypes::shift) {
                    shift_op = true;
                } else if (ftype == Bodo_FTypes::transform) {
                    transform_op = true;
                } else if (ftype == Bodo_FTypes::ngroup) {
                    ngroup_op = true;
                } else if (ftype == Bodo_FTypes::window) {
                    window_op = true;
                }
                break;
            }
        }
        // In case of ngroup: previous loop will be skipped
        // As num_funcs will be 0 since ngroup output is single column
        // regardless of number of input and key columns
        // So, set flags for ngroup here.
        if (num_funcs == 0 && ftypes[0] == Bodo_FTypes::ngroup) {
            ngroup_op = true;
            // these operations first require shuffling the data to
            // gather all rows with the same key in the same process
            if (is_parallel) shuffle_before_update = true;
            // these operations require extended group info
            req_extended_group_info = true;
        }
        if (nunique_op) {
            if (nunique_count == num_funcs) nunique_only = true;
            ev.add_attribute("nunique_only", nunique_only);
        }

        // if gb.head and data is distribute, last column is key-sort column.
        int head_i = int(head_op && is_parallel);
        // Add key-sorting-column for gb.head() to sort output at the end
        // this is relevant only if data is distributed.
        if (head_i) {
            add_head_key_sort_column();
        }
        int k = 0;
        for (uint64_t icol = 0; icol < in_table->ncols() - index_i - head_i;
             icol++) {
            array_info* a = in_table->columns[icol];
            if (a->arr_type == bodo_array_type::DICT) {
                // Convert the local dictionary to global for hashing purposes
                make_dictionary_global_and_unique(a, is_parallel);
            }
        }

        // get hashes of keys
        // NOTE: this has to be num_keys and not n_shuffle_keys
        // to avoid having a far off estimated nunique_hashes
        // which could lead to having large chance of map insertion collisions.
        // See [BE-3371] for more context.
        hashes = hash_keys_table(in_table, num_keys, SEED_HASH_PARTITION,
                                 is_parallel);
        size_t nunique_hashes_global = 0;
        // get estimate of number of unique hashes to guide optimization.
        // if shuffle_before_update=true we are going to shuffle everything
        // first so we don't need statistics of current hashes
        if (is_parallel && !shuffle_before_update) {
            if (nunique_op) {
                // nunique_hashes_global is currently only used for gb.nunique
                // heuristic
                std::tie(nunique_hashes, nunique_hashes_global) =
                    get_nunique_hashes_global(hashes, in_table->nrows(),
                                              is_parallel);
            } else {
                nunique_hashes =
                    get_nunique_hashes(hashes, in_table->nrows(), is_parallel);
            }
        } else if (!is_parallel) {
            nunique_hashes =
                get_nunique_hashes(hashes, in_table->nrows(), is_parallel);
        }

        if (is_parallel && (dispatch_table == nullptr) && !has_udf &&
            !shuffle_before_update) {
            // If the estimated number of groups (given by nunique_hashes)
            // is similar to the number of input rows, then it's better to
            // shuffle first instead of doing a local reduction

            // TODO To do this with UDF functions we need to generate
            // two versions of UDFs at compile time (one for
            // shuffle_before_update=true and one for
            // shuffle_before_update=false)

            int shuffle_before_update_local = 0;
            double local_expected_avg_group_size;
            if (nunique_hashes == 0) {
                local_expected_avg_group_size = 1.0;
            } else {
                local_expected_avg_group_size =
                    in_table->nrows() / double(nunique_hashes);
            }
            // XXX what threshold is best? Here we say on average we expect
            // every group to shrink.
            if (local_expected_avg_group_size <= 2.0) {
                shuffle_before_update_local = 1;
            }
            ev.add_attribute("local_expected_avg_group_size",
                             local_expected_avg_group_size);
            ev.add_attribute("shuffle_before_update_local",
                             shuffle_before_update_local);
            // global count of ranks that decide to shuffle before update
            int shuffle_before_update_count;
            MPI_Allreduce(&shuffle_before_update_local,
                          &shuffle_before_update_count, 1, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD);
            // TODO Need a better threshold or cost model to decide when
            // to shuffle: https://bodo.atlassian.net/browse/BE-1140
            int num_ranks;
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            if (shuffle_before_update_count >= num_ranks * 0.5) {
                shuffle_before_update = true;
            }
        }

        if (shuffle_before_update) {
            // If we are using a subset of keys we need to use the hash function
            // based on the actual number of shuffle keys. Note: This shouldn't
            // matter in the other cases because we will recompute the hashes
            // based on the number of shuffle keys if we update then shuffle and
            // num_keys == n_shuffle_keys for nunique.
            if (num_keys != n_shuffle_keys) {
                delete[] hashes;
                hashes = hash_keys_table(in_table, n_shuffle_keys,
                                         SEED_HASH_PARTITION, is_parallel);
            }

            // Code below is equivalent to:
            // table_info* in_table = shuffle_table(in_table, num_keys)
            // We do this more complicated construction because we may
            // need the hashes and comm_info later.
            comm_info_ptr = new mpi_comm_info(in_table->columns);
            comm_info_ptr->set_counts(hashes, is_parallel);
            // shuffle_table_kernel steals the reference but we still
            // need it for the code after C++ groupby
            for (auto a : in_table->columns) incref_array(a);
            in_table = shuffle_table_kernel(in_table, hashes, *comm_info_ptr,
                                            is_parallel);
            has_reverse_shuffle = cumulative_op || shift_op || transform_op ||
                                  ngroup_op || window_op;
            if (!has_reverse_shuffle) {
                delete[] hashes;
                delete comm_info_ptr;
            } else {
                // preserve input table hashes for reverse shuffle at the end
                in_hashes = hashes;
            }
            hashes = nullptr;
        } else if (nunique_op && is_parallel) {
            // **NOTE**: gb_nunique_preprocess can set
            // shuffle_before_update=true in some cases
            gb_nunique_preprocess(ftypes, num_funcs, nunique_hashes_global);
        }

        // a combine operation is only necessary when data is distributed and
        // a shuffle has not been done at the start of the groupby pipeline
        do_combine = is_parallel && !shuffle_before_update;

        array_info* index_col = nullptr;
        if (input_has_index)
            // if gb.head() exclude head_op column as well (if data is
            // distributed).
            index_col =
                in_table->columns[in_table->columns.size() - 1 - head_i];

        // construct the column sets, one for each (input_column, func) pair.
        // ftypes is an array of function types received from generated code,
        // and has one ftype for each (input_column, func) pair
        k = 0;
        n_udf = 0;
        for (uint64_t i = num_keys; i < in_table->ncols() - index_i - head_i;
             i++, k++) {  // for each data column
            array_info* col = in_table->columns[i];
            int start = func_offsets[k];
            int end = func_offsets[k + 1];
            for (int j = start; j != end;
                 j++) {  // for each function applied to this column
                if (ftypes[j] == Bodo_FTypes::nunique &&
                    (nunique_tables.size() > 0)) {
                    array_info* nunique_col =
                        nunique_tables[i]->columns[num_keys];
                    col_sets.push_back(makeColSet(
                        nunique_col, index_col, ftypes[j], do_combine, skipna,
                        periods, transform_func, n_udf, is_parallel,
                        window_ascending, window_na_position, udf_n_redvars,
                        udf_table, udf_table_idx, nunique_tables[i],
                        use_sql_rules));
                } else {
                    col_sets.push_back(makeColSet(
                        col, index_col, ftypes[j], do_combine, skipna, periods,
                        transform_func, n_udf, is_parallel, window_ascending,
                        window_na_position, udf_n_redvars, udf_table,
                        udf_table_idx, nullptr, use_sql_rules));
                }
                if (ftypes[j] == Bodo_FTypes::udf ||
                    ftypes[j] == Bodo_FTypes::gen_udf) {
                    udf_table_idx += (1 + udf_n_redvars[n_udf]);
                    n_udf++;
                    if (ftypes[j] == Bodo_FTypes::gen_udf) {
                        gen_udf_col_sets.push_back(
                            dynamic_cast<GeneralUdfColSet*>(col_sets.back()));
                    }
                }
                ev.add_attribute("g_column_ftype_" + std::to_string(j),
                                 ftypes[j]);
            }
        }
        // This is needed if aggregation was just size/ngroup operation, it will
        // skip loop (ncols = num_keys + index_i)
        if (col_sets.size() == 0 && (ftypes[0] == Bodo_FTypes::size ||
                                     ftypes[0] == Bodo_FTypes::ngroup)) {
            col_sets.push_back(makeColSet(
                in_table->columns[0], index_col, ftypes[0], do_combine, skipna,
                periods, transform_func, n_udf, is_parallel, window_ascending,
                window_na_position, udf_n_redvars, udf_table, udf_table_idx,
                nullptr, use_sql_rules));
        }
        // Add key-sort column and index to col_sets
        // to apply head_computation on them as well.
        if (head_op && return_index) {
            // index-column
            col_sets.push_back(
                makeColSet(index_col, index_col, Bodo_FTypes::head, do_combine,
                           skipna, periods, transform_func, n_udf, is_parallel,
                           window_ascending, window_na_position, udf_n_redvars,
                           udf_table, udf_table_idx, nullptr, use_sql_rules));
            if (head_i) {
                array_info* col =
                    in_table->columns[in_table->columns.size() - 1];
                col_sets.push_back(makeColSet(
                    col, index_col, Bodo_FTypes::head, do_combine, skipna,
                    periods, transform_func, n_udf, is_parallel,
                    window_ascending, window_na_position, udf_n_redvars,
                    udf_table, udf_table_idx, nullptr, use_sql_rules));
            }
        }
        in_table->id = 0;
        ev.add_attribute("g_shuffle_before_update",
                         static_cast<size_t>(shuffle_before_update));
        ev.add_attribute("g_do_combine", static_cast<size_t>(do_combine));
    }

    ~GroupbyPipeline() {
        for (auto col_set : col_sets) delete col_set;
        if (hashes != nullptr) delete[] hashes;
    }
    /**
     * @brief
     * Create key-sort column used to sort table at the end.
     * Set its values and add as the last column in in_table.
     * Column values is in range(start, start+nrows).
     * Each rank will compute its range by identifying
     * start/end index of its set of rows.
     * @return ** void
     */
    void add_head_key_sort_column() {
        array_info* head_sort_col =
            alloc_array(in_table->nrows(), 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::UINT64, 0, 0);
        int64_t num_ranks = dist_get_size();
        int64_t my_rank = dist_get_rank();
        // Gather the number of rows on every rank
        int64_t num_rows = in_table->nrows();
        std::vector<int64_t> num_rows_ranks(num_ranks);
        MPI_Allgather(&num_rows, 1, MPI_INT64_T, num_rows_ranks.data(), 1,
                      MPI_INT64_T, MPI_COMM_WORLD);

        // Determine the start/end row number of each rank
        int64_t rank_start_row, rank_end_row;
        rank_end_row = std::accumulate(num_rows_ranks.begin(),
                                       num_rows_ranks.begin() + my_rank + 1, 0);
        rank_start_row = rank_end_row - num_rows;
        // generate start/end range
        for (int64_t i = 0; i < num_rows; i++) {
            uint64_t& val = getv<uint64_t>(head_sort_col, i);
            val = rank_start_row + i;
        }
        in_table->columns.push_back(head_sort_col);
    }

    /**
     * This is the main control flow of the Groupby pipeline.
     */
    table_info* run(int64_t* n_out_rows) {
        update();
        if (shuffle_before_update) {
            if (in_table != orig_in_table)
                // in_table is temporary table created in C++
                delete_table_decref_arrays(in_table);
        }
        if (is_parallel && !shuffle_before_update) {
            shuffle();
            combine();
        }

        eval();
        // For gb.head() operation, if data is distributed,
        // sort table based on head_sort_col column.
        if (head_op && is_parallel) {
            sort_gb_head_output();
        }
        return getOutputTable(n_out_rows);
    }
    /**
     * @brief
     * 1. Put head_sort_col at the beginning of the table.
     * 2. Sort table based on this column.
     * 3. Remove head_sort_col.
     * @return ** void
     */
    void sort_gb_head_output() {
        // Move sort column to the front.
        std::vector<array_info*>::iterator pos = cur_table->columns.end() - 1;
        std::rotate(cur_table->columns.begin(), pos, pos + 1);
        // whether to put NaN first or last.
        // Does not matter in this case (no NaN, values are range(nrows))
        int64_t asc_pos = 1;
        int64_t zero = 0;
        cur_table = sort_values_table(cur_table, 1, &asc_pos, &asc_pos, &zero,
                                      nullptr, nullptr, is_parallel);
        // Remove key-sort column
        delete_info_decref_array(cur_table->columns[0]);
        cur_table->columns.erase(cur_table->columns.begin());
    }

   private:
    int64_t compute_head_row_list(grouping_info const& grp_info,
                                  std::vector<int64_t>& head_row_list) {
        // keep track of how many rows found per group so far.
        std::vector<int64_t> nrows_per_grp(grp_info.num_groups);
        int64_t count = 0;  // how many rows found so far
        uint64_t iRow = 0;  // index looping over all rows
        for (iRow = 0; iRow < in_table->nrows(); iRow++) {
            int64_t igrp = grp_info.row_to_group[iRow];
            if (igrp != -1 && nrows_per_grp[igrp] < head_n) {
                nrows_per_grp[igrp]++;
                head_row_list.push_back(iRow);
                count++;
            }
        }
        return count;
    }
    /**
     * The update step groups rows in the input table based on keys, and
     * aggregates them based on the function to be applied to the columns.
     * More specifically, it will invoke the update method of each column set.
     */
    void update() {
        tracing::Event ev("update", is_parallel);
        in_table->num_keys = num_keys;
        std::vector<table_info*> tables;
        // If nunique_only and nunique_tables.size() > 0 then all of the input
        // data is in nunique_tables
        if (!(nunique_only && nunique_tables.size() > 0))
            tables.push_back(in_table);
        for (auto it = nunique_tables.begin(); it != nunique_tables.end(); it++)
            tables.push_back(it->second);

        if (req_extended_group_info) {
            const bool consider_missing = cumulative_op || shift_op ||
                                          transform_op || ngroup_op ||
                                          window_op;
            get_group_info_iterate(tables, hashes, nunique_hashes, grp_infos,
                                   consider_missing, key_dropna, is_parallel);
        } else {
            get_group_info(tables, hashes, nunique_hashes, grp_infos, true,
                           key_dropna, is_parallel);
        }
        grouping_info& grp_info = grp_infos[0];
        grp_info.dispatch_table = dispatch_table;
        grp_info.dispatch_info = dispatch_info;
        grp_info.mode = 1;
        num_groups = grp_info.num_groups;
        int64_t update_col_len = num_groups;
        std::vector<int64_t> head_row_list;
        if (head_op) {
            update_col_len = compute_head_row_list(grp_infos[0], head_row_list);
        }

        // Now if we have multiple tables, this step recombines them into
        // a single update table. There could be multiple tables if different
        // operations shuffle at different times. For example nunique + sum
        // in test_711.py e2e tests.
        update_table = cur_table = new table_info();
        if (cumulative_op || shift_op || transform_op || head_op || ngroup_op ||
            window_op) {
            num_keys = 0;  // there are no key columns in output of cumulative
                           // operations
        } else {
            alloc_init_keys(tables, update_table);
        }

        for (auto col_set : col_sets) {
            std::vector<array_info*> list_arr;
            col_set->alloc_update_columns(update_col_len, list_arr);
            for (auto& e_arr : list_arr) {
                update_table->columns.push_back(e_arr);
            }
            auto head_col = dynamic_cast<HeadColSet*>(col_set);
            if (head_col) head_col->set_head_row_list(head_row_list);
            col_set->update(grp_infos);
        }
        // gb.head() already added the index to the tables columns.
        // This is need to do head_computation on it as well.
        // since it will not be the same length as the in_table.
        if (!head_op && return_index) {
            update_table->columns.push_back(
                copy_array(in_table->columns.back()));
        }
        if (n_udf > 0) {
            int n_gen_udf = gen_udf_col_sets.size();
            if (n_udf > n_gen_udf)
                // regular UDFs
                udf_info.update(in_table, update_table,
                                grp_info.row_to_group.data());
            if (n_gen_udf > 0) {
                table_info* general_in_table = new table_info();
                for (auto udf_col_set : gen_udf_col_sets)
                    udf_col_set->fill_in_columns(general_in_table, grp_info);
                udf_info.general_udf(grp_info.num_groups, general_in_table,
                                     update_table);
                delete_table_decref_arrays(general_in_table);
            }
        }
    }

    /**
     * Shuffles the update table and updates the column sets with the newly
     * shuffled table.
     */
    void shuffle() {
        tracing::Event ev("shuffle", is_parallel);
        int64_t num_shuffle_keys = n_shuffle_keys;
        // If we do a reverse shuffle there is no benefit to keeping the shuffle
        // keys as the data doesn't stay shuffled. nunique is heavily optimized
        // so we cannot yet use a subset of keys to shuffle
        if (has_reverse_shuffle || nunique_op) {
            num_shuffle_keys = num_keys;
        }
        ev.add_attribute("passed_n_shuffle_keys", n_shuffle_keys);
        ev.add_attribute("num_shuffle_keys", num_shuffle_keys);
        table_info* shuf_table =
            shuffle_table(update_table, num_shuffle_keys, is_parallel);

        // NOTE: shuffle_table_kernel decrefs input arrays
        delete_table(update_table);
        update_table = cur_table = shuf_table;

        // update column sets with columns from shuffled table
        auto it = update_table->columns.begin() + num_keys;
        for (auto col_set : col_sets) {
            it = col_set->update_after_shuffle(it);
        }
    }

    /**
     * The combine step is performed after update and shuffle. It groups rows
     * in shuffled table based on keys, and aggregates them based on the
     * function to be applied to the columns. More specifically, it will invoke
     * the combine method of each column set.
     */
    void combine() {
        tracing::Event ev("combine", is_parallel);
        update_table->num_keys = num_keys;
        grp_infos.clear();
        std::vector<table_info*> tables = {update_table};
        get_group_info(tables, hashes, nunique_hashes, grp_infos, false,
                       key_dropna, is_parallel);
        grouping_info& grp_info = grp_infos[0];
        num_groups = grp_info.num_groups;
        grp_info.dispatch_table = dispatch_table;
        grp_info.dispatch_info = dispatch_info;
        grp_info.mode = 2;

        combine_table = cur_table = new table_info();
        alloc_init_keys({update_table}, combine_table);
        std::vector<array_info*> list_arr;
        for (auto col_set : col_sets) {
            std::vector<array_info*> list_arr;
            col_set->alloc_combine_columns(num_groups, list_arr);
            for (auto& e_arr : list_arr) {
                combine_table->columns.push_back(e_arr);
            }
            col_set->combine(grp_info);
        }
        if (n_udf > 0) {
            udf_info.combine(update_table, combine_table,
                             grp_info.row_to_group.data());
        }
        delete_table_decref_arrays(update_table);
    }

    /**
     * The eval step generates the final result (output column) for each column
     * set. It call the eval method of each column set.
     */
    void eval() {
        tracing::Event ev("eval", is_parallel);
        for (auto col_set : col_sets) col_set->eval(grp_infos[0]);
        // only regular UDFs need eval step
        if (n_udf - gen_udf_col_sets.size() > 0) {
            udf_info.eval(cur_table);
        }
    }

    /**
     * Returns the final output table which is the result of the groupby.
     */
    table_info* getOutputTable(int64_t* n_out_rows) {
        if (maintain_input_size) {
            // These operations are all defined to maintain the same
            // length as the input.
            *n_out_rows = orig_in_table->nrows();
        } else {
            *n_out_rows = cur_table->nrows();
        }
        table_info* out_table = new table_info();
        if (return_key) {
            out_table->columns.assign(cur_table->columns.begin(),
                                      cur_table->columns.begin() + num_keys);
        }

        // gb.head() with distirbuted data sorted the table so col_sets no
        // longer reflects final
        // output columns.
        if (head_op && is_parallel) {
            for (uint64_t i = 0; i < cur_table->ncols(); i++) {
                out_table->columns.push_back(cur_table->columns[i]);
            }
        } else {
            for (BasicColSet* col_set : col_sets) {
                out_table->columns.push_back(col_set->getOutputColumn());
            }
            // gb.head() already added index to out_table.
            if (!head_op && return_index) {
                out_table->columns.push_back(cur_table->columns.back());
            }
        }
        if ((cumulative_op || shift_op || transform_op || ngroup_op ||
             window_op) &&
            is_parallel) {
            table_info* revshuf_table = reverse_shuffle_table_kernel(
                out_table, in_hashes, *comm_info_ptr);
            delete[] in_hashes;
            delete comm_info_ptr;
            delete_table(out_table);
            out_table = revshuf_table;
        }
        delete cur_table;
        return out_table;
    }

    /**
     * We enter this algorithm at the beginning of the groupby pipeline, if
     * there are gb.nunique operations and there is no other operation that
     * requires shuffling before update. This algorithm decides, for each
     * nunique column, whether all ranks drop duplicates locally for that column
     * based on average local cardinality estimates across all ranks, and will
     * also decide how to shuffle all the nunique columns (it will use the same
     * scheme to shuffle all the nunique columns since the decision is not
     * based on the characteristics on any particular column). There are two
     * strategies for shuffling:
     * a) Shuffle based on groupby keys. Shuffles nunique data to its final
     *    destination. If there are no other groupby operations other than
     *    nunique then this equals shuffle_before_update=true and we just
     *    need an update and eval step (no second shuffle and combine). But
     *    if there are other operations mixed in, for simplicity we will do
     *    update, shuffle and combine step for nunique columns even though
     *    the nunique data is already in the final destination.
     * b) Shuffle based on keys+value. This is done if the number of *global*
     *    groups is small compared to the number of ranks, since shuffling
     *    based on keys in this case can generate significant load imbalance.
     *    In this case the update step calculates number of unique values
     *    for (key, value) tuples, the second shuffle (after update) collects
     *    the nuniques for a given group on the same rank, and the combine sums
     *    them.
     * @param ftypes: list of groupby function types passed directly from
     * GroupbyPipeline constructor.
     * @param num_funcs: number of functions in ftypes
     * @param nunique_hashes_global: estimated number of global unique hashes
     * of groupby keys (gives an estimate of global number of unique groups)
     */
    void gb_nunique_preprocess(int* ftypes, int num_funcs,
                               size_t nunique_hashes_global) {
        tracing::Event ev("gb_nunique_preprocess", is_parallel);
        if (!is_parallel) {
            throw std::runtime_error(
                "gb_nunique_preprocess called for non-distributed data");
        }
        if (shuffle_before_update) {
            throw std::runtime_error(
                "gb_nunique_preprocess called with shuffle_before_update=true");
        }

        // If it's just nunique we set table_id_counter to 0 because we won't
        // add in_table to our list of tables. Otherwise, set to 1 as 0 is
        // reserved for in_table
        int table_id_counter = 1;
        if (nunique_only) {
            table_id_counter = 0;
        }

        static constexpr float threshold_of_fraction_of_unique_hash = 0.5;
        ev.add_attribute("g_threshold_of_fraction_of_unique_hash",
                         threshold_of_fraction_of_unique_hash);

        // If the number of global groups is small we need to shuffle
        // based on keys *and* values to maximize data distribution
        // and improve scaling. If we only spread based on keys, scaling
        // will be limited by the number of groups.
        int num_ranks;
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        size_t num_ranks_unsigned = size_t(num_ranks);
        // When number of groups starts to approximate the number of ranks
        // there will be a high chance that a single rank ends up with 2-3
        // times the load (number of groups) than others after shuffling
        // TODO investigate what is the best threshold:
        // https://bodo.atlassian.net/browse/BE-1308
        const bool shuffle_by_keys_and_value =
            (nunique_hashes_global <= num_ranks_unsigned * 3);
        ev.add_attribute("g_nunique_shuffle_by_keys_and_values",
                         shuffle_by_keys_and_value);

        for (int i = 0, col_idx = num_keys; i < num_funcs; i++, col_idx++) {
            if (ftypes[i] != Bodo_FTypes::nunique) {
                continue;
            }

            table_info* tmp = new table_info();
            tmp->columns.assign(in_table->columns.begin(),
                                in_table->columns.begin() + num_keys);
            tmp->num_keys = num_keys;
            tmp->columns.push_back(in_table->columns[col_idx]);

            // --------- drop local duplicates ---------
            // If we know that the |set(values)| / len(values)
            // is low on all ranks then it should be beneficial to
            // drop local duplicates before the shuffle.

            const size_t n_rows = static_cast<size_t>(in_table->nrows());
            // get hashes of keys+value
            uint32_t* key_value_hashes = new uint32_t[n_rows];
            memcpy(key_value_hashes, hashes, sizeof(uint32_t) * n_rows);
            // TODO: do a hash combine that writes to an empty hash
            // array to avoid memcpy?
            hash_array_combine(key_value_hashes, tmp->columns[num_keys], n_rows,
                               SEED_HASH_PARTITION,
                               /*global_dict_needed=*/true, is_parallel);

            // Compute the local fraction of unique hashes
            size_t nunique_keyval_hashes =
                get_nunique_hashes(key_value_hashes, n_rows, is_parallel);
            float local_fraction_unique_hashes =
                static_cast<float>(nunique_keyval_hashes) /
                static_cast<float>(n_rows);
            float global_fraction_unique_hashes;
            if (ev.is_tracing())
                ev.add_attribute("nunique_" + std::to_string(i) +
                                     "_local_fraction_unique_hashes",
                                 local_fraction_unique_hashes);
            MPI_Allreduce(&local_fraction_unique_hashes,
                          &global_fraction_unique_hashes, 1, MPI_FLOAT, MPI_SUM,
                          MPI_COMM_WORLD);
            global_fraction_unique_hashes /= static_cast<float>(num_ranks);
            ev.add_attribute("g_nunique_" + std::to_string(i) +
                                 "_global_fraction_unique_hashes",
                             global_fraction_unique_hashes);
            const bool drop_duplicates = global_fraction_unique_hashes <
                                         threshold_of_fraction_of_unique_hash;
            ev.add_attribute(
                "g_nunique_" + std::to_string(i) + "_drop_duplicates",
                drop_duplicates);

            // Regardless of whether we drop duplicates or not, the references
            // to the original input arrays are going to be decremented (by
            // either drop_duplicates_table_inner or shuffle_table), but we
            // still need the references for the code after C++ groupby
            for (auto a : tmp->columns) incref_array(a);
            table_info* tmp2 = nullptr;
            if (drop_duplicates) {
                // Set dropna to false because skipna is handled at
                // a later step. Setting dropna=True here removes NA
                // from the keys, which we do not want
                tmp2 = drop_duplicates_table_inner(
                    tmp, tmp->ncols(), 0, 1, is_parallel, false,
                    /*drop_duplicates_dict=*/true, key_value_hashes);
                delete tmp;
                tmp = tmp2;
            }

            // --------- shuffle column ---------
            if (shuffle_by_keys_and_value) {
                if (drop_duplicates) {
                    // Note that tmp here no longer contains the
                    // original input arrays
                    tmp2 = shuffle_table(tmp, tmp->ncols(), is_parallel);
                } else {
                    // Since the arrays are unmodified we can reuse the hashes
                    tmp2 = shuffle_table(tmp, tmp->ncols(), is_parallel, false,
                                         key_value_hashes);
                }
            } else {
                if (drop_duplicates) {
                    tmp2 = shuffle_table(tmp, num_keys, is_parallel);
                } else {
                    tmp2 = shuffle_table(tmp, num_keys, is_parallel, false,
                                         hashes);
                }
            }
            delete[] key_value_hashes;
            delete tmp;
            tmp2->num_keys = num_keys;
            tmp2->id = table_id_counter++;
            nunique_tables[col_idx] = tmp2;
        }

        if (!shuffle_by_keys_and_value && nunique_only)
            // We have shuffled the data to its final destination so this is
            // equivalent to shuffle_before_update=true and we don't need to
            // do a combine step
            shuffle_before_update = true;

        if (nunique_only) {
            // in the case of nunique_only the hashes that we calculated in
            // GroupbyPipeline() are not valid, since we have shuffled all of
            // the input columns
            delete[] hashes;
            hashes = nullptr;
        }
    }

    /**
     * @brief Get key_col given a group number
     *
     * @param group[in]: group number
     * @param from_tables[in] list of tables
     * @param key_col_idx[in]
     * @return std::tuple<array_info*, int64_t> Tuple of the column and the row
     * containing the group.
     */
    std::tuple<array_info*, int64_t> find_key_for_group(
        int64_t group, const std::vector<table_info*>& from_tables,
        int64_t key_col_idx) {
        for (size_t k = 0; k < grp_infos.size(); k++) {
            int64_t key_row = grp_infos[k].group_to_first_row[group];
            if (key_row >= 0) {
                array_info* key_col = (*from_tables[k])[key_col_idx];
                return {key_col, key_row};
            }
        }
        throw std::runtime_error("No valid row found for group: " +
                                 std::to_string(group));
    }

    /**
     * Allocate and fill key columns, based on grouping info. It uses the
     * values of key columns from from_table to populate out_table.
     */
    void alloc_init_keys(std::vector<table_info*> from_tables,
                         table_info* out_table) {
        int64_t key_row = 0;
        for (int64_t i = 0; i < num_keys; i++) {
            array_info* key_col = (*from_tables[0])[i];
            array_info* new_key_col = nullptr;
            if (key_col->arr_type == bodo_array_type::NUMPY ||
                key_col->arr_type == bodo_array_type::CATEGORICAL ||
                key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                new_key_col =
                    alloc_array(num_groups, 1, 1, key_col->arr_type,
                                key_col->dtype, 0, key_col->num_categories);
                int64_t dtype_size = numpy_item_size[key_col->dtype];
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i);
                    memcpy(new_key_col->data1 + j * dtype_size,
                           key_col->data1 + key_row * dtype_size, dtype_size);
                }
                if (key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                    for (size_t j = 0; j < num_groups; j++) {
                        std::tie(key_col, key_row) =
                            find_key_for_group(j, from_tables, i);
                        bool bit = key_col->get_null_bit(key_row);
                        new_key_col->set_null_bit(j, bit);
                    }
                }
            }
            if (key_col->arr_type == bodo_array_type::DICT) {
                array_info* key_indices = key_col->info2;
                array_info* new_key_indices =
                    alloc_array(num_groups, -1, -1, key_indices->arr_type,
                                key_indices->dtype, 0, 0);
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i);
                    // Update key_indices with the new key col
                    key_indices = key_col->info2;
                    new_key_indices->at<dict_indices_t>(j) =
                        key_indices->at<dict_indices_t>(key_row);
                    bool bit = key_indices->get_null_bit(key_row);
                    new_key_indices->set_null_bit(j, bit);
                }
                new_key_col = new array_info(
                    bodo_array_type::DICT, key_col->dtype,
                    new_key_indices->length, -1, -1, NULL, NULL, NULL,
                    new_key_indices->null_bitmask, NULL, NULL, NULL, NULL, 0, 0,
                    0, key_col->has_global_dictionary,
                    key_col->has_deduped_local_dictionary,
                    key_col->has_sorted_dictionary, key_col->info1,
                    new_key_indices);
                // incref because they share the same dictionary array
                incref_array(key_col->info1);
            }
            if (key_col->arr_type == bodo_array_type::STRING) {
                // new key col will have num_groups rows containing the
                // string for each group
                int64_t n_chars = 0;  // total number of chars of all keys for
                                      // this column
                offset_t* in_offsets;
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i);
                    in_offsets = (offset_t*)key_col->data2;
                    n_chars += in_offsets[key_row + 1] - in_offsets[key_row];
                }
                new_key_col =
                    alloc_array(num_groups, n_chars, 1, key_col->arr_type,
                                key_col->dtype, 0, key_col->num_categories);

                offset_t* out_offsets = (offset_t*)new_key_col->data2;
                offset_t pos = 0;
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i);
                    in_offsets = (offset_t*)key_col->data2;
                    offset_t start_offset = in_offsets[key_row];
                    offset_t str_len = in_offsets[key_row + 1] - start_offset;
                    out_offsets[j] = pos;
                    memcpy(&new_key_col->data1[pos],
                           &key_col->data1[start_offset], str_len);
                    pos += str_len;
                    bool bit = key_col->get_null_bit(key_row);
                    new_key_col->set_null_bit(j, bit);
                }
                out_offsets[num_groups] = pos;
            }
            if (key_col->arr_type == bodo_array_type::LIST_STRING) {
                // new key col will have num_groups rows containing the
                // list string for each group
                int64_t n_strings = 0;  // total number of strings of all keys
                                        // for this column
                int64_t n_chars = 0;    // total number of chars of all keys for
                                        // this column
                offset_t* in_index_offsets;
                offset_t* in_data_offsets;
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i);
                    in_index_offsets = (offset_t*)key_col->data3;
                    in_data_offsets = (offset_t*)key_col->data2;
                    n_strings += in_index_offsets[key_row + 1] -
                                 in_index_offsets[key_row];
                    n_chars += in_data_offsets[in_index_offsets[key_row + 1]] -
                               in_data_offsets[in_index_offsets[key_row]];
                }
                new_key_col = alloc_array(num_groups, n_strings, n_chars,
                                          key_col->arr_type, key_col->dtype, 0,
                                          key_col->num_categories);
                uint8_t* in_sub_null_bitmask =
                    (uint8_t*)key_col->sub_null_bitmask;
                uint8_t* out_sub_null_bitmask =
                    (uint8_t*)new_key_col->sub_null_bitmask;
                offset_t* out_index_offsets = (offset_t*)new_key_col->data3;
                offset_t* out_data_offsets = (offset_t*)new_key_col->data2;
                offset_t pos_data = 0;
                offset_t pos_index = 0;
                out_data_offsets[0] = 0;
                out_index_offsets[0] = 0;
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i);
                    in_index_offsets = (offset_t*)key_col->data3;
                    in_data_offsets = (offset_t*)key_col->data2;
                    offset_t size_index = in_index_offsets[key_row + 1] -
                                          in_index_offsets[key_row];
                    offset_t pos_start = in_index_offsets[key_row];
                    for (offset_t i_str = 0; i_str < size_index; i_str++) {
                        offset_t len_str =
                            in_data_offsets[pos_start + i_str + 1] -
                            in_data_offsets[pos_start + i_str];
                        pos_index++;
                        out_data_offsets[pos_index] =
                            out_data_offsets[pos_index - 1] + len_str;
                        bool bit =
                            GetBit(in_sub_null_bitmask, pos_start + i_str);
                        SetBitTo(out_sub_null_bitmask, pos_index, bit);
                    }
                    out_index_offsets[j + 1] = pos_index;
                    // Now the strings themselves
                    offset_t in_start_offset =
                        in_data_offsets[in_index_offsets[key_row]];
                    offset_t n_chars_o =
                        in_data_offsets[in_index_offsets[key_row + 1]] -
                        in_data_offsets[in_index_offsets[key_row]];
                    memcpy(&new_key_col->data1[pos_data],
                           &key_col->data1[in_start_offset], n_chars_o);
                    pos_data += n_chars_o;
                    bool bit = key_col->get_null_bit(key_row);
                    new_key_col->set_null_bit(j, bit);
                }
            }
            out_table->columns.push_back(new_key_col);
        }
    }

    table_info*
        orig_in_table;  // original input table of groupby received from Python
    table_info* in_table;  // input table of groupby
    int64_t num_keys;
    table_info* dispatch_table;  // input dispatching table of pivot_table
    table_info* dispatch_info;   // input dispatching info of pivot_table
    bool is_parallel;
    bool return_key;
    bool return_index;
    bool key_dropna;
    std::vector<BasicColSet*> col_sets;
    std::vector<GeneralUdfColSet*> gen_udf_col_sets;
    table_info* udf_table;
    int* udf_n_redvars;
    // total number of UDFs applied to input columns (includes regular and
    // general UDFs)
    int n_udf = 0;
    int udf_table_idx = 0;
    // shuffling before update requires more communication and is needed
    // when one of the groupby functions is
    // median/nunique/cumsum/cumprod/cummin/cummax/shift/transform
    bool shuffle_before_update = false;
    bool cumulative_op = false;
    bool shift_op = false;
    bool transform_op = false;
    bool nunique_op = false;
    bool head_op = false;
    bool ngroup_op = false;
    bool window_op = false;
    bool has_reverse_shuffle = false;
    int64_t head_n;
    bool req_extended_group_info = false;
    bool do_combine;
    bool maintain_input_size;
    int64_t n_shuffle_keys;
    bool use_sql_rules;

    // column position in in_table -> table that contains key columns + one
    // nunique column after [dropping local duplicates] + shuffling
    std::map<int, table_info*> nunique_tables;
    bool nunique_only = false;  // there are only groupby nunique operations

    udfinfo_t udf_info;

    table_info* update_table = nullptr;
    table_info* combine_table = nullptr;
    table_info* cur_table = nullptr;

    std::vector<grouping_info> grp_infos;
    size_t num_groups;
    // shuffling stuff
    uint32_t* in_hashes = nullptr;
    mpi_comm_info* comm_info_ptr = nullptr;
    uint32_t* hashes = nullptr;
    size_t nunique_hashes = 0;
};

table_info* groupby_and_aggregate(
    table_info* in_table, int64_t num_keys, bool input_has_index, int* ftypes,
    int* func_offsets, int* udf_nredvars, bool is_parallel, bool skipdropna,
    int64_t periods, int64_t transform_func, int64_t head_n, bool return_key,
    bool return_index, bool key_dropna, void* update_cb, void* combine_cb,
    void* eval_cb, void* general_udfs_cb, table_info* udf_dummy_table,
    int64_t* n_out_rows, bool window_ascending, bool window_na_position,
    bool maintain_input_size, int64_t n_shuffle_keys, bool use_sql_rules) {
    try {
        tracing::Event ev("groupby_and_aggregate", is_parallel);
        int strategy = determine_groupby_strategy(
            in_table, num_keys, ftypes, func_offsets, input_has_index);
        ev.add_attribute("g_strategy", strategy);

        auto implement_strategy0 = [&]() -> table_info* {
            table_info* dispatch_info = nullptr;
            table_info* dispatch_table = nullptr;
            GroupbyPipeline groupby(
                in_table, num_keys, dispatch_table, dispatch_info,
                input_has_index, is_parallel, ftypes, func_offsets,
                udf_nredvars, udf_dummy_table, (udf_table_op_fn)update_cb,
                (udf_table_op_fn)combine_cb, (udf_eval_fn)eval_cb,
                (udf_general_fn)general_udfs_cb, skipdropna, periods,
                transform_func, head_n, return_key, return_index, key_dropna,
                window_ascending, window_na_position, maintain_input_size,
                n_shuffle_keys, use_sql_rules);

            table_info* ret_table = groupby.run(n_out_rows);
            return ret_table;
        };
        auto implement_categorical_exscan =
            [&](array_info* cat_column) -> table_info* {
            table_info* ret_table =
                mpi_exscan_computation(cat_column, in_table, num_keys, ftypes,
                                       func_offsets, is_parallel, skipdropna,
                                       return_key, return_index, use_sql_rules);
            *n_out_rows = in_table->nrows();
            return ret_table;
        };
        if (strategy == 0) return implement_strategy0();
        if (strategy == 1) {
            array_info* cat_column = in_table->columns[0];
            return implement_categorical_exscan(cat_column);
        }
        if (strategy == 2) {
            array_info* cat_column = compute_categorical_index(
                in_table, num_keys, is_parallel, key_dropna);
            if (cat_column ==
                nullptr) {  // It turns out that there are too many
                            // different keys for exscan to be ok.
                return implement_strategy0();
            } else {
                table_info* ret_table =
                    implement_categorical_exscan(cat_column);
                delete_info_decref_array(cat_column);
                return ret_table;
            }
        }
        return nullptr;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
