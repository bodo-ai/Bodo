// Copyright (C) 2023 Bodo Inc. All rights reserved.

#ifndef _GROUPBY_HASHING_H_INCLUDED
#define _GROUPBY_HASHING_H_INCLUDED
#include "_array_utils.h"

/**
 * This file declares any types/structs that are used for creating Hashmaps
 * unique to groupby.
 */

/**
 * Look up a hash in a table with 32 bit hashes.
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashLookupIn32bitTable {
    uint32_t operator()(const int64_t iRow) const { return hashes[iRow]; }
    uint32_t* hashes;
};

/**
 * Check if keys are equal by lookup in a table with 32 bit hashes.
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualLookupIn32bitTable {
    bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        return TestEqualJoin(table, table, iRowA, iRowB, n_keys, true);
    }
    int64_t n_keys;
    table_info* table;
};

/**
 * @brief Function used to deallocate the memory occupied by a hashmap.
 *
 * @tparam CalledFromGroupInfo Was this function called from group_info?
 * @tparam Map The type of the hashmap.
 * @param hashes The hashes of the keys to be freed.
 * @param key_to_group The hashmap to deallocate.
 * @param is_parallel Is this function called in parallel? This is used for
 * tracing.
 */
template <bool CalledFromGroupInfo, typename Map>
void do_map_dealloc(uint32_t*& hashes, Map& key_to_group, bool is_parallel) {
    tracing::Event ev_dealloc(CalledFromGroupInfo
                                  ? "get_group_info_dealloc"
                                  : "get_groupby_labels_dealloc",
                              is_parallel);
    delete[] hashes;
    hashes = nullptr;  // updates hashes ptr at caller

    if (ev_dealloc.is_tracing()) {
        ev_dealloc.add_attribute("map size", key_to_group.size());
        ev_dealloc.add_attribute("map bucket_count",
                                 key_to_group.bucket_count());
        ev_dealloc.add_attribute("map load_factor", key_to_group.load_factor());
        ev_dealloc.add_attribute("map max_load_factor",
                                 key_to_group.max_load_factor());
    }
    key_to_group.clear();
    key_to_group.reserve(0);  // try to force dealloc of hash map
    ev_dealloc.finalize();
}

// Hashing info for the MPI_EXSCAN PATH

/**
 * Compute hash for `compute_categorical_index`
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashComputeCategoricalIndex {
    size_t operator()(size_t const iRow) const {
        if (iRow < n_rows_full) {
            return static_cast<size_t>(hashes_full[iRow]);
        } else {
            return static_cast<size_t>(hashes_in_table[iRow - n_rows_full]);
        }
    }
    uint32_t* hashes_full;
    uint32_t* hashes_in_table;
    size_t n_rows_full;
};

/**
 * Key comparison for `compute_categorical_index`
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashEqualComputeCategoricalIndex {
    bool operator()(size_t const iRowA, size_t const iRowB) const {
        size_t jRowA, jRowB, shift_A, shift_B;
        if (iRowA < n_rows_full) {
            shift_A = 0;
            jRowA = iRowA;
        } else {
            shift_A = num_keys;
            jRowA = iRowA - n_rows_full;
        }
        if (iRowB < n_rows_full) {
            shift_B = 0;
            jRowB = iRowB;
        } else {
            shift_B = num_keys;
            jRowB = iRowB - n_rows_full;
        }
        bool test =
            TestEqual(*concat_column, num_keys, shift_A, jRowA, shift_B, jRowB);
        return test;
    }
    int64_t num_keys;
    size_t n_rows_full;
    std::vector<array_info*>* concat_column;
};

#endif  // _GROUPBY_HASHING_H_INCLUDED
