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

#endif  // _GROUPBY_HASHING_H_INCLUDED
