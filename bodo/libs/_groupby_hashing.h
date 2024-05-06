// Copyright (C) 2023 Bodo Inc. All rights reserved.

#pragma once

#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_murmurhash3.h"

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
    std::shared_ptr<uint32_t[]>& hashes;
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
    std::shared_ptr<table_info> table;
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
void do_map_dealloc(std::shared_ptr<uint32_t[]>& hashes, Map& key_to_group,
                    bool is_parallel) {
    tracing::Event ev_dealloc(CalledFromGroupInfo
                                  ? "get_group_info_dealloc"
                                  : "get_groupby_labels_dealloc",
                              is_parallel);
    hashes.reset();
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
    std::shared_ptr<uint32_t[]>& hashes_full;
    std::shared_ptr<uint32_t[]>& hashes_in_table;
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
    std::vector<std::shared_ptr<array_info>>* concat_column;
};

// HASHING FOR NUNIQUE

/**
 * Compute hash for numpy or nullable_int_bool bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashNuniqueComputationNumpyOrNullableIntBool {
    uint32_t operator()(const int64_t i) const {
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
               arr->arr_type == bodo_array_type::NUMPY);
        uint32_t retval = 0;
        if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
            arr->dtype == Bodo_CTypes::_BOOL) {
            // Nullable booleans store 1 bit per boolean
            bool bit = GetBit(
                (uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(), i);
            hash_inner_32<bool>(&bit, seed, &retval);
        } else {
            // The implementation of data1 is the same for both numpy and
            // nullable
            char* ptr = arr->data1<bodo_array_type::NUMPY>() + i * siztype;
            hash_string_32(ptr, siztype, seed, &retval);
        }
        return retval;
    }
    std::shared_ptr<array_info> arr;
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
        if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
            arr->dtype == Bodo_CTypes::_BOOL) {
            // Nullable booleans store 1 bit per boolean
            bool bit1 = GetBit(
                (uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(), i1);
            bool bit2 = GetBit(
                (uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(), i2);
            return bit1 == bit2;
        } else {
            assert(arr->arr_type == bodo_array_type::NUMPY ||
                   arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
            // data1 is identical for NUMPY and NULLABLE_INT_BOOL, so we fix it
            // to NUMPY for better performance.
            char* ptr1 = arr->data1<bodo_array_type::NUMPY>() + i1 * siztype;
            char* ptr2 = arr->data1<bodo_array_type::NUMPY>() + i2 * siztype;
            return memcmp(ptr1, ptr2, siztype) == 0;
        }
    }

    std::shared_ptr<array_info> arr;
    size_t siztype;
};

/**
 * Compute hash for string bodo types.
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashNuniqueComputationString {
    size_t operator()(const int64_t i) const {
        assert(this->arr->arr_type == bodo_array_type::STRING);
        char* val_chars =
            this->arr->data1<bodo_array_type::STRING>() + in_offsets[i];
        size_t len = in_offsets[i + 1] - in_offsets[i];
        uint32_t val;
        hash_string_32(val_chars, len, seed, &val);
        return size_t(val);
    }
    std::shared_ptr<array_info> arr;
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
        assert(this->arr->arr_type == bodo_array_type::STRING);
        size_t len1 = in_offsets[i1 + 1] - in_offsets[i1];
        size_t len2 = in_offsets[i2 + 1] - in_offsets[i2];
        if (len1 != len2) {
            return false;
        }
        char* ptr1 = arr->data1<bodo_array_type::STRING>() + in_offsets[i1];
        char* ptr2 = arr->data1<bodo_array_type::STRING>() + in_offsets[i2];
        return memcmp(ptr1, ptr2, len1) == 0;
    }

    std::shared_ptr<array_info> arr;
    offset_t* in_offsets;
};

/**
 * Tools to define a custom hashing scheme for int128 if one does not already
 * exist.
 */
template <typename T>
concept Hashable = requires(T a) {
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept Int128NotHashable = !Hashable<T> && std::is_same_v<T, __int128_t>;

/**
 * Function to safely combine two hash outputs into another hash value.
 * https://github.com/boostorg/container_hash/blob/504857692148d52afe7110bcb96cf837b0ced9d7/include/boost/container_hash/hash.hpp#L337
 *
 * @param[in,out] h1: first input hash value, which is modified in place
 * to be combined with the second input hash yet still have the properties
 * of a hash function.
 * @param[in] h2: second input hash value.
 */
inline void hash_combine(size_t& h, size_t k) {
    const size_t m = 0xc6a4a7935bd1e995;
    const int r = 47;
    k *= m;
    k ^= k >> r;
    k *= m;
    h ^= k;
    h *= m;
    // Prevent zeros from hashing to zero
    h += 0xe6546b64;
}

namespace std {
template <Int128NotHashable T>
struct hash<T> {
    uint32_t operator()(T var) const {
        // Obtain the hash values of each of the halves of the
        // 128 bit integer
        size_t h1 = std::hash<int64_t>{}((int64_t)var);
        size_t h2 = std::hash<int64_t>{}((int64_t)(var >> 64));
        // Combine the hashes of the two halves
        hash_combine(h1, h2);
        return h1;
    }
};
}  // namespace std
