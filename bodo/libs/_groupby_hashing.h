// Copyright (C) 2023 Bodo Inc. All rights reserved.

#ifndef _GROUPBY_HASHING_H_INCLUDED
#define _GROUPBY_HASHING_H_INCLUDED
#include "_array_hash.h"
#include "_array_utils.h"
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
    std::vector<array_info*>* concat_column;
};

// HASHING FOR NUNIQUE

/**
 * Compute hash for numpy or nullable_int_bool bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashNuniqueComputationNumpyOrNullableIntBool {
    uint32_t operator()(const int64_t i) const {
        uint32_t retval = 0;
        if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
            arr->dtype == Bodo_CTypes::_BOOL) {
            // Nullable booleans store 1 bit per boolean
            bool bit = GetBit((uint8_t*)arr->data1(), i);
            hash_inner_32<bool>(&bit, seed, &retval);
        } else {
            char* ptr = arr->data1() + i * siztype;
            hash_string_32(ptr, siztype, seed, &retval);
        }
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
        if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
            arr->dtype == Bodo_CTypes::_BOOL) {
            // Nullable booleans store 1 bit per boolean
            bool bit1 = GetBit((uint8_t*)arr->data1(), i1);
            bool bit2 = GetBit((uint8_t*)arr->data1(), i2);
            return bit1 == bit2;
        } else {
            char* ptr1 = arr->data1() + i1 * siztype;
            char* ptr2 = arr->data1() + i2 * siztype;
            return memcmp(ptr1, ptr2, siztype) == 0;
        }
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
        char* val_chars = arr->data1() + in_data_offsets[in_index_offsets[i]];
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
        if (bit1 != bit2) {
            return false;  // That first case, might not be necessary.
        }
        size_t len1 = in_index_offsets[i1 + 1] - in_index_offsets[i1];
        size_t len2 = in_index_offsets[i2 + 1] - in_index_offsets[i2];
        if (len1 != len2) {
            return false;
        }
        for (size_t u = 0; u < len1; u++) {
            offset_t len_str1 = in_data_offsets[in_index_offsets[i1] + 1] -
                                in_data_offsets[in_index_offsets[i1]];
            offset_t len_str2 = in_data_offsets[in_index_offsets[i2] + 1] -
                                in_data_offsets[in_index_offsets[i2]];
            if (len_str1 != len_str2) {
                return false;
            }
            bool bit1 = GetBit(sub_null_bitmask, in_index_offsets[i1]);
            bool bit2 = GetBit(sub_null_bitmask, in_index_offsets[i2]);
            if (bit1 != bit2) {
                return false;
            }
        }
        offset_t nb_char1 = in_data_offsets[in_index_offsets[i1 + 1]] -
                            in_data_offsets[in_index_offsets[i1]];
        offset_t nb_char2 = in_data_offsets[in_index_offsets[i2 + 1]] -
                            in_data_offsets[in_index_offsets[i2]];
        if (nb_char1 != nb_char2) {
            return false;
        }
        char* ptr1 = arr->data1() +
                     sizeof(offset_t) * in_data_offsets[in_index_offsets[i1]];
        char* ptr2 = arr->data1() +
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
        char* val_chars = arr->data1() + in_offsets[i];
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
        char* ptr1 = arr->data1() + in_offsets[i1];
        char* ptr2 = arr->data1() + in_offsets[i2];
        return memcmp(ptr1, ptr2, len1) == 0;
    }

    array_info* arr;
    offset_t* in_offsets;
};

#endif  // _GROUPBY_HASHING_H_INCLUDED
