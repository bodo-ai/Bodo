// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _ARRAY_HASH_H_INCLUDED
#define _ARRAY_HASH_H_INCLUDED

#include "_array_utils.h"
#include "_bodo_common.h"

#define SEED_HASH_PARTITION 0xb0d01289
#define SEED_HASH_MULTIKEY 0xb0d01288
#define SEED_HASH_DROPDUPLI 0xb0d01287
#define SEED_HASH_JOIN 0xb0d01286
#define SEED_HASH_GROUPBY_SHUFFLE 0xb0d01285
#define SEED_HASH_PIVOT_SHUFFLE 0xb0d01285
#define SEED_HASH_CONTAINER 0xb0d01284

void hash_array_combine(uint32_t* out_hashes, array_info* array, size_t n_rows,
                        const uint32_t seed, bool global_dict_needed);

/**
 * Function for the computation of hashes for keys
 *
 * @param key_arrs: input keys to hashThe hashes on output.
 * @param seed: the seed of the computation.
 * @param is_parallel: whether we run in parallel or not.
 * @return hash keys
 *
 */
uint32_t* hash_keys(std::vector<array_info*> const& key_arrs,
                    const uint32_t seed, bool is_parallel,
                    bool global_dict_needed = true);

uint32_t* coherent_hash_keys(std::vector<array_info*> const& key_arrs,
                             std::vector<array_info*> const& ref_key_arrs,
                             const uint32_t seed);

void hash_array(uint32_t* out_hashes, array_info* array, size_t n_rows,
                const uint32_t seed, bool is_parallel, bool global_dict_needed);

/**
 * Function for the getting table keys and returning its hashes
 *
 * @param in_table: the input table
 * @param num_keys : the number of keys
 * @param seed: the seed of the computation.
 * @param is_parallel: whether we run in parallel or not.
 * @return hash keys
 *
 */
inline uint32_t* hash_keys_table(table_info* in_table, size_t num_keys,
                                 uint32_t seed, bool is_parallel) {
    tracing::Event ev("hash_keys_table", is_parallel);
    std::vector<array_info*> key_arrs(in_table->columns.begin(),
                                      in_table->columns.begin() + num_keys);
    return hash_keys(key_arrs, seed, is_parallel);
}

inline uint32_t* coherent_hash_keys_table(table_info* in_table,
                                          table_info* ref_table,
                                          size_t num_keys, uint32_t seed) {
    std::vector<array_info*> key_arrs(in_table->columns.begin(),
                                      in_table->columns.begin() + num_keys);
    std::vector<array_info*> ref_key_arrs(
        ref_table->columns.begin(), ref_table->columns.begin() + num_keys);
    return coherent_hash_keys(key_arrs, ref_key_arrs, seed);
}

/**
 * Multi column key that can be used to hash the value of multiple columns
 * in a dataframe row in C++ hash tables/sets.
 * NOTE: This assumes the key columns are the first columns in the table.
 */
struct multi_col_key {
    uint32_t hash;
    table_info* table;
    int64_t row;

    multi_col_key(uint32_t _hash, table_info* _table, int64_t _row)
        : hash(_hash), table(_table), row(_row) {}

    bool operator==(const multi_col_key& other) const {
        for (int64_t i = 0; i < table->num_keys; i++) {
            array_info* c1 = table->columns[i];
            array_info* c2 = other.table->columns[i];
            size_t siztype;
            switch (c1->arr_type) {
                case bodo_array_type::ARROW: {
                    int64_t pos1_s = row;
                    int64_t pos1_e = row + 1;
                    int64_t pos2_s = other.row;
                    int64_t pos2_e = other.row + 1;
                    bool na_position_bis = true;
                    int test = ComparisonArrowColumn(c1->array, pos1_s, pos1_e,
                                                     c2->array, pos2_s, pos2_e,
                                                     na_position_bis);
                    if (test != 0) return false;
                }
                    continue;
                case bodo_array_type::DICT: {
                    if (c1->has_global_dictionary &&
                        c2->has_global_dictionary) {
                        if (c1->info1 != c2->info1) {
                            throw std::runtime_error(
                                "multi-key-hashing dictionary the columns are "
                                "not unified.");
                        }
                        return c1->info2->at<dict_indices_t>(row) ==
                               c2->info2->at<dict_indices_t>(other.row);
                    } else {
                        throw std::runtime_error(
                            "multi-key-hashing dictionary array requires "
                            "global dictionary");
                    }
                }
                    continue;
                case bodo_array_type::NULLABLE_INT_BOOL:
                    if (c1->get_null_bit(row) != c2->get_null_bit(other.row))
                        return false;
                    if (!c1->get_null_bit(row)) continue;
                case bodo_array_type::CATEGORICAL:  // Even in missing case
                                                    // (value -1) this works
                case bodo_array_type::NUMPY:
                    siztype = numpy_item_size[c1->dtype];
                    if (memcmp(c1->data1 + siztype * row,
                               c2->data1 + siztype * other.row, siztype) != 0) {
                        return false;
                    }
                    continue;
                case bodo_array_type::STRING: {
                    uint8_t* c1_null_bitmask = (uint8_t*)c1->null_bitmask;
                    uint8_t* c2_null_bitmask = (uint8_t*)c2->null_bitmask;
                    if (GetBit(c1_null_bitmask, row) !=
                        GetBit(c2_null_bitmask, other.row))
                        return false;
                    offset_t* c1_offsets = (offset_t*)c1->data2;
                    offset_t* c2_offsets = (offset_t*)c2->data2;
                    offset_t c1_str_len = c1_offsets[row + 1] - c1_offsets[row];
                    offset_t c2_str_len =
                        c2_offsets[other.row + 1] - c2_offsets[other.row];
                    if (c1_str_len != c2_str_len) return false;
                    char* c1_str = c1->data1 + c1_offsets[row];
                    char* c2_str = c2->data1 + c2_offsets[other.row];
                    if (memcmp(c1_str, c2_str, c1_str_len) != 0) return false;
                }
                    continue;
                case bodo_array_type::LIST_STRING: {
                    uint8_t* c1_null_bitmask = (uint8_t*)c1->null_bitmask;
                    uint8_t* c2_null_bitmask = (uint8_t*)c2->null_bitmask;
                    if (GetBit(c1_null_bitmask, row) !=
                        GetBit(c2_null_bitmask, other.row))
                        return false;
                    uint8_t* c1_sub_null_bitmask =
                        (uint8_t*)c1->sub_null_bitmask;
                    uint8_t* c2_sub_null_bitmask =
                        (uint8_t*)c2->sub_null_bitmask;
                    offset_t* c1_index_offsets = (offset_t*)c1->data3;
                    offset_t* c2_index_offsets = (offset_t*)c2->data3;
                    offset_t* c1_data_offsets = (offset_t*)c1->data2;
                    offset_t* c2_data_offsets = (offset_t*)c2->data2;
                    // Comparing the number of strings.
                    offset_t c1_index_len =
                        c1_index_offsets[row + 1] - c1_index_offsets[row];
                    offset_t c2_index_len = c2_index_offsets[other.row + 1] -
                                            c2_index_offsets[other.row];
                    if (c1_index_len != c2_index_len) return false;
                    // comparing the length of the strings.
                    for (offset_t u = 0; u < c1_index_len; u++) {
                        offset_t size_data1 =
                            c1_data_offsets[c1_index_offsets[row] + u + 1] -
                            c1_data_offsets[c1_index_offsets[row] + u];
                        offset_t size_data2 =
                            c2_data_offsets[c2_index_offsets[other.row] + u +
                                            1] -
                            c2_data_offsets[c2_index_offsets[other.row] + u];
                        if (size_data1 != size_data2) return false;
                        bool str_bit1 = GetBit(c1_sub_null_bitmask,
                                               c1_index_offsets[row] + u);
                        bool str_bit2 = GetBit(c2_sub_null_bitmask,
                                               c2_index_offsets[other.row] + u);
                        if (str_bit1 != str_bit2) return false;
                    }
                    // Now comparing the strings. Their length is the same since
                    // we pass above check
                    offset_t common_len =
                        c1_data_offsets[c1_index_offsets[row + 1]] -
                        c1_data_offsets[c1_index_offsets[row]];
                    char* c1_strB =
                        c1->data1 + c1_data_offsets[c1_index_offsets[row]];
                    char* c2_strB =
                        c2->data1 +
                        c2_data_offsets[c2_index_offsets[other.row]];
                    if (memcmp(c1_strB, c2_strB, common_len) != 0) return false;
                    continue;
                }
                default: {
                    throw std::runtime_error(
                        "multi_col_key_hash : Unsupported type");
                }
                    continue;
            }
        }
        return true;
    }
};

struct multi_col_key_hash {
    std::size_t operator()(const multi_col_key& k) const { return k.hash; }
};

/**
 * Unifies dictionaries of DICT arrays arr1 and arr2
 * If the arrays have a reference to the same dictionary then
 * this function does nothing and returns.
 * Replaces old dictionary with the new one.
 * Updates the indices to conform to the new dictionary.
 */
void unify_dictionaries(array_info* arr1, array_info* arr2);

// For hashing function involving dictionaries of dictionary-encoded arrays
struct HashDict {
    uint32_t operator()(const size_t iRow) const {
        if (iRow < global_array_rows) {
            return global_array_hashes[iRow];
        } else {
            return local_array_hashes[iRow - global_array_rows];
        }
    }
    const size_t global_array_rows;
    const uint32_t* global_array_hashes;
    const uint32_t* local_array_hashes;
};

// For key comparison involving dictionaries of dictionary-encoded arrays
struct KeyEqualDict {
    bool operator()(const size_t iRowA, const size_t iRowB) const {
        size_t jRowA, jRowB;
        array_info *dict_A, *dict_B;
        if (iRowA < global_array_rows) {
            dict_A = global_dictionary;
            jRowA = iRowA;
        } else {
            dict_A = local_dictionary;
            jRowA = iRowA - global_array_rows;
        }
        if (iRowB < global_array_rows) {
            dict_B = global_dictionary;
            jRowB = iRowB;
        } else {
            dict_B = local_dictionary;
            jRowB = iRowB - global_array_rows;
        }
        // TODO inline?
        return TestEqualColumn(dict_A, jRowA, dict_B, jRowB, true);
    }
    // global_dict_len, global_dictionary, local_dictionary
    size_t global_array_rows;
    array_info* global_dictionary;
    array_info* local_dictionary;
};

#endif  // _ARRAY_HASH_H_INCLUDED
