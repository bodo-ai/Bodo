// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_bodo_common.h"
#include "_murmurhash3.h"


/**
 * Computation of the inner hash of the functions. This covers the NUMPY case.
 *
 * @param out_hashes: The hashes on output.
 * @param array: the list of data in input.
 * @param n_rows: the number of rows of the table.
 * @param seed: the seed of the computation.
 *
 */
template <class T>
static void hash_array_inner(uint32_t* out_hashes, T* data, size_t n_rows,
                             const uint32_t seed) {
    for (size_t i = 0; i < n_rows; i++) {
        hash_inner_32<T>(&data[i], seed, &out_hashes[i]);
    }
}


/**
 * Computation of the hashes for the case of list of strings array column. Covers LIST_STRING
 *
 * @param out_hashes: The hashes on output.
 * @param data: the strings
 * @param data_offsets: the data offsets (that separates the strings)
 * @param index_offsets: the index offsets (for separating the block of strings)
 * @param null_bitmap: the bitmap array for the values.
 * @param n_rows: the number of rows of the table.
 * @param seed: the seed of the computation.
 *
 * The hash is computed in 3 stages:
 * 1) The hash of the concatenated strings
 * 2) The hash of the string length
 * 3) The hash of the bitmask
 */
static void hash_array_list_string(uint32_t* out_hashes,
                                   char* data, uint32_t* data_offsets, uint32_t* index_offsets, uint8_t* null_bitmask,
                                   size_t n_rows,
                                   const uint32_t seed) {
    uint32_t start_index_offset = 0;
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t hash1, hash2, hash3;
        // First the hash from the strings.
        uint32_t end_index_offset = index_offsets[i + 1];
        uint32_t len1 = data_offsets[end_index_offset] - data_offsets[start_index_offset];
        std::string val(&data[data_offsets[start_index_offset]], len1);
        const char* val_chars1 = val.c_str();
        hash_string_32(val_chars1, (const int)len1, seed, &hash1);
        // Second the hash from the length of strings (approx that most strings have less than 256 characters)
        uint32_t len2=end_index_offset - start_index_offset;
        std::vector<char> V(len2);
        for (size_t j=0; j<len2; j++) {
          uint32_t n_chars = data_offsets[start_index_offset+j+1] - data_offsets[start_index_offset+j];
          V[j] = (char)n_chars;
        }
        const char* val_chars2 = V.data();
        hash_string_32(val_chars2, (const int)len2, hash1, &hash2);
        // Third the hash from whether it is missing or not
        bool bit = GetBit(null_bitmask, i);
        char val_sing = bit;
        hash_string_32(&val_sing, 1, hash2, &hash3);
        out_hashes[i] = hash3;
        start_index_offset = end_index_offset;
    }
}


/**
 * Computation of the hashes for the case of strings array column. Covers STRING
 *
 * @param out_hashes: The hashes on output.
 * @param data: the strings
 * @param offsets: the offsets (that separates the strings)
 * @param null_bitmap: the bitmap array for the values.
 * @param n_rows: the number of rows of the table.
 * @param seed: the seed of the computation.
 *
 * Right now, the bitmask is not used in the computation, which
 * may be a problem to consider later on.
 */
static void hash_array_string(uint32_t* out_hashes, char* data,
                              uint32_t* offsets, uint8_t* null_bitmask, size_t n_rows,
                              const uint32_t seed) {
    uint32_t start_offset = 0;
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t end_offset = offsets[i + 1];
        uint32_t len = end_offset - start_offset;
        std::string val(&data[start_offset], len);
        const char* val_chars = val.c_str();
        hash_string_32(val_chars, (const int)len, seed, &out_hashes[i]);
        start_offset = end_offset;
    }
}


/**
 * Top function for the computation of the hashes. It calls all the other hash functions.
 *
 * @param out_hashes: The hashes on output.
 * @param array: the list of columns in input
 * @param n_rows: the number of rows of the table.
 * @param seed: the seed of the computation.
 *
 */
void hash_array(uint32_t* out_hashes, array_info* array, size_t n_rows,
                const uint32_t seed) {
    // dispatch to proper function
    // TODO: general dispatcher
    // XXX: assumes nullable array data for nulls is always consistent
    if (array->dtype == Bodo_CTypes::_BOOL)
        return hash_array_inner<bool>(out_hashes, (bool*)array->data1, n_rows,
                                      seed);
    if (array->dtype == Bodo_CTypes::INT8)
        return hash_array_inner<int8_t>(out_hashes, (int8_t*)array->data1,
                                        n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT8)
        return hash_array_inner<uint8_t>(out_hashes, (uint8_t*)array->data1,
                                         n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT16)
        return hash_array_inner<int16_t>(out_hashes, (int16_t*)array->data1,
                                         n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT16)
        return hash_array_inner<uint16_t>(out_hashes, (uint16_t*)array->data1,
                                          n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT32)
        return hash_array_inner<int32_t>(out_hashes, (int32_t*)array->data1,
                                         n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT32)
        return hash_array_inner<uint32_t>(out_hashes, (uint32_t*)array->data1,
                                          n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT64)
        return hash_array_inner<int64_t>(out_hashes, (int64_t*)array->data1,
                                         n_rows, seed);
    if (array->dtype == Bodo_CTypes::DECIMAL)
        return hash_array_inner<decimal_value_cpp>(out_hashes, (decimal_value_cpp*)array->data1,
                                         n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT64)
        return hash_array_inner<uint64_t>(out_hashes, (uint64_t*)array->data1,
                                          n_rows, seed);
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA)
        return hash_array_inner<int64_t>(out_hashes, (int64_t*)array->data1,
                                          n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return hash_array_inner<float>(out_hashes, (float*)array->data1, n_rows,
                                       seed);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return hash_array_inner<double>(out_hashes, (double*)array->data1,
                                        n_rows, seed);
    if (array->arr_type == bodo_array_type::STRING)
        return hash_array_string(out_hashes,
                                 (char*)array->data1, (uint32_t*)array->data2,
                                 (uint8_t*)array->null_bitmask,
                                 n_rows, seed);
    if (array->arr_type == bodo_array_type::LIST_STRING)
        return hash_array_list_string(out_hashes,
                                      (char*)array->data1,(uint32_t*)array->data2,
                                      (uint32_t*)array->data3,(uint8_t*)array->null_bitmask,
                                      n_rows, seed);
    Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid data type for hash");
}


template <class T>
static void hash_array_combine_inner(uint32_t* out_hashes, T* data,
                                     size_t n_rows, const uint32_t seed) {
    // hash combine code from boost
    // https://github.com/boostorg/container_hash/blob/504857692148d52afe7110bcb96cf837b0ced9d7/include/boost/container_hash/hash.hpp#L313
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t out_hash = 0;
        hash_inner_32<T>(&data[i], seed, &out_hash);
        out_hashes[i] ^=
            out_hash + 0x9e3779b9 + (out_hashes[i] << 6) + (out_hashes[i] >> 2);
    }
}


static void hash_array_combine_string(uint32_t* out_hashes, char* data,
                                      uint32_t* offsets, size_t n_rows,
                                      const uint32_t seed) {
    uint32_t start_offset = 0;
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t end_offset = offsets[i + 1];
        uint32_t len = end_offset - start_offset;
        std::string val(&data[start_offset], len);

        uint32_t out_hash = 0;

        const char* val_chars = val.c_str();
        hash_string_32(val_chars, (const int)len, seed, &out_hash);
        out_hashes[i] ^=
            out_hash + 0x9e3779b9 + (out_hashes[i] << 6) + (out_hashes[i] >> 2);
        start_offset = end_offset;
    }
}

static void hash_array_combine(uint32_t* out_hashes, array_info* array,
                               size_t n_rows, const uint32_t seed) {
    // dispatch to proper function
    // TODO: general dispatcher
    if (array->dtype == Bodo_CTypes::_BOOL)
        return hash_array_combine_inner<bool>(out_hashes, (bool*)array->data1,
                                              n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT8)
        return hash_array_combine_inner<int8_t>(
            out_hashes, (int8_t*)array->data1, n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT8)
        return hash_array_combine_inner<uint8_t>(
            out_hashes, (uint8_t*)array->data1, n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT16)
        return hash_array_combine_inner<int16_t>(
            out_hashes, (int16_t*)array->data1, n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT16)
        return hash_array_combine_inner<uint16_t>(
            out_hashes, (uint16_t*)array->data1, n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT32)
        return hash_array_combine_inner<int32_t>(
            out_hashes, (int32_t*)array->data1, n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT32)
        return hash_array_combine_inner<uint32_t>(
            out_hashes, (uint32_t*)array->data1, n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT64)
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT64)
        return hash_array_combine_inner<uint64_t>(
            out_hashes, (uint64_t*)array->data1, n_rows, seed);
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA)
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return hash_array_combine_inner<float>(out_hashes, (float*)array->data1,
                                               n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return hash_array_combine_inner<double>(
            out_hashes, (double*)array->data1, n_rows, seed);
    if (array->arr_type == bodo_array_type::STRING)
        return hash_array_combine_string(out_hashes, (char*)array->data1,
                                         (uint32_t*)array->data2, n_rows, seed);
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "Invalid data type for hash combine");
}

uint32_t* hash_keys(std::vector<array_info*> const& key_arrs,
                    const uint32_t seed) {
    size_t n_rows = (size_t)key_arrs[0]->length;
    uint32_t* hashes = new uint32_t[n_rows];
    // hash first array
    hash_array(hashes, key_arrs[0], n_rows, seed);
    // combine other array hashes
    for (size_t i = 1; i < key_arrs.size(); i++) {
        hash_array_combine(hashes, key_arrs[i], n_rows, seed);
    }
    return hashes;
}
