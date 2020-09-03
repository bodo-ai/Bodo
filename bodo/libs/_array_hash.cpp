// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <arrow/api.h>
#include "_bodo_common.h"
#include "_murmurhash3.h"
#include "_array_utils.h"
#include "_array_hash.h"

#undef DEBUG_HASH

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
 * Computation of the hashes for the case of list of strings array column.
 * Covers LIST_STRING
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
static void combine_hash_array_list_string(uint32_t* out_hashes, char* data,
                                           uint32_t* data_offsets,
                                           uint32_t* index_offsets,
                                           uint8_t* null_bitmask,
                                           uint8_t* sub_null_bitmask,
                                           size_t n_rows) {
    uint32_t start_index_offset = 0;
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t hash1, hash2, hash3;
        // First the hash from the strings.
        uint32_t end_index_offset = index_offsets[i + 1];
        uint32_t len1 =
            data_offsets[end_index_offset] - data_offsets[start_index_offset];
        std::string val(&data[data_offsets[start_index_offset]], len1);
        const char* val_chars1 = val.c_str();
        uint32_t seed = out_hashes[i];
        hash_string_32(val_chars1, (const int)len1, seed, &hash1);
        // Second the hash from the length of strings (approx that most strings
        // have less than 256 characters)
        uint32_t len2 = end_index_offset - start_index_offset;
        // This vectors encodes the length of the strings
        std::vector<char> V(len2);
        // This vector encodes the bitmask of the strings and the bitmask of the list itself
        std::vector<char> V2(len2+1);
        for (size_t j = 0; j < len2; j++) {
            uint32_t n_chars = data_offsets[start_index_offset + j + 1] -
                               data_offsets[start_index_offset + j];
            V[j] = (char)n_chars;
            V2[j+1] = GetBit(sub_null_bitmask, start_index_offset + j);
        }
        hash_string_32(V.data(), (const int)len2, hash1, &hash2);
        // Third the hash from whether it is missing or not
        V2[0] = GetBit(null_bitmask, i);
        hash_string_32(V2.data(), len2+1, hash2, &hash3);
        out_hashes[i] = hash3;
        start_index_offset = end_index_offset;
    }
}




/**
 * Computation of the hashes for the case of list of strings array column.
 * Covers LIST_STRING
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
static void hash_array_list_string(uint32_t* out_hashes, char* data,
                                   uint32_t* data_offsets,
                                   uint32_t* index_offsets,
                                   uint8_t* null_bitmask,
                                   uint8_t* sub_null_bitmask,
                                   size_t n_rows,
                                   const uint32_t seed) {
    uint32_t start_index_offset = 0;
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t hash1, hash2, hash3;
        // First the hash from the strings.
        uint32_t end_index_offset = index_offsets[i + 1];
        uint32_t len1 =
            data_offsets[end_index_offset] - data_offsets[start_index_offset];
        std::string val(&data[data_offsets[start_index_offset]], len1);
        const char* val_chars1 = val.c_str();
        hash_string_32(val_chars1, (const int)len1, seed, &hash1);
        // Second the hash from the length of strings (approx that most strings
        // have less than 256 characters)
        uint32_t len2 = end_index_offset - start_index_offset;
        // This vectors encodes the length of the strings
        std::vector<char> V(len2);
        // This vector encodes the bitmask of the strings and the bitmask of the list itself
        std::vector<char> V2(len2+1);
        for (size_t j = 0; j < len2; j++) {
            uint32_t n_chars = data_offsets[start_index_offset + j + 1] -
                               data_offsets[start_index_offset + j];
            V[j] = (char)n_chars;
            V2[j+1] = GetBit(sub_null_bitmask, start_index_offset + j);
        }
        hash_string_32(V.data(), (const int)len2, hash1, &hash2);
        // Third the hash from whether it is missing or not
        V2[0] = GetBit(null_bitmask, i);
        hash_string_32(V2.data(), len2+1, hash2, &hash3);
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
                              uint32_t* offsets, uint8_t* null_bitmask,
                              size_t n_rows, const uint32_t seed) {
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
 * Computation of the hashes for the offsets
 *
 * @param out_hashes: The hashes on input/output.
 * @param list_offsets: the offsets in the input_array to consider.
 * @param n_rows: the number of rows of the array.
 * @param input_array: the array in input.
 *
 * One approximation is the casting to char of the algorithm.
 */
template <typename T>
void apply_arrow_offset_hash(uint32_t* out_hashes,
                             std::vector<uint32_t> const& list_offsets,
                             size_t n_rows, T const& input_array) {
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        int64_t off1 = input_array->value_offset(list_offsets[i_row]);
        int64_t off2 = input_array->value_offset(list_offsets[i_row + 1]);
        char e_len = (char)(off2 - off1);
        hash_string_32(&e_len, 1, out_hashes[i_row], &out_hashes[i_row]);
    }
}

/**
 * Computation of the hashes for the bitmasks
 *
 * @param out_hashes: The hashes on input/output.
 * @param list_offsets: the offsets in the input_array to consider.
 * @param n_rows: the number of rows of the array.
 * @param input_array: the array in input.
 *
 * The bitmask is encoded as a 8 bit integer. This is of course
 * an approximation if the size is greater than 8 but ok for hashes.
 */
template <typename T>
void apply_arrow_bitmask_hash(uint32_t* out_hashes,
                              std::vector<uint32_t> const& list_offsets,
                              size_t n_rows, T const& input_array) {
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        uint8_t val = 0;
        uint8_t pow = 1;
        for (uint32_t idx = list_offsets[i_row]; idx < list_offsets[i_row + 1];
             idx++) {
            int val_i = (int)input_array->IsNull(idx);
            val += pow * val_i;
            pow *= 2;
        }
        char val_c = (char)val;
        hash_string_32(&val_c, 1, out_hashes[i_row], &out_hashes[i_row]);
    }
}

/**
 * Computation of the hashes for the strings
 *
 * @param out_hashes: The hashes on input/output.
 * @param list_offsets: the offsets in the input_array to consider.
 * @param n_rows: the number of rows of the array.
 * @param input_array: the array in input.
 *
 */
void apply_arrow_string_hashes(
    uint32_t* out_hashes, std::vector<uint32_t> const& list_offsets,
    size_t const& n_rows,
    std::shared_ptr<arrow::StringArray> const& input_array) {
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        for (uint32_t idx = list_offsets[i_row]; idx < list_offsets[i_row + 1];
             idx++) {
            if (input_array->IsNull(idx)) {
                char val_c = 1;
                hash_string_32(&val_c, 1, out_hashes[i_row],
                               &out_hashes[i_row]);
            } else {
                std::string e_str = input_array->GetString(idx);
                hash_string_32(e_str.c_str(), e_str.size(), out_hashes[i_row],
                               &out_hashes[i_row]);
            }
        }
    }
}

/**
 * Computation of the hashes for numerical values.
 *
 * @param out_hashes: The hashes on input/output.
 * @param list_offsets: the offsets in the input_array to consider.
 * @param n_rows: the number of rows of the array.
 * @param values: the list of values in temmplated array.
 * @param input_array: the array in input.
 */
void apply_arrow_numeric_hash(
    uint32_t* out_hashes, std::vector<uint32_t> const& list_offsets,
    size_t const& n_rows,
    std::shared_ptr<arrow::PrimitiveArray> const& primitive_array) {
    arrow::Type::type typ = primitive_array->type()->id();
    Bodo_CTypes::CTypeEnum bodo_typ = arrow_to_bodo_type(typ);
    uint64_t siztype = numpy_item_size[bodo_typ];
    char* value_ptr = (char*)primitive_array->values()->data();
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        for (uint32_t idx = list_offsets[i_row]; idx < list_offsets[i_row + 1];
             idx++) {
            char* value_ptr_shift = value_ptr + siztype * idx;
            hash_string_32(value_ptr_shift, siztype, out_hashes[i_row],
                           &out_hashes[i_row]);
        }
    }
}

/** It is the recursive algorithm for computing the hash.
 * It is done sequentially in order to consider entries one by one.
 * The use of *list_offsets* on input is warranted since when we go deeper
 * with the LIST type, this creates some different list_offsets to use.
 *
 * @param : out_hashes, the hashes on input/output
 * @param : list_offsets, the list of offsets (of length n_rows+1)
 * @param : n_rows, the number of rows in input
 * @param : input_array, the input array put in argument.
 */
void hash_arrow_array(uint32_t* out_hashes,
                      std::vector<uint32_t> const& list_offsets,
                      size_t const& n_rows,
                      std::shared_ptr<arrow::Array> const& input_array) {
#ifdef DEBUG_HASH
    std::cout << "Beginning of hash_arrow_array\n";
#endif
    if (input_array->type_id() == arrow::Type::LIST) {
        auto list_array =
            std::dynamic_pointer_cast<arrow::ListArray>(input_array);
        apply_arrow_offset_hash(out_hashes, list_offsets, n_rows, list_array);
        std::vector<uint32_t> list_offsets_out(n_rows + 1);
        for (size_t i_row = 0; i_row <= n_rows; i_row++)
            list_offsets_out[i_row] =
                list_array->value_offset(list_offsets[i_row]);
        hash_arrow_array(out_hashes, list_offsets_out, n_rows,
                         list_array->values());
        apply_arrow_bitmask_hash(out_hashes, list_offsets, n_rows, input_array);
    } else if (input_array->type_id() == arrow::Type::STRUCT) {
        auto struct_array =
            std::dynamic_pointer_cast<arrow::StructArray>(input_array);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_array->type());
        for (int i_field = 0; i_field < struct_type->num_fields(); i_field++)
            hash_arrow_array(out_hashes, list_offsets, n_rows,
                             struct_array->field(i_field));
        apply_arrow_bitmask_hash(out_hashes, list_offsets, n_rows, input_array);
    } else if (input_array->type_id() == arrow::Type::STRING) {
        auto str_array =
            std::dynamic_pointer_cast<arrow::StringArray>(input_array);
        apply_arrow_offset_hash(out_hashes, list_offsets, n_rows, str_array);
        apply_arrow_string_hashes(out_hashes, list_offsets, n_rows, str_array);
        apply_arrow_bitmask_hash(out_hashes, list_offsets, n_rows, str_array);
    } else {
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        apply_arrow_numeric_hash(out_hashes, list_offsets, n_rows, primitive_array);
        apply_arrow_bitmask_hash(out_hashes, list_offsets, n_rows, primitive_array);
    }
#ifdef DEBUG_HASH
    std::cout << "Ending of hash_arrow_array\n";
#endif
}

/**
 * Top function for the computation of the hashes. It calls all the other hash
 * functions.
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
    if (array->arr_type == bodo_array_type::ARROW) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing arrow column\n";
#endif
        std::vector<uint32_t> list_offsets(n_rows + 1);
        for (uint32_t i = 0; i <= n_rows; i++) list_offsets[i] = i;
        for (uint32_t i = 0; i < n_rows; i++) out_hashes[i] = seed;
        return hash_arrow_array(out_hashes, list_offsets, n_rows, array->array);
    }
    if (array->dtype == Bodo_CTypes::_BOOL) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing bool column\n";
#endif
        return hash_array_inner<bool>(out_hashes, (bool*)array->data1, n_rows,
                                      seed);
    }
    if (array->dtype == Bodo_CTypes::INT8) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing int8 column\n";
#endif
        return hash_array_inner<int8_t>(out_hashes, (int8_t*)array->data1,
                                        n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT8) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing uint8 column\n";
#endif
        return hash_array_inner<uint8_t>(out_hashes, (uint8_t*)array->data1,
                                         n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT16) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing int16 column\n";
#endif
        return hash_array_inner<int16_t>(out_hashes, (int16_t*)array->data1,
                                         n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT16) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing uint16 column\n";
#endif
        return hash_array_inner<uint16_t>(out_hashes, (uint16_t*)array->data1,
                                          n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT32) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing int32 column\n";
#endif
        return hash_array_inner<int32_t>(out_hashes, (int32_t*)array->data1,
                                         n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT32) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing uint32 column\n";
#endif
        return hash_array_inner<uint32_t>(out_hashes, (uint32_t*)array->data1,
                                          n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT64) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing int64 column\n";
#endif
        return hash_array_inner<int64_t>(out_hashes, (int64_t*)array->data1,
                                         n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::DECIMAL) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing decimal column\n";
#endif
        return hash_array_inner<decimal_value_cpp>(
            out_hashes, (decimal_value_cpp*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT64) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing uint64 column\n";
#endif
        return hash_array_inner<uint64_t>(out_hashes, (uint64_t*)array->data1,
                                          n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing date / datetime / timedelta column\n";
#endif
        return hash_array_inner<int64_t>(out_hashes, (int64_t*)array->data1,
                                         n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::FLOAT32) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing float column\n";
#endif
        return hash_array_inner<float>(out_hashes, (float*)array->data1, n_rows,
                                       seed);
    }
    if (array->dtype == Bodo_CTypes::FLOAT64) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing double column\n";
#endif
        return hash_array_inner<double>(out_hashes, (double*)array->data1,
                                        n_rows, seed);
    }
    if (array->arr_type == bodo_array_type::STRING) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing STRING column\n";
#endif
        return hash_array_string(out_hashes, (char*)array->data1,
                                 (uint32_t*)array->data2,
                                 (uint8_t*)array->null_bitmask, n_rows, seed);
    }
    if (array->arr_type == bodo_array_type::LIST_STRING) {
#ifdef DEBUG_HASH
        std::cout << "HASH: Processing LIST_STRING column\n";
#endif
        return hash_array_list_string(
            out_hashes, (char*)array->data1, (uint32_t*)array->data2,
            (uint32_t*)array->data3, (uint8_t*)array->null_bitmask,
            (uint8_t*)array->sub_null_bitmask, n_rows, seed);
    }
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
#ifdef DEBUG_HASH
    std::cout << "Beginning of hash_array_combine\n";
#endif
    if (array->arr_type == bodo_array_type::ARROW) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing arrow column\n";
#endif
        std::vector<uint32_t> list_offsets(n_rows + 1);
        for (uint32_t i = 0; i <= n_rows; i++) list_offsets[i] = i;
        return hash_arrow_array(out_hashes, list_offsets, n_rows, array->array);
    }
    if (array->arr_type == bodo_array_type::STRING) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing string column\n";
#endif
        return hash_array_combine_string(out_hashes, (char*)array->data1,
                                         (uint32_t*)array->data2, n_rows, seed);
    }
    if (array->arr_type == bodo_array_type::LIST_STRING) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing list(string) column\n";
#endif
        return combine_hash_array_list_string(out_hashes, (char*)array->data1,
            (uint32_t*)array->data2, (uint32_t*)array->data3,
            (uint8_t*)array->null_bitmask,(uint8_t*)array->sub_null_bitmask, n_rows);
    }
    if (array->dtype == Bodo_CTypes::_BOOL) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing bool column\n";
#endif
        return hash_array_combine_inner<bool>(out_hashes, (bool*)array->data1,
                                              n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT8) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing int8 column\n";
#endif
        return hash_array_combine_inner<int8_t>(
            out_hashes, (int8_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT8) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing uint8 column\n";
#endif
        return hash_array_combine_inner<uint8_t>(
            out_hashes, (uint8_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT16) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing int16 column\n";
#endif
        return hash_array_combine_inner<int16_t>(
            out_hashes, (int16_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT16) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing uint16 column\n";
#endif
        return hash_array_combine_inner<uint16_t>(
            out_hashes, (uint16_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT32) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing int32 column\n";
#endif
        return hash_array_combine_inner<int32_t>(
            out_hashes, (int32_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT32) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing uint32 column\n";
#endif
        return hash_array_combine_inner<uint32_t>(
            out_hashes, (uint32_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT64) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing int64 column\n";
#endif
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT64) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing uint64 column\n";
#endif
        return hash_array_combine_inner<uint64_t>(
            out_hashes, (uint64_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing date / datetime / timedelta column\n";
#endif
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::FLOAT32) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing float column\n";
#endif
        return hash_array_combine_inner<float>(out_hashes, (float*)array->data1,
                                               n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::FLOAT64) {
#ifdef DEBUG_HASH
        std::cout << "Combine HASH: Processing double column\n";
#endif
        return hash_array_combine_inner<double>(
            out_hashes, (double*)array->data1, n_rows, seed);
    }
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "Invalid data type for hash combine");
}

uint32_t* hash_keys(std::vector<array_info*> const& key_arrs,
                    const uint32_t seed) {
#ifdef DEBUG_HASH
    std::cout << "Beginning of hash_keys. key_arrs=\n";
    DEBUG_PrintSetOfColumn(std::cout, key_arrs);
    DEBUG_PrintRefct(std::cout, key_arrs);
#endif
    size_t n_rows = (size_t)key_arrs[0]->length;
#ifdef DEBUG_HASH
    std::cout << "n_rows=" << n_rows << "\n";
#endif
    uint32_t* hashes = new uint32_t[n_rows];
    // hash first array
#ifdef DEBUG_HASH
    std::cout << "Before hash_array\n";
#endif
    hash_array(hashes, key_arrs[0], n_rows, seed);
#ifdef DEBUG_HASH
    std::cout << "After hash_array\n";
#endif
    // combine other array hashes
    for (size_t i = 1; i < key_arrs.size(); i++) {
        hash_array_combine(hashes, key_arrs[i], n_rows, seed);
    }
#ifdef DEBUG_HASH
    for (size_t i_row=0; i_row<n_rows; i_row++)
      std::cout << "hash_keys : i_row=" << i_row << " hash=" << hashes[i_row] << "\n";
    std::cout << "Ending of hash_keys\n";
#endif
    return hashes;
}
