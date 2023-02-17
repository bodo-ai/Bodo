// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_hash.h"
#include <Python.h>
#include <arrow/api.h>
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_murmurhash3.h"

/**
 * Computation of the NA value hash
 * @param seed: the seed of the computation.
 * @param[out] hash_value: The hashes on output.
 * @param use_murmurhash: Use the murmurhash hash algorithm
 * (currently only used for Iceberg bucket transformation)
 * TODO: [BE-975] Use this to trigger with hash_array_inner.
 */
static void hash_na_val(const uint32_t seed, uint32_t* hash_value,
                        bool use_murmurhash = false) {
    int64_t val = 1;
    if (use_murmurhash)
        hash_inner_murmurhash3_x86_32<int64_t>(&val, seed, hash_value);
    else
        hash_inner_32<int64_t>(&val, seed, hash_value);
}
/**
 * Computation of the inner hash of the functions. This covers the NUMPY case.
 *
 * @param out_hashes: The hashes on output.
 * @param data: the list of data in input.
 * @param n_rows: the number of rows of the table.
 * @param seed: the seed of the computation.
 * @param null_bitmask: the null_bitmask of the data.
 * @param use_murmurhash: Use the murmurhash hash algorithm
 * (currently only used for Iceberg bucket transformation)
 *
 */
template <typename T>
static typename std::enable_if<!std::is_floating_point<T>::value, void>::type
hash_array_inner(uint32_t* out_hashes, T* data, size_t n_rows,
                 const uint32_t seed, uint8_t* null_bitmask,
                 bool use_murmurhash = false) {
    if (null_bitmask) {
        uint32_t na_hash;
        hash_na_val(seed, &na_hash, use_murmurhash);
        for (size_t i = 0; i < n_rows; i++) {
            if (use_murmurhash)
                hash_inner_murmurhash3_x86_32<T>(&data[i], seed,
                                                 &out_hashes[i]);
            else
                hash_inner_32<T>(&data[i], seed, &out_hashes[i]);
            if (!GetBit(null_bitmask, i)) out_hashes[i] = na_hash;
        }
    } else {
        for (size_t i = 0; i < n_rows; i++)
            if (use_murmurhash)
                hash_inner_murmurhash3_x86_32<T>(&data[i], seed,
                                                 &out_hashes[i]);
            else
                hash_inner_32<T>(&data[i], seed, &out_hashes[i]);
    }
}

/*
 * Copied largely from Numpy
 * https://github.com/numpy/numpy/blob/548bc6826b597ab79b9c1451b79ec8d23db9d444/numpy/core/src/common/npy_pycompat.h#L7
 *
 * In Python 3.10a7 (or b1), python started using the identity for the hash
 * when a value is NaN.  See https://bugs.python.org/issue43475
 */
#if PY_VERSION_HEX > 0x030a00a6
#define Npy_HashDouble _Py_HashDouble
#else
static inline Py_hash_t Npy_HashDouble(PyObject* __UNUSED__(identity),
                                       double val) {
    return _Py_HashDouble(val);
}
#endif

// Discussion on hashing floats:
// https://stackoverflow.com/questions/4238122/hash-function-for-floats

template <typename T>
static typename std::enable_if<std::is_floating_point<T>::value, void>::type
hash_array_inner(uint32_t* out_hashes, T* data, size_t n_rows,
                 const uint32_t seed, bool use_murmurhash = false) {
    for (size_t i = 0; i < n_rows; i++) {
        Py_hash_t py_hash = Npy_HashDouble(nullptr, data[i]);
        if (use_murmurhash)
            hash_inner_murmurhash3_x86_32<Py_hash_t>(&py_hash, seed,
                                                     &out_hashes[i]);
        else
            hash_inner_32<Py_hash_t>(&py_hash, seed, &out_hashes[i]);
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
                                           offset_t* data_offsets,
                                           offset_t* index_offsets,
                                           uint8_t* null_bitmask,
                                           uint8_t* sub_null_bitmask,
                                           size_t n_rows) {
    offset_t start_index_offset = 0;
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t hash1, hash2, hash3;
        // First the hash from the strings.
        offset_t end_index_offset = index_offsets[i + 1];
        offset_t len1 =
            data_offsets[end_index_offset] - data_offsets[start_index_offset];
        const char* val_chars1 = &data[data_offsets[start_index_offset]];
        // Use existing hash as seed for next hashing step, hence
        // "combining" the hashes
        uint32_t seed = out_hashes[i];
        hash_string_32(val_chars1, (const int)len1, seed, &hash1);
        // Second the hash from the length of strings (approx that most strings
        // have less than 256 characters)
        offset_t len2 = end_index_offset - start_index_offset;
        // This vectors encodes the length of the strings
        std::vector<char> V(len2);
        // This vector encodes the bitmask of the strings and the bitmask of the
        // list itself
        std::vector<char> V2(len2 + 1);
        for (size_t j = 0; j < len2; j++) {
            offset_t n_chars = data_offsets[start_index_offset + j + 1] -
                               data_offsets[start_index_offset + j];
            V[j] = (char)n_chars;
            V2[j + 1] = GetBit(sub_null_bitmask, start_index_offset + j);
        }
        hash_string_32(V.data(), (const int)len2, hash1, &hash2);
        // Third the hash from whether it is missing or not
        V2[0] = GetBit(null_bitmask, i);
        hash_string_32(V2.data(), len2 + 1, hash2, &hash3);
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
 * @param use_murmurhash: use murmurhash3_x86_32 hashes (used by Iceberg).
 * Default: false
 *
 * The hash is computed in 3 stages:
 * 1) The hash of the concatenated strings
 * 2) The hash of the string length
 * 3) The hash of the bitmask
 */
static void hash_array_list_string(
    uint32_t* out_hashes, char* data, offset_t* data_offsets,
    offset_t* index_offsets, uint8_t* null_bitmask, uint8_t* sub_null_bitmask,
    size_t n_rows, const uint32_t seed, bool use_murmurhash = false) {
    offset_t start_index_offset = 0;
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t hash1, hash2, hash3;
        // First the hash from the strings.
        offset_t end_index_offset = index_offsets[i + 1];
        offset_t len1 =
            data_offsets[end_index_offset] - data_offsets[start_index_offset];
        const char* val_chars1 = &data[data_offsets[start_index_offset]];
        if (use_murmurhash)
            hash_string_murmurhash3_x86_32(val_chars1, (const int)len1, seed,
                                           &hash1);
        else
            hash_string_32(val_chars1, (const int)len1, seed, &hash1);
        // Second the hash from the length of strings (approx that most strings
        // have less than 256 characters)
        offset_t len2 = end_index_offset - start_index_offset;
        // This vectors encodes the length of the strings
        std::vector<char> V(len2);
        // This vector encodes the bitmask of the strings and the bitmask of the
        // list itself
        std::vector<char> V2(len2 + 1);
        for (size_t j = 0; j < len2; j++) {
            offset_t n_chars = data_offsets[start_index_offset + j + 1] -
                               data_offsets[start_index_offset + j];
            V[j] = (char)n_chars;
            V2[j + 1] = GetBit(sub_null_bitmask, start_index_offset + j);
        }
        if (use_murmurhash)
            hash_string_murmurhash3_x86_32(V.data(), (const int)len2, hash1,
                                           &hash2);
        else
            hash_string_32(V.data(), (const int)len2, hash1, &hash2);
        // Third the hash from whether it is missing or not
        V2[0] = GetBit(null_bitmask, i);
        if (use_murmurhash)
            hash_string_murmurhash3_x86_32(V2.data(), len2 + 1, hash2, &hash3);
        else
            hash_string_32(V2.data(), len2 + 1, hash2, &hash3);
        out_hashes[i] = hash3;
        start_index_offset = end_index_offset;
    }
}

/**
 * Computation of the NA string hash
 * @param seed: the seed of the computation.
 * @param[out] hash_value: The hashes on output.
 * @param use_murmurhash: Use the murmurhash hash algorithm
 * (currently only used for Iceberg bucket transformation)
 */
static void hash_na_string(const uint32_t seed, uint32_t* hash_value,
                           bool use_murmurhash = false) {
    char val_c = 1;
    if (use_murmurhash)
        hash_string_murmurhash3_x86_32(&val_c, 1, seed, hash_value);
    else
        hash_string_32(&val_c, 1, seed, hash_value);
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
 * @param is_parallel: whether we run in parallel or not.
 * @param use_murmurhash: use murmurhash3_x86_32 hashes (used by Iceberg).
 * Default: false
 *
 * Right now, the bitmask is not used in the computation, which
 * may be a problem to consider later on.
 */
static void hash_array_string(uint32_t* out_hashes, char* data,
                              offset_t* offsets, uint8_t* null_bitmask,
                              size_t n_rows, const uint32_t seed,
                              bool is_parallel, bool use_murmurhash = false) {
    tracing::Event ev("hash_array_string", is_parallel);
    offset_t start_offset = 0;
    uint32_t na_hash;
    hash_na_string(seed, &na_hash, use_murmurhash);
    for (size_t i = 0; i < n_rows; i++) {
        offset_t end_offset = offsets[i + 1];
        offset_t len = end_offset - start_offset;
        // val is null
        if (is_na(null_bitmask, i)) {
            out_hashes[i] = na_hash;
        } else {
            const char* val_chars = &data[start_offset];
            if (use_murmurhash)
                hash_string_murmurhash3_x86_32(val_chars, (const int)len, seed,
                                               &out_hashes[i]);
            else
                hash_string_32(val_chars, (const int)len, seed, &out_hashes[i]);
        }
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
 * @param use_murmurhash: use murmurhash3_x86_32 hashes (used by Iceberg).
 * Default: false
 *
 * One approximation is the casting to char of the algorithm.
 */
template <typename T>
void apply_arrow_offset_hash(uint32_t* out_hashes,
                             std::vector<offset_t> const& list_offsets,
                             size_t n_rows, T const& input_array,
                             bool use_murmurhash = false) {
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        int64_t off1 = input_array->value_offset(list_offsets[i_row]);
        int64_t off2 = input_array->value_offset(list_offsets[i_row + 1]);
        char e_len = (char)(off2 - off1);
        if (use_murmurhash)
            hash_string_murmurhash3_x86_32(&e_len, 1, out_hashes[i_row],
                                           &out_hashes[i_row]);
        else
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
                              std::vector<offset_t> const& list_offsets,
                              size_t n_rows, T const& input_array) {
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        uint8_t val = 0;
        uint8_t pow = 1;
        for (offset_t idx = list_offsets[i_row]; idx < list_offsets[i_row + 1];
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
    uint32_t* out_hashes, std::vector<offset_t> const& list_offsets,
    size_t const& n_rows,
#if OFFSET_BITWIDTH == 32
    std::shared_ptr<arrow::StringArray> const& input_array) {
#else
    std::shared_ptr<arrow::LargeStringArray> const& input_array) {
#endif
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        for (offset_t idx = list_offsets[i_row]; idx < list_offsets[i_row + 1];
             idx++) {
            if (input_array->IsNull(idx)) {
                char val_c = 1;
                hash_string_32(&val_c, 1, out_hashes[i_row],
                               &out_hashes[i_row]);
            } else {
                std::string_view e_str = ArrowStrArrGetView(input_array, idx);
                hash_string_32(e_str.data(), e_str.size(), out_hashes[i_row],
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
    uint32_t* out_hashes, std::vector<offset_t> const& list_offsets,
    size_t const& n_rows,
    std::shared_ptr<arrow::PrimitiveArray> const& primitive_array) {
    Bodo_CTypes::CTypeEnum bodo_typ =
        arrow_to_bodo_type(primitive_array->type());
    uint64_t siztype = numpy_item_size[bodo_typ];
    char* value_ptr = (char*)primitive_array->values()->data();
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        for (offset_t idx = list_offsets[i_row]; idx < list_offsets[i_row + 1];
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
 * @param out_hashes: the hashes on input/output
 * @param list_offsets: the list of offsets (of length n_rows+1)
 * @param n_rows: the number of rows in input
 * @param input_array: the input array put in argument.
 */
void hash_arrow_array(uint32_t* out_hashes,
                      std::vector<offset_t> const& list_offsets,
                      size_t const& n_rows,
                      std::shared_ptr<arrow::Array> const& input_array) {
#if OFFSET_BITWIDTH == 32
    if (input_array->type_id() == arrow::Type::LIST) {
        auto list_array =
            std::dynamic_pointer_cast<arrow::ListArray>(input_array);
#else
    if (input_array->type_id() == arrow::Type::LARGE_LIST) {
        auto list_array =
            std::dynamic_pointer_cast<arrow::LargeListArray>(input_array);
#endif
        apply_arrow_offset_hash(out_hashes, list_offsets, n_rows, list_array);
        std::vector<offset_t> list_offsets_out(n_rows + 1);
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
#if OFFSET_BITWIDTH == 32
    } else if (input_array->type_id() == arrow::Type::STRING) {
        auto str_array =
            std::dynamic_pointer_cast<arrow::StringArray>(input_array);
#else
    } else if (input_array->type_id() == arrow::Type::LARGE_STRING) {
        auto str_array =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(input_array);
#endif
        apply_arrow_offset_hash(out_hashes, list_offsets, n_rows, str_array);
        apply_arrow_string_hashes(out_hashes, list_offsets, n_rows, str_array);
        apply_arrow_bitmask_hash(out_hashes, list_offsets, n_rows, str_array);
    } else {
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        apply_arrow_numeric_hash(out_hashes, list_offsets, n_rows,
                                 primitive_array);
        apply_arrow_bitmask_hash(out_hashes, list_offsets, n_rows,
                                 primitive_array);
    }
}

/**
 * Top function for the computation of the hashes. It calls all the other hash
 * functions.
 *
 * @param[out] out_hashes: The hashes on output.
 * @param[in] array: the list of columns in input
 * @param n_rows: the number of rows of the table.
 * @param seed: the seed of the computation.
 * @param is_parallel: whether we run in parallel or not.
 * @param global_dict_needed: this only applies to hashing of dictionary-encoded
 * arrays. This parameter specifies whether the dictionary has to be global
 * or not (for correctness or for performance -for example avoiding collisions
 * after shuffling-). This is context-dependent.
 * @param use_murmurhash: use murmurhash3_x86_32 hashes (used by Iceberg).
 * Default: false
 */
void hash_array(uint32_t* out_hashes, array_info* array, size_t n_rows,
                const uint32_t seed, bool is_parallel, bool global_dict_needed,
                bool use_murmurhash) {
    // dispatch to proper function
    // TODO: general dispatcher
    // XXX: assumes nullable array data for nulls is always consistent
    if (array->arr_type == bodo_array_type::ARROW) {
        std::vector<offset_t> list_offsets(n_rows + 1);
        for (offset_t i = 0; i <= n_rows; i++) list_offsets[i] = i;
        for (offset_t i = 0; i < n_rows; i++) out_hashes[i] = seed;
        if (use_murmurhash)
            throw std::runtime_error(
                "_array_hash::hash_array: MurmurHash not supported for Arrow "
                "arrays.");
        return hash_arrow_array(out_hashes, list_offsets, n_rows, array->array);
    }
    if (array->arr_type == bodo_array_type::STRING) {
        return hash_array_string(out_hashes, (char*)array->data1,
                                 (offset_t*)array->data2,
                                 (uint8_t*)array->null_bitmask, n_rows, seed,
                                 is_parallel, use_murmurhash);
    }
    if (array->arr_type == bodo_array_type::DICT) {
        if ((array->has_global_dictionary &&
             array->has_deduped_local_dictionary) ||
            !is_parallel || !global_dict_needed) {
            // in this case we can just hash the indices since the dictionary is
            // synchronized across ranks or is only needed for a local
            // operation where hashing based on local dictionary won't affect
            // correctness or performance
            return hash_array_inner<dict_indices_t>(
                out_hashes, (dict_indices_t*)array->info2->data1, n_rows, seed,
                (uint8_t*)array->info2->null_bitmask, use_murmurhash);
        } else {
            // 3 options:
            // - Convert to global dictionary now
            // - Require the conversion to have happened before calling this
            // function
            // - Access the strings to get globally consistent hashes (this is
            // not efficient
            //   if we are going to end up converting to global dictionary as
            //   part of the operation that called hash_array())
            throw std::runtime_error(
                "hashing dictionary array requires global dictionary "
                "with unique values in this context");
        }
    }
    if (array->arr_type == bodo_array_type::LIST_STRING) {
        return hash_array_list_string(
            out_hashes, (char*)array->data1, (offset_t*)array->data2,
            (offset_t*)array->data3, (uint8_t*)array->null_bitmask,
            (uint8_t*)array->sub_null_bitmask, n_rows, seed, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array_inner<bool>(out_hashes, (bool*)array->data1, n_rows,
                                      seed, (uint8_t*)array->null_bitmask,
                                      use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::INT8) {
        return hash_array_inner<int8_t>(
            out_hashes, (int8_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::UINT8) {
        return hash_array_inner<uint8_t>(
            out_hashes, (uint8_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::INT16) {
        return hash_array_inner<int16_t>(
            out_hashes, (int16_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::UINT16) {
        return hash_array_inner<uint16_t>(
            out_hashes, (uint16_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::INT32) {
        return hash_array_inner<int32_t>(
            out_hashes, (int32_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::UINT32) {
        return hash_array_inner<uint32_t>(
            out_hashes, (uint32_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::INT64) {
        return hash_array_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::DECIMAL) {
        return hash_array_inner<decimal_value_cpp>(
            out_hashes, (decimal_value_cpp*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::UINT64) {
        return hash_array_inner<uint64_t>(
            out_hashes, (uint64_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    // TODO: [BE-4106] Split Time into Time32 and Time64
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA) {
        return hash_array_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::FLOAT32) {
        return hash_array_inner<float>(out_hashes, (float*)array->data1, n_rows,
                                       seed, use_murmurhash);
    }
    if (array->dtype == Bodo_CTypes::FLOAT64) {
        return hash_array_inner<double>(out_hashes, (double*)array->data1,
                                        n_rows, seed, use_murmurhash);
    }
    Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid data type for hash");
}

// ------- boost hash combine function for 32-bit hashes -------

// https://github.com/boostorg/container_hash/blob/504857692148d52afe7110bcb96cf837b0ced9d7/include/boost/container_hash/hash.hpp#L60
#if defined(_MSC_VER)
#define BOOST_FUNCTIONAL_HASH_ROTL32(x, r) _rotl(x, r)
#else
#define BOOST_FUNCTIONAL_HASH_ROTL32(x, r) (x << r) | (x >> (32 - r))
#endif

// https://github.com/boostorg/container_hash/blob/504857692148d52afe7110bcb96cf837b0ced9d7/include/boost/container_hash/hash.hpp#L316
static inline void hash_combine_boost(uint32_t& h1, uint32_t k1) {
    // This is a single 32-bit murmur iteration.
    // See this comment and its discussion for more information:
    // https://stackoverflow.com/a/50978188

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    k1 *= c1;
    k1 = BOOST_FUNCTIONAL_HASH_ROTL32(k1, 15);
    k1 *= c2;

    h1 ^= k1;
    h1 = BOOST_FUNCTIONAL_HASH_ROTL32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
}

// -------------------------------------------------------------

template <class T>
static typename std::enable_if<!std::is_floating_point<T>::value, void>::type
hash_array_combine_inner(uint32_t* out_hashes, T* data, size_t n_rows,
                         const uint32_t seed, uint8_t* null_bitmask) {
    if (null_bitmask) {
        uint32_t na_hash;
        hash_na_val(seed, &na_hash);
        uint32_t out_hash = 0;
        for (size_t i = 0; i < n_rows; i++) {
            if (!GetBit(null_bitmask, i))
                out_hash = na_hash;
            else
                hash_inner_32<T>(&data[i], seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    } else {
        uint32_t out_hash = 0;
        for (size_t i = 0; i < n_rows; i++) {
            hash_inner_32<T>(&data[i], seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    }
}

// Discussion on hashing floats:
// https://stackoverflow.com/questions/4238122/hash-function-for-floats

template <class T>
static typename std::enable_if<std::is_floating_point<T>::value, void>::type
hash_array_combine_inner(uint32_t* out_hashes, T* data, size_t n_rows,
                         const uint32_t seed) {
    uint32_t out_hash = 0;
    for (size_t i = 0; i < n_rows; i++) {
        Py_hash_t py_hash = Npy_HashDouble(nullptr, data[i]);
        hash_inner_32<Py_hash_t>(&py_hash, seed, &out_hash);
        hash_combine_boost(out_hashes[i], out_hash);
    }
}

static void hash_array_combine_string(uint32_t* out_hashes, char* data,
                                      offset_t* offsets, uint8_t* null_bitmask,
                                      size_t n_rows, const uint32_t seed) {
    offset_t start_offset = 0;
    uint32_t na_hash;
    hash_na_string(seed, &na_hash);
    for (size_t i = 0; i < n_rows; i++) {
        offset_t end_offset = offsets[i + 1];
        offset_t len = end_offset - start_offset;

        uint32_t out_hash = 0;
        if (is_na(null_bitmask, i)) {
            out_hash = na_hash;
        } else {
            const char* val_chars = &data[start_offset];
            hash_string_32(val_chars, (const int)len, seed, &out_hash);
        }
        hash_combine_boost(out_hashes[i], out_hash);
        start_offset = end_offset;
    }
}

// See hash_array for documentation of parameters
void hash_array_combine(uint32_t* out_hashes, array_info* array, size_t n_rows,
                        const uint32_t seed, bool global_dict_needed,
                        bool is_parallel) {
    // dispatch to proper function
    // TODO: general dispatcher
    if (array->arr_type == bodo_array_type::ARROW) {
        std::vector<offset_t> list_offsets(n_rows + 1);
        for (offset_t i = 0; i <= n_rows; i++) list_offsets[i] = i;
        return hash_arrow_array(out_hashes, list_offsets, n_rows, array->array);
    }
    if (array->arr_type == bodo_array_type::STRING) {
        return hash_array_combine_string(
            out_hashes, (char*)array->data1, (offset_t*)array->data2,
            (uint8_t*)array->null_bitmask, n_rows, seed);
    }
    if (array->arr_type == bodo_array_type::DICT) {
        if ((array->has_global_dictionary &&
             array->has_deduped_local_dictionary) ||
            !global_dict_needed || !is_parallel) {
            // in this case we can just hash the indices since the dictionary is
            // synchronized across ranks or is only needed for a local
            // operation where hashing based on local dictionary won't affect
            // correctness or performance
            return hash_array_combine_inner<dict_indices_t>(
                out_hashes, (dict_indices_t*)array->info2->data1, n_rows, seed,
                (uint8_t*)array->null_bitmask);
        } else {
            // 3 options:
            // - Convert to global dictionary now
            // - Require the conversion to have happened before calling this
            // function
            // - Access the strings to get globally consistent hashes (this is
            // not efficient
            //   if we are going to end up converting to global dictionary as
            //   part of the operation that called hash_array())
            throw std::runtime_error(
                "hashing dictionary array requires global dictionary "
                "with unique values in this context");
        }
    }
    if (array->arr_type == bodo_array_type::LIST_STRING) {
        return combine_hash_array_list_string(
            out_hashes, (char*)array->data1, (offset_t*)array->data2,
            (offset_t*)array->data3, (uint8_t*)array->null_bitmask,
            (uint8_t*)array->sub_null_bitmask, n_rows);
    }
    if (array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array_combine_inner<bool>(out_hashes, (bool*)array->data1,
                                              n_rows, seed,
                                              (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::INT8) {
        return hash_array_combine_inner<int8_t>(
            out_hashes, (int8_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::UINT8) {
        return hash_array_combine_inner<uint8_t>(
            out_hashes, (uint8_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::INT16) {
        return hash_array_combine_inner<int16_t>(
            out_hashes, (int16_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::UINT16) {
        return hash_array_combine_inner<uint16_t>(
            out_hashes, (uint16_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::INT32) {
        return hash_array_combine_inner<int32_t>(
            out_hashes, (int32_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::UINT32) {
        return hash_array_combine_inner<uint32_t>(
            out_hashes, (uint32_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::INT64) {
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::UINT64) {
        return hash_array_combine_inner<uint64_t>(
            out_hashes, (uint64_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    // TODO: [BE-4106] Split Time into Time32 and Time64
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA) {
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::FLOAT32) {
        return hash_array_combine_inner<float>(out_hashes, (float*)array->data1,
                                               n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::FLOAT64) {
        return hash_array_combine_inner<double>(
            out_hashes, (double*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::DECIMAL ||
        array->dtype == Bodo_CTypes::INT128) {
        return hash_array_combine_inner<decimal_value_cpp>(
            out_hashes, (decimal_value_cpp*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "Invalid data type for hash combine");
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, double>::type
get_value(T val) {
    if (isnan(val))  // I wrote that because I am not sure nan have unique
                     // binary representation
        return std::nan("");
    return val;
}

template <typename T>
typename std::enable_if<!std::is_floating_point<T>::value, double>::type
get_value(T val) {
    return val;
}

template <typename T>
void coherent_hash_array_inner_uint64(uint32_t* out_hashes, array_info* array,
                                      size_t n_rows, const uint32_t seed) {
    T* data = (T*)array->data1;
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            uint64_t val = data[i];
            hash_inner_32<uint64_t>(&val, seed, &out_hashes[i]);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask;
        uint32_t na_hash;
        hash_na_val(seed, &na_hash);
        for (size_t i = 0; i < n_rows; i++) {
            uint64_t val = data[i];
            hash_inner_32<uint64_t>(&val, seed, &out_hashes[i]);
            if (!GetBit(null_bitmask, i)) out_hashes[i] = na_hash;
        }
    }
}

template <typename T>
void coherent_hash_array_inner_int64(uint32_t* out_hashes, array_info* array,
                                     size_t n_rows, const uint32_t seed) {
    T* data = (T*)array->data1;
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            int64_t val = data[i];
            hash_inner_32<int64_t>(&val, seed, &out_hashes[i]);
            // For numpy, all entries are true, no need to increment.
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask;
        uint32_t na_hash;
        hash_na_val(seed, &na_hash);
        for (size_t i = 0; i < n_rows; i++) {
            int64_t val = data[i];
            hash_inner_32<int64_t>(&val, seed, &out_hashes[i]);
            if (!GetBit(null_bitmask, i)) out_hashes[i] = na_hash;
        }
    }
}

template <typename T>
void coherent_hash_array_inner_double(uint32_t* out_hashes, array_info* array,
                                      size_t n_rows, const uint32_t seed) {
    T* data = (T*)array->data1;
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            double val = get_value(data[i]);
            hash_inner_32<double>(&val, seed, &out_hashes[i]);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask;
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(null_bitmask, i);
            double val;
            if (bit)
                val = get_value(data[i]);
            else
                val = std::nan("");
            hash_inner_32<double>(&val, seed, &out_hashes[i]);
        }
    }
}

void coherent_hash_array(uint32_t* out_hashes, array_info* array,
                         array_info* ref_array, size_t n_rows,
                         const uint32_t seed, bool is_parallel = true) {
    if ((array->arr_type == bodo_array_type::DICT) &&
        (array->info1 != ref_array->info1)) {
        // This implementation of coherent_hash_array hashes data based on
        // the values in the indices array. To do this, we make and enforce
        // a few assumptions
        //
        // 1. Both arrays are dictionary encoded. This is enforced in join.py
        // where determine_table_cast_map requires either both inputs to be
        // dictionary encoded or neither.
        //
        // 2. Both arrays share the exact same dictionary. This occurs in
        // unify_dictionaries and is checked above.
        //
        // 3. The dictionary does not contain any duplicate values. This is
        // enforced by the has_deduped_local_dictionary check in
        // unify_dictionaries and is updated by
        // make_dictionary_global_and_unique. In particular,
        // make_dictionary_global_and_unique contains a drop duplicates step
        // that ensures all values are unique. If the dictionary is modified
        // by some other means (e.g. Python), then we assume that it also
        // updates the flags appropriately.
        throw std::runtime_error(
            "coherent_hash_array: don't know if arrays have unified "
            "dictionary");
    }

    // For those types, no type conversion is ever needed.
    if (array->arr_type == bodo_array_type::ARROW ||
        array->arr_type == bodo_array_type::STRING ||
        array->arr_type == bodo_array_type::LIST_STRING) {
        return hash_array(out_hashes, array, n_rows, seed, is_parallel, true);
    }
    // Now we are in NUMPY / NULLABLE_INT_BOOL. Getting into hot waters.
    // For DATE / TIME / DATETIME / TIMEDELTA / DECIMAL no type conversion is
    // allowed
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::TIME ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA ||
        array->dtype == Bodo_CTypes::DECIMAL ||
        array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array(out_hashes, array, n_rows, seed, is_parallel, true);
    }
    // If we have the same type on left or right then no need
    if (array->arr_type == ref_array->arr_type ||
        array->dtype == ref_array->dtype) {
        return hash_array(out_hashes, array, n_rows, seed, is_parallel, true);
    }
    // If both are unsigned int, we convert to uint64_t
    if (is_unsigned_integer(array->dtype) &&
        is_unsigned_integer(ref_array->dtype)) {
        if (array->dtype == Bodo_CTypes::UINT8)
            return coherent_hash_array_inner_uint64<uint8_t>(out_hashes, array,
                                                             n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT16)
            return coherent_hash_array_inner_uint64<uint16_t>(out_hashes, array,
                                                              n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT32)
            return coherent_hash_array_inner_uint64<uint32_t>(out_hashes, array,
                                                              n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT64)
            return coherent_hash_array_inner_uint64<uint64_t>(out_hashes, array,
                                                              n_rows, seed);
    }
    // If both are integer (signed or unsigned), we convert to int64_t
    if (is_integer(array->dtype) && is_integer(ref_array->dtype)) {
        if (array->dtype == Bodo_CTypes::UINT8)
            return coherent_hash_array_inner_int64<uint8_t>(out_hashes, array,
                                                            n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT16)
            return coherent_hash_array_inner_int64<uint16_t>(out_hashes, array,
                                                             n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT32)
            return coherent_hash_array_inner_int64<uint32_t>(out_hashes, array,
                                                             n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT64)
            return coherent_hash_array_inner_int64<uint64_t>(out_hashes, array,
                                                             n_rows, seed);
        if (array->dtype == Bodo_CTypes::INT8)
            return coherent_hash_array_inner_int64<int8_t>(out_hashes, array,
                                                           n_rows, seed);
        if (array->dtype == Bodo_CTypes::INT16)
            return coherent_hash_array_inner_int64<int16_t>(out_hashes, array,
                                                            n_rows, seed);
        if (array->dtype == Bodo_CTypes::INT32)
            return coherent_hash_array_inner_int64<int32_t>(out_hashes, array,
                                                            n_rows, seed);
        if (array->dtype == Bodo_CTypes::INT64)
            return coherent_hash_array_inner_int64<int64_t>(out_hashes, array,
                                                            n_rows, seed);
    }
    // In all other cases, we convert to double
    if (array->dtype == Bodo_CTypes::UINT8)
        return coherent_hash_array_inner_double<uint8_t>(out_hashes, array,
                                                         n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT16)
        return coherent_hash_array_inner_double<uint16_t>(out_hashes, array,
                                                          n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT32)
        return coherent_hash_array_inner_double<uint32_t>(out_hashes, array,
                                                          n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT64)
        return coherent_hash_array_inner_double<uint64_t>(out_hashes, array,
                                                          n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT8)
        return coherent_hash_array_inner_double<int8_t>(out_hashes, array,
                                                        n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT16)
        return coherent_hash_array_inner_double<int16_t>(out_hashes, array,
                                                         n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT32)
        return coherent_hash_array_inner_double<int32_t>(out_hashes, array,
                                                         n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT64)
        return coherent_hash_array_inner_double<int64_t>(out_hashes, array,
                                                         n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return coherent_hash_array_inner_double<float>(out_hashes, array,
                                                       n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return coherent_hash_array_inner_double<double>(out_hashes, array,
                                                        n_rows, seed);
}

template <typename T>
void coherent_hash_array_combine_inner_uint64(uint32_t* out_hashes,
                                              array_info* array, size_t n_rows,
                                              const uint32_t seed) {
    T* data = (T*)array->data1;
    uint32_t out_hash;
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            uint64_t val = data[i];
            hash_inner_32<uint64_t>(&val, seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask;
        for (size_t i = 0; i < n_rows; i++) {
            uint64_t val = data[i];
            hash_inner_32<uint64_t>(&val, seed, &out_hash);
            if (!GetBit(null_bitmask, i)) out_hash++;
            hash_combine_boost(out_hashes[i], out_hash);
        }
    }
}

template <typename T>
void coherent_hash_array_combine_inner_int64(uint32_t* out_hashes,
                                             array_info* array, size_t n_rows,
                                             const uint32_t seed) {
    T* data = (T*)array->data1;
    uint32_t out_hash;
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            int64_t val = data[i];
            hash_inner_32<int64_t>(&val, seed, &out_hash);
            // For numpy, all entries are true, no need to increment.
            hash_combine_boost(out_hashes[i], out_hash);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask;
        for (size_t i = 0; i < n_rows; i++) {
            int64_t val = data[i];
            hash_inner_32<int64_t>(&val, seed, &out_hash);
            if (!GetBit(null_bitmask, i)) out_hash++;
            hash_combine_boost(out_hashes[i], out_hash);
        }
    }
}

template <typename T>
void coherent_hash_array_combine_inner_double(uint32_t* out_hashes,
                                              array_info* array, size_t n_rows,
                                              const uint32_t seed) {
    T* data = (T*)array->data1;
    uint32_t out_hash;
    if (array->arr_type == bodo_array_type::NUMPY) {
        uint32_t out_hash;
        for (size_t i = 0; i < n_rows; i++) {
            double val = get_value(data[i]);
            hash_inner_32<double>(&val, seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask;
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(null_bitmask, i);
            double val;
            if (bit)
                val = get_value(data[i]);
            else
                val = std::nan("");
            hash_inner_32<double>(&val, seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    }
}

void coherent_hash_array_combine(uint32_t* out_hashes, array_info* array,
                                 array_info* ref_array, size_t n_rows,
                                 const uint32_t seed, bool is_parallel) {
    // For those types, no type conversion is ever needed.
    if (array->arr_type == bodo_array_type::ARROW ||
        array->arr_type == bodo_array_type::STRING ||
        array->arr_type == bodo_array_type::LIST_STRING) {
        return hash_array_combine(out_hashes, array, n_rows, seed, true,
                                  is_parallel);
    }
    // Now we are in NUMPY / NULLABLE_INT_BOOL. Getting into hot waters.
    // For DATE / DATETIME / TIMEDELTA / DECIMAL no type conversion is allowed
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA ||
        array->dtype == Bodo_CTypes::DECIMAL ||
        array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array_combine(out_hashes, array, n_rows, seed, true,
                                  is_parallel);
    }
    // If we have the same type on left or right then no need
    if (array->arr_type == ref_array->arr_type ||
        array->dtype == ref_array->dtype) {
        return hash_array_combine(out_hashes, array, n_rows, seed, true,
                                  is_parallel);
    }
    // If both are unsigned int, we convert to uint64_t
    if (is_unsigned_integer(array->dtype) &&
        is_unsigned_integer(ref_array->dtype)) {
        if (array->dtype == Bodo_CTypes::UINT8)
            return coherent_hash_array_combine_inner_uint64<uint8_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT16)
            return coherent_hash_array_combine_inner_uint64<uint16_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT32)
            return coherent_hash_array_combine_inner_uint64<uint32_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT64)
            return coherent_hash_array_combine_inner_uint64<uint64_t>(
                out_hashes, array, n_rows, seed);
    }
    // If both are integer (signed or unsigned), we convert to int64_t
    if (is_integer(array->dtype) && is_integer(ref_array->dtype)) {
        if (array->dtype == Bodo_CTypes::UINT8)
            return coherent_hash_array_combine_inner_int64<uint8_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT16)
            return coherent_hash_array_combine_inner_int64<uint16_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT32)
            return coherent_hash_array_combine_inner_int64<uint32_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::UINT64)
            return coherent_hash_array_combine_inner_int64<uint64_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::INT8)
            return coherent_hash_array_combine_inner_int64<int8_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::INT16)
            return coherent_hash_array_combine_inner_int64<int16_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::INT32)
            return coherent_hash_array_combine_inner_int64<int32_t>(
                out_hashes, array, n_rows, seed);
        if (array->dtype == Bodo_CTypes::INT64)
            return coherent_hash_array_combine_inner_int64<int64_t>(
                out_hashes, array, n_rows, seed);
    }
    // In all other cases, we convert to double
    if (array->dtype == Bodo_CTypes::UINT8)
        return coherent_hash_array_combine_inner_double<uint8_t>(
            out_hashes, array, n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT16)
        return coherent_hash_array_combine_inner_double<uint16_t>(
            out_hashes, array, n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT32)
        return coherent_hash_array_combine_inner_double<uint32_t>(
            out_hashes, array, n_rows, seed);
    if (array->dtype == Bodo_CTypes::UINT64)
        return coherent_hash_array_combine_inner_double<uint64_t>(
            out_hashes, array, n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT8)
        return coherent_hash_array_combine_inner_double<int8_t>(
            out_hashes, array, n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT16)
        return coherent_hash_array_combine_inner_double<int16_t>(
            out_hashes, array, n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT32)
        return coherent_hash_array_combine_inner_double<int32_t>(
            out_hashes, array, n_rows, seed);
    if (array->dtype == Bodo_CTypes::INT64)
        return coherent_hash_array_combine_inner_double<int64_t>(
            out_hashes, array, n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return coherent_hash_array_combine_inner_double<float>(
            out_hashes, array, n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return coherent_hash_array_combine_inner_double<double>(
            out_hashes, array, n_rows, seed);
}

/* The coherent_hash_keys is for computing hashes for join computation.
   What can happen is that columns have different type but we need to have
   coherent hash.
   ---
   Examples of pairs of type that we need to support
   1) uint8_t / uint32_t
   2) int8_t / uint32_t
   3) int32_t / float32
   4) nullable int16_t / double
   and to have coherent hashes on both sides.
   ---
   @param key_arrs: the keys for which we want the hash
   @param ref_key_arrs: the keys on the other side. Used only for their
   arr_type/dtype
   @param seed: the seed used as input
   @param is_parallel: Is the input data distributed
   @return returning the list of hashes.
 */
uint32_t* coherent_hash_keys(std::vector<array_info*> const& key_arrs,
                             std::vector<array_info*> const& ref_key_arrs,
                             const uint32_t seed, bool is_parallel) {
    tracing::Event ev("coherent_hash_keys", is_parallel);
    size_t n_rows = (size_t)key_arrs[0]->length;
    uint32_t* hashes = new uint32_t[n_rows];
    coherent_hash_array(hashes, key_arrs[0], ref_key_arrs[0], n_rows, seed,
                        is_parallel);
    for (size_t i = 1; i < key_arrs.size(); i++) {
        coherent_hash_array_combine(hashes, key_arrs[i], ref_key_arrs[i],
                                    n_rows, seed, is_parallel);
    }
    return hashes;
}

uint32_t* hash_keys(std::vector<array_info*> const& key_arrs,
                    const uint32_t seed, bool is_parallel,
                    bool global_dict_needed) {
    tracing::Event ev("hash_keys", is_parallel);
    size_t n_rows = (size_t)key_arrs[0]->length;
    uint32_t* hashes = new uint32_t[n_rows];
    // hash first array
    hash_array(hashes, key_arrs[0], n_rows, seed, is_parallel,
               global_dict_needed);
    // combine other array hashes
    for (size_t i = 1; i < key_arrs.size(); i++) {
        hash_array_combine(hashes, key_arrs[i], n_rows, seed,
                           global_dict_needed, is_parallel);
    }
    return hashes;
}

/**
 * @brief Verify that the dictionary arrays attempted to be unified have
 * satified the requirements for unification.
 *
 * @param arrs The arrays to unify.
 * @param is_parallels If each array is parallel.
 */
void ensure_dicts_can_unify(std::vector<array_info*>& arrs,
                            std::vector<bool>& is_parallels) {
    for (size_t i = 0; i < arrs.size(); i++) {
        if (is_parallels[i] && !arrs[i]->has_global_dictionary) {
            throw std::runtime_error(
                "unify_dictionaries: array does not have global dictionary");
        }
        if (!arrs[i]->has_deduped_local_dictionary) {
            throw std::runtime_error(
                "unify_dictionaries: array's dictionary has duplicate "
                "values");
        }
    }
}

/**
 * @brief Create a Hashmap that that compares several arrays that are inserted
 * one at a time.
 *
 * @param arrs[in] The arrays that may need to be inserted.
 * @param hashes[in] The vector where hashes will be inserted.
 * @param stored_arrs[in] The vector where arrays will be inserted
 * @return UNORD_MAP_CONTAINER<std::pair<size_t, size_t>, dict_indices_t>* A
 * pointer to the heap allocated hashmap.
 */
UNORD_MAP_CONTAINER<std::pair<size_t, size_t>, dict_indices_t, HashMultiArray,
                    MultiArrayInfoEqual>*
create_several_array_hashmap(std::vector<array_info*>& arrs,
                             std::vector<uint32_t*>& hashes,
                             std::vector<array_info*>& stored_arrs) {
    // hash map mapping dictionary values of arr1 and arr2 to index in unified
    // dictionary
    HashMultiArray hash_fct{hashes};
    MultiArrayInfoEqual equal_fct{stored_arrs};
    UNORD_MAP_CONTAINER<std::pair<size_t, size_t>, dict_indices_t,
                        HashMultiArray, MultiArrayInfoEqual>*
        dict_value_to_unified_index =
            new UNORD_MAP_CONTAINER<std::pair<size_t, size_t>, dict_indices_t,
                                    HashMultiArray, MultiArrayInfoEqual>(
                {}, hash_fct, equal_fct);
    // Estimate how much to reserve. We could get an accurate
    // estimate with hyperloglog but it seems unnecessary for this use case.
    // For now we reserve initial capacity as the max size of any of the
    // dictionaries.
    std::vector<size_t> lengths(arrs.size());
    for (size_t i = 0; i < arrs.size(); i++) {
        lengths[i] = arrs[i]->info1->length;
    }
    size_t max_length = *std::max_element(lengths.begin(), lengths.end());
    dict_value_to_unified_index->reserve(max_length);
    return dict_value_to_unified_index;
}

/**
 * @brief Inserts the initial dictionary to the hashmap.
 * This dictionary is guaranteed to be unique so we never need to check
 * if it is already in the hashmap.
 *
 * @param[in] dict_value_to_unified_index The hashmap
 * @param[out] hashes The vector of hashes used by the hashmap.
 * @param[out] stored_arrs The vector of arrays used by the hashmap.
 * @param[in] dict THe input dictionary.
 * @param[in] offsets The array of offsets. This is used to determine
 * how many characters will need to be inserted into the new dictionary.
 * @param[in] len The length of the dictionary
 * @param[in, out] next_index The next index to insert in the hashmap.
 * @param[in, out] n_chars The number of chars needed by the data that matches
 * the keys in the hashmap.
 * @param[in] hash_seed Seed for hashing
 */
void insert_initial_dict_to_multiarray_hashmap(
    UNORD_MAP_CONTAINER<std::pair<size_t, size_t>, dict_indices_t,
                        HashMultiArray, MultiArrayInfoEqual>*
        dict_value_to_unified_index,
    std::vector<uint32_t*>& hashes, std::vector<array_info*>& stored_arrs,
    array_info* dict, offset_t const* const offsets, const size_t len,
    dict_indices_t& next_index, size_t& n_chars, const uint32_t hash_seed) {
    uint32_t* arr_hashes = new uint32_t[len];
    hash_array(arr_hashes, dict, len, hash_seed, false,
               /*global_dict_needed=*/false);
    // Insert the hashes and the array
    hashes.push_back(arr_hashes);
    stored_arrs.push_back(dict);
    // Update the number of chars
    n_chars += offsets[len];
    // Insert the dictionary values
    for (size_t j = 0; j < len; j++) {
        // Set the first n elements in the hash map, each of which is
        // always unique.
        dict_indices_t& index =
            (*dict_value_to_unified_index)[std::pair<size_t, size_t>(j, 0)];
        index = next_index++;
    }
}

/**
 * @brief Inserts a new dictionary to the hashmap.
 *
 * @param[in] dict_value_to_unified_index The hashmap
 * @param[out] hashes The vector of hashes used by the hashmap.
 * @param[out] stored_arrs The vector of arrays used by the hashmap.
 * @param[out] arr_index_map The vector of mapping the current indices
 * to the indices in the final dictionary.
 * @param[out] unique_indices_all_arrs The vector that stores the vector
 * of row numbers for the unique indices in each newly inserted array.
 * @param[in] dict THe input dictionary.
 * @param[in] offsets The array of offsets. This is used to determine
 * how many characters will need to be inserted into the new dictionary.
 * @param[in] len The length of the dictionary
 * @param[in, out] next_index The next index to insert in the hashmap.
 * @param[in, out] n_chars The number of chars needed by the data that matches
 * the keys in the hashmap.
 * @param[in] arr_num What number array being inserted is this?
 * @param[in] hash_seed Seed for hashing
 */
void insert_new_dict_to_multiarray_hashmap(
    UNORD_MAP_CONTAINER<std::pair<size_t, size_t>, dict_indices_t,
                        HashMultiArray, MultiArrayInfoEqual>*
        dict_value_to_unified_index,
    std::vector<uint32_t*>& hashes, std::vector<array_info*>& stored_arrs,
    std::vector<dict_indices_t>& arr_index_map,
    std::vector<std::vector<dict_indices_t>*>& unique_indices_all_arrs,
    array_info* dict, offset_t const* const offsets, const size_t len,
    dict_indices_t& next_index, size_t& n_chars, size_t arr_num,
    const uint32_t hash_seed) {
    uint32_t* arr_hashes = new uint32_t[len];
    hash_array(arr_hashes, dict, len, hash_seed, false,
               /*global_dict_needed=*/false);
    // Insert the hashes and the array
    hashes.push_back(arr_hashes);
    stored_arrs.push_back(dict);
    // Create a vector to store the indices of the unique strings in the
    // current arr.
    std::vector<dict_indices_t>* unique_indices =
        new std::vector<dict_indices_t>();
    // Store the mapping of indices for this array
    // Insert the dictionary values
    for (size_t j = 0; j < len; j++) {
        // Set the first n elements in the hash map, each of which is
        // always unique.
        dict_indices_t& index =
            (*dict_value_to_unified_index)[std::pair<size_t, size_t>(j,
                                                                     arr_num)];
        // Hashmap's return 0 if there is no match
        if (index == 0) {
            // found new string
            index = next_index++;
            n_chars += (offsets[j + 1] - offsets[j]);
            unique_indices->emplace_back(j);
        }
        arr_index_map[j] = index - 1;
    }
    // Add this array's unique indices to the list of all arrays.
    unique_indices_all_arrs.emplace_back(unique_indices);
}

/**
 * @brief Update the indices for this array. If there is only one reference to
 * the dict_array remaining we can update the array inplace without
 * allocating a new array.
 *
 * @param arr The array whose indices need to be updated.
 * @param arr_index_map Mapping from the indices in this array to the indices
 * in the new dictionary.
 */
void replace_dict_arr_indices(array_info* arr,
                              std::vector<dict_indices_t>& arr_index_map) {
    // Update the indices for this array. If there is only one reference to
    // the dict_array remaining we can update the array inplace without
    // allocating a new array.
    bool inplace = (arr->info2->meminfo->refct == 1);
    if (!inplace) {
        array_info* indices = copy_array(arr->info2);
        delete_info_decref_array(arr->info2);
        arr->info2 = indices;
        arr->null_bitmask = indices->null_bitmask;
    }

    uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;

    for (size_t j = 0; j < arr->info2->length; j++) {
        if (GetBit(null_bitmask, j)) {
            dict_indices_t& index = arr->info2->at<dict_indices_t>(j);
            index = arr_index_map[index];
        }
    }
}

void unify_several_dictionaries(std::vector<array_info*>& arrs,
                                std::vector<bool>& is_parallels) {
    // Validate the inputs
    ensure_dicts_can_unify(arrs, is_parallels);
    // Keep a vector of hashes for each array. That will be checked. We will
    // update this dynamically to avoid need to constantly rehash/update the
    // array.
    std::vector<uint32_t*> hashes;
    // Keep a vector of array infos
    std::vector<array_info*> stored_arrs;
    // Create the hash table. We will dynamically fill the vector of
    // hashes and dictionaries as we go.
    const uint32_t hash_seed = SEED_HASH_JOIN;
    UNORD_MAP_CONTAINER<std::pair<size_t, size_t>, dict_indices_t,
                        HashMultiArray, MultiArrayInfoEqual>*
        dict_value_to_unified_index =
            create_several_array_hashmap(arrs, hashes, stored_arrs);

    // The first dictionary will always be entirely included in the output
    // unified dictionary.
    array_info* base_dict = arrs[0]->info1;
    const size_t base_len = static_cast<size_t>(base_dict->length);
    offset_t const* const base_offsets = (offset_t*)base_dict->data2;
    bool added_first = false;
    size_t arr_num = 1;
    size_t n_chars = 0;
    dict_indices_t next_index = 1;

    // Keep track of the unique indices for each array. We will use this to
    // build the final dictionary. We will omit the base dictionary.
    std::vector<std::vector<dict_indices_t>*> unique_indices_all_arrs;

    for (size_t i = 1; i < arrs.size(); i++) {
        // Process the dictionaries 1 at a time. To do this we always insert
        // any new dictionary entries in order by array (first all of arr1,
        // then anything new from arr2, etc). As a result, this means that the
        // entries in arr{i} can never modify the indices of arr{i-1}.
        array_info* curr_arr = arrs[i];
        array_info* curr_dict = curr_arr->info1;
        offset_t const* const curr_dict_offsets = (offset_t*)curr_dict->data2;

        // Using this realization, we can then conclude that we can simply
        // process the dictionaries in order and then update the dictionaries at
        // the end.
        if (curr_dict == base_dict) {
            // If this dictionary matches the first one, we will
            // not add entries or update the indices.
            continue;
        }
        if (!added_first) {
            // If this is the first arr we are adding we need to insert
            // the base dictionary.
            insert_initial_dict_to_multiarray_hashmap(
                dict_value_to_unified_index, hashes, stored_arrs, base_dict,
                base_offsets, base_len, next_index, n_chars, hash_seed);
            added_first = true;
        }
        // Add the elements for the ith array.
        const size_t curr_len = static_cast<size_t>(curr_dict->length);
        // Store the mapping of indices for this array
        std::vector<dict_indices_t> arr_index_map(curr_len);

        insert_new_dict_to_multiarray_hashmap(
            dict_value_to_unified_index, hashes, stored_arrs, arr_index_map,
            unique_indices_all_arrs, curr_dict, curr_dict_offsets, curr_len,
            next_index, n_chars, arr_num, hash_seed);

        replace_dict_arr_indices(curr_arr, arr_index_map);

        // Update the array number.
        arr_num += 1;
    }
    if (!added_first) {
        // No dictionary was modified so we can just return.
        return;
    }

    delete dict_value_to_unified_index;
    // Free all of the hashes
    for (size_t i = 0; i < hashes.size(); i++) {
        delete[] hashes[i];
    }

    // Now that we have all of the dictionary elements we can create the
    // dictionary. The next_index is always num_strings + 1, so we can use that
    // to get the length of the dictionary.
    size_t n_strings = next_index - 1;
    array_info* new_dict = alloc_string_array(n_strings, n_chars, 0);
    offset_t* new_dict_str_offsets = (offset_t*)new_dict->data2;

    // Initialize the offset and string index to the end of the base dictionary
    offset_t cur_offset = base_offsets[base_len];
    int64_t cur_offset_idx = base_len + 1;

    // copy offsets from arr1 into new_dict_str_offsets
    memcpy(new_dict_str_offsets, base_offsets,
           cur_offset_idx * sizeof(offset_t));
    // copy strings from arr1 into new_dict
    memcpy(new_dict->data1, base_dict->data1, cur_offset);
    for (size_t i = 0; i < unique_indices_all_arrs.size(); i++) {
        std::vector<dict_indices_t>*& arr_unique_indices =
            unique_indices_all_arrs[i];
        // Load the relevant array. This is the i+1 array we stored for the
        // hashmap because we skip the base array.
        array_info* dict_arr = stored_arrs[i + 1];
        offset_t const* const arr_offsets = (offset_t*)dict_arr->data2;

        for (dict_indices_t j : *arr_unique_indices) {
            offset_t str_len = arr_offsets[j + 1] - arr_offsets[j];
            memcpy(new_dict->data1 + cur_offset,
                   dict_arr->data1 + arr_offsets[j], str_len);
            cur_offset += str_len;
            new_dict_str_offsets[cur_offset_idx++] = cur_offset;
        }
        delete arr_unique_indices;
    }

    // replace old dictionaries with a new one
    for (size_t i = 0; i < arrs.size(); i++) {
        delete_info_decref_array(arrs[i]->info1);
        arrs[i]->info1 = new_dict;
        if (i != 0) {
            // This same array is now in N places so we need to incref it.
            incref_array(new_dict);
        }
    }
}

void unify_dictionaries(array_info* arr1, array_info* arr2,
                        bool arr1_is_parallel, bool arr2_is_parallel) {
    // Validate the inputs
    std::vector<array_info*> arrs = {arr1, arr2};
    std::vector<bool> is_parallel = {arr1_is_parallel, arr2_is_parallel};
    ensure_dicts_can_unify(arrs, is_parallel);

    if (arr1->info1 == arr2->info1) return;  // dictionaries are the same

    // Note we insert the dictionaries in order (arr1 then arr2). Since we have
    // ensured there are no duplicates this means that only the indices in arr2
    // can change and the entire dictionary in arr1 will be in the unified dict.

    const size_t arr1_dictionary_len = static_cast<size_t>(arr1->info1->length);
    const size_t arr2_dictionary_len = static_cast<size_t>(arr2->info1->length);
    // this vector will be used to map old indices to new ones
    std::vector<dict_indices_t> arr2_index_map(arr2_dictionary_len);

    const uint32_t hash_seed = SEED_HASH_JOIN;
    uint32_t* arr1_hashes = new uint32_t[arr1_dictionary_len];
    uint32_t* arr2_hashes = new uint32_t[arr2_dictionary_len];
    hash_array(arr1_hashes, arr1->info1, arr1_dictionary_len, hash_seed, false,
               /*global_dict_needed=*/false);
    hash_array(arr2_hashes, arr2->info1, arr2_dictionary_len, hash_seed, false,
               /*global_dict_needed=*/false);

    // hash map mapping dictionary values of arr1 and arr2 to index in unified
    // dictionary
    HashDict hash_fct{arr1_dictionary_len, arr1_hashes, arr2_hashes};
    KeyEqualDict equal_fct{arr1_dictionary_len, arr1->info1,
                           arr2->info1 /*, is_na_equal*/};
    UNORD_MAP_CONTAINER<size_t, dict_indices_t, HashDict,
                        KeyEqualDict>* dict_value_to_unified_index =
        new UNORD_MAP_CONTAINER<size_t, dict_indices_t, HashDict, KeyEqualDict>(
            {}, hash_fct, equal_fct);
    // Size of new dictionary could end up as large as
    // arr1_dictionary_len + arr2_dictionary_len. We could get an accurate
    // estimate with hyperloglog but it seems unnecessary for this use case.
    // For now we reserve initial capacity as the max size of the two
    dict_value_to_unified_index->reserve(
        std::max(arr1_dictionary_len, arr2_dictionary_len));

    // this vector stores indices of the strings in arr2 that will be
    // part of the unified dictionary. All of array 1's strings will always
    // be part of the unified dictionary.
    std::vector<size_t> arr2_unique_strs;
    arr2_unique_strs.reserve(arr2_dictionary_len);

    offset_t const* const arr1_str_offsets = (offset_t*)arr1->info1->data2;
    int64_t n_chars = arr1_str_offsets[arr1_dictionary_len];

    dict_indices_t next_index = 1;
    for (size_t i = 0; i < arr1_dictionary_len; i++) {
        // TODO: Move into the constructor
        // Set the first n elements in the hash map, each of which is
        // always unique.
        dict_indices_t& index = (*dict_value_to_unified_index)[i];
        index = next_index++;
    }

    offset_t const* const arr2_str_offsets = (offset_t*)arr2->info1->data2;
    for (size_t i = 0; i < arr2_dictionary_len; i++) {
        dict_indices_t& index =
            (*dict_value_to_unified_index)[i + arr1_dictionary_len];
        if (index == 0) {
            // found new string
            index = next_index++;
            n_chars += (arr2_str_offsets[i + 1] - arr2_str_offsets[i]);
            arr2_unique_strs.emplace_back(i);
        }
        arr2_index_map[i] = index - 1;
    }
    int64_t n_strings = arr1_dictionary_len + arr2_unique_strs.size();
    delete dict_value_to_unified_index;
    delete[] arr1_hashes;
    delete[] arr2_hashes;

    array_info* new_dict = alloc_string_array(n_strings, n_chars, 0);
    offset_t* new_dict_str_offsets = (offset_t*)new_dict->data2;

    // Initialize the offset and string index to the end of arr1's dictionary
    offset_t cur_offset = arr1_str_offsets[arr1_dictionary_len];
    int64_t cur_offset_idx = arr1_dictionary_len + 1;

    // copy offsets from arr1 into new_dict_str_offsets
    memcpy(new_dict_str_offsets, arr1_str_offsets,
           cur_offset_idx * sizeof(offset_t));
    // copy strings from arr1 into new_dict
    memcpy(new_dict->data1, arr1->info1->data1, cur_offset);
    for (auto i : arr2_unique_strs) {
        offset_t str_len = arr2_str_offsets[i + 1] - arr2_str_offsets[i];
        memcpy(new_dict->data1 + cur_offset,
               arr2->info1->data1 + arr2_str_offsets[i], str_len);
        cur_offset += str_len;
        new_dict_str_offsets[cur_offset_idx++] = cur_offset;
    }

    // replace old dictionaries with new one
    delete_info_decref_array(arr1->info1);
    delete_info_decref_array(arr2->info1);
    arr1->info1 = new_dict;
    arr2->info1 = new_dict;
    incref_array(new_dict);

    // convert old indices to new ones for arr2
    replace_dict_arr_indices(arr2, arr2_index_map);
}
