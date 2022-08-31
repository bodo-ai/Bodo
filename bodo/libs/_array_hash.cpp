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
 * TODO: [BE-975] Use this to trigger with hash_array_inner.
 */
static void hash_na_val(const uint32_t seed, uint32_t* hash_value) {
    int64_t val = 1;
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
 *
 */
template <typename T>
static typename std::enable_if<!std::is_floating_point<T>::value, void>::type
hash_array_inner(uint32_t* out_hashes, T* data, size_t n_rows,
                 const uint32_t seed, uint8_t* null_bitmask) {
    if (null_bitmask) {
        uint32_t na_hash;
        hash_na_val(seed, &na_hash);
        for (size_t i = 0; i < n_rows; i++) {
            hash_inner_32<T>(&data[i], seed, &out_hashes[i]);
            if (!GetBit(null_bitmask, i)) out_hashes[i] = na_hash;
        }
    } else {
        for (size_t i = 0; i < n_rows; i++)
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
                 const uint32_t seed) {
    for (size_t i = 0; i < n_rows; i++) {
        Py_hash_t py_hash = Npy_HashDouble(nullptr, data[i]);
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
        std::string val(&data[data_offsets[start_index_offset]], len1);
        const char* val_chars1 = val.c_str();
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
 *
 * The hash is computed in 3 stages:
 * 1) The hash of the concatenated strings
 * 2) The hash of the string length
 * 3) The hash of the bitmask
 */
static void hash_array_list_string(uint32_t* out_hashes, char* data,
                                   offset_t* data_offsets,
                                   offset_t* index_offsets,
                                   uint8_t* null_bitmask,
                                   uint8_t* sub_null_bitmask, size_t n_rows,
                                   const uint32_t seed) {
    offset_t start_index_offset = 0;
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t hash1, hash2, hash3;
        // First the hash from the strings.
        offset_t end_index_offset = index_offsets[i + 1];
        offset_t len1 =
            data_offsets[end_index_offset] - data_offsets[start_index_offset];
        std::string val(&data[data_offsets[start_index_offset]], len1);
        const char* val_chars1 = val.c_str();
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
 * Computation of the NA string hash
 * @param seed: the seed of the computation.
 * @param[out] hash_value: The hashes on output.
 */
static void hash_na_string(const uint32_t seed, uint32_t* hash_value) {
    char val_c = 1;
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
 *
 * Right now, the bitmask is not used in the computation, which
 * may be a problem to consider later on.
 */
static void hash_array_string(uint32_t* out_hashes, char* data,
                              offset_t* offsets, uint8_t* null_bitmask,
                              size_t n_rows, const uint32_t seed,
                              bool is_parallel) {
    tracing::Event ev("hash_array_string", is_parallel);
    offset_t start_offset = 0;
    uint32_t na_hash;
    hash_na_string(seed, &na_hash);
    for (size_t i = 0; i < n_rows; i++) {
        offset_t end_offset = offsets[i + 1];
        offset_t len = end_offset - start_offset;
        std::string val(&data[start_offset], len);
        // val is null
        if (is_na(null_bitmask, i)) {
            out_hashes[i] = na_hash;
        } else {
            const char* val_chars = val.c_str();
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
 *
 * One approximation is the casting to char of the algorithm.
 */
template <typename T>
void apply_arrow_offset_hash(uint32_t* out_hashes,
                             std::vector<offset_t> const& list_offsets,
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
 * @param : out_hashes, the hashes on input/output
 * @param : list_offsets, the list of offsets (of length n_rows+1)
 * @param : n_rows, the number of rows in input
 * @param : input_array, the input array put in argument.
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
 * after shuffling-). This is context-dependent
 */
void hash_array(uint32_t* out_hashes, array_info* array, size_t n_rows,
                const uint32_t seed, bool is_parallel,
                bool global_dict_needed) {
    // dispatch to proper function
    // TODO: general dispatcher
    // XXX: assumes nullable array data for nulls is always consistent
    if (array->arr_type == bodo_array_type::ARROW) {
        std::vector<offset_t> list_offsets(n_rows + 1);
        for (offset_t i = 0; i <= n_rows; i++) list_offsets[i] = i;
        for (offset_t i = 0; i < n_rows; i++) out_hashes[i] = seed;
        return hash_arrow_array(out_hashes, list_offsets, n_rows, array->array);
    }
    if (array->arr_type == bodo_array_type::STRING) {
        return hash_array_string(
            out_hashes, (char*)array->data1, (offset_t*)array->data2,
            (uint8_t*)array->null_bitmask, n_rows, seed, is_parallel);
    }
    if (array->arr_type == bodo_array_type::DICT) {
        if (array->has_global_dictionary || !is_parallel ||
            !global_dict_needed) {
            // in this case we can just hash the indices since the dictionary is
            // synchronized across ranks or is only needed for a local
            // operation where hashing based on local dictionary won't affect
            // correctness or performance
            return hash_array_inner<dict_indices_t>(
                out_hashes, (dict_indices_t*)array->info2->data1, n_rows, seed,
                (uint8_t*)array->info2->null_bitmask);
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
                "hashing dictionary array requires global dictionary in this "
                "context");
        }
    }
    if (array->arr_type == bodo_array_type::LIST_STRING) {
        return hash_array_list_string(
            out_hashes, (char*)array->data1, (offset_t*)array->data2,
            (offset_t*)array->data3, (uint8_t*)array->null_bitmask,
            (uint8_t*)array->sub_null_bitmask, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array_inner<bool>(out_hashes, (bool*)array->data1, n_rows,
                                      seed, (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::INT8) {
        return hash_array_inner<int8_t>(out_hashes, (int8_t*)array->data1,
                                        n_rows, seed,
                                        (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::UINT8) {
        return hash_array_inner<uint8_t>(out_hashes, (uint8_t*)array->data1,
                                         n_rows, seed,
                                         (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::INT16) {
        return hash_array_inner<int16_t>(out_hashes, (int16_t*)array->data1,
                                         n_rows, seed,
                                         (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::UINT16) {
        return hash_array_inner<uint16_t>(out_hashes, (uint16_t*)array->data1,
                                          n_rows, seed,
                                          (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::INT32) {
        return hash_array_inner<int32_t>(out_hashes, (int32_t*)array->data1,
                                         n_rows, seed,
                                         (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::UINT32) {
        return hash_array_inner<uint32_t>(out_hashes, (uint32_t*)array->data1,
                                          n_rows, seed,
                                          (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::INT64) {
        return hash_array_inner<int64_t>(out_hashes, (int64_t*)array->data1,
                                         n_rows, seed,
                                         (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::DECIMAL) {
        return hash_array_inner<decimal_value_cpp>(
            out_hashes, (decimal_value_cpp*)array->data1, n_rows, seed,
            (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::UINT64) {
        return hash_array_inner<uint64_t>(out_hashes, (uint64_t*)array->data1,
                                          n_rows, seed,
                                          (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA) {
        return hash_array_inner<int64_t>(out_hashes, (int64_t*)array->data1,
                                         n_rows, seed,
                                         (uint8_t*)array->null_bitmask);
    }
    if (array->dtype == Bodo_CTypes::FLOAT32) {
        return hash_array_inner<float>(out_hashes, (float*)array->data1, n_rows,
                                       seed);
    }
    if (array->dtype == Bodo_CTypes::FLOAT64) {
        return hash_array_inner<double>(out_hashes, (double*)array->data1,
                                        n_rows, seed);
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
static void hash_array_combine_inner(uint32_t* out_hashes, T* data,
                                     size_t n_rows, const uint32_t seed) {
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t out_hash = 0;
        hash_inner_32<T>(&data[i], seed, &out_hash);
        hash_combine_boost(out_hashes[i], out_hash);
    }
}

static void hash_array_combine_string(uint32_t* out_hashes, char* data,
                                      offset_t* offsets, size_t n_rows,
                                      const uint32_t seed) {
    offset_t start_offset = 0;
    for (size_t i = 0; i < n_rows; i++) {
        offset_t end_offset = offsets[i + 1];
        offset_t len = end_offset - start_offset;
        std::string val(&data[start_offset], len);

        uint32_t out_hash = 0;

        const char* val_chars = val.c_str();
        hash_string_32(val_chars, (const int)len, seed, &out_hash);
        hash_combine_boost(out_hashes[i], out_hash);
        start_offset = end_offset;
    }
}

// See hash_array for documentation of parameters
void hash_array_combine(uint32_t* out_hashes, array_info* array, size_t n_rows,
                        const uint32_t seed, bool global_dict_needed) {
    // dispatch to proper function
    // TODO: general dispatcher
    if (array->arr_type == bodo_array_type::ARROW) {
        std::vector<offset_t> list_offsets(n_rows + 1);
        for (offset_t i = 0; i <= n_rows; i++) list_offsets[i] = i;
        return hash_arrow_array(out_hashes, list_offsets, n_rows, array->array);
    }
    if (array->arr_type == bodo_array_type::STRING) {
        return hash_array_combine_string(out_hashes, (char*)array->data1,
                                         (offset_t*)array->data2, n_rows, seed);
    }
    if (array->arr_type == bodo_array_type::DICT) {
        if (array->has_global_dictionary || !global_dict_needed) {
            // in this case we can just hash the indices since the dictionary is
            // synchronized across ranks or is only needed for a local
            // operation where hashing based on local dictionary won't affect
            // correctness or performance
            return hash_array_combine_inner<dict_indices_t>(
                out_hashes, (dict_indices_t*)array->info2->data1, n_rows, seed);
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
                "hashing dictionary array requires global dictionary in this "
                "context");
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
                                              n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT8) {
        return hash_array_combine_inner<int8_t>(
            out_hashes, (int8_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT8) {
        return hash_array_combine_inner<uint8_t>(
            out_hashes, (uint8_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT16) {
        return hash_array_combine_inner<int16_t>(
            out_hashes, (int16_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT16) {
        return hash_array_combine_inner<uint16_t>(
            out_hashes, (uint16_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT32) {
        return hash_array_combine_inner<int32_t>(
            out_hashes, (int32_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT32) {
        return hash_array_combine_inner<uint32_t>(
            out_hashes, (uint32_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::INT64) {
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::UINT64) {
        return hash_array_combine_inner<uint64_t>(
            out_hashes, (uint64_t*)array->data1, n_rows, seed);
    }
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA) {
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1, n_rows, seed);
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
            out_hashes, (decimal_value_cpp*)array->data1, n_rows, seed);
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
        // enforced by the has_global_dictionary check in unify_dictionaries
        // and is updated by convert_local_dictionary_to_global. In particular,
        // convert_local_dictionary_to_global contains a drop duplicates step
        // that ensures all values are unique. If the dictionary is made global
        // by some other means (e.g. Python), then we assume that is also
        // unique.
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
    // For DATE / DATETIME / TIMEDELTA / DECIMAL no type conversion is allowed
    if (array->dtype == Bodo_CTypes::DATE ||
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
                                 const uint32_t seed) {
    // For those types, no type conversion is ever needed.
    if (array->arr_type == bodo_array_type::ARROW ||
        array->arr_type == bodo_array_type::STRING ||
        array->arr_type == bodo_array_type::LIST_STRING) {
        return hash_array_combine(out_hashes, array, n_rows, seed, true);
    }
    // Now we are in NUMPY / NULLABLE_INT_BOOL. Getting into hot waters.
    // For DATE / DATETIME / TIMEDELTA / DECIMAL no type conversion is allowed
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA ||
        array->dtype == Bodo_CTypes::DECIMAL ||
        array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array_combine(out_hashes, array, n_rows, seed, true);
    }
    // If we have the same type on left or right then no need
    if (array->arr_type == ref_array->arr_type ||
        array->dtype == ref_array->dtype) {
        return hash_array_combine(out_hashes, array, n_rows, seed, true);
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
   @return returning the list of hashes.
 */
uint32_t* coherent_hash_keys(std::vector<array_info*> const& key_arrs,
                             std::vector<array_info*> const& ref_key_arrs,
                             const uint32_t seed) {
    size_t n_rows = (size_t)key_arrs[0]->length;
    uint32_t* hashes = new uint32_t[n_rows];
    coherent_hash_array(hashes, key_arrs[0], ref_key_arrs[0], n_rows, seed);
    for (size_t i = 1; i < key_arrs.size(); i++) {
        coherent_hash_array_combine(hashes, key_arrs[i], ref_key_arrs[i],
                                    n_rows, seed);
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
                           global_dict_needed);
    }
    return hashes;
}

void unify_dictionaries(array_info* arr1, array_info* arr2) {
    // TODO To simplify things for now we require dictionaries to be global
    if (!arr1->has_global_dictionary)
        throw std::runtime_error(
            "unify_dictionaries: first array does not have global dictionary");

    if (!arr2->has_global_dictionary)
        throw std::runtime_error(
            "unify_dictionaries: second array does not have global dictionary");

    if (arr1->info1 == arr2->info1) return;  // dictionaries are the same

    const size_t arr1_dictionary_len = static_cast<size_t>(arr1->info1->length);
    const size_t arr2_dictionary_len = static_cast<size_t>(arr2->info1->length);
    // these vectors will be used to map old indices to new ones
    std::vector<dict_indices_t> arr1_index_map(arr1_dictionary_len);
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
    UNORD_MAP_CONTAINER<size_t, dict_indices_t, HashDict, KeyEqualDict>
        dict_value_to_unified_index({}, hash_fct, equal_fct);
    // Size of new dictionary could end up as large as
    // arr1_dictionary_len + arr2_dictionary_len. We could get an accurate
    // estimate with hyperloglog but it seems unnecessary for this use case.
    // For now we reserve initial capacity as the max size of the two
    dict_value_to_unified_index.reserve(
        std::max(arr1_dictionary_len, arr2_dictionary_len));

    // this vector stores indices of the strings (in arr1 and arr2) that will be
    // part of the unified dictionary. index values greater than
    // arr1_dictionary_len correspond to arr2 (for these we substract
    // arr1_dictionary_len to get the index value)
    std::vector<size_t> unique_strs;
    unique_strs.reserve(std::max(arr1_dictionary_len, arr2_dictionary_len));

    dict_indices_t next_index = 1;
    int64_t n_chars = 0;

    offset_t const* const arr1_str_offsets = (offset_t*)arr1->info1->data2;
    for (size_t i = 0; i < arr1_dictionary_len; i++) {
        dict_indices_t& index = dict_value_to_unified_index[i];
        if (index == 0) {
            // found new string
            index = next_index++;
            n_chars += (arr1_str_offsets[i + 1] - arr1_str_offsets[i]);
            unique_strs.emplace_back(i);
        }
        arr1_index_map[i] = index - 1;
    }

    offset_t const* const arr2_str_offsets = (offset_t*)arr2->info1->data2;
    for (size_t i = 0; i < arr2_dictionary_len; i++) {
        dict_indices_t& index =
            dict_value_to_unified_index[i + arr1_dictionary_len];
        if (index == 0) {
            // found new string
            index = next_index++;
            n_chars += (arr2_str_offsets[i + 1] - arr2_str_offsets[i]);
            unique_strs.emplace_back(i + arr1_dictionary_len);
        }
        arr2_index_map[i] = index - 1;
    }
    int64_t n_strings = unique_strs.size();
    dict_value_to_unified_index.clear();
    dict_value_to_unified_index.reserve(0);  // try to force dealloc of hashmap
    delete[] arr1_hashes;
    delete[] arr2_hashes;

    array_info* new_dict = alloc_string_array(n_strings, n_chars, 0);
    offset_t* new_dict_str_offsets = (offset_t*)new_dict->data2;
    offset_t cur_offset = 0;
    int64_t cur_str = 0;
    offset_t str_len;
    for (auto i : unique_strs) {
        if (i < arr1_dictionary_len) {
            str_len = arr1_str_offsets[i + 1] - arr1_str_offsets[i];
            memcpy(new_dict->data1 + cur_offset,
                   arr1->info1->data1 + arr1_str_offsets[i], str_len);
        } else {
            str_len = arr2_str_offsets[i - arr1_dictionary_len + 1] -
                      arr2_str_offsets[i - arr1_dictionary_len];
            memcpy(
                new_dict->data1 + cur_offset,
                arr2->info1->data1 + arr2_str_offsets[i - arr1_dictionary_len],
                str_len);
        }
        new_dict_str_offsets[cur_str++] = cur_offset;
        cur_offset += str_len;
    }
    new_dict_str_offsets[n_strings] = n_chars;

    // replace old dictionaries with new one
    delete_info_decref_array(arr1->info1);
    delete_info_decref_array(arr2->info1);
    arr1->info1 = new_dict;
    arr2->info1 = new_dict;
    incref_array(new_dict);

    // convert old indices to new ones

    bool inplace = (arr1->info2->meminfo->refct == 1);
    if (!inplace) {
        array_info* indices = copy_array(arr1->info2);
        delete_info_decref_array(arr1->info2);
        arr1->info2 = indices;
        arr1->null_bitmask = indices->null_bitmask;
    }

    uint8_t* null_bitmask1 = (uint8_t*)arr1->null_bitmask;

    for (size_t i = 0; i < arr1->info2->length; i++) {
        if (GetBit(null_bitmask1, i)) {
            dict_indices_t& index = arr1->info2->at<dict_indices_t>(i);
            index = arr1_index_map[index];
        }
    }

    inplace = (arr2->info2->meminfo->refct == 1);
    if (!inplace) {
        array_info* indices = copy_array(arr2->info2);
        delete_info_decref_array(arr2->info2);
        arr2->info2 = indices;
        arr2->null_bitmask = indices->null_bitmask;
    }

    uint8_t* null_bitmask2 = (uint8_t*)arr2->null_bitmask;

    for (size_t i = 0; i < arr2->info2->length; i++) {
        if (GetBit(null_bitmask2, i)) {
            dict_indices_t& index = arr2->info2->at<dict_indices_t>(i);
            index = arr2_index_map[index];
        }
    }
}
