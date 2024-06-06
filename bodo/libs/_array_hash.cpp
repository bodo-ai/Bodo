// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_hash.h"
#include <Python.h>
#include <arrow/api.h>
#include <arrow/array/data.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/compute/api_vector.h>
#include <span>
#include "_array_utils.h"
#include "_bodo_common.h"

/**
 * Computation of the NA value hash
 * @param seed: the seed of the computation.
 * @param[out] hash_value: The hashes on output.
 * @tparam use_murmurhash: Use the murmurhash hash algorithm
 * (currently only used for Iceberg bucket transformation)
 * TODO: [BE-975] Use this to trigger with hash_array_inner.
 */
template <bool use_murmurhash = false>
static void hash_na_val(const uint32_t seed, uint32_t* hash_value) {
    int64_t val = 1;
    if (use_murmurhash) {
        hash_inner_murmurhash3_x86_32<int64_t>(&val, seed, hash_value);
    } else {
        hash_inner_32<int64_t>(&val, seed, hash_value);
    }
}
/**
 * Computation of the inner hash of the functions. This covers the NUMPY case.
 *
 * @param out_hashes: The hashes on output.
 * @param data: the list of data in input.
 * @param n_rows: Number of rows to hash starting from start_row_offset, i.e. we
 * will hash rows at indices [start_row_offset, start_row_offset + n_rows - 1].
 * @param seed: the seed of the computation.
 * @param null_bitmask: the null_bitmask of the data.
 * @tparam use_murmurhash: Use the murmurhash hash algorithm (currently only
 * used for Iceberg bucket transformation)
 * @param start_row_offset Index of the first row to hash. Defaults to 0. This
 * is useful in streaming hash join when we want to compute hashes incrementally
 * on the tables.
 *
 */
template <typename T, typename hashes_t, bool use_murmurhash = false>
    requires(!std::floating_point<T> && hashes_arr_type<hashes_t>)
static void hash_array_inner(const hashes_t& out_hashes, T* data, size_t n_rows,
                             const uint32_t seed, uint8_t* null_bitmask,
                             size_t start_row_offset = 0) {
    if (null_bitmask) {
        uint32_t na_hash;
        hash_na_val<use_murmurhash>(seed, &na_hash);
        for (size_t i = 0; i < n_rows; i++) {
            if (use_murmurhash) {
                hash_inner_murmurhash3_x86_32<T>(&data[start_row_offset + i],
                                                 seed, &out_hashes[i]);
            } else {
                hash_inner_32<T>(&data[start_row_offset + i], seed,
                                 &out_hashes[i]);
            }
            if (!GetBit(null_bitmask, start_row_offset + i)) {
                out_hashes[i] = na_hash;
            }
        }
    } else {
        for (size_t i = 0; i < n_rows; i++) {
            if (use_murmurhash) {
                hash_inner_murmurhash3_x86_32<T>(&data[start_row_offset + i],
                                                 seed, &out_hashes[i]);
            } else {
                hash_inner_32<T>(&data[start_row_offset + i], seed,
                                 &out_hashes[i]);
            }
        }
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
template <typename T, typename hashes_t, bool use_murmurhash = false>
    requires(std::floating_point<T> && hashes_arr_type<hashes_t>)
static void hash_array_inner(const hashes_t& out_hashes, T* data, size_t n_rows,
                             const uint32_t seed, uint8_t* null_bitmask,
                             size_t start_row_offset = 0) {
    if (null_bitmask) {
        uint32_t na_hash;
        hash_na_val<use_murmurhash>(seed, &na_hash);
        for (size_t i = 0; i < n_rows; i++) {
            Py_hash_t py_hash =
                Npy_HashDouble(nullptr, data[start_row_offset + i]);
            if (use_murmurhash) {
                hash_inner_murmurhash3_x86_32<Py_hash_t>(&py_hash, seed,
                                                         &out_hashes[i]);
            } else {
                hash_inner_32<Py_hash_t>(&py_hash, seed, &out_hashes[i]);
            }
            if (!GetBit(null_bitmask, start_row_offset + i)) {
                out_hashes[i] = na_hash;
            }
        }
    } else {
        for (size_t i = 0; i < n_rows; i++) {
            Py_hash_t py_hash =
                Npy_HashDouble(nullptr, data[start_row_offset + i]);
            if (use_murmurhash) {
                hash_inner_murmurhash3_x86_32<Py_hash_t>(&py_hash, seed,
                                                         &out_hashes[i]);
            } else {
                hash_inner_32<Py_hash_t>(&py_hash, seed, &out_hashes[i]);
            }
        }
    }
}

/**
 * Computation of the inner hash of the functions. This covers the Nullable
 * boolean case where 1 bit is stored for each boolean value.
 *
 * @param out_hashes: The hashes on output.
 * @param data: the list of data in input.
 * @param n_rows: Number of rows to hash starting from start_row_offset, i.e. we
 * will hash rows at indices [start_row_offset, start_row_offset + n_rows - 1].
 * @param seed: the seed of the computation.
 * @param null_bitmask: the null_bitmask of the data.
 * @param start_row_offset Index of the first row to hash. Defaults to 0. This
 * is useful in streaming hash join when we want to compute hashes incrementally
 * on the tables.
 * @tparam use_murmurhash: Use the murmurhash hash algorithm
 * (currently only used for Iceberg bucket transformation)
 *
 */
template <typename hashes_t, bool use_murmurhash = false>
    requires(hashes_arr_type<hashes_t>)
static void hash_array_inner_nullable_boolean(const hashes_t& out_hashes,
                                              uint8_t* data, size_t n_rows,
                                              const uint32_t seed,
                                              uint8_t* null_bitmask,
                                              size_t start_row_offset = 0) {
    uint32_t na_hash;
    hash_na_val<use_murmurhash>(seed, &na_hash);
    for (size_t i = 0; i < n_rows; i++) {
        bool bit = GetBit(data, start_row_offset + i);
        if (use_murmurhash) {
            hash_inner_murmurhash3_x86_32<bool>(&bit, seed, &out_hashes[i]);
        } else {
            hash_inner_32<bool>(&bit, seed, &out_hashes[i]);
        }
        if (!GetBit(null_bitmask, start_row_offset + i)) {
            out_hashes[i] = na_hash;
        }
    }
}

/**
 * Computation of the NA string hash
 * @tparam use_murmurhash: Use the murmurhash hash algorithm
 * (currently only used for Iceberg bucket transformation)
 * @param seed: the seed of the computation.
 * @param[out] hash_value: The hashes on output.
 */
template <bool use_murmurhash = false>
static void hash_na_string(const uint32_t seed, uint32_t* hash_value) {
    char val_c = 1;
    if (use_murmurhash) {
        hash_string_murmurhash3_x86_32(&val_c, 1, seed, hash_value);
    } else {
        hash_string_32(&val_c, 1, seed, hash_value);
    }
}
/**
 * Computation of the hashes for the case of strings array column. Covers STRING
 *
 * @tparam use_murmurhash: use murmurhash3_x86_32 hashes (used by Iceberg).
 * Default: false
 * @param out_hashes: The hashes on output.
 * @param data: the strings
 * @param offsets: the offsets (that separates the strings)
 * @param null_bitmap: the bitmap array for the values.
 * @param n_rows: Number of rows to hash starting from start_row_offset, i.e. we
 * will hash rows at indices [start_row_offset, start_row_offset + n_rows - 1].
 * @param seed: the seed of the computation.
 * @param is_parallel: whether we run in parallel or not.
 * @param start_row_offset Index of the first row to hash. Defaults to 0. This
 * is useful in streaming hash join when we want to compute hashes incrementally
 * on the tables.
 *
 * Right now, the bitmask is not used in the computation, which
 * may be a problem to consider later on.
 */
template <typename hashes_t, bool use_murmurhash = false>
    requires(hashes_arr_type<hashes_t>)
static void hash_array_string(const hashes_t& out_hashes, char* data,
                              offset_t* offsets, uint8_t* null_bitmask,
                              size_t n_rows, const uint32_t seed,
                              bool is_parallel, size_t start_row_offset = 0) {
    tracing::Event ev("hash_array_string", is_parallel);
    offset_t start_offset = offsets[start_row_offset];
    uint32_t na_hash;
    hash_na_string<use_murmurhash>(seed, &na_hash);
    for (size_t i = 0; i < n_rows; i++) {
        offset_t end_offset = offsets[start_row_offset + i + 1];
        offset_t len = end_offset - start_offset;
        // val is null
        if (is_na(null_bitmask, start_row_offset + i)) {
            out_hashes[i] = na_hash;
        } else {
            const char* val_chars = &data[start_offset];
            if (use_murmurhash) {
                hash_string_murmurhash3_x86_32(val_chars, (const int)len, seed,
                                               &out_hashes[i]);
            } else {
                hash_string_32(val_chars, (const int)len, seed, &out_hashes[i]);
            }
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
 * @tparam use_murmurhash: use murmurhash3_x86_32 hashes (used by Iceberg).
 * Default: false
 *
 * One approximation is the casting to char of the algorithm.
 */
template <typename T, typename hashes_t, bool use_murmurhash = false>
    requires(hashes_arr_type<hashes_t>)
void apply_arrow_offset_hash(const hashes_t& out_hashes,
                             const std::span<const offset_t> list_offsets,
                             size_t n_rows, T const& input_array) {
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        int64_t off1 = input_array->value_offset(list_offsets[i_row]);
        int64_t off2 = input_array->value_offset(list_offsets[i_row + 1]);
        char e_len = (char)(off2 - off1);
        if (use_murmurhash) {
            hash_string_murmurhash3_x86_32(&e_len, 1, out_hashes[i_row],
                                           &out_hashes[i_row]);
        } else {
            hash_string_32(&e_len, 1, out_hashes[i_row], &out_hashes[i_row]);
        }
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
template <typename T, typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void apply_arrow_bitmask_hash(const hashes_t& out_hashes,
                              const std::span<const offset_t> list_offsets,
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
template <typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void apply_arrow_string_hashes(
    const hashes_t& out_hashes, const std::span<const offset_t> list_offsets,
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
                std::string_view e_str = input_array->GetView(idx);
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
 * @param values: the list of values in templated array.
 * @param input_array: the array in input.
 */
template <typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void apply_arrow_numeric_hash(
    const hashes_t& out_hashes, const std::span<const offset_t> list_offsets,
    size_t const& n_rows,
    std::shared_ptr<arrow::PrimitiveArray> const& primitive_array) {
    std::shared_ptr<arrow::DataType> type = primitive_array->type();
    if (type->id() == arrow::Type::BOOL) {
        // Boolean arrays have 1 bit per entry so they need special handling
        uint8_t* data_ptr = (uint8_t*)primitive_array->values()->data();
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            for (offset_t idx = list_offsets[i_row];
                 idx < list_offsets[i_row + 1]; idx++) {
                bool bit = arrow::bit_util::GetBit(data_ptr, idx);
                hash_inner_32<bool>(&bit, out_hashes[i_row],
                                    &out_hashes[i_row]);
            }
        }
    } else {
        Bodo_CTypes::CTypeEnum bodo_typ = arrow_to_bodo_type(type->id());
        uint64_t siztype = numpy_item_size[bodo_typ];
        char* value_ptr = (char*)primitive_array->values()->data();
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            for (offset_t idx = list_offsets[i_row];
                 idx < list_offsets[i_row + 1]; idx++) {
                char* value_ptr_shift = value_ptr + siztype * idx;
                hash_string_32(value_ptr_shift, siztype, out_hashes[i_row],
                               &out_hashes[i_row]);
            }
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
 * @param start_row_offset: where to start hashing input_array
 * @param input_array: the input array put in argument.
 */
template <typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void hash_arrow_array(const hashes_t& out_hashes,
                      const std::span<const offset_t> list_offsets,
                      size_t const& n_rows,
                      std::shared_ptr<arrow::Array> const& input_array,
                      size_t start_row_offset) {
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
        bodo::vector<offset_t> list_offsets_out(n_rows + 1);
        for (size_t i_row = 0; i_row <= n_rows; i_row++)
            list_offsets_out[i_row] =
                list_array->value_offset(list_offsets[i_row]);
        hash_arrow_array(out_hashes, list_offsets_out, n_rows,
                         list_array->values(), start_row_offset);
        apply_arrow_bitmask_hash(out_hashes, list_offsets, n_rows, input_array);
    } else if (input_array->type_id() == arrow::Type::STRUCT) {
        auto struct_array =
            std::dynamic_pointer_cast<arrow::StructArray>(input_array);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_array->type());
        for (int i_field = 0; i_field < struct_type->num_fields(); i_field++)
            hash_arrow_array(out_hashes, list_offsets, n_rows,
                             struct_array->field(i_field), start_row_offset);
        apply_arrow_bitmask_hash(out_hashes, list_offsets, n_rows, input_array);
    } else if (input_array->type_id() == arrow::Type::MAP) {
        auto map_array =
            std::dynamic_pointer_cast<arrow::MapArray>(input_array);
        apply_arrow_offset_hash(out_hashes, list_offsets, n_rows, map_array);

        size_t nkeys = map_array->value_offset(list_offsets[n_rows]) -
                       map_array->value_offset(list_offsets[0]);
        auto sort_indices = arrow::UInt64Builder();
        auto reserve_res = sort_indices.Reserve(nkeys);
        if (!reserve_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "hash_arrow_array: Unable to reserve sort_indices");
        }

        // For every row, get the relevant slice of the child arrays, get the
        // sort indices, shift them by the row offset, and append to the sort
        // indices array.
        bodo::vector<offset_t> list_offsets_out(n_rows + 1);
        list_offsets_out[n_rows] = nkeys;
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            // Get the sort indices of the relevant slice of the keys
            offset_t row_offset = map_array->value_offset(list_offsets[i_row]);
            list_offsets_out[i_row] =
                row_offset - map_array->value_offset(start_row_offset);
            auto slice_sort_indices = get_sort_indices_of_slice_arrow(
                map_array->keys(), row_offset,
                map_array->value_offset(list_offsets[i_row + 1]));
            // Shift the sort indices by the row offset
            auto offset_slice_sort_indices_res = arrow::compute::Add(
                slice_sort_indices, arrow::Int32Scalar(row_offset));
            if (!offset_slice_sort_indices_res.ok()) [[unlikely]] {
                throw std::runtime_error(
                    "hash_arrow_array: Unable to offset sort_indices");
            }
            auto offset_slice_sort_indices =
                offset_slice_sort_indices_res.ValueOrDie().array();
            // Append the sort indices to the sort indices array
            auto append_res = sort_indices.AppendArraySlice(
                *offset_slice_sort_indices, 0, slice_sort_indices->length);
            if (!append_res.ok()) [[unlikely]] {
                throw std::runtime_error(
                    "hash_arrow_array: Unable to append sort_indices");
            }
        }
        // Finish the sort indices array
        auto sort_indices_array_res = sort_indices.Finish();
        if (!sort_indices_array_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "hash_arrow_array: Unable to finish sort_indices");
        }
        auto sort_indices_array = sort_indices_array_res.ValueOrDie();
        // Get the sorted keys
        auto sorted_keys_res =
            arrow::compute::Take(map_array->keys(), sort_indices_array);
        if (!sorted_keys_res.ok()) [[unlikely]] {
            throw std::runtime_error("hash_arrow_array: Unable to sort array");
        }
        auto sorted_keys = sorted_keys_res.ValueOrDie().make_array();
        // Get the sorted values
        auto sorted_vals_res =
            arrow::compute::Take(map_array->items(), sort_indices_array);
        if (!sorted_vals_res.ok()) [[unlikely]] {
            throw std::runtime_error("hash_arrow_array: Unable to sort array");
        }
        auto sorted_vals = sorted_vals_res.ValueOrDie().make_array();
        // Hash the sorted keys and values
        hash_arrow_array(out_hashes, list_offsets_out, n_rows, sorted_keys,
                         start_row_offset);
        hash_arrow_array(out_hashes, list_offsets_out, n_rows, sorted_vals,
                         start_row_offset);
        // Hash the null bitmask
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
    } else if (input_array->type_id() == arrow::Type::DICTIONARY) {
        // TODO: this is only needed to handle hash_array_combine incorrectly
        // handling nested types. This should be removed and proper support for
        // hashing nested arrays should be implemented instead.
        std::shared_ptr<arrow::DictionaryArray> dict_array =
            std::dynamic_pointer_cast<arrow::DictionaryArray>(input_array);
        auto dictionary = dict_array->dictionary();
        std::shared_ptr<uint32_t[]> dict_hashes =
            bodo::make_shared_arr<uint32_t>(dictionary->length(),
                                            bodo::BufferPool::DefaultPtr());
        hash_arrow_array(dict_hashes, list_offsets, n_rows, dictionary,
                         start_row_offset);

        // Define a hash value for nulls
        uint32_t null_hash = 0;
        char val_c = 1;
        hash_string_32(&val_c, 1, 0, &null_hash);

        auto indices = dict_array->indices();
        for (size_t i = 0; i < n_rows; i++) {
            if (indices->IsNull(i)) {
                out_hashes[i] = null_hash;
            } else {
                // Note that GetValueIndex is not reccomended for performant
                // code, but this codepath should be removed in general
                out_hashes[i] = dict_hashes[dict_array->GetValueIndex(i)];
            }
        }
    } else if (arrow::is_primitive(*input_array->type()) ||
               input_array->type_id() == arrow::Type::DECIMAL128) {
        // Casting DECIMAL128 Arrays to PrimitiveArray succeeds, but arrow
        // doesn't return true for is_primitive(DECIMAL128) - we might not be
        // getting a good quality hash function this way though, and we may want
        // to revisit this with a bespoke hash for decimal128.
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        apply_arrow_numeric_hash(out_hashes, list_offsets, n_rows,
                                 primitive_array);
        apply_arrow_bitmask_hash(out_hashes, list_offsets, n_rows,
                                 primitive_array);
    } else {
        throw std::runtime_error("hash_arrow_array: Unsupported array type: " +
                                 input_array->type()->ToString());
    }
}

/**
 * Top function for the computation of the hashes. It calls all the other hash
 * functions.
 *
 * @tparam use_murmurhash: use murmurhash3_x86_32 hashes (used by Iceberg).
 * Default: false
 * @param[out] out_hashes: The hashes on output.
 * @param[in] array: the list of columns in input
 * @param n_rows: Number of rows to hash starting from start_row_offset, i.e. we
 * will hash rows at indices [start_row_offset, start_row_offset + n_rows - 1].
 * @param seed: the seed of the computation.
 * @param is_parallel: whether we run in parallel or not.
 * @param global_dict_needed: this only applies to hashing of dictionary-encoded
 * arrays. This parameter specifies whether the dictionary has to be global
 * or not (for correctness or for performance -for example avoiding collisions
 * after shuffling-). This is context-dependent. This is ignored if dict_hashes
 * are provided
 * @param dict_hashes: dictionary hashes for dict-encoded string arrays. Integer
 * indices are hashed if not specified (which can cause errors if used across
 * arrays with incompatible dictionaries). Default: nullptr.
 * @param start_row_offset Index of the first row to hash. Defaults to 0. This
 * is useful in streaming hash join when we want to compute hashes incrementally
 * on the tables.
 */
template <typename hashes_t, bool use_murmurhash>
    requires(hashes_arr_type<hashes_t>)
void hash_array(const hashes_t& out_hashes, std::shared_ptr<array_info> array,
                size_t n_rows, const uint32_t seed, bool is_parallel,
                bool global_dict_needed,
                std::shared_ptr<bodo::vector<uint32_t>> dict_hashes,
                size_t start_row_offset) {
    // dispatch to proper function
    // TODO: general dispatcher
    // XXX: assumes nullable array data for nulls is always consistent
    if (array->arr_type == bodo_array_type::STRUCT ||
        array->arr_type == bodo_array_type::ARRAY_ITEM ||
        array->arr_type == bodo_array_type::MAP) {
        bodo::vector<offset_t> list_offsets(n_rows + 1);
        for (offset_t i = 0; i <= n_rows; i++)
            list_offsets[i] = i + start_row_offset;
        for (offset_t i = 0; i < n_rows; i++)
            out_hashes[i] = seed;
        if (use_murmurhash) {
            throw std::runtime_error(
                "_array_hash::hash_array: MurmurHash not supported for Arrow "
                "arrays.");
        }
        return hash_arrow_array(out_hashes, list_offsets, n_rows,
                                to_arrow(array), start_row_offset);
    }
    if (array->arr_type == bodo_array_type::STRING) {
        return hash_array_string<hashes_t, use_murmurhash>(
            out_hashes, (char*)array->data1<bodo_array_type::STRING>(),
            (offset_t*)array->data2<bodo_array_type::STRING>(),
            (uint8_t*)array->null_bitmask<bodo_array_type::STRING>(), n_rows,
            seed, is_parallel, start_row_offset);
    }
    if (array->arr_type == bodo_array_type::DICT) {
        // Use provided dictionary indices if specified, otherwise hash indices
        if (dict_hashes != nullptr) {
            if (use_murmurhash) {
                throw std::runtime_error(
                    "hash_array: use_murmurhash=true not supported when "
                    "dictionary hashes are provided");
            }
            uint32_t na_hash;
            hash_na_val<use_murmurhash>(seed, &na_hash);
            uint8_t* null_bitmask =
                (uint8_t*)array->child_arrays[1]
                    ->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
            dict_indices_t* dict_inds =
                (dict_indices_t*)array->child_arrays[1]
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL>();
            for (size_t i = 0; i < n_rows; i++) {
                out_hashes[i] =
                    GetBit(null_bitmask, start_row_offset + i)
                        ? (*dict_hashes)[dict_inds[start_row_offset + i]]
                        : na_hash;
            }
            return;
        }
        std::shared_ptr<array_info>& dict = array->child_arrays[0];
        if ((dict->is_globally_replicated && dict->is_locally_unique) ||
            !is_parallel || !global_dict_needed) {
            // in this case we can just hash the indices since the dictionary is
            // synchronized across ranks or is only needed for a local
            // operation where hashing based on local dictionary won't affect
            // correctness or performance
            return hash_array_inner<dict_indices_t, hashes_t, use_murmurhash>(
                out_hashes,
                (dict_indices_t*)array->child_arrays[1]
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                n_rows, seed,
                (uint8_t*)array->child_arrays[1]
                    ->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
                start_row_offset);
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
    if (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
        array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array_inner_nullable_boolean<hashes_t, use_murmurhash>(
            out_hashes,
            (uint8_t*)array->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
            n_rows, seed,
            (uint8_t*)array->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
            start_row_offset);
    }

    if (array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array_inner<bool, hashes_t, use_murmurhash>(
            out_hashes, (bool*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::INT8) {
        return hash_array_inner<int8_t, hashes_t, use_murmurhash>(
            out_hashes, (int8_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::UINT8) {
        return hash_array_inner<uint8_t, hashes_t, use_murmurhash>(
            out_hashes, (uint8_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::INT16) {
        return hash_array_inner<int16_t, hashes_t, use_murmurhash>(
            out_hashes, (int16_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::UINT16) {
        return hash_array_inner<uint16_t, hashes_t, use_murmurhash>(
            out_hashes, (uint16_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::INT32 ||
        array->dtype == Bodo_CTypes::DATE) {
        return hash_array_inner<int32_t, hashes_t, use_murmurhash>(
            out_hashes, (int32_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::UINT32) {
        return hash_array_inner<uint32_t, hashes_t, use_murmurhash>(
            out_hashes, (uint32_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::INT64) {
        return hash_array_inner<int64_t, hashes_t, use_murmurhash>(
            out_hashes, (int64_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::DECIMAL) {
        return hash_array_inner<__int128, hashes_t, use_murmurhash>(
            out_hashes, (__int128*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::UINT64) {
        return hash_array_inner<uint64_t, hashes_t, use_murmurhash>(
            out_hashes, (uint64_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    // TODO: [BE-4106] Split Time into Time32 and Time64
    // NOTE: TimestampTZ hash only on data1 (ignores the offset buffer since
    // values in data1 is already normalized in UTC values)
    if (array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIME ||
        array->dtype == Bodo_CTypes::TIMESTAMPTZ ||
        array->dtype == Bodo_CTypes::TIMEDELTA) {
        return hash_array_inner<int64_t, hashes_t, use_murmurhash>(
            out_hashes, (int64_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::FLOAT32) {
        return hash_array_inner<float, hashes_t, use_murmurhash>(
            out_hashes, (float*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::FLOAT64) {
        return hash_array_inner<double, hashes_t, use_murmurhash>(
            out_hashes, (double*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid data type for hash");
}

// Explicitly initialize the required templates for loader to be able to
// find them statically.
template void hash_array<std::unique_ptr<uint32_t[]>, true>(
    const std::unique_ptr<uint32_t[]>& out_hashes,
    std::shared_ptr<array_info> array, size_t n_rows, const uint32_t seed,
    bool is_parallel, bool global_dict_needed,
    std::shared_ptr<bodo::vector<uint32_t>> dict_hashes,
    size_t start_row_offset);

template void hash_array<std::shared_ptr<uint32_t[]>, true>(
    const std::shared_ptr<uint32_t[]>& out_hashes,
    std::shared_ptr<array_info> array, size_t n_rows, const uint32_t seed,
    bool is_parallel, bool global_dict_needed,
    std::shared_ptr<bodo::vector<uint32_t>> dict_hashes,
    size_t start_row_offset);

template void hash_array<std::unique_ptr<uint32_t[]>, false>(
    const std::unique_ptr<uint32_t[]>& out_hashes,
    std::shared_ptr<array_info> array, size_t n_rows, const uint32_t seed,
    bool is_parallel, bool global_dict_needed,
    std::shared_ptr<bodo::vector<uint32_t>> dict_hashes,
    size_t start_row_offset);

template void hash_array<std::shared_ptr<uint32_t[]>, false>(
    const std::shared_ptr<uint32_t[]>& out_hashes,
    std::shared_ptr<array_info> array, size_t n_rows, const uint32_t seed,
    bool is_parallel, bool global_dict_needed,
    std::shared_ptr<bodo::vector<uint32_t>> dict_hashes,
    size_t start_row_offset);

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

template <class T, typename hashes_t>
    requires(hashes_arr_type<hashes_t> && !std::floating_point<T>)
static void hash_array_combine_inner(const hashes_t& out_hashes, T* data,
                                     size_t n_rows, const uint32_t seed,
                                     uint8_t* null_bitmask,
                                     size_t start_row_offset = 0) {
    if (null_bitmask) {
        uint32_t na_hash;
        hash_na_val(seed, &na_hash);
        uint32_t out_hash = 0;
        for (size_t i = 0; i < n_rows; i++) {
            if (!GetBit(null_bitmask, start_row_offset + i)) {
                out_hash = na_hash;
            } else {
                hash_inner_32<T>(&data[start_row_offset + i], seed, &out_hash);
            }
            hash_combine_boost(out_hashes[i], out_hash);
        }
    } else {
        uint32_t out_hash = 0;
        for (size_t i = 0; i < n_rows; i++) {
            hash_inner_32<T>(&data[start_row_offset + i], seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    }
}

// Discussion on hashing floats:
// https://stackoverflow.com/questions/4238122/hash-function-for-floats

template <class T, typename hashes_t>
    requires(hashes_arr_type<hashes_t> && std::floating_point<T>)
static void hash_array_combine_inner(const hashes_t& out_hashes, T* data,
                                     size_t n_rows, const uint32_t seed,
                                     uint8_t* null_bitmask,
                                     size_t start_row_offset = 0) {
    if (null_bitmask) {
        uint32_t na_hash;
        hash_na_val(seed, &na_hash);
        uint32_t out_hash = 0;
        for (size_t i = 0; i < n_rows; i++) {
            if (!GetBit(null_bitmask, start_row_offset + i)) {
                out_hash = na_hash;
            } else {
                Py_hash_t py_hash =
                    Npy_HashDouble(nullptr, data[start_row_offset + i]);
                hash_inner_32<Py_hash_t>(&py_hash, seed, &out_hash);
            }
            hash_combine_boost(out_hashes[i], out_hash);
        }
    } else {
        uint32_t out_hash = 0;
        for (size_t i = 0; i < n_rows; i++) {
            Py_hash_t py_hash =
                Npy_HashDouble(nullptr, data[start_row_offset + i]);
            hash_inner_32<Py_hash_t>(&py_hash, seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    }
}

/**
 * Computation of the inner hash combine function. This covers the Nullable
 * boolean case where 1 bit is stored for each boolean value.
 *
 * @param out_hashes: The hashes on output.
 * @param data: the list of data in input.
 * @param n_rows: Number of rows to hash starting from start_row_offset, i.e. we
 * will hash rows at indices [start_row_offset, start_row_offset + n_rows - 1].
 * @param seed: the seed of the computation.
 * @param null_bitmask: the null_bitmask of the data.
 * @param start_row_offset Index of the first row to hash. Defaults to 0. This
 * is useful in streaming hash join when we want to compute hashes incrementally
 * on the tables.
 *
 */
template <typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
static void hash_array_combine_inner_nullable_boolean(
    const hashes_t& out_hashes, uint8_t* data, size_t n_rows,
    const uint32_t seed, uint8_t* null_bitmask, size_t start_row_offset = 0) {
    uint32_t na_hash;
    hash_na_val(seed, &na_hash);
    uint32_t out_hash = 0;
    for (size_t i = 0; i < n_rows; i++) {
        if (!GetBit(null_bitmask, start_row_offset + i)) {
            out_hash = na_hash;
        } else {
            bool bit = GetBit(data, start_row_offset + i);
            hash_inner_32<bool>(&bit, seed, &out_hash);
        }
        hash_combine_boost(out_hashes[i], out_hash);
    }
}

template <typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
static void hash_array_combine_string(const hashes_t& out_hashes, char* data,
                                      offset_t* offsets, uint8_t* null_bitmask,
                                      size_t n_rows, const uint32_t seed,
                                      size_t start_row_offset = 0) {
    offset_t start_offset = offsets[start_row_offset];
    uint32_t na_hash;
    hash_na_string(seed, &na_hash);
    for (size_t i = 0; i < n_rows; i++) {
        offset_t end_offset = offsets[start_row_offset + i + 1];
        offset_t len = end_offset - start_offset;

        uint32_t out_hash = 0;
        if (is_na(null_bitmask, start_row_offset + i)) {
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
template <typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void hash_array_combine(const hashes_t& out_hashes,
                        std::shared_ptr<array_info> array, size_t n_rows,
                        const uint32_t seed, bool global_dict_needed,
                        bool is_parallel,
                        std::shared_ptr<bodo::vector<uint32_t>> dict_hashes,
                        size_t start_row_offset) {
    // dispatch to proper function
    // TODO: general dispatcher
    if (array->arr_type == bodo_array_type::STRUCT ||
        array->arr_type == bodo_array_type::ARRAY_ITEM ||
        array->arr_type == bodo_array_type::MAP) {
        // TODO: stop using arrow hash
        bodo::vector<offset_t> list_offsets(n_rows + 1);
        for (offset_t i = 0; i <= n_rows; i++) {
            list_offsets[i] = i + start_row_offset;
        }
        return hash_arrow_array(out_hashes, list_offsets, n_rows,
                                to_arrow(array), start_row_offset);
    }
    if (array->arr_type == bodo_array_type::STRING) {
        return hash_array_combine_string(
            out_hashes, (char*)array->data1<bodo_array_type::STRING>(),
            (offset_t*)array->data2<bodo_array_type::STRING>(),
            (uint8_t*)array->null_bitmask<bodo_array_type::STRING>(), n_rows,
            seed, start_row_offset);
    }
    if (array->arr_type == bodo_array_type::DICT) {
        // Use provided dictionary indices if specified, otherwise hash indices
        if (dict_hashes != nullptr) {
            uint32_t na_hash;
            hash_na_val(seed, &na_hash);
            uint8_t* null_bitmask =
                (uint8_t*)array->child_arrays[1]
                    ->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
            dict_indices_t* dict_inds =
                (dict_indices_t*)array->child_arrays[1]
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL>();

            uint32_t out_hash = 0;
            for (size_t i = 0; i < n_rows; i++) {
                if (!GetBit(null_bitmask, start_row_offset + i)) {
                    out_hash = na_hash;
                } else {
                    out_hash = (*dict_hashes)[dict_inds[start_row_offset + i]];
                }
                hash_combine_boost(out_hashes[i], out_hash);
            }
            return;
        }
        std::shared_ptr<array_info>& dict = array->child_arrays[0];
        if ((dict->is_globally_replicated && dict->is_locally_unique) ||
            !global_dict_needed || !is_parallel) {
            // in this case we can just hash the indices since the dictionary is
            // synchronized across ranks or is only needed for a local
            // operation where hashing based on local dictionary won't affect
            // correctness or performance
            return hash_array_combine_inner<dict_indices_t>(
                out_hashes,
                (dict_indices_t*)array->child_arrays[1]
                    ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                n_rows, seed,
                (uint8_t*)array->null_bitmask<bodo_array_type::DICT>(),
                start_row_offset);
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
    if (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
        array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array_combine_inner_nullable_boolean(
            out_hashes,
            (uint8_t*)array->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
            n_rows, seed,
            (uint8_t*)array->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
            start_row_offset);
    }

    if (array->dtype == Bodo_CTypes::_BOOL) {
        return hash_array_combine_inner<bool>(
            out_hashes, (bool*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::INT8) {
        return hash_array_combine_inner<int8_t>(
            out_hashes, (int8_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::UINT8) {
        return hash_array_combine_inner<uint8_t>(
            out_hashes, (uint8_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::INT16) {
        return hash_array_combine_inner<int16_t>(
            out_hashes, (int16_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::UINT16) {
        return hash_array_combine_inner<uint16_t>(
            out_hashes, (uint16_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::INT32 ||
        array->dtype == Bodo_CTypes::DATE) {
        return hash_array_combine_inner<int32_t>(
            out_hashes, (int32_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::UINT32) {
        return hash_array_combine_inner<uint32_t>(
            out_hashes, (uint32_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::INT64) {
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::UINT64) {
        return hash_array_combine_inner<uint64_t>(
            out_hashes, (uint64_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    // TODO: [BE-4106] Split Time into Time32 and Time64
    // NOTE: TimestampTZ hash only on data1 (ignores the offset buffer since
    // values in data1 is already normalized in UTC values)
    if (array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIME ||
        array->dtype == Bodo_CTypes::TIMESTAMPTZ ||
        array->dtype == Bodo_CTypes::TIMEDELTA) {
        return hash_array_combine_inner<int64_t>(
            out_hashes, (int64_t*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::FLOAT32) {
        return hash_array_combine_inner<float>(
            out_hashes, (float*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::FLOAT64) {
        return hash_array_combine_inner<double>(
            out_hashes, (double*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    if (array->dtype == Bodo_CTypes::DECIMAL ||
        array->dtype == Bodo_CTypes::INT128) {
        return hash_array_combine_inner<__int128>(
            out_hashes, (__int128*)array->data1(), n_rows, seed,
            (uint8_t*)array->null_bitmask(), start_row_offset);
    }
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "Invalid data type for hash combine");
}

// Explicitly initialize the required templates for loader to be able to
// find them statically.
template void hash_array_combine<std::unique_ptr<uint32_t[]>>(
    const std::unique_ptr<uint32_t[]>& out_hashes,
    std::shared_ptr<array_info> array, size_t n_rows, const uint32_t seed,
    bool global_dict_needed, bool is_parallel,
    std::shared_ptr<bodo::vector<uint32_t>> dict_hashes,
    size_t start_row_offset);

template void hash_array_combine<std::shared_ptr<uint32_t[]>>(
    const std::shared_ptr<uint32_t[]>& out_hashes,
    std::shared_ptr<array_info> array, size_t n_rows, const uint32_t seed,
    bool global_dict_needed, bool is_parallel,
    std::shared_ptr<bodo::vector<uint32_t>> dict_hashes,
    size_t start_row_offset);

template <typename T>
    requires std::floating_point<T>
double get_value(T val) {
    // I wrote that because I am not sure nan have unique
    // binary representation
    if (isnan(val)) {
        return std::nan("");
    }
    return val;
}

template <typename T>
    requires(!std::floating_point<T>)
double get_value(T val) {
    return val;
}

template <typename T, typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void coherent_hash_array_inner_uint64(const hashes_t& out_hashes,
                                      std::shared_ptr<array_info> array,
                                      size_t n_rows, const uint32_t seed) {
    T* data = (T*)array->data1();
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            uint64_t val = data[i];
            hash_inner_32<uint64_t>(&val, seed, &out_hashes[i]);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask();
        uint32_t na_hash;
        hash_na_val(seed, &na_hash);
        for (size_t i = 0; i < n_rows; i++) {
            uint64_t val = data[i];
            hash_inner_32<uint64_t>(&val, seed, &out_hashes[i]);
            if (!GetBit(null_bitmask, i)) {
                out_hashes[i] = na_hash;
            }
        }
    }
}

template <typename T, typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void coherent_hash_array_inner_int64(const hashes_t& out_hashes,
                                     std::shared_ptr<array_info> array,
                                     size_t n_rows, const uint32_t seed) {
    T* data = (T*)array->data1();
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            int64_t val = data[i];
            hash_inner_32<int64_t>(&val, seed, &out_hashes[i]);
            // For numpy, all entries are true, no need to increment.
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask();
        uint32_t na_hash;
        hash_na_val(seed, &na_hash);
        for (size_t i = 0; i < n_rows; i++) {
            int64_t val = data[i];
            hash_inner_32<int64_t>(&val, seed, &out_hashes[i]);
            if (!GetBit(null_bitmask, i)) {
                out_hashes[i] = na_hash;
            }
        }
    }
}

template <typename T, typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void coherent_hash_array_inner_double(const hashes_t& out_hashes,
                                      std::shared_ptr<array_info> array,
                                      size_t n_rows, const uint32_t seed) {
    T* data = (T*)array->data1();
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            double val = get_value(data[i]);
            hash_inner_32<double>(&val, seed, &out_hashes[i]);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask();
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(null_bitmask, i);
            double val;
            if (bit) {
                val = get_value(data[i]);
            } else {
                val = std::nan("");
            }
            hash_inner_32<double>(&val, seed, &out_hashes[i]);
        }
    }
}

template <typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void coherent_hash_array(const hashes_t& out_hashes,
                         std::shared_ptr<array_info> array,
                         std::shared_ptr<array_info> ref_array, size_t n_rows,
                         const uint32_t seed, bool is_parallel = true) {
    if ((array->arr_type == bodo_array_type::DICT) &&
        !is_matching_dictionary(array->child_arrays[0],
                                ref_array->child_arrays[0])) {
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
        // enforced by the is_locally_unique check in
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
    if (array->arr_type == bodo_array_type::STRUCT ||
        array->arr_type == bodo_array_type::ARRAY_ITEM ||
        array->arr_type == bodo_array_type::MAP ||
        array->arr_type == bodo_array_type::STRING) {
        return hash_array(out_hashes, array, n_rows, seed, is_parallel, true);
    }
    // Now we are in NUMPY / NULLABLE_INT_BOOL. Getting into hot waters.
    // For DATE / TIME / DATETIME / TIMEDELTA / TIMESTAMPTZ / DECIMAL no type
    // conversion is allowed
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::TIME ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA ||
        array->arr_type == bodo_array_type::TIMESTAMPTZ ||
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

template <typename T, typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void coherent_hash_array_combine_inner_uint64(const hashes_t& out_hashes,
                                              std::shared_ptr<array_info> array,
                                              size_t n_rows,
                                              const uint32_t seed) {
    T* data = (T*)array->data1();
    uint32_t out_hash;
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            uint64_t val = data[i];
            hash_inner_32<uint64_t>(&val, seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask();
        for (size_t i = 0; i < n_rows; i++) {
            uint64_t val = data[i];
            hash_inner_32<uint64_t>(&val, seed, &out_hash);
            if (!GetBit(null_bitmask, i)) {
                out_hash++;
            }
            hash_combine_boost(out_hashes[i], out_hash);
        }
    }
}

template <typename T, typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void coherent_hash_array_combine_inner_int64(const hashes_t& out_hashes,
                                             std::shared_ptr<array_info> array,
                                             size_t n_rows,
                                             const uint32_t seed) {
    T* data = (T*)array->data1();
    uint32_t out_hash;
    if (array->arr_type == bodo_array_type::NUMPY) {
        for (size_t i = 0; i < n_rows; i++) {
            int64_t val = data[i];
            hash_inner_32<int64_t>(&val, seed, &out_hash);
            // For numpy, all entries are true, no need to increment.
            hash_combine_boost(out_hashes[i], out_hash);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask();
        for (size_t i = 0; i < n_rows; i++) {
            int64_t val = data[i];
            hash_inner_32<int64_t>(&val, seed, &out_hash);
            if (!GetBit(null_bitmask, i)) {
                out_hash++;
            }
            hash_combine_boost(out_hashes[i], out_hash);
        }
    }
}

template <typename T, typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void coherent_hash_array_combine_inner_double(const hashes_t& out_hashes,
                                              std::shared_ptr<array_info> array,
                                              size_t n_rows,
                                              const uint32_t seed) {
    T* data = (T*)array->data1();
    uint32_t out_hash;
    if (array->arr_type == bodo_array_type::NUMPY) {
        uint32_t out_hash;
        for (size_t i = 0; i < n_rows; i++) {
            double val = get_value(data[i]);
            hash_inner_32<double>(&val, seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    } else {  // We are in NULLABLE_INT_BOOL
        uint8_t* null_bitmask = (uint8_t*)array->null_bitmask();
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(null_bitmask, i);
            double val;
            if (bit) {
                val = get_value(data[i]);
            } else {
                val = std::nan("");
            }
            hash_inner_32<double>(&val, seed, &out_hash);
            hash_combine_boost(out_hashes[i], out_hash);
        }
    }
}

template <typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void coherent_hash_array_combine(const hashes_t& out_hashes,
                                 std::shared_ptr<array_info> array,
                                 std::shared_ptr<array_info> ref_array,
                                 size_t n_rows, const uint32_t seed,
                                 bool is_parallel) {
    // For those types, no type conversion is ever needed.
    if (array->arr_type == bodo_array_type::STRUCT ||
        array->arr_type == bodo_array_type::ARRAY_ITEM ||
        array->arr_type == bodo_array_type::MAP ||
        array->arr_type == bodo_array_type::STRING) {
        return hash_array_combine(out_hashes, array, n_rows, seed, true,
                                  is_parallel);
    }
    // Now we are in NUMPY / NULLABLE_INT_BOOL. Getting into hot waters.
    // For DATE / DATETIME / TIMEDELTA / TIMESTAMPTZ/ DECIMAL no type conversion
    // is allowed
    if (array->dtype == Bodo_CTypes::DATE ||
        array->dtype == Bodo_CTypes::DATETIME ||
        array->dtype == Bodo_CTypes::TIMEDELTA ||
        array->arr_type == bodo_array_type::TIMESTAMPTZ ||
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
std::unique_ptr<uint32_t[]> coherent_hash_keys(
    std::vector<std::shared_ptr<array_info>> const& key_arrs,
    std::vector<std::shared_ptr<array_info>> const& ref_key_arrs,
    const uint32_t seed, bool is_parallel) {
    tracing::Event ev("coherent_hash_keys", is_parallel);
    size_t n_rows = (size_t)key_arrs[0]->length;
    std::unique_ptr<uint32_t[]> hashes = std::make_unique<uint32_t[]>(n_rows);
    coherent_hash_array(hashes, key_arrs[0], ref_key_arrs[0], n_rows, seed,
                        is_parallel);
    for (size_t i = 1; i < key_arrs.size(); i++) {
        coherent_hash_array_combine(hashes, key_arrs[i], ref_key_arrs[i],
                                    n_rows, seed, is_parallel);
    }
    return hashes;
}

template <typename hashes_t>
    requires(hashes_arr_type<hashes_t>)
void hash_keys(
    const hashes_t& out_hashes,
    std::vector<std::shared_ptr<array_info>> const& key_arrs,
    const uint32_t seed, bool is_parallel, bool global_dict_needed,
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes,
    size_t start_row_offset, int64_t n_rows_) {
    tracing::Event ev("hash_keys", is_parallel);
    size_t n_rows = key_arrs[0]->length - start_row_offset;
    if (n_rows_ != -1) {
        n_rows = std::min<size_t>(n_rows_, n_rows);
    }
    // hash first array
    hash_array(
        out_hashes, key_arrs[0], n_rows, seed, is_parallel, global_dict_needed,
        dict_hashes == nullptr ? nullptr : (*dict_hashes)[0], start_row_offset);
    // combine other array out_hashes
    for (size_t i = 1; i < key_arrs.size(); i++) {
        hash_array_combine(out_hashes, key_arrs[i], n_rows, seed,
                           global_dict_needed, is_parallel,
                           dict_hashes == nullptr ? nullptr : (*dict_hashes)[i],
                           start_row_offset);
    }
}

// Explicitly initialize the required templates for loader to be able to
// find them statically.
template void hash_keys<std::unique_ptr<uint32_t[]>>(
    const std::unique_ptr<uint32_t[]>& out_hashes,
    std::vector<std::shared_ptr<array_info>> const& key_arrs,
    const uint32_t seed, bool is_parallel, bool global_dict_needed,
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes,
    size_t start_row_offset, int64_t n_rows_);

template void hash_keys<std::shared_ptr<uint32_t[]>>(
    const std::shared_ptr<uint32_t[]>& out_hashes,
    std::vector<std::shared_ptr<array_info>> const& key_arrs,
    const uint32_t seed, bool is_parallel, bool global_dict_needed,
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes,
    size_t start_row_offset, int64_t n_rows_);

std::unique_ptr<uint32_t[]> hash_keys(
    std::vector<std::shared_ptr<array_info>> const& key_arrs,
    const uint32_t seed, bool is_parallel, bool global_dict_needed,
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
        dict_hashes,
    size_t start_row_offset, int64_t n_rows_) {
    size_t n_rows = key_arrs[0]->length - start_row_offset;
    if (n_rows_ != -1) {
        n_rows = std::min<size_t>(n_rows_, n_rows);
    }
    std::unique_ptr<uint32_t[]> out_hashes =
        std::make_unique<uint32_t[]>(n_rows);
    hash_keys(out_hashes, key_arrs, seed, is_parallel, global_dict_needed,
              dict_hashes, start_row_offset, n_rows);
    return out_hashes;
}

/**
 * @brief Verify that the dictionary arrays attempted to be unified have
 * satisfied the requirements for unification.
 *
 * @param arrs The arrays to unify.
 * @param is_parallels If each array is parallel.
 */
void ensure_dicts_can_unify(std::vector<std::shared_ptr<array_info>>& arrs,
                            std::vector<bool>& is_parallels) {
    for (size_t i = 0; i < arrs.size(); i++) {
        std::shared_ptr<array_info>& dict = arrs[i]->child_arrays[0];
        if (is_parallels[i] && !dict->is_globally_replicated) {
            throw std::runtime_error(
                "unify_dictionaries: array does not have global dictionary");
        }
        if (!dict->is_locally_unique) {
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
 * @return bodo::unord_map_container<std::pair<size_t, size_t>, dict_indices_t>*
 * A pointer to the heap allocated hashmap.
 */
bodo::unord_map_container<std::pair<size_t, size_t>, dict_indices_t,
                          HashMultiArray, MultiArrayInfoEqual>*
create_several_array_hashmap(
    std::vector<std::shared_ptr<array_info>>& arrs,
    std::vector<std::shared_ptr<uint32_t[]>>& hashes,
    std::vector<std::shared_ptr<array_info>>& stored_arrs) {
    // hash map mapping dictionary values of arr1 and arr2 to index in unified
    // dictionary
    HashMultiArray hash_fct{hashes};
    MultiArrayInfoEqual equal_fct{stored_arrs};
    auto* dict_value_to_unified_index =
        new bodo::unord_map_container<std::pair<size_t, size_t>, dict_indices_t,
                                      HashMultiArray, MultiArrayInfoEqual>(
            {}, hash_fct, equal_fct);
    // Estimate how much to reserve. We could get an accurate
    // estimate with hyperloglog but it seems unnecessary for this use case.
    // For now we reserve initial capacity as the max size of any of the
    // dictionaries.
    bodo::vector<size_t> lengths(arrs.size());
    for (size_t i = 0; i < arrs.size(); i++) {
        lengths[i] = arrs[i]->child_arrays[0]->length;
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
    bodo::unord_map_container<std::pair<size_t, size_t>, dict_indices_t,
                              HashMultiArray, MultiArrayInfoEqual>*
        dict_value_to_unified_index,
    std::vector<std::shared_ptr<uint32_t[]>>& hashes,
    std::vector<std::shared_ptr<array_info>>& stored_arrs,
    std::shared_ptr<array_info> dict, offset_t const* const offsets,
    const size_t len, dict_indices_t& next_index, size_t& n_chars,
    const uint32_t hash_seed) {
    std::unique_ptr<uint32_t[]> arr_hashes = std::make_unique<uint32_t[]>(len);
    hash_array(arr_hashes, dict, len, hash_seed, false,
               /*global_dict_needed=*/false);
    // Insert the hashes and the array
    hashes.push_back(std::move(arr_hashes));
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
template <typename Alloc>
void insert_new_dict_to_multiarray_hashmap(
    bodo::unord_map_container<std::pair<size_t, size_t>, dict_indices_t,
                              HashMultiArray, MultiArrayInfoEqual>*
        dict_value_to_unified_index,
    std::vector<std::shared_ptr<uint32_t[]>>& hashes,
    std::vector<std::shared_ptr<array_info>>& stored_arrs,
    std::vector<dict_indices_t, Alloc>& arr_index_map,
    std::vector<bodo::vector<dict_indices_t>*>& unique_indices_all_arrs,
    std::shared_ptr<array_info> dict, offset_t const* const offsets,
    const size_t len, dict_indices_t& next_index, size_t& n_chars,
    size_t arr_num, const uint32_t hash_seed) {
    std::unique_ptr<uint32_t[]> arr_hashes = std::make_unique<uint32_t[]>(len);
    hash_array(arr_hashes, dict, len, hash_seed, false,
               /*global_dict_needed=*/false);
    // Insert the hashes and the array
    hashes.push_back(std::move(arr_hashes));
    stored_arrs.push_back(dict);
    // Create a vector to store the indices of the unique strings in the
    // current arr.
    auto* unique_indices = new bodo::vector<dict_indices_t>();
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
void replace_dict_arr_indices(
    std::shared_ptr<array_info> arr,
    const std::span<const dict_indices_t> arr_index_map) {
    // Update the indices for this array. If there is only one reference to
    // the dict_array remaining we can update the array inplace without
    // allocating a new array.
    bool inplace = (arr->child_arrays[1]->buffers[0]->getMeminfo()->refct == 1);
    if (!inplace) {
        std::shared_ptr<array_info> indices = copy_array(arr->child_arrays[1]);
        arr->child_arrays[1] = indices;
    }

    uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask();

    for (size_t j = 0; j < arr->child_arrays[1]->length; j++) {
        if (GetBit(null_bitmask, j)) {
            dict_indices_t& index =
                arr->child_arrays[1]
                    ->at<dict_indices_t, bodo_array_type::NULLABLE_INT_BOOL>(j);
            index = arr_index_map[index];
        }
    }
}

void unify_several_dictionaries(std::vector<std::shared_ptr<array_info>>& arrs,
                                std::vector<bool>& is_parallels) {
    // Validate the inputs
    ensure_dicts_can_unify(arrs, is_parallels);
    // Keep a vector of hashes for each array. That will be checked. We will
    // update this dynamically to avoid need to constantly rehash/update the
    // array.
    std::vector<std::shared_ptr<uint32_t[]>> hashes;
    // Keep a vector of array infos
    std::vector<std::shared_ptr<array_info>> stored_arrs;
    // Create the hash table. We will dynamically fill the vector of
    // hashes and dictionaries as we go.
    const uint32_t hash_seed = SEED_HASH_JOIN;
    auto* dict_value_to_unified_index =
        create_several_array_hashmap(arrs, hashes, stored_arrs);

    // The first dictionary will always be entirely included in the output
    // unified dictionary.
    std::shared_ptr<array_info> base_dict = arrs[0]->child_arrays[0];
    const size_t base_len = static_cast<size_t>(base_dict->length);
    offset_t const* const base_offsets =
        (offset_t*)base_dict->data2<bodo_array_type::STRING>();
    bool added_first = false;
    size_t arr_num = 1;
    size_t n_chars = 0;
    dict_indices_t next_index = 1;

    // Keep track of the unique indices for each array. We will use this to
    // build the final dictionary. We will omit the base dictionary.
    std::vector<bodo::vector<dict_indices_t>*> unique_indices_all_arrs;

    for (size_t i = 1; i < arrs.size(); i++) {
        // Process the dictionaries 1 at a time. To do this we always insert
        // any new dictionary entries in order by array (first all of arr1,
        // then anything new from arr2, etc). As a result, this means that the
        // entries in arr{i} can never modify the indices of arr{i-1}.
        std::shared_ptr<array_info> curr_arr = arrs[i];
        std::shared_ptr<array_info> curr_dict = curr_arr->child_arrays[0];
        offset_t const* const curr_dict_offsets =
            (offset_t*)curr_dict->data2<bodo_array_type::STRING>();

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
        bodo::vector<dict_indices_t> arr_index_map(curr_len);

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
        hashes[i].reset();
    }

    // Now that we have all of the dictionary elements we can create the
    // dictionary. The next_index is always num_strings + 1, so we can use that
    // to get the length of the dictionary.
    size_t n_strings = next_index - 1;
    // ensure_dicts_can_unify requires each array be globally replicated.
    // This is either because the whole array is replicated or the dictionary
    // is.
    bool is_globally_replicated = true;
    // ensure_dicts_can_unify requires each array is unique.
    bool is_locally_unique = true;
    std::shared_ptr<array_info> new_dict =
        alloc_string_array(Bodo_CTypes::STRING, n_strings, n_chars, -1, 0,
                           is_globally_replicated, is_locally_unique);
    offset_t* new_dict_str_offsets =
        (offset_t*)new_dict->data2<bodo_array_type::STRING>();

    // Initialize the offset and string index to the end of the base dictionary
    offset_t cur_offset = base_offsets[base_len];
    int64_t cur_offset_idx = base_len + 1;

    // copy offsets from arr1 into new_dict_str_offsets
    memcpy(new_dict_str_offsets, base_offsets,
           cur_offset_idx * sizeof(offset_t));
    // copy strings from arr1 into new_dict
    memcpy(new_dict->data1<bodo_array_type::STRING>(),
           base_dict->data1<bodo_array_type::STRING>(), cur_offset);
    for (size_t i = 0; i < unique_indices_all_arrs.size(); i++) {
        bodo::vector<dict_indices_t>*& arr_unique_indices =
            unique_indices_all_arrs[i];
        // Load the relevant array. This is the i+1 array we stored for the
        // hashmap because we skip the base array.
        std::shared_ptr<array_info> dict_arr = stored_arrs[i + 1];
        assert(dict_arr->arr_type == bodo_array_type::STRING);
        offset_t const* const arr_offsets =
            (offset_t*)dict_arr->data2<bodo_array_type::STRING>();

        for (dict_indices_t j : *arr_unique_indices) {
            offset_t str_len = arr_offsets[j + 1] - arr_offsets[j];
            memcpy(new_dict->data1<bodo_array_type::STRING>() + cur_offset,
                   dict_arr->data1<bodo_array_type::STRING>() + arr_offsets[j],
                   str_len);
            cur_offset += str_len;
            new_dict_str_offsets[cur_offset_idx++] = cur_offset;
        }
        delete arr_unique_indices;
    }

    // replace old dictionaries with a new one
    for (size_t i = 0; i < arrs.size(); i++) {
        arrs[i]->child_arrays[0] = new_dict;
    }
}

void unify_dictionaries(std::shared_ptr<array_info> arr1,
                        std::shared_ptr<array_info> arr2, bool arr1_is_parallel,
                        bool arr2_is_parallel) {
    // Validate the inputs
    std::vector<std::shared_ptr<array_info>> arrs = {arr1, arr2};
    std::vector<bool> is_parallel = {arr1_is_parallel, arr2_is_parallel};
    ensure_dicts_can_unify(arrs, is_parallel);

    if (is_matching_dictionary(arr1->child_arrays[0], arr2->child_arrays[0])) {
        return;  // dictionaries are the same
    }

    // Note we insert the dictionaries in order (arr1 then arr2). Since we have
    // ensured there are no duplicates this means that only the indices in arr2
    // can change and the entire dictionary in arr1 will be in the unified dict.

    const size_t arr1_dictionary_len =
        static_cast<size_t>(arr1->child_arrays[0]->length);
    const size_t arr2_dictionary_len =
        static_cast<size_t>(arr2->child_arrays[0]->length);
    // this vector will be used to map old indices to new ones
    bodo::vector<dict_indices_t> arr2_index_map(arr2_dictionary_len);

    const uint32_t hash_seed = SEED_HASH_JOIN;
    std::unique_ptr<uint32_t[]> arr1_hashes =
        std::make_unique<uint32_t[]>(arr1_dictionary_len);
    std::unique_ptr<uint32_t[]> arr2_hashes =
        std::make_unique<uint32_t[]>(arr2_dictionary_len);
    hash_array(arr1_hashes, arr1->child_arrays[0], arr1_dictionary_len,
               hash_seed, false,
               /*global_dict_needed=*/false);
    hash_array(arr2_hashes, arr2->child_arrays[0], arr2_dictionary_len,
               hash_seed, false,
               /*global_dict_needed=*/false);

    // hash map mapping dictionary values of arr1 and arr2 to index in unified
    // dictionary
    HashDict hash_fct{arr1_dictionary_len, arr1_hashes, arr2_hashes};
    KeyEqualDict equal_fct{arr1_dictionary_len, arr1->child_arrays[0],
                           arr2->child_arrays[0] /*, is_na_equal*/};
    auto* dict_value_to_unified_index =
        new bodo::unord_map_container<size_t, dict_indices_t, HashDict,
                                      KeyEqualDict>({}, hash_fct, equal_fct);
    // Size of new dictionary could end up as large as
    // arr1_dictionary_len + arr2_dictionary_len. We could get an accurate
    // estimate with hyperloglog but it seems unnecessary for this use case.
    // For now we reserve initial capacity as the max size of the two
    dict_value_to_unified_index->reserve(
        std::max(arr1_dictionary_len, arr2_dictionary_len));

    // this vector stores indices of the strings in arr2 that will be
    // part of the unified dictionary. All of array 1's strings will always
    // be part of the unified dictionary.
    bodo::vector<size_t> arr2_unique_strs;
    arr2_unique_strs.reserve(arr2_dictionary_len);

    offset_t const* const arr1_str_offsets =
        (offset_t*)arr1->child_arrays[0]->data2();
    int64_t n_chars = arr1_str_offsets[arr1_dictionary_len];

    dict_indices_t next_index = 1;
    for (size_t i = 0; i < arr1_dictionary_len; i++) {
        // TODO: Move into the constructor
        // Set the first n elements in the hash map, each of which is
        // always unique.
        dict_indices_t& index = (*dict_value_to_unified_index)[i];
        index = next_index++;
    }

    offset_t const* const arr2_str_offsets =
        (offset_t*)arr2->child_arrays[0]->data2();
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
    arr1_hashes.reset();
    arr2_hashes.reset();

    // ensure_dicts_can_unify requires each array be globally replicated.
    // This is either because the whole array is replicated or the dictionary
    // is.
    bool is_globally_replicated = true;
    // ensure_dicts_can_unify requires each array is unique.
    bool is_locally_unique = true;
    std::shared_ptr<array_info> new_dict =
        alloc_string_array(Bodo_CTypes::STRING, n_strings, n_chars, -1, 0,
                           is_globally_replicated, is_locally_unique);
    offset_t* new_dict_str_offsets = (offset_t*)new_dict->data2();

    // Initialize the offset and string index to the end of arr1's dictionary
    offset_t cur_offset = arr1_str_offsets[arr1_dictionary_len];
    int64_t cur_offset_idx = arr1_dictionary_len + 1;

    // copy offsets from arr1 into new_dict_str_offsets
    memcpy(new_dict_str_offsets, arr1_str_offsets,
           cur_offset_idx * sizeof(offset_t));
    // copy strings from arr1 into new_dict
    memcpy(new_dict->data1<bodo_array_type::STRING>(),
           arr1->child_arrays[0]->data1<bodo_array_type::STRING>(), cur_offset);
    for (auto i : arr2_unique_strs) {
        offset_t str_len = arr2_str_offsets[i + 1] - arr2_str_offsets[i];
        memcpy(new_dict->data1<bodo_array_type::STRING>() + cur_offset,
               arr2->child_arrays[0]->data1<bodo_array_type::STRING>() +
                   arr2_str_offsets[i],
               str_len);
        cur_offset += str_len;
        new_dict_str_offsets[cur_offset_idx++] = cur_offset;
    }

    arr1->child_arrays[0] = new_dict;
    arr2->child_arrays[0] = new_dict;

    // convert old indices to new ones for arr2
    replace_dict_arr_indices(arr2, arr2_index_map);
}

// CACHE FOR LIKE KERNEL DICT-ENCODING CASE

like_kernel_cache_t* alloc_like_kernel_cache(uint64_t reserve_size) noexcept {
    auto cache = new like_kernel_cache_t();
    cache->reserve(reserve_size);
    return cache;
}

void add_to_like_kernel_cache(like_kernel_cache_t* cache, uint32_t idx1,
                              uint32_t idx2, bool val) noexcept {
    // Concatenate the two uint32_t indices into a uint64_t to form the key.
    // Primary reason for doing this is that std::hash doesn't support hashing
    // std::pair.
    uint64_t key = (((uint64_t)idx1) << 32) | ((uint64_t)idx2);
    (*cache)[key] = static_cast<int8_t>(val);
}

int8_t check_like_kernel_cache(like_kernel_cache_t* cache, uint32_t idx1,
                               uint32_t idx2) noexcept {
    // Concatenate the two uint32_t indices into a uint64_t to form the key.
    uint64_t key = (((uint64_t)idx1) << 32) | ((uint64_t)idx2);
    auto iter = cache->find(key);
    if (iter == cache->end()) {
        // Miss
        return -1;
    }
    return (*cache)[key];
}

void dealloc_like_kernel_cache(like_kernel_cache_t* cache) noexcept {
    delete cache;
}
