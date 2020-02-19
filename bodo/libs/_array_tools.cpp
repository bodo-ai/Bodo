// Copyright (C) 2019 Bodo Inc. All rights reserved.
/**
 * @file _array_tools.cpp
 * @author Ehsan (ehsan@bodo-inc.com)
 * @brief Tools for handling bodo array, such as passing to C/C++
 * structures/functions
 * @date 2019-10-06
 */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <string>
#include "_bodo_common.h"
#include "_distributed.h"
#include "_murmurhash3.cpp"
#include "gfx/timsort.hpp"
#include "mpi.h"

#undef USE_STD
#define USE_TSL_ROBIN
#undef USE_TSL_SPARSE
#undef USE_TSL_HOPSCOTCH

#ifdef USE_STD
# include <unordered_map>
# include <unordered_set>
# define MAP_CONTAINER std::unordered_map
# define SET_CONTAINER std::unordered_set
#endif
#ifdef USE_TSL_ROBIN
# include <include/tsl/robin_map.h>
# include <include/tsl/robin_set.h>
# define MAP_CONTAINER tsl::robin_map
# define SET_CONTAINER tsl::robin_set
#endif
#ifdef USE_TSL_SPARSE
# include <include/tsl/sparse_map.h>
# include <include/tsl/sparse_set.h>
# define MAP_CONTAINER tsl::sparse_map
# define SET_CONTAINER tsl::sparse_set
#endif
#ifdef USE_TSL_HOPSCOTCH
# include <include/tsl/hopscotch_map.h>
# include <include/tsl/hopscotch_set.h>
# define MAP_CONTAINER tsl::hopscotch_map
# define SET_CONTAINER tsl::hopscotch_set
#endif




#define ALIGNMENT 64  // preferred alignment for AVX512

#define SEED_HASH_PARTITION 0xb0d01289


void CheckEqualityArrayType(array_info* arr1, array_info* arr2)
{
    if (arr1->arr_type != arr2->arr_type) {
        PyErr_SetString(PyExc_RuntimeError,
                        "array_info passed to Cpp code have different arr_type");
        return;
    }
    if (arr1->arr_type != bodo_array_type::STRING) {
        if (arr1->dtype != arr2->dtype) {
            PyErr_SetString(PyExc_RuntimeError,
                            "array_info passed to Cpp code have different dtype");
            return;
        }
    }
}


array_info* string_array_to_info(uint64_t n_items, uint64_t n_chars, char* data,
                                 char* offsets, char* null_bitmap,
                                 NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::STRING, Bodo_CTypes::STRING, n_items,
                          n_chars, data, offsets, NULL, null_bitmap, meminfo,
                          NULL);
}

array_info* numpy_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NUMPY,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, data,
                          NULL, NULL, NULL, meminfo, NULL);
}

array_info* nullable_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                   char* null_bitmap, NRT_MemInfo* meminfo,
                                   NRT_MemInfo* meminfo_bitmask) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, data,
                          NULL, NULL, null_bitmap, meminfo, meminfo_bitmask);
}

void info_to_string_array(array_info* info, uint64_t* n_items,
                          uint64_t* n_chars, char** data, char** offsets,
                          char** null_bitmap, NRT_MemInfo** meminfo) {
    if (info->arr_type != bodo_array_type::STRING) {
        PyErr_SetString(PyExc_RuntimeError,
                        "info_to_string_array requires string input");
        return;
    }
    *n_items = info->length;
    *n_chars = info->n_sub_elems;
    *data = info->data1;
    *offsets = info->data2;
    *null_bitmap = info->null_bitmask;
    *meminfo = info->meminfo;
    return;
}

void info_to_numpy_array(array_info* info, uint64_t* n_items, char** data,
                         NRT_MemInfo** meminfo) {
    if (info->arr_type != bodo_array_type::NUMPY) {
        PyErr_SetString(PyExc_RuntimeError,
                        "info_to_numpy_array requires numpy input");
        return;
    }
    *n_items = info->length;
    *data = info->data1;
    *meminfo = info->meminfo;
    return;
}

void info_to_nullable_array(array_info* info, uint64_t* n_items,
                            uint64_t* n_bytes, char** data, char** null_bitmap,
                            NRT_MemInfo** meminfo,
                            NRT_MemInfo** meminfo_bitmask) {
    if (info->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "info_to_nullable_array requires nullable input");
        return;
    }
    *n_items = info->length;
    *n_bytes = (info->length + 7) >> 3;
    *data = info->data1;
    *null_bitmap = info->null_bitmask;
    *meminfo = info->meminfo;
    *meminfo_bitmask = info->meminfo_bitmask;
    return;
}

// for numpy arrays, this maps dtype to sizeof, thus avoiding
// get_item_size() function call
std::vector<size_t> numpy_item_size(BODO_NUMPY_ARRAY_NUM_DTYPES);

array_info* alloc_numpy(int64_t length, Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    return new array_info(bodo_array_type::NUMPY, typ_enum, length, -1, data,
                          NULL, NULL, NULL, meminfo, NULL);
}

array_info* alloc_nullable_array(int64_t length,
                                 Bodo_CTypes::CTypeEnum typ_enum,
                                 int64_t extra_null_bytes) {
    int64_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    NRT_MemInfo* meminfo_bitmask =
        NRT_MemInfo_alloc_safe_aligned(n_bytes * sizeof(uint8_t), ALIGNMENT);
    char* null_bitmap = (char*)meminfo_bitmask->data;
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL, typ_enum, length,
                          -1, data, NULL, NULL, null_bitmap, meminfo,
                          meminfo_bitmask);
}

array_info* alloc_string_array(int64_t length, int64_t n_chars,
                               int64_t extra_null_bytes) {
    // extra_null_bytes are necessary for communication buffers around the edges
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_dtor_safe(
        sizeof(str_arr_payload), (NRT_dtor_function)dtor_string_array);
    str_arr_payload* payload = (str_arr_payload*)meminfo->data;
    allocate_string_array(&(payload->offsets), &(payload->data),
                          &(payload->null_bitmap), length, n_chars,
                          extra_null_bytes);
    return new array_info(bodo_array_type::STRING, Bodo_CTypes::STRING, length,
                          n_chars, payload->data, (char*)payload->offsets, NULL,
                          (char*)payload->null_bitmap, meminfo, NULL);
}

table_info* arr_info_list_to_table(array_info** arrs, int64_t n_arrs) {
    std::vector<array_info*> columns = std::vector<array_info*>(n_arrs);
    for (size_t i = 0; i < (size_t)n_arrs; i++) {
        columns[i] = arrs[i];
    }
    return new table_info(columns);
}

array_info* info_from_table(table_info* table, int64_t col_ind) {
    return table->columns[col_ind];
}

void delete_table(table_info* table) {
    for (array_info* a : table->columns) {
        delete a;
    }
    delete table;
}

void free_array(array_info* arr);

void delete_table_free_arrays(table_info* table) {
    for (array_info* a : table->columns) {
        free_array(a);
        delete a;
    }
    delete table;
}

template <class T>
void hash_array_inner(uint32_t* out_hashes, T* data, size_t n_rows,
                      const uint32_t seed) {
    for (size_t i = 0; i < n_rows; i++) {
        hash_inner_32<T>(&data[i], seed, &out_hashes[i]);
    }
}

void hash_array_string(uint32_t* out_hashes, char* data, uint32_t* offsets,
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

void hash_array(uint32_t* out_hashes, array_info* array, size_t n_rows,
                const uint32_t seed) {
    // dispatch to proper function
    // TODO: general dispatcher
    // XXX: assumes nullable array data for nulls is always consistent
    if (array->dtype == Bodo_CTypes::_BOOL)
        return hash_array_inner<bool>(out_hashes, (bool*)array->data1,
                                      n_rows, seed);
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
    if (array->dtype == Bodo_CTypes::UINT64)
        return hash_array_inner<uint64_t>(out_hashes, (uint64_t*)array->data1,
                                          n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return hash_array_inner<float>(out_hashes, (float*)array->data1, n_rows,
                                       seed);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return hash_array_inner<double>(out_hashes, (double*)array->data1,
                                        n_rows, seed);
    if (array->arr_type == bodo_array_type::STRING)
        return hash_array_string(out_hashes, (char*)array->data1,
                                 (uint32_t*)array->data2, n_rows, seed);
    PyErr_SetString(PyExc_RuntimeError, "Invalid data type for hash");
}

template <class T>
void hash_array_combine_inner(uint32_t* out_hashes, T* data, size_t n_rows,
                              const uint32_t seed) {
    // hash combine code from boost
    // https://github.com/boostorg/container_hash/blob/504857692148d52afe7110bcb96cf837b0ced9d7/include/boost/container_hash/hash.hpp#L313
    for (size_t i = 0; i < n_rows; i++) {
        uint32_t out_hash = 0;
        hash_inner_32<T>(&data[i], seed, &out_hash);
        out_hashes[i] ^=
            out_hash + 0x9e3779b9 + (out_hashes[i] << 6) + (out_hashes[i] >> 2);
    }
}

void hash_array_combine_string(uint32_t* out_hashes, char* data,
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

void hash_array_combine(uint32_t* out_hashes, array_info* array, size_t n_rows,
                        const uint32_t seed) {
    // dispatch to proper function
    // TODO: general dispatcher
    if (array->dtype == Bodo_CTypes::_BOOL)
        return hash_array_combine_inner<bool>(
            out_hashes, (bool*)array->data1, n_rows, seed);
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
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return hash_array_combine_inner<float>(out_hashes, (float*)array->data1,
                                               n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return hash_array_combine_inner<double>(
            out_hashes, (double*)array->data1, n_rows, seed);
    if (array->arr_type == bodo_array_type::STRING)
        return hash_array_combine_string(out_hashes, (char*)array->data1,
                                         (uint32_t*)array->data2, n_rows, seed);
    PyErr_SetString(PyExc_RuntimeError, "Invalid data type for hash combine");
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

template <class T>
void fill_send_array_inner(T* send_buff, T* data, uint32_t* hashes,
                           std::vector<int> const& send_disp, int n_pes,
                           size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int ind = tmp_offset[node];
        send_buff[ind] = data[i];
        tmp_offset[node]++;
    }
}


template <class T>
void fill_recv_data_inner(T* recv_buff, T* data, uint32_t* hashes,
                           std::vector<int> const& send_disp, int n_pes,
                           size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int ind = tmp_offset[node];
        data[i] = recv_buff[ind];
        tmp_offset[node]++;
    }
}






void fill_send_array_string_inner(char* send_data_buff,
                                  uint32_t* send_length_buff, char* arr_data,
                                  uint32_t* arr_offsets, uint32_t* hashes,
                                  std::vector<int> const& send_disp,
                                  std::vector<int> const& send_disp_char, int n_pes,
                                  size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    std::vector<int> tmp_offset_char(send_disp_char);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        // write length
        int ind = tmp_offset[node];
        uint32_t str_len = arr_offsets[i + 1] - arr_offsets[i];
        send_length_buff[ind] = str_len;
        tmp_offset[node]++;
        // write data
        int c_ind = tmp_offset_char[node];
        memcpy(&send_data_buff[c_ind], &arr_data[arr_offsets[i]], str_len);
        tmp_offset_char[node] += str_len;
    }
}

void fill_send_array_null_inner(uint8_t* send_null_bitmask,
                                uint8_t* array_null_bitmask, uint32_t* hashes,
                                std::vector<int> const& send_disp_null, int n_pes,
                                size_t n_rows) {
    std::vector<int> tmp_offset(n_pes, 0);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int ind = tmp_offset[node];
        // write null bit
        bool bit = GetBit(array_null_bitmask, i);
        uint8_t* out_bitmap = &send_null_bitmask[send_disp_null[node]];
        SetBitTo(out_bitmap, ind, bit);
        tmp_offset[node]++;
    }
    return;
}

void fill_send_array(array_info* send_arr, array_info* array, uint32_t* hashes,
                     std::vector<int> const& send_disp,
                     std::vector<int> const& send_disp_char,
                     std::vector<int> const& send_disp_null, int n_pes) {
    size_t n_rows = (size_t)array->length;
    // dispatch to proper function
    // TODO: general dispatcher
    if (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL)
        fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask,
                                   (uint8_t*)array->null_bitmask, hashes,
                                   send_disp_null, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::_BOOL)
        return fill_send_array_inner<bool>((bool*)send_arr->data1,
                                             (bool*)array->data1, hashes,
                                             send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT8)
        return fill_send_array_inner<int8_t>((int8_t*)send_arr->data1,
                                             (int8_t*)array->data1, hashes,
                                             send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT8)
        return fill_send_array_inner<uint8_t>((uint8_t*)send_arr->data1,
                                              (uint8_t*)array->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT16)
        return fill_send_array_inner<int16_t>((int16_t*)send_arr->data1,
                                              (int16_t*)array->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT16)
        return fill_send_array_inner<uint16_t>((uint16_t*)send_arr->data1,
                                               (uint16_t*)array->data1, hashes,
                                               send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT32)
        return fill_send_array_inner<int32_t>((int32_t*)send_arr->data1,
                                              (int32_t*)array->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT32)
        return fill_send_array_inner<uint32_t>((uint32_t*)send_arr->data1,
                                               (uint32_t*)array->data1, hashes,
                                               send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT64)
        return fill_send_array_inner<int64_t>((int64_t*)send_arr->data1,
                                              (int64_t*)array->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT64)
        return fill_send_array_inner<uint64_t>((uint64_t*)send_arr->data1,
                                               (uint64_t*)array->data1, hashes,
                                               send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return fill_send_array_inner<float>((float*)send_arr->data1,
                                            (float*)array->data1, hashes,
                                            send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return fill_send_array_inner<double>((double*)send_arr->data1,
                                             (double*)array->data1, hashes,
                                             send_disp, n_pes, n_rows);
    if (array->arr_type == bodo_array_type::STRING)
        fill_send_array_string_inner(
            (char*)send_arr->data1, (uint32_t*)send_arr->data2,
            (char*)array->data1, (uint32_t*)array->data2, hashes, send_disp,
            send_disp_char, n_pes, n_rows);
    fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask,
                               (uint8_t*)array->null_bitmask, hashes,
                               send_disp_null, n_pes, n_rows);
    return;
    PyErr_SetString(PyExc_RuntimeError, "Invalid data type for send fill");
}

void calc_disp(std::vector<int>& disps, std::vector<int>& counts) {
    size_t n = counts.size();
    disps[0] = 0;
    for (size_t i = 1; i < n; i++) disps[i] = disps[i - 1] + counts[i - 1];
    return;
}

struct mpi_comm_info {
    int n_pes;
    std::vector<array_info*> arrays;
    size_t n_rows;
    bool has_nulls;
    // generally required MPI counts
    std::vector<int> send_count;
    std::vector<int> recv_count;
    std::vector<int> send_disp;
    std::vector<int> recv_disp;
    // counts required for string arrays
    std::vector<std::vector<int>> send_count_char;
    std::vector<std::vector<int>> recv_count_char;
    std::vector<std::vector<int>> send_disp_char;
    std::vector<std::vector<int>> recv_disp_char;
    // counts for arrays with null bitmask
    std::vector<int> send_count_null;
    std::vector<int> recv_count_null;
    std::vector<int> send_disp_null;
    std::vector<int> recv_disp_null;
    size_t n_null_bytes;

    explicit mpi_comm_info(int _n_pes, std::vector<array_info*>& _arrays)
        : n_pes(_n_pes), arrays(_arrays) {
        n_rows = arrays[0]->length;
        has_nulls = false;
        for (array_info* arr_info : arrays) {
            if (arr_info->arr_type == bodo_array_type::STRING ||
                arr_info->arr_type == bodo_array_type::NULLABLE_INT_BOOL)
                has_nulls = true;
        }
        n_null_bytes=0;
        // init counts
        send_count = std::vector<int>(n_pes, 0);
        recv_count = std::vector<int>(n_pes);
        send_disp = std::vector<int>(n_pes);
        recv_disp = std::vector<int>(n_pes);
        // init counts for string arrays
        for (array_info* arr_info : arrays) {
            if (arr_info->arr_type == bodo_array_type::STRING) {
                send_count_char.emplace_back(std::vector<int>(n_pes, 0));
                recv_count_char.emplace_back(std::vector<int>(n_pes));
                send_disp_char.emplace_back(std::vector<int>(n_pes));
                recv_disp_char.emplace_back(std::vector<int>(n_pes));
            } else {
                send_count_char.emplace_back(std::vector<int>());
                recv_count_char.emplace_back(std::vector<int>());
                send_disp_char.emplace_back(std::vector<int>());
                recv_disp_char.emplace_back(std::vector<int>());
            }
        }
        if (has_nulls) {
            send_count_null = std::vector<int>(n_pes);
            recv_count_null = std::vector<int>(n_pes);
            send_disp_null = std::vector<int>(n_pes);
            recv_disp_null = std::vector<int>(n_pes);
        }
    }

    void set_counts(uint32_t* hashes) {
        // get send count
        for (size_t i = 0; i < n_rows; i++) {
            size_t node = (size_t)hashes[i] % (size_t)n_pes;
            send_count[node]++;
        }

        // get recv count
        MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1,
                     MPI_INT, MPI_COMM_WORLD);

        // get displacements
        calc_disp(send_disp, send_count);
        calc_disp(recv_disp, recv_count);

        // counts for string arrays
        for (size_t i = 0; i < arrays.size(); i++) {
            array_info* arr_info = arrays[i];
            if (arr_info->arr_type == bodo_array_type::STRING) {
                // send counts
                std::vector<int>& char_counts = send_count_char[i];
                uint32_t* offsets = (uint32_t*)arr_info->data2;
                for (size_t i = 0; i < n_rows; i++) {
                    int str_len = offsets[i + 1] - offsets[i];
                    size_t node = (size_t)hashes[i] % (size_t)n_pes;
                    char_counts[node] += str_len;
                }
                // get recv count
                MPI_Alltoall(char_counts.data(), 1, MPI_INT,
                             recv_count_char[i].data(), 1, MPI_INT,
                             MPI_COMM_WORLD);
                // get displacements
                calc_disp(send_disp_char[i], char_counts);
                calc_disp(recv_disp_char[i], recv_count_char[i]);
            }
        }
        if (has_nulls) {
            for (size_t i = 0; i < size_t(n_pes); i++) {
                send_count_null[i] = (send_count[i] + 7) >> 3;
                recv_count_null[i] = (recv_count[i] + 7) >> 3;
            }
            calc_disp(send_disp_null, send_count_null);
            calc_disp(recv_disp_null, recv_count_null);
            n_null_bytes = std::accumulate(recv_count_null.begin(),
                                           recv_count_null.end(), 0);
        }
        return;
    }
};

/* Internal function. Convert counts to displacements
 */
void convert_len_arr_to_offset(uint32_t* offsets, size_t const& num_strs) {
    uint32_t curr_offset = 0;
    for (size_t i = 0; i < num_strs; i++) {
        uint32_t val = offsets[i];
        offsets[i] = curr_offset;
        curr_offset += val;
    }
    offsets[num_strs] = curr_offset;
    return;
}

void copy_gathered_null_bytes(uint8_t* null_bitmask,
                              std::vector<uint8_t> const& tmp_null_bytes,
                              std::vector<int> const& recv_count_null,
                              std::vector<int> const& recv_count) {
    size_t curr_tmp_byte = 0;  // current location in buffer with all data
    size_t curr_str = 0;       // current string in output bitmap
    // for each chunk
    for (size_t i = 0; i < recv_count.size(); i++) {
        size_t n_strs = recv_count[i];
        size_t n_bytes = recv_count_null[i];
        const uint8_t* chunk_bytes = &tmp_null_bytes[curr_tmp_byte];
        // for each string in chunk
        for (size_t j = 0; j < n_strs; j++) {
            SetBitTo(null_bitmask, curr_str, GetBit(chunk_bytes, j));
            curr_str += 1;
        }
        curr_tmp_byte += n_bytes;
    }
    return;
}

void shuffle_array(array_info* send_arr, array_info* out_arr,
                   std::vector<int> const& send_count,
                   std::vector<int> const& recv_count,
                   std::vector<int> const& send_disp,
                   std::vector<int> const& recv_disp,
                   std::vector<int> const& send_count_char,
                   std::vector<int> const& recv_count_char,
                   std::vector<int> const& send_disp_char,
                   std::vector<int> const& recv_disp_char,
                   std::vector<int> const& send_count_null,
                   std::vector<int> const& recv_count_null,
                   std::vector<int> const& send_disp_null,
                   std::vector<int> const& recv_disp_null,
                   std::vector<uint8_t>& tmp_null_bytes) {
    // strings need data and length comm
    if (send_arr->arr_type == bodo_array_type::STRING) {
        // string lengths
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
        MPI_Alltoallv(send_arr->data2, send_count.data(), send_disp.data(),
                      mpi_typ, out_arr->data2, recv_count.data(),
                      recv_disp.data(), mpi_typ, MPI_COMM_WORLD);
        convert_len_arr_to_offset((uint32_t*)out_arr->data2,
                                  (size_t)out_arr->length);
        // string data
        mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Alltoallv(send_arr->data1, send_count_char.data(),
                      send_disp_char.data(), mpi_typ, out_arr->data1,
                      recv_count_char.data(), recv_disp_char.data(), mpi_typ,
                      MPI_COMM_WORLD);

    } else {  // Numpy/nullable arrays
        MPI_Datatype mpi_typ = get_MPI_typ(send_arr->dtype);
        MPI_Alltoallv(send_arr->data1, send_count.data(), send_disp.data(),
                      mpi_typ, out_arr->data1, recv_count.data(),
                      recv_disp.data(), mpi_typ, MPI_COMM_WORLD);
    }
    if (send_arr->arr_type == bodo_array_type::STRING ||
        send_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // nulls
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Alltoallv(send_arr->null_bitmask, send_count_null.data(),
                      send_disp_null.data(), mpi_typ, tmp_null_bytes.data(),
                      recv_count_null.data(), recv_disp_null.data(), mpi_typ,
                      MPI_COMM_WORLD);
        copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask,
                                 tmp_null_bytes, recv_count_null, recv_count);
    }
    return;
}

array_info* alloc_array(int64_t length, int64_t n_sub_elems,
                        bodo_array_type::arr_type_enum arr_type,
                        Bodo_CTypes::CTypeEnum dtype,
                        int64_t extra_null_bytes) {
    if (arr_type == bodo_array_type::STRING)
        return alloc_string_array(length, n_sub_elems, extra_null_bytes);

    // nullable array
    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL)
        return alloc_nullable_array(length, dtype, extra_null_bytes);

    // Numpy
    // TODO: error check
    return alloc_numpy(length, dtype);
}

void free_array(array_info* arr) {
    // string array
    if (arr->arr_type == bodo_array_type::STRING) {
        // data
        delete[] arr->data1;
        // offsets
        delete[] arr->data2;
        // nulls
        if (arr->null_bitmask != nullptr) delete[] arr->null_bitmask;
    } else {                 // Numpy or nullable array
        free(arr->meminfo);  // TODO: decref for cleanup?
        if (arr->meminfo_bitmask != NULL) free(arr->meminfo_bitmask);
    }
    return;
}

table_info* shuffle_table_kernel(table_info* in_table, uint32_t* hashes, int n_pes, mpi_comm_info const& comm_info) {
    int total_recv = std::accumulate(comm_info.recv_count.begin(),
                                     comm_info.recv_count.end(), 0);
    size_t n_cols = in_table->ncols();
    std::vector<int> n_char_recvs(n_cols);
    for (size_t i = 0; i < n_cols; i++)
        n_char_recvs[i] =
            std::accumulate(comm_info.recv_count_char[i].begin(),
                            comm_info.recv_count_char[i].end(), 0);

    // fill send buffer and send
    std::vector<array_info*> out_arrs;
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<uint8_t> tmp_null_bytes(comm_info.n_null_bytes);
    for (size_t i = 0; i < n_cols; i++) {
        array_info* in_arr = in_table->columns[i];
        array_info* send_arr =
            alloc_array(n_rows, in_arr->n_sub_elems, in_arr->arr_type,
                        in_arr->dtype, 2 * n_pes);
        array_info* out_arr = alloc_array(total_recv, n_char_recvs[i],
                                          in_arr->arr_type, in_arr->dtype, 0);
        fill_send_array(send_arr, in_arr, hashes, comm_info.send_disp,
                        comm_info.send_disp_char[i], comm_info.send_disp_null,
                        n_pes);

        shuffle_array(send_arr, out_arr,
                      comm_info.send_count, comm_info.recv_count,
                      comm_info.send_disp, comm_info.recv_disp,
                      comm_info.send_count_char[i], comm_info.recv_count_char[i],
                      comm_info.send_disp_char[i], comm_info.recv_disp_char[i],
                      comm_info.send_count_null, comm_info.recv_count_null,
                      comm_info.send_disp_null, comm_info.recv_disp_null,
                      tmp_null_bytes);

        out_arrs.push_back(out_arr);
        free_array(send_arr);
        delete send_arr;
    }

    return new table_info(out_arrs);
}


table_info* shuffle_table(table_info* in_table, int64_t n_keys) {
    // error checking
    if (in_table->ncols() <= 0 || n_keys <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return NULL;
    }
    int n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    mpi_comm_info comm_info(n_pes, in_table->columns);
    // computing the hash data structure
    std::vector<array_info*> key_arrs = std::vector<array_info*>(
        in_table->columns.begin(), in_table->columns.begin() + n_keys);
    uint32_t seed = SEED_HASH_PARTITION;
    uint32_t* hashes = hash_keys(key_arrs, seed);

    comm_info.set_counts(hashes);

    table_info* table = shuffle_table_kernel(in_table, hashes, n_pes, comm_info);
    delete [] hashes;
    return table;
}



/** Getting the computing node on which a row belongs to
 *
 * The template paramter is T.
 * @param in_table: the input table
 * @param n_keys  : the number of keys to be used for the hash
 * @param n_pes   : the number of processor considered
 * @return the table containing a single column with the nodes
 */
table_info* compute_node_partition_by_hash(table_info* in_table, int64_t n_keys, int64_t n_pes)
{
#undef DEBUG_COMP_HASH
    int64_t n_rows = in_table->nrows();
    std::vector<array_info*> key_arrs = std::vector<array_info*>(
        in_table->columns.begin(), in_table->columns.begin() + n_keys);
    uint32_t seed = SEED_HASH_PARTITION;
    uint32_t* hashes = hash_keys(key_arrs, seed);
    //
    std::vector<array_info*> out_arrs;
    array_info* out_arr = alloc_array(n_rows, -1, bodo_array_type::NUMPY,
                                      Bodo_CTypes::INT32, 0);
#ifdef DEBUG_COMP_HASH
    std::cout << "COMPUTE_HASH\n";
#endif
    for (int64_t i_row=0; i_row<n_rows; i_row++) {
      int32_t node_id = hashes[i_row] % n_pes;
      out_arr->at<int32_t>(i_row) = node_id;
#ifdef DEBUG_COMP_HASH
      std::cout << "i_row=" << i_row << " node_id=" << node_id << "\n";
#endif
    }
    out_arrs.push_back(out_arr);
    return new table_info(out_arrs);
}



/** Getting the expression of a T value as a vector of characters
 *
 * The template paramter is T.
 * @param val the value in the type T.
 * @return the vector of characters on output
 */
template <typename T>
std::vector<char> GetCharVector(T const& val) {
    const T* valptr = &val;
    const char* charptr = (char*)valptr;
    std::vector<char> V(sizeof(T));
    for (size_t u = 0; u < sizeof(T); u++) V[u] = charptr[u];
    return V;
}




/** Getting the expression of a string of characters as a T value
 *
 * The template paramter is T.
 * @param dtype the bodo data type.
 * @param ptr the value of the pointer passed in argument
 * @return the value as a T value.
 */
template<typename T>
T GetTentry(char* ptr)
{
  T* ptr_T = (T*)ptr;
  return *ptr_T;
}

/** Getting the expression of a string of characters as a double value
 *
 * @param dtype the bodo data type.
 * @param ptr the value of the pointer passed in argument
 * @return the value as a double.
 */
double GetDoubleEntry(Bodo_CTypes::CTypeEnum dtype, char* ptr)
{
    if (dtype == Bodo_CTypes::INT8) return GetTentry<int8_t>(ptr);
    if (dtype == Bodo_CTypes::UINT8) return GetTentry<uint8_t>(ptr);
    if (dtype == Bodo_CTypes::INT16) return GetTentry<int16_t>(ptr);
    if (dtype == Bodo_CTypes::UINT16) return GetTentry<uint16_t>(ptr);
    if (dtype == Bodo_CTypes::INT32) return GetTentry<int32_t>(ptr);
    if (dtype == Bodo_CTypes::UINT32) return GetTentry<uint32_t>(ptr);
    if (dtype == Bodo_CTypes::INT64) return GetTentry<int64_t>(ptr);
    if (dtype == Bodo_CTypes::UINT64) return GetTentry<uint64_t>(ptr);
    if (dtype == Bodo_CTypes::FLOAT32) return GetTentry<float>(ptr);
    if (dtype == Bodo_CTypes::FLOAT64) return GetTentry<double>(ptr);
    return 0;
}




/* The NaN entry used in the case a normal value is not available.
 *
 * The choice are done in following way:
 * ---int8_t / int16_t / int32_t / int64_t : return a -1 value
 * ---uint8_t / uint16_t / uint32_t / uint64_t : return a 0 value
 * ---float / double : return a NaN
 * This is obviously not perfect as -1 can be a legitimate value, but here goes.
 *
 * @param the dtype used.
 * @return the list of characters in output.
 */
std::vector<char> RetrieveNaNentry(Bodo_CTypes::CTypeEnum const& dtype) {
    if (dtype == Bodo_CTypes::_BOOL) return GetCharVector<bool>(false);
    if (dtype == Bodo_CTypes::INT8) return GetCharVector<int8_t>(-1);
    if (dtype == Bodo_CTypes::UINT8) return GetCharVector<uint8_t>(0);
    if (dtype == Bodo_CTypes::INT16) return GetCharVector<int16_t>(-1);
    if (dtype == Bodo_CTypes::UINT16) return GetCharVector<uint16_t>(0);
    if (dtype == Bodo_CTypes::INT32) return GetCharVector<int32_t>(-1);
    if (dtype == Bodo_CTypes::UINT32) return GetCharVector<uint32_t>(0);
    if (dtype == Bodo_CTypes::INT64) return GetCharVector<int64_t>(-1);
    if (dtype == Bodo_CTypes::UINT64) return GetCharVector<uint64_t>(0);
    if (dtype == Bodo_CTypes::FLOAT32) return GetCharVector<float>(std::nanf("1"));
    if (dtype == Bodo_CTypes::FLOAT64) return GetCharVector<double>(std::nan("1"));
    return {};
}

/** Printing the string expression of an entry in the column
 *
 * @param dtype: the data type on input
 * @param ptrdata: The pointer to the data (its length is determined by dtype)
 * @return The string on output.
 */
std::string GetStringExpression(Bodo_CTypes::CTypeEnum const& dtype,
                                char* ptrdata) {
    if (dtype == Bodo_CTypes::_BOOL) {
        bool* ptr = (bool*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::INT8) {
        int8_t* ptr = (int8_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::UINT8) {
        uint8_t* ptr = (uint8_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::INT16) {
        int16_t* ptr = (int16_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::UINT16) {
        uint16_t* ptr = (uint16_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::INT32) {
        int32_t* ptr = (int32_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::UINT32) {
        uint32_t* ptr = (uint32_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::INT64) {
        int64_t* ptr = (int64_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::UINT64) {
        uint64_t* ptr = (uint64_t*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::FLOAT32) {
        float* ptr = (float*)ptrdata;
        return std::to_string(*ptr);
    }
    if (dtype == Bodo_CTypes::FLOAT64) {
        double* ptr = (double*)ptrdata;
        return std::to_string(*ptr);
    }
    return "no matching type";
}

/* This is a function used by "DEBUG_PrintSetOfColumn"
 * It takes a column and returns a vector of string on output
 *
 * @param arr is the pointer.
 * @return The vector of strings to be used later.
 */
std::vector<std::string> DEBUG_PrintColumn(array_info* arr) {
    size_t nRow = arr->length;
    std::vector<std::string> ListStr(nRow);
    std::string strOut;
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        uint64_t siztype = numpy_item_size[arr->dtype];
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = GetBit(null_bitmask, iRow);
            if (bit) {
                char* ptrdata1 = &(arr->data1[siztype * iRow]);
                strOut = GetStringExpression(arr->dtype, ptrdata1);
            } else {
                strOut = "false";
            }
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::NUMPY) {
        uint64_t siztype = numpy_item_size[arr->dtype];
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            char* ptrdata1 = &(arr->data1[siztype * iRow]);
            strOut = GetStringExpression(arr->dtype, ptrdata1);
            ListStr[iRow] = strOut;
        }
    }
    if (arr->arr_type == bodo_array_type::STRING) {
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        uint32_t* data2 = (uint32_t*)arr->data2;
        char* data1 = arr->data1;
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = GetBit(null_bitmask, iRow);
            if (bit) {
                uint32_t start_pos = data2[iRow];
                uint32_t end_pos = data2[iRow + 1];
                uint32_t len = end_pos - start_pos;
                char* strname;
                strname = new char[len + 1];
                for (uint32_t i = 0; i < len; i++) {
                    strname[i] = data1[start_pos + i];
                }
                strname[len] = '\0';
                strOut = strname;
                delete[] strname;
            } else {
                strOut = "false";
            }
            ListStr[iRow] = strOut;
        }
    }
    return ListStr;
}

/** The DEBUG_PrintSetOfColumn is printing the contents of the table to
 * the output stream.
 * All cases are supported (NUMPY, SRING, NULLABLE_INT_BOOL) as well as
 * all integer and floating types.
 *
 * The number of rows in the columns do not have to be the same.
 *
 * @param the output stream (e.g. std::cerr or std::cout)
 * @param ListArr the list of columns in input
 * @return Nothing. Everything is put in the stream
 */
void DEBUG_PrintSetOfColumn(std::ostream& os,
                            std::vector<array_info*> const& ListArr) {
    int nCol = ListArr.size();
    if (nCol == 0) {
        os << "Nothing to print really\n";
        return;
    }
    std::vector<int> ListLen(nCol);
    int nRowMax = 0;
    for (int iCol = 0; iCol < nCol; iCol++) {
        int nRow = ListArr[iCol]->length;
        if (nRow > nRowMax) nRowMax = nRow;
        ListLen[iCol] = nRow;
    }
    std::vector<std::vector<std::string>> ListListStr;
    for (int iCol = 0; iCol < nCol; iCol++) {
        std::vector<std::string> LStr = DEBUG_PrintColumn(ListArr[iCol]);
        for (int iRow = ListLen[iCol]; iRow < nRowMax; iRow++)
            LStr.emplace_back("");
        ListListStr.emplace_back(LStr);
    }
    std::vector<std::string> ListStrOut(nRowMax);
    for (int iRow = 0; iRow < nRowMax; iRow++) {
        std::string str = std::to_string(iRow) + " :";
        ListStrOut[iRow] = str;
    }
    for (int iCol = 0; iCol < nCol; iCol++) {
        std::vector<int> ListLen(nRowMax);
        size_t maxlen = 0;
        for (int iRow = 0; iRow < nRowMax; iRow++) {
            size_t elen = ListListStr[iCol][iRow].size();
            ListLen[iRow] = elen;
            if (elen > maxlen) maxlen = elen;
        }
        for (int iRow = 0; iRow < nRowMax; iRow++) {
            std::string str = ListStrOut[iRow] + " " + ListListStr[iCol][iRow];
            size_t diff = maxlen - ListLen[iRow];
            for (size_t u = 0; u < diff; u++) str += " ";
            ListStrOut[iRow] = str;
        }
    }
    for (int iRow = 0; iRow < nRowMax; iRow++) os << ListStrOut[iRow] << "\n";
}

/** This is a function used for debugging.
 * It prints the nature of the columns of the tables
 *
 * @param the output stream (for example std::cerr or std::cout)
 * @param The list of columns in output
 * @return nothing. Everything is printed to the stream
 */
void DEBUG_PrintRefct(std::ostream& os,
                      std::vector<array_info*> const& ListArr) {
    int nCol = ListArr.size();
    auto GetType = [](bodo_array_type::arr_type_enum arr_type) -> std::string {
        if (arr_type == bodo_array_type::NULLABLE_INT_BOOL) return "NULLABLE";
        if (arr_type == bodo_array_type::NUMPY) return "NUMPY";
        return "STRING";
    };
    auto GetNRTinfo = [](NRT_MemInfo* meminf) -> std::string {
        if (meminf == NULL) return "NULL";
        return "(refct=" + std::to_string(meminf->refct) + ")";
    };
    for (int iCol = 0; iCol < nCol; iCol++) {
        os << "iCol=" << iCol << " : " << GetType(ListArr[iCol]->arr_type)
           << " dtype=" << ListArr[iCol]->dtype
           << " : meminfo=" << GetNRTinfo(ListArr[iCol]->meminfo)
           << " meminfo_bitmask=" << GetNRTinfo(ListArr[iCol]->meminfo_bitmask)
           << "\n";
    }
}

/** This function uses the combinatorial information computed in the
 * "ListPairWrite" array. The other arguments shift1, shift2 and ChoiceColumn
 * are for the choice of column
 * ---For inserting a left data, ChoiceColumn = 0 indicates retrieving column
 * from the left.
 * ---For inserting a right data, ChoiceColumn = 1 indicates retrieving column
 * from the right.
 * ---For inserting a key, we need to access both to left and right columns.
 *    This corresponds to the columns shift1 and shift2.
 *
 * The code considers all the cases in turn and creates the new array from it.
 *
 * The keys in output re used twice: In the left and on the right and so they
 * are outputed twice.
 *
 * No error is thrown but input is assumed to be coherent.
 *
 * @param in_table is the input table.
 * @param ListPairWrite is the vector of list of pairs for the writing of the
 * output table
 * @param shift1 is the first shift (of the left array)
 * @param shift2 is the second shift (of the left array)
 * @param ChoiceColumn is the chosen option
 * @return one column of the table output.
 */
array_info* RetrieveArray(
    table_info* const& in_table,
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> const& ListPairWrite,
    size_t const& shift1, size_t const& shift2, int const& ChoiceColumn) {
    size_t nRowOut = ListPairWrite.size();
    array_info* out_arr = NULL;
    /* The function for computing the returning values
     * In the output is the column index to use and the row index to use.
     * The row index may be -1 though.
     *
     * @param the row index in the output
     * @return the pair (column,row) to be used.
     */
    auto get_iRow =
        [&](size_t const& iRowIn) -> std::pair<size_t, std::ptrdiff_t> {
        std::pair<std::ptrdiff_t, std::ptrdiff_t> pairLRcolumn =
            ListPairWrite[iRowIn];
        if (ChoiceColumn == 0) return {shift1, pairLRcolumn.first};
        if (ChoiceColumn == 1) return {shift2, pairLRcolumn.second};
        if (pairLRcolumn.first != -1) return {shift1, pairLRcolumn.first};
        return {shift2, pairLRcolumn.second};
    };
    // eshift is the in_table index used for the determination
    // of arr_type and dtype of the returned column.
    size_t eshift;
    if (ChoiceColumn == 0) eshift = shift1;
    if (ChoiceColumn == 1) eshift = shift2;
    if (ChoiceColumn == 2) eshift = shift1;
    if (in_table->columns[eshift]->arr_type == bodo_array_type::STRING) {
        // In the first case of STRING, we have to deal with offsets first so we
        // need one first loop to determine the needed length. In the second
        // loop, the assignation is made. If the entries are missing then the
        // bitmask is set to false.
        int64_t n_chars = 0;
        std::vector<uint32_t> ListSizes(nRowOut);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            uint32_t size = 0;
            if (pairShiftRow.second >= 0) {
                uint32_t* in_offsets =
                    (uint32_t*)in_table->columns[pairShiftRow.first]->data2;
                uint32_t end_offset = in_offsets[pairShiftRow.second + 1];
                uint32_t start_offset = in_offsets[pairShiftRow.second];
                size = end_offset - start_offset;
            }
            ListSizes[iRow] = size;
            n_chars += size;
        }
        out_arr =
            alloc_array(nRowOut, n_chars, in_table->columns[eshift]->arr_type,
                        in_table->columns[eshift]->dtype, 0);
        uint8_t* out_null_bitmask = (uint8_t*)out_arr->null_bitmask;
        uint32_t pos = 0;
        uint32_t* out_offsets = (uint32_t*)out_arr->data2;
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            uint32_t size = ListSizes[iRow];
            out_offsets[iRow] = pos;
            bool bit = false;
            if (pairShiftRow.second >= 0) {
                uint8_t* in_null_bitmask =
                    (uint8_t*)in_table->columns[pairShiftRow.first]
                        ->null_bitmask;
                uint32_t* in_offsets =
                    (uint32_t*)in_table->columns[pairShiftRow.first]->data2;
                uint32_t start_offset = in_offsets[pairShiftRow.second];
                for (uint32_t i = 0; i < size; i++) {
                    out_arr->data1[pos] = in_table->columns[pairShiftRow.first]
                                              ->data1[start_offset];
                    pos++;
                    start_offset++;
                }
                bit = GetBit(in_null_bitmask, pairShiftRow.second);
            }
            SetBitTo(out_null_bitmask, iRow, bit);
        }
        out_offsets[nRowOut] = pos;
    }
    if (in_table->columns[eshift]->arr_type ==
        bodo_array_type::NULLABLE_INT_BOOL) {
        // In the case of NULLABLE array, we do a single loop for
        // assigning the arrays.
        // We do not need to reassign the pointers, only their size
        // suffices for the copy.
        // In the case of missing array a value of false is assigned
        // to the bitmask.
        out_arr = alloc_array(nRowOut, -1, in_table->columns[eshift]->arr_type,
                              in_table->columns[eshift]->dtype, 0);
        uint8_t* out_null_bitmask = (uint8_t*)out_arr->null_bitmask;
        uint64_t siztype = numpy_item_size[in_table->columns[eshift]->dtype];
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            bool bit = false;
            if (pairShiftRow.second >= 0) {
                uint8_t* in_null_bitmask =
                    (uint8_t*)in_table->columns[pairShiftRow.first]
                        ->null_bitmask;
                for (uint64_t u = 0; u < siztype; u++)
                    out_arr->data1[siztype * iRow + u] =
                        in_table->columns[pairShiftRow.first]
                            ->data1[siztype * pairShiftRow.second + u];
                bit = GetBit(in_null_bitmask, pairShiftRow.second);
            }
            SetBitTo(out_null_bitmask, iRow, bit);
        }
    }
    if (in_table->columns[eshift]->arr_type == bodo_array_type::NUMPY) {
        // In the case of NUMPY array we have only to put a single
        // entry.
        // In the case of missing data we have to assign a NaN and that is
        // not easy in general and done in the RetrieveNaNentry.
        // According to types:
        // ---signed integer: value -1
        // ---unsigned integer: value 0
        // ---floating point: std::nan as here both notions match.
        out_arr = alloc_array(nRowOut, -1, in_table->columns[eshift]->arr_type,
                              in_table->columns[eshift]->dtype, 0);
        uint64_t siztype = numpy_item_size[in_table->columns[eshift]->dtype];
        std::vector<char> vectNaN =
            RetrieveNaNentry(in_table->columns[eshift]->dtype);
        for (size_t iRow = 0; iRow < nRowOut; iRow++) {
            std::pair<size_t, std::ptrdiff_t> pairShiftRow = get_iRow(iRow);
            //
            if (pairShiftRow.second >= 0) {
                for (uint64_t u = 0; u < siztype; u++)
                    out_arr->data1[siztype * iRow + u] =
                        in_table->columns[pairShiftRow.first]
                            ->data1[siztype * pairShiftRow.second + u];
            } else {
                for (uint64_t u = 0; u < siztype; u++)
                    out_arr->data1[siztype * iRow + u] = vectNaN[u];
            }
        }
    }
    return out_arr;
};


/** This code test if two keys are equal (Before that the hash should have been
 * used) It is used that way because we assume that the left key have the same
 * type as the right keys. The computation is for just one column and it is used
 * keys or the right keys. Equality means that the column/rows are the same.
 *
 * @param arr1 the first column for the comparison
 * @param iRow1 the row of the first key
 * @param arr2 the second columne for the comparison
 * @param iRow2 the row of the second key
 * @return True if they are equal and false otherwise.
 */
bool TestEqualColumn(array_info* arr1, int64_t pos1, array_info* arr2, int64_t pos2)
{
    if (arr1->arr_type == bodo_array_type::NUMPY) {
        // In the case of NUMPY, we compare the values for concluding.
        uint64_t siztype = numpy_item_size[arr1->dtype];
        char* ptr1 = arr1->data1 + siztype * pos1;
        char* ptr2 = arr2->data1 + siztype * pos2;
        if (memcmp(ptr1, ptr2, siztype) != 0) return false;
    }
    if (arr1->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // NULLABLE case. We need to consider the bitmask and the values.
        uint8_t* null_bitmask1 = (uint8_t*)arr1->null_bitmask;
        uint8_t* null_bitmask2 = (uint8_t*)arr2->null_bitmask;
        bool bit1 = GetBit(null_bitmask1, pos1);
        bool bit2 = GetBit(null_bitmask2, pos2);
        // If one bitmask is T and the other the reverse then they are
        // clearly not equal.
        if (bit1 != bit2) return false;
        // If both bitmasks are false, then it does not matter what value
        // they are storing. Comparison is the same as for NUMPY.
        if (bit1) {
            uint64_t siztype = numpy_item_size[arr1->dtype];
            char* ptr1 = arr1->data1 + siztype * pos1;
            char* ptr2 = arr2->data1 + siztype * pos2;
            if (memcmp(ptr1, ptr2, siztype) != 0) return false;
        }
    }
    if (arr1->arr_type == bodo_array_type::STRING) {
        // For STRING case we need to deal bitmask and the values.
        uint8_t* null_bitmask1 = (uint8_t*)arr1->null_bitmask;
        uint8_t* null_bitmask2 = (uint8_t*)arr2->null_bitmask;
        bool bit1 = GetBit(null_bitmask1, pos1);
        bool bit2 = GetBit(null_bitmask2, pos2);
        // If bitmasks are different then we conclude they are not equal.
        if (bit1 != bit2) return false;
        // If bitmasks are both false, then no need to compare the string
        // values.
        if (bit1) {
            // Here we consider the shifts in data2 for the comparison.
            uint32_t* data2_1 = (uint32_t*)arr1->data2;
            uint32_t* data2_2 = (uint32_t*)arr2->data2;
            uint32_t len1 = data2_1[pos1 + 1] - data2_1[pos1];
            uint32_t len2 = data2_2[pos2 + 1] - data2_2[pos2];
            // If string lengths are different then they are different.
            if (len1 != len2) return false;
            // Now we iterate over the characters for the comparison.
            uint32_t pos1_prev = data2_1[pos1];
            uint32_t pos2_prev = data2_2[pos2];
            char* data1_1 = arr1->data1 + pos1_prev;
            char* data1_2 = arr2->data1 + pos2_prev;
            if (memcmp(data1_1, data1_2, len1) != 0) return false;
        }
    }
    return true;
};



/** This code test if two keys are equal (Before that the hash should have been
 * used) It is used that way because we assume that the left key have the same
 * type as the right keys. The shift is used to precise whether we use the left
 * keys or the right keys. Equality means that all the columns are the same.
 * Thus the test iterates over the columns and if one is different then result
 * is false. We consider all types of bodo_array_type
 *
 * @param columns the vector of columns
 * @param n_key the number of keys considered for the comparison
 * @param shift_key1 the column shift for the first key
 * @param iRow1 the row of the first key
 * @param shift_key2 the column for the second key
 * @param iRow2 the row of the second key
 * @return True if they are equal and false otherwise.
 */
bool TestEqual(std::vector<array_info*> const& columns, size_t const& n_key,
               size_t const& shift_key1, size_t const& iRow1,
               size_t const& shift_key2, size_t const& iRow2) {
    // iteration over the list of key for the comparison.
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        bool test = TestEqualColumn(columns[shift_key1 + iKey], iRow1, columns[shift_key2 + iKey], iRow2);
        if (!test)
          return false;
    }
    // If all keys are equal then we are ok and the keys are equals.
    return true;
};

/**
 * The comparison function for integer types.
 *
 * @param ptr1: char* pointer to the first value
 * @param ptr2: char* pointer to the second value
 * @param na_position: true for NaN being last, false for NaN being first (not used)
 * @return 1 if *ptr1 < *ptr2
 */
template <typename T>
inline typename std::enable_if<!std::is_floating_point<T>::value,int>::type NumericComparison_T(char* ptr1, char* ptr2, bool const& na_position) {
    T* ptr1_T = (T*)ptr1;
    T* ptr2_T = (T*)ptr2;
    if (*ptr1_T > *ptr2_T) return -1;
    if (*ptr1_T < *ptr2_T) return 1;
    return 0;
}

/**
 * The comparison function for floating points.
 * If na_position = True then the NaN are considered larger than any other.
 *
 * @param ptr1: char* pointer to the first value
 * @param ptr2: char* pointer to the second value
 * @param na_position: true for NaN being larger, false for NaN being smallest
 * @return 1 if *ptr1 < *ptr2
 */
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value,int>::type NumericComparison_T(char* ptr1, char* ptr2, bool const& na_position) {
    T* ptr1_T = (T*)ptr1;
    T* ptr2_T = (T*)ptr2;
    T val1 = *ptr1_T;
    T val2 = *ptr2_T;
    if (isnan(val1) && isnan(val2)) return 0;
    if (isnan(val2)) {
      if (na_position)
        return 1;
      return -1;
    }
    if (isnan(val1)) {
      if (na_position)
        return -1;
      return 1;
    }
    if (val1 > val2) return -1;
    if (val1 < val2) return 1;
    return 0;
}


/**
 * The comparison function for innteger/floating point
 * If na_position = True then the NaN are considered larger than any other.
 *
 * @param ptr1: char* pointer to the first value
 * @param ptr2: char* pointer to the second value
 * @param na_position: true for NaN being last, false for NaN being first
 * @return 1 if *ptr1 < *ptr2, 0 if equal and -1 if >
 */
int NumericComparison(Bodo_CTypes::CTypeEnum const& dtype, char* ptr1,
                      char* ptr2, bool const& na_position) {
    if (dtype == Bodo_CTypes::_BOOL)
        return NumericComparison_T<bool>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::INT8)
        return NumericComparison_T<int8_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT8)
        return NumericComparison_T<uint8_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::INT16)
        return NumericComparison_T<int16_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT16)
        return NumericComparison_T<uint16_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::INT32)
        return NumericComparison_T<int32_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT32)
        return NumericComparison_T<uint32_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::INT64)
        return NumericComparison_T<int64_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT64)
        return NumericComparison_T<uint64_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::FLOAT32)
        return NumericComparison_T<float>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::FLOAT64)
        return NumericComparison_T<double>(ptr1, ptr2, na_position);
    PyErr_SetString(PyExc_RuntimeError,
                    "Invalid dtype put on input to NumericComparison");
    return 0;
}

/** This code test keys if two keys are greater or equal
 * The code is done so as to give identical results to the Python comparison.
 * It is used that way because we assume that the left key have the same type as
 * the right keys. The shift is used to precise whether we use the left keys or
 * the right keys. 0 means that the columns are equals and 1,-1 that the keys
 * are different. Thus the test iterates over the columns and if one is
 * different then we can conclude. We consider all types of bodo_array_type.
 *
 * @param in_table the input table
 * @param n_key the number of keys considered for the comparison
 * @param vect_ascending the vector of ascending values for the comparison
 * @param shift_key1 the column shift for the first key
 * @param iRow1 the row of the first key
 * @param shift_key2 the column for the second key
 * @param iRow2 the row of the second key
 * @param na_position: if true NaN values are largest, if false smallest.
 * @return true if (shift_key1,iRow1) < (shift_key2,iRow2) , false otherwise
 */
bool KeyComparisonAsPython(std::vector<array_info*> const& columns,
                           size_t const& n_key, int64_t* vect_ascending,
                           size_t const& shift_key1, size_t const& iRow1,
                           size_t const& shift_key2, size_t const& iRow2,
                           bool const& na_position) {
    // iteration over the list of key for the comparison.
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        bool ascending = vect_ascending[iKey];
        auto ProcessOutput=[&](int const& value) -> bool {
            if (ascending) {
                return value > 0;
            }
            return value < 0;
        };
        bool na_position_bis = (!na_position) ^ ascending;
        if (columns[shift_key1 + iKey]->arr_type == bodo_array_type::NUMPY) {
            // In the case of NUMPY, we compare the values for concluding.
            uint64_t siztype = numpy_item_size[columns[shift_key1 + iKey]->dtype];
            char* ptr1 = columns[shift_key1 + iKey]->data1 + (siztype * iRow1);
            char* ptr2 = columns[shift_key2 + iKey]->data1 + (siztype * iRow2);
            int test = NumericComparison(columns[shift_key1 + iKey]->dtype,
                                         ptr1, ptr2, na_position_bis);
            if (test != 0) return ProcessOutput(test);
        }
        if (columns[shift_key1 + iKey]->arr_type ==
            bodo_array_type::NULLABLE_INT_BOOL) {
            // NULLABLE case. We need to consider the bitmask and the values.
            uint8_t* null_bitmask1 =
                (uint8_t*)columns[shift_key1 + iKey]->null_bitmask;
            uint8_t* null_bitmask2 =
                (uint8_t*)columns[shift_key2 + iKey]->null_bitmask;
            bool bit1 = GetBit(null_bitmask1, iRow1);
            bool bit2 = GetBit(null_bitmask2, iRow2);
            // If one bitmask is T and the other the reverse then they are
            // clearly not equal.
            if (bit1 && !bit2) {
              if (na_position_bis) return ProcessOutput(1);
              return ProcessOutput(-1);
            }
            if (!bit1 && bit2) {
              if (na_position_bis) return ProcessOutput(-1);
              return ProcessOutput(1);
            }
            // If both bitmasks are false, then it does not matter what value
            // they are storing. Comparison is the same as for NUMPY.
            if (bit1) {
                uint64_t siztype =
                    numpy_item_size[columns[shift_key1 + iKey]->dtype];
                char* ptr1 =
                    columns[shift_key1 + iKey]->data1 + (siztype * iRow1);
                char* ptr2 =
                    columns[shift_key2 + iKey]->data1 + (siztype * iRow2);
                int test = NumericComparison(columns[shift_key1 + iKey]->dtype,
                                             ptr1, ptr2, na_position_bis);
                if (test != 0) return ProcessOutput(test);
            }
        }
        if (columns[shift_key1 + iKey]->arr_type == bodo_array_type::STRING) {
            // For STRING case we need to deal bitmask and the values.
            uint8_t* null_bitmask1 =
                (uint8_t*)columns[shift_key1 + iKey]->null_bitmask;
            uint8_t* null_bitmask2 =
                (uint8_t*)columns[shift_key2 + iKey]->null_bitmask;
            bool bit1 = GetBit(null_bitmask1, iRow1);
            bool bit2 = GetBit(null_bitmask2, iRow2);
            // If bitmasks are different then we can conclude the comparison
            if (bit1 && !bit2) {
              if (na_position_bis) return ProcessOutput(1);
              return ProcessOutput(-1);
            }
            if (!bit1 && bit2) {
              if (na_position_bis) return ProcessOutput(-1);
              return ProcessOutput(1);
            }
            // If bitmasks are both false, then no need to compare the string
            // values.
            if (bit1) {
                // Here we consider the shifts in data2 for the comparison.
                uint32_t* data2_1 =
                    (uint32_t*)columns[shift_key1 + iKey]->data2;
                uint32_t* data2_2 =
                    (uint32_t*)columns[shift_key2 + iKey]->data2;
                uint32_t len1 = data2_1[iRow1 + 1] - data2_1[iRow1];
                uint32_t len2 = data2_2[iRow2 + 1] - data2_2[iRow2];
                // Compute minimal length
                uint32_t minlen = len1;
                if (len2 < len1) minlen = len2;
                // From the common characters, we may be able to conclude.
                uint32_t pos1_prev = data2_1[iRow1];
                uint32_t pos2_prev = data2_2[iRow2];
                char* data1_1 =
                    (char*)columns[shift_key1 + iKey]->data1 + pos1_prev;
                char* data1_2 =
                    (char*)columns[shift_key2 + iKey]->data1 + pos2_prev;
                int test = std::strncmp(data1_2, data1_1, minlen);
                if (test != 0) return ProcessOutput(test);
                // If not, we may be able to conclude via the string length.
                if (len1 > len2) return ProcessOutput(-1);
                if (len1 < len2) return ProcessOutput(1);
            }
        }
    }
    // If all keys are equal then we return false
    return false;
};

/** This function does the joining of the table and returns the joined
 * table
 *
 * This implementation follows the Shared partition procedure.
 * The data is partitioned and shuffled with the _gen_par_shuffle.
 *
 * The first stage is the partitioning of the data by using hashes array
 * and unordered map array.
 *
 * Afterwards, secondary partitioning is done is the hashes match.
 * Then the pairs of left/right origins are created for subsequent
 * work. If a left key has no matching on the right, then value -1
 * is put (thus the std::ptrdiff_t type is used).
 *
 * External function used are "RetrieveArray" and "TestEqual"
 *
 * We need to merge all the arrays in input because we could not
 * have empty arrays.
 *
 * is_left and is_right correspond
 *   "inner" : is_left = T, is_right = T
 *   "outer" : is_left = F, is_right = F
 *   "left"  : is_left = T, is_right = F
 *   "right" : is_left = F, is_right = T
 *
 * @param in_table : the joined left and right tables.
 * @param n_key_t : the number of columns of keys on input
 * @param n_data_left_t : the number of columns of data on the left
 * @param n_data_right_t : the number of columns of data on the right
 * @param is_left : whether we do merging on the left
 * @param is_right : whether we do merging on the right.
 * @return the returned table used in the code.
 */
table_info* hash_join_table(table_info* in_table, int64_t n_key_t,
                            int64_t n_data_left_t, int64_t n_data_right_t,
                            int64_t* vect_same_key, bool is_left,
                            bool is_right) {
#undef DEBUG_JOIN
#ifdef DEBUG_JOIN
    std::cout << "IN_TABLE:\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
#endif
    size_t n_key = size_t(n_key_t);
    size_t n_data_left = size_t(n_data_left_t);
    size_t n_data_right = size_t(n_data_right_t);
    size_t n_tot_left = n_key + n_data_left;
    size_t n_tot_right = n_key + n_data_right;
    size_t sum_dim = 2 * n_key + n_data_left + n_data_right;
    size_t n_col = in_table->ncols();
    for (size_t iKey=0; iKey<n_key; iKey++)
      CheckEqualityArrayType(in_table->columns[iKey], in_table->columns[n_tot_left + iKey]);
    if (n_col != sum_dim) {
        PyErr_SetString(PyExc_RuntimeError, "incoherent dimensions");
        return NULL;
    }
#ifdef DEBUG_JOIN
    std::cout << "pointer=" << vect_same_key << "\n";
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        int64_t val = vect_same_key[iKey];
        std::cout << "iKey=" << iKey << " vect_same_key[iKey]=" << val << "\n";
    }
#endif
    // This is a hack because we may access vect_same_key_b above n_key
    // even if that is irrelevant to the computation.
    //
    size_t n_rows_left = (size_t)in_table->columns[0]->length;
    size_t n_rows_right = (size_t)in_table->columns[n_tot_left]->length;
    //
    std::vector<array_info*> key_arrs_left = std::vector<array_info*>(
        in_table->columns.begin(), in_table->columns.begin() + n_key);
    uint32_t seed = 0xb0d01288;
    uint32_t* hashes_left = hash_keys(key_arrs_left, seed);

    std::vector<array_info*> key_arrs_right = std::vector<array_info*>(
        in_table->columns.begin() + n_tot_left,
        in_table->columns.begin() + n_tot_left + n_key);
    uint32_t* hashes_right = hash_keys(key_arrs_right, seed);
#ifdef DEBUG_JOIN
    for (size_t i = 0; i < n_rows_left; i++)
        std::cout << "i=" << i << " hashes_left=" << hashes_left[i] << "\n";
    for (size_t i = 0; i < n_rows_right; i++)
        std::cout << "i=" << i << " hashes_right=" << hashes_right[i] << "\n";
#endif
    int ChoiceOpt;
    bool short_table_work,
        long_table_work;  // This corresponds to is_left/is_right
    size_t short_table_shift,
        long_table_shift;  // This corresponds to the shift for left and right.
    size_t short_table_rows, long_table_rows;  // the number of rows
    uint32_t *short_table_hashes, *long_table_hashes;
#ifdef DEBUG_JOIN
    std::cout << "n_rows_left=" << n_rows_left
              << " n_rows_right=" << n_rows_right << "\n";
#endif
    if (n_rows_left < n_rows_right) {
        ChoiceOpt = 0;
        // short = left and long = right
        short_table_work = is_left;
        long_table_work = is_right;
        short_table_shift = 0;
        long_table_shift = n_tot_left;
        short_table_rows = n_rows_left;
        long_table_rows = n_rows_right;
        short_table_hashes = hashes_left;
        long_table_hashes = hashes_right;
    } else {
        ChoiceOpt = 1;
        // short = right and long = left
        short_table_work = is_right;
        long_table_work = is_left;
        short_table_shift = n_tot_left;
        long_table_shift = 0;
        short_table_rows = n_rows_right;
        long_table_rows = n_rows_left;
        short_table_hashes = hashes_right;
        long_table_hashes = hashes_left;
    }
#ifdef DEBUG_JOIN
    std::cout << "ChoiceOpt=" << ChoiceOpt << "\n";
    std::cout << "short_table_rows=" << short_table_rows
              << " long_table_rows=" << long_table_rows << "\n";
#endif
    /* This is a function for comparing the rows.
     * This is the first lambda used as argument for the unordered map container.
     *
     * rows can be in the left or the right tables.
     * If iRow < short_table_rows then it is in the first table.
     * If iRow >= short_table_rows then it is in the second table.
     *
     * Note that the hash is size_t (so 8 bytes on x86-64) while
     * the hashes array are int32_t (so 4 bytes)
     *
     * @param iRow is the first row index for the comparison
     * @return true/false depending on the case.
     */
    std::function<size_t(size_t)> hash_fct = [&](size_t iRow) -> size_t {
        if (iRow < short_table_rows)
            return short_table_hashes[iRow];
        else
            return long_table_hashes[iRow - short_table_rows];
    };
    /* This is a function for testing equality of rows.
     * This is used as second argument for the unordered map container.
     *
     * rows can be in the left or the right tables.
     * If iRow < short_table_rows then it is in the first table.
     * If iRow >= short_table_rows then it is in the second table.
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true/false depending on equality or not.
     */
    std::function<bool(size_t,size_t)> equal_fct = [&](size_t iRowA, size_t iRowB) -> bool {
        size_t jRowA, jRowB;
        size_t shift_A, shift_B;
        if (iRowA < short_table_rows) {
            shift_A = short_table_shift;
            jRowA = iRowA;
        } else {
            shift_A = long_table_shift;
            jRowA = iRowA - short_table_rows;
        }
        if (iRowB < short_table_rows) {
            shift_B = short_table_shift;
            jRowB = iRowB;
        } else {
            shift_B = long_table_shift;
            jRowB = iRowB - short_table_rows;
        }
        return TestEqual(in_table->columns, n_key, shift_A, jRowA, shift_B,
                         jRowB);
    };
    // The entList contains the hash of the short table.
    // We address the entry by the row index. We store all the rows which are
    // identical in the std::vector.
    MAP_CONTAINER <size_t, std::vector<size_t>, std::function<size_t(size_t)>,std::function<bool(size_t,size_t)>> entList({}, hash_fct, equal_fct);
    // The loop over the short table.
    // entries are stored one by one and all of them are put even if identical
    // in value.
    for (size_t i_short = 0; i_short < short_table_rows; i_short++) {
#ifdef DEBUG_JOIN
        std::cout << "i_short=" << i_short << "\n";
#endif
        std::vector<size_t>& group = entList[i_short];
        group.emplace_back(i_short);
    }
    size_t nEnt = entList.size();
#ifdef DEBUG_JOIN
    std::cout << "nEnt=" << nEnt << "\n";
#endif
    //
    // Now iterating and determining how many entries we have to do.
    //

    //
    // ListPairWrite is the table used for the output
    // It precises the index used for the writing of the output table.
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
    // This precise whether a short table entry has been used or not.
    std::vector<int> ListStatus(nEnt, 0);
    // We now iterate over all entries of the long table in order to get
    // the entries in the ListPairWrite.
    for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
        size_t i_long_shift = i_long + short_table_rows;
        auto iter = entList.find(i_long_shift);
        if (iter == entList.end()) {
            if (long_table_work) ListPairWrite.push_back({-1, i_long});
        } else {
            // If the short table entry are present in output as well, then
            // we need to keep track whether they are used or not by the long
            // table.
            if (short_table_work) {
                auto index = std::distance(entList.begin(), iter);
                ListStatus[index] = 1;
            }
            for (auto& j_short : iter->second)
                ListPairWrite.push_back({j_short, i_long});
        }
    }
    // if short_table is in output then we need to check
    // if they are used by the long table and if so use them on output.
    if (short_table_work) {
        auto iter = entList.begin();
        size_t iter_s = 0;
        while (iter != entList.end()) {
            if (ListStatus[iter_s] == 0) {
                for (auto& j_short : iter->second) {
                    ListPairWrite.push_back({j_short, -1});
                }
            }
            iter++;
            iter_s++;
        }
#ifdef DEBUG_JOIN
        std::cout << "AFTER : iter_s=" << iter_s << "\n";
#endif
    }
#ifdef DEBUG_JOIN
    size_t nbPair = ListPairWrite.size();
    for (size_t iPair = 0; iPair < nbPair; iPair++)
        std::cout << "iPair=" << iPair
                  << " ePair=" << ListPairWrite[iPair].first << " , "
                  << ListPairWrite[iPair].second << "\n";
#endif
    std::vector<array_info*> out_arrs;
    // Inserting the left data
    for (size_t i = 0; i < n_tot_left; i++) {
        if (i < n_key && vect_same_key[i < n_key ? i : 0] == 1) {
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(RetrieveArray(in_table, ListPairWrite, i,
                                                 n_tot_left + i, 2));
            } else {
                out_arrs.emplace_back(RetrieveArray(in_table, ListPairWrite,
                                                 n_tot_left + i, i, 2));
            }
        } else {
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(
                    RetrieveArray(in_table, ListPairWrite, i, -1, 0));
            } else {
                out_arrs.emplace_back(
                    RetrieveArray(in_table, ListPairWrite, -1, i, 1));
            }
        }
    }
    // Inserting the right data
    for (size_t i = 0; i < n_tot_right; i++) {
        if (i < n_key && vect_same_key[i < n_key ? i : 0] == 1) {
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(RetrieveArray(in_table, ListPairWrite, i,
                                                 n_tot_left + i, 2));
            } else {
                out_arrs.emplace_back(RetrieveArray(in_table, ListPairWrite,
                                                 n_tot_left + i, i, 2));
            }
        } else {
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(RetrieveArray(in_table, ListPairWrite, -1,
                                                 n_tot_left + i, 1));
            } else {
                out_arrs.emplace_back(RetrieveArray(in_table, ListPairWrite,
                                                 n_tot_left + i, -1, 0));
            }
        }
    }
#ifdef DEBUG_JOIN
    std::cout << "hash_join_table, output information\n";
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
    DEBUG_PrintRefct(std::cout, out_arrs);
    std::cout << "Finally leaving\n";
#endif
    //
    delete[] hashes_left;
    delete[] hashes_right;
    return new table_info(out_arrs);
}






/**
 * Enum of aggregation, combine and eval functions used by groubpy.
 * Some functions like sum can be used for multiple purposes, like aggregation
 * and combine. Some operations like sum don't need eval.
 */
struct Bodo_FTypes {
    // !!! IMPORTANT: this is supposed to match the positions in
    // supported_agg_funcs in aggregate.py
    enum FTypeEnum {
        sum,
        count,
        nunique,
        median,
        cumsum,
        cumprod,
        mean,
        min,
        max,
        prod,
        var,
        std,
        udf,
        num_funcs,  // num_funcs is used to know how many functions up to this
                    // point
        mean_eval,
        var_eval,
        std_eval
    };
};

/**
 * Function pointer for groupby update and combine operations that are
 * executed in JIT-compiled code (also see udfinfo_t).
 *
 * @param input table
 * @param output table
 * @param row to group mapping (tells to which group -row in output table-
          the row i of input table goes to)
 */
typedef void (*udf_table_op_fn)(table_info* in_table, table_info* out_table,
                                int64_t* row_to_group);
/**
 * Function pointer for groupby eval operation that is executed in JIT-compiled
 * code (also see udfinfo_t).
 *
 * @param table containing the output columns and reduction variables columns
 */
typedef void (*udf_eval_fn)(table_info*);

/*
 * This struct stores info that is used when groupby.agg() has JIT-compiled
 * user-defined functions. Such JIT-compiled code will be invoked by the C++
 * library via function pointers.
 */
struct udfinfo_t {
    /*
     * This empty table is used to tell the C++ library the types to use
     * to allocate the columns (output and redvar) for udfs
     */
    table_info* udf_table_dummy;
    /*
     * Function pointer to "update" code which performs the initial
     * local groupby and aggregation.
     */
    udf_table_op_fn update;
    /*
     * Function pointer to "combine" code which combines the results
     * after shuffle.
     */
    udf_table_op_fn combine;
    /*
     * Function pointer to "eval" code which performs post-processing and
     * sets the final output value for each group.
     */
    udf_eval_fn eval;
};

/**
 * This template is used for functions that take two values of the same dtype.
 */
template <typename T, int ftype, typename Enable = void>
struct aggfunc {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    static void apply(T& v1, T& v2) {}
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::sum,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    /**
     * Aggregation function for sum. Modifies current sum if value is not a nan
     *
     * @param[in,out] current sum value, and holds the result
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 += v2; }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::sum,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) v1 += v2;
    }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::min,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    /**
     * Aggregation function for min. Modifies current min if value is not a nan
     *
     * @param[in,out] current min value (or nan for floats if no min value found
     * yet)
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 = std::min(v1, v2); }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::min,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2))
            v1 = std::min(v2, v1);  // std::min(x,NaN) = x
                                    // (v1 is initialized as NaN)
    }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::max,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    /**
     * Aggregation function for max. Modifies current max if value is not a nan
     *
     * @param[in,out] current max value (or nan for floats if no max value found
     * yet)
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 = std::max(v1, v2); }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::max,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) {
            v1 = std::max(v2, v1);  // std::max(x,NaN) = x
                                    // (v1 is initialized as NaN)
        }
    }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::prod,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    /**
     * Aggregation function for product. Modifies current product if value is
     * not a nan
     *
     * @param[in,out] current product
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 *= v2; }
};

template<>
struct aggfunc<bool, Bodo_FTypes::prod> {
    static void apply(bool& v1, bool& v2) { v1 = v1 && v2; }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::prod,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) v1 *= v2;
    }
};


template <int ftype, typename Enable = void>
struct aggstring {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    static void apply(std::string& v1, std::string& v2) {}
};



template<>
struct aggstring<Bodo_FTypes::sum> {
    static void apply(std::string& v1, std::string& v2) {
        v1 += v2;
    }
};

template<>
struct aggstring<Bodo_FTypes::min> {
    static void apply(std::string& v1, std::string& v2) {
        v1 = std::min(v1, v2);
    }
};

template<>
struct aggstring<Bodo_FTypes::max> {
    static void apply(std::string& v1, std::string& v2) {
        v1 = std::max(v1, v2);
    }
};





template <typename T, typename Enable = void>
struct count_agg {
    /**
     * Aggregation function for count. Increases count if value is not a nan
     *
     * @param[in,out] current count
     * @param second input value.
     */
    static void apply(int64_t& v1, T& v2);
};

template <typename T>
struct count_agg<
    T, typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    static void apply(int64_t& v1, T& v2) { v1 += 1; }
};

template <typename T>
struct count_agg<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(int64_t& v1, T& v2) {
        if (!isnan(v2)) v1 += 1;
    }
};

template <typename T, typename Enable = void>
struct mean_agg {
    /**
     * Aggregation function for mean. Modifies count and sum of observed input
     * values
     *
     * @param[in,out] contains the current sum of observed values
     * @param an observed input value
     * @param[in,out] count: current number of observations
     */
    static void apply(double& v1, T& v2, uint64_t& count);
};

template <typename T>
struct mean_agg<
    T, typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    static void apply(double& v1, T& v2, uint64_t& count) {
        v1 += (double)v2;
        count += 1;
    }
};

template <typename T>
struct mean_agg<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(double& v1, T& v2, uint64_t& count) {
        if (!isnan(v2)) {
            v1 += (double)v2;
            count += 1;
        }
    }
};

/**
 * Final evaluation step for mean, which calculates the mean based on the
 * sum of observed values and the number of values.
 *
 * @param[in,out] sum of observed values, will be modified to contain the mean
 * @param count: number of observations
 */
static void mean_eval(double& result, uint64_t& count) { result /= count; }

template <typename T, typename Enable = void>
struct var_agg {
    /**
     * Aggregation function for variance. Modifies count, mean and m2 (sum of
     * squares of differences from the current mean) based on the observed input
     * values. See
     * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
     * for more information.
     *
     * @param[in] observed value
     * @param[in,out] count: current number of observations
     * @param[in,out] mean_x: current mean
     * @param[in,out] m2: sum of squares of differences from the current mean
     */
    static void apply(T& v2, uint64_t& count, double& mean_x, double& m2);
};

template <typename T>
struct var_agg<
    T, typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    inline static void apply(T& v2, uint64_t& count, double& mean_x,
                             double& m2) {
        count += 1;
        double delta = (double)v2 - mean_x;
        mean_x += delta / count;
        double delta2 = (double)v2 - mean_x;
        m2 += delta * delta2;
    }
};

template <typename T>
struct var_agg<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    inline static void apply(T& v2, uint64_t& count, double& mean_x,
                             double& m2) {
        if (!isnan(v2)) {
            count += 1;
            double delta = (double)v2 - mean_x;
            mean_x += delta / count;
            double delta2 = (double)v2 - mean_x;
            m2 += delta * delta2;
        }
    }
};

/**
 * Perform combine operation for variance, which for a set of rows belonging
 * to the same group with count (# observations), mean and m2, reduces
 * the values to one row. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param input array of counts (can include multiple values per group and
 * multiple groups)
 * @param input array of means (can include multiple values per group and
 * multiple groups)
 * @param input array of m2 (can include multiple values per group and multiple
 * groups)
 * @param output array of counts (one row per group)
 * @param output array of means (one row per group)
 * @param output array of m2 (one row per group)
 * @param maps row numbers in input arrays to row number in output array
 */
void var_combine(array_info* count_col_in, array_info* mean_col_in,
                 array_info* m2_col_in, array_info* count_col_out,
                 array_info* mean_col_out, array_info* m2_col_out,
                 const std::vector<int64_t>& row_to_group) {
    for (int64_t i = 0; i < count_col_in->length; i++) {
        uint64_t& count_a = count_col_out->at<uint64_t>(row_to_group[i]);
        uint64_t& count_b = count_col_in->at<uint64_t>(i);
        double& mean_a = mean_col_out->at<double>(row_to_group[i]);
        double& mean_b = mean_col_in->at<double>(i);
        double& m2_a = m2_col_out->at<double>(row_to_group[i]);
        double& m2_b = m2_col_in->at<double>(i);

        uint64_t count = count_a + count_b;
        double delta = mean_b - mean_a;
        mean_a = (count_a * mean_a + count_b * mean_b) / count;
        m2_a = m2_a + m2_b + delta * delta * count_a * count_b / count;
        count_a = count;
    }
}

/**
 * Perform final evaluation step for variance, which calculates the variance
 * based on the count and m2 values. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param[in,out] stores the calculated variance
 * @param count: number of observations
 * @param m2: sum of squares of differences from the current mean
 */
static void var_eval(double& result, uint64_t& count, double& m2) {
    if (count == 0)
        result = std::numeric_limits<double>::quiet_NaN();
    else
        result = m2 / (count - 1);
}

/**
 * Perform final evaluation step for std, which calculates the standard
 * deviation based on the count and m2 values. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param[in,out] stores the calculated std
 * @param count: number of observations
 * @param m2: sum of squares of differences from the current mean
 */
static void std_eval(double& result, uint64_t& count, double& m2) {
    if (count == 0)
        result = std::numeric_limits<double>::quiet_NaN();
    else
        result = sqrt(m2 / (count - 1));
}




/** Data structure used for the computation of groups.

    @data row_to_group       : This takes the index and returns the group
    @data group_to_first_row : This takes the group index and return the first row index.
    @data next_row_in_group  : for a row in the list returns the next row in the list if existent.
                               if non-existent value is -1.
    @data list_missing       : list of rows which are missing and NaNs.

    This is only one data structure but it has two use cases.
    -- get_group_info computes only the entries row_to_group and group_to_first_row.
       This is the data structure used for groupby operations such as sum, mean, etc. for which
       the full group structure does not need to be known.
    -- get_group_info_iterate computes all the entries. This is needed for some operations such
       as nunique, median, cumsum, cumprod.
       The entry list_missing is computed only for cumsum and cumprod and computed only if needed.
 */
struct grouping_info {
    std::vector<int64_t> row_to_group;
    std::vector<int64_t> group_to_first_row;
    std::vector<int64_t> next_row_in_group;
    std::vector<int64_t> list_missing;
};




/**
 * Apply a function to a column(s), save result to (possibly reduced) output
 * column(s) Semantics of this function right now vary depending on function
 * type (ftype).
 *
 * @param column containing input values
 * @param output column
 * @param auxiliary input/output columns used for mean, var, std
 * @param maps row numbers in input columns to group numbers (for reduction
 * operations)
 */
template <typename T, int ftype>
void apply_to_column(array_info* in_col, array_info* out_col,
                     std::vector<array_info*>& aux_cols,
                     const grouping_info &grp_info) {
    switch (in_col->arr_type) {
        case bodo_array_type::NUMPY:
            if (ftype == Bodo_FTypes::mean) {
                array_info* count_col = aux_cols[0];
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        mean_agg<T>::apply(
                            out_col->at<double>(grp_info.row_to_group[i]),
                            in_col->at<T>(i),
                            count_col->at<uint64_t>(grp_info.row_to_group[i]));
            } else if (ftype == Bodo_FTypes::mean_eval) {
                for (int64_t i = 0; i < in_col->length; i++)
                    mean_eval(out_col->at<double>(i), in_col->at<uint64_t>(i));
            } else if (ftype == Bodo_FTypes::var) {
                array_info* count_col = aux_cols[0];
                array_info* mean_col = aux_cols[1];
                array_info* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        var_agg<T>::apply(
                            in_col->at<T>(i),
                            count_col->at<uint64_t>(grp_info.row_to_group[i]),
                            mean_col->at<double>(grp_info.row_to_group[i]),
                            m2_col->at<double>(grp_info.row_to_group[i]));
            } else if (ftype == Bodo_FTypes::var_eval) {
                array_info* count_col = aux_cols[0];
                array_info* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++)
                    var_eval(out_col->at<double>(i), count_col->at<uint64_t>(i),
                             m2_col->at<double>(i));
            } else if (ftype == Bodo_FTypes::std_eval) {
                array_info* count_col = aux_cols[0];
                array_info* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++)
                    std_eval(out_col->at<double>(i), count_col->at<uint64_t>(i),
                             m2_col->at<double>(i));
            } else if (ftype == Bodo_FTypes::count) {
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        count_agg<T>::apply(
                            out_col->at<int64_t>(grp_info.row_to_group[i]),
                            in_col->at<T>(i));
            } else {
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        aggfunc<T, ftype>::apply(
                            out_col->at<T>(grp_info.row_to_group[i]), in_col->at<T>(i));
            }
            return;
        // for strings, we are only supporting count for now, and count function
        // works for strings because the input value doesn't matter
        case bodo_array_type::STRING:
            switch (ftype) {
                case Bodo_FTypes::count:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            count_agg<T>::apply(
                                out_col->at<int64_t>(grp_info.row_to_group[i]),
                                in_col->at<T>(i));
                    }
                    return;
                default:
                    size_t num_groups = grp_info.group_to_first_row.size();
                    std::vector<std::string> ListString(num_groups);
                    char* data = in_col->data1;
                    uint32_t* offsets = (uint32_t*)in_col->data2;
                    uint8_t* null_bitmask_i = (uint8_t*)in_col->null_bitmask;
                    uint8_t* null_bitmask_o = (uint8_t*)out_col->null_bitmask;
                    // Computing the strings used in output.
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) && GetBit(null_bitmask_i, i)) {
                            uint32_t start_offset = offsets[i];
                            uint32_t end_offset = offsets[i+1];
                            uint32_t len = end_offset - start_offset;
                            int64_t i_grp = grp_info.row_to_group[i];
                            std::string val(&data[start_offset], len);
                            if (GetBit(null_bitmask_o, i_grp)) {
                                aggstring<ftype>::apply(ListString[i_grp], val);
                            } else {
                                ListString[i_grp] = val;
                                SetBitTo(null_bitmask_o, i_grp, true);
                            }
                        }
                    }
                    // Determining the number of characters in output.
                    size_t nb_char = 0;
                    for (int64_t i_grp = 0; i_grp < int64_t(num_groups); i_grp++) {
                        if (GetBit(null_bitmask_o, i_grp))
                            nb_char += ListString[i_grp].size();
                    }
                    // Doing the additional needed allocations
                    delete [] out_col->data1;
                    out_col->data1 = new char[nb_char];
                    out_col->n_sub_elems = nb_char;
                    // Writing the strings in output
                    char* data_o = out_col->data1;
                    uint32_t* offsets_o = (uint32_t*)out_col->data2;
                    uint32_t pos = 0;
                    for (int64_t i_grp = 0; i_grp < int64_t(num_groups); i_grp++) {
                        offsets_o[i_grp] = pos;
                        if (GetBit(null_bitmask_o, i_grp)) {
                            int len = ListString[i_grp].size();
                            memcpy(data_o, ListString[i_grp].data(), len);
                            data_o += len;
                            pos += len;
                        }
                    }
                    offsets_o[num_groups] = pos;
                    return;
            }
        case bodo_array_type::NULLABLE_INT_BOOL:
            switch (ftype) {
                case Bodo_FTypes::count:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            count_agg<T>::apply(
                                out_col->at<int64_t>(grp_info.row_to_group[i]),
                                in_col->at<T>(i));
                    }
                    return;
                case Bodo_FTypes::mean:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i)) {
                            mean_agg<T>::apply(
                                out_col->at<double>(grp_info.row_to_group[i]),
                                in_col->at<T>(i),
                                aux_cols[0]->at<uint64_t>(grp_info.row_to_group[i]));
                        }
                    }
                    return;
                case Bodo_FTypes::var:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            var_agg<T>::apply(
                                in_col->at<T>(i),
                                aux_cols[0]->at<uint64_t>(grp_info.row_to_group[i]),
                                aux_cols[1]->at<double>(grp_info.row_to_group[i]),
                                aux_cols[2]->at<double>(grp_info.row_to_group[i]));
                    }
                    return;
                default:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i)) {
                            aggfunc<T, ftype>::apply(
                                out_col->at<T>(grp_info.row_to_group[i]),
                                in_col->at<T>(i));
                            SetBitTo((uint8_t*)out_col->null_bitmask,
                                     grp_info.row_to_group[i], true);
                        }
                    }
                    return;
            }
        default:
            PyErr_SetString(PyExc_RuntimeError,
                            "apply_to_column: incorrect array type");
            return;
    }
}




/**
 * Invokes the correct template instance of apply_to_column depending on
 * function (ftype) and dtype. See 'apply_to_column'
 *
 * @param column containing input values
 * @param output column
 * @param auxiliary input/output columns used for mean, var, std
 * @param maps row numbers in input columns to group numbers (for reduction
 * operations)
 * @param function to apply
 */
void do_apply_to_column(array_info* in_col, array_info* out_col,
                        std::vector<array_info*>& aux_cols,
                        const grouping_info &grp_info, int ftype) {
    if (in_col->arr_type == bodo_array_type::STRING) {
        switch (ftype) {
            // NOTE: The int template argument is not used in this call to apply_to_column
            case Bodo_FTypes::sum:
                return apply_to_column<int, Bodo_FTypes::sum>(
                    in_col, out_col, aux_cols, grp_info);
            case Bodo_FTypes::min:
                return apply_to_column<int, Bodo_FTypes::min>(
                    in_col, out_col, aux_cols, grp_info);
            case Bodo_FTypes::max:
                return apply_to_column<int, Bodo_FTypes::max>(
                    in_col, out_col, aux_cols, grp_info);
        }
    }
    if (ftype == Bodo_FTypes::count) {
        switch (in_col->dtype) {
            case Bodo_CTypes::FLOAT32:
                // data will only be used to check for nans
                return apply_to_column<float, Bodo_FTypes::count>(
                    in_col, out_col, aux_cols, grp_info);
            case Bodo_CTypes::FLOAT64:
                // data will only be used to check for nans
                return apply_to_column<double, Bodo_FTypes::count>(
                    in_col, out_col, aux_cols, grp_info);
            default:
                // data will be ignored in this case, so type doesn't matter
                return apply_to_column<int8_t, Bodo_FTypes::count>(
                    in_col, out_col, aux_cols, grp_info);
        }
    }
    
    switch (in_col->dtype) {
        case Bodo_CTypes::_BOOL:
            switch (ftype) {
                case Bodo_FTypes::min:
                    return apply_to_column<bool, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<bool, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<bool, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                default:
                    PyErr_SetString(PyExc_RuntimeError,
                        "unsuported aggregation for boolean type column");
                    return;
            }
        case Bodo_CTypes::INT8:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int8_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int8_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int8_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int8_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int8_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int8_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT8:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint8_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint8_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint8_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint8_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint8_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint8_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int16_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int16_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int16_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int16_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int16_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int16_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint16_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint16_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint16_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint16_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint16_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint16_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int32_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int32_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int32_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int32_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int32_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int32_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint32_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint32_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint32_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint32_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint32_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint32_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint64_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint64_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint64_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint64_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint64_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint64_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::FLOAT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<float, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<float, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<float, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<float, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<float, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<float, Bodo_FTypes::mean_eval>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<float, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<float, Bodo_FTypes::var_eval>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<float, Bodo_FTypes::std_eval>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::FLOAT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<double, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<double, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<double, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<double, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<double, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<double, Bodo_FTypes::mean_eval>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<double, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<double, Bodo_FTypes::var_eval>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<double, Bodo_FTypes::std_eval>(
                        in_col, out_col, aux_cols, grp_info);
            }
        default:
            fprintf(stderr, "do_apply_to_column: invalid array dtype\n");
            return;
    }
}

/**
 * Multi column key used for hashing keys to determine group membership in
 * groupby
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
                case bodo_array_type::NULLABLE_INT_BOOL:
                    if (GetBit((uint8_t*)c1->null_bitmask, row) !=
                        GetBit((uint8_t*)c2->null_bitmask, other.row))
                        return false;
                    if (!GetBit((uint8_t*)c1->null_bitmask, row)) continue;
                case bodo_array_type::NUMPY:
                    siztype = numpy_item_size[c1->dtype];
                    if (memcmp(c1->data1 + siztype * row,
                               c2->data1 + siztype * other.row, siztype) != 0) {
                        return false;
                    }
                    continue;
                case bodo_array_type::STRING:
                    uint32_t* c1_offsets = (uint32_t*)c1->data2;
                    uint32_t* c2_offsets = (uint32_t*)c2->data2;
                    uint32_t c1_str_len = c1_offsets[row + 1] - c1_offsets[row];
                    uint32_t c2_str_len =
                        c2_offsets[other.row + 1] - c2_offsets[other.row];
                    if (c1_str_len != c2_str_len) return false;
                    char* c1_str = c1->data1 + c1_offsets[row];
                    char* c2_str = c2->data1 + c2_offsets[other.row];
                    if (strncmp(c1_str, c2_str, c1_str_len) != 0) return false;
            }
        }
        return true;
    }
};

struct key_hash {
    std::size_t operator()(const multi_col_key& k) const { return k.hash; }
};



bool does_keys_have_nulls(std::vector<array_info*> const& key_cols)
{
  for (auto key_col : key_cols) {
    if ((key_col->arr_type == bodo_array_type::NUMPY &&
         (key_col->dtype == Bodo_CTypes::FLOAT32 ||
          key_col->dtype == Bodo_CTypes::FLOAT64)) ||
        key_col->arr_type == bodo_array_type::STRING ||
        key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
      return true;
    }
  }
  return false;
}


bool does_row_has_nulls(std::vector<array_info*> const& key_cols, int64_t const& i)
{
  for (auto key_col : key_cols) {
    if (key_col->arr_type == bodo_array_type::STRING ||
        key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
      if (!GetBit((uint8_t*)key_col->null_bitmask, i)) return true;
    } else if (key_col->arr_type == bodo_array_type::NUMPY) {
      if ((key_col->dtype == Bodo_CTypes::FLOAT32 &&
           isnan(key_col->at<float>(i))) ||
          (key_col->dtype == Bodo_CTypes::FLOAT64 &&
           isnan(key_col->at<double>(i)))) return true;
    }
  }
  return false;
}

/**
 * Given a table with n key columns, this function calculates the row to group
 * mapping for every row based on its key.
 * For every row in the table, this only does *one* lookup in the hash map.
 *
 * @param the table
 * @param[out] vector that maps row number in the table to a group number
 * @param[out] vector that maps group number to the first row in the table
 *                that belongs to that group
 */
void get_group_info(table_info& table, std::vector<int64_t>& row_to_group,
                    std::vector<int64_t>& group_to_first_row,
                    bool check_for_null_keys) {
    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table.columns.begin(), table.columns.begin() + table.num_keys);
    uint32_t seed = 0xb0d01288;
    uint32_t* hashes = hash_keys(key_cols, seed);

    row_to_group.reserve(table.nrows());
    // start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values I record in the output go from
    // 0 to num_groups - 1)
    int next_group = 1;
    MAP_CONTAINER <multi_col_key, int64_t, key_hash> key_to_group;
    bool key_is_nullable = false;
    if (check_for_null_keys) {
      key_is_nullable = does_keys_have_nulls(key_cols);
    }
    for (int64_t i = 0; i < table.nrows(); i++) {
        if (key_is_nullable) {
            if (does_row_has_nulls(key_cols, i)) {
                row_to_group.emplace_back(-1);
                continue;
            }
        }
        multi_col_key key(hashes[i], &table, i);
        int64_t& group = key_to_group[key];  // this inserts 0 into the map if
                                             // key doesn't exist
        if (group == 0) {
            group = next_group++;  // this updates the value in the map without
                                   // another lookup
            group_to_first_row.emplace_back(i);
        }
        row_to_group.emplace_back(group - 1);
    }
    delete[] hashes;
}

/**
 * Given a table with n key columns, this function calculates the row to group
 * mapping for every row based on its key.
 * For every row in the table, this only does *one* lookup in the hash map.
 *
 * @param            table: the table
 * @param consider_missing: whether to return the list of missing rows or not
 * @return vector that maps group number to the first row in the table
 *                that belongs to that group
 */
grouping_info get_group_info_iterate(table_info* table, bool consider_missing) {
    std::vector<int64_t> row_to_group(table->nrows());
    std::vector<int64_t> group_to_first_row;
    std::vector<int64_t> next_row_in_group(table->nrows(), -1);
    std::vector<int64_t> active_group_repr;
    std::vector<int64_t> list_missing;

    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table->columns.begin(), table->columns.begin() + table->num_keys);
    uint32_t seed = 0xb0d01288;
    uint32_t* hashes = hash_keys(key_cols, seed);

    bool key_is_nullable = does_keys_have_nulls(key_cols);
    // start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values I record in the output go from
    // 0 to num_groups - 1)
    int next_group = 1;
    MAP_CONTAINER <multi_col_key, int64_t, key_hash> key_to_group;
    for (int64_t i = 0; i < table->nrows(); i++) {
        if (key_is_nullable) {
            if (does_row_has_nulls(key_cols, i)) {
                row_to_group[i] = -1;
                if (consider_missing)
                  list_missing.push_back(i);
                continue;
            }
        }
        multi_col_key key(hashes[i], table, i);
        int64_t& group = key_to_group[key];  // this inserts 0 into the map if
                                             // key doesn't exist
        if (group == 0) {
            group = next_group++;  // this updates the value in the map without
                                   // another lookup
            group_to_first_row.emplace_back(i);
            active_group_repr.emplace_back(i);
        } else {
            int64_t prev_elt = active_group_repr[group - 1];
            next_row_in_group[prev_elt] = i;
            active_group_repr[group - 1] = i;
        }
        row_to_group[i] = group - 1;
    }
    delete[] hashes;
    return {std::move(row_to_group), std::move(group_to_first_row), std::move(next_row_in_group), std::move(list_missing)};
}

/**
 * Initialize an output column that will be used to store the result of an
 * aggregation function. Initialization depends on the function:
 * default: zero initialization
 * prod: 1
 * min: max dtype value, or quiet_NaN if float (so that result is nan if all
 * input values are nan) max: min dtype value, or quiet_NaN if float (so that
 * result is nan if all input values are nan)
 *
 * @param output column
 * @param function identifier
 */
void aggfunc_output_initialize(array_info* out_col, int ftype) {
    if (out_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        if (ftype == Bodo_FTypes::min || ftype == Bodo_FTypes::max)
            // if input is all nulls, max and min output will be null
            InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length, false);
        else
            // for other functions (count, sum, etc.) output will never be null
            InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length, true);
    }
    if (out_col->arr_type == bodo_array_type::STRING) {
        InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length, false);
    }
    switch (ftype) {
        case Bodo_FTypes::prod:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    std::fill((bool*)out_col->data1,
                              (bool*)out_col->data1 + out_col->length, true);
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1,
                              (int8_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1,
                              (uint8_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1,
                              (int16_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1,
                              (uint16_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1,
                              (int32_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1,
                              (uint32_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1,
                              (uint64_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::FLOAT32:
                    std::fill((float*)out_col->data1,
                              (float*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::FLOAT64:
                    std::fill((double*)out_col->data1,
                              (double*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::STRING:
                default:
                    PyErr_SetString(PyExc_RuntimeError,
                                    "unsupported/not implemented");
                    return;
            }
        case Bodo_FTypes::min:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    std::fill((bool*)out_col->data1,
                              (bool*)out_col->data1 + out_col->length,
                              true);
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1,
                              (int8_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int8_t>::max());
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1,
                              (uint8_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint8_t>::max());
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1,
                              (int16_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int16_t>::max());
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1,
                              (uint16_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint16_t>::max());
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1,
                              (int32_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int32_t>::max());
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1,
                              (uint32_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint32_t>::max());
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1,
                              (uint64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint64_t>::max());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1,
                              (float*)out_col->data1 + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1,
                              (double*)out_col->data1 + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                case Bodo_CTypes::STRING:
                    // Nothing to initilize with in the case of strings.
                    return;
                default:
                    PyErr_SetString(PyExc_RuntimeError,
                                    "unsupported/not implemented");
                    return;
            }
        case Bodo_FTypes::max:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    std::fill((bool*)out_col->data1,
                              (bool*)out_col->data1 + out_col->length,
                              false);
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1,
                              (int8_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int8_t>::min());
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1,
                              (uint8_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint8_t>::min());
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1,
                              (int16_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int16_t>::min());
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1,
                              (uint16_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint16_t>::min());
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1,
                              (int32_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int32_t>::min());
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1,
                              (uint32_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint32_t>::min());
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1,
                              (uint64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint64_t>::min());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1,
                              (float*)out_col->data1 + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1,
                              (double*)out_col->data1 + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                case Bodo_CTypes::STRING:
                    // nothing to initialize in the case of strings
                    return;
                default:
                    PyErr_SetString(PyExc_RuntimeError,
                                    "unsupported/not implemented");
                    return;
            }
        default:
            // zero initialize
            memset(out_col->data1, 0,
                   numpy_item_size[out_col->dtype] * out_col->length);
    }
}

/**
 * Returns the array type and dtype required for output columns based on the
 * aggregation function and input dtype.
 *
 * @param function identifier
 * @param[in,out] array type (caller sets a default, this function only changes
 * in certain cases)
 * @param[in,out] output dtype (caller sets a default, this function only
 * changes in certain cases)
 * @param true if column is key column (in this case ignore because output type
 * will be the same)
 */
void get_groupby_output_dtype(int ftype,
                              bodo_array_type::arr_type_enum& array_type,
                              Bodo_CTypes::CTypeEnum& dtype, bool is_key) {
    if (is_key) return;
    switch (ftype) {
        case Bodo_FTypes::nunique:
        case Bodo_FTypes::count:
            array_type = bodo_array_type::NUMPY;
            dtype = Bodo_CTypes::INT64;
            return;
        case Bodo_FTypes::median:
        case Bodo_FTypes::mean:
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
            array_type = bodo_array_type::NUMPY;
            dtype = Bodo_CTypes::FLOAT64;
            return;
        default:
            return;
    }
}

std::vector<Bodo_FTypes::FTypeEnum> combine_funcs(Bodo_FTypes::num_funcs);


/*
 An instance of GroupbyPipeline class manages a groupby operation. In a
 groupby operation, an arbitrary number of functions can be applied to each
 input column. The functions can vary between input columns. Each combination
 of (input column, function) is an operation that produces a column in the
 output table. The computation of each (input column, function) pair is
 encapsulated in what is called a "column set" (for lack of a better name).
 There are different column sets for different types of operations (e.g. var,
 mean, median, udfs, basic operations...). Each column set creates,
 initializes, operates on and manages the arrays needed to perform its
 computation. Different column set types may require different number of
 columns and dtypes. The main control flow of groupby is in
 GroupbyPipeline::run(). It invokes update, shuffle, combine and eval steps
 (as needed), and these steps iterate through the column sets and invoke
 their operations.
*/

/*
 * This is the base column set class which is used by most operations (like
 * sum, prod, count, etc.). Several subclasses also rely on some of the methods
 * of this base class.
 */
class BasicColSet {
public:

    /**
     * Construct column set corresponding to function of type ftype applied to
     * the input column in_col
     * @param input column of groupby associated with this column set
     * @param ftype associated with this column set
     * @param tells the column set whether GroupbyPipeline is going to perform
     *        a combine operation or not. If false, this means that either
     *        shuffling is not necessary or that it will be done at the
     *        beginning of the pipeline.
     */
    BasicColSet(array_info *in_col, int ftype, bool combine_step) :
        in_col(in_col), ftype(ftype), combine_step(combine_step) {}
    virtual ~BasicColSet() {}

    /**
     * Allocate my columns for update step.
     * @param number of groups found in the input table
     * @param[in,out] vector of columns of update table. This method adds
     *                columns to this vector.
     */
    virtual void alloc_update_columns(size_t num_groups, std::vector<array_info*> &out_cols) {
        bodo_array_type::arr_type_enum arr_type = in_col->arr_type;
        Bodo_CTypes::CTypeEnum dtype = in_col->dtype;
        // calling this modifies arr_type and dtype
        get_groupby_output_dtype(ftype, arr_type, dtype, false);
        out_cols.push_back(alloc_array(num_groups, 1, arr_type, dtype, 0));
        update_cols.push_back(out_cols.back());
    }

    /**
     * Perform update step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void update(const grouping_info &grp_info) {
        std::vector<array_info*> aux_cols;
        aggfunc_output_initialize(update_cols[0], ftype);
        do_apply_to_column(in_col, update_cols[0],
                           aux_cols, grp_info, ftype);
    }

    /**
     * When GroupbyPipeline shuffles the table after update, the column set
     * needs to be updated with the columns from the new shuffled table. This
     * method is called by GroupbyPipeline with an iterator pointing to my
     * first column. The column set will update its columns and return an
     * iterator pointing to the next set of columns.
     * @param iterator pointing to the first column in this column set
     */
    virtual std::vector<array_info*>::iterator update_after_shuffle(
      std::vector<array_info*>::iterator &it) {
        update_cols.assign(it, it + update_cols.size());
        return it + update_cols.size();
    }

    /**
     * Allocate my columns for combine step.
     * @param number of groups found in the input table (which is the update table)
     * @param[in,out] vector of columns of combine table. This method adds
     *                columns to this vector.
     */
    virtual void alloc_combine_columns(size_t num_groups, std::vector<array_info*> &out_cols) {
        Bodo_FTypes::FTypeEnum combine_ftype = combine_funcs[ftype];
        for (auto col : update_cols) {
            bodo_array_type::arr_type_enum arr_type = col->arr_type;
            Bodo_CTypes::CTypeEnum dtype = col->dtype;
            // calling this modifies arr_type and dtype
            get_groupby_output_dtype(combine_ftype, arr_type, dtype, false);
            out_cols.push_back(alloc_array(num_groups, 1, arr_type, dtype, 0));
            combine_cols.push_back(out_cols.back());
        }
    }

    /**
     * Perform combine step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void combine(const grouping_info &grp_info) {
        Bodo_FTypes::FTypeEnum combine_ftype = combine_funcs[ftype];
        std::vector<array_info*> aux_cols(combine_cols.begin() + 1, combine_cols.end());
        for (auto col : combine_cols)
            aggfunc_output_initialize(col, combine_ftype);
        do_apply_to_column(update_cols[0], combine_cols[0],
                           aux_cols, grp_info, combine_ftype);
    }

    /**
     * Perform eval step for this column set. This will fill the output column
     * with the final result of the aggregation operation corresponding to this
     * column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void eval(const grouping_info &grp_info) {}

    /**
     * Obtain the final output column resulting from the groupby operation on
     * this column set. This will free all other intermediate or auxiliary
     * columns (if any) used by the column set (like reduction variables).
     */
    virtual array_info* getOutputColumn() {
        std::vector<array_info*> *mycols;
        if (combine_step)
            mycols = &combine_cols;
        else
            mycols = &update_cols;
        array_info* out_col = mycols->at(0);
        for (auto it = mycols->begin() + 1; it != mycols->end(); it++) {
            array_info *a = *it;
            free_array(a);
            delete a;
        }
        return out_col;
    }

protected:
    friend class GroupbyPipeline;
    array_info *in_col; // the input column (from groupby input table) to which this column set corresponds to
    int ftype;
    bool combine_step; // GroupbyPipeline is going to perform a combine operation or not
    std::vector<array_info*> update_cols; // columns for update step
    std::vector<array_info*> combine_cols; // columns for combine step
};

class MeanColSet : public BasicColSet {
public:

    MeanColSet(array_info *in_col, bool combine_step) :
        BasicColSet(in_col, Bodo_FTypes::mean, combine_step) {}
    virtual ~MeanColSet() {}

    virtual void alloc_update_columns(size_t num_groups, std::vector<array_info*> &out_cols) {
        array_info* c1 = alloc_array(num_groups, 1, bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64, 0); // for sum and result
        array_info* c2 = alloc_array(num_groups, 1, bodo_array_type::NUMPY, Bodo_CTypes::UINT64, 0); // for counts
        out_cols.push_back(c1);
        out_cols.push_back(c2);
        update_cols.push_back(c1);
        update_cols.push_back(c2);
    }

    virtual void update(const grouping_info &grp_info) {
        std::vector<array_info*> aux_cols = {update_cols[1]};
        aggfunc_output_initialize(update_cols[0], ftype);
        aggfunc_output_initialize(update_cols[1], ftype);
        do_apply_to_column(in_col, update_cols[0],
                           aux_cols, grp_info, ftype);
    }

    virtual void combine(const grouping_info &grp_info) {
        std::vector<array_info*> aux_cols;
        aggfunc_output_initialize(combine_cols[0], Bodo_FTypes::sum);
        aggfunc_output_initialize(combine_cols[1], Bodo_FTypes::sum);
        do_apply_to_column(update_cols[0], combine_cols[0],
                           aux_cols, grp_info, Bodo_FTypes::sum);
        do_apply_to_column(update_cols[1], combine_cols[1],
                           aux_cols, grp_info, Bodo_FTypes::sum);
    }

    virtual void eval(const grouping_info &grp_info) {
        std::vector<array_info*> aux_cols;
        if (combine_step)
            do_apply_to_column(combine_cols[1], combine_cols[0], aux_cols,
                               grp_info, Bodo_FTypes::mean_eval);
        else
            do_apply_to_column(update_cols[1], update_cols[0], aux_cols,
                               grp_info, Bodo_FTypes::mean_eval);
    }
};

class VarStdColSet : public BasicColSet {
public:

    VarStdColSet(array_info *in_col, int ftype, bool combine_step) : BasicColSet(in_col, ftype, combine_step) {}
    virtual ~VarStdColSet() {}

    virtual void alloc_update_columns(size_t num_groups, std::vector<array_info*> &out_cols) {
        if (!combine_step) {
            // need to create output column now
            array_info* col = alloc_array(num_groups, 1, bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64, 0); // for result
            out_cols.push_back(col);
            update_cols.push_back(col);
        }
        array_info* count_col = alloc_array(num_groups, 1,
                                            bodo_array_type::NUMPY,
                                            Bodo_CTypes::UINT64, 0);
        array_info* mean_col = alloc_array(num_groups, 1,
                                           bodo_array_type::NUMPY,
                                           Bodo_CTypes::FLOAT64, 0);
        array_info* m2_col = alloc_array(num_groups, 1,
                                         bodo_array_type::NUMPY,
                                         Bodo_CTypes::FLOAT64, 0);
        aggfunc_output_initialize(count_col, Bodo_FTypes::count); // zero initialize
        aggfunc_output_initialize(mean_col, Bodo_FTypes::count); // zero initialize
        aggfunc_output_initialize(m2_col, Bodo_FTypes::count); // zero initialize
        out_cols.push_back(count_col);
        out_cols.push_back(mean_col);
        out_cols.push_back(m2_col);
        update_cols.push_back(count_col);
        update_cols.push_back(mean_col);
        update_cols.push_back(m2_col);
    }

    virtual void update(const grouping_info &grp_info) {
        if (!combine_step) {
            std::vector<array_info*> aux_cols = {update_cols[1],
                                                 update_cols[2],
                                                 update_cols[3]};
            do_apply_to_column(in_col, update_cols[1],
                               aux_cols, grp_info, ftype);
        } else {
            std::vector<array_info*> aux_cols = {update_cols[0],
                                                 update_cols[1],
                                                 update_cols[2]};
            do_apply_to_column(in_col, update_cols[0],
                               aux_cols, grp_info, ftype);
        }
    }

    virtual void alloc_combine_columns(size_t num_groups, std::vector<array_info*> &out_cols) {
        array_info* col = alloc_array(num_groups, 1, bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64, 0); // for result
        out_cols.push_back(col);
        combine_cols.push_back(col);
        BasicColSet::alloc_combine_columns(num_groups, out_cols);
    }

    virtual void combine(const grouping_info &grp_info) {
        array_info* count_col_in = update_cols[0];
        array_info* mean_col_in = update_cols[1];
        array_info* m2_col_in = update_cols[2];
        array_info* count_col_out = combine_cols[1];
        array_info* mean_col_out = combine_cols[2];
        array_info* m2_col_out = combine_cols[3];
        aggfunc_output_initialize(count_col_out, Bodo_FTypes::count);
        aggfunc_output_initialize(mean_col_out, Bodo_FTypes::count);
        aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count);
        var_combine(count_col_in, mean_col_in, m2_col_in, count_col_out,
                    mean_col_out, m2_col_out, grp_info.row_to_group);
    }

    virtual void eval(const grouping_info &grp_info) {
        std::vector<array_info*> *mycols;
        if (combine_step)
            mycols = &combine_cols;
        else
            mycols = &update_cols;

        std::vector<array_info*> aux_cols = {mycols->at(1),
                                             mycols->at(2),
                                             mycols->at(3)};
        if (ftype == Bodo_FTypes::var)
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols,
                               grp_info, Bodo_FTypes::var_eval);
        else
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols,
                               grp_info, Bodo_FTypes::std_eval);
    }
};

class UdfColSet : public BasicColSet {
public:

    UdfColSet(array_info *in_col, bool combine_step, table_info *udf_table, int udf_table_idx,
              int n_redvars) :
                  BasicColSet(in_col, Bodo_FTypes::udf, combine_step),
                  udf_table(udf_table),
                  udf_table_idx(udf_table_idx),
                  n_redvars(n_redvars) {}
    virtual ~UdfColSet() {}

    virtual void alloc_update_columns(size_t num_groups, std::vector<array_info*> &out_cols) {
        int offset = 0;
        if (combine_step) offset = 1;
        // for update table we only need redvars (skip first column which is
        // output column)
        for (int i=udf_table_idx + offset; i < udf_table_idx + 1 + n_redvars; i++) {
            // we get the type from the udf dummy table that was passed to C++ library
            bodo_array_type::arr_type_enum arr_type = udf_table->columns[i]->arr_type;
            Bodo_CTypes::CTypeEnum dtype = udf_table->columns[i]->dtype;
            out_cols.push_back(alloc_array(num_groups, 1, arr_type, dtype, 0));
            if (!combine_step) update_cols.push_back(out_cols.back());
        }
    }

    virtual void update(const grouping_info &grp_info) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

    virtual std::vector<array_info*>::iterator update_after_shuffle(
      std::vector<array_info*>::iterator &it) {
        // UdfColSet doesn't keep the update cols, return the updated iterator
        return it + n_redvars;
    }

    virtual void alloc_combine_columns(size_t num_groups, std::vector<array_info*> &out_cols) {
        for (int i=udf_table_idx; i < udf_table_idx + 1 + n_redvars; i++) {
            // we get the type from the udf dummy table that was passed to C++ library
            bodo_array_type::arr_type_enum arr_type = udf_table->columns[i]->arr_type;
            Bodo_CTypes::CTypeEnum dtype = udf_table->columns[i]->dtype;
            out_cols.push_back(alloc_array(num_groups, 1, arr_type, dtype, 0));
            combine_cols.push_back(out_cols.back());
        }
    }

    virtual void combine(const grouping_info &grp_info) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

    virtual void eval(const grouping_info &grp_info) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

private:
    table_info *udf_table; // the table containing type info for UDF columns
    int udf_table_idx; // index to my information in the udf table
    int n_redvars; // number of redvar columns this UDF uses
};

// TODO moving GroupbyPipeline and related code to bottom of file will avoid
// need for this forward declaration
void median_computation(array_info* arr, array_info* out_arr,
                               grouping_info const& grp_inf, bool const& skipna);

class MedianColSet : public BasicColSet {
public:

    MedianColSet(array_info *in_col, bool _skipna) :
        BasicColSet(in_col, Bodo_FTypes::median, false), skipna(_skipna) {}
    virtual ~MedianColSet() {}

    virtual void update(const grouping_info &grp_info) {
        median_computation(in_col, update_cols[0], grp_info, skipna);
    }

private:
    bool skipna;
};

// TODO moving GroupbyPipeline and related code to bottom of file will avoid
// need for this forward declaration
void nunique_computation(array_info* arr, array_info* out_arr, grouping_info const& grp_inf,
                                bool const& dropna);

class NUniqueColSet : public BasicColSet {
public:

    NUniqueColSet(array_info *in_col, bool _dropna) :
        BasicColSet(in_col, Bodo_FTypes::nunique, false), dropna(_dropna) {}
    virtual ~NUniqueColSet() {}

    virtual void update(const grouping_info &grp_info) {
        nunique_computation(in_col, update_cols[0], grp_info, dropna);
    }

private:
    bool dropna;
};

// TODO moving GroupbyPipeline and related code to bottom of file will avoid
// need for this forward declaration
void cumsum_cumprod_computation(array_info* arr, array_info* out_arr, grouping_info const& grp_inf,
                                       int32_t const& ftype, bool const& skipna);

class CumOpColSet : public BasicColSet {
public:

    CumOpColSet(array_info *in_col, int ftype, bool _skipna) :
        BasicColSet(in_col, ftype, false), skipna(_skipna) {}
    virtual ~CumOpColSet() {}

    virtual void alloc_update_columns(size_t num_groups, std::vector<array_info*> &out_cols) {
        // NOTE: output size of cum ops is the same as input size
        //       (NOT the number of groups)
        out_cols.push_back(alloc_array(in_col->length, 1, in_col->arr_type,
                                       in_col->dtype, 0));
        update_cols.push_back(out_cols.back());
    }

    virtual void update(const grouping_info &grp_info) {
        cumsum_cumprod_computation(in_col, update_cols[0], grp_info, ftype, skipna);
    }

private:
    bool skipna;
};

class GroupbyPipeline {

public:

    GroupbyPipeline(table_info* _in_table, int64_t _num_keys, bool _is_parallel,
                    int *ftypes, int *func_offsets, int *_udf_nredvars,
                    table_info* _udf_table,
                    udf_table_op_fn update_cb,
                    udf_table_op_fn combine_cb,
                    udf_eval_fn eval_cb, bool skipna) :
                    in_table(_in_table), num_keys(_num_keys), is_parallel(_is_parallel),
                    udf_table(_udf_table), udf_n_redvars(_udf_nredvars) {

        udf_info = {udf_table, update_cb, combine_cb, eval_cb};

        // NOTE cumulative operations (cumsum, cumprod, etc.) cannot be mixed
        // with non cumulative ops. This is checked at compile time in
        // aggregate.py

        for (int i=0; i < func_offsets[in_table->ncols() - num_keys]; i++) {
            int ftype = ftypes[i];
            if (ftype == Bodo_FTypes::nunique || ftype == Bodo_FTypes::median ||
                ftype == Bodo_FTypes::cumsum || ftype == Bodo_FTypes::cumprod) {
                // these operations first require shuffling the data to
                // gather all rows with the same key in the same process
                if (is_parallel)
                    shuffle_before_update = true;
                // these operations require extended group info
                req_extended_group_info = true;
                if (ftype == Bodo_FTypes::cumsum || ftype == Bodo_FTypes::cumprod)
                    cumulative_op = true;
                break;
            }
        }
        if (shuffle_before_update)
            in_table = shuffle_table(in_table, num_keys);
        // a combine operation is only necessary when data is distributed and
        // a shuffle has not been done at the start of the groupby pipeline
        do_combine = is_parallel && !shuffle_before_update;

        // construct the column sets, one for each (input_column, func) pair
        // ftypes is an array of function types received from generated code,
        // and has one ftype for each (input_column, func) pair
        int k=0;
        for (int64_t i=num_keys; i < in_table->ncols(); i++, k++) {
            array_info* col = in_table->columns[i];
            int start = func_offsets[k];
            int end = func_offsets[k + 1];
            for (int j=start; j != end; j++) {
                col_sets.push_back(makeColSet(col, ftypes[j], do_combine, skipna));
            }
        }
    }

    ~GroupbyPipeline() {
        for (auto col_set : col_sets)
            delete col_set;
    }

    /**
     * This is the main control flow of the Groupby pipeline.
     */
    table_info* run() {
        update();
        if (shuffle_before_update)
            // in_table was created in C++ during shuffling and not needed anymore
            delete_table_free_arrays(in_table);
        if (is_parallel && !shuffle_before_update) {
            shuffle();
            combine();
        }
        eval();
        return getOutputTable();
    }

private:

    /**
     * Construct and return a column set based on the ftype.
     * @param groupby input column associated with this column set.
     * @param ftype function type associated with this column set.
     * @param do_combine whether GroupbyPipeline will perform combine operation
     *        or not.
     * @param skipna option used for nunique, cumsum, cumprod
     */
    BasicColSet* makeColSet(array_info *in_col, int ftype, bool do_combine,
                            bool skipna) {
        BasicColSet* col_set;
        switch (ftype) {
            case Bodo_FTypes::udf:
                col_set = new UdfColSet(in_col, do_combine, udf_table, udf_table_idx, udf_n_redvars[n_udf]);
                udf_table_idx += (1 + udf_n_redvars[n_udf]);
                n_udf++;
                return col_set;
            case Bodo_FTypes::median:
                 return new MedianColSet(in_col, skipna);
            case Bodo_FTypes::nunique:
                 return new NUniqueColSet(in_col, skipna);
            case Bodo_FTypes::cumsum:
            case Bodo_FTypes::cumprod:
                 return new CumOpColSet(in_col, ftype, skipna);
            case Bodo_FTypes::mean:
                 return new MeanColSet(in_col, do_combine);
            case Bodo_FTypes::var:
            case Bodo_FTypes::std:
                return new VarStdColSet(in_col, ftype, do_combine);
            default:
                return new BasicColSet(in_col, ftype, do_combine);
        }
    }

    /**
     * The update step groups rows in the input table based on keys, and
     * aggregates them based on the function to be applied to the columns.
     * More specifically, it will invoke the update method of each column set.
     */
    void update() {
        in_table->num_keys = num_keys;
        if (req_extended_group_info) {
            bool consider_missing = cumulative_op;
            grp_info = get_group_info_iterate(in_table, consider_missing);
        } else
            get_group_info(*in_table, grp_info.row_to_group, grp_info.group_to_first_row, true);
        num_groups = grp_info.group_to_first_row.size();

        update_table = cur_table = new table_info();
        if (cumulative_op)
            num_keys = 0; // there are no key columns in output of cumsum, etc.
        else
            alloc_init_keys(in_table, update_table);

        for (auto col_set : col_sets) {
            col_set->alloc_update_columns(num_groups, update_table->columns);
            col_set->update(grp_info);
        }
        if (n_udf > 0)
            udf_info.update(in_table, update_table, grp_info.row_to_group.data());
    }

    /**
     * Shuffles the update table and updates the column sets with the newly
     * shuffled table.
     */
    void shuffle() {
        table_info* shuf_table = shuffle_table(update_table, num_keys);
        delete_table_free_arrays(update_table);
        update_table = cur_table = shuf_table;

        // update column sets with columns from shuffled table
        auto it = update_table->columns.begin() + num_keys;
        for (auto col_set : col_sets)
            it = col_set->update_after_shuffle(it);
    }

    /**
     * The combine step is performed after update and shuffle. It groups rows
     * in shuffled table based on keys, and aggregates them based on the
     * function to be applied to the columns. More specifically, it will invoke
     * the combine method of each column set.
     */
    void combine() {
        grp_info.row_to_group.clear();
        grp_info.group_to_first_row.clear();
        update_table->num_keys = num_keys;
        get_group_info(*update_table, grp_info.row_to_group, grp_info.group_to_first_row, false);
        num_groups = grp_info.group_to_first_row.size();

        combine_table = cur_table = new table_info();
        alloc_init_keys(update_table, combine_table);
        for (auto col_set : col_sets) {
            col_set->alloc_combine_columns(num_groups, combine_table->columns);
            col_set->combine(grp_info);
        }
        if (n_udf > 0)
            udf_info.combine(update_table, combine_table, grp_info.row_to_group.data());
        delete_table_free_arrays(update_table);
    }

    /**
     * The eval step generates the final result (output column) for each column
     * set. It call the eval method of each column set.
     */
    void eval() {
        for (auto col_set : col_sets)
            col_set->eval(grp_info);
        if (n_udf > 0)
            udf_info.eval(cur_table);
    }

    /**
     * Returns the final output table which is the result of the groupby.
     */
    table_info *getOutputTable() {
        table_info *out_table = new table_info();
        out_table->columns.assign(cur_table->columns.begin(),
                                  cur_table->columns.begin() + num_keys);
        for (BasicColSet* col_set : col_sets)
            out_table->columns.push_back(col_set->getOutputColumn());
        delete cur_table;
        return out_table;
    }

    /**
     * Allocate and fill key columns, based on grouping info. It uses the
     * values of key columns from from_table to populate out_table.
     */
    void alloc_init_keys(table_info* from_table, table_info *out_table) {
        for (int64_t i = 0; i < num_keys; i++) {
            const array_info* key_col = (*from_table)[i];
            array_info* new_key_col;
            if (key_col->arr_type == bodo_array_type::STRING) {
                // new key col will have num_groups rows containing the
                // string for each group
                int64_t n_chars = 0;   // total number of chars of all keys for
                                       // this column
                uint32_t* in_offsets = (uint32_t*)key_col->data2;
                for (size_t j = 0; j < num_groups; j++) {
                    int64_t row = grp_info.group_to_first_row[j];
                    n_chars += in_offsets[row + 1] - in_offsets[row];
                }
                new_key_col = alloc_array(num_groups, n_chars, key_col->arr_type, key_col->dtype, 0);

                uint8_t* in_null_bitmask = (uint8_t*)key_col->null_bitmask;
                uint8_t* out_null_bitmask = (uint8_t*)new_key_col->null_bitmask;
                uint32_t* out_offsets = (uint32_t*)new_key_col->data2;
                uint32_t pos = 0;
                for (size_t j = 0; j < num_groups; j++) {
                    size_t in_row = grp_info.group_to_first_row[j];
                    uint32_t start_offset = in_offsets[in_row];
                    uint32_t str_len = in_offsets[in_row + 1] - start_offset;
                    out_offsets[j] = pos;
                    memcpy(&new_key_col->data1[pos], &key_col->data1[start_offset],
                           str_len);
                    pos += str_len;
                    SetBitTo(out_null_bitmask, j, GetBit(in_null_bitmask, in_row));
                }
                out_offsets[num_groups] = pos;
            } else {
                new_key_col = alloc_array(num_groups, 1, key_col->arr_type, key_col->dtype, 0);
                int64_t dtype_size = numpy_item_size[key_col->dtype];
                for (size_t j = 0; j < num_groups; j++)
                    memcpy(new_key_col->data1 + j * dtype_size,
                           key_col->data1 + grp_info.group_to_first_row[j] * dtype_size,
                           dtype_size);
            }
            out_table->columns.push_back(new_key_col);
        }
    }

private:
    table_info *in_table; // input table of groupby
    int64_t num_keys;
    bool is_parallel;
    std::vector<BasicColSet*> col_sets;
    table_info *udf_table;
    int *udf_n_redvars;
    int n_udf = 0;
    int udf_table_idx = 0;
    // shuffling before update requires more communication and is needed
    // when one of the groupby functions is median/nunique/cumsum/cumprod
    bool shuffle_before_update = false;
    bool cumulative_op = false;
    bool req_extended_group_info = false;
    bool do_combine;

    udfinfo_t udf_info;

    table_info* update_table = nullptr;
    table_info* combine_table = nullptr;
    table_info* cur_table = nullptr;

    grouping_info grp_info;
    size_t num_groups;
};



/**
 * The isnan operation done for all types.
 *
 * @param eVal: the T value
 * @return true if for integer and true/false for floating point
 */
template <typename T>
inline typename std::enable_if<!std::is_floating_point<T>::value,bool>::type isnan_T(char* ptr) {
  return false;
}
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value,bool>::type isnan_T(char* ptr) {
  T* ptr_d = (T*)ptr;
  return isnan(*ptr_d);
}



/**
 * The cumsum_cumprod_computation function. It uses the symbolic information
 * to compute the cumsum/cumprod.
 *
 * @param The column on which we do the computation
 * @param The array containing information on how the rows are organized
 * @param skipna: Whether to skip NaN values or not.
 * @return the returning array.
 */
template<typename T>
void cumsum_cumprod_computation_T(array_info* arr, array_info* out_arr, grouping_info const& grp_inf,
                                         int32_t const& ftype, bool const& skipna) {
    size_t num_group = grp_inf.group_to_first_row.size();
    if (arr->arr_type == bodo_array_type::STRING) {
        PyErr_SetString(PyExc_RuntimeError,
                        "There is no median for the string case");
        return;
    }
    size_t siztype = numpy_item_size[arr->dtype];
    auto cum_computation=[&](std::function<std::pair<bool,T>(int64_t)> const& get_entry, std::function<void(int64_t, std::pair<bool,T> const&)> const& set_entry) -> void {
        for (size_t igrp=0; igrp<num_group; igrp++) {
            int64_t i = grp_inf.group_to_first_row[igrp];
            T initVal = 0;
            if (ftype == Bodo_FTypes::cumprod)
                initVal = 1;
            std::pair<bool,T> ePair{false, initVal};
            while (true) {
                std::pair<bool,T> fPair = get_entry(i);
                if (fPair.first) { // the value is a NaN.
                    if (skipna) {
                        set_entry(i, fPair);
                    }
                    else {
                        ePair = fPair;
                        set_entry(i, ePair);
                    }
                }
                else { // The value is a normal one.
                    if (ftype == Bodo_FTypes::cumsum)
                        ePair.second += fPair.second;
                    else
                        ePair.second *= fPair.second;
                    set_entry(i, ePair);
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
        }
        T eVal_nan = GetTentry<T>(RetrieveNaNentry(arr->dtype).data());
        std::pair<bool,T> pairNaN{true,eVal_nan};
        for (auto & idx_miss : grp_inf.list_missing)
            set_entry(idx_miss, pairNaN);
    };

    if (arr->arr_type == bodo_array_type::NUMPY) {
        cum_computation([=](int64_t pos) -> std::pair<bool,T> {
            char* ptr = arr->data1 + pos * siztype;
            bool isna = isnan_T<T>(ptr);
            T eVal = GetTentry<T>(ptr);
            return {isna, eVal};
          },
          [=](int64_t pos, std::pair<bool,T> const& ePair) -> void {
            out_arr->at<T>(pos) = ePair.second;
          });
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask_i = (uint8_t*)arr->null_bitmask;
        uint8_t* null_bitmask_o = (uint8_t*)out_arr->null_bitmask;
        cum_computation([=](int64_t pos) -> std::pair<bool,T> {
            char* ptr = arr->data1 + pos * siztype;
            return {!GetBit(null_bitmask_i,pos), GetTentry<T>(ptr)};
          },
          [=](int64_t pos, std::pair<bool,T> const& ePair) -> void {
            SetBitTo(null_bitmask_o, pos, ePair.first);
            out_arr->at<T>(pos) = ePair.second;
          });
    }
}






void cumsum_cumprod_computation(array_info* arr, array_info* out_arr, grouping_info const& grp_inf,
                                       int32_t const& ftype, bool const& skipna) {
    if (arr->dtype == Bodo_CTypes::INT8)
        return cumsum_cumprod_computation_T<int8_t>(arr, out_arr, grp_inf, ftype, skipna);
    if (arr->dtype == Bodo_CTypes::UINT8)
        return cumsum_cumprod_computation_T<uint8_t>(arr, out_arr, grp_inf, ftype, skipna);

    if (arr->dtype == Bodo_CTypes::INT16)
        return cumsum_cumprod_computation_T<int16_t>(arr, out_arr, grp_inf, ftype, skipna);
    if (arr->dtype == Bodo_CTypes::UINT16)
        return cumsum_cumprod_computation_T<uint16_t>(arr, out_arr, grp_inf, ftype, skipna);

    if (arr->dtype == Bodo_CTypes::INT32)
        return cumsum_cumprod_computation_T<int32_t>(arr, out_arr, grp_inf, ftype, skipna);
    if (arr->dtype == Bodo_CTypes::UINT32)
        return cumsum_cumprod_computation_T<uint32_t>(arr, out_arr, grp_inf, ftype, skipna);

    if (arr->dtype == Bodo_CTypes::INT64)
        return cumsum_cumprod_computation_T<int64_t>(arr, out_arr, grp_inf, ftype, skipna);
    if (arr->dtype == Bodo_CTypes::UINT64)
        return cumsum_cumprod_computation_T<uint64_t>(arr, out_arr, grp_inf, ftype, skipna);

    if (arr->dtype == Bodo_CTypes::FLOAT32)
        return cumsum_cumprod_computation_T<float>(arr, out_arr, grp_inf, ftype, skipna);
    if (arr->dtype == Bodo_CTypes::FLOAT64)
        return cumsum_cumprod_computation_T<double>(arr, out_arr, grp_inf, ftype, skipna);
}


/**
 * The median_computation function. It uses the symbolic information to compute
 * the median results.
 *
 * @param The column on which we do the computation
 * @param The array containing information on how the rows are organized
 * @param skipna: Whether to skip NaN values or not.
 */
void median_computation(array_info* arr, array_info* out_arr,
                               grouping_info const& grp_inf, bool const& skipna) {
    size_t num_group = grp_inf.group_to_first_row.size();
    std::function<bool(size_t)> isnan_entry;
    size_t siztype = numpy_item_size[arr->dtype];
    if (arr->arr_type == bodo_array_type::STRING) {
        PyErr_SetString(PyExc_RuntimeError,
                        "There is no median for the string case");
        return;
    }
    auto median_operation=[&](std::function<bool(size_t)> const& isnan_entry) -> void {
        for (size_t igrp=0; igrp<num_group; igrp++) {
            int64_t i = grp_inf.group_to_first_row[igrp];
            std::vector<double> ListValue;
            bool HasNaN = false;
            while (true) {
                if (!isnan_entry(i)) {
                    char* ptr = arr->data1 + i * siztype;
                    double eVal = GetDoubleEntry(arr->dtype, ptr);
                    ListValue.emplace_back(eVal);
                }
                else {
                    if (!skipna) {
                        HasNaN = true;
                        break;
                    }
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
            auto GetKthValue=[&](size_t const& pos) -> double {
                std::nth_element(ListValue.begin(), ListValue.begin() + pos, ListValue.end());
                return ListValue[pos];
            };
            double valReturn;
            if (HasNaN) {
                valReturn = std::nan("1");
            }
            else {
                size_t len = ListValue.size();
                int res = len % 2;
                if (res == 0) {
                    size_t kMid1 = len / 2;
                    size_t kMid2 = kMid1 - 1;
                    valReturn = (GetKthValue(kMid1) + GetKthValue(kMid2)) / 2;
                }
                else {
                    size_t kMid = len / 2;
                    valReturn = GetKthValue(kMid);
                }
            }
            out_arr->at<double>(igrp) = valReturn;
        }
    };
    if (arr->arr_type == bodo_array_type::NUMPY) {
        median_operation([=](size_t pos) -> bool {
            if (arr->dtype == Bodo_CTypes::FLOAT32) {
                return isnan(arr->at<float>(pos));
            }
            if (arr->dtype == Bodo_CTypes::FLOAT64) {
                return isnan(arr->at<double>(pos));
            }
            return false;
        });
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        median_operation([=](size_t pos) -> bool {
            return !GetBit(null_bitmask,pos);
        });
    }
}


/**
 * The nunique_computation function. It uses the symbolic information to compute
 * the nunique results.
 *
 * @param The column on which we do the computation
 * @param The array containing information on how the rows are organized
 * @param The boolean dropna indicating whether we drop or not the NaN values from the
 *   nunique computation.
 */
void nunique_computation(array_info* arr, array_info* out_arr, grouping_info const& grp_inf,
                                bool const& dropna) {
    size_t num_group = grp_inf.group_to_first_row.size();
    if (arr->arr_type == bodo_array_type::NUMPY) {
        /**
         * Check if a pointer points to a NaN or not
         *
         * @param the char* pointer
         * @param the type of the data in input
         */
        auto isnan_entry=[&](char* ptr) -> bool {
            if (arr->dtype == Bodo_CTypes::FLOAT32) {
                float* ptr_f = (float*)ptr;
                return isnan(*ptr_f);
            }
            if (arr->dtype == Bodo_CTypes::FLOAT64) {
                double* ptr_d = (double*)ptr;
                return isnan(*ptr_d);
            }
            return false;
        };
        size_t siztype = numpy_item_size[arr->dtype];
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            std::function<size_t(int64_t)> hash_fct=[&](int64_t i) -> size_t {
                char *ptr = arr->data1 + i * siztype;
                size_t retval = 0;
                memcpy(&retval, ptr, std::min(siztype, sizeof(size_t)));
                return retval;
            };
            std::function<bool(int64_t,int64_t)> equal_fct=[&](int64_t i1, int64_t i2) -> bool {
                char *ptr1 = arr->data1 + i1 * siztype;
                char *ptr2 = arr->data1 + i2 * siztype;
                return memcmp(ptr1, ptr2, siztype) == 0;
            };
            SET_CONTAINER <int64_t, std::function<size_t(int64_t)>, std::function<bool(int64_t,int64_t)>> eset({}, hash_fct, equal_fct);
            int64_t i = grp_inf.group_to_first_row[igrp];
            bool HasNullRow = false;
            while (true) {
                char* ptr = arr->data1 + (i * siztype);
                if (!isnan_entry(ptr)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }
    if (arr->arr_type == bodo_array_type::STRING) {
        uint32_t* in_offsets = (uint32_t*)arr->data2;
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        uint32_t seed = 0xb0d01280;

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            std::function<size_t(int64_t)> hash_fct=[&](int64_t i) -> size_t {
                char* val_chars = arr->data1 + in_offsets[i];
                int len = in_offsets[i + 1] - in_offsets[i];
                uint32_t val;
                hash_string_32(val_chars, len, seed, &val);
                return size_t(val);
            };
            std::function<bool(int64_t,int64_t)> equal_fct=[&](int64_t i1, int64_t i2) -> bool {
                size_t len1 = in_offsets[i1 + 1] - in_offsets[i1];
                size_t len2 = in_offsets[i2 + 1] - in_offsets[i2];
                if (len1 != len2) return false;
                char *ptr1 = arr->data1 + in_offsets[i1];
                char *ptr2 = arr->data1 + in_offsets[i2];
                return strncmp(ptr1, ptr2, len1) == 0;
            };
            SET_CONTAINER <int64_t, std::function<size_t(int64_t)>, std::function<bool(int64_t,int64_t)>> eset({}, hash_fct, equal_fct);
            int64_t i = grp_inf.group_to_first_row[igrp];
            bool HasNullRow = false;
            while (true) {
                if (GetBit(null_bitmask, i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        size_t siztype = numpy_item_size[arr->dtype];
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            std::function<size_t(int64_t)> hash_fct=[&](int64_t i) -> size_t {
                char *ptr = arr->data1 + i * siztype;
                size_t retval = 0;
                size_t *size_t_ptrA = &retval;
                char* size_t_ptrB = (char*)size_t_ptrA;
                for (size_t i=0; i<std::min(siztype, sizeof(size_t)); i++)
                    size_t_ptrB[i] = ptr[i];
                return retval;
            };
            std::function<bool(int64_t,int64_t)> equal_fct=[&](int64_t i1, int64_t i2) -> bool {
                char *ptr1 = arr->data1 + i1 * siztype;
                char *ptr2 = arr->data1 + i2 * siztype;
                return memcmp(ptr1, ptr2, siztype) == 0;
            };
            SET_CONTAINER <int64_t, std::function<size_t(int64_t)>, std::function<bool(int64_t,int64_t)>> eset({}, hash_fct, equal_fct);
            int64_t i = grp_inf.group_to_first_row[igrp];
            bool HasNullRow = false;
            while (true) {
                if (GetBit(null_bitmask, i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }
}

/**
 * Compute the boolean array on output corresponds to the "isin" function in matlab.
 * each group, writes the result to a new output table containing one row per
 * group.
 *
 * @param out_arr the boolean array on output.
 * @param in_arr the list of values on input
 * @param in_values the list of values that we need to check with
 */
void array_isin_kernel(array_info* out_arr, array_info* in_arr, array_info* in_values)
{
    CheckEqualityArrayType(in_arr, in_values);
    if (out_arr->dtype != Bodo_CTypes::_BOOL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "array out_arr should be a boolean array");
        return;
    }
    uint32_t seed = 0xb0d01d80;

    int64_t len_values = in_values->length;
    uint32_t* hashes_values = new uint32_t[len_values];
    hash_array(hashes_values, in_values, (size_t)len_values, seed);

    int64_t len_in_arr = in_arr->length;
    uint32_t* hashes_in_arr = new uint32_t[len_in_arr];
    hash_array(hashes_in_arr, in_arr, (size_t)len_in_arr, seed);

    std::function<bool(int64_t,int64_t)> equal_fct=[&](int64_t const& pos1, int64_t const& pos2) -> bool {
      int64_t pos1_b, pos2_b;
      array_info *arr1_b, *arr2_b;
      if (pos1 < len_values) {
        arr1_b = in_values;
        pos1_b = pos1;
      }
      else {
        arr1_b = in_arr;
        pos1_b = pos1 - len_values;
      }
      if (pos2 < len_values) {
        arr2_b = in_values;
        pos2_b = pos2;
      }
      else {
        arr2_b = in_arr;
        pos2_b = pos2 - len_values;
      }
      return TestEqualColumn(arr1_b, pos1_b, arr2_b, pos2_b);
    };
    std::function<size_t(int64_t)> hash_fct=[&](int64_t const& pos) -> size_t {
      int64_t value;
      if (pos < len_values)
        value = hashes_values[pos];
      else
        value = hashes_in_arr[pos - len_values];
      return (size_t)value;
    };
    SET_CONTAINER <size_t, std::function<size_t(int64_t)>, std::function<bool(int64_t,int64_t)>> eset({}, hash_fct, equal_fct);
    for (int64_t pos=0; pos<len_values; pos++) {
      eset.insert(pos);
    }
    for (int64_t pos=0; pos<len_in_arr; pos++) {
      bool test = eset.count(pos + len_values) == 1;
      out_arr->at<bool>(pos) = test;
    }
    delete [] hashes_in_arr;
    delete [] hashes_values;
}






/**
 * Compute the boolean array on output corresponds to the "isin" function in matlab.
 * each group, writes the result to a new output table containing one row per
 * group.
 *
 * @param out_arr the boolean array on output.
 * @param in_arr the list of values on input
 * @param in_values the list of values that we need to check with
 * @param is_parallel, whether the computation is parallel or not.
 */
void array_isin(array_info* out_arr, array_info* in_arr, array_info* in_values, bool is_parallel)
{
    if (!is_parallel) {
        return array_isin_kernel(out_arr, in_arr, in_values);
    }
    std::vector<array_info*> vect_in_values={in_values};
    table_info table_in_values = table_info(vect_in_values);
    std::vector<array_info*> vect_in_arr={in_arr};
    table_info table_in_arr = table_info(vect_in_arr);

    int64_t num_keys = 1;
    table_info* shuf_table_in_values = shuffle_table(&table_in_values, num_keys);
    // we need the comm_info and hashes for the reverse shuffling
    int n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    mpi_comm_info comm_info(n_pes, table_in_arr.columns);
    uint32_t seed = SEED_HASH_PARTITION;
    uint32_t* hashes = hash_keys(vect_in_arr, seed);
    comm_info.set_counts(hashes);
    table_info* shuf_table_in_arr = shuffle_table_kernel(&table_in_arr, hashes, n_pes, comm_info);
    // Creation of the output array.
    int64_t len = shuf_table_in_arr->columns[0]->length;
    array_info* shuf_out_arr = alloc_array(len, -1, out_arr->arr_type, out_arr->dtype, 0);
    // Calling isin on the shuffled info
    array_isin_kernel(shuf_out_arr, shuf_table_in_arr->columns[0], shuf_table_in_values->columns[0]);

    // Deleting the data after usage
    delete_table_free_arrays(shuf_table_in_values);
    delete_table_free_arrays(shuf_table_in_arr);
    // Now the reverse shuffling operation. Since the array out_arr is not directly handled
    // by the comm_info, we have to get out hands dirty.
    MPI_Datatype mpi_typ = get_MPI_typ(out_arr->dtype);
    size_t n_rows=out_arr->length;
    std::vector<uint8_t> tmp_recv(n_rows);
    MPI_Alltoallv(shuf_out_arr->data1,
                  comm_info.recv_count.data(), comm_info.recv_disp.data(),
                  mpi_typ, tmp_recv.data(),
                  comm_info.send_count.data(), comm_info.send_disp.data(),
                  mpi_typ, MPI_COMM_WORLD);
    fill_recv_data_inner<uint8_t>(tmp_recv.data(), (uint8_t*)out_arr->data1, hashes,
                                  comm_info.send_disp, n_pes,
                                  n_rows);
    // freeing just before returning.
    free_array(shuf_out_arr);
    delete [] hashes;
}

/**
 * This operation groups rows in a distributed table based on keys, and applies
 * a function(s) to a set of columns in each group, producing one output column
 * for each (input column, function) pair. The general algorithm works as
 * follows:
 * a) Group and Update: Each process does the following with its local table:
 *   - Determine to which group each row in the input table belongs to by using
 *     a hash table on the key columns (obtaining a row to group mapping).
 *   - Allocate output table (one row per group -most of the time- or one row
 *     per input row for cumulative operations)
 *   - Initialize output columns (depends on aggregation function)
 *   - Update: apply function to input columns, write result to output (either
 *     directly to output data column or to temporary reduction variable
 *     columns). Uses the row_to_group mapping computed above.
 * b) Shuffle: If the table is distributed, do a parallel shuffle of the
 *    current output table to gather the rows that are part of the same group
 *    on the same process.
 * c) Group and Combine: after the shuffle, a process can end up with multiple
 *    rows belonging to the same group, so we repeat the grouping of a) with
 *    the new (shuffled) table, and apply a possibly different function
 *    ("combine").
 * d) Eval: for functions that required redvar columns, this computes the
 *    final result from the value in the redvar columns and writes it to the
 *    output data columns. This step is only needed for certain functions
 *    like mean, var, std and agg. Redvar columns are deleted afterwards.
 *
 * @param input table
 * @param number of key columns in the table
 * @param functions to apply (see Bodo_FTypes::FTypeEnum)
 * @param the functions to apply to input col i are in ftypes, in range
 *        func_offsets[i] to func_offsets[i+1]
 * @param udf_nredvars[i] is the number of redvar columns needed by udf i
 * @param true if needs to run in parallel (distributed data on multiple
 *        processes)
 * @param skipdropna: whether to drop NaN values or not from the computation
 *                    (dropna for nunique and skipna for median/cumsum/cumprod)
 * @param external 'update' function (a function pointer).
 *        For ftype=udf, the update step happens in external JIT-compiled code,
 *        which must initialize redvar columns and apply the update function.
 * @param external 'combine' function (a function pointer).
 *        For ftype=udf, external code does the combine step (apply combine
 *        function to current table)
 * @param external 'eval' function (a function pointer).
 *        For ftype=udf, external code does the eval step.
 * @param dummy table containing type info for output and redvars columns for
 *        udfs
 */
table_info* groupby_and_aggregate(table_info* in_table, int64_t num_keys,
                                  int *ftypes, int *func_offsets, int *udf_nredvars,
                                  bool is_parallel, bool skipdropna,
                                  void* update_cb, void* combine_cb, void* eval_cb,
                                  table_info* udf_dummy_table) {

    GroupbyPipeline groupby(in_table, num_keys, is_parallel, ftypes, func_offsets,
                            udf_nredvars,
                            udf_dummy_table, (udf_table_op_fn)update_cb,
                            (udf_table_op_fn)combine_cb, (udf_eval_fn)eval_cb,
                            skipdropna);

    return groupby.run();
}

/**
 * Implementation of the sort_values functionality in C++
 * Notes:
 * - We depend on the timsort code from https://github.com/timsort/cpp-TimSort
 *   which provides stable sort taking care of already sorted parts.
 * - We use lambda for the call functions.
 * - The KeyComparisonAsPython is used for having the comparison as in Python.
 *
 * @param input table
 * @param number of key columns in the table used for the comparison
 * @param ascending, whether to sort ascending or not
 * @param na_position, true corresponds to last, false to first
 */
table_info* sort_values_table(table_info* in_table, int64_t n_key_t,
                              int64_t* vect_ascending, bool na_position) {
    size_t n_rows = (size_t)in_table->nrows();
    size_t n_cols = (size_t)in_table->ncols();
    size_t n_key = size_t(n_key_t);
#undef DEBUG_SORT
#ifdef DEBUG_SORT
    std::cout << "n_key_t=" << n_key_t << " na_position=" << na_position << "\n";
    for (int64_t iKey=0; iKey<n_key_t; iKey++)
      std::cerr << "iKey=" << iKey << "/" << n_key_t << "  vect_ascending=" << vect_ascending[iKey] << "\n";
    std::cout << "INPUT:\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
    std::cout << "n_rows=" << n_rows << " n_cols=" << n_cols
              << " n_key=" << n_key << "\n";
#endif
    std::vector<size_t> V(n_rows);
    for (size_t i = 0; i < n_rows; i++) V[i] = i;
    std::function<bool(size_t, size_t)> f = [&](size_t const& iRow1,
                                                size_t const& iRow2) -> bool {
        size_t shift_key1 = 0, shift_key2 = 0;
        return KeyComparisonAsPython(in_table->columns, n_key, vect_ascending,
                                     shift_key1, iRow1,
                                     shift_key2, iRow2, na_position);
    };
    gfx::timsort(V.begin(), V.end(), f);
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite(
        n_rows);
    for (size_t i = 0; i < n_rows; i++) ListPairWrite[i] = {V[i], -1};
    //
    std::vector<array_info*> out_arrs;
    // Inserting the left data
    for (size_t i_col = 0; i_col < n_cols; i_col++)
        out_arrs.emplace_back(
            RetrieveArray(in_table, ListPairWrite, i_col, -1, 0));
        //
#ifdef DEBUG_SORT
    std::cout << "OUTPUT:\n";
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
    DEBUG_PrintRefct(std::cout, out_arrs);
#endif
    return new table_info(out_arrs);
}





/** This function is the inner function for the dropping of duplicated rows.
 * This C++ code is used for the drop_duplicates.
 * Two support cases:
 * ---The local computation where we store two values (first and last) in order
 *    to deal with all eventualities
 * ---The final case where depending on the case we store the first, last or
 *    none if more than 2 are considered.
 *
 * As for the join, this relies on using hash keys for the partitionning.
 * The computation is done locally.
 *
 * External function used are "RetrieveArray" and "TestEqual"
 *
 * @param in_table : the input table
 * @param sum_value: the uint64_t containing all the values together.
 * @param keep: integer specifying the expected behavior.
 *        keep = 0 corresponds to the case of keep="first" keep first entry
 *        keep = 1 corresponds to the case of keep="last" keep last entry
 *        keep = 2 corresponds to the case of keep=False : remove all duplicates
 * @param step: integer specifying the work done
 *              2 corresponds to the first step of the operation where we collate the rows
 *                on the computational node
 *              1 corresponds to the second step of the operation after the rows have been
 *                merged on the computation
 * @return the vector of pointers to be used.
 */
table_info* drop_duplicates_table_inner(table_info* in_table, int64_t num_keys,
                                        int64_t keep, int step) {
#undef DEBUG_DD
    size_t n_col = in_table->ncols();
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<array_info*> key_arrs(num_keys);
    for (size_t iKey = 0; iKey < size_t(num_keys); iKey++)
        key_arrs[iKey] = in_table->columns[iKey];
#ifdef DEBUG_DD
    std::cout << "INPUT:\n";
    std::cout << "n_col=" << n_col << " n_rows=" << n_rows << " num_keys=" << num_keys
              << "\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
#endif

    uint32_t seed = 0xb0d01287;
    uint32_t* hashes = hash_keys(key_arrs, seed);
    /* This is a function for computing the hash (here returning computed value)
     * This is the first function passed as argument for the map function.
     *
     * Note that the hash is a size_t (as requested by standard and so 8 bytes
     * on x86-64) but our hashes are int32_t
     *
     * @param iRow is the first row index for the comparison
     * @return the hash itself
     */
    std::function<size_t(size_t)> hash_fct = [&](size_t const& iRow) -> size_t {
        return size_t(hashes[iRow]);
    };
    /* This is a function for testing equality of rows.
     * This is the second lambda passed to the map function.
     *
     * We use the TestEqual function precedingly defined.
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true/false depending on the case.
     */
    std::function<bool(size_t,size_t)> equal_fct = [&](size_t const& iRowA, size_t const& iRowB) -> bool {
        size_t shift_A = 0, shift_B = 0;
        bool test = TestEqual(key_arrs, num_keys, shift_A, iRowA, shift_B, iRowB);
        return test;
    };
    // The entSet contains the hash of the table.
    // We address the entry by the row index.
    MAP_CONTAINER <size_t, size_t, std::function<size_t(size_t)>, std::function<bool(size_t,size_t)>> entSet({}, hash_fct, equal_fct);
    // The loop over the short table.
    // entries are stored one by one and all of them are put even if identical
    // in value.
    //
    // In the first case we keep only one entry.
    auto RetrievePair1=[&]() -> std::vector<std::pair<std::ptrdiff_t,std::ptrdiff_t>> {
        std::vector<int64_t> ListRow;
        uint64_t next_ent = 0;
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            size_t& group = entSet[i_row];
            if (group == 0) {
                next_ent++;
                group = next_ent;
                ListRow.emplace_back(i_row);
            } else {
                size_t pos = group - 1;
                if (keep == 0) {  // keep first entry. So do nothing here
                }
                if (keep == 1) {  // keep last entry. So update the list
                    ListRow[pos] = i_row;
                }
                if (keep == 2) {  // Case of False. So put it to -1.
                    ListRow[pos] = -1;
                }
            }
        }
        std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
        for (auto& eRow : ListRow) {
          if (eRow != -1) ListPairWrite.push_back({eRow, -1});
        }
        return std::move(ListPairWrite);
    };
    // In this case we store the pairs of values, the first and the last.
    // This allows to reach conclusions in all possible cases.
    auto RetrievePair2=[&]() -> std::vector<std::pair<std::ptrdiff_t,std::ptrdiff_t>> {
        std::vector<std::pair<int64_t,int64_t>> ListRowPair;
        size_t next_ent = 0;
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            size_t& group = entSet[i_row];
            if (group == 0) {
                next_ent++;
                group = next_ent;
                ListRowPair.push_back({i_row,-1});
            } else {
                size_t pos = group - 1;
                ListRowPair[pos].second = i_row;
            }
        }
        std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
        for (auto& eRowPair : ListRowPair) {
          if (eRowPair.first  != -1) ListPairWrite.push_back({eRowPair.first,  -1});
          if (eRowPair.second != -1) ListPairWrite.push_back({eRowPair.second, -1});
        }
        return std::move(ListPairWrite);
    };
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
    if (step == 1 || keep == 0 || keep == 1)
      ListPairWrite = RetrievePair1();
    else
      ListPairWrite = RetrievePair2();
#ifdef DEBUG_DD
    std::cout << "|ListPairWrite|=" << ListPairWrite.size() << "\n";
#endif
    // Now building the out_arrs array.
    std::vector<array_info*> out_arrs;
    // Inserting the left data
    for (size_t i_col = 0; i_col < n_col; i_col++)
        out_arrs.emplace_back(
            RetrieveArray(in_table, ListPairWrite, i_col, -1, 0));
    //
    delete[] hashes;
#ifdef DEBUG_DD
    std::cout << "OUTPUT:\n";
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
    DEBUG_PrintRefct(std::cout, out_arrs);
#endif
    return new table_info(out_arrs);
}


/** This function is the function for the dropping of duplicated rows.
 * This C++ code should provide following functionality of pandas
 * drop_duplicates:
 * ---possibility of selecting columns for the identification
 * ---possibility of keeping first, last or removing all entries with duplicate
 * inplace operation for keeping the data in the same place is another problem.
 *
 * @param in_table : the input table
 * @param is_parallel: the boolean specifying if the computation is parallel or not.
 * @param num_keys: the number of keys used for the computation
 * @param keep: integer specifying the expected behavior.
 *        keep = 0 corresponds to the case of keep="first" keep first entry
 *        keep = 1 corresponds to the case of keep="last" keep last entry
 *        keep = 2 corresponds to the case of keep=False : remove all duplicates
 * @return the vector of pointers to be used.
 */
table_info* drop_duplicates_table(table_info* in_table, bool is_parallel,
                                  int64_t num_keys, int64_t keep) {
#ifdef DEBUG_DD
  std::cout << "is_parallel=" << is_parallel << "\n";
#endif
  // serial case
  if (!is_parallel) {
    return drop_duplicates_table_inner(in_table, num_keys, keep, 1);
  }
  // parallel case
  // pre reduction of duplicates
#ifdef DEBUG_DD
  std::cout << "Before the drop duplicates on the local nodes\n";
#endif
  table_info* red_table = drop_duplicates_table_inner(in_table, num_keys, keep, 2);
  // shuffling of values
#ifdef DEBUG_DD
  std::cout << "Before the shuffling\n";
#endif
  table_info* shuf_table = shuffle_table(red_table, num_keys);
  delete_table_free_arrays(red_table);
  // reduction after shuffling
#ifdef DEBUG_DD
  std::cout << "Before the second shuffling\n";
#endif
  table_info* ret_table = drop_duplicates_table_inner(shuf_table, num_keys, keep, 1);
  delete_table_free_arrays(shuf_table);
#ifdef DEBUG_DD
  std::cout << "Final returning table\n";
  DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
  DEBUG_PrintRefct(std::cout, ret_table->columns);
#endif
  // returning table
  return ret_table;
}



PyMODINIT_FUNC PyInit_array_tools_ext(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "array_tools_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // init numpy
    import_array();

    numpy_item_size[Bodo_CTypes::_BOOL] = sizeof(bool);
    numpy_item_size[Bodo_CTypes::INT8] = sizeof(int8_t);
    numpy_item_size[Bodo_CTypes::UINT8] = sizeof(uint8_t);
    numpy_item_size[Bodo_CTypes::INT16] = sizeof(int16_t);
    numpy_item_size[Bodo_CTypes::UINT16] = sizeof(uint16_t);
    numpy_item_size[Bodo_CTypes::INT32] = sizeof(int32_t);
    numpy_item_size[Bodo_CTypes::UINT32] = sizeof(uint32_t);
    numpy_item_size[Bodo_CTypes::INT64] = sizeof(int64_t);
    numpy_item_size[Bodo_CTypes::UINT64] = sizeof(uint64_t);
    numpy_item_size[Bodo_CTypes::FLOAT32] = sizeof(float);
    numpy_item_size[Bodo_CTypes::FLOAT64] = sizeof(double);

    PyObject *np_mod = PyImport_ImportModule("numpy");
    PyObject *dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "bool");
    if ((size_t)PyNumber_AsSsize_t(PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) != sizeof(bool)) {
        PyErr_SetString(PyExc_RuntimeError, "bool size mismatch between C++ and NumPy!");
        return NULL;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float32");
    if ((size_t)PyNumber_AsSsize_t(PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) != sizeof(float)) {
        PyErr_SetString(PyExc_RuntimeError, "float32 size mismatch between C++ and NumPy!");
        return NULL;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float64");
    if ((size_t)PyNumber_AsSsize_t(PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) != sizeof(double)) {
        PyErr_SetString(PyExc_RuntimeError, "float64 size mismatch between C++ and NumPy!");
        return NULL;
    }

    // this mapping is used by BasicColSet operations to know what combine
    // function to use for a given aggregation function
    combine_funcs[Bodo_FTypes::sum] = Bodo_FTypes::sum;
    combine_funcs[Bodo_FTypes::count] = Bodo_FTypes::sum;
    combine_funcs[Bodo_FTypes::mean] =
        Bodo_FTypes::sum;  // sum totals and counts
    combine_funcs[Bodo_FTypes::min] = Bodo_FTypes::min;
    combine_funcs[Bodo_FTypes::max] = Bodo_FTypes::max;
    combine_funcs[Bodo_FTypes::prod] = Bodo_FTypes::prod;

    // DEC_MOD_METHOD(string_array_to_info);
    PyObject_SetAttrString(m, "string_array_to_info",
                           PyLong_FromVoidPtr((void*)(&string_array_to_info)));
    PyObject_SetAttrString(m, "numpy_array_to_info",
                           PyLong_FromVoidPtr((void*)(&numpy_array_to_info)));
    PyObject_SetAttrString(
        m, "nullable_array_to_info",
        PyLong_FromVoidPtr((void*)(&nullable_array_to_info)));
    PyObject_SetAttrString(m, "info_to_string_array",
                           PyLong_FromVoidPtr((void*)(&info_to_string_array)));
    PyObject_SetAttrString(m, "info_to_numpy_array",
                           PyLong_FromVoidPtr((void*)(&info_to_numpy_array)));
    PyObject_SetAttrString(
        m, "info_to_nullable_array",
        PyLong_FromVoidPtr((void*)(&info_to_nullable_array)));
    PyObject_SetAttrString(m, "alloc_numpy",
                           PyLong_FromVoidPtr((void*)(&alloc_numpy)));
    PyObject_SetAttrString(m, "alloc_string_array",
                           PyLong_FromVoidPtr((void*)(&alloc_string_array)));
    PyObject_SetAttrString(
        m, "arr_info_list_to_table",
        PyLong_FromVoidPtr((void*)(&arr_info_list_to_table)));
    PyObject_SetAttrString(m, "info_from_table",
                           PyLong_FromVoidPtr((void*)(&info_from_table)));
    PyObject_SetAttrString(m, "delete_table",
                           PyLong_FromVoidPtr((void*)(&delete_table)));
    PyObject_SetAttrString(m, "shuffle_table",
                           PyLong_FromVoidPtr((void*)(&shuffle_table)));
    PyObject_SetAttrString(m, "hash_join_table",
                           PyLong_FromVoidPtr((void*)(&hash_join_table)));
    PyObject_SetAttrString(m, "sort_values_table",
                           PyLong_FromVoidPtr((void*)(&sort_values_table)));
    PyObject_SetAttrString(
        m, "drop_duplicates_table",
        PyLong_FromVoidPtr((void*)(&drop_duplicates_table)));
    PyObject_SetAttrString(m, "groupby_and_aggregate",
                           PyLong_FromVoidPtr((void*)(&groupby_and_aggregate)));
    PyObject_SetAttrString(m, "array_isin",
                           PyLong_FromVoidPtr((void*)(&array_isin)));
    PyObject_SetAttrString(m, "compute_node_partition_by_hash",
                           PyLong_FromVoidPtr((void*)(&compute_node_partition_by_hash)));
    return m;
}
