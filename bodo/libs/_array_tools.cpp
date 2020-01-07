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
                           std::vector<int>& send_disp, int n_pes,
                           size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int ind = tmp_offset[node];
        send_buff[ind] = data[i];
        tmp_offset[node]++;
    }
}

void fill_send_array_string_inner(char* send_data_buff,
                                  uint32_t* send_length_buff, char* arr_data,
                                  uint32_t* arr_offsets, uint32_t* hashes,
                                  std::vector<int>& send_disp,
                                  std::vector<int>& send_disp_char, int n_pes,
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
                                std::vector<int>& send_disp_null, int n_pes,
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
                     std::vector<int>& send_disp,
                     std::vector<int>& send_disp_char,
                     std::vector<int>& send_disp_null, int n_pes) {
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
    std::vector<uint8_t> tmp_null_bytes;

    explicit mpi_comm_info(int _n_pes, std::vector<array_info*>& _arrays)
        : n_pes(_n_pes), arrays(_arrays) {
        n_rows = arrays[0]->length;
        has_nulls = false;
        for (array_info* arr_info : arrays) {
            if (arr_info->arr_type == bodo_array_type::STRING ||
                arr_info->arr_type == bodo_array_type::NULLABLE_INT_BOOL)
                has_nulls = true;
        }
        // init counts
        send_count = std::vector<int>(n_pes, 0);
        recv_count = std::vector<int>(n_pes);
        send_disp = std::vector<int>(n_pes);
        recv_disp = std::vector<int>(n_pes);
        // init counts for string arrays
        for (array_info* arr_info : arrays) {
            if (arr_info->arr_type == bodo_array_type::STRING) {
                send_count_char.push_back(std::vector<int>(n_pes, 0));
                recv_count_char.push_back(std::vector<int>(n_pes));
                send_disp_char.push_back(std::vector<int>(n_pes));
                recv_disp_char.push_back(std::vector<int>(n_pes));
            } else {
                send_count_char.push_back(std::vector<int>());
                recv_count_char.push_back(std::vector<int>());
                send_disp_char.push_back(std::vector<int>());
                recv_disp_char.push_back(std::vector<int>());
            }
        }
        if (has_nulls) {
            send_count_null = std::vector<int>(n_pes);
            recv_count_null = std::vector<int>(n_pes);
            send_disp_null = std::vector<int>(n_pes);
            recv_disp_null = std::vector<int>(n_pes);
        }
    }

    void get_counts(uint32_t* hashes) {
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
            size_t n_null_bytes = std::accumulate(recv_count_null.begin(),
                                                  recv_count_null.end(), 0);
            tmp_null_bytes = std::vector<uint8_t>(n_null_bytes);
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

table_info* shuffle_table(table_info* in_table, int64_t n_keys) {
    // error checking
    if (in_table->ncols() <= 0 || n_keys <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return NULL;
    }

    // declare comm auxiliary data structures
    int n_pes, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    mpi_comm_info comm_info(n_pes, in_table->columns);

    size_t n_rows = (size_t)in_table->nrows();
    size_t n_cols = in_table->ncols();
    std::vector<array_info*> key_arrs = std::vector<array_info*>(
        in_table->columns.begin(), in_table->columns.begin() + n_keys);

    // get hashes
    uint32_t seed = 0xb0d01289;
    uint32_t* hashes = hash_keys(key_arrs, seed);

    comm_info.get_counts(hashes);

    int total_recv = std::accumulate(comm_info.recv_count.begin(),
                                     comm_info.recv_count.end(), 0);
    std::vector<int> n_char_recvs(n_cols);
    for (size_t i = 0; i < n_cols; i++)
        n_char_recvs[i] =
            std::accumulate(comm_info.recv_count_char[i].begin(),
                            comm_info.recv_count_char[i].end(), 0);

    // printf("%d total count %d\n", rank, total_recv);

    // fill send buffer and send
    std::vector<array_info*> out_arrs;
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

        shuffle_array(send_arr, out_arr, comm_info.send_count,
                      comm_info.recv_count, comm_info.send_disp,
                      comm_info.recv_disp, comm_info.send_count_char[i],
                      comm_info.recv_count_char[i], comm_info.send_disp_char[i],
                      comm_info.recv_disp_char[i], comm_info.send_count_null,
                      comm_info.recv_count_null, comm_info.send_disp_null,
                      comm_info.recv_disp_null, comm_info.tmp_null_bytes);

        out_arrs.push_back(out_arr);
        free_array(send_arr);
        delete send_arr;
    }

    // clean up
    delete[] hashes;

    return new table_info(out_arrs);
}

/** Getting the expression of a T value as a vector of characters
 *
 * The template paramter is T.
 * @param val the value in the type T.
 * @return the vector of characters on output
 */
template <typename T>
std::vector<char> GetVector(T const& val) {
    const T* valptr = &val;
    const char* charptr = (char*)valptr;
    std::vector<char> V(sizeof(T));
    for (size_t u = 0; u < sizeof(T); u++) V[u] = charptr[u];
    return V;
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
    if (dtype == Bodo_CTypes::_BOOL) return GetVector<bool>(false);
    if (dtype == Bodo_CTypes::INT8) return GetVector<int8_t>(-1);
    if (dtype == Bodo_CTypes::UINT8) return GetVector<uint8_t>(0);
    if (dtype == Bodo_CTypes::INT16) return GetVector<int16_t>(-1);
    if (dtype == Bodo_CTypes::UINT16) return GetVector<uint16_t>(0);
    if (dtype == Bodo_CTypes::INT32) return GetVector<int32_t>(-1);
    if (dtype == Bodo_CTypes::UINT32) return GetVector<uint32_t>(0);
    if (dtype == Bodo_CTypes::INT64) return GetVector<int64_t>(-1);
    if (dtype == Bodo_CTypes::UINT64) return GetVector<uint64_t>(0);
    if (dtype == Bodo_CTypes::FLOAT32) return GetVector<float>(std::nanf("1"));
    if (dtype == Bodo_CTypes::FLOAT64) return GetVector<double>(std::nan("1"));
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
            LStr.push_back("");
        ListListStr.push_back(LStr);
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
 * type as the right keys. The shift is used to precise whether we use the left
 * keys or the right keys. Equality means that all the columns are the same.
 * Thus the test iterates over the columns and if one is different then result
 * is false. We consider all types of bodo_array_type
 *
 * This function is currently unused but could be used in the future.
 *
 * @param in_table the input table
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
        if (columns[shift_key1 + iKey]->arr_type == bodo_array_type::NUMPY) {
            // In the case of NUMPY, we compare the values for concluding.
            uint64_t siztype = numpy_item_size[columns[shift_key1 + iKey]->dtype];
            for (uint64_t u = 0; u < siztype; u++) {
                if (columns[shift_key1 + iKey]->data1[siztype * iRow1 + u] !=
                    columns[shift_key2 + iKey]->data1[siztype * iRow2 + u])
                    return false;
            }
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
            if (bit1 != bit2) return false;
            // If both bitmasks are false, then it does not matter what value
            // they are storing. Comparison is the same as for NUMPY.
            if (bit1) {
                uint64_t siztype =
                    numpy_item_size[columns[shift_key1 + iKey]->dtype];
                for (uint64_t u = 0; u < siztype; u++) {
                    if (columns[shift_key1 + iKey]
                            ->data1[siztype * iRow1 + u] !=
                        columns[shift_key2 + iKey]->data1[siztype * iRow2 + u])
                        return false;
                }
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
            // If bitmasks are different then we conclude they are not equal.
            if (bit1 != bit2) return false;
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
                // If string lengths are different then they are different.
                if (len1 != len2) return false;
                // Now we iterate over the characters for the comparison.
                uint32_t pos1_prev = data2_1[iRow1];
                uint32_t pos2_prev = data2_2[iRow2];
                char* data1_1 = (char*)columns[shift_key1 + iKey]->data1;
                char* data1_2 = (char*)columns[shift_key2 + iKey]->data1;
                for (uint32_t pos = 0; pos < len1; pos++) {
                    uint32_t pos1 = pos1_prev + pos;
                    uint32_t pos2 = pos2_prev + pos;
                    if (data1_1[pos1] != data1_2[pos2]) return false;
                }
            }
        }
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
 * @return 1 if *ptr1 < *ptr2
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
        group.push_back(i_short);
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
                out_arrs.push_back(RetrieveArray(in_table, ListPairWrite, i,
                                                 n_tot_left + i, 2));
            } else {
                out_arrs.push_back(RetrieveArray(in_table, ListPairWrite,
                                                 n_tot_left + i, i, 2));
            }
        } else {
            if (ChoiceOpt == 0) {
                out_arrs.push_back(
                    RetrieveArray(in_table, ListPairWrite, i, -1, 0));
            } else {
                out_arrs.push_back(
                    RetrieveArray(in_table, ListPairWrite, -1, i, 1));
            }
        }
    }
    // Inserting the right data
    for (size_t i = 0; i < n_tot_right; i++) {
        if (i < n_key && vect_same_key[i < n_key ? i : 0] == 1) {
            if (ChoiceOpt == 0) {
                out_arrs.push_back(RetrieveArray(in_table, ListPairWrite, i,
                                                 n_tot_left + i, 2));
            } else {
                out_arrs.push_back(RetrieveArray(in_table, ListPairWrite,
                                                 n_tot_left + i, i, 2));
            }
        } else {
            if (ChoiceOpt == 0) {
                out_arrs.push_back(RetrieveArray(in_table, ListPairWrite, -1,
                                                 n_tot_left + i, 1));
            } else {
                out_arrs.push_back(RetrieveArray(in_table, ListPairWrite,
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
        mean,
        min,
        max,
        prod,
        var,
        std,
        agg,
        num_funcs,  // num_funcs is used to know how many functions up to this
                    // point
        mean_eval,
        var_combine,
        var_eval,
        std_eval
    };
};

/**
 * Function pointer for groupby update and combine operations that are
 * executed in JIT-compiled code (also see agginfo_t).
 *
 * @param input table
 * @param output table
 * @param row to group mapping (tells to which group -row in output table-
          the row i of input table goes to)
 */
typedef void (*agg_table_op_fn)(table_info* in_table, table_info* out_table,
                                int64_t* row_to_group);
/**
 * Function pointer for groupby eval operation that is executed in JIT-compiled
 * code (also see agginfo_t).
 *
 * @param table containing the output columns and reduction variables columns
 */
typedef void (*agg_eval_fn)(table_info*);

/*
 * This struct stores info that is currently used for groupby.agg() where
 * code such as user-defined functions is JIT-compiled and used for
 * aggregation. Such JIT-compiled code will be invoked by the C++ library
 * via function pointers.
 */
struct agginfo_t {
    /*
     * This empty table is used to tell the C++ library the types to use
     * to allocate the output table (also has the type for redvar columns)
     */
    table_info* out_table_dummy;
    /*
     * Function pointer to "update" code which performs the initial
     * local groupby and aggregation.
     */
    agg_table_op_fn update;
    /*
     * Function pointer to "combine" code which combines the results
     * after shuffle.
     */
    agg_table_op_fn combine;
    /*
     * Function pointer to "eval" code which performs post-processing and
     * sets the final output value for each group.
     */
    agg_eval_fn eval;
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

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::prod,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) v1 *= v2;
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
    result = sqrt(m2 / (count - 1));
}









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
                     const std::vector<int64_t>& row_to_group) {
    switch (in_col->arr_type) {
        case bodo_array_type::NUMPY:
            if (ftype == Bodo_FTypes::mean) {
                array_info* count_col = aux_cols[0];
                for (int64_t i = 0; i < in_col->length; i++)
                    if (row_to_group[i] != -1)
                        mean_agg<T>::apply(
                            out_col->at<double>(row_to_group[i]),
                            in_col->at<T>(i),
                            count_col->at<uint64_t>(row_to_group[i]));
            } else if (ftype == Bodo_FTypes::mean_eval) {
                for (int64_t i = 0; i < in_col->length; i++)
                    mean_eval(out_col->at<double>(i), in_col->at<uint64_t>(i));
            } else if (ftype == Bodo_FTypes::var) {
                array_info* count_col = aux_cols[0];
                array_info* mean_col = aux_cols[1];
                array_info* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++)
                    if (row_to_group[i] != -1)
                        var_agg<T>::apply(
                            in_col->at<T>(i),
                            count_col->at<uint64_t>(row_to_group[i]),
                            mean_col->at<double>(row_to_group[i]),
                            m2_col->at<double>(row_to_group[i]));
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
                    if (row_to_group[i] != -1)
                        count_agg<T>::apply(
                            out_col->at<int64_t>(row_to_group[i]),
                            in_col->at<T>(i));
            } else {
                for (int64_t i = 0; i < in_col->length; i++)
                    if (row_to_group[i] != -1)
                        aggfunc<T, ftype>::apply(
                            out_col->at<T>(row_to_group[i]), in_col->at<T>(i));
            }
            return;
        // for strings, we are only supporting count for now, and count function
        // works for strings because the input value doesn't matter
        case bodo_array_type::STRING:
            assert(ftype == Bodo_FTypes::count);
        case bodo_array_type::NULLABLE_INT_BOOL:
            switch (ftype) {
                case Bodo_FTypes::count:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            count_agg<T>::apply(
                                out_col->at<int64_t>(row_to_group[i]),
                                in_col->at<T>(i));
                    }
                    return;
                case Bodo_FTypes::mean:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i)) {
                            mean_agg<T>::apply(
                                out_col->at<double>(row_to_group[i]),
                                in_col->at<T>(i),
                                aux_cols[0]->at<uint64_t>(row_to_group[i]));
                        }
                    }
                    return;
                case Bodo_FTypes::var:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            var_agg<T>::apply(
                                in_col->at<T>(i),
                                aux_cols[0]->at<uint64_t>(row_to_group[i]),
                                aux_cols[1]->at<double>(row_to_group[i]),
                                aux_cols[2]->at<double>(row_to_group[i]));
                    }
                    return;
                default:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i)) {
                            aggfunc<T, ftype>::apply(
                                out_col->at<T>(row_to_group[i]),
                                in_col->at<T>(i));
                            SetBitTo((uint8_t*)out_col->null_bitmask,
                                     row_to_group[i], true);
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
                        const std::vector<int64_t>& row_to_group, int ftype) {
    if (ftype == Bodo_FTypes::count) {
        switch (in_col->dtype) {
            case Bodo_CTypes::FLOAT32:
                // data will only be used to check for nans
                return apply_to_column<float, Bodo_FTypes::count>(
                    in_col, out_col, aux_cols, row_to_group);
            case Bodo_CTypes::FLOAT64:
                // data will only be used to check for nans
                return apply_to_column<double, Bodo_FTypes::count>(
                    in_col, out_col, aux_cols, row_to_group);
            default:
                // data will be ignored in this case, so type doesn't matter
                return apply_to_column<int8_t, Bodo_FTypes::count>(
                    in_col, out_col, aux_cols, row_to_group);
        }
    }
    switch (in_col->dtype) {
        case Bodo_CTypes::_BOOL:
            switch (ftype) {
                case Bodo_FTypes::min:
                    return apply_to_column<bool, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<bool, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<bool, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                default:
                    PyErr_SetString(PyExc_RuntimeError,
                        "unsuported aggregation for boolean type column");
                    return;
            }
        case Bodo_CTypes::INT8:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int8_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<int8_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<int8_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<int8_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<int8_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int8_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
            }
        case Bodo_CTypes::UINT8:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint8_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<uint8_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<uint8_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint8_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint8_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint8_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
            }
        case Bodo_CTypes::INT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int16_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<int16_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<int16_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<int16_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<int16_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int16_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
            }
        case Bodo_CTypes::UINT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint16_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<uint16_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<uint16_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint16_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint16_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint16_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
            }
        case Bodo_CTypes::INT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int32_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<int32_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<int32_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<int32_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<int32_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int32_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
            }
        case Bodo_CTypes::UINT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint32_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<uint32_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<uint32_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint32_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint32_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint32_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
            }
        case Bodo_CTypes::INT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
            }
        case Bodo_CTypes::UINT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint64_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<uint64_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<uint64_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint64_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint64_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint64_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
            }
        case Bodo_CTypes::FLOAT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<float, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<float, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<float, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<float, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<float, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<float, Bodo_FTypes::mean_eval>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<float, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<float, Bodo_FTypes::var_eval>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<float, Bodo_FTypes::std_eval>(
                        in_col, out_col, aux_cols, row_to_group);
            }
        case Bodo_CTypes::FLOAT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<double, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::min:
                    return apply_to_column<double, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::max:
                    return apply_to_column<double, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::prod:
                    return apply_to_column<double, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean:
                    return apply_to_column<double, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<double, Bodo_FTypes::mean_eval>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<double, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<double, Bodo_FTypes::var_eval>(
                        in_col, out_col, aux_cols, row_to_group);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<double, Bodo_FTypes::std_eval>(
                        in_col, out_col, aux_cols, row_to_group);
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
                row_to_group.push_back(-1);
                continue;
            }
        }
        multi_col_key key(hashes[i], &table, i);
        int64_t& group = key_to_group[key];  // this inserts 0 into the map if
                                             // key doesn't exist
        if (group == 0) {
            group = next_group++;  // this updates the value in the map without
                                   // another lookup
            group_to_first_row.push_back(i);
        }
        row_to_group.push_back(group - 1);
    }
    delete[] hashes;
}


struct grouping_info {
    std::vector<int64_t> row_to_group;
    std::vector<int64_t> group_to_first_row;
    std::vector<int64_t> next_row_in_group;
};

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
grouping_info get_group_info_iterate(table_info* table) {
    std::vector<int64_t> row_to_group;
    std::vector<int64_t> group_to_first_row;
    std::vector<int64_t> next_row_in_group(table->nrows(), -1);
    std::vector<int64_t> active_group_repr;

    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table->columns.begin(), table->columns.begin() + table->num_keys);
    uint32_t seed = 0xb0d01288;
    uint32_t* hashes = hash_keys(key_cols, seed);

    bool key_is_nullable = does_keys_have_nulls(key_cols);
    row_to_group.reserve(table->nrows());
    // start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values I record in the output go from
    // 0 to num_groups - 1)
    int next_group = 1;
    MAP_CONTAINER <multi_col_key, int64_t, key_hash> key_to_group;
    for (int64_t i = 0; i < table->nrows(); i++) {
        if (key_is_nullable) {
            if (does_row_has_nulls(key_cols, i)) {
                row_to_group.push_back(-1);
                continue;
            }
        }
        multi_col_key key(hashes[i], table, i);
        int64_t& group = key_to_group[key];  // this inserts 0 into the map if
                                             // key doesn't exist
        if (group == 0) {
            group = next_group++;  // this updates the value in the map without
                                   // another lookup
            group_to_first_row.push_back(i);
            active_group_repr.push_back(i);
        } else {
            int64_t prev_elt = active_group_repr[group - 1];
            next_row_in_group[prev_elt] = i;
            active_group_repr[group - 1] = i;
        }
        row_to_group.push_back(group - 1);
    }
    delete[] hashes;
    return {std::move(row_to_group), std::move(group_to_first_row), std::move(next_row_in_group)};
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
 * Create temporary columns needed by mean, var, std and agg (to store
 * reduction variables).
 *
 * @param[in,out] table output columns. The created columns are added to this
 * vector
 * @param number of keys in table
 * @param number of columns in table before adding auxiliar columns
 * @param number of groups, which equals the number of rows that we need to
 * create
 * @param function identifier
 * @param[in,out] stores the created auxiliary columns
 * @param info structure for agg operation
 */
void create_auxiliary_cols(std::vector<array_info*>& out_cols, int64_t num_keys,
                           int64_t ncols, int64_t num_groups, int ftype,
                           std::vector<std::vector<array_info*>>& aux_cols,
                           const agginfo_t& agginfo) {
    switch (ftype) {
        case Bodo_FTypes::mean:
            for (int64_t i = num_keys; i < ncols; i++) {
                array_info* aux_col =
                    alloc_array(num_groups, 1, bodo_array_type::NUMPY,
                                Bodo_CTypes::UINT64, 0);
                out_cols.push_back(aux_col);
                aux_cols.emplace_back();
                aux_cols.back().push_back(aux_col);
                // auxiliary column for mean will record the count
                aggfunc_output_initialize(aux_col, Bodo_FTypes::count);
            }
            return;
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
            for (int64_t i = num_keys; i < ncols; i++) {
                array_info* count_col =
                    alloc_array(num_groups, 1, bodo_array_type::NUMPY,
                                Bodo_CTypes::UINT64, 0);
                array_info* mean_col =
                    alloc_array(num_groups, 1, bodo_array_type::NUMPY,
                                Bodo_CTypes::FLOAT64, 0);
                array_info* m2_col =
                    alloc_array(num_groups, 1, bodo_array_type::NUMPY,
                                Bodo_CTypes::FLOAT64, 0);
                out_cols.push_back(count_col);
                out_cols.push_back(mean_col);
                out_cols.push_back(m2_col);
                aux_cols.emplace_back();
                aux_cols.back().push_back(count_col);
                aux_cols.back().push_back(mean_col);
                aux_cols.back().push_back(m2_col);
                // auxiliary column for mean will record the count
                aggfunc_output_initialize(count_col, Bodo_FTypes::count);
                aggfunc_output_initialize(mean_col, Bodo_FTypes::count);
                aggfunc_output_initialize(m2_col, Bodo_FTypes::count);
            }
            return;
        case Bodo_FTypes::agg:
            for (int64_t i=ncols; i < agginfo.out_table_dummy->ncols(); i++) {
                bodo_array_type::arr_type_enum arr_type = agginfo.out_table_dummy->columns[i]->arr_type;
                Bodo_CTypes::CTypeEnum dtype = agginfo.out_table_dummy->columns[i]->dtype;
                out_cols.push_back(alloc_array(num_groups, 1, arr_type, dtype, 0));
            }
            return;
        default:
            return;
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











/**
 * Allocate output tables (intermediary or final) for groupby.
 *
 * @param input table
 * @param number of keys in input table
 * @param total number of output columns to allocate
 * @param number of groups (number of rows in output table)
 * @param groupby aggregation function type
 * @param info structure for agg operation
 * @param vector that maps group number to the first row in input table
 *        that belongs to that group
 * @return the newly allocated output table and columns
 */
table_info* groupby_alloc_table(const table_info& in_table, int64_t n_keys,
                                int64_t n_out_cols,
                                int64_t num_groups, int ftype,
                                const agginfo_t& agginfo,
                                const std::vector<int64_t>& group_to_first_row) {
    std::vector<array_info*> out_cols(n_out_cols);
    for (int64_t i = 0; i < n_out_cols; i++) {
        if (ftype == Bodo_FTypes::agg && i >= n_keys) {
            // for output data columns in 'agg' case, we need to get the type
            // from the dummy table that was passed to C++ library
            bodo_array_type::arr_type_enum arr_type = agginfo.out_table_dummy->columns[i]->arr_type;
            Bodo_CTypes::CTypeEnum dtype = agginfo.out_table_dummy->columns[i]->dtype;
            out_cols[i] = alloc_array(num_groups, 1, arr_type, dtype, 0);
        } else {
            const array_info* in_arr = in_table[i];
            bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
            Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
            // calling this modifies arr_type and dtype
            get_groupby_output_dtype(ftype, arr_type, dtype, i < n_keys);
            if (arr_type == bodo_array_type::STRING) {
                // this is a special case: key columns containing strings.
                // in this case, output col will have num_groups rows
                // containing the string for each group
                assert(i < n_keys);  // assert that this is a key column
                int64_t n_chars = 0;   // total number of chars of all keys for this
                                       // column
                uint32_t* in_offsets = (uint32_t*)in_arr->data2;
                for (int64_t j = 0; j < num_groups; j++) {
                    int64_t row = group_to_first_row[j];
                    n_chars += in_offsets[row + 1] - in_offsets[row];
                }
                out_cols[i] = alloc_array(num_groups, n_chars, arr_type, dtype, 0);
            } else {
                out_cols[i] = alloc_array(num_groups, 1, arr_type, dtype, 0);
            }
        }
    }

    return new table_info(out_cols);
}

/**
 * Initialize key columns in groupby output tables (intermediary or final
 * tables).
 *
 * @param input table
 * @param[in,out] output table
 * @param number of keys in tables
 * @param number of groups
 * @param vector that maps group number to the first row in input table
 *        that belongs to that group
 */
void groupby_init_keys(const table_info& in_table, table_info& out_table,
                       int64_t n_keys, int64_t num_groups,
                       const std::vector<int64_t>& group_to_first_row) {
    for (int64_t j = 0; j < n_keys; j++) {
        const array_info* in_col = in_table[j];
        array_info* out_col = out_table[j];
        if (in_col->arr_type == bodo_array_type::NUMPY ||
            in_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            int64_t dtype_size = numpy_item_size[out_col->dtype];
            for (int64_t i = 0; i < num_groups; i++)
                memcpy(out_col->data1 + i * dtype_size,
                       in_col->data1 + group_to_first_row[i] * dtype_size,
                       dtype_size);
        } else if (in_col->arr_type == bodo_array_type::STRING) {
            uint8_t* in_null_bitmask = (uint8_t*)in_col->null_bitmask;
            uint32_t* in_offsets = (uint32_t*)in_col->data2;
            uint8_t* out_null_bitmask = (uint8_t*)out_col->null_bitmask;
            uint32_t* out_offsets = (uint32_t*)out_col->data2;
            uint32_t pos = 0;
            for (int64_t i = 0; i < num_groups; i++) {
                size_t in_row = group_to_first_row[i];
                uint32_t start_offset = in_offsets[in_row];
                uint32_t str_len = in_offsets[in_row + 1] - start_offset;
                out_offsets[i] = pos;
                memcpy(&out_col->data1[pos], &in_col->data1[start_offset],
                       str_len);
                pos += str_len;
                SetBitTo(out_null_bitmask, i, GetBit(in_null_bitmask, in_row));
            }
            out_offsets[num_groups] = pos;
        }
    }
}

/**
 * Groups the rows in a table based on key, applies a function to the rows in
 * each group, writes the result to a new output table containing one row per
 * group.
 *
 * @param input table (contains keys and input data columns)
 * @param number of key columns in the table
 * @param number of data columns
 * @param function to apply
 * @param number of functions to apply per data column (used in agg)
 * @param info structure for agg operation
 * @return output table which contains key columns, output data columns, and
 * optionally redvar columns (depends on function to apply). Note that
 * currently, which columns are populated and how depends on the function
 */
table_info* groupby_update(table_info& in_table, int64_t num_keys,
                           int64_t num_data_cols,
                           int ftype,
                           int32_t num_funcs,
                           const agginfo_t& agginfo) {

    int64_t n_out_cols = num_keys + num_data_cols * num_funcs;
    std::vector<int64_t> inrows_to_group;
    std::vector<int64_t> group_to_first_row;
    in_table.num_keys = num_keys;
    get_group_info(in_table, inrows_to_group, group_to_first_row, true);
    int64_t num_groups = group_to_first_row.size();

    // create output table with *uninitialized* columns
    // output table has one row per group
    table_info* out_table = groupby_alloc_table(in_table, num_keys, n_out_cols,
                                                num_groups, ftype, agginfo,
                                                group_to_first_row);

    // some aggregation functions like 'mean' need auxiliary columns to store
    // reduction variables such as the count of valid elements
    std::vector<std::vector<array_info*>> aux_cols;
    create_auxiliary_cols(out_table->columns, num_keys, n_out_cols, num_groups,
                          ftype, aux_cols, agginfo);

    // set key values in output table
    groupby_init_keys(in_table, *out_table, num_keys, num_groups,
                      group_to_first_row);

    if (ftype == Bodo_FTypes::agg) { // user defined function
        agginfo.update(&in_table, out_table, inrows_to_group.data());
    } else {
        for (int64_t j = num_keys; j < n_out_cols; j++) {
            aggfunc_output_initialize((*out_table)[j], ftype);
            do_apply_to_column(in_table[j], (*out_table)[j],
                               aux_cols[j - num_keys], inrows_to_group, ftype);
        }
    }

    return out_table;
}

/**
 * Combine operation. After shuffle, groups the rows in a table based on key,
 * applies a combine function to the rows in each group, writes the result to a
 * new output table containing one row per group.
 *
 * @param input table after shuffle (contains keys, output data columns and
 * optionally redvar data columns resulting from 'update' operation)
 * @param number of key columns in the table
 * @param number of data columns
 * @param function to apply
 * @param info structure for agg operation
 * @return output table which contains key columns, output data columns, and
 * optionally redvar columns (if they were added in 'update'). Note that
 * currently, which columns are populated and how depends on the function
 */
table_info* groupby_combine(table_info& in_table, int64_t num_keys,
                            int64_t num_data_cols, int ftype,
                            const agginfo_t& agginfo) {

    int64_t n_out_cols = in_table.ncols();
    std::vector<int64_t> inrows_to_group;
    std::vector<int64_t> group_to_first_row;
    in_table.num_keys = num_keys;
    get_group_info(in_table, inrows_to_group, group_to_first_row, false);
    int64_t num_groups = group_to_first_row.size();

    // create output table with *uninitialized* columns
    // output table has one row per group
    table_info* out_table = groupby_alloc_table(in_table, num_keys, n_out_cols,
                                                num_groups, ftype, agginfo,
                                                group_to_first_row);
    std::vector<std::vector<array_info*>> aux_cols;

    // set key values in output table
    groupby_init_keys(in_table, *out_table, num_keys, num_groups,
                      group_to_first_row);

    if (ftype == Bodo_FTypes::agg) { // user defined function
        agginfo.combine(&in_table, out_table, inrows_to_group.data());
    } else if (ftype != Bodo_FTypes::var_combine) {
        for (int64_t j = num_keys; j < n_out_cols; j++) {
            aggfunc_output_initialize((*out_table)[j], ftype);
            do_apply_to_column(in_table[j], (*out_table)[j],
                               aux_cols[j - num_keys], inrows_to_group, ftype);
        }
    } else {
        for (int64_t i = 0; i < num_data_cols; i++) {
            // get count, mean, m2 cols for this data col
            int idx = num_keys + num_data_cols + i * 3;
            array_info* count_col_in = in_table[idx];
            array_info* mean_col_in = in_table[idx + 1];
            array_info* m2_col_in = in_table[idx + 2];
            array_info* count_col_out = (*out_table)[idx];
            array_info* mean_col_out = (*out_table)[idx + 1];
            array_info* m2_col_out = (*out_table)[idx + 2];
            aggfunc_output_initialize(count_col_out, Bodo_FTypes::count);
            aggfunc_output_initialize(mean_col_out, Bodo_FTypes::count);
            aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count);
            var_combine(count_col_in, mean_col_in, m2_col_in, count_col_out,
                        mean_col_out, m2_col_out, inrows_to_group);
        }
    }
    return out_table;
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
array_info* nunique_computation(array_info* arr, grouping_info const& grp_inf,
                                bool const& dropna) {
    size_t num_group = grp_inf.group_to_first_row.size();
    array_info* out_arr = alloc_array(num_group, 1, bodo_array_type::NUMPY,
                                      Bodo_CTypes::INT64, 0);
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
    return out_arr;
}






/**
 * Groups the rows in a table based on key, applies the nunique operation to the rows in
 * each group, writes the result to a new output table containing one row per
 * group.
 *
 * @param input table
 * @param number of key columns in the table
 * @param number of data columns
 * @param whether to drop NaN values or not from the computation
 */
table_info* groupby_and_nunique(table_info* in_table, int64_t num_keys,
                                int64_t num_data_cols, bool const& dropna) {
#undef DEBUG_NUNIQUE
#ifdef DEBUG_NUNIQUE
    std::cout << "IN_TABLE:\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
#endif
    const int64_t ncols = in_table->ncols();
    in_table->num_keys = num_keys;
    grouping_info grp_inf = get_group_info_iterate(in_table);
    size_t num_groups = grp_inf.group_to_first_row.size();
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite(
        num_groups);
    for (size_t igrp = 0; igrp < num_groups; igrp++) {
        ListPairWrite[igrp] = {grp_inf.group_to_first_row[igrp], -1};
    }

    std::vector<array_info*> out_arrs;
    for (int64_t i_col = 0; i_col < num_keys; i_col++) {
        out_arrs.push_back(
            RetrieveArray(in_table, ListPairWrite, i_col, -1, 0));
    }
    for (int64_t i_col = num_keys; i_col < ncols; i_col++) {
        out_arrs.push_back(
            nunique_computation(in_table->columns[i_col], grp_inf, dropna));
    }
#ifdef DEBUG_NUNIQUE
    std::cout << "OUT_TABLE:\n";
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
    DEBUG_PrintRefct(std::cout, out_arrs);
#endif
    return new table_info(out_arrs);
}








/**
 * Applies an evaluation function to each row in input table, writing the
 * final result to output data columns. Removes redvar columns when done.
 *
 * @param input table (contains keys, output data columns and optionally
 *        redvar columns if added during 'update')
 * @param number of key columns in the table
 * @param number of data columns
 * @param function to apply (see Bodo_FTypes::FTypeEnum)
 * @param number of functions to apply per data column (used in agg)
 * @param info structure for agg operation
 */
void groupby_eval(table_info& in_table, int64_t num_keys, int64_t n_data_cols,
                  int32_t ftype, int64_t num_funcs, const agginfo_t& agginfo) {
    std::vector<array_info*> aux_cols;
    std::vector<int64_t> row_to_group;
    bool remove_aux_cols = false;
    switch (ftype) {
        case Bodo_FTypes::mean:
            for (int64_t i = num_keys; i < num_keys + n_data_cols; i++) {
                do_apply_to_column(in_table[i + n_data_cols], in_table[i],
                                   aux_cols, row_to_group,
                                   Bodo_FTypes::mean_eval);
            }
            remove_aux_cols = true;
            break;
        case Bodo_FTypes::std:
        case Bodo_FTypes::var:
            for (int64_t j = num_keys, z = 0; j < num_keys + n_data_cols;
                 j++, z++) {
                int64_t idx = num_keys + n_data_cols + z * 3;
                aux_cols.push_back(in_table[idx]);
                aux_cols.push_back(in_table[idx + 1]);
                aux_cols.push_back(in_table[idx + 2]);
                if (ftype == Bodo_FTypes::var)
                    do_apply_to_column(in_table[0], in_table[j], aux_cols,
                                       row_to_group, Bodo_FTypes::var_eval);
                else
                    do_apply_to_column(in_table[0], in_table[j], aux_cols,
                                       row_to_group, Bodo_FTypes::std_eval);
                aux_cols.clear();
            }
            remove_aux_cols = true;
            break;
        case Bodo_FTypes::agg:
            agginfo.eval(&in_table);
            remove_aux_cols = true;
            break;
        default:
            return;
    }
    if (remove_aux_cols) {
        for (int64_t i = num_keys + n_data_cols*num_funcs; i < in_table.ncols(); i++)
            free_array(in_table[i]);
        in_table.columns.resize(num_keys + n_data_cols*num_funcs);
    }
}

std::vector<Bodo_FTypes::FTypeEnum> combine_funcs(Bodo_FTypes::num_funcs);


table_info* groupby_and_aggregate_nunique(table_info* in_table, int64_t num_keys,
                                          bool is_parallel) {
    // perform initial local aggregation
    int64_t num_data_cols = in_table->ncols() - num_keys;
    table_info* work_table;
    if (is_parallel) {
      work_table = shuffle_table(in_table, num_keys);
    } else {
      work_table = in_table;
    }
    bool dropna=true;
    // TODO: implement correct use of dropna in the computation.
    // See https://github.com/Bodo-inc/Bodo/issues/270
    table_info* out_table = groupby_and_nunique(work_table, num_keys, num_data_cols, dropna);
    if (is_parallel) delete_table_free_arrays(work_table);
    return out_table;
}


/**
 * This operation groups rows in a distributed table based on keys, and applies
 * a function(s) to a set of columns in each group (producing one row per
 * group). For each input column for which a function is applied, there will be
 * an output column with the result. The algorithm works as follows:
 * a) Group and Update: Each process does the following with its local table:
 *   - Determine to which group each row in the input table belongs to by using
 *     a hash table on the key columns (obtaining a row to group mapping).
 *   - Allocate output table (one row per group)
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
 * IMPORTANT: Currently there are cases where multiple functions are applied
 * to each column, but those functions run in numba-generated code (even if
 * they are built-in functions like sum). This C++ code needs to know how many
 * functions there are (passed via 'num_funcs'). But the number of
 * functions could vary between columns. Currently this only supports the same
 * number of functions per column.
 *
 * TODOs:
 * - offload multi-function case with built-ins to C++
 *   (https://github.com/Bodo-inc/Bodo/issues/277)
 * - support variable number of functions per column
 *
 * @param input table
 * @param number of key columns in the table
 * @param function to apply (see Bodo_FTypes::FTypeEnum)
 * @param Number of functions to apply per input data column (currently only
 *        used with ftype=agg)
 * @param true if needs to run in parallel (do a parallel shuffle)
 * @param external 'update' function (a function pointer).
 *        If ftype=agg, the update step happens in external JIT-compiled code,
 *        which must initialize output and/or redvar columns and apply the
 *        update function.
 * @param external 'combine' function (a function pointer).
 *        If ftype=agg, external code does the combine step (apply combine
 *        function to current table)
 * @param external 'eval' function (a function pointer).
 *        If ftype=agg, external code does the eval step.
 * @param dummy table containing type info for output table (for ftype=agg)
 */
table_info* groupby_and_aggregate(table_info* in_table, int64_t num_keys,
                                  int32_t ftype, int32_t num_funcs, bool is_parallel,
                                  void* update_cb, void* combine_cb, void* eval_cb,
                                  table_info* out_table_dummy) {

    agginfo_t agginfo = {out_table_dummy, (agg_table_op_fn)update_cb,
                         (agg_table_op_fn)combine_cb, (agg_eval_fn)eval_cb};

    // perform initial local aggregation
    int64_t num_data_cols = in_table->ncols() - num_keys;
    table_info* aggr_local =
        groupby_update(*in_table, num_keys, num_data_cols, ftype, num_funcs, agginfo);

    // shuffle step
    table_info* shuf_table = aggr_local;
    if (is_parallel) {
        shuf_table = shuffle_table(aggr_local, num_keys);
        delete_table_free_arrays(aggr_local);
    }

    // combine step
    table_info* out_table = groupby_combine(
        *shuf_table, num_keys, num_data_cols, combine_funcs[ftype], agginfo);
    delete_table_free_arrays(shuf_table);

    // eval step
    groupby_eval(*out_table, num_keys, num_data_cols, ftype, num_funcs, agginfo);

    return out_table;
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
    std::cout << "ascending=" << ascending << " na_position=" << na_position << "\n";
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
        out_arrs.push_back(
            RetrieveArray(in_table, ListPairWrite, i_col, -1, 0));
        //
#ifdef DEBUG_SORT
    std::cout << "OUTPUT:\n";
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
    DEBUG_PrintRefct(std::cout, out_arrs);
#endif
    return new table_info(out_arrs);
}





/** This function is the kernel function for the dropping of duplicated rows.
 * This C++ code should provide following functionality of pandas
 * drop_duplicates:
 * ---possibility of selecting columns for the identification
 * ---possibility of keeping first, last or removing all entries with duplicate
 * inplace operation for keeping the data in the same place is another problem.
 *
 * As for the join, this relies on using hash keys for the partitionning.
 *
 * External function used are "RetrieveArray" and "TestEqual"
 *
 * @param in_table : the input table
 * @param sum_value: the uint64_t containing all the values together.
 * @param keep: integer specifying the expected behavior.
 *        keep = 0 corresponds to the case of keep="first" keep first entry
 *        keep = 1 corresponds to the case of keep="last" keep last entry
 *        keep = 2 corresponds to the case of keep=False : remove all duplicates
 * @param in_place: True for returning the entry to the same place and False for
 * the opposite case.
 * @return the vector of pointers to be used.
 */
table_info* drop_duplicates_table_outplace(table_info* in_table,
                                           int64_t* subset_vect, int64_t keep) {
#undef DEBUG_DD
    size_t n_col = in_table->ncols();
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<array_info*> key_arrs;
    for (size_t iCol = 0; iCol < n_col; iCol++)
        if (subset_vect[iCol] == 1) key_arrs.push_back(in_table->columns[iCol]);
    size_t n_key = key_arrs.size();
#ifdef DEBUG_DD
    std::cout << "INPUT:\n";
    std::cout << "n_col=" << n_col << " n_rows=" << n_rows << " n_key=" << n_key
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
        bool test = TestEqual(key_arrs, n_key, shift_A, iRowA, shift_B, iRowB);
        return test;
    };
    // The entList contains the hash of the short table.
    // We address the entry by the row index. We store all the rows which are
    // identical in the std::vector.
    MAP_CONTAINER <size_t, size_t, std::function<size_t(size_t)>, std::function<bool(size_t,size_t)>> entSet({}, hash_fct, equal_fct);
    // The loop over the short table.
    // entries are stored one by one and all of them are put even if identical
    // in value.
    std::vector<int64_t> ListRow;
    uint64_t next_ent = 0;
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        size_t& group = entSet[i_row];
        if (group == 0) {
            next_ent++;
            group = next_ent;
            ListRow.push_back(i_row);
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
#ifdef DEBUG_DD
    std::cout << "|ListPairWrite|=" << ListPairWrite.size() << "\n";
#endif
    // Now building the out_arrs array.
    std::vector<array_info*> out_arrs;
    // Inserting the left data
    for (size_t i_col = 0; i_col < n_col; i_col++)
        out_arrs.push_back(
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
        PyErr_SetString(PyExc_RuntimeError, "bool size mistmatch between C++ and NumPy!");
        return NULL;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float32");
    if ((size_t)PyNumber_AsSsize_t(PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) != sizeof(float)) {
        PyErr_SetString(PyExc_RuntimeError, "float32 size mistmatch between C++ and NumPy!");
        return NULL;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float64");
    if ((size_t)PyNumber_AsSsize_t(PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) != sizeof(double)) {
        PyErr_SetString(PyExc_RuntimeError, "float64 size mistmatch between C++ and NumPy!");
        return NULL;
    }

    combine_funcs[Bodo_FTypes::sum] = Bodo_FTypes::sum;
    combine_funcs[Bodo_FTypes::count] = Bodo_FTypes::sum;
    combine_funcs[Bodo_FTypes::mean] =
        Bodo_FTypes::sum;  // sum totals and counts
    combine_funcs[Bodo_FTypes::min] = Bodo_FTypes::min;
    combine_funcs[Bodo_FTypes::max] = Bodo_FTypes::max;
    combine_funcs[Bodo_FTypes::prod] = Bodo_FTypes::prod;
    combine_funcs[Bodo_FTypes::var] = Bodo_FTypes::var_combine;
    combine_funcs[Bodo_FTypes::std] = Bodo_FTypes::var_combine;
    combine_funcs[Bodo_FTypes::agg] = Bodo_FTypes::agg;

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
        m, "drop_duplicates_table_outplace",
        PyLong_FromVoidPtr((void*)(&drop_duplicates_table_outplace)));
    PyObject_SetAttrString(m, "groupby_and_aggregate",
                           PyLong_FromVoidPtr((void*)(&groupby_and_aggregate)));
    PyObject_SetAttrString(m, "groupby_and_aggregate_nunique",
                           PyLong_FromVoidPtr((void*)(&groupby_and_aggregate_nunique)));
    return m;
}
