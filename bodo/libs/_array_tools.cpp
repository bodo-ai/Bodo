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
#include <iostream>
#include <numeric>
#include <map>
#include <unordered_map>
#include "_bodo_common.h"
#include "_distributed.h"
#include "_murmurhash3.cpp"
#include "mpi.h"
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

int64_t get_item_size(Bodo_CTypes::CTypeEnum typ_enum) {
    switch (typ_enum) {
        case Bodo_CTypes::INT8:
            return sizeof(int8_t);
        case Bodo_CTypes::UINT8:
            return sizeof(uint8_t);
        case Bodo_CTypes::INT16:
            return sizeof(int16_t);
        case Bodo_CTypes::UINT16:
            return sizeof(uint16_t);
        case Bodo_CTypes::INT32:
            return sizeof(int32_t);
        case Bodo_CTypes::UINT32:
            return sizeof(uint32_t);
        case Bodo_CTypes::INT64:
            return sizeof(int64_t);
        case Bodo_CTypes::UINT64:
            return sizeof(uint64_t);
        case Bodo_CTypes::FLOAT32:
            return sizeof(float);
        case Bodo_CTypes::FLOAT64:
            return sizeof(double);
        case Bodo_CTypes::STRING:
            PyErr_SetString(PyExc_RuntimeError,
                            "Invalid item size call on string type");
            return 0;
    }
    PyErr_SetString(PyExc_RuntimeError,
                    "Invalid item size call on unknown type");
    return 0;
}

array_info* alloc_numpy(int64_t length, Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t size = length * get_item_size(typ_enum);
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    return new array_info(bodo_array_type::NUMPY, typ_enum, length, -1, data,
                          NULL, NULL, NULL, meminfo, NULL);
}

array_info* alloc_nullable_array(int64_t length,
                                 Bodo_CTypes::CTypeEnum typ_enum,
                                 int64_t extra_null_bytes) {
    int64_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    int64_t size = length * get_item_size(typ_enum);
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
    return;
}

template <class T>
void hash_array_inner(uint32_t* out_hashes, T* data, size_t n_rows, const uint32_t seed) {
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

void hash_array(uint32_t* out_hashes, array_info* array, size_t n_rows, const uint32_t seed) {
    // dispatch to proper function
    // TODO: general dispatcher
    // XXX: assumes nullable array data for nulls is always consistent
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
        return hash_array_inner<float>(out_hashes, (float*)array->data1,
                                       n_rows, seed);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return hash_array_inner<double>(out_hashes, (double*)array->data1,
                                        n_rows, seed);
    if (array->arr_type == bodo_array_type::STRING)
        return hash_array_string(out_hashes, (char*)array->data1,
                                 (uint32_t*)array->data2, n_rows, seed);
    PyErr_SetString(PyExc_RuntimeError, "Invalid data type for hash");
}

template <class T>
void hash_array_combine_inner(uint32_t* out_hashes, T* data, size_t n_rows, const uint32_t seed) {
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
                               uint32_t* offsets, size_t n_rows, const uint32_t seed) {
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

void hash_array_combine(uint32_t* out_hashes, array_info* array,
                        size_t n_rows, const uint32_t seed) {
    // dispatch to proper function
    // TODO: general dispatcher
    if (array->dtype == Bodo_CTypes::INT8)
        return hash_array_combine_inner<int8_t>(out_hashes,
                                                (int8_t*)array->data1, n_rows, seed);
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
        return hash_array_combine_inner<double>(out_hashes,
                                                (double*)array->data1, n_rows, seed);
    if (array->arr_type == bodo_array_type::STRING)
        return hash_array_combine_string(out_hashes, (char*)array->data1,
                                         (uint32_t*)array->data2, n_rows, seed);
    PyErr_SetString(PyExc_RuntimeError, "Invalid data type for hash combine");
}

uint32_t* hash_keys(std::vector<array_info*> key_arrs, const uint32_t seed) {
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

/* 
 */
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



void shuffle_array(
    array_info* send_arr, array_info* out_arr, std::vector<int> const& send_count,
    std::vector<int> const& recv_count, std::vector<int> const& send_disp,
    std::vector<int> const& recv_disp, std::vector<int> const& send_count_char,
    std::vector<int> const& recv_count_char, std::vector<int> const& send_disp_char,
    std::vector<int> const& recv_disp_char, std::vector<int> const& send_count_null,
    std::vector<int> const& recv_count_null, std::vector<int> const& send_disp_null,
    std::vector<int> const& recv_disp_null, std::vector<uint8_t>& tmp_null_bytes) {
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
    if (in_table->columns.size() <= 0 || n_keys <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return NULL;
    }

    // declare comm auxiliary data structures
    int n_pes, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    mpi_comm_info comm_info(n_pes, in_table->columns);

    size_t n_rows = (size_t)in_table->columns[0]->length;
    size_t n_cols = in_table->columns.size();
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






template<typename T>
std::vector<char> GetVector(T const& val)
{
  const T* valptr= &val;
  const char* charptr = (char*)valptr;
  std::vector<char> V(sizeof(T));
  for (size_t u=0; u<sizeof(T); u++)
    V[u] = charptr[u];
  return V;
}

std::vector<char> RetrieveNaNentry(Bodo_CTypes::CTypeEnum const& dtype)
{
  if (dtype == Bodo_CTypes::INT8)
    return GetVector<int8_t>(0);
  if (dtype == Bodo_CTypes::UINT8)
    return GetVector<uint8_t>(0);
  if (dtype == Bodo_CTypes::INT16)
    return GetVector<int16_t>(0);
  if (dtype == Bodo_CTypes::UINT16)
    return GetVector<uint16_t>(0);
  if (dtype == Bodo_CTypes::INT32)
    return GetVector<int32_t>(0);
  if (dtype == Bodo_CTypes::UINT32)
    return GetVector<uint32_t>(0);
  if (dtype == Bodo_CTypes::INT64)
    return GetVector<int64_t>(0);
  if (dtype == Bodo_CTypes::UINT64)
    return GetVector<uint64_t>(0);
  if (dtype == Bodo_CTypes::FLOAT32)
    return GetVector<float>(std::nanf("1"));
  if (dtype == Bodo_CTypes::FLOAT64)
    return GetVector<double>(std::nan("1"));
  return {};
}


std::string GetStringExpression(Bodo_CTypes::CTypeEnum const& dtype, char* ptrdata)
{
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
  //
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
  //
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


std::vector<std::string> PrintColumn(array_info* arr)
{
  size_t nbRow=arr->length;
  std::vector<std::string> ListStr(nbRow);
  std::string strOut;
  if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
    uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
    uint64_t siztype = get_item_size(arr->dtype);
    for (size_t iRow=0; iRow<nbRow; iRow++) {
      bool bit=GetBit(null_bitmask, iRow);
      if (bit) {
        char* ptrdata1 = &(arr->data1[siztype*iRow]);
        strOut = GetStringExpression(arr->dtype, ptrdata1);
      }
      else {
        strOut="false";
      }
      ListStr[iRow] = strOut;
    }
  }
  if (arr->arr_type == bodo_array_type::NUMPY) {
    uint64_t siztype = get_item_size(arr->dtype);
    for (size_t iRow=0; iRow<nbRow; iRow++) {
      char* ptrdata1 = &(arr->data1[siztype*iRow]);
      strOut = GetStringExpression(arr->dtype, ptrdata1);
      ListStr[iRow] = strOut;
    }
  }
  if (arr->arr_type == bodo_array_type::STRING) {
    uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
    uint32_t* data2 = (uint32_t*)arr->data2;
    char* data1 = arr->data1;
    for (size_t iRow=0; iRow<nbRow; iRow++) {
      bool bit=GetBit(null_bitmask, iRow);
      if (bit) {
        uint32_t start_pos = data2[iRow];
        uint32_t end_pos   = data2[iRow+1];
        uint32_t len = end_pos - start_pos;
        char* strname;
        strname = new char[len+1];
        for (uint32_t i=0; i<len; i++) {
          strname[i] = data1[start_pos + i];
        }
        strname[len] = '\0';
        strOut = strname;
        delete [] strname;
      }
      else {
        strOut="false";
      }
      ListStr[iRow] = strOut;
    }
  }
  return ListStr;
}

void PrintSetOfColumn(std::ostream & os, std::vector<array_info*> const& ListArr)
{
  int nbCol=ListArr.size();
  if (nbCol == 0) {
    os << "Nothing to print really\n";
    return;
  }
  std::vector<int> ListLen(nbCol);
  int nbRowMax=0;
  for (int iCol=0; iCol<nbCol; iCol++) {
    int nbRow=ListArr[iCol]->length;
    if (nbRow > nbRowMax)
      nbRowMax = nbRow;
    ListLen[iCol] = nbRow;
  }
  std::vector<std::vector<std::string>> ListListStr;
  for (int iCol=0; iCol<nbCol; iCol++) {
    std::vector<std::string> LStr = PrintColumn(ListArr[iCol]);
    for (int iRow=ListLen[iCol]; iRow<nbRowMax; iRow++)
      LStr.push_back("");
    ListListStr.push_back(LStr);
  }
  std::vector<std::string> ListStrOut(nbRowMax);
  for (int iRow=0; iRow<nbRowMax; iRow++) {
    std::string str = std::to_string(iRow) + " :";
    ListStrOut[iRow]=str;
  }
  for (int iCol=0; iCol<nbCol; iCol++) {
    std::vector<int> ListLen(nbRowMax);
    size_t maxlen=0;
    for (int iRow=0; iRow<nbRowMax; iRow++) {
      size_t elen = ListListStr[iCol][iRow].size();
      ListLen[iRow] = elen;
      if (elen > maxlen)
        maxlen = elen;
    }
    for (int iRow=0; iRow<nbRowMax; iRow++) {
      std::string str = ListStrOut[iRow] + " " + ListListStr[iCol][iRow];
      size_t diff = maxlen - ListLen[iRow];
      for (size_t u=0; u<diff; u++)
        str += " ";
      ListStrOut[iRow] = str;
    }
  }
  for (int iRow=0; iRow<nbRowMax; iRow++)
    os << ListStrOut[iRow] << "\n";
}


void PrintRefct(std::ostream &os, std::vector<array_info*> const& ListArr)
{
  int nbCol=ListArr.size();
  auto GetType=[](bodo_array_type::arr_type_enum arr_type) -> std::string {
    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL)
      return "NULLABLE";
    if (arr_type == bodo_array_type::NUMPY)
      return "NUMPY";
    return "STRING";
  };
  auto GetNRTinfo=[](NRT_MemInfo* meminf) -> std::string {
    if (meminf == NULL)
      return "NULL";
    return "(refct=" + std::to_string(meminf->refct) + ")";
  };
  for (int iCol=0; iCol<nbCol; iCol++) {
    os << "iCol=" << iCol << " : " << GetType(ListArr[iCol]->arr_type) << " : meminfo=" << GetNRTinfo(ListArr[iCol]->meminfo) << " meminfo_bitmask=" << GetNRTinfo(ListArr[iCol]->meminfo_bitmask) << "\n";
  }
}



/*
  This implementation follows the Shared partition procedure.
  The data is partitioned and shuffled with the _gen_par_shuffle.
  ---
  The first stage is the partitioning of the data by using hashes
  and std::unordered_map array.
  ---
  Afterwards, secondary partitioning is done is the hashes match.
  Then the pairs of left/right origins are created for subsequent
  work. If a left key has no matching on the right, then value -1
  is put (thus the std::ptrdiff_t type is used).
  ---
  Then the arrays in output are subsequently created by RetrieveArray.
  Another interesting function is "TestEqual" for testing key equality.
  ---
  We use a single array on input since empty arrays cause problem for
  Python which cannot determine their type.
  Further debugging and optimization are needed.
 */
table_info* hash_join_table(table_info* in_table, int64_t nb_key_t, int64_t nb_data_left_t, int64_t nb_data_right_t, bool is_left, bool is_right)
{
  std::cout << "nb_key_t=" << nb_key_t << " nb_data_left_t=" << nb_data_left_t << " nb_data_right_t=" << nb_data_right_t << "\n";
  std::cout << "hash_join_table, step 1 is_left=" << is_left << " is_right=" << is_right << "\n";
  std::cout << "IN_TABLE:\n";
  PrintSetOfColumn(std::cout, in_table->columns);
  size_t nb_key = size_t(nb_key_t);
  size_t nb_data_left = size_t(nb_data_left_t);
  size_t nb_data_right = size_t(nb_data_right_t);
  size_t sum_dim = 2*nb_key + nb_data_left + nb_data_right;
  size_t nb_col = in_table->columns.size();
  if (nb_col != sum_dim) {
    PyErr_SetString(PyExc_RuntimeError, "incoherent dimensions");
    return NULL;
  }
  for (size_t i=0; i<nb_col; i++)
    std::cout << "1: i=" << i << " dtype=" << in_table->columns[i]->dtype << "\n";
  //
  size_t n_rows_left = (size_t)in_table->columns[0]->length;
  size_t n_rows_right = (size_t)in_table->columns[nb_key]->length;
  std::cout << "n_rows_left=" << n_rows_left << " n_rows_right=" << n_rows_right << "\n";
  //
  std::cout << "hash_join_table, step 2\n";
  std::vector<array_info*> key_arrs_left = std::vector<array_info*>(in_table->columns.begin(), in_table->columns.begin() + nb_key);
  uint32_t seed = 0xb0d01288;
  uint32_t* hashes_left = hash_keys(key_arrs_left, seed);

  std::cout << "hash_join_table, step 3\n";
  std::vector<array_info*> key_arrs_right = std::vector<array_info*>(in_table->columns.begin()+nb_key, in_table->columns.begin() + 2*nb_key);
  uint32_t* hashes_right = hash_keys(key_arrs_right, seed);
  //
  for (size_t i=0; i<n_rows_left; i++)
    std::cout << "i=" << i << " hashes_left=" << hashes_left[i] << "\n";
  for (size_t i=0; i<n_rows_right; i++)
    std::cout << "i=" << i << " hashes_right=" << hashes_right[i] << "\n";

  

  std::cout << "hash_join_table, step 4\n";
  std::unordered_map<uint32_t, std::vector<std::pair<int,size_t>>> ListEnt;
  // What we do here is actually an optimization (maybe premature?)
  // We insert entries on the left and right but if some entries
  // are not going to get used anyway, then we do not insert them.
  // The output of this is suboptimal and there are some duplication
  // but this is unavoidable at this point.
  if (is_left) {
    // inserting the left entries.
    for (size_t iRowL=0; iRowL<n_rows_left; iRowL++) {
      std::pair<int,size_t> eEnt{0,iRowL};
      uint32_t eKey = hashes_left[iRowL];
      if (ListEnt.count(eKey) == 0) {
        ListEnt[eKey] = {eEnt};
      }
      else {
        ListEnt[eKey].push_back(eEnt);
      }
    }
    for (size_t iRowR=0; iRowR<n_rows_right; iRowR++) {
      std::pair<int,size_t> eEnt{1,iRowR};
      uint32_t eKey=hashes_right[iRowR];
      if (ListEnt.count(eKey) == 0) {
        if (is_right)
          ListEnt[eKey] = {eEnt};
      }
      else {
        ListEnt[eKey].push_back(eEnt);
      }
    }
  }
  else {
    for (size_t iRowR=0; iRowR<n_rows_right; iRowR++) {
      std::pair<int,size_t> eEnt{1,iRowR};
      uint32_t eKey=hashes_right[iRowR];
      if (ListEnt.count(eKey) == 0) {
        ListEnt[eKey] = {eEnt};
      }
      else {
        ListEnt[eKey].push_back(eEnt);
      }
    }
    for (size_t iRowL=0; iRowL<n_rows_left; iRowL++) {
      std::pair<int,size_t> eEnt{0,iRowL};
      uint32_t eKey = hashes_left[iRowL];
      if (ListEnt.count(eKey) == 0) {
        if (is_left)
          ListEnt[eKey] = {eEnt};
      }
      else {
        ListEnt[eKey].push_back(eEnt);
      }
    }
  }
  std::cout << "|ListEnt|=" << ListEnt.size() << "\n";
  //
  // Testing equality of entries
  //
  std::cout << "hash_join_table, step 5\n";
  // This code test if two keys are equal (Before that the hash should have been used)
  // It is used that way because we assume that the left key have the same type as the
  // right keys.
  // The shift is used to precise whether we use the left keys or the right keys.
  // Equality means that all the columns are the same.
  // Thus the test iterates over the columns and if one is different then result is false.
  // We consider all types of bodo_array_type
  auto TestEqual=[&](size_t const& shift_key1, size_t const& iRow1, size_t const& shift_key2, size_t const& iRow2) -> bool {
    //    std::cout << "TestEqual, begin\n";
    for (size_t iKey=0; iKey<nb_key; iKey++) {
      //      std::cout << "iKey=" << iKey << " nb_key=" << nb_key << "\n";
      if (in_table->columns[shift_key1+iKey]->arr_type == bodo_array_type::NUMPY) {
        //        std::cout << "  Case 1\n";
        uint64_t siztype = get_item_size(in_table->columns[shift_key1+iKey]->dtype);
        for (uint64_t u=0; u<siztype; u++) {
          if (in_table->columns[shift_key1+iKey]->data1[siztype*iRow1 + u] != in_table->columns[shift_key2+iKey]->data1[siztype*iRow2 + u])
            return false;
        }
      }
      if (in_table->columns[shift_key1+iKey]->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        //        std::cout << "  Case 2\n";
        uint8_t* null_bitmask1 = (uint8_t*)in_table->columns[shift_key1+iKey]->null_bitmask;
        uint8_t* null_bitmask2 = (uint8_t*)in_table->columns[shift_key2+iKey]->null_bitmask;
        bool bit1 = GetBit(null_bitmask1, iRow1);
        bool bit2 = GetBit(null_bitmask2, iRow2);
        if (bit1 != bit2)
          return false;
        if (bit1) {
          uint64_t siztype = get_item_size(in_table->columns[shift_key1+iKey]->dtype);
          for (uint64_t u=0; u<siztype; u++) {
            if (in_table->columns[shift_key1+iKey]->data1[siztype*iRow1 + u] != in_table->columns[shift_key2+iKey]->data1[siztype*iRow2 + u])
              return false;
          }
        }
      }
      if (in_table->columns[shift_key1+iKey]->arr_type == bodo_array_type::STRING) {
        uint8_t* null_bitmask1 = (uint8_t*)in_table->columns[shift_key1+iKey]->null_bitmask;
        uint8_t* null_bitmask2 = (uint8_t*)in_table->columns[shift_key2+iKey]->null_bitmask;
        bool bit1 = GetBit(null_bitmask1, iRow1);
        bool bit2 = GetBit(null_bitmask2, iRow2);
        if (bit1 != bit2)
          return false;
        if (bit1) {
          uint32_t* data2_1 = (uint32_t*)in_table->columns[shift_key1+iKey]->data2;
          uint32_t* data2_2 = (uint32_t*)in_table->columns[shift_key2+iKey]->data2;
          uint32_t len1 = data2_1[iRow1+1] - data2_1[iRow1];
          uint32_t len2 = data2_2[iRow2+1] - data2_2[iRow2];
          if (len1 != len2)
            return false;
          uint32_t pos1_prev = data2_1[iRow1];
          uint32_t pos2_prev = data2_2[iRow2];
          char* data1_1 = (char*)in_table->columns[shift_key1+iKey]->data1;
          char* data1_2 = (char*)in_table->columns[shift_key2+iKey]->data1;
          for (uint32_t pos=0; pos<len1; pos++) {
            uint32_t pos1=pos1_prev + pos;
            uint32_t pos2=pos2_prev + pos;
            if (data1_1[pos1] != data1_2[pos2])
              return false;
          }
        }
      }
    }
    return true;
  };
  //
  // Now iterating and determining how many entries we have to do.
  //
  std::cout << "hash_join_table, step 6\n";
  std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
  for (auto & ePair : ListEnt) {
    //    std::cout << "hash_join_table, step 6.1\n";
    std::vector<size_t> ListEntL, ListEntR;
    for (auto fPair : ePair.second) {
      int idx=fPair.first;
      size_t iRow=fPair.second;
      if (idx == 0)
        ListEntL.push_back(iRow);
      if (idx == 1)
        ListEntR.push_back(iRow);
    }
    std::cout << "|ListEntL|=" << ListEntL.size() << " |ListEntR|=" << ListEntR.size() << "\n";
    //    std::cout << "hash_join_table, step 6.2\n";
    //
    // This function takes a list of lines in the in_table and return the list of line
    // by blocks if they have the same key.
    // The entry "shift" specifies the starting point of the keys.
    // This is because the keys are done on the left or right in the same data structure.
    //
    // The algorithm is nothing special with quadratic run-time in worst case.
    // It is expected not to be a problem since different keys with the same hash should
    // be rare. If all the keys with same hash are also identical then the running time
    // is linear.
    auto GetBlocks=[&](size_t const& shift, std::vector<size_t> const& ListEnt) -> std::vector<std::vector<size_t>> {
      int len=ListEnt.size();
      std::vector<int> ListStatus(len,0);
      std::vector<std::vector<size_t>> ListListEnt;
      for (int i1=0; i1<len; i1++) {
        if (ListStatus[i1] == 0) {
          ListStatus[i1]=1;
          std::vector<size_t> eList{ListEnt[i1]};
          for (int i2=i1+1; i2<len; i2++) {
            if (ListStatus[i2] == 0) {
              bool test=TestEqual(shift, ListEnt[i1], shift, ListEnt[i2]);
              if (test) {
                ListStatus[i2]=1;
                eList.push_back(ListEnt[i2]);
              }
            }
          }
          //          std::cout << "|eList|=" << eList.size() << "\n";
          ListListEnt.push_back(eList);
        }
      }
      //      std::cout << "|ListListEnt|=" << ListListEnt.size() << "\n";
      return ListListEnt;
    };
    //    std::cout << "hash_join_table, step 6.3\n";

    std::vector<std::vector<size_t>> ListListEntL = GetBlocks(0, ListEntL);
    //    std::cout << "hash_join_table, step 6.4\n";
    std::vector<std::vector<size_t>> ListListEntR = GetBlocks(nb_key, ListEntR);
    //    std::cout << "hash_join_table, step 6.5\n";
    size_t nbBlockL=ListListEntL.size();
    size_t nbBlockR=ListListEntR.size();
    std::cout << "nbBlockL=" << nbBlockL << " nbBlockR=" << nbBlockR << "\n";

    if (is_left) {
      //      std::cout << "hash_join_table, step 6.6\n";
      auto GetEntry=[&](size_t const& iRowL) -> std::ptrdiff_t {
        //        std::cout << "GetEntry : nbBlockR=" << nbBlockR << "\n";
        for (size_t iR=0; iR<nbBlockR; iR++) {
          //          std::cout << "GetEntry : iR=" << iR << " |ListListEntR[iR]|=" << ListListEntR[iR].size() << "\n";
          bool test = TestEqual(0, iRowL, nb_key, ListListEntR[iR][0]);
          //          std::cout << "GetEntry : test=" << test << "\n";
          if (test)
            return iR;
        }
        return -1;
      };
      //      std::cout << "hash_join_table, step 6.7\n";
      std::vector<int> ListStatus(nbBlockR,0);
      for (size_t iL=0; iL<nbBlockL; iL++) {
        //        std::cout << "iL=" << iL << " |ListListEntL[iL]|=" << ListListEntL[iL].size() << "\n";
        std::ptrdiff_t iR=GetEntry(ListListEntL[iL][0]);
        //        std::cout << "iR=" << iR << "\n";
        if (iR == -1) {
          for (auto & uL : ListListEntL[iL])
            ListPairWrite.push_back({uL, -1});
        }
        else {
          if (is_right)
            ListStatus[iR]=1;
          for (auto & uL : ListListEntL[iL])
            for (auto & uR : ListListEntR[iR])
              ListPairWrite.push_back({uL,uR});
        }
      }
      if (is_right) {
        for (size_t iR=0; iR<nbBlockR; iR++)
          if (ListStatus[iR] == 0)
            for (auto & uR : ListListEntR[iR])
              ListPairWrite.push_back({-1,uR});
      }
      //      std::cout << "hash_join_table, step 6.8\n";
    }
    else {
      //      std::cout << "hash_join_table, step 6.9\n";
      auto GetEntry=[&](size_t const& iRowR) -> std::ptrdiff_t {
        for (size_t iL=0; iL<nbBlockL; iL++) {
          bool test = TestEqual(0, ListListEntL[iL][0], nb_key, iRowR);
          if (test)
            return iL;
        }
        return -1;
      };
      //      std::cout << "hash_join_table, step 6.10\n";
      for (size_t iR=0; iR<nbBlockR; iR++) {
        std::ptrdiff_t iL=GetEntry(ListListEntR[iR][0]);
        if (iL == -1) {
          if (is_right) {
            for (auto & uR : ListListEntR[iR])
              ListPairWrite.push_back({-1, uR});
          }
        }
        else {
          std::cout << "|ListListEntL[iL]|=" << ListListEntL[iL].size() << " |ListListEntR[iR]|=" << ListListEntR[iR].size() << "\n";
          for (auto & uL : ListListEntL[iL])
            for (auto & uR : ListListEntR[iR])
              ListPairWrite.push_back({uL,uR});
        }
      }
      std::cout << "Now |ListPairWrite|=" << ListPairWrite.size() << "\n";
      //      std::cout << "hash_join_table, step 6.11\n";
    }
  }
  size_t nbRowOut = ListPairWrite.size();
  for (size_t iRowOut=0; iRowOut<nbRowOut; iRowOut++)
    std::cout << "iRowOut=" << iRowOut << " epair=" << ListPairWrite[iRowOut].first << " , " << ListPairWrite[iRowOut].second << "\n";

  std::cout << "hash_join_table, step 7 nbRowOut=" << nbRowOut << "\n";
  //
  // This function uses the combinatorial information computed in the "ListPairWrite"
  // array.
  // The other arguments shift1, shift2 and ChoiceColumn are for the choice of column
  // ---For inserting a left data,  call (i+2*nb_key, -1                        , 0)
  // ---For inserting a right data, call (-1        ,  i+2*nb_key + nb_data_left, 1)
  // ---For the key, we need to put both the left and right key column (shift1, shift2)
  //    and ChoiceColumn = 2
  //
  // The code considers all the cases in turn and creates the new array from it.
  //
  std::vector<array_info*> out_arrs;
  auto RetrieveArray = [&](size_t const& shift1, size_t const& shift2, int const& ChoiceColumn) -> void {
    array_info* out_arr = NULL;
    std::cout << "--------------------------------------------------------------\n";
    auto get_iRow=[&](size_t const& iRowIn) -> std::pair<size_t,std::ptrdiff_t> {
      std::pair<std::ptrdiff_t, std::ptrdiff_t> epair = ListPairWrite[iRowIn];
      if (ChoiceColumn == 0)
        return {shift1, epair.first};
      if (ChoiceColumn == 1)
        return {shift2, epair.second};
      if (epair.first != -1)
        return {shift1, epair.first};
      return {shift2, epair.second};
    };
    std::cout << "shift1=" << shift1 << " shift2=" << shift2 << " ChoiceColumn=" << ChoiceColumn << "\n";
    size_t eshift;
    if (ChoiceColumn == 0)
      eshift=shift1;
    if (ChoiceColumn == 1)
      eshift=shift2;
    if (ChoiceColumn == 2)
      eshift=shift1;
    std::cout << "RetrieveArray, step 1\n";
    if (in_table->columns[eshift]->arr_type == bodo_array_type::STRING) {
      std::cout << "RetrieveArray, step 2\n";
      int64_t n_chars=0;
      std::vector<uint32_t> ListSizes(nbRowOut);
      for (size_t iRow=0; iRow<nbRowOut; iRow++) {
        std::pair<size_t,std::ptrdiff_t> epair = get_iRow(iRow);
        uint32_t size=0;
        if (epair.second >= 0) {
          uint32_t* in_offsets = (uint32_t*)in_table->columns[epair.first]->data2;
          uint32_t end_offset = in_offsets[epair.second + 1];
          uint32_t start_offset = in_offsets[epair.second];
          size = end_offset - start_offset;
        }
        std::cout << "iRow=" << iRow << " size=" << size << "\n";
        ListSizes[iRow] = size;
        n_chars += size;
      }
      std::cout << "RetrieveArray, step 2.1 nbRowOut=" << nbRowOut << " n_chars=" << n_chars << "\n";
      out_arr = alloc_array(nbRowOut, n_chars,
                            in_table->columns[eshift]->arr_type,
                            in_table->columns[eshift]->dtype, 0);
      std::cout << "RetrieveArray, step 2.2\n";
      uint8_t* out_null_bitmask = (uint8_t*)out_arr->null_bitmask;
      uint32_t pos = 0;
      uint32_t* out_offsets = (uint32_t*)out_arr->data2;
      std::cout << "RetrieveArray, step 2.3\n";
      for (size_t iRow=0; iRow<nbRowOut; iRow++) {
        std::pair<size_t,std::ptrdiff_t> epair=get_iRow(iRow);
        uint32_t size = ListSizes[iRow];
        out_offsets[iRow] = pos;
        std::cout << "  iRow=" << iRow << " size=" << size << " pos=" << pos << " epair.second=" << epair.second << "\n";
        bool bit=false;
        if (epair.second >= 0) {
          uint8_t* in_null_bitmask = (uint8_t*)in_table->columns[epair.first]->null_bitmask;
          uint32_t* in_offsets = (uint32_t*)in_table->columns[epair.first]->data2;
          uint32_t start_offset = in_offsets[epair.second];
          for (uint32_t i=0; i<size; i++) {
            out_arr->data1[pos] = in_table->columns[epair.first]->data1[start_offset];
            pos++;
            start_offset++;
          }
          bit = GetBit(in_null_bitmask, epair.second);
          std::cout << "    bit=" << bit << "\n";
        }
        std::cout << "  iRow=" << iRow << " bit=" << bit << "\n";
        SetBitTo(out_null_bitmask, iRow, bit);
      }
      std::cout << "RetrieveArray, step 2.4 pos=" << pos << "\n";
      out_offsets[nbRowOut] = pos;
      std::cout << "RetrieveArray, step 2.5\n";
    }
    if (in_table->columns[eshift]->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
      std::cout << "RetrieveArray, step 3\n";
      out_arr = alloc_array(nbRowOut, -1,
                            in_table->columns[eshift]->arr_type,
                            in_table->columns[eshift]->dtype, 0);
      uint8_t* out_null_bitmask = (uint8_t*)out_arr->null_bitmask;
      uint64_t siztype = get_item_size(in_table->columns[eshift]->dtype);
      for (size_t iRow=0; iRow<nbRowOut; iRow++) {
        std::pair<size_t,std::ptrdiff_t> epair=get_iRow(iRow);
        bool bit=false;
        if (epair.second >= 0) {
          uint8_t* in_null_bitmask = (uint8_t*)in_table->columns[epair.first]->null_bitmask;
          for (uint64_t u=0; u<siztype; u++)
            out_arr->data1[siztype*iRow + u] = in_table->columns[epair.first]->data1[siztype*epair.second + u];
          //
          bit = GetBit(in_null_bitmask, epair.second);
        }
        SetBitTo(out_null_bitmask, iRow, bit);
      }
    }
    if (in_table->columns[eshift]->arr_type == bodo_array_type::NUMPY) {
      std::cout << "RetrieveArray, step 4\n";
      out_arr = alloc_array(nbRowOut, -1,
                            in_table->columns[eshift]->arr_type,
                            in_table->columns[eshift]->dtype, 0);
      uint64_t siztype = get_item_size(in_table->columns[eshift]->dtype);
      std::cout << "siztype=" << siztype << "\n";
      std::vector<char> vectNaN = RetrieveNaNentry(in_table->columns[eshift]->dtype);
      for (size_t iRow=0; iRow<nbRowOut; iRow++) {
        std::pair<size_t,std::ptrdiff_t> epair=get_iRow(iRow);
        std::cout << "iRow=" << iRow << " epair=" << epair.first << " , " << epair.second << "\n";
        //
        if (epair.second >= 0) {
          for (uint64_t u=0; u<siztype; u++)
            out_arr->data1[siztype*iRow + u] = in_table->columns[epair.first]->data1[siztype*epair.second + u];
        }
        else {
          for (uint64_t u=0; u<siztype; u++)
            out_arr->data1[siztype*iRow + u] = vectNaN[u];
        }
      }
    }
    out_arrs.push_back(out_arr);
  };
  std::cout << "hash_join_table, step 8\n";
  std::cout << "nb_key=" << nb_key << " nb_data_left=" << nb_data_left << " nb_data_right=" << nb_data_right << "\n";
  for (size_t i=0; i<nb_col; i++)
    std::cout << "2: i=" << i << " dtype=" << in_table->columns[i]->dtype << "\n";
  for (size_t i=0; i<nb_data_left; i++) {
    std::cout << "1: i=" << i << "\n";
    RetrieveArray(i + 2*nb_key, -1, 0);
  }
  std::cout << "hash_join_table, step 9\n";
  for (size_t i=0; i<nb_data_right; i++) {
    std::cout << "2: i=" << i << "\n";
    RetrieveArray(-1, i + 2*nb_key + nb_data_left, 1);
  }
  std::cout << "hash_join_table, step 10\n";
  for (size_t i=0; i<nb_key; i++) {
    std::cout << "3: i=" << i << "\n";
    RetrieveArray(i, i+nb_key, 2);
  }
  std::cout << "hash_join_table, step 11\n";
  for (size_t i=0; i<nb_key; i++) {
    std::cout << "4: i=" << i << "\n";
    RetrieveArray(i, i+nb_key, 2);
  }
  std::cout << "hash_join_table, step 12\n";
  PrintSetOfColumn(std::cout, out_arrs);
  PrintRefct(std::cout, out_arrs);
  std::cout << "hash_join_table, step 13\n";
  //
  delete[] hashes_left;
  delete[] hashes_right;
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
    return m;
}
