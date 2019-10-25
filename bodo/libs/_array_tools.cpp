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
                              std::vector<uint8_t>& tmp_null_bytes,
                              std::vector<int> const& recv_count_null,
                              std::vector<int> const& recv_count) {
    size_t curr_tmp_byte = 0;  // current location in buffer with all data
    size_t curr_str = 0;       // current string in output bitmap
    // for each chunk
    for (size_t i = 0; i < recv_count.size(); i++) {
        size_t n_strs = recv_count[i];
        size_t n_bytes = recv_count_null[i];
        uint8_t* chunk_bytes = &tmp_null_bytes[curr_tmp_byte];
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
    for (size_t i = 0; i < (size_t)n_cols; i++)
        n_char_recvs[i] =
            std::accumulate(comm_info.recv_count_char[i].begin(),
                            comm_info.recv_count_char[i].end(), 0);

    // printf("%d total count %d\n", rank, total_recv);

    // fill send buffer and send
    std::vector<array_info*> out_arrs;
    for (size_t i = 0; i < (size_t)n_cols; i++) {
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


/* There is no need for using MPI since this has already been done
   following the _gen_par_shuffle.

 */
table_info* hash_join_table(table_info* key_left, table_info* key_right,
                            table_info* data_left, table_info* data_right,
                            bool is_left, bool is_right)
{
  //
  size_t nb_key_left = key_left->columns.size();
  size_t nb_key_right = key_right->columns.size();
  if (nb_key_left != nb_key_right) {
    PyErr_SetString(PyExc_RuntimeError, "key_left is inconsistent with key_right");
    return NULL;
  }
  size_t nb_key=nb_key_left;
  size_t nb_data_left = data_left->columns.size();
  size_t nb_data_right = data_right->columns.size();
  size_t n_rows = (size_t)key_left->columns[0]->length;
  //
  //
  std::vector<array_info*> key_arrs_left = std::vector<array_info*>(key_left->columns.begin(), key_left->columns.end());
  uint32_t seed = 0xb0d01288;
  uint32_t* hashes_left = hash_keys(key_arrs_left, seed);

  std::vector<array_info*> key_arrs_right = std::vector<array_info*>(key_right->columns.begin(), key_right->columns.end());
  uint32_t* hashes_right = hash_keys(key_arrs_right, seed);

  std::unordered_map<uint32_t, std::vector<std::pair<int,size_t>>> ListEnt;
  if (is_left) {
    // inserting the left entries.
    for (size_t iRowL=0; iRowL<n_rows; iRowL++) {
      std::pair<int,size_t> eEnt{0,iRowL};
      uint32_t eKey = hashes_left[iRowL];
      if (ListEnt.count(eKey) == 0) {
        ListEnt[eKey] = {eEnt};
      }
      else {
        ListEnt[eKey].push_back(eEnt);
      }
    }
    for (size_t iRowR=0; iRowR<n_rows; iRowR++) {
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
    for (size_t iRowR=0; iRowR<n_rows; iRowR++) {
      std::pair<int,size_t> eEnt{1,iRowR};
      uint32_t eKey=hashes_right[iRowR];
      if (ListEnt.count(eKey) == 0) {
        ListEnt[eKey] = {eEnt};
      }
      else {
        ListEnt[eKey].push_back(eEnt);
      }
    }
    for (size_t iRowL=0; iRowL<n_rows; iRowL++) {
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
  //
  // Testing equality of entries
  //
  auto TestEqual=[&](table_info* key1, size_t const& iRow1, table_info* key2, size_t const& iRow2) -> bool {
    for (size_t iKey=0; iKey<nb_key; iKey++) {
      if (key1->columns[iKey]->arr_type == bodo_array_type::NUMPY) {
        uint64_t siztype = get_item_size(key1->columns[iKey]->dtype);
        for (uint64_t u=0; u<siztype; u++) {
          if (key1->columns[iKey]->data1[siztype*iRow1 + u] != key2->columns[iKey]->data1[siztype*iRow2 + u])
            return false;
        }
      }
      if (key1->columns[iKey]->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask1 = (uint8_t*)key1->columns[iKey]->null_bitmask;
        uint8_t* null_bitmask2 = (uint8_t*)key2->columns[iKey]->null_bitmask;
        bool bit1 = GetBit(null_bitmask1, iRow1);
        bool bit2 = GetBit(null_bitmask2, iRow2);
        if (bit1 != bit2)
          return false;
        uint64_t siztype = get_item_size(key1->columns[iKey]->dtype);
        for (uint64_t u=0; u<siztype; u++) {
          if (key1->columns[iKey]->data1[siztype*iRow1 + u] != key2->columns[iKey]->data1[siztype*iRow2 + u])
            return false;
        }
      }
      if (key1->columns[iKey]->arr_type == bodo_array_type::STRING) {
        uint32_t pos1_prev = ((uint32_t*)key1->columns[iKey]->data2)[iRow1];
        uint32_t pos1_next = ((uint32_t*)key1->columns[iKey]->data2)[iRow1+1];
        uint32_t pos2_prev = ((uint32_t*)key2->columns[iKey]->data2)[iRow2];
        uint32_t pos2_next = ((uint32_t*)key2->columns[iKey]->data2)[iRow2+1];
        uint32_t len1 = pos1_next - pos1_prev;
        uint32_t len2 = pos2_next - pos2_prev;
        if (len1 != len2)
          return false;
        for (uint32_t pos=0; pos<len1; pos++) {
          uint32_t pos1=pos1_prev + pos;
          uint32_t pos2=pos2_prev + pos;
          char charL = ((char*)key1->columns[iKey]->data1)[pos1];
          char charR = ((char*)key2->columns[iKey]->data1)[pos2];
          if (charL != charR)
            return false;
        }
      }
    }
    return true;
  };
  //
  // Now iterating and determining how many entries we have to do.
  //
  std::vector<std::pair<size_t, size_t>> ListPairWrite;
  for (auto & ePair : ListEnt) {
    std::vector<size_t> ListEntL, ListEntR;
    for (auto fPair : ePair.second) {
      int iSize=fPair.first;
      size_t iRow=fPair.second;
      if (iSize == 0)
        ListEntL.push_back(iRow);
      if (iSize == 1)
        ListEntR.push_back(iRow);
    }
    auto GetBlocks=[&](table_info* table, std::vector<size_t> const& ListEnt) -> std::vector<std::vector<size_t>> {
      int len=ListEnt.size();
      std::vector<int> ListStatus(len,0);
      std::vector<std::vector<size_t>> ListListEnt;
      for (int i1=0; i1<len; i1++) {
        if (ListStatus[i1] == 0) {
          ListStatus[i1]=1;
          std::vector<size_t> eList{ListEnt[i1]};
          for (int i2=i1+1; i2<len; i2++) {
            if (ListStatus[i2] == 0) {
              bool test=TestEqual(table, ListEnt[i1], table, ListEnt[i2]);
              if (test) {
                ListStatus[i2]=1;
                eList.push_back(ListEnt[i2]);
              }
            }
          }
          ListListEnt.push_back(eList);
        }
      }
      return ListListEnt;
    };

    std::vector<std::vector<size_t>> ListListEntL = GetBlocks(key_left, ListEntL);
    std::vector<std::vector<size_t>> ListListEntR = GetBlocks(key_right, ListEntR);
    int nbBlockL=ListListEntL.size();
    int nbBlockR=ListListEntL.size();

    if (is_left) {
      auto GetEntry=[&](size_t const& iRowL) -> int {
        for (int iR=0; iR<nbBlockR; iR++) {
          bool test = TestEqual(key_left, iRowL, key_right, ListListEntR[iR][0]);
          if (test)
            return iR;
        }
        return -1;
      };
      for (int iL=0; iL<nbBlockL; iL++) {
        int iR=GetEntry(ListListEntL[iL][0]);
        if (iR == -1) {
          for (auto & uL : ListListEntL[iL])
            ListPairWrite.push_back({uL, -1});
        }
        else {
          for (auto & uL : ListListEntL[iL])
            for (auto & uR : ListListEntR[iR])
              ListPairWrite.push_back({uL,uR});
        }
      }
    }
    else {
      auto GetEntry=[&](size_t const& iRowR) -> int {
        for (int iL=0; iL<nbBlockL; iL++) {
          bool test = TestEqual(key_left, ListListEntL[iL][0], key_right, iRowR);
          if (test)
            return iL;
        }
        return -1;
      };
      for (int iR=0; iR<nbBlockR; iR++) {
        int iL=GetEntry(ListListEntR[iR][0]);
        if (iL == -1) {
          for (auto & uR : ListListEntR[iR])
            ListPairWrite.push_back({-1, uR});
        }
        else {
          for (auto & uL : ListListEntL[iL])
            for (auto & uR : ListListEntR[iR])
              ListPairWrite.push_back({uL,uR});
        }
      }
    }
  }
  size_t nbRowOut = ListPairWrite.size();
  //
  //
  //
  auto RetrieveArray = [&](array_info* in_arr, int const& ChoiceColumn) -> array_info* {
    auto get_iRow=[&](size_t const& iRowIn) -> size_t {
      if (ChoiceColumn == 0)
        return ListPairWrite[iRowIn].first;
      return ListPairWrite[iRowIn].second;
    };
    if (in_arr->arr_type == bodo_array_type::STRING) {
      int64_t n_chars=0;
      uint32_t* in_offsets = (uint32_t*)in_arr->data2;
      std::vector<uint32_t> ListSizes(nbRowOut);
      for (size_t iRow=0; iRow<nbRowOut; iRow++) {
        size_t iRowW=get_iRow(iRow);
        uint32_t end_offset = in_offsets[iRowW + 1];
        uint32_t start_offset = in_offsets[iRowW];
        uint32_t size = end_offset - start_offset;
        ListSizes[iRow] = size;
        n_chars += size;
      }
      array_info* out_arr = alloc_array(nbRowOut, n_chars,
                                        in_arr->arr_type, in_arr->dtype, 0);
      uint32_t pos = 0;
      uint32_t* out_offsets = (uint32_t*)out_arr->data2;
      for (size_t iRow=0; iRow<nbRowOut; iRow++) {
        size_t iRowW=get_iRow(iRow);
        uint32_t start_offset = in_offsets[iRowW];
        uint32_t size = ListSizes[iRow];
        out_offsets[iRow] = pos;
        for (uint32_t i=0; i<size; i++) {
          out_arr->data1[pos] = in_arr->data1[start_offset];
          pos++;
          start_offset++;
        }
        pos += size;
      }
      out_offsets[nbRowOut] = pos;
      return out_arr;
    }
    if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
      array_info* out_arr = alloc_array(nbRowOut, -1,
                                        in_arr->arr_type, in_arr->dtype, 0);
      uint8_t* in_null_bitmask = (uint8_t*)in_arr->null_bitmask;
      uint8_t* out_null_bitmask = (uint8_t*)out_arr->null_bitmask;
      uint64_t siztype = get_item_size(in_arr->dtype);
      for (size_t iRow=0; iRow<nbRowOut; iRow++) {
        size_t iRowW=get_iRow(iRow);
        //
        for (uint64_t u=0; u<siztype; u++)
          out_arr->data1[siztype*iRow + u] = in_arr->data1[siztype*iRowW + u];
        //
        bool bit = GetBit(in_null_bitmask, iRowW);
        SetBitTo(out_null_bitmask, iRow, bit);
      }
      return out_arr;
    }
    //
    // NUMPY case
    array_info* out_arr = alloc_array(nbRowOut, -1,
                                      in_arr->arr_type, in_arr->dtype, 0);
    uint64_t siztype = get_item_size(in_arr->dtype);
    for (size_t iRow=0; iRow<nbRowOut; iRow++) {
      size_t iRowW=get_iRow(iRow);
      //
        for (uint64_t u=0; u<siztype; u++)
          out_arr->data1[siztype*iRow + u] = in_arr->data1[siztype*iRowW + u];
    }
    return out_arr;
  };
  std::vector<array_info*> out_arrs;
  for (size_t i=0; i<nb_data_left; i++)
    out_arrs.push_back(RetrieveArray(data_left->columns[i], 0));
  for (size_t i=0; i<nb_data_right; i++)
    out_arrs.push_back(RetrieveArray(data_right->columns[i], 1));
  for (size_t i=0; i<nb_key; i++)
    out_arrs.push_back(RetrieveArray(key_left->columns[i], 0));
  for (size_t i=0; i<nb_key; i++)
    out_arrs.push_back(RetrieveArray(key_right->columns[i], 1));
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
