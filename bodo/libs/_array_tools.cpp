/**
 * @file _array_tools.cpp
 * @author Ehsan (ehsan@bodo-inc.com)
 * @brief Tools for handling bodo array, such as passing to C/C++ structures/functions
 * @date 2019-10-06
 */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "mpi.h"
#include <cstdio>
#include <numeric>
#include <iostream>
#include "_bodo_common.h"
#define ALIGNMENT 64  // preferred alignment for AVX512


array_info* string_array_to_info(uint64_t n_items, uint64_t n_chars, char* data, char *offsets, char* null_bitmap, NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::STRING, Bodo_CTypes::STRING, n_items, n_chars, data, offsets, NULL, null_bitmap, meminfo);
}


array_info* numpy_array_to_info(uint64_t n_items, char* data, int typ_enum, NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NUMPY, (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, data, NULL, NULL, NULL, meminfo);
}


void info_to_string_array(array_info* info, uint64_t* n_items, uint64_t* n_chars, char** data, char** offsets, char** null_bitmap, NRT_MemInfo** meminfo)
{
    if (info->arr_type != bodo_array_type::STRING)
    {
        PyErr_SetString(PyExc_RuntimeError, "info_to_string_array requires string input");
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


void info_to_numpy_array(array_info* info, uint64_t* n_items, char** data, NRT_MemInfo** meminfo)
{
    if (info->arr_type != bodo_array_type::NUMPY)
    {
        PyErr_SetString(PyExc_RuntimeError, "info_to_numpy_array requires numpy input");
        return;
    }
    *n_items = info->length;
    *data = info->data1;
    *meminfo = info->meminfo;
    return;
}


int64_t get_item_size(Bodo_CTypes::CTypeEnum typ_enum)
{
    switch (typ_enum)
    {
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
        PyErr_SetString(PyExc_RuntimeError, "Invalid item size call on string type");
        return 0;
    }
    PyErr_SetString(PyExc_RuntimeError, "Invalid item size call on unknown type");
    return 0;
}


array_info* alloc_numpy(int64_t length, Bodo_CTypes::CTypeEnum typ_enum)
{
    int64_t size = length * get_item_size(typ_enum);
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    return new array_info(bodo_array_type::NUMPY, typ_enum, length, -1, data, NULL, NULL, NULL, meminfo);
}


array_info* alloc_string_array(int64_t length, int64_t n_chars)
{
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_dtor_safe(sizeof(str_arr_payload), (NRT_dtor_function)dtor_string_array);
    str_arr_payload* payload = (str_arr_payload*)meminfo->data;
    allocate_string_array(&(payload->offsets), &(payload->data), &(payload->null_bitmap), length, n_chars);
    return new array_info(bodo_array_type::STRING, Bodo_CTypes::STRING, length, n_chars, payload->data, (char*)payload->offsets, NULL, (char*)payload->null_bitmap, meminfo);
}



table_info* arr_info_list_to_table(array_info** arrs, int64_t n_arrs)
{
    std::vector<array_info*> columns = std::vector<array_info*>(n_arrs);
    for(size_t i=0; i<n_arrs; i++)
    {
        columns[i] = arrs[i];
    }
    return new table_info(columns);
}


array_info* info_from_table(table_info* table, int64_t col_ind)
{
    return table->columns[col_ind];
}


void delete_table(table_info* table)
{
    for(array_info* a: table->columns)
    {
        delete a;
    }
    delete table;
    return;
}


template <class T>
void hash_array_inner(int *out_hashes, T* data, int64_t n_rows)
{
    for (size_t i=0; i<n_rows; i++)
        out_hashes[i] = std::hash<T>{}(data[i]);
}


void hash_array(int *out_hashes, array_info* array, size_t n_rows)
{
    // dispatch to proper function
    // TODO: general dispatcher
    // TODO: string
    if (array->dtype == Bodo_CTypes::INT8)
        return hash_array_inner<int8_t>(out_hashes, (int8_t*)array->data1, n_rows);
    if (array->dtype == Bodo_CTypes::UINT8)
        return hash_array_inner<uint8_t>(out_hashes, (uint8_t*)array->data1, n_rows);
    if (array->dtype == Bodo_CTypes::INT16)
        return hash_array_inner<int16_t>(out_hashes, (int16_t*)array->data1, n_rows);
    if (array->dtype == Bodo_CTypes::UINT16)
        return hash_array_inner<uint16_t>(out_hashes, (uint16_t*)array->data1, n_rows);
    if (array->dtype == Bodo_CTypes::INT32)
        return hash_array_inner<int32_t>(out_hashes, (int32_t*)array->data1, n_rows);
    if (array->dtype == Bodo_CTypes::UINT32)
        return hash_array_inner<uint32_t>(out_hashes, (uint32_t*)array->data1, n_rows);
    if (array->dtype == Bodo_CTypes::INT64)
        return hash_array_inner<int64_t>(out_hashes, (int64_t*)array->data1, n_rows);
    if (array->dtype == Bodo_CTypes::UINT64)
        return hash_array_inner<uint64_t>(out_hashes, (uint64_t*)array->data1, n_rows);
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return hash_array_inner<float>(out_hashes, (float*)array->data1, n_rows);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return hash_array_inner<double>(out_hashes, (double*)array->data1, n_rows);
    PyErr_SetString(PyExc_RuntimeError, "Invalid data type for hash");
}


int* hash_keys(std::vector<array_info*> key_arrs)
{
    size_t n_rows = (size_t)key_arrs[0]->length;
    int *hashes = new int[n_rows];
    hash_array(hashes, key_arrs[0], n_rows);
    return hashes;
}


table_info* shuffle_table(table_info* in_table, int64_t n_keys)
{
    // error checking
    if (in_table->columns.size() <= 0 || n_keys <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return NULL;
    }

    // declare comm auxiliary data structures
    int n_pes, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    std::vector<int> send_count(n_pes, 0);
    std::vector<int> recv_count(n_pes);
    std::vector<int> send_disp(n_pes);
    std::vector<int> recv_disp(n_pes);
    std::vector<int> tmp_offset(n_pes);

    size_t n_rows = (size_t) in_table->columns[0]->length;
    std::vector<array_info*> key_arrs = std::vector<array_info*>(in_table->columns.begin(), in_table->columns.begin() + n_keys);

    // get hashes
    int* hashes = hash_keys(key_arrs);

    // get send count
    for(size_t i=0; i<n_rows; i++)
        send_count[hashes[i] % n_pes]++;

    // get recv count
    MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
    // printf("%d send counts %d %d\n", rank, send_count[0], send_count[1]);
    // printf("%d recv counts %d %d\n", rank, recv_count[0], recv_count[1]);

    // calc disps
    send_disp[0] = 0;
    for(int i=1; i<n_pes; i++)
        send_disp[i] = send_disp[i-1] + send_count[i-1];

    recv_disp[0] = 0;
    for(int i=1; i<n_pes; i++)
        recv_disp[i] = recv_disp[i-1] + recv_count[i-1];

    int total_recv = std::accumulate(recv_count.begin(), recv_count.end(), 0);
    // printf("%d total count %d\n", rank, total_recv);

    // allocate output array
    array_info *out_keys = alloc_numpy(total_recv, Bodo_CTypes::INT64);
    int64_t *out_k = (int64_t *)out_keys->data1;
    array_info *send_keys = alloc_numpy(total_recv, Bodo_CTypes::INT64);
    int64_t *send_k = (int64_t *)send_keys->data1;


    array_info *keys = in_table->columns[0];
    int64_t *k_data = (int64_t *)keys->data1;


    // fill send buffer
    tmp_offset = send_disp;
    // printf("rank %d offsets %d %d\n", rank, tmp_offset[0], tmp_offset[1]);
    for(size_t i=0; i<n_rows; i++) {
        int64_t k = k_data[i];
        int node = k%n_pes;
        int ind = tmp_offset[node];
        send_k[ind] = k;
        tmp_offset[node]++;
    }
    // printf("rank %d offsets %d %d\n", rank, tmp_offset[0], tmp_offset[1]);

    MPI_Alltoallv(send_k, send_count.data(), send_disp.data(), MPI_LONG_LONG_INT,
        out_k, recv_count.data(), recv_disp.data(), MPI_LONG_LONG_INT, MPI_COMM_WORLD);

    delete[] send_keys->meminfo;
    delete[] hashes;

    std::vector<array_info*> out_cols;
    out_cols.push_back(out_keys);
    return new table_info(out_cols);
}



PyMODINIT_FUNC PyInit_array_tools_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT, "array_tools_ext", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    // init numpy
    import_array();

    // DEC_MOD_METHOD(string_array_to_info);
    PyObject_SetAttrString(m, "string_array_to_info",
                            PyLong_FromVoidPtr((void*)(&string_array_to_info)));
    PyObject_SetAttrString(m, "numpy_array_to_info",
                            PyLong_FromVoidPtr((void*)(&numpy_array_to_info)));
    PyObject_SetAttrString(m, "info_to_string_array",
                            PyLong_FromVoidPtr((void*)(&info_to_string_array)));
    PyObject_SetAttrString(m, "info_to_numpy_array",
                            PyLong_FromVoidPtr((void*)(&info_to_numpy_array)));
    PyObject_SetAttrString(m, "alloc_numpy",
                            PyLong_FromVoidPtr((void*)(&alloc_numpy)));
    PyObject_SetAttrString(m, "alloc_string_array",
                            PyLong_FromVoidPtr((void*)(&alloc_string_array)));
    PyObject_SetAttrString(m, "arr_info_list_to_table",
                            PyLong_FromVoidPtr((void*)(&arr_info_list_to_table)));
    PyObject_SetAttrString(m, "info_from_table",
                            PyLong_FromVoidPtr((void*)(&info_from_table)));
    PyObject_SetAttrString(m, "delete_table",
                            PyLong_FromVoidPtr((void*)(&delete_table)));
    PyObject_SetAttrString(m, "shuffle_table",
                            PyLong_FromVoidPtr((void*)(&shuffle_table)));

    return m;
}

