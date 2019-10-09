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
#include "_distributed.h"
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


template <class T>
void fill_send_array_inner(T* send_buff, T* data, int *hashes, std::vector<int> &send_disp, int n_pes, size_t n_rows)
{
    std::vector<int> tmp_offset(send_disp);
    for(size_t i=0; i<n_rows; i++) {
        int node = hashes[i] % n_pes;
        int ind = tmp_offset[node];
        send_buff[ind] = data[i];
        tmp_offset[node]++;
    }
}


void fill_send_array(array_info* send_arr, array_info *array, int *hashes, std::vector<int> &send_disp, int n_pes)
{
    size_t n_rows = (size_t)array->length;
    // dispatch to proper function
    // TODO: general dispatcher
    // TODO: string
    if (array->dtype == Bodo_CTypes::INT8)
        return fill_send_array_inner<int8_t>((int8_t*)send_arr->data1, (int8_t*)array->data1, hashes, send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT8)
        return fill_send_array_inner<uint8_t>((uint8_t*)send_arr->data1, (uint8_t*)array->data1, hashes, send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT16)
        return fill_send_array_inner<int16_t>((int16_t*)send_arr->data1, (int16_t*)array->data1, hashes, send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT16)
        return fill_send_array_inner<uint16_t>((uint16_t*)send_arr->data1, (uint16_t*)array->data1, hashes, send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT32)
        return fill_send_array_inner<int32_t>((int32_t*)send_arr->data1, (int32_t*)array->data1, hashes, send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT32)
        return fill_send_array_inner<uint32_t>((uint32_t*)send_arr->data1, (uint32_t*)array->data1, hashes, send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT64)
        return fill_send_array_inner<int64_t>((int64_t*)send_arr->data1, (int64_t*)array->data1, hashes, send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT64)
        return fill_send_array_inner<uint64_t>((uint64_t*)send_arr->data1, (uint64_t*)array->data1, hashes, send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return fill_send_array_inner<float>((float*)send_arr->data1, (float*)array->data1, hashes, send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return fill_send_array_inner<double>((double*)send_arr->data1, (double*)array->data1, hashes, send_disp, n_pes, n_rows);
    PyErr_SetString(PyExc_RuntimeError, "Invalid data type for send fill");
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

    size_t n_rows = (size_t) in_table->columns[0]->length;
    std::vector<array_info*> key_arrs = std::vector<array_info*>(in_table->columns.begin(), in_table->columns.begin() + n_keys);

    // get hashes
    int* hashes = hash_keys(key_arrs);

    // get send count
    for(size_t i=0; i<n_rows; i++)
        send_count[hashes[i] % n_pes]++;

    // get recv count
    MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // calc disps
    send_disp[0] = 0;
    for(int i=1; i<n_pes; i++)
        send_disp[i] = send_disp[i-1] + send_count[i-1];

    recv_disp[0] = 0;
    for(int i=1; i<n_pes; i++)
        recv_disp[i] = recv_disp[i-1] + recv_count[i-1];

    int total_recv = std::accumulate(recv_count.begin(), recv_count.end(), 0);
    // printf("%d total count %d\n", rank, total_recv);

    // allocate send and output arrays
    std::vector<array_info*> out_key_arrs;
    std::vector<array_info*> send_key_arrs;
    for (size_t i=0; i<(size_t)n_keys; i++)
    {
        send_key_arrs.push_back(alloc_numpy(total_recv, key_arrs[i]->dtype));
        out_key_arrs.push_back(alloc_numpy(total_recv, key_arrs[i]->dtype));
    }

    // fill send buffer and send
    for (size_t i=0; i<(size_t)n_keys; i++)
    {
        fill_send_array(send_key_arrs[i], key_arrs[i], hashes, send_disp, n_pes);
        MPI_Datatype mpi_typ = get_MPI_typ(key_arrs[i]->dtype);
        MPI_Alltoallv(send_key_arrs[i]->data1, send_count.data(), send_disp.data(), mpi_typ,
            out_key_arrs[i]->data1, recv_count.data(), recv_disp.data(), mpi_typ, MPI_COMM_WORLD);
    }

    // clean up
    for (size_t i=0; i<(size_t)n_keys; i++)
        delete[] send_key_arrs[i]->meminfo;
    delete[] hashes;

    return new table_info(out_key_arrs);
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

