/**
 * @file _array_tools.cpp
 * @author Ehsan (ehsan@bodo-inc.com)
 * @brief Tools for handling bodo array, such as passing to C/C++ structures/functions
 * @date 2019-10-06
 */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cstdio>
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


array_info* alloc_numpy(int64_t length, Bodo_CTypes::CTypeEnum typ_enum)
{
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(length, ALIGNMENT);
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

    return m;
}

