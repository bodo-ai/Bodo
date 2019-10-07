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

    return m;
}

