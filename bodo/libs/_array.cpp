// Copyright (C) 2019 Bodo Inc. All rights reserved.
/**
 * @author Ehsan (ehsan@bodo-inc.com)
 * @brief Bodo array and table C++ library to allocate arrays, perform parallel
 * shuffle, perform array and table operations like join, groupby
 * @date 2019-10-06
 */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "_array_operations.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_groupby.h"
#include "_join.h"
#include "_shuffle.h"

MPI_Datatype decimal_mpi_type = MPI_DATATYPE_NULL;

array_info* list_string_array_to_info(uint64_t n_items, uint64_t n_strings, uint64_t n_chars,
                                      char* data, char* data_offsets, char* index_offsets,
                                      char* null_bitmap, NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::LIST_STRING, Bodo_CTypes::LIST_STRING, n_items,
                          n_strings, n_chars, data, data_offsets, index_offsets, null_bitmap, meminfo,
                          NULL);
}

array_info* string_array_to_info(uint64_t n_items, uint64_t n_chars, char* data,
                                 char* offsets, char* null_bitmap,
                                 NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::STRING, Bodo_CTypes::STRING, n_items,
                          n_chars, -1, data, offsets, NULL, null_bitmap, meminfo,
                          NULL);
}

array_info* numpy_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NUMPY,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1, data,
                          NULL, NULL, NULL, meminfo, NULL);
}

array_info* nullable_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                   char* null_bitmap, NRT_MemInfo* meminfo,
                                   NRT_MemInfo* meminfo_bitmask) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1, data,
                          NULL, NULL, null_bitmap, meminfo, meminfo_bitmask);
}

array_info* decimal_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                  char* null_bitmap, NRT_MemInfo* meminfo,
                                  NRT_MemInfo* meminfo_bitmask,
                                  int32_t precision, int32_t scale) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1, data,
                          NULL, NULL, null_bitmap, meminfo, meminfo_bitmask,
                          precision, scale);
}


void info_to_list_string_array(array_info* info,
                               uint64_t* n_items, uint64_t* n_strings, uint64_t* n_chars,
                               char** data, char** data_offsets, char** index_offsets,
                               char** null_bitmap, NRT_MemInfo** meminfo) {
    if (info->arr_type != bodo_array_type::LIST_STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "info_to_list_string_array requires list string input");
        return;
    }
    *n_items = info->length;
    *n_strings = info->n_sub_elems;
    *n_chars = info->n_sub_sub_elems;
    *data = info->data1;
    *data_offsets = info->data2;
    *index_offsets = info->data3;
    *null_bitmap = info->null_bitmask;
    *meminfo = info->meminfo;
}



void info_to_string_array(array_info* info, uint64_t* n_items,
                          uint64_t* n_chars, char** data, char** offsets,
                          char** null_bitmap, NRT_MemInfo** meminfo) {
    if (info->arr_type != bodo_array_type::STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "info_to_string_array requires string input");
        return;
    }
    *n_items = info->length;
    *n_chars = info->n_sub_elems;
    *data = info->data1;
    *offsets = info->data2;
    *null_bitmap = info->null_bitmask;
    *meminfo = info->meminfo;
}

void info_to_numpy_array(array_info* info, uint64_t* n_items, char** data,
                         NRT_MemInfo** meminfo) {
    if (info->arr_type != bodo_array_type::NUMPY) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "info_to_numpy_array requires numpy input");
        return;
    }
    *n_items = info->length;
    *data = info->data1;
    *meminfo = info->meminfo;
}

void info_to_nullable_array(array_info* info, uint64_t* n_items,
                            uint64_t* n_bytes, char** data, char** null_bitmap,
                            NRT_MemInfo** meminfo,
                            NRT_MemInfo** meminfo_bitmask) {
    if (info->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "info_to_nullable_array requires nullable input");
        return;
    }
    *n_items = info->length;
    *n_bytes = (info->length + 7) >> 3;
    *data = info->data1;
    *null_bitmap = info->null_bitmask;
    *meminfo = info->meminfo;
    *meminfo_bitmask = info->meminfo_bitmask;
}

table_info* arr_info_list_to_table(array_info** arrs, int64_t n_arrs) {
    std::vector<array_info*> columns(arrs, arrs + n_arrs);
    return new table_info(columns);
}

array_info* info_from_table(table_info* table, int64_t col_ind) {
    return table->columns[col_ind];
}

PyMODINIT_FUNC PyInit_array_ext(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "array_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // init numpy
    import_array();

    bodo_common_init();

    // initialize decimal_mpi_type
    // TODO: free when program exits
    if (decimal_mpi_type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(2, MPI_LONG_LONG_INT, &decimal_mpi_type);
        MPI_Type_commit(&decimal_mpi_type);
    }

    groupby_init();

    // DEC_MOD_METHOD(string_array_to_info);
    PyObject_SetAttrString(m, "list_string_array_to_info",
                           PyLong_FromVoidPtr((void*)(&list_string_array_to_info)));
    PyObject_SetAttrString(m, "string_array_to_info",
                           PyLong_FromVoidPtr((void*)(&string_array_to_info)));
    PyObject_SetAttrString(m, "numpy_array_to_info",
                           PyLong_FromVoidPtr((void*)(&numpy_array_to_info)));
    PyObject_SetAttrString(
        m, "nullable_array_to_info",
        PyLong_FromVoidPtr((void*)(&nullable_array_to_info)));
    PyObject_SetAttrString(m, "decimal_array_to_info",
                           PyLong_FromVoidPtr((void*)(&decimal_array_to_info)));
    PyObject_SetAttrString(m, "info_to_string_array",
                           PyLong_FromVoidPtr((void*)(&info_to_string_array)));
    PyObject_SetAttrString(m, "info_to_list_string_array",
                           PyLong_FromVoidPtr((void*)(&info_to_list_string_array)));
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
    PyObject_SetAttrString(m, "drop_duplicates_table",
                           PyLong_FromVoidPtr((void*)(&drop_duplicates_table)));
    PyObject_SetAttrString(m, "groupby_and_aggregate",
                           PyLong_FromVoidPtr((void*)(&groupby_and_aggregate)));
    PyObject_SetAttrString(m, "array_isin",
                           PyLong_FromVoidPtr((void*)(&array_isin)));
    PyObject_SetAttrString(
        m, "compute_node_partition_by_hash",
        PyLong_FromVoidPtr((void*)(&compute_node_partition_by_hash)));
    return m;
}
