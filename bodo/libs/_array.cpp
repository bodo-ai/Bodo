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

array_info* decimal_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                  char* null_bitmap, NRT_MemInfo* meminfo,
                                  NRT_MemInfo* meminfo_bitmask,
                                  int32_t precision, int32_t scale) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, data,
                          NULL, NULL, null_bitmap, meminfo, meminfo_bitmask,
                          precision, scale);
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

/**
 * @brief count the total number of data elements in list(item) arrays
 *
 * @param list_arr_obj list(item) array object
 * @return int64_t total number of data elements
 */
int64_t count_total_elems_list_array(PyObject* list_arr_obj) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return -1;                     \
    }
    // get pd.NA object to check for new NA kind
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    Py_ssize_t n = PyObject_Size(list_arr_obj);
    int64_t n_lists = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* s = PySequence_GetItem(list_arr_obj, i);
        CHECK(s, "getting element failed");
        // Pandas stores NA as either None or nan
        if (!(s == Py_None ||
              (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s)))) ||
            s == C_NA) {
            n_lists += PyObject_Size(s);
        }
        Py_DECREF(s);
    }
    return n_lists;
#undef CHECK
}

/**
 * @brief convert Python object to native value and copies it to data buffer
 *
 * @param data data buffer for output native value
 * @param ind index within the buffer for output
 * @param item python object to read
 * @param dtype data type
 */
inline void copy_item_to_buffer(uint8_t* data, Py_ssize_t ind, PyObject* item,
                                Bodo_CTypes::CTypeEnum dtype) {
    if (dtype == Bodo_CTypes::INT64) {
        int64_t* ptr = (int64_t*)data;
        ptr[ind] = PyLong_AsLongLong(item);
    } else if (dtype == Bodo_CTypes::FLOAT64) {
        double* ptr = (double*)data;
        ptr[ind] = PyFloat_AsDouble(item);
    } else
        std::cerr << "data type " << dtype
                  << " not supported for unboxing list(item) array."
                  << std::endl;
}

/**
 * @brief compute offsets, data, and null_bitmap values for list(item) array
 * from an array of lists of values.
 *
 * @param list_item_arr_obj Python Sequence object, intended to be an array of
 * lists of items.
 * @param data data buffer to be filled with all values
 * @param offsets offsets buffer to be filled with computed offsets
 * @param null_bitmap nulls buffer to be filled
 * @param dtype data type of values, currently only float64 and int64 supported.
 */
void list_item_array_from_sequence(PyObject* list_arr_obj, uint8_t* data,
                                   uint32_t* offsets, uint8_t* null_bitmap,
                                   Bodo_CTypes::CTypeEnum dtype) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }

    CHECK(PySequence_Check(list_arr_obj), "expecting a PySequence");
    CHECK(data && offsets && null_bitmap, "buffer arguments must not be NULL");

    // get pd.NA object to check for new NA kind
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    Py_ssize_t n = PyObject_Size(list_arr_obj);

    int64_t curr_item_ind = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        offsets[i] = curr_item_ind;
        PyObject* s = PySequence_GetItem(list_arr_obj, i);
        CHECK(s, "getting list(item) array element failed");
        // Pandas stores NA as either None or nan
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA) {
            // set null bit to 0
            SetBitTo(null_bitmap, i, 0);
        } else {
            // set null bit to 1
            null_bitmap[i / 8] |= kBitmask[i % 8];
            // check list
            CHECK(PySequence_Check(s), "expecting a list");
            Py_ssize_t n_items = PyObject_Size(s);
            for (Py_ssize_t j = 0; j < n_items; j++) {
                PyObject* v = PySequence_GetItem(s, j);
                copy_item_to_buffer(data, curr_item_ind, v, dtype);
                Py_DECREF(v);
                curr_item_ind++;
            }
            Py_DECREF(s);
        }
    }
    offsets[n] = curr_item_ind;

    Py_DECREF(pd_mod);
    Py_DECREF(C_NA);
#undef CHECK
}

/**
 * @brief convert native value from buffer to Python object
 *
 * @param data data buffer for input native value (only int64/float64 currently)
 * @param ind index within the buffer
 * @param dtype data type
 * @return python object for value
 */
inline PyObject* value_to_pyobject(const uint8_t* data, int64_t ind,
                                   Bodo_CTypes::CTypeEnum dtype) {
    // TODO: support other types
    if (dtype == Bodo_CTypes::INT64) {
        int64_t* ptr = (int64_t*)data;
        return PyLong_FromLongLong(ptr[ind]);
    } else if (dtype == Bodo_CTypes::FLOAT64) {
        double* ptr = (double*)data;
        return PyFloat_FromDouble(ptr[ind]);
    } else
        std::cerr << "data type " << dtype
                  << " not supported for boxing list(item) array." << std::endl;
    return NULL;
}

/**
 * @brief create a numpy array of lists of item objects from a ListItemArray
 *
 * @param num_lists number of lists in input array
 * @param buffer all values
 * @param offsets offsets to data
 * @param null_bitmap null bitmask
 * @param dtype data type of values (currently, only int/float)
 * @return numpy array of list of item objects
 */
void* np_array_from_list_item_array(int64_t num_lists, const uint8_t* buffer,
                                    const uint32_t* offsets,
                                    const uint8_t* null_bitmap,
                                    Bodo_CTypes::CTypeEnum dtype) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return NULL;                   \
    }

    // allocate array and get nan object
    npy_intp dims[] = {num_lists};
    PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    CHECK(ret, "allocating numpy array failed");
    int err;
    PyObject* np_mod = PyImport_ImportModule("numpy");
    CHECK(np_mod, "importing numpy module failed");
    PyObject* nan_obj = PyObject_GetAttrString(np_mod, "nan");
    CHECK(nan_obj, "getting np.nan failed");

    size_t curr_value = 0;
    // for each list item
    for (int64_t i = 0; i < num_lists; ++i) {
        // set nan if item is NA
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (is_na(null_bitmap, i)) {
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, nan_obj);
            CHECK(err == 0, "setting item in numpy array failed");
            continue;
        }

        // alloc list item
        Py_ssize_t n_vals = (Py_ssize_t)(offsets[i + 1] - offsets[i]);
        PyObject* l = PyList_New(n_vals);

        for (Py_ssize_t j = 0; j < n_vals; j++) {
            PyObject* s = value_to_pyobject(buffer, curr_value, dtype);
            CHECK(s, "creating Python int/float object failed");
            PyList_SET_ITEM(l, j, s);  // steals reference to s!
            curr_value++;
        }

        err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, l);
        CHECK(err == 0, "setting item in numpy array failed");
        Py_DECREF(l);
    }

    Py_DECREF(np_mod);
    Py_DECREF(nan_obj);
    return ret;
#undef CHECK
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
    PyObject_SetAttrString(
        m, "count_total_elems_list_array",
        PyLong_FromVoidPtr((void*)(&count_total_elems_list_array)));
    PyObject_SetAttrString(
        m, "list_item_array_from_sequence",
        PyLong_FromVoidPtr((void*)(&list_item_array_from_sequence)));
    PyObject_SetAttrString(
        m, "np_array_from_list_item_array",
        PyLong_FromVoidPtr((void*)(&np_array_from_list_item_array)));
    return m;
}
