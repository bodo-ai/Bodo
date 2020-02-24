// Copyright (C) 2020 Bodo Inc. All rights reserved.
// C/C++ code for DecimalArray handling
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>
#include "_bodo_common.h"
#include "arrow/util/decimal.h"

#define BYTES_PER_DECIMAL 16

extern "C" {

void* box_decimal_array(int64_t n, const uint8_t* data,
                        const uint8_t* null_bitmap, int scale);

PyMODINIT_FUNC PyInit_decimal_ext(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "decimal_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // init numpy
    import_array();

    PyObject_SetAttrString(m, "box_decimal_array",
                           PyLong_FromVoidPtr((void*)(&box_decimal_array)));
    return m;
}

/// @brief  Box native DecimalArray data to Numpy object array of
/// decimal.Decimal items
/// @return Numpy object array of decimal.Decimal
/// @param[in] n number of values
/// @param[in] data pointer to 128-bit values
/// @param[in] null_bitmap bitvector representing nulls (Arrow format)
void* box_decimal_array(int64_t n, const uint8_t* data,
                        const uint8_t* null_bitmap, int scale) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return NULL;                   \
    }

    auto gilstate = PyGILState_Ensure();

    npy_intp dims[] = {n};
    PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    CHECK(ret, "allocating numpy array failed");
    int err;

    // get decimal.Decimal constructor
    PyObject* dec_mod = PyImport_ImportModule("decimal");
    CHECK(dec_mod, "importing decimal module failed");
    PyObject* decimal_constructor = PyObject_GetAttrString(dec_mod, "Decimal");
    CHECK(decimal_constructor, "getting decimal.Decimal failed");

    for (int64_t i = 0; i < n; ++i) {
        // using Arrow's Decimal128 to get string representation
        // and call decimal.Decimal(), similar to Arrow to pandas code.
        uint64_t high_bytes =
            *(uint64_t*)(data + i * BYTES_PER_DECIMAL + sizeof(uint64_t));
        uint64_t low_bytes = *(uint64_t*)(data + i * BYTES_PER_DECIMAL);
        arrow::Decimal128 arrow_decimal(high_bytes, low_bytes);
        std::string str = arrow_decimal.ToString(scale);
        PyObject* d =
            PyObject_CallFunction(decimal_constructor, "s", str.c_str());

        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (!is_na(null_bitmap, i))
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, d);
        else
            // TODO: replace None with pd.NA when Pandas switch to pd.NA
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, Py_None);
        CHECK(err == 0, "setting item in numpy array failed");
        Py_DECREF(d);
    }

    Py_DECREF(decimal_constructor);
    Py_DECREF(dec_mod);

    PyGILState_Release(gilstate);
    return ret;
#undef CHECK
}

}  // extern "C"
