// Copyright (C) 2020 Bodo Inc. All rights reserved.
// C/C++ code for DecimalArray handling
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>
#include "_bodo_common.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/decimal.h"

// using scale 18 when converting from Python Decimal objects (same as Spark)
#define PY_DECIMAL_SCALE 18

std::string decimal_to_std_string(arrow::Decimal128 const& arrow_decimal, int const& scale) {
    std::string str = arrow_decimal.ToString(scale);
    // str may be of the form 0.45000000000 or 4.000000000
    size_t last_char = str.length();
    while(true) {
        if (str[last_char-1] != '0')
            break;
        last_char--;
    }
    // position reduce str to 0.45  or 4.
    if (str[last_char-1] == '.')
        last_char--;
    // position reduce str to 0.45 or 4
    str = str.substr(0,last_char);
    if (str == "0.E-18")
        return "0";
    return str;
}

std::string decimal_value_cpp_to_std_string(decimal_value_cpp const & val, int const& scale) {
    arrow::Decimal128 arrow_decimal(val.high, val.low);
    return decimal_to_std_string(arrow_decimal, scale);
}

bool operator<(decimal_value_cpp const& left, decimal_value_cpp const& right)
{
    arrow::Decimal128 arrow_decimal_left(left.high, left.low);
    arrow::Decimal128 arrow_decimal_right(right.high, right.low);
    return arrow_decimal_left < arrow_decimal_right;
}

double decimal_to_double(decimal_value_cpp const& val)
{
    int scale = 18;
    std::string str = decimal_value_cpp_to_std_string(val, scale);
    return std::stod(str);
}

extern "C" {

void* box_decimal_array(int64_t n, const uint8_t* data,
                        const uint8_t* null_bitmap, int scale);
void unbox_decimal_array(PyObject* obj, int64_t n, uint8_t* data,
                         uint8_t* null_bitmap);
void unbox_decimal(PyObject* obj, uint8_t* data);

#pragma pack(1)
struct decimal_value {
    int64_t low;
    int64_t high;
};

void decimal_to_str(decimal_value val, NRT_MemInfo** meminfo_ptr,
                    int64_t* len_ptr, int scale);

PyMODINIT_FUNC PyInit_decimal_ext(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "decimal_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // init numpy
    import_array();

    // decimal_value should be exactly 128 bits to match Python
    if (sizeof(decimal_value) != 16)
        std::cerr << "invalid decimal struct size" << std::endl;

    PyObject_SetAttrString(m, "box_decimal_array",
                           PyLong_FromVoidPtr((void*)(&box_decimal_array)));
    PyObject_SetAttrString(m, "unbox_decimal_array",
                           PyLong_FromVoidPtr((void*)(&unbox_decimal_array)));
    PyObject_SetAttrString(m, "unbox_decimal",
                           PyLong_FromVoidPtr((void*)(&unbox_decimal)));
    PyObject_SetAttrString(m, "decimal_to_str",
                           PyLong_FromVoidPtr((void*)(&decimal_to_str)));
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
        std::string str = decimal_to_std_string(arrow_decimal, scale);
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


/**
 * @brief unbox a single Decimal object into a native Decimal128Type
 *
 * @param obj single decimal object
 * @param data pointer to 128-bit data
 */
void unbox_decimal(PyObject* obj, uint8_t* data_ptr)
{
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return;                        \
    }
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK(res.status().ok(), msg)             \
    lhs = std::move(res).ValueOrDie();

    auto gilstate = PyGILState_Ensure();

    PyObject* decimal_mod = PyImport_ImportModule("decimal");
    CHECK(decimal_mod, "importing decimal module failed");
    PyObject* str_obj = PyObject_Str(obj);
    CHECK(str_obj, "str(decimal) failed");
    // extracting the 128 integer
    arrow::Decimal128 d128, d128_18;
    int32_t precision;
    int32_t scale;
    arrow::Status status = arrow::Decimal128::FromString(
        (const char*)PyUnicode_DATA(str_obj), &d128, &precision,
        &scale);
    CHECK(status.ok(), "decimal rescale faild");
    // rescale decimal to 18 (default scale converting from Python)
    CHECK_ARROW_AND_ASSIGN(d128.Rescale(scale, PY_DECIMAL_SCALE),
                           "decimal rescale error", d128_18);
    d128_18.ToBytes(data_ptr);
    Py_DECREF(str_obj);
    Py_DECREF(decimal_mod);

    PyGILState_Release(gilstate);
}




/**
 * @brief unbox ndarray of Decimal objects into native DecimalArray
 *
 * @param obj ndarray object of Decimal objects
 * @param n number of values
 * @param data pointer to 128-bit data buffer
 * @param null_bitmap pointer to null_bitmap buffer
 */
void unbox_decimal_array(PyObject* obj, int64_t n, uint8_t* data,
                         uint8_t* null_bitmap) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return;                        \
    }
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK(res.status().ok(), msg)             \
    lhs = std::move(res).ValueOrDie();

    auto gilstate = PyGILState_Ensure();

    CHECK(PySequence_Check(obj), "expecting a PySequence");
    CHECK(n >= 0 && data && null_bitmap, "output arguments must not be NULL");

    // get pd.NA object to check for new NA kind
    // simple equality check is enough since the object is a singleton
    // example:
    // https://github.com/pandas-dev/pandas/blob/fcadff30da9feb3edb3acda662ff6143b7cb2d9f/pandas/_libs/missing.pyx#L57
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    arrow::Status status;

    for (int64_t i = 0; i < n; ++i) {
        PyObject* s = PySequence_GetItem(obj, i);
        CHECK(s, "getting element failed");
        // Pandas stores NA as either None, nan, or pd.NA
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA) {
            // null bit
            ::arrow::BitUtil::ClearBit(null_bitmap, i);
            memset(data + i * BYTES_PER_DECIMAL, 0, BYTES_PER_DECIMAL);
        } else {
            // set not null
            ::arrow::BitUtil::SetBit(null_bitmap, i);
            // construct Decimal128 from str(decimal)
            PyObject* s_str_obj = PyObject_Str(s);
            CHECK(s_str_obj, "str(decimal) failed");
            arrow::Decimal128 d128, d128_18;
            int32_t precision;
            int32_t scale;
            status = arrow::Decimal128::FromString(
                (const char*)PyUnicode_DATA(s_str_obj), &d128, &precision,
                &scale);
            CHECK(status.ok(), "decimal rescale faild");
            // rescale decimal to 18 (default scale converting from Python)
            CHECK_ARROW_AND_ASSIGN(d128.Rescale(scale, PY_DECIMAL_SCALE),
                                   "decimal rescale error", d128_18);
            // write to data array
            uint8_t* data_ptr = (data + i * BYTES_PER_DECIMAL);
            d128_18.ToBytes(data_ptr);
            Py_DECREF(s_str_obj);
        }
        Py_DECREF(s);
    }

    Py_DECREF(C_NA);
    Py_DECREF(pd_mod);

    PyGILState_Release(gilstate);

    return;
#undef CHECK
}

/**
 * @brief convert decimal128 value to string and create a memptr pointer with
 * the data
 *
 * @param val decimal128 input value
 * @param meminfo_ptr output memptr pointer to set
 * @param len_ptr output length to set
 * @param scale scale parameter of decimal128
 */
void decimal_to_str(decimal_value val, NRT_MemInfo** meminfo_ptr,
                    int64_t* len_ptr, int scale) {
    // Creating the arrow_decimal value
    arrow::Decimal128 arrow_decimal(val.high, val.low);
    // Getting the string
    std::string str = decimal_to_std_string(arrow_decimal, scale);
    // Now doing the boxing.
    int64_t l = (int64_t)str.length();
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe(l + 1);
    memcpy(meminfo->data, str.c_str(), l);
    ((char*)meminfo->data)[l] = 0;
    *len_ptr = l;
    *meminfo_ptr = meminfo;
}

}  // extern "C"
