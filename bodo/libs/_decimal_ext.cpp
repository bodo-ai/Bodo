// Copyright (C) 2020 Bodo Inc. All rights reserved.
// C/C++ code for DecimalArray handling
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>

#include <fmt/format.h>
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/decimal.h"

// using scale 18 when converting from Python Decimal objects (same as Spark)
#define PY_DECIMAL_SCALE 18

#pragma pack(1)
struct decimal_value {
    int64_t low;
    int64_t high;
};

/**
 * @brief enum for designating comparison operators passed from Python.
 * Most match with definition in decimal_arr_ext.py.
 */
enum CmpOp {
    LT = 0,
    LE = 1,
    EQ = 2,
    NE = 3,
    GT = 4,
    GE = 5,
};

std::string decimal_to_std_string(arrow::Decimal128 const& arrow_decimal,
                                  int const& scale) {
    // TODO(srilman): I think Arrow Decimal128::ToString() had trailing zeros in
    // past, but when I tested, it didn't. Maybe we can get rid of this
    // function entirely
    std::string str = arrow_decimal.ToString(scale);
    if (str.find('.') == std::string::npos) {
        return str;
    }

    // str may be of the form 0.45000000000 or 4.000000000
    size_t last_char = str.length();
    while (true) {
        if (str[last_char - 1] != '0')
            break;
        last_char--;
    }
    // position reduce str to 0.45  or 4.
    if (str[last_char - 1] == '.')
        last_char--;

    // Slice String to New Range
    str = str.substr(0, last_char);
    if (str == "0.E-18")
        return "0";
    return str;
}

std::string int128_decimal_to_std_string(__int128 const& val,
                                         int const& scale) {
    arrow::Decimal128 arrow_decimal((int64_t)(val >> 64), (int64_t)(val));
    return decimal_to_std_string(arrow_decimal, scale);
}

double decimal_to_double(__int128 const& val, uint8_t scale) {
    // TODO: Zero-copy (cast __int128 to int64[2] for Decimal128 constructor)
    // Can't figure out how to do this in C++
    arrow::Decimal128 dec((int64_t)(val >> 64), (int64_t)(val));
    return dec.ToDouble(scale);
}

double decimal_to_double_py_entry(decimal_value val, uint8_t scale) {
    auto high = static_cast<uint64_t>(val.high);
    auto low = static_cast<uint64_t>(val.low);
    return decimal_to_double((static_cast<__int128>(high) << 64) | low, scale);
}

decimal_value int64_to_decimal(int64_t value) {
    arrow::Decimal128 dec(value);
    auto low_bits = dec.low_bits();
    auto high_bits = dec.high_bits();
    return {static_cast<int64_t>(low_bits), high_bits};
}

void* box_decimal_array(int64_t n, const uint8_t* data,
                        const uint8_t* null_bitmap, int scale);
void unbox_decimal_array(PyObject* obj, int64_t n, uint8_t* data,
                         uint8_t* null_bitmap);
void unbox_decimal(PyObject* obj, uint8_t* data);

void decimal_to_str(decimal_value val, NRT_MemInfo** meminfo_ptr,
                    int64_t* len_ptr, int scale);

arrow::Decimal128 str_to_decimal(const std::string& str_val);

/**
 * @brief Get the Arrow function name for comparison operator
 * See https://arrow.apache.org/docs/cpp/compute.html#comparisons
 *
 * @param op_enum operator enum value
 * @return std::string Arrow function name
 */
std::string get_arrow_function_name(int32_t op_enum) {
    switch (op_enum) {
        case CmpOp::LT:
            return "less";
        case CmpOp::LE:
            return "less_equal";
        case CmpOp::EQ:
            return "equal";
        case CmpOp::NE:
            return "not_equal";
        case CmpOp::GT:
            return "greater";
        case CmpOp::GE:
            return "greater_equal";
        default:
            throw std::runtime_error(
                fmt::format("get_arrow_function_name: invalid op {}", op_enum));
    }
}

/**
 * @brief Call Arrow compute function for comparison operator
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0 first argument for comparison
 * @param arg1 second argument for comparison
 * @param is_scalar_arg0 first argument is scalar (passed as a one element
 * array)
 * @param is_scalar_arg1 second argument is scalar (passed as a one element
 * array)
 * @return array_info* output of comparison which is a boolean array
 */
array_info* arrow_compute_cmp_py_entry(int32_t op_enum, array_info* arg0,
                                       array_info* arg1, bool is_scalar_arg0,
                                       bool is_scalar_arg1) {
    try {
        // convert inputs to Arrow and handle scalars
        std::shared_ptr<arrow::Array> arr0 =
            to_arrow(std::shared_ptr<array_info>(arg0));
        std::shared_ptr<arrow::Array> arr1 =
            to_arrow(std::shared_ptr<array_info>(arg1));

        arrow::Datum arg0_datum(arr0);
        arrow::Datum arg1_datum(arr1);

        // convert scalar arguments passed as one element arrays back to scalar
        if (is_scalar_arg0) {
            arrow::Result<std::shared_ptr<arrow::Scalar>> getitem_res =
                arr0->GetScalar(0);
            if (!getitem_res.ok()) [[unlikely]] {
                throw std::runtime_error(
                    fmt::format("arrow_compute_cmp_py_entry: Error in Arrow "
                                "getitem for first argument: {}",
                                getitem_res.status().message()));
            }
            arg0_datum = getitem_res.ValueOrDie();
        }

        if (is_scalar_arg1) {
            arrow::Result<std::shared_ptr<arrow::Scalar>> getitem_res =
                arr1->GetScalar(0);
            if (!getitem_res.ok()) [[unlikely]] {
                throw std::runtime_error(
                    fmt::format("arrow_compute_cmp_py_entry: Error in Arrow "
                                "getitem for second argument: {}",
                                getitem_res.status().message()));
            }
            arg1_datum = getitem_res.ValueOrDie();
        }

        // call Arrow compute function
        arrow::Result<arrow::Datum> cmp_res = arrow::compute::CallFunction(
            get_arrow_function_name(op_enum), {arg0_datum, arg1_datum});
        if (!cmp_res.ok()) [[unlikely]] {
            throw std::runtime_error(fmt::format(
                "arrow_compute_cmp_py_entry: Error in Arrow compute: {}",
                cmp_res.status().message()));
        }
        std::shared_ptr<arrow::Array> out_arrow_arr =
            cmp_res.ValueOrDie().make_array();

        // convert Arrow output to Bodo array
        std::shared_ptr<array_info> out_arr =
            arrow_array_to_bodo(out_arrow_arr);
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief call Arrow to compare scalars
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0_scalar first argument of comparison
 * @param arg1_scalar second argument of comparison
 * @return bool output of comparison
 */
template <class T0, class T1>
inline bool arrow_compute_cmp_scalar(int32_t op_enum, T0 arg0_scalar,
                                     T1 arg1_scalar) {
    // call Arrow compute function
    arrow::Result<arrow::Datum> cmp_res = arrow::compute::CallFunction(
        get_arrow_function_name(op_enum), {arg0_scalar, arg1_scalar});
    if (!cmp_res.ok()) [[unlikely]] {
        throw std::runtime_error(
            fmt::format("arrow_compute_cmp_scalar: Error in "
                        "Arrow compute: {}",
                        cmp_res.status().message()));
    }

    // return bool output
    std::shared_ptr<arrow::BooleanScalar> out =
        std::static_pointer_cast<arrow::BooleanScalar>(
            cmp_res.ValueOrDie().scalar());
    return out->value;
}

/**
 * @brief compare decimal scalar to integer scalar
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0 first argument of comparison (decimal)
 * @param precision decimal argument's precision
 * @param scale decimal argument's scale
 * @param arg1 second argument (int)
 * @return bool output of comparison
 */
bool arrow_compute_cmp_decimal_int_py_entry(int32_t op_enum, decimal_value arg0,
                                            int32_t precision, int32_t scale,
                                            int64_t arg1) {
    try {
        // convert input to Arrow scalars
        arrow::Decimal128 arrow_decimal(arg0.high, arg0.low);
        arrow::Decimal128Scalar arg0_scalar(
            arrow_decimal, arrow::decimal128(precision, scale));
        arrow::Int64Scalar arg1_scalar(arg1);
        return arrow_compute_cmp_scalar(op_enum, arg0_scalar, arg1_scalar);

    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
    }
}

/**
 * @brief compare decimal scalar to float scalar
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0 first argument of comparison (decimal)
 * @param precision decimal argument's precision
 * @param scale decimal argument's scale
 * @param arg1 second argument (float)
 * @return bool output of comparison
 */
bool arrow_compute_cmp_decimal_float_py_entry(int32_t op_enum,
                                              decimal_value arg0,
                                              int32_t precision, int32_t scale,
                                              double arg1) {
    try {
        // convert input to Arrow scalars
        arrow::Decimal128 arrow_decimal(arg0.high, arg0.low);
        arrow::Decimal128Scalar arg0_scalar(
            arrow_decimal, arrow::decimal128(precision, scale));
        arrow::DoubleScalar arg1_scalar(arg1);
        return arrow_compute_cmp_scalar(op_enum, arg0_scalar, arg1_scalar);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
    }
}

/**
 * @brief compare decimal scalar to decimal scalar
 *
 * @param op_enum enum designating comparison operator to call
 * @param arg0 first argument of comparison (decimal)
 * @param precision0 first argument's precision
 * @param scale0 first argument's scale
 * @param arg1 second argument (decimal)
 * @param precision1 second argument's precision
 * @param scale1 second argument's scale
 * @return bool output of comparison
 */
bool arrow_compute_cmp_decimal_decimal_py_entry(
    int32_t op_enum, decimal_value arg0, int32_t precision0, int32_t scale0,
    int32_t precision1, int32_t scale1, decimal_value arg1) {
    try {
        // convert input to Arrow scalars
        arrow::Decimal128 arrow_decimal0(arg0.high, arg0.low);
        arrow::Decimal128Scalar arg0_scalar(
            arrow_decimal0, arrow::decimal128(precision0, scale0));
        arrow::Decimal128 arrow_decimal1(arg1.high, arg1.low);
        arrow::Decimal128Scalar arg1_scalar(
            arrow_decimal1, arrow::decimal128(precision1, scale1));
        return arrow_compute_cmp_scalar(op_enum, arg0_scalar, arg1_scalar);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return false;
    }
}

PyMODINIT_FUNC PyInit_decimal_ext(void) {
    PyObject* m;
    MOD_DEF(m, "decimal_ext", "No docs", NULL);
    if (m == NULL)
        return NULL;

    // init numpy
    import_array();

    bodo_common_init();

    // decimal_value should be exactly 128 bits to match Python
    if (sizeof(decimal_value) != 16)
        std::cerr << "invalid decimal struct size" << std::endl;

    // These are all C functions, so they don't throw any exceptions.
    // We might still need to add better error handling in the future.
    SetAttrStringFromVoidPtr(m, box_decimal_array);
    SetAttrStringFromVoidPtr(m, unbox_decimal_array);
    SetAttrStringFromVoidPtr(m, unbox_decimal);

    SetAttrStringFromVoidPtr(m, decimal_to_str);
    SetAttrStringFromVoidPtr(m, str_to_decimal);
    SetAttrStringFromVoidPtr(m, decimal_to_double_py_entry);
    SetAttrStringFromVoidPtr(m, int64_to_decimal);

    SetAttrStringFromVoidPtr(m, arrow_compute_cmp_py_entry);
    SetAttrStringFromVoidPtr(m, arrow_compute_cmp_decimal_int_py_entry);
    SetAttrStringFromVoidPtr(m, arrow_compute_cmp_decimal_float_py_entry);
    SetAttrStringFromVoidPtr(m, arrow_compute_cmp_decimal_decimal_py_entry);

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
#undef CHECK
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
void unbox_decimal(PyObject* obj, uint8_t* data_ptr) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return;                        \
    }
#undef CHECK_ARROW_AND_ASSIGN
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
        (const char*)PyUnicode_DATA(str_obj), &d128, &precision, &scale);
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
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return;                        \
    }
#undef CHECK_ARROW_AND_ASSIGN
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
            ::arrow::bit_util::ClearBit(null_bitmap, i);
            memset(data + i * BYTES_PER_DECIMAL, 0, BYTES_PER_DECIMAL);
        } else {
            // set not null
            ::arrow::bit_util::SetBit(null_bitmap, i);
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
    NRT_MemInfo* meminfo = alloc_meminfo(l + 1);
    memcpy(meminfo->data, str.c_str(), l);
    ((char*)meminfo->data)[l] = 0;
    *len_ptr = l;
    *meminfo_ptr = meminfo;
}

arrow::Decimal128 str_to_decimal(const std::string& str_val) {
    /*
        str_to_decimal ignores a scale and precision value from
        Python code because the user cannot specify them directly. Precision
        currently matches the arrow default value of 38 and scale is set in
        the global variable. If this should ever change this function will need
        to be adjusted.
    */
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return -1;                     \
    }
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK(res.status().ok(), msg)             \
    lhs = std::move(res).ValueOrDie();

    auto gilstate = PyGILState_Ensure();
    // Create the decimal array value from a string, precision, and scale
    arrow::Decimal128 decimal;
    int32_t precision;
    int32_t scale;
    arrow::Status status =
        arrow::Decimal128::FromString(str_val, &decimal, &precision, &scale);
    CHECK(status.ok(), "decimal rescale failed");
    // rescale decimal to 18 (default scale converting from Python)
    CHECK_ARROW_AND_ASSIGN(decimal.Rescale(scale, PY_DECIMAL_SCALE),
                           "decimal rescale error", decimal);
    return decimal;
#undef CHECK
}
