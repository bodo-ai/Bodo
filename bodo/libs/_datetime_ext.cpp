// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>

#include "_bodo_common.h"
#include "_datetime_ext.h"
#include "arrow/util/bit_util.h"

extern "C" {

/**
 * @brief Computes the python `ret, d = divmod(d, unit)`.
 * copied from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L509
 * @param d[in,out] dt64 value
 * @param unit division unit
 * @return npy_int64 division results
 */
npy_int64 extract_unit(npy_datetime* d, npy_datetime unit) {
    assert(unit > 0);
    npy_int64 div = *d / unit;
    npy_int64 mod = *d % unit;
    if (mod < 0) {
        mod += unit;
        div -= 1;
    }
    assert(mod >= 0);
    *d = mod;
    return div;
}

/**
 * @brief extracts year and days from dt64 value, and updates to the remaining
 * dt64 from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L603
 * @param dt[in,out] dt64
 * @param year[out] extracted year
 * @param days[out] extracted days
 */
static void extract_year_days(npy_datetime* dt, int64_t* year, int64_t* days) {
    //
    npy_int64 perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;
    *days = extract_unit(dt, perday);  // NOTE: dt is updated here as well
    *year = days_to_yearsdays(days);
}

/**
 * @brief Get extracts month and day from days offset within a year
 * from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L230
 * @param year[in]
 * @param days[in]
 * @param month[out]
 * @param day[out]
 */
static void get_month_day(npy_int64 year, npy_int64 days, npy_int64* month,
                          npy_int64* day) {
    const int* month_lengths;
    int i;

    month_lengths = days_per_month_table[is_leapyear(year)];

    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            *month = i + 1;
            *day = days + 1;
            return;
        } else {
            days -= month_lengths[i];
        }
    }
}

// copeid from Pandas, but input is changed from npy_datetime to year/month/day
// fields
// https://github.com/pandas-dev/pandas/blob/b8043724c48890e86fda0265ad5b6ac3d31f1940/pandas/_libs/tslibs/src/datetime/np_datetime.c#L106
/*
 * Calculates the days offset from the 1970 epoch.
 */
npy_int64 get_datetimestruct_days(int64_t dt_year, int dt_month, int dt_day) {
    int i, month;
    npy_int64 year, days = 0;
    const int* month_lengths;

    year = dt_year - 1970;
    days = year * 365;

    /* Adjust for leap years */
    if (days >= 0) {
        /*
         * 1968 is the closest leap year before 1970.
         * Exclude the current year, so add 1.
         */
        year += 1;
        /* Add one day for each 4 years */
        days += year / 4;
        /* 1900 is the closest previous year divisible by 100 */
        year += 68;
        /* Subtract one day for each 100 years */
        days -= year / 100;
        /* 1600 is the closest previous year divisible by 400 */
        year += 300;
        /* Add one day for each 400 years */
        days += year / 400;
    } else {
        /*
         * 1972 is the closest later year after 1970.
         * Include the current year, so subtract 2.
         */
        year -= 2;
        /* Subtract one day for each 4 years */
        days += year / 4;
        /* 2000 is the closest later year divisible by 100 */
        year -= 28;
        /* Add one day for each 100 years */
        days -= year / 100;
        /* 2000 is also the closest later year divisible by 400 */
        /* Subtract one day for each 400 years */
        days += year / 400;
    }

    month_lengths = days_per_month_table[is_leapyear(dt_year)];
    month = dt_month - 1;

    /* Add the months */
    for (i = 0; i < month; ++i) {
        days += month_lengths[i];
    }

    /* Add the days */
    days += dt_day - 1;

    return days;
}

// copeid from Pandas, but input is changed from npy_datetime to
// year/month/day/... fields only the ns frequency is used
// https://github.com/pandas-dev/pandas/blob/b8043724c48890e86fda0265ad5b6ac3d31f1940/pandas/_libs/tslibs/src/datetime/np_datetime.c#L405
/*
 * Converts a datetime from a datetimestruct to a dt64 value
 */
npy_datetime npy_datetimestruct_to_datetime(int64_t year, int month, int day,
                                            int hour, int min, int sec,
                                            int us) {
    int ps = 0;
    npy_int64 days = get_datetimestruct_days(year, month, day);
    return ((((days * 24 + hour) * 60 + min) * 60 + sec) * 1000000 + us) *
               1000 +
           ps / 1000;
}

/**
 * @brief Box native datetime_date_array data to Numpy object array of
 * datetime.date items
 * @return Numpy object array of datetime.date
 * @param[in] n number of values
 * @param[in] data pointer to 64-bit values
 * @param[in] null_bitmap bitvector representing nulls (Arrow format)
 */
void* box_datetime_date_array(int64_t n, const int64_t* data,
                              const uint8_t* null_bitmap) {
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

    // get datetime.date constructor
    PyObject* datetime = PyImport_ImportModule("datetime");
    CHECK(datetime, "importing datetime module failed");
    PyObject* datetime_date_constructor =
        PyObject_GetAttrString(datetime, "date");
    CHECK(datetime_date_constructor, "getting datetime.date failed");

    for (int64_t i = 0; i < n; ++i) {
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (!is_na(null_bitmap, i)) {
            int64_t val = data[i];
            int64_t year = val >> 32;
            int64_t month = (val >> 16) & 0xFFFF;
            int64_t day = val & 0xFFFF;
            PyObject* d = PyObject_CallFunction(datetime_date_constructor,
                                                "LLL", year, month, day);
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, d);
            Py_DECREF(d);
        } else {
            // TODO: replace None with pd.NA when Pandas switch to pd.NA
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, Py_None);
        }
        CHECK(err == 0, "setting item in numpy array failed");
    }

    Py_DECREF(datetime_date_constructor);
    Py_DECREF(datetime);

    PyGILState_Release(gilstate);

    return ret;
#undef CHECK
}

/**
 * @brief unbox ndarray of datetime.date objects into native datetime.date array
 *
 * @param obj ndarray object of datetime.date objects
 * @param n number of values
 * @param data pointer to 64-bit data buffer
 * @param null_bitmap pointer to null_bitmap buffer
 */
void unbox_datetime_date_array(PyObject* obj, int64_t n, int64_t* data,
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
    // Pandas usually stores NaT for date arrays
    PyObject* C_NAT = PyObject_GetAttrString(pd_mod, "NaT");
    CHECK(C_NAT, "getting pd.NaT failed");

    arrow::Status status;

    for (int64_t i = 0; i < n; ++i) {
        PyObject* s = PySequence_GetItem(obj, i);
        CHECK(s, "getting element failed");
        // Pandas stores NA as either None, nan, or pd.NA
        bool value_bitmap;
        int64_t value_data;
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA || s == C_NAT) {
            value_bitmap = false;
            // Set na data to a legal value for array getitem.
            value_data = (1970L << 32) + (1L << 16) + 1L;
        } else {
            PyObject* year_obj = PyObject_GetAttrString(s, "year");
            PyObject* month_obj = PyObject_GetAttrString(s, "month");
            PyObject* day_obj = PyObject_GetAttrString(s, "day");

            int64_t year = PyLong_AsLongLong(year_obj);
            int64_t month = PyLong_AsLongLong(month_obj);
            int64_t day = PyLong_AsLongLong(day_obj);
            value_data = (year << 32) + (month << 16) + day;
            Py_DECREF(year_obj);
            Py_DECREF(month_obj);
            Py_DECREF(day_obj);
            if (year == -1 && month == -1 && day == -1)
                value_bitmap = false;
            else
                value_bitmap = true;
        }
        data[i] = value_data;
        if (value_bitmap) {
            ::arrow::bit_util::SetBit(null_bitmap, i);
        } else {
            ::arrow::bit_util::ClearBit(null_bitmap, i);
        }
        Py_DECREF(s);
    }

    Py_DECREF(C_NA);
    Py_DECREF(C_NAT);
    Py_DECREF(pd_mod);

    PyGILState_Release(gilstate);

    return;
#undef CHECK
}

/**
 * @brief Box native datetime_date_array data to Numpy object array of
 * datetime.timedelta items
 * @return Numpy object array of datetime.timedelta
 * @param[in] n number of values
 * @param[in] days_data pointer to 64-bit values for days
 * @param[in] seconds_data pointer to 64-bit values for seconds
 * @param[in] microseconds_data pointer to 64-bit values for microseconds
 * @param[in] null_bitmap bitvector representing nulls (Arrow format)
 */
void* box_datetime_timedelta_array(int64_t n, const int64_t* days_data,
                              const int64_t* seconds_data,
                              const int64_t* microseconds_data,
                              const uint8_t* null_bitmap) {
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

    // get datetime.timedelta constructor
    PyObject* datetime = PyImport_ImportModule("datetime");
    CHECK(datetime, "importing datetime module failed");
    PyObject* datetime_timedelta_constructor =
        PyObject_GetAttrString(datetime, "timedelta");
    CHECK(datetime_timedelta_constructor, "getting datetime.timedelta failed");

    for (int64_t i = 0; i < n; ++i) {
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (!is_na(null_bitmap, i)) {
            // To avoid needing to supply kwargs we note that the first 3 args
            // are the required fields
            int64_t days = days_data[i];
            int64_t seconds = seconds_data[i];
            int64_t microseconds = microseconds_data[i];
            PyObject* d = PyObject_CallFunction(datetime_timedelta_constructor,
                                        "LLL", days, seconds, microseconds);
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, d);
            Py_DECREF(d);
        } else {
            // TODO: replace None with pd.NA when Pandas switch to pd.NA
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, Py_None);
        }
        CHECK(err == 0, "setting item in numpy array failed");
    }

    Py_DECREF(datetime_timedelta_constructor);
    Py_DECREF(datetime);

    PyGILState_Release(gilstate);

    return ret;
#undef CHECK
}

/**
 * @brief unbox ndarray of datetime.timedelta objects into native datetime.timedelta array
 *
 * @param obj ndarray object of datetime.timedelta objects
 * @param n number of values
 * @param days_data pointer to 64-bit values for days
 * @param seconds_data pointer to 64-bit values for seconds
 * @param microseconds_data pointer to 64-bit values for microseconds
 * @param null_bitmap pointer to null_bitmap buffer
 */
void unbox_datetime_timedelta_array(PyObject* obj, int64_t n,
        int64_t* days_data, int64_t* seconds_data,
        int64_t* microseconds_data, uint8_t* null_bitmap) {
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
    CHECK(n >= 0 && days_data && seconds_data
            && microseconds_data && null_bitmap,
            "output arguments must not be NULL");

    // get pd.NA object to check for new NA kind
    // simple equality check is enough since the object is a singleton
    // example:
    // https://github.com/pandas-dev/pandas/blob/fcadff30da9feb3edb3acda662ff6143b7cb2d9f/pandas/_libs/missing.pyx#L57
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");
    // Pandas usually stores NaT for date arrays
    PyObject* C_NAT = PyObject_GetAttrString(pd_mod, "NaT");
    CHECK(C_NAT, "getting pd.NaT failed");

    arrow::Status status;

    for (int64_t i = 0; i < n; ++i) {
        PyObject* s = PySequence_GetItem(obj, i);
        CHECK(s, "getting element failed");
        // Pandas stores NA as either None, nan, or pd.NA
        bool value_bitmap;
        int64_t days_val;
        int64_t seconds_val;
        int64_t microseconds_val;
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA || s == C_NAT) {
            value_bitmap = false;
            days_val = 0;
            seconds_val = 0;
            microseconds_val = 0;
        } else {
            PyObject* days_obj = PyObject_GetAttrString(s, "days");
            PyObject* seconds_obj = PyObject_GetAttrString(s, "seconds");
            PyObject* microseconds_obj = PyObject_GetAttrString(s,
                    "microseconds");

            days_val = PyLong_AsLongLong(days_obj);
            seconds_val = PyLong_AsLongLong(seconds_obj);
            microseconds_val = PyLong_AsLongLong(microseconds_obj);
            Py_DECREF(days_obj);
            Py_DECREF(seconds_obj);
            Py_DECREF(microseconds_obj);
            if (days_val == -1 && seconds_val == -1
                    && microseconds_val == -1)
                value_bitmap = false;
            else
                value_bitmap = true;
        }
        days_data[i] = days_val;
        seconds_data[i] = seconds_val;
        microseconds_data[i] = microseconds_val;
        if (value_bitmap) {
            ::arrow::bit_util::SetBit(null_bitmap, i);
        } else {
            ::arrow::bit_util::ClearBit(null_bitmap, i);
        }
        Py_DECREF(s);
    }

    Py_DECREF(C_NA);
    Py_DECREF(C_NAT);
    Py_DECREF(pd_mod);

    PyGILState_Release(gilstate);

    return;
#undef CHECK
}


/**
 * @brief box a date_offset object. Null fields should be omitted. 0 is included if has_kws.
 * @return DateOffset PyObject
 * @param n n value for DateOffset
 * @param normalize normalize value for Dateoffset
 * @param fields_arr Array of fields that may be initialized
 * @param has_kws Bool for if any non-nano fields are initialized. This impacts behavior
 */
PyObject* box_date_offset(int64_t n, bool normalize, int64_t fields_arr[18], bool has_kws) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return nullptr;                \
    }
    auto gilstate = PyGILState_Ensure();

    // set of fields names for the array. They are divided into 2 groups based upon
    // if their default value can be used to determine if the value was/wasn't included.
    // Those ending with s cannot be distinguished so they will always be added if has_kws
    // This is fine because 0 is the same behavior as not being included, so long as 1 field
    // is included.
    const char* fields[2][9] = {{"years", "months", "weeks", "days", "hours", "minutes", "seconds", "microseconds", "nanoseconds"},
    {"year", "month", "day", "weekday", "hour", "minute", "second", "microsecond", "nanosecond"}};

    int64_t default_values[2] = {0, -1};
    // Vector of pyobjs for tracking decref
    std::vector<PyObject *> pyobjs;

    // Create a kwargs dict
    PyObject* kwargs = PyDict_New();
    CHECK(kwargs, "Creating kwargs dict failed");
    pyobjs.push_back(kwargs);
    if (has_kws) {
        // If has_kws all fields that are non-null or cannot be distinguished
        // need to be added to the Python dictionary.
        for (int64_t i = 0; i < 2; ++i) {
            int64_t default_value = default_values[i];
            for (int64_t j = 0; j < 9; ++j) {
                int64_t field_value = fields_arr[i * 9 + j];
                const char* field_name = fields[i][j];
                if (field_value != default_value) {
                    PyObject* field_obj = Py_BuildValue("s", field_name);
                    CHECK(field_obj, "Creating name obj for kwargs failed");
                    PyObject* value_obj = Py_BuildValue("l", field_value);
                    CHECK(value_obj, "Creating value obj for kwargs failed");
                    CHECK(PyDict_SetItem(kwargs, field_obj, value_obj) != -1, "Dict setitem failed");
                    pyobjs.push_back(field_obj);
                    pyobjs.push_back(value_obj);
                }
            }
        }
    }
    PyObject* n_obj = Py_BuildValue("l", n);
    CHECK(n_obj, "Creating n object failed");
    PyObject* normalize_obj = PyBool_FromLong(normalize);
    CHECK(normalize_obj, "Creating normalize object failed");
    PyObject *args = PyTuple_Pack(2, n_obj, normalize_obj);
    CHECK(args, "Creating *args failed");
    pyobjs.push_back(n_obj);
    pyobjs.push_back(normalize_obj);
    pyobjs.push_back(args);
    PyObject* module = PyImport_ImportModule("pandas.tseries.offsets");
    CHECK(module, "importing pandas.tseries.offsets module failed");
    PyObject *function = PyObject_GetAttrString(module, "DateOffset");
    pyobjs.push_back(module);
    pyobjs.push_back(function);
    PyObject* date_offset_obj = PyObject_Call(function, args, kwargs);
    CHECK(date_offset_obj, "DateOffset constructor failed");
    // Make sure to update reference counting
    for (auto& obj: pyobjs) {
        Py_DECREF(obj);
    }
    PyGILState_Release(gilstate);

    return date_offset_obj;
#undef CHECK
}


/**
 * @brief unbox a date_offset object. Missing fields should get their null value.
 * @return boolean for if the obj has any kws
 * @param obj pd.tseries.offsets.DateOffset object
 * @param fields_arr Array of fields that must be initialized
 */
bool unbox_date_offset(PyObject* obj, int64_t fields_arr[18]) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return false;                \
    }
    auto gilstate = PyGILState_Ensure();

    // set of fields, split by default value if missing
    const char* fields[2][9] = {{"years", "months", "weeks", "days", "hours", "minutes", "seconds", "microseconds", "nanoseconds"},
    {"year", "month", "day", "weekday", "hour", "minute", "second", "microsecond", "nanosecond"}};

    int64_t default_values[2] = {0, -1};
    bool has_kws = false;

    for (int64_t i = 0; i < 2; ++i) {
        int64_t default_value = default_values[i];
        for (int64_t j = 0; j < 9; ++j) {
            const char* field_name = fields[i][j];
            int64_t field_value = default_value;

            if (PyObject_HasAttrString(obj, field_name)) {
                has_kws = true;
                PyObject* field_obj = PyObject_GetAttrString(obj, field_name);
                CHECK(field_obj, "Selecting field from DateOffset Obj failed")
                field_value = PyLong_AsLongLong(field_obj);
                Py_DECREF(field_obj);
            }
            fields_arr[(i * 9) + j] = field_value;
        }
    }
    PyGILState_Release(gilstate);

    return has_kws;
#undef CHECK
}



PyMODINIT_FUNC PyInit_hdatetime_ext(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hdatetime_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // init numpy
    import_array();

    // initalize memory alloc/tracking system in _meminfo.h
    bodo_common_init();

    // These are all C functions, so they don't throw any exceptions.
    // We might still need to add better error handling in the future.

    PyObject_SetAttrString(m, "get_isocalendar",
                           PyLong_FromVoidPtr((void*)(&get_isocalendar)));
    PyObject_SetAttrString(m, "extract_year_days",
                           PyLong_FromVoidPtr((void*)(&extract_year_days)));

    PyObject_SetAttrString(m, "get_month_day",
                           PyLong_FromVoidPtr((void*)(&get_month_day)));

    PyObject_SetAttrString(
        m, "npy_datetimestruct_to_datetime",
        PyLong_FromVoidPtr((void*)(&npy_datetimestruct_to_datetime)));

    PyObject_SetAttrString(
        m, "box_datetime_date_array",
        PyLong_FromVoidPtr((void*)(&box_datetime_date_array)));

    PyObject_SetAttrString(
        m, "unbox_datetime_date_array",
        PyLong_FromVoidPtr((void*)(&unbox_datetime_date_array)));

    PyObject_SetAttrString(
        m, "box_datetime_timedelta_array",
        PyLong_FromVoidPtr((void*)(&box_datetime_timedelta_array)));

    PyObject_SetAttrString(
        m, "unbox_datetime_timedelta_array",
        PyLong_FromVoidPtr((void*)(&unbox_datetime_timedelta_array)));

    PyObject_SetAttrString(
        m, "unbox_date_offset",
        PyLong_FromVoidPtr((void*)(&unbox_date_offset)));

    PyObject_SetAttrString(
        m, "box_date_offset",
        PyLong_FromVoidPtr((void*)(&box_date_offset)));

    return m;
}

}  // extern "C"
