// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>

#include "_bodo_common.h"
#include "arrow/util/bit_util.h"

extern "C" {

static int is_leapyear(npy_int64 year) {
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 || (year % 400) == 0);
}

static const int days_per_month_table[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};

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
 * @brief Modifies '*days_' to be the day offset within the year, and returns
 * the year. copied from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L166
 * @param days_[in,out] input: total days output: day offset within the year
 * @return npy_int64 output year
 */
static npy_int64 days_to_yearsdays(npy_int64* days_) {
    const npy_int64 days_per_400years = (400 * 365 + 100 - 4 + 1);
    /* Adjust so it's relative to the year 2000 (divisible by 400) */
    npy_int64 days = (*days_) - (365 * 30 + 7);
    npy_int64 year;

    /* Break down the 400 year cycle to get the year and day within the year */
    if (days >= 0) {
        year = 400 * (days / days_per_400years);
        days = days % days_per_400years;
    } else {
        year = 400 * ((days - (days_per_400years - 1)) / days_per_400years);
        days = days % days_per_400years;
        if (days < 0) {
            days += days_per_400years;
        }
    }

    /* Work out the year/day within the 400 year cycle */
    if (days >= 366) {
        year += 100 * ((days - 1) / (100 * 365 + 25 - 1));
        days = (days - 1) % (100 * 365 + 25 - 1);
        if (days >= 365) {
            year += 4 * ((days + 1) / (4 * 365 + 1));
            days = (days + 1) % (4 * 365 + 1);
            if (days >= 366) {
                year += (days - 1) / 365;
                days = (days - 1) % 365;
            }
        }
    }

    *days_ = days;
    return year + 2000;
}

/**
 * @brief extracts year and days from dt64 value, and updates to the remaining
 * dt64 from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L603
 * @param dt[in,out] dt64
 * @param year[out] extracted year
 * @param days[out] extracted days
 */
static void extract_year_days(npy_datetime* dt, npy_int64* year,
                              npy_int64* days) {
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
        } else
            // TODO: replace None with pd.NA when Pandas switch to pd.NA
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, Py_None);
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

    // get pd.isna object to call in the loop to check for NAs
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* isna_call_obj = PyObject_GetAttrString(pd_mod, "isna");
    CHECK(isna_call_obj, "getting pd.isna failed");

    arrow::Status status;

    for (int64_t i = 0; i < n; ++i) {
        PyObject* s = PySequence_GetItem(obj, i);
        CHECK(s, "getting element failed");
        // Pandas stores NA as either None, nan, or pd.NA
        PyObject* isna_obj =
            PyObject_CallFunctionObjArgs(isna_call_obj, s, NULL);
        if (PyObject_IsTrue(isna_obj)) {
            // null bit
            ::arrow::BitUtil::ClearBit(null_bitmap, i);
            data[i] = 0;
        } else {
            // set not null
            ::arrow::BitUtil::SetBit(null_bitmap, i);
            PyObject* year_obj = PyObject_GetAttrString(s, "year");
            PyObject* month_obj = PyObject_GetAttrString(s, "month");
            PyObject* day_obj = PyObject_GetAttrString(s, "day");

            int64_t year = PyLong_AsLongLong(year_obj);
            int64_t month = PyLong_AsLongLong(month_obj);
            int64_t day = PyLong_AsLongLong(day_obj);
            data[i] = (year << 32) + (month << 16) + day;

            Py_DECREF(year_obj);
            Py_DECREF(month_obj);
            Py_DECREF(day_obj);
        }
        Py_DECREF(s);
        Py_DECREF(isna_obj);
    }

    Py_DECREF(isna_call_obj);
    Py_DECREF(pd_mod);

    PyGILState_Release(gilstate);

    return;
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

    return m;
}

}  // extern "C"
