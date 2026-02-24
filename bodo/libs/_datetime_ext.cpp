#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>

#include "_bodo_common.h"
#include "_datetime_ext.h"
#include "_datetime_utils.h"
#include "arrow/util/bit_util.h"

extern "C" {

/** Signature definitions for private helper functions. **/
npy_int64 extract_unit(npy_datetime* d, npy_datetime unit);
int64_t get_day_of_year(int64_t year, int64_t month, int64_t day);
int64_t dayofweek(int64_t y, int64_t m, int64_t d);
void get_isocalendar(int64_t year, int64_t month, int64_t day,
                     int64_t* year_res, int64_t* week_res, int64_t* dow_res);
void extract_year_days(npy_datetime* dt, int64_t* year, int64_t* days);
npy_int64 get_datetimestruct_days(int64_t dt_year, int dt_month, int dt_day);
npy_datetime npy_datetimestruct_to_datetime(int64_t year, int month, int day,
                                            int hour, int min, int sec, int us);

/*
 * Array used to calculate dayofweek
 * Taken from Pandas:
 * https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L20
 */
const int sakamoto_arr[12] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};

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

/*
 * Return the ordinal day-of-year for the given day.
 * Copied from Pandas:
 * https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L215
 */
int64_t get_day_of_year(int64_t year, int64_t month, int64_t day) {
    int64_t mo_off = month_offset[(13 * is_leapyear(year)) + (month - 1)];
    return mo_off + day;
}

/*
 * Find the day of week for the date described by the Y/M/D triple y, m, d
 * using Sakamoto's method, from wikipedia.
 * Copied from Pandas:
 * https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L83
 */
int64_t dayofweek(int64_t y, int64_t m, int64_t d) {
    y -= (m < 3);
    int64_t day = (y + y / 4 - y / 100 + y / 400 + sakamoto_arr[m - 1] + d) % 7;
    // convert to python day
    return (day + 6) % 7;
}

/*
 * Implementation of a general get_isocalendar to be reused
 * by various pandas types. Places return values inside the allocated
 * buffers. Iso calendar returns the year, week number, and day of the
 * week for a given date described by a year month and day.
 */
void get_isocalendar(int64_t year, int64_t month, int64_t day,
                     int64_t* year_res, int64_t* week_res, int64_t* dow_res) {
    /*
     * In the Gregorian calendar, week 1 is considered the week of the first
     * Thursday of the month.
     * https://en.wikipedia.org/wiki/ISO_week_date#First_week As a result, we
     * need special handling depending on the first day of the year.
     */

    /* Implementation taken from Pandas.
     * https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L161
     */
    int64_t doy = get_day_of_year(year, month, day);
    int64_t dow = dayofweek(year, month, day);

    // estimate
    int64_t iso_week = (doy - 1) - dow + 3;
    if (iso_week >= 0) {
        iso_week = (iso_week / 7) + 1;
    }

    // verify
    if (iso_week < 0) {
        if ((iso_week > -2) || (iso_week == -2 && is_leapyear(year - 1))) {
            iso_week = 53;
        } else {
            iso_week = 52;
        }
    } else if (iso_week == 53) {
        if ((31 - day + dow) < 3) {
            iso_week = 1;
        }
    }

    int64_t iso_year = year;
    if (iso_week == 1 && month == 12) {
        iso_year += 1;
    } else if (iso_week >= 52 && month == 1) {
        iso_year -= 1;
    }

    // Assign the calendar values to the pointers
    *year_res = iso_year;
    *week_res = iso_week;
    // Add 1 to day value for 1 indexing
    *dow_res = dow + 1;
}

/**
 * @brief extracts year and days from dt64 value, and updates to the remaining
 * dt64 from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L603
 * @param dt[in,out] dt64
 * @param year[out] extracted year
 * @param days[out] extracted days
 */
void extract_year_days(npy_datetime* dt, int64_t* year, int64_t* days) {
    //
    npy_int64 perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;
    *days = extract_unit(dt, perday);  // NOTE: dt is updated here as well
    *year = days_to_yearsdays(days);
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

// copied from Pandas, but input is changed from npy_datetime to
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

const static int64_t _NANOS_PER_MICRO = 1000;
const static int64_t _NANOS_PER_MILLI = 1000 * _NANOS_PER_MICRO;
const static int64_t _NANOS_PER_SECOND = 1000 * _NANOS_PER_MILLI;
const static int64_t _NANOS_PER_MINUTE = 60 * _NANOS_PER_SECOND;
const static int64_t _NANOS_PER_HOUR = 60 * _NANOS_PER_MINUTE;

/**
 * @brief Box native time_array data to Numpy object array of
 * bodo.types.Time items
 * @return Numpy object array of bodo.types.Time
 * @param[in] n number of values
 * @param[in] data pointer to 64-bit values
 * @param[in] null_bitmap bit vector representing nulls (Arrow format)
 * @param[in] precision number of decimal places to use for Time values
 */
void* box_time_array(int64_t n, const int64_t* data, const uint8_t* null_bitmap,
                     uint8_t precision) {
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

    // get bodo.types.Time constructor
    PyObject* bodo = PyImport_ImportModule("bodo");
    CHECK(bodo, "importing bodo module failed");
    PyObject* bodo_types = PyObject_GetAttrString(bodo, "types");
    CHECK(bodo_types, "getting bodo.types module failed");
    PyObject* bodo_time_constructor =
        PyObject_GetAttrString(bodo_types, "Time");
    CHECK(bodo_time_constructor, "getting bodo.types.Time failed");

    for (int64_t i = 0; i < n; ++i) {
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (!is_na(null_bitmap, i)) {
            int64_t val = data[i];
            int64_t nanosecond = val % _NANOS_PER_MICRO;
            int64_t microsecond = (val % _NANOS_PER_MILLI) / _NANOS_PER_MICRO;
            int64_t millisecond = (val % _NANOS_PER_SECOND) / _NANOS_PER_MILLI;
            int64_t second = (val % _NANOS_PER_MINUTE) / _NANOS_PER_SECOND;
            int64_t minute = (val % _NANOS_PER_HOUR) / _NANOS_PER_MINUTE;
            int64_t hour = val / _NANOS_PER_HOUR;
            PyObject* d = PyObject_CallFunction(
                bodo_time_constructor, "LLLLLLL", hour, minute, second,
                millisecond, microsecond, nanosecond, precision);
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, d);
            Py_DECREF(d);
        } else {
            // TODO: replace None with pd.NA when Pandas switch to pd.NA
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, Py_None);
        }
        CHECK(err == 0, "setting item in numpy array failed");
    }

    Py_DECREF(bodo_time_constructor);
    Py_DECREF(bodo_types);
    Py_DECREF(bodo);

    PyGILState_Release(gilstate);

    return ret;
#undef CHECK
}

/**
 * @brief unbox ndarray of Time objects into native Time array
 *
 * @param obj ndarray object of Time objects
 * @param n number of values
 * @param data pointer to 64-bit data buffer
 * @param null_bitmap pointer to null_bitmap buffer
 */
void unbox_time_array(PyObject* obj, int64_t n, int64_t* data,
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
    // Pandas usually stores NaT for date arrays
    PyObject* C_NAT = PyObject_GetAttrString(pd_mod, "NaT");
    CHECK(C_NAT, "getting pd.NaT failed");
    PyObject* datetime = PyImport_ImportModule("datetime");
    CHECK(datetime, "importing datetime module failed");
    PyObject* datetime_time_class = PyObject_GetAttrString(datetime, "time");

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
            value_data = 0;
        } else {
            // Pyarrow boxes into datetime.time objects.
            // TODO: Determine how to get the underlying arrow data.
            if (PyObject_IsInstance(s, datetime_time_class)) {
                PyObject* hour_obj = PyObject_GetAttrString(s, "hour");
                PyObject* minute_obj = PyObject_GetAttrString(s, "minute");
                PyObject* second_obj = PyObject_GetAttrString(s, "second");
                PyObject* microsecond_obj =
                    PyObject_GetAttrString(s, "microsecond");

                int64_t hour = PyLong_AsLongLong(hour_obj);
                int64_t minute = PyLong_AsLongLong(minute_obj);
                int64_t second = PyLong_AsLongLong(second_obj);
                int64_t microsecond = PyLong_AsLongLong(microsecond_obj);

                value_data =
                    hour * _NANOS_PER_HOUR + minute * _NANOS_PER_MINUTE +
                    second * _NANOS_PER_SECOND + microsecond * _NANOS_PER_MICRO;
                Py_DECREF(hour_obj);
                Py_DECREF(minute_obj);
                Py_DECREF(second_obj);
                Py_DECREF(microsecond_obj);
            } else {
                PyObject* value_obj = PyObject_GetAttrString(s, "value");
                value_data = PyLong_AsLongLong(value_obj);
                Py_DECREF(value_obj);
            }
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

    Py_DECREF(datetime_time_class);
    Py_DECREF(datetime);
    Py_DECREF(C_NA);
    Py_DECREF(C_NAT);
    Py_DECREF(pd_mod);

    PyGILState_Release(gilstate);

    return;
#undef CHECK
}

/**
 * @brief box a date_offset object. Null fields should be omitted. 0 is included
 * if has_kws.
 * @return DateOffset PyObject
 * @param n n value for DateOffset
 * @param normalize normalize value for Dateoffset
 * @param fields_arr Array of fields that may be initialized
 * @param has_kws Bool for if any non-nano fields are initialized. This impacts
 * behavior
 */
PyObject* box_date_offset(int64_t n, bool normalize, int64_t fields_arr[18],
                          bool has_kws) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return nullptr;                \
    }
    auto gilstate = PyGILState_Ensure();

    // set of fields names for the array. They are divided into 2 groups based
    // upon if their default value can be used to determine if the value
    // was/wasn't included. Those ending with s cannot be distinguished so they
    // will always be added if has_kws This is fine because 0 is the same
    // behavior as not being included, so long as 1 field is included.
    const char* fields[2][9] = {
        {"years", "months", "weeks", "days", "hours", "minutes", "seconds",
         "microseconds", "nanoseconds"},
        {"year", "month", "day", "weekday", "hour", "minute", "second",
         "microsecond", "nanosecond"}};

    int64_t default_values[2] = {0, -1};
    // Vector of pyobjs for tracking decref
    std::vector<PyObject*> pyobjs;

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
                    PyObject* value_obj = Py_BuildValue("L", field_value);
                    CHECK(value_obj, "Creating value obj for kwargs failed");
                    CHECK(PyDict_SetItem(kwargs, field_obj, value_obj) != -1,
                          "Dict setitem failed");
                    pyobjs.push_back(field_obj);
                    pyobjs.push_back(value_obj);
                }
            }
        }
    }
    PyObject* n_obj = Py_BuildValue("L", n);
    CHECK(n_obj, "Creating n object failed");
    PyObject* normalize_obj = PyBool_FromLong(normalize);
    CHECK(normalize_obj, "Creating normalize object failed");
    PyObject* args = PyTuple_Pack(2, n_obj, normalize_obj);
    CHECK(args, "Creating *args failed");
    pyobjs.push_back(n_obj);
    pyobjs.push_back(normalize_obj);
    pyobjs.push_back(args);
    PyObject* module = PyImport_ImportModule("pandas.tseries.offsets");
    CHECK(module, "importing pandas.tseries.offsets module failed");
    PyObject* function = PyObject_GetAttrString(module, "DateOffset");
    pyobjs.push_back(module);
    pyobjs.push_back(function);
    PyObject* date_offset_obj = PyObject_Call(function, args, kwargs);
    CHECK(date_offset_obj, "DateOffset constructor failed");
    // Make sure to update reference counting
    for (auto& obj : pyobjs) {
        Py_DECREF(obj);
    }
    PyGILState_Release(gilstate);

    return date_offset_obj;
#undef CHECK
}

/**
 * @brief unbox a date_offset object. Missing fields should get their null
 * value.
 * @return boolean for if the obj has any kws
 * @param obj pd.tseries.offsets.DateOffset object
 * @param fields_arr Array of fields that must be initialized
 */
bool unbox_date_offset(PyObject* obj, int64_t fields_arr[18]) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return false;                  \
    }
    auto gilstate = PyGILState_Ensure();

    // set of fields, split by default value if missing
    const char* fields[2][9] = {
        {"years", "months", "weeks", "days", "hours", "minutes", "seconds",
         "microseconds", "nanoseconds"},
        {"year", "month", "day", "weekday", "hour", "minute", "second",
         "microsecond", "nanosecond"}};

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

/**
 * @brief unbox ndarray of TimestampTZ objects into native int64/int64 arrays
 *
 * @param obj ndarray object of Time objects
 * @param n number of values
 * @param data pointer to 64-bit timestamp buffer
 * @param data pointer to 16-bit offset buffer
 * @param null_bitmap pointer to null_bitmap buffer
 */
void unbox_timestamptz_array(PyObject* obj, int64_t n, int64_t* data_ts,
                             int16_t* data_offset, uint8_t* null_bitmap) {
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
    CHECK(n >= 0 && data_ts && data_offset && null_bitmap,
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
        int64_t ts_data;
        int16_t ts_offset;
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA || s == C_NAT) {
            value_bitmap = false;
            // Set na data to a legal value for array getitem.
            ts_data = 0;
            ts_offset = 0;
        } else {
            PyObject* ts_obj = PyObject_GetAttrString(s, "utc_timestamp");
            PyObject* ts_value_obj = PyObject_GetAttrString(ts_obj, "value");
            ts_data = PyLong_AsLongLong(ts_value_obj);
            Py_DECREF(ts_value_obj);
            Py_DECREF(ts_obj);

            PyObject* ts_offset_obj =
                PyObject_GetAttrString(s, "offset_minutes");
            ts_offset = (int16_t)PyLong_AsLongLong(ts_offset_obj);
            Py_DECREF(ts_offset_obj);

            value_bitmap = true;
        }
        data_ts[i] = ts_data;
        data_offset[i] = ts_offset;
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
 * @brief Box native timestamptz_array data to Numpy object array of
 * bodo.types.TimestampTZ items
 * @return Numpy object array of bodo.types.TimestampTZ
 * @param[in] n number of values
 * @param[in] data timestamp pointer to 64-bit values
 * @param[in] data offset pointer to 64-bit values
 * @param[in] null_bitmap bit vector representing nulls (Arrow format)
 */
void* box_timestamptz_array(int64_t n, const int64_t* data_ts,
                            const int16_t* data_offset,
                            const uint8_t* null_bitmap, uint8_t precision) {
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

    // get pd.Timestamp constructor
    PyObject* pandas = PyImport_ImportModule("pandas");
    CHECK(pandas, "importing bodo module failed");
    PyObject* timestamp_constructor =
        PyObject_GetAttrString(pandas, "Timestamp");
    CHECK(timestamp_constructor, "getting pandas.Timestamp failed");

    // get bodo.types.TimestampTZ constructor
    PyObject* bodo = PyImport_ImportModule("bodo");
    CHECK(bodo, "importing bodo module failed");
    PyObject* bodo_types = PyObject_GetAttrString(bodo, "types");
    CHECK(bodo_types, "getting bodo.types module failed");
    PyObject* bodo_timestamptz_constructor =
        PyObject_GetAttrString(bodo_types, "TimestampTZ");
    CHECK(bodo_timestamptz_constructor,
          "getting bodo.types.TimestampTZ failed");

    for (int64_t i = 0; i < n; ++i) {
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (!is_na(null_bitmap, i)) {
            int64_t ts_val = data_ts[i];
            PyObject* ts =
                PyObject_CallFunction(timestamp_constructor, "L", ts_val);

            int16_t offset = data_offset[i];
            // NOTE: int64_t casting of offset is necessary on Windows
            PyObject* ts_tz =
                PyObject_CallFunction(bodo_timestamptz_constructor, "OL", ts,
                                      static_cast<int64_t>(offset));
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, ts_tz);

            Py_DECREF(ts);
            Py_DECREF(ts_tz);
        } else {
            // TODO: replace None with pd.NA when Pandas switch to pd.NA
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, Py_None);
        }
        CHECK(err == 0, "setting item in numpy array failed");
    }

    Py_DECREF(bodo_timestamptz_constructor);
    Py_DECREF(bodo_types);
    Py_DECREF(bodo);
    Py_DECREF(timestamp_constructor);
    Py_DECREF(pandas);

    PyGILState_Release(gilstate);

    return ret;
#undef CHECK
}

PyMODINIT_FUNC PyInit_hdatetime_ext(void) {
    PyObject* m;
    MOD_DEF(m, "hdatetime_ext", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    // init numpy
    import_array();

    // initalize memory alloc/tracking system in _meminfo.h
    bodo_common_init();

    // These are all C functions, so they don't throw any exceptions.
    // We might still need to add better error handling in the future.
    SetAttrStringFromVoidPtr(m, get_isocalendar);
    SetAttrStringFromVoidPtr(m, extract_year_days);
    SetAttrStringFromVoidPtr(m, get_month_day);
    SetAttrStringFromVoidPtr(m, npy_datetimestruct_to_datetime);
    SetAttrStringFromVoidPtr(m, box_time_array);
    SetAttrStringFromVoidPtr(m, unbox_time_array);
    SetAttrStringFromVoidPtr(m, unbox_date_offset);
    SetAttrStringFromVoidPtr(m, box_date_offset);
    SetAttrStringFromVoidPtr(m, get_days_from_date);
    SetAttrStringFromVoidPtr(m, unbox_timestamptz_array);
    SetAttrStringFromVoidPtr(m, box_timestamptz_array);

    return m;
}

}  // extern "C"
