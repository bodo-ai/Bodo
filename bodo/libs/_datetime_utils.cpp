// Copyright (C) 2022 Bodo Inc. All rights reserved.

/**
 * Datetime helper functions used in files that are
 * likely too large to be inlined.
 */

#include "_datetime_utils.h"
#include <Python.h>
#include <iostream>

#include "../libs/_datetime_ext.h"
#include "_datetime_ext.h"

/**
 * @brief copied from Arrow since not in exported APIs
 * https://github.com/apache/arrow/blob/329c9944554ddb142b0a2ac26a4abdf477636e37/cpp/src/arrow/python/datetime.cc#L58
 * Calculates the days offset from the 1970 epoch.
 *
 * @param date_year year >= 1970
 * @param date_month month [1, 12]
 * @param date_day day [1, 31]
 * @return int64_t The days offset from the 1970 epoch
 */
int64_t get_days_from_date(int64_t date_year, int64_t date_month,
                           int64_t date_day) {
    int64_t year = date_year - 1970;
    int64_t days = year * 365;

    // Adjust for leap years
    if (days >= 0) {
        // 1968 is the closest leap year before 1970.
        // Exclude the current year, so add 1.
        year += 1;
        // Add one day for each 4 years
        days += year / 4;
        // 1900 is the closest previous year divisible by 100
        year += 68;
        // Subtract one day for each 100 years
        days -= year / 100;
        // 1600 is the closest previous year divisible by 400
        year += 300;
        // Add one day for each 400 years
        days += year / 400;
    } else {
        // 1972 is the closest later year after 1970.
        // Include the current year, so subtract 2.
        year -= 2;
        // Subtract one day for each 4 years
        days += year / 4;
        // 2000 is the closest later year divisible by 100
        year -= 28;
        // Add one day for each 100 years
        days -= year / 100;
        // 2000 is also the closest later year divisible by 400
        // Subtract one day for each 400 years
        days += year / 400;
    }

    // Add the months
    days += month_offset[(13 * is_leapyear(date_year)) + (date_month - 1)];

    // Add the days
    days += date_day - 1;

    return days;
}

/**
 * @brief Modifies '*days_' to be the day offset within the year, and returns
 * the year. copied from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L166
 * @param days_[in,out] input: total days output: day offset within the year
 * @return int64_t output year
 */
int64_t days_to_yearsdays(int64_t* days_) {
    const int64_t days_per_400years = (400 * 365 + 100 - 4 + 1);
    /* Adjust so it's relative to the year 2000 (divisible by 400) */
    int64_t days = (*days_) - (365 * 30 + 7);
    int64_t year;

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

void get_month_day(int64_t year, int64_t days, int64_t* month, int64_t* day) {
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

PyObject* py_date_from_int(size_t ordinal) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return NULL;                   \
    }

    auto gilstate = PyGILState_Ensure();

    // get datetime.date constructor
    PyObject* datetime = PyImport_ImportModule("datetime");
    CHECK(datetime, "importing datetime module failed");
    PyObject* datetime_date_constructor =
        PyObject_GetAttrString(datetime, "date");
    CHECK(datetime_date_constructor, "getting datetime.date failed");
    int64_t day = ordinal;
    int64_t month = 0;
    int64_t year = days_to_yearsdays(&day);
    get_month_day(year, day, &month, &day);

    auto result = PyObject_CallFunction(datetime_date_constructor, "LLL", year,
                                        month, day);

    Py_DECREF(datetime_date_constructor);
    Py_DECREF(datetime);
    PyGILState_Release(gilstate);
    return result;
}

PyObject* py_timestamp_from_int(int64_t val) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return NULL;                   \
    }

    auto gilstate = PyGILState_Ensure();

    // get pd.Timestamp constructor
    PyObject* pandas = PyImport_ImportModule("pandas");
    CHECK(pandas, "importing pandas module failed");
    PyObject* timestamp_constructor =
        PyObject_GetAttrString(pandas, "Timestamp");
    CHECK(timestamp_constructor, "getting pandas.Timestamp failed");

    auto result = PyObject_CallFunction(timestamp_constructor, "L", val);

    Py_DECREF(pandas);
    Py_DECREF(timestamp_constructor);
    PyGILState_Release(gilstate);
    return result;
}
