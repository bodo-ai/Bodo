#pragma once

#include <Python.h>
#include <cstddef>
#include <cstdint>

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
                           int64_t date_day);

/**
 * @brief Modifies '*days_' to be the day offset within the year, and returns
 * the year. copied from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L166
 * @param days_[in,out] input: total days output: day offset within the year
 * @return int64_t output year
 */
int64_t days_to_yearsdays(int64_t* days_);

/**
 * @brief Get extracts month and day from days offset within a year
 * from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L230
 * @param year[in]
 * @param days[in]
 * @param month[out]
 * @param day[out]
 */
void get_month_day(int64_t year, int64_t days, int64_t* month, int64_t* day);

/**
 * @brief converts an integer to a Python datetime.date object.
 * @param[in] ordinal: the ordinal value of the day.
 * For example, ordinal 1 = January 1st, the year 1.
 */
PyObject* py_date_from_int(size_t ordinal);

/**
 * @brief converts an integer to a Python Timestamp object.
 * @param[in] val: the number of nanoseconds since midnight
 * on January 1st, 1970.
 */
PyObject* py_timestamp_from_int(int64_t val);
