// Copyright (C) 2020 Bodo Inc. All rights reserved.
#ifndef BODO_DATETIME_H_INCLUDED_
#define BODO_DATETIME_H_INCLUDED_
#include <cstdint>

static int is_leapyear(int64_t year) {
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 || (year % 400) == 0);
}

/*
 * Taken from Pandas:
 * https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L22
 *
 * The first 13 entries give the month days elapsed as of the first of month N
 * (or the total number of days in the year for N=13) in non-leap years.
 * The remaining 13 entries give the days elapsed in leap years.
 */
static const int month_offset[26] = {
    0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365,
    0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};

// Days per month, regular year and leap year
static const int days_per_month_table[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};

#endif /* BODO_DATETIME_H_INCLUDED_ */
