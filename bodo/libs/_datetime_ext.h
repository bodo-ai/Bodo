// Copyright (C) 2020 Bodo Inc. All rights reserved.
#ifndef BODO_DATETIME_H_INCLUDED_
#define BODO_DATETIME_H_INCLUDED_
#include <cstdint>

static int is_leapyear(int64_t year) {
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 || (year % 400) == 0);
}

// Days per month, regular year and leap year
static const int days_per_month_table[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};

/*
 * Array used to calculate dayofweek
 * Taken from Pandas:
 * https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L20
 */
static const int sakamoto_arr[12] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};

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

/*
 * Return the ordinal day-of-year for the given day.
 * Copied from Pandas:
 * https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L215
 */
static int64_t get_day_of_year(int64_t year, int64_t month, int64_t day) {
    int64_t mo_off = month_offset[(13 * is_leapyear(year)) + (month - 1)];
    return mo_off + day;
}

/*
 * Find the day of week for the date described by the Y/M/D triple y, m, d
 * using Sakamoto's method, from wikipedia.
 * Copied from Pandas:
 * https://github.com/pandas-dev/pandas/blob/e088ea31a897929848caa4b5ce3db9d308c604db/pandas/_libs/tslibs/ccalendar.pyx#L83
 */
static int64_t dayofweek(int64_t y, int64_t m, int64_t d) {
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
static void get_isocalendar(int64_t year, int64_t month, int64_t day,
                            int64_t* year_res, int64_t* week_res,
                            int64_t* dow_res) {
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

// copied from Arrow since not in exported APIs
// https://github.com/apache/arrow/blob/329c9944554ddb142b0a2ac26a4abdf477636e37/cpp/src/arrow/python/datetime.cc#L58
// Calculates the days offset from the 1970 epoch.
static int64_t get_days_from_date(int64_t date_year, int64_t date_month,
                                  int64_t date_day) {
    int64_t i, month;
    int64_t year, days = 0;
    const int* month_lengths;

    year = date_year - 1970;
    days = year * 365;

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
static int64_t days_to_yearsdays(int64_t* days_) {
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

#endif /* BODO_DATETIME_H_INCLUDED_ */
