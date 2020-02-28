// Copyright (C) 2020 Bodo Inc. All rights reserved.
#ifndef BODO_DATETIME_H_
#define BODO_DATETIME_H_
#include <cstdint>


static int is_leapyear(int64_t year) {
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 || (year % 400) == 0);
}

static const int days_per_month_table[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};



/**
 * @brief Modifies '*days_' to be the day offset within the year, and returns
 * the year. copied from Pandas:
 * https://github.com/pandas-dev/pandas/blob/844dc4a4fb8d213303085709aa4a3649400ed51a/pandas/_libs/tslibs/src/datetime/np_datetime.c#L166
 * @param days_[in,out] input: total days output: day offset within the year
 * @return int64_t output year
 */
static int64_t days_to_yearsdays(int64_t *days_) {
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


#endif /* BODO_DATETIME_H_ */
