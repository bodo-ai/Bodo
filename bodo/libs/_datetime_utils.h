// Copyright (C) 2022 Bodo Inc. All rights reserved.
#ifndef BODO_DATETIME_UTILS_H_INCLUDED_
#define BODO_DATETIME_UTILS_H_INCLUDED_

#include <stdint.h>

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

#endif /* BODO_DATETIME_UTILS_H_INCLUDED_ */
