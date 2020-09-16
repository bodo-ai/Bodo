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

// Implementation of a general get_isocalendar to be reused
// by various pandas types. Places return values inside the allocated
// buffers. Iso calendar returns the year, week number, and day of the
// week for a given date described by a year month and day
static void get_isocalendar(int64_t date_year, int64_t date_month, 
        int64_t date_day, int64_t* year_res, int64_t* week_res, int64_t* dow_res) {

    /*
     * NOTE: Pandas has weird behavior and it doesn't seem to be well defined.
     * The documentation is sparse:
     * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.isocalendar.html
     * Weird bugs also seem to happen with certain calendar inputs (but not all). This seems to
     * typically occur at the start of specific years believing they are part of the previous year
     * Example:
     * >>> pd.Timestamp(2000, 1, 1, 1).isocalendar()
     * (1999, 52, 6)
     * Our implementation does not have this weird behavior/these bugs
     * >>> pd.Timestamp(2000, 1, 1, 1).isocalendar()
     * (2000, 1, 7)
     * It also seems that for these years the problems will persist each day of that year.
     * Pandas also has issues with input validation. Early years are not rejected by crash due to 
     * storage issues.
     */


    /*
     * Calculate the weekday of the day before 1601 for easy
     * leap year calculation (but pandas won't accept 1601-1677.
     * This year started on a Monday and we will calculate years with
     * it. However we want to add the days so we are "starting" at the
     * end of 1600.
     *
     * For now calculations assume Monday = 0 -> Sunday = 6 for calcuating weeks.
     * Panda does -> Monday = 1, Sunday = 7, so we will add 1 to our output.
     */
     int64_t earliest_year = 1601;

    // Start 1 day early to make the math simpler
    int64_t start_day = 6;

    int64_t year_difference = date_year - earliest_year;
    int64_t num_leap_years = (year_difference / 4) - (year_difference / 100) + (year_difference / 400);

    //  Our day offset differs form the constant value for 1601 by the number of years
    // + number_leap years between them
    int64_t start_offset = year_difference + num_leap_years;
    start_day = (start_day + start_offset) % 7;

    int year_type = 0;
    if (is_leapyear(date_year)) {
        year_type = 1;
    }
    for (int i = 0; i < date_month - 1; i++) {
        start_day += days_per_month_table[year_type][i];
    }

    // Include the days from this month
    start_day += date_day;

    // Assign the calendar values to the pointers
    *year_res = date_year;
    // Add 1 to week and day values for 1 indexing
    *week_res = (start_day / 7) + 1;
    *dow_res = (start_day % 7) + 1;
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

  month_lengths = days_per_month_table[is_leapyear(date_year)];
  month = date_month - 1;

  // Add the months
  for (i = 0; i < month; ++i) {
    days += month_lengths[i];
  }

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


#endif /* BODO_DATETIME_H_INCLUDED_ */
