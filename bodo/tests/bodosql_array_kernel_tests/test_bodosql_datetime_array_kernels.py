# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL date/time functions
"""


import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.timezone_common import generate_date_trunc_func
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "args, answers",
    [
        pytest.param(
            (
                pd.Series([1, 1, 2, 2, -1, -1, 16, 16, -50, -50]),
                pd.Series([pd.Timestamp("2018-1-1"), None] * 5),
            ),
            {
                "years": pd.Series(
                    [
                        pd.Timestamp("2019-1-1"),
                        None,
                        pd.Timestamp("2020-1-1"),
                        None,
                        pd.Timestamp("2017-1-1"),
                        None,
                        pd.Timestamp("2034-1-1"),
                        None,
                        pd.Timestamp("1968-1-1"),
                        None,
                    ]
                ),
                "months": pd.Series(
                    [
                        pd.Timestamp("2018-2-1"),
                        None,
                        pd.Timestamp("2018-3-1"),
                        None,
                        pd.Timestamp("2017-12-1"),
                        None,
                        pd.Timestamp("2019-5-1"),
                        None,
                        pd.Timestamp("2013-11-1"),
                        None,
                    ]
                ),
                "weeks": pd.Series(
                    [
                        pd.Timestamp("2018-1-8"),
                        None,
                        pd.Timestamp("2018-1-15"),
                        None,
                        pd.Timestamp("2017-12-25"),
                        None,
                        pd.Timestamp("2018-4-23"),
                        None,
                        pd.Timestamp("2017-1-16"),
                        None,
                    ]
                ),
                "days": pd.Series(
                    [
                        pd.Timestamp("2018-1-2"),
                        None,
                        pd.Timestamp("2018-1-3"),
                        None,
                        pd.Timestamp("2017-12-31"),
                        None,
                        pd.Timestamp("2018-1-17"),
                        None,
                        pd.Timestamp("2017-11-12"),
                        None,
                    ]
                ),
            },
            id="all_vector",
        ),
        pytest.param(
            (
                100,
                pd.Series(pd.date_range("1999-12-20", "1999-12-30", 11)),
            ),
            {
                "years": pd.Series(pd.date_range("2099-12-20", "2099-12-30", 11)),
                "months": pd.Series(pd.date_range("2008-04-20", "2008-04-30", 11)),
                "weeks": pd.Series(pd.date_range("2001-11-19", "2001-11-29", 11)),
                "days": pd.Series(pd.date_range("2000-03-29", "2000-04-08", 11)),
            },
            id="scalar_vector",
        ),
        pytest.param(
            (300, pd.Timestamp("1776-7-4")),
            {
                "years": pd.Timestamp("2076-7-4"),
                "months": pd.Timestamp("1801-7-4"),
                "weeks": pd.Timestamp("1782-4-4"),
                "days": pd.Timestamp("1777-4-30"),
            },
            id="all_scalar",
        ),
        pytest.param(
            (None, pd.Timestamp("1776-7-4")),
            {
                "years": None,
                "months": None,
                "weeks": None,
                "days": None,
            },
            id="scalar_null",
        ),
    ],
)
@pytest.mark.parametrize(
    "unit",
    [
        "years",
        "months",
        "weeks",
        "days",
    ],
)
def test_add_interval_date_units(unit, args, answers, memory_leak_check):
    if any(isinstance(arg, pd.Series) for arg in args):
        fn_str = f"lambda amount, start_dt: pd.Series(bodo.libs.bodosql_array_kernels.add_interval_{unit}(amount, start_dt))"
    else:
        fn_str = f"lambda amount, start_dt: bodo.libs.bodosql_array_kernels.add_interval_{unit}(amount, start_dt)"
    impl = eval(fn_str)

    check_func(
        impl,
        args,
        py_output=answers[unit],
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args, answers",
    [
        pytest.param(
            (
                pd.Series([1, 30, -10, 42, 1234, -654321]),
                pd.Series(
                    [pd.Timestamp("2015-03-14")] * 3
                    + [None]
                    + [pd.Timestamp("2015-03-14")] * 2
                ),
            ),
            {
                "hours": pd.Series(
                    [
                        pd.Timestamp("2015-3-14 01:00:00"),
                        pd.Timestamp("2015-3-15 06:00:00"),
                        pd.Timestamp("2015-3-13 14:00:00"),
                        None,
                        pd.Timestamp("2015-5-4 10:00:00"),
                        pd.Timestamp("1940-7-21 15:00:00"),
                    ]
                ),
                "minutes": pd.Series(
                    [
                        pd.Timestamp("2015-3-14 00:01:00"),
                        pd.Timestamp("2015-3-14 00:30:00"),
                        pd.Timestamp("2015-3-13 23:50:00"),
                        None,
                        pd.Timestamp("2015-3-14 20:34:00"),
                        pd.Timestamp("2013-12-14 14:39:00"),
                    ]
                ),
                "seconds": pd.Series(
                    [
                        pd.Timestamp("2015-3-14 00:00:01"),
                        pd.Timestamp("2015-3-14 00:00:30"),
                        pd.Timestamp("2015-3-13 23:59:50"),
                        None,
                        pd.Timestamp("2015-3-14 00:20:34"),
                        pd.Timestamp("2015-3-6 10:14:39"),
                    ]
                ),
                "milliseconds": pd.Series(
                    [
                        pd.Timestamp("2015-3-14 00:00:00.001000"),
                        pd.Timestamp("2015-3-14 00:00:00.030000"),
                        pd.Timestamp("2015-03-13 23:59:59.990000"),
                        None,
                        pd.Timestamp("2015-3-14 00:00:01.234000"),
                        pd.Timestamp("2015-03-13 23:49:05.679000"),
                    ]
                ),
                "microseconds": pd.Series(
                    [
                        pd.Timestamp("2015-3-14 00:00:00.000001"),
                        pd.Timestamp("2015-3-14 00:00:00.000030"),
                        pd.Timestamp("2015-3-13 23:59:59.999990"),
                        None,
                        pd.Timestamp("2015-3-14 00:00:00.001234"),
                        pd.Timestamp("2015-03-13 23:59:59.345679"),
                    ]
                ),
                "nanoseconds": pd.Series(
                    [
                        pd.Timestamp("2015-3-14 00:00:00.000000001"),
                        pd.Timestamp("2015-3-14 00:00:00.000000030"),
                        pd.Timestamp("2015-03-13 23:59:59.999999990"),
                        None,
                        pd.Timestamp("2015-3-14 00:00:00.000001234"),
                        pd.Timestamp("2015-03-13 23:59:59.999345679"),
                    ]
                ),
            },
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series([16 ** (i + 1) for i in range(5)]),
                pd.Timestamp("2022-11-9 09:42:30.151121521"),
            ),
            {
                "hours": pd.Series(
                    [
                        pd.Timestamp("2022-11-10 01:42:30.151121521"),
                        pd.Timestamp("2022-11-20 1:42:30.151121521"),
                        pd.Timestamp("2023-4-29 1:42:30.151121521"),
                        pd.Timestamp("2030-5-2 1:42:30.151121521"),
                        pd.Timestamp("2142-6-24 1:42:30.151121521"),
                    ]
                ),
                "minutes": pd.Series(
                    [
                        pd.Timestamp("2022-11-09 09:58:30.151121521"),
                        pd.Timestamp("2022-11-09 13:58:30.151121521"),
                        pd.Timestamp("2022-11-12 05:58:30.151121521"),
                        pd.Timestamp("2022-12-24 21:58:30.151121521"),
                        pd.Timestamp("2024-11-06 13:58:30.151121521"),
                    ]
                ),
                "seconds": pd.Series(
                    [
                        pd.Timestamp("2022-11-09 09:42:46.151121521"),
                        pd.Timestamp("2022-11-09 09:46:46.151121521"),
                        pd.Timestamp("2022-11-09 10:50:46.151121521"),
                        pd.Timestamp("2022-11-10 03:54:46.151121521"),
                        pd.Timestamp("2022-11-21 12:58:46.151121521"),
                    ]
                ),
                "milliseconds": pd.Series(
                    [
                        pd.Timestamp("2022-11-09 09:42:30.167121521"),
                        pd.Timestamp("2022-11-09 09:42:30.407121521"),
                        pd.Timestamp("2022-11-09 09:42:34.247121521"),
                        pd.Timestamp("2022-11-09 09:43:35.687121521"),
                        pd.Timestamp("2022-11-09 09:59:58.727121521"),
                    ]
                ),
                "microseconds": pd.Series(
                    [
                        pd.Timestamp("2022-11-09 09:42:30.151137521"),
                        pd.Timestamp("2022-11-09 09:42:30.151377521"),
                        pd.Timestamp("2022-11-09 09:42:30.155217521"),
                        pd.Timestamp("2022-11-09 09:42:30.216657521"),
                        pd.Timestamp("2022-11-09 09:42:31.199697521"),
                    ]
                ),
                "nanoseconds": pd.Series(
                    [
                        pd.Timestamp("2022-11-09 09:42:30.151121537"),
                        pd.Timestamp("2022-11-09 09:42:30.151121777"),
                        pd.Timestamp("2022-11-09 09:42:30.151125617"),
                        pd.Timestamp("2022-11-09 09:42:30.151187057"),
                        pd.Timestamp("2022-11-09 09:42:30.152170097"),
                    ]
                ),
            },
            id="vector_scalar",
        ),
        pytest.param(
            (
                300,
                pd.Timestamp("1986-2-27 20:10:15.625"),
            ),
            {
                "hours": pd.Timestamp("1986-03-12 08:10:15.625000"),
                "minutes": pd.Timestamp("1986-02-28 01:10:15.625000"),
                "seconds": pd.Timestamp("1986-02-27 20:15:15.625000"),
                "milliseconds": pd.Timestamp("1986-02-27 20:10:15.925000"),
                "microseconds": pd.Timestamp("1986-02-27 20:10:15.625300"),
                "nanoseconds": pd.Timestamp("1986-02-27 20:10:15.625000300"),
            },
            id="all_scalar",
        ),
        pytest.param(
            (40, None),
            {
                "hours": None,
                "minutes": None,
                "seconds": None,
                "milliseconds": None,
                "microseconds": None,
                "nanoseconds": None,
            },
            id="scalar_null",
        ),
    ],
)
@pytest.mark.parametrize(
    "unit",
    [
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ],
)
def test_add_interval_time_units(unit, args, answers, memory_leak_check):
    if any(isinstance(arg, pd.Series) for arg in args):
        fn_str = f"lambda amount, start_dt: pd.Series(bodo.libs.bodosql_array_kernels.add_interval_{unit}(amount, start_dt))"
    else:
        fn_str = f"lambda amount, start_dt: bodo.libs.bodosql_array_kernels.add_interval_{unit}(amount, start_dt)"
    impl = eval(fn_str)

    check_func(
        impl,
        args,
        py_output=answers[unit],
        check_dtype=False,
        reset_index=True,
    )


def make_add_interval_tz_test(amount, start, target, is_vector):
    """
    Takes in a start/end timestamp string and converts them to tz-aware timestamps
    with a timestamp chosen based on the amount added. Either keeps as a scalar
    or converts to a vector.
    """
    timezones = [
        "US/Pacific",
        "US/Mountain",
        "US/Eastern",
        "Europe/Madrid",
        "Australia/Sydney",
        "Poland",
        "US/Alaska",
    ]
    tz = timezones[amount % len(timezones)]
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(target, tz=tz)
    if is_vector:
        start_vector = pd.Series([start_ts, None] * 3)
        end_vector = pd.Series([end_ts, None] * 3)
        return start_vector, end_vector
    else:
        return start_ts, end_ts


@pytest.mark.parametrize(
    "is_vector", [pytest.param(True, id="vector"), pytest.param(False, id="scalar")]
)
@pytest.mark.parametrize(
    "unit, amount, start, answer",
    [
        pytest.param("years", 5, "mar", "2025-3-7 12:00:00", id="years-mar"),
        pytest.param(
            "months",
            14,
            "mar",
            "2021-5-7 12:00:00",
            id="months-mar",
            marks=pytest.mark.slow,
        ),
        pytest.param("weeks", 110, "mar", "2022-4-16 12:00:00", id="weeks-mar"),
        pytest.param(
            "days",
            3,
            "mar",
            "2020-3-10 12:00:00",
            id="days-mar",
            marks=pytest.mark.slow,
        ),
        pytest.param("hours", 20, "mar", "2020-3-8 8:00:00", id="hours-mar"),
        pytest.param(
            "seconds",
            86490,
            "mar",
            "2020-3-8 12:01:30",
            id="seconds-mar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "milliseconds",
            105359250,
            "mar",
            "2020-3-8 17:15:59.250",
            id="milliseconds-mar",
        ),
        pytest.param(
            "years",
            2,
            "nov",
            "2022-11-1 00:00:00",
            id="years-nov",
            marks=pytest.mark.slow,
        ),
        pytest.param("months", 1, "nov", "2020-12-1 00:00:00", id="months-nov"),
        pytest.param(
            "weeks",
            2,
            "nov",
            "2020-11-15 00:00:00",
            id="weeks-nov",
            marks=pytest.mark.slow,
        ),
        pytest.param("days", 1, "nov", "2020-11-2 00:00:00", id="days-nov"),
        pytest.param(
            "hours",
            6,
            "nov",
            "2020-11-1 06:00:00",
            id="hours-nov",
            marks=pytest.mark.slow,
        ),
        pytest.param("minutes", 255, "nov", "2020-11-1 04:15:00", id="minutes-nov"),
        pytest.param(
            "seconds",
            19921,
            "nov",
            "2020-11-1 05:32:01",
            id="seconds-nov",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "nanoseconds",
            15150123456789,
            "nov",
            "2020-11-1 04:12:30.123456789",
            id="nanoseconds-nov",
        ),
    ],
)
def test_add_interval_tz(unit, amount, start, answer, is_vector, memory_leak_check):
    """Tests the add_interval_xxx kernels on timezone data. All the vector
    tests use a starting date right before the spring daylight savings jump
    of 2020, and all the scalar tests use a starting date right before hte
    fall daylight savings jump of 2020.

    Current expected behavior: daylight savings behavior ignored."""

    # Map mar/nov to the corresponding starting timestamps
    if start == "mar":
        starting_dt = "2020-3-7 12:00:00"
    else:
        starting_dt = "2020-11-1 00:00:00"

    start, answer = make_add_interval_tz_test(amount, starting_dt, answer, is_vector)

    if any(isinstance(arg, pd.Series) for arg in (amount, start)):
        fn_str = f"lambda amount, start_dt: pd.Series(bodo.libs.bodosql_array_kernels.add_interval_{unit}(amount, start_dt))"
    else:
        fn_str = f"lambda amount, start_dt: bodo.libs.bodosql_array_kernels.add_interval_{unit}(amount, start_dt)"
    impl = eval(fn_str)

    check_func(
        impl,
        (amount, start),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(
                [
                    pd.Timestamp(d)
                    for d in pd.date_range("2018-01-01", "2019-01-01", periods=20)
                ]
                + [None, None]
                + [
                    pd.Timestamp(d)
                    for d in pd.date_range("1970-01-01", "2108-01-01", periods=20)
                ]
            ),
            id="vector",
        ),
        pytest.param(pd.Timestamp("2000-10-29"), id="scalar"),
    ],
)
def dates_scalar_vector(request):
    """A fixture of either a single timestamp, or a series of timestamps from
    various year/month ranges with some nulls inserted. Uses pd.Series on
    concatenated lists instead of pd.concat since the date_range outputs
    a DatetimeIndex with a potentially inconvenient dtype when combinined."""
    return request.param


def test_dayname(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.dayname(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.dayname(arr)

    # Simulates DAYNAME on a single row
    def dayname_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.day_name()

    dayname_answer = vectorized_sol((dates_scalar_vector,), dayname_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=dayname_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_dayofmonth(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.dayofmonth(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.dayofmonth(arr)

    # Simulates DAYOFMONTH on a single row
    def dayofmonth_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.day

    dayname_answer = vectorized_sol((dates_scalar_vector,), dayofmonth_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=dayname_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_dayofweek(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.dayofweek(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.dayofweek(arr)

    # Simulates DAYOFWEEK on a single row
    def dayofweek_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            dow_dict = {
                "Monday": 1,
                "Tuesday": 2,
                "Wednesday": 3,
                "Thursday": 4,
                "Friday": 5,
                "Saturday": 6,
                "Sunday": 0,
            }
            return dow_dict[elem.day_name()]

    dayname_answer = vectorized_sol((dates_scalar_vector,), dayofweek_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=dayname_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_dayofweekiso(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.dayofweekiso(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.dayofweekiso(arr)

    # Simulates DAYOFWEEKISO on a single row
    def dayofweekiso_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            dow_dict = {
                "Monday": 1,
                "Tuesday": 2,
                "Wednesday": 3,
                "Thursday": 4,
                "Friday": 5,
                "Saturday": 6,
                "Sunday": 7,
            }
            return dow_dict[elem.day_name()]

    dayname_answer = vectorized_sol(
        (dates_scalar_vector,), dayofweekiso_scalar_fn, None
    )
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=dayname_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_dayofyear(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.dayofyear(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.dayofyear(arr)

    # Simulates DAYOFYEAR on a single row
    def dayofyear_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.dayofyear

    dayname_answer = vectorized_sol((dates_scalar_vector,), dayofyear_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=dayname_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "days",
    [
        pytest.param(
            pd.Series(pd.array([0, 1, -2, 4, 8, None, -32])),
            id="vector",
        ),
        pytest.param(
            42,
            id="scalar",
        ),
    ],
)
def test_int_to_days(days, memory_leak_check):
    def impl(days):
        return pd.Series(bodo.libs.bodosql_array_kernels.int_to_days(days))

    # avoid pd.Series() conversion for scalar output
    if isinstance(days, int):
        impl = lambda days: bodo.libs.bodosql_array_kernels.int_to_days(days)

    # Simulates int_to_days on a single row
    def itd_scalar_fn(days):
        if pd.isna(days):
            return None
        else:
            return pd.Timedelta(days=days)

    itd_answer = vectorized_sol((days,), itd_scalar_fn, "timedelta64[ns]")
    check_func(
        impl,
        (days,),
        py_output=itd_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_last_day(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.last_day(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.last_day(arr)

    # Simulates LAST_DAY on a single row
    def last_day_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem + pd.tseries.offsets.MonthEnd(n=0, normalize=True)

    last_day_answer = vectorized_sol((dates_scalar_vector,), last_day_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=last_day_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(
            pd.Series(
                [
                    pd.Timestamp("2015-01-31", tz="US/Pacific"),
                    pd.Timestamp("2016-02-29 01:36:01.737418240", tz="US/Pacific"),
                    pd.Timestamp("2017-03-27 08:43:00", tz="US/Pacific"),
                    None,
                    pd.Timestamp("2018-04-25 00:00:21", tz="US/Pacific"),
                    pd.Timestamp("2019-05-23 16:00:00", tz="US/Pacific"),
                    pd.Timestamp("2020-06-21 05:40:01", tz="US/Pacific"),
                    None,
                    pd.Timestamp("2021-07-19", tz="US/Pacific"),
                    pd.Timestamp("2022-08-17 07:48:01.254654976", tz="US/Pacific"),
                    pd.Timestamp("2023-09-15 08:00:00", tz="US/Pacific"),
                    None,
                    pd.Timestamp("2024-10-13 00:00:45.511627776", tz="US/Pacific"),
                    pd.Timestamp("2025-11-11 16:15:00", tz="US/Pacific"),
                    pd.Timestamp("2026-12-09 11:16:01.467226624", tz="US/Pacific"),
                    None,
                    pd.Timestamp("2017-02-1", tz="US/Pacific"),
                    pd.Timestamp("2020-03-1", tz="US/Pacific"),
                    pd.Timestamp("2024-11-1", tz="US/Pacific"),
                ]
            ),
            id="vector-pacific",
        ),
        pytest.param(pd.Timestamp("2022-3-1", tz="Poland"), id="scalar-poland"),
    ],
)
def test_last_day_tz_aware(arr, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.last_day(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(arr, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.last_day(arr)

    # Simulates LAST_DAY on a single row
    def last_day_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem + pd.tseries.offsets.MonthEnd(n=0, normalize=True)

    last_day_answer = vectorized_sol((arr,), last_day_scalar_fn, None)
    check_func(
        impl,
        (arr,),
        py_output=last_day_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array([2001, 2002, 2003, 2004, 2005, None, 2007])),
                pd.Series(pd.array([None, 32, 90, 180, 150, 365, 225])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                2007,
                pd.Series(pd.array([1, 10, 40, None, 80, 120, 200, 350, 360, None])),
            ),
            id="scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param((2018, 300), id="all_scalar"),
    ],
)
def test_makedate(args, memory_leak_check):
    def impl(year, day):
        return pd.Series(bodo.libs.bodosql_array_kernels.makedate(year, day))

    # Avoid pd.Series() conversion for scalar output
    if isinstance(args[0], int) and isinstance(args[1], int):
        impl = lambda year, day: bodo.libs.bodosql_array_kernels.makedate(year, day)

    # Simulates MAKEDATE on a single row
    def makedate_scalar_fn(year, day):
        if pd.isna(year) or pd.isna(day):
            return None
        else:
            return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(
                day - 1, unit="D"
            )

    makedate_answer = vectorized_sol(args, makedate_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=makedate_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "seconds",
    [
        pytest.param(
            pd.Series(pd.array([0, 1, -2, 4, 8, None, -32, 100000])),
            id="vector",
        ),
        pytest.param(
            42,
            id="scalar",
        ),
    ],
)
def test_second_timestamp(seconds, memory_leak_check):
    def impl(seconds):
        return pd.Series(bodo.libs.bodosql_array_kernels.second_timestamp(seconds))

    # Avoid pd.Series() conversion for scalar output
    if isinstance(seconds, int):
        impl = lambda seconds: bodo.libs.bodosql_array_kernels.second_timestamp(seconds)

    # Simulates second_timestamp on a single row
    def second_scalar_fn(seconds):
        if pd.isna(seconds):
            return None
        else:
            return pd.Timestamp(seconds, unit="s")

    second_answer = vectorized_sol((seconds,), second_scalar_fn, "datetime64[ns]")
    check_func(
        impl,
        (seconds,),
        py_output=second_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_monthname(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.monthname(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.monthname(arr)

    # Simulates MONTHNAME on a single row
    def monthname_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.month_name()

    monthname_answer = vectorized_sol((dates_scalar_vector,), monthname_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=monthname_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "arg",
    [
        pd.Timestamp("2007-10-07"),
        pd.Series(
            [None] * 2
            + list(pd.date_range("2020-10-01", freq="11D", periods=30))
            + [None]
        ).values,
    ],
)
def test_to_days(arg, memory_leak_check):
    args = (arg,)

    def impl(arg):
        return pd.Series(bodo.libs.bodosql_array_kernels.to_days(arg))

    # avoid Series conversion for scalar output
    if isinstance(arg, pd.Timestamp):
        impl = lambda arg: bodo.libs.bodosql_array_kernels.to_days(arg)

    # Simulates to_days on a single row
    def to_days_scalar_fn(dt64):
        if pd.isna(dt64):
            return None
        else:
            # Handle the scalar Timestamp case.
            if isinstance(dt64, pd.Timestamp):
                dt64 = dt64.value
            return (np.int64(dt64) // 86400000000000) + 719528

    to_days_answer = vectorized_sol((arg,), to_days_scalar_fn, pd.Int64Dtype())
    check_func(
        impl,
        args,
        py_output=to_days_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "arg",
    [
        pd.Timestamp("2007-10-07"),
        pd.Timestamp("2007-10-07", tz="US/Pacific"),
        pd.Series(
            [None] * 2
            + list(pd.date_range("2020-10-01", freq="11D", periods=30))
            + [None]
        ).values,
        pd.Series(
            [None] * 2
            + list(pd.date_range("2020-10-01", freq="11D", periods=30, tz="US/Pacific"))
            + [None]
        ).array,
    ],
)
def test_to_seconds(arg, memory_leak_check):
    args = (arg,)

    def impl(arg):
        return pd.Series(bodo.libs.bodosql_array_kernels.to_seconds(arg))

    # avoid Series conversion for scalar output
    if isinstance(arg, pd.Timestamp):
        impl = lambda arg: bodo.libs.bodosql_array_kernels.to_seconds(arg)

    # Simulates to_seconds on a single row
    def to_seconds_scalar_fn(dt64):
        if pd.isna(dt64):
            return None
        else:
            # Handle the scalar Timestamp case.
            if isinstance(dt64, pd.Timestamp):
                dt64 = dt64.value
            return (np.int64(dt64) // 1000000000) + 62167219200

    to_seconds_answer = vectorized_sol((arg,), to_seconds_scalar_fn, pd.Int64Dtype())
    check_func(
        impl,
        args,
        py_output=to_seconds_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "arg",
    [
        733321,
        pd.Series(
            [None] * 2 + list(np.arange(738064, 738394, 11)) + [None], dtype="Int64"
        ).array,
    ],
)
def test_from_days(arg, memory_leak_check):
    args = (arg,)

    def impl(arg):
        return pd.Series(bodo.libs.bodosql_array_kernels.from_days(arg))

    # avoid Series conversion for scalar output
    if isinstance(arg, int):
        impl = lambda arg: bodo.libs.bodosql_array_kernels.from_days(arg)

    # Simulates from_days on a single row
    def from_days_scalar_fn(int_val):
        if pd.isna(int_val):
            return None
        else:
            return pd.Timestamp((int_val - 719528) * 86400000000000)

    from_days_answer = vectorized_sol((arg,), from_days_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=from_days_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.date_range("2018-01-01", "2019-01-01", periods=20)),
                pd.Series(["su"] * 20),
            )
        ),
        pytest.param(
            (
                pd.Series(pd.date_range("2019-01-01", "2020-01-01", periods=21)),
                pd.Series(["mo", "tu", "we", "th", "fr", "sa", "su"] * 3),
            )
        ),
    ],
)
def test_next_previous_day(args, memory_leak_check):
    def next_impl(arr0, arr1):
        return pd.Series(bodo.libs.bodosql_array_kernels.next_day(arr0, arr1))

    def prev_impl(arr0, arr1):
        return pd.Series(bodo.libs.bodosql_array_kernels.previous_day(arr0, arr1))

    dow_map = {"mo": 0, "tu": 1, "we": 2, "th": 3, "fr": 4, "sa": 5, "su": 6}
    # Simulates next/previous_day on a single row
    def next_prev_day_scalar_fn(is_prev=False):
        mlt = -1 if is_prev else 1

        def impl(ts, day):
            if pd.isna(ts) or pd.isna(day):
                return None
            else:
                return pd.Timestamp(
                    (
                        ts
                        + mlt
                        * pd.Timedelta(
                            days=7 - ((mlt * (ts.dayofweek - dow_map[day])) % 7)
                        )
                    ).date()
                )

        return impl

    next_day_answer = vectorized_sol(
        args, next_prev_day_scalar_fn(), np.datetime64, manual_coercion=True
    )
    check_func(
        next_impl,
        args,
        py_output=next_day_answer,
        check_dtype=False,
        reset_index=True,
    )
    previous_day_answer = vectorized_sol(
        args, next_prev_day_scalar_fn(True), np.datetime64, manual_coercion=True
    )
    check_func(
        prev_impl,
        args,
        py_output=previous_day_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_weekday(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.weekday(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.weekday(arr)

    # Simulates WEEKDAY on a single row
    def weekday_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.weekday()

    weekday_answer = vectorized_sol((dates_scalar_vector,), weekday_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=weekday_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_yearofweekiso(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.yearofweekiso(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.yearofweekiso(arr)

    # Simulates YEAROFWEEKISO on a single row
    def yearofweekiso_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.isocalendar()[0]

    yearofweekiso_answer = vectorized_sol(
        (dates_scalar_vector,), yearofweekiso_scalar_fn, None
    )
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=yearofweekiso_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.tz_aware
@pytest.mark.parametrize(
    "ts_val",
    [
        pd.Timestamp("2022-11-07 04:23:12", tz="US/Pacific"),
        pd.Series(
            [None] * 4
            + list(
                pd.date_range("1/1/2022", periods=30, freq="7D6H7s", tz="US/Pacific")
            )
            + [None] * 2
        ),
    ],
)
def test_tz_aware_interval_add_date_offset(ts_val, memory_leak_check):
    """
    Tests tz_aware_interval_add with a date_offset as the interval.
    """

    def impl(ts_val, date_offset):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.tz_aware_interval_add(ts_val, date_offset)
        )

    # avoid pd.Series() conversion for scalar output
    if isinstance(ts_val, pd.Timestamp):
        impl = lambda ts_val, date_offset: bodo.libs.bodosql_array_kernels.tz_aware_interval_add(
            ts_val, date_offset
        )

    date_offset = pd.DateOffset(months=-2)

    # Simulates the add on a single row
    def tz_aware_interval_add_scalar_fn(ts_val, date_offset):
        if pd.isna(ts_val):
            return None
        else:
            return ts_val + date_offset

    answer = vectorized_sol(
        (ts_val, date_offset), tz_aware_interval_add_scalar_fn, None
    )
    check_func(
        impl,
        (ts_val, date_offset),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.tz_aware
@pytest.mark.parametrize(
    "ts_val",
    [
        pd.Timestamp("2022-11-07 04:23:12", tz="US/Pacific"),
        pd.Series(
            [None] * 4
            + list(
                pd.date_range("1/1/2022", periods=30, freq="7D6H7s", tz="US/Pacific")
            )
            + [None] * 2
        ),
    ],
)
def test_tz_aware_interval_add_timedelta(ts_val, memory_leak_check):
    """
    Tests tz_aware_interval_add with a timedelta as the interval.
    """

    def impl(ts_val, timedelta):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.tz_aware_interval_add(ts_val, timedelta)
        )

    # avoid pd.Series() conversion for scalar output
    if isinstance(ts_val, pd.Timestamp):
        impl = lambda ts_val, timedelta: bodo.libs.bodosql_array_kernels.tz_aware_interval_add(
            ts_val, timedelta
        )

    # Note we assume the days as the only unit in the expected output
    timedelta = pd.Timedelta(days=2)

    # Simulates the add on a single row
    def tz_aware_interval_add_scalar_fn(ts_val, timedelta):
        if pd.isna(ts_val):
            return None
        else:
            # First compute the day movement.
            new_ts = ts_val.normalize() + timedelta
            # Now restore the fields
            return pd.Timestamp(
                year=new_ts.year,
                month=new_ts.month,
                day=new_ts.day,
                hour=ts_val.hour,
                minute=ts_val.minute,
                second=ts_val.second,
                microsecond=ts_val.microsecond,
                nanosecond=ts_val.nanosecond,
                tz=new_ts.tz,
            )

    answer = vectorized_sol((ts_val, timedelta), tz_aware_interval_add_scalar_fn, None)
    check_func(
        impl,
        (ts_val, timedelta),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "part",
    [
        "quarter",
        "yyyy",
        "week",
        "mm",
        "days",
        "hour",
        "minute",
        "S",
        "ms",
        "us",
        "nsecond",
    ],
)
@pytest.mark.parametrize(
    "ts_input",
    [
        pytest.param(
            pd.Timestamp(
                year=2022,
                month=11,
                day=6,
                hour=11,
                minute=4,
                second=12,
                microsecond=241,
                nanosecond=31,
            ),
            id="scalar-tz-naive",
        ),
        pytest.param(
            pd.Timestamp(
                year=2022,
                month=11,
                day=6,
                hour=11,
                minute=4,
                second=12,
                microsecond=241,
                nanosecond=31,
                tz="Australia/Lord_Howe",
            ),
            id="scalar-tz-aware",
        ),
        pytest.param(
            pd.Series(
                [None] * 4
                + list(
                    pd.date_range(
                        "2022-11-6 04:12:41.432433", periods=20, freq="11D3H5us"
                    )
                )
                + [None] * 2
            ).values,
            id="vector-tz-naive",
        ),
        pytest.param(
            pd.Series(
                [None] * 4
                + list(
                    pd.date_range(
                        "2022-11-6 04:12:41.432433",
                        periods=20,
                        freq="11D3H5us",
                        tz="US/Pacific",
                    )
                )
                + [None] * 2
            ).array,
            id="vector-tz-aware",
        ),
    ],
)
def test_date_trunc(part, ts_input, memory_leak_check):
    """
    Tests date_trunc array kernel on various inputs, testing all the different code paths
    in the generated kernel.
    """

    def impl(part, arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.date_trunc(part, arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(ts_input, pd.Timestamp):
        impl = lambda part, arr: bodo.libs.bodosql_array_kernels.date_trunc(part, arr)

    # Simulates date_trunc on a single row
    def date_trunc_scalar_fn(part, ts_input):
        return generate_date_trunc_func(part)(ts_input)

    answer = vectorized_sol((part, ts_input), date_trunc_scalar_fn, None)
    check_func(
        impl,
        (part, ts_input),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


def test_add_interval_optional(memory_leak_check):
    def impl(tz_naive_ts, tz_aware_ts, int_val, flag0, flag1):
        arg0 = tz_naive_ts if flag0 else None
        arg1 = tz_aware_ts if flag0 else None
        arg2 = int_val if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.add_interval_months(arg2, arg0),
            bodo.libs.bodosql_array_kernels.add_interval_days(arg2, arg1),
            bodo.libs.bodosql_array_kernels.add_interval_hours(arg2, arg0),
            bodo.libs.bodosql_array_kernels.add_interval_minutes(arg2, arg1),
        )

    A = pd.Timestamp("2018-04-01")
    B = pd.Timestamp("2023-11-05 00:45:00", tz="US/Mountain")
    C = 240
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            a0 = pd.Timestamp("2038-04-01") if flag0 and flag1 else None
            a1 = (
                pd.Timestamp("2024-7-02 00:45:00", tz="US/Mountain")
                if flag0 and flag1
                else None
            )
            a2 = pd.Timestamp("2018-04-11") if flag0 and flag1 else None
            a3 = (
                pd.Timestamp("2023-11-05 04:45:00", tz="US/Mountain")
                if flag0 and flag1
                else None
            )
            check_func(
                impl,
                (A, B, C, flag0, flag1),
                py_output=(a0, a1, a2, a3),
            )


@pytest.mark.slow
def test_calendar_optional(memory_leak_check):
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return (
            bodo.libs.bodosql_array_kernels.last_day(arg0),
            bodo.libs.bodosql_array_kernels.dayname(arg0),
            bodo.libs.bodosql_array_kernels.monthname(arg0),
            bodo.libs.bodosql_array_kernels.weekday(arg0),
            bodo.libs.bodosql_array_kernels.yearofweekiso(arg0),
            bodo.libs.bodosql_array_kernels.makedate(arg1, arg2),
            bodo.libs.bodosql_array_kernels.dayofweek(arg0),
            bodo.libs.bodosql_array_kernels.dayofmonth(arg0),
            bodo.libs.bodosql_array_kernels.dayofyear(arg0),
        )

    A, B, C = pd.Timestamp("2018-04-01"), 2005, 365
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                a0 = pd.Timestamp("2018-04-30") if flag0 else None
                a1 = "Sunday" if flag0 else None
                a2 = "April" if flag0 else None
                a3 = 6 if flag0 else None
                a4 = 2018 if flag0 else None
                a5 = pd.Timestamp("2005-12-31") if flag1 and flag2 else None
                a6 = 0 if flag0 else None
                a7 = 1 if flag0 else None
                a8 = 91 if flag0 else None
                check_func(
                    impl,
                    (A, B, C, flag0, flag1, flag2),
                    py_output=(a0, a1, a2, a3, a4, a5, a6, a7, a8),
                )


@pytest.mark.slow
def test_option_timestamp(memory_leak_check):
    def impl(A, flag):
        arg = A if flag else None
        return bodo.libs.bodosql_array_kernels.second_timestamp(arg)

    for flag in [True, False]:
        A = pd.Timestamp(1000000, unit="s") if flag else None
        check_func(
            impl,
            (1000000, flag),
            py_output=A,
        )


@pytest.mark.slow
def test_option_int_to_days(memory_leak_check):
    def impl(A, flag):
        arg = A if flag else None
        return bodo.libs.bodosql_array_kernels.int_to_days(arg)

    for flag in [True, False]:
        answer = pd.Timedelta(days=10) if flag else None
        check_func(impl, (10, flag), py_output=answer)


@pytest.mark.slow
def test_option_to_days(memory_leak_check):
    def impl(A, flag):
        arg = A if flag else None
        return bodo.libs.bodosql_array_kernels.to_days(arg)

    for flag in [True, False]:
        answer = 733042 if flag else None
        check_func(
            impl,
            (pd.Timestamp("2007-01-01"), flag),
            py_output=answer,
        )


@pytest.mark.slow
def test_option_from_days(memory_leak_check):
    def impl(A, flag):
        arg = A if flag else None
        return bodo.libs.bodosql_array_kernels.from_days(arg)

    for flag in [True, False]:
        answer = pd.Timestamp("2007-01-01") if flag else None
        check_func(
            impl,
            (733042, flag),
            py_output=answer,
        )


@pytest.mark.slow
def test_option_to_seconds(memory_leak_check):
    def impl(A, flag):
        arg = A if flag else None
        return bodo.libs.bodosql_array_kernels.to_seconds(arg)

    for flag in [True, False]:
        answer = 63334828800 if flag else None
        check_func(
            impl,
            (pd.Timestamp("2007-01-01"), flag),
            py_output=answer,
        )


@pytest.mark.tz_aware
@pytest.mark.slow
def test_option_tz_aware_interval_add_date_offset(memory_leak_check):
    """
    Tests tz_aware_interval_add optional support with a date_offset as the interval.
    """

    def impl(ts_val, date_offset, flag0, flag1):
        arg0 = ts_val if flag0 else None
        arg1 = date_offset if flag1 else None
        return bodo.libs.bodosql_array_kernels.tz_aware_interval_add(arg0, arg1)

    ts_val = pd.Timestamp("2022-11-05", tz="US/Pacific")
    date_offset = pd.DateOffset(months=2)
    expected_add = ts_val + date_offset

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = expected_add if flag0 and flag1 else None
            check_func(
                impl,
                (ts_val, date_offset, flag0, flag1),
                py_output=answer,
            )


@pytest.mark.tz_aware
@pytest.mark.slow
def test_option_tz_aware_interval_add_timedelta(memory_leak_check):
    """
    Tests tz_aware_interval_add optional support with a timedelta as the interval.
    """

    def impl(ts_val, date_offset, flag0, flag1):
        arg0 = ts_val if flag0 else None
        arg1 = date_offset if flag1 else None
        return bodo.libs.bodosql_array_kernels.tz_aware_interval_add(arg0, arg1)

    ts_val = pd.Timestamp("2022-11-05 04:23:12", tz="US/Pacific")
    date_offset = pd.Timedelta(days=2)
    expected_add = pd.Timestamp("2022-11-07 04:23:12", tz="US/Pacific")

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = expected_add if flag0 and flag1 else None
            check_func(
                impl,
                (ts_val, date_offset, flag0, flag1),
                py_output=answer,
            )


@pytest.mark.slow
def test_option_date_trunc(memory_leak_check):
    """
    Tests date_trunc array kernel on optional inputs
    """

    def impl(part, ts, flag0, flag1):
        arg0 = part if flag0 else None
        arg1 = ts if flag1 else None
        return bodo.libs.bodosql_array_kernels.date_trunc(arg0, arg1)

    part = "day"
    ts = pd.Timestamp("2022-11-6 12:40:45", tz="US/Pacific")

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (
                pd.Timestamp("2022-11-6", tz="US/Pacific") if flag0 and flag1 else None
            )
            check_func(
                impl,
                (part, ts, flag0, flag1),
                py_output=answer,
            )
