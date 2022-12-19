# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL date/time functions
"""


import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
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
def test_add_interval_date_units(unit, args, answers):
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
def test_add_interval_time_units(unit, args, answers):
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


def test_dayname(dates_scalar_vector):
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


def test_dayofmonth(dates_scalar_vector):
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


def test_dayofweek(dates_scalar_vector):
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


def test_dayofweekiso(dates_scalar_vector):
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


def test_dayofyear(dates_scalar_vector):
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
            pd.Series(pd.array([0, 1, -2, 4, 8, None, -32, 10000])),
            id="vector",
        ),
        pytest.param(
            42,
            id="scalar",
        ),
    ],
)
def test_day_timestamp(days):
    def impl(days):
        return pd.Series(bodo.libs.bodosql_array_kernels.day_timestamp(days))

    # avoid pd.Series() conversion for scalar output
    if isinstance(days, int):
        impl = lambda days: bodo.libs.bodosql_array_kernels.day_timestamp(days)

    # Simulates day_timestamp on a single row
    def days_scalar_fn(days):
        if pd.isna(days):
            return None
        else:
            return pd.Timestamp(days, unit="D")

    days_answer = vectorized_sol((days,), days_scalar_fn, "datetime64[ns]")
    check_func(
        impl,
        (days,),
        py_output=days_answer,
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
def test_int_to_days(days):
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


def test_last_day(dates_scalar_vector):
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
def test_makedate(args):
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
def test_second_timestamp(seconds):
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


def test_monthname(dates_scalar_vector):
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
    "args",
    [
        pytest.param(
            (
                pd.concat(
                    [
                        pd.Series(
                            [
                                pd.Timestamp(d)
                                for d in pd.date_range(
                                    "2018-01-01", "2019-01-01", periods=20
                                )
                            ]
                            + [None, None]
                        )
                    ]
                ),
                pd.Series(pd.date_range("2005-01-01", "2020-01-01", periods=22)),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        pd.Timestamp(d)
                        for d in pd.date_range("2018-01-01", "2019-01-01", periods=20)
                    ]
                    + [None, None]
                ),
                pd.Timestamp("2018-06-05"),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Timestamp("2000-10-29"), pd.Timestamp("1992-03-25")), id="all_scalar"
        ),
    ],
)
def test_month_diff(args):
    def impl(arr0, arr1):
        return pd.Series(bodo.libs.bodosql_array_kernels.month_diff(arr0, arr1))

    # avoid pd.Series() conversion for scalar output
    if not isinstance(args[0], pd.Series) and not isinstance(args[1], pd.Series):
        impl = lambda arr0, arr1: bodo.libs.bodosql_array_kernels.month_diff(arr0, arr1)

    # Simulates month diff on a single row
    def md_scalar_fn(ts1, ts2):
        if pd.isna(ts1) or pd.isna(ts2):
            return None
        else:
            floored_delta = (ts1.year - ts2.year) * 12 + (ts1.month - ts2.month)
            remainder = ((ts1 - pd.DateOffset(months=floored_delta)) - ts2).value
            remainder = 1 if remainder > 0 else (-1 if remainder < 0 else 0)
            if floored_delta > 0 and remainder < 0:
                actual_month_delta = floored_delta - 1
            elif floored_delta < 0 and remainder > 0:
                actual_month_delta = floored_delta + 1
            else:
                actual_month_delta = floored_delta
            return -actual_month_delta

    days_answer = vectorized_sol(args, md_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        args,
        py_output=days_answer,
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
def test_next_previous_day(args):
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


def test_weekday(dates_scalar_vector):
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


def test_yearofweekiso(dates_scalar_vector):
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


@pytest.mark.slow
def test_calendar_optional():
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
def test_option_timestamp():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.second_timestamp(arg0),
            bodo.libs.bodosql_array_kernels.day_timestamp(arg1),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            A0 = pd.Timestamp(1000000, unit="s") if flag0 else None
            A1 = pd.Timestamp(10000, unit="D") if flag1 else None
            check_func(
                impl,
                (1000000, 10000, flag0, flag1),
                py_output=(A0, A1),
            )


@pytest.mark.slow
def test_option_int_to_days():
    def impl(A, flag):
        arg = A if flag else None
        return bodo.libs.bodosql_array_kernels.int_to_days(arg)

    for flag in [True, False]:
        answer = pd.Timedelta(days=10) if flag else None
        check_func(impl, (10, flag), py_output=answer)


@pytest.mark.slow
def test_option_month_diff():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.month_diff(arg0, arg1)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = 42 if flag0 and flag1 else None
            check_func(
                impl,
                (pd.Timestamp("2007-01-01"), pd.Timestamp("2010-07-04"), flag0, flag1),
                py_output=answer,
            )
