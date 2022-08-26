# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL date/time functions
"""


import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param(
            pd.concat(
                [
                    pd.Series(pd.date_range("2018-01-01", "2019-01-01", periods=20)),
                    pd.Series([None, None]),
                    pd.Series(pd.date_range("1970-01-01", "2108-01-01", periods=20)),
                ]
            ),
            id="vector",
        ),
        pytest.param(pd.Timestamp("2000-10-29"), id="scalar"),
    ],
)
def dates_scalar_vector(request):
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

    days_answer = vectorized_sol(
        (days,), days_scalar_fn, np.datetime64, manual_coercion=True
    )
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
            return np.timedelta64(days, "D")

    itd_answer = vectorized_sol(
        (days,), itd_scalar_fn, np.timedelta64, manual_coercion=True
    )
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
            return np.datetime64(
                elem + pd.tseries.offsets.MonthEnd(n=0, normalize=True)
            )

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
            return np.datetime64(
                pd.Timestamp(year=year, month=1, day=1)
                + pd.Timedelta(day - 1, unit="D")
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

    second_answer = vectorized_sol(
        (seconds,), second_scalar_fn, np.datetime64, manual_coercion=True
    )
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
                            pd.date_range("2018-01-01", "2019-01-01", periods=20)
                        ),
                        pd.Series([None, None]),
                    ]
                ),
                pd.Series(pd.date_range("2005-01-01", "2020-01-01", periods=22)),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(pd.date_range("2018-01-01", "2019-01-01", periods=20)),
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
        )

    A, B, C = pd.Timestamp("2018-04-01"), 2005, 365
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                a0 = np.datetime64("2018-04-30") if flag0 else None
                a1 = "Sunday" if flag0 else None
                a2 = "April" if flag0 else None
                a3 = 6 if flag0 else None
                a4 = 2018 if flag0 else None
                a5 = np.datetime64("2005-12-31") if flag1 and flag2 else None
                check_func(
                    impl,
                    (A, B, C, flag0, flag1, flag2),
                    py_output=(a0, a1, a2, a3, a4, a5),
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
            A0 = np.datetime64(pd.Timestamp(1000000, unit="s")) if flag0 else None
            A1 = np.datetime64(pd.Timestamp(10000, unit="D")) if flag1 else None
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
        answer = np.timedelta64(pd.Timedelta(days=10)) if flag else None
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
