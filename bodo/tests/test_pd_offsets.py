# Copyright (C) 2020 Bodo Inc. All rights reserved.
""" 
    Test File for pd.tseries.offsets types.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pd.tseries.offsets.MonthEnd(),
        pytest.param(
            pd.tseries.offsets.MonthEnd(n=4, normalize=False),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.MonthEnd(n=4, normalize=True),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.tseries.offsets.MonthEnd(n=-2), marks=pytest.mark.slow),
        pytest.param(
            pd.tseries.offsets.MonthEnd(n=-1, normalize=True), marks=pytest.mark.slow
        ),
    ]
)
def month_end_value(request):
    return request.param


def test_constant_lowering_month_end(month_end_value, memory_leak_check):
    def test_impl():
        return month_end_value

    check_func(test_impl, ())


def test_month_end_boxing(month_end_value, memory_leak_check):
    """
    Test boxing and unboxing of pd.tseries.offsets.MonthEnd()
    """

    def test_impl(me_obj):
        return me_obj

    check_func(test_impl, (month_end_value,))


def test_month_end_constructor(memory_leak_check):
    def test_impl1():
        return pd.tseries.offsets.MonthEnd()

    def test_impl2():
        return pd.tseries.offsets.MonthEnd(n=4, normalize=True)

    def test_impl3():
        return pd.tseries.offsets.MonthEnd(-2)

    check_func(test_impl1, ())
    check_func(test_impl2, ())
    check_func(test_impl3, ())


def test_month_end_n(month_end_value, memory_leak_check):
    def test_impl(me_obj):
        return me_obj.n

    check_func(test_impl, (month_end_value,))


def test_month_end_normalize(month_end_value, memory_leak_check):
    def test_impl(me_obj):
        return me_obj.normalize

    check_func(test_impl, (month_end_value,))


def test_month_end_add_datetime(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=30, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (month_end_value, datetime_val))
    check_func(test_impl, (datetime_val, month_end_value))


@pytest.mark.slow
def test_month_end_add_datetime_boundary(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=31, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (month_end_value, datetime_val))
    check_func(test_impl, (datetime_val, month_end_value))


def test_month_end_add_timestamp(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    timestamp_val = pd.Timestamp(
        year=2020,
        month=10,
        day=30,
        hour=22,
        minute=12,
        second=45,
        microsecond=99320,
        nanosecond=891,
    )
    check_func(test_impl, (month_end_value, timestamp_val))
    check_func(test_impl, (timestamp_val, month_end_value))


@pytest.mark.slow
def test_month_end_add_timestamp_boundary(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    timestamp_val = pd.Timestamp(
        year=2020,
        month=10,
        day=31,
        hour=22,
        minute=12,
        second=45,
        microsecond=99320,
        nanosecond=891,
    )
    check_func(test_impl, (month_end_value, timestamp_val))
    check_func(test_impl, (timestamp_val, month_end_value))


def test_month_end_add_date_timestamp(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=30,
    )
    check_func(test_impl, (month_end_value, date_val))
    check_func(test_impl, (date_val, month_end_value))


@pytest.mark.slow
def test_month_end_add_date_boundary(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=31,
    )
    check_func(test_impl, (month_end_value, date_val))
    check_func(test_impl, (date_val, month_end_value))


def test_month_end_add_series(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    S = pd.Series(pd.date_range(start="2018-04-24", end="2020-04-29", periods=5))
    check_func(test_impl, (month_end_value, S))
    check_func(test_impl, (S, month_end_value))


def test_month_end_sub_datetime(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=30, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (datetime_val, month_end_value))


@pytest.mark.slow
def test_month_end_sub_datetime_boundary(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=31, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (datetime_val, month_end_value))


def test_month_end_sub_timestamp(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    timestamp_val = pd.Timestamp(
        year=2020,
        month=10,
        day=30,
        hour=22,
        minute=12,
        second=45,
        microsecond=99320,
        nanosecond=891,
    )
    check_func(test_impl, (timestamp_val, month_end_value))


@pytest.mark.slow
def test_month_end_sub_timestamp_boundary(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    timestamp_val = pd.Timestamp(
        year=2020,
        month=10,
        day=31,
        hour=22,
        minute=12,
        second=45,
        microsecond=99320,
        nanosecond=891,
    )
    check_func(test_impl, (timestamp_val, month_end_value))


def test_month_end_sub_date_timestamp(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=30,
    )
    check_func(test_impl, (date_val, month_end_value))


@pytest.mark.slow
def test_month_end_sub_date_boundary(month_end_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=31,
    )
    check_func(test_impl, (date_val, month_end_value))


def test_month_end_sub_series(month_end_value, memory_leak_check):
    def test_impl(S, val):
        return S - val

    S = pd.Series(pd.date_range(start="2018-04-24", end="2020-04-29", periods=5))
    check_func(test_impl, (S, month_end_value))


def test_month_end_neg(month_end_value, memory_leak_check):
    def test_impl(me):
        return -me

    check_func(test_impl, (month_end_value,))


@pytest.fixture(
    params=[
        pytest.param(
            pd.tseries.offsets.DateOffset(),
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                normalize=False,
                year=2020,
                month=10,
                day=5,
                weekday=0,
                hour=23,
                minute=55,
                second=10,
                microsecond=41423,
                nanosecond=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                normalize=True,
                year=2020,
                month=10,
                day=5,
                hour=23,
                minute=55,
                second=10,
                microsecond=41423,
                nanosecond=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                normalize=False,
                year=2020,
                month=10,
                day=5,
                hour=23,
                minute=55,
                second=10,
                microsecond=41423,
                nanosecond=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                normalize=False,
                years=-3,
                months=4,
                weeks=2,
                days=4,
                hours=23,
                minutes=55,
                seconds=10,
                microseconds=41423,
                nanoseconds=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                normalize=True,
                years=-3,
                months=4,
                weeks=2,
                days=4,
                hours=23,
                minutes=55,
                seconds=10,
                microseconds=41423,
                nanoseconds=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                normalize=False,
                years=-3,
                months=4,
                weeks=2,
                days=4,
                hours=23,
                minutes=55,
                seconds=10,
                microseconds=41423,
                nanoseconds=121,
                year=2020,
                month=10,
                day=17,
                hour=23,
                minute=55,
                second=10,
                microsecond=41423,
                nanosecond=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                normalize=True,
                years=-3,
                months=4,
                weeks=2,
                days=4,
                hours=23,
                minutes=55,
                seconds=10,
                microseconds=41423,
                nanoseconds=121,
                year=2020,
                month=10,
                day=2,
                weekday=5,
                hour=23,
                minute=55,
                second=10,
                microsecond=41423,
                nanosecond=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                n=2,
                years=-3,
                months=4,
                weeks=2,
                days=4,
                hours=23,
                minutes=55,
                seconds=10,
                microseconds=41423,
                nanoseconds=121,
                year=2020,
                month=10,
                day=2,
                weekday=0,
                hour=23,
                minute=55,
                second=10,
                microsecond=41423,
                nanosecond=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                n=-3,
                years=-3,
                months=4,
                weeks=2,
                days=4,
                hours=23,
                minutes=55,
                seconds=10,
                microseconds=41423,
                nanoseconds=121,
                year=2020,
                month=10,
                day=5,
                hour=23,
                minute=55,
                second=10,
                microsecond=41423,
                nanosecond=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                n=6,
                normalize=True,
                years=-3,
                months=4,
                weeks=2,
                days=4,
                hours=23,
                minutes=55,
                seconds=10,
                microseconds=41423,
                nanoseconds=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                n=6,
                nanoseconds=121,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.DateOffset(
                n=6,
                nanosecond=121,
            ),
            marks=pytest.mark.slow,
        ),
    ]
)
def date_offset_value(request):
    return request.param


def test_constant_lowering_date_offset(date_offset_value, memory_leak_check):
    # Objects won't match exactly, so test boxing by checking that addition in Python
    # has the same result
    def test_impl():
        return date_offset_value

    py_output = test_impl()
    bodo_output = bodo.jit(test_impl)()
    timestamp_val = pd.Timestamp(
        year=2020,
        month=10,
        day=30,
        hour=22,
        minute=12,
        second=45,
        microsecond=99320,
        nanosecond=891,
    )
    assert timestamp_val + py_output == timestamp_val + bodo_output


def test_date_offset_boxing(date_offset_value, memory_leak_check):
    """
    Test boxing and unboxing of pd.tseries.offsets.DateOffset()
    """

    # Objects won't match exactly, so test boxing by checking that addition in Python
    # has the same result
    def test_impl(do_obj):
        return do_obj

    py_output = test_impl(date_offset_value)
    bodo_output = bodo.jit(test_impl)(date_offset_value)
    timestamp_val = pd.Timestamp(
        year=2020,
        month=10,
        day=30,
        hour=22,
        minute=12,
        second=45,
        microsecond=99320,
        nanosecond=891,
    )
    assert timestamp_val + py_output == timestamp_val + bodo_output


def test_date_offset_add_timestamp(date_offset_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    timestamp_val = pd.Timestamp(
        year=2020,
        month=10,
        day=30,
        hour=22,
        minute=12,
        second=45,
        microsecond=99320,
        nanosecond=891,
    )
    check_func(test_impl, (date_offset_value, timestamp_val))
    check_func(test_impl, (timestamp_val, date_offset_value))
