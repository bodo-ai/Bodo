# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
    Test File for pd.tseries.offsets types.
"""
import datetime

import pandas as pd
import pytest

import bodo
from bodo.pandas_compat import pandas_version
from bodo.tests.utils import check_func


@pytest.fixture(params=[0, -1, 3, -4])
def offset_multiplier(request):
    """returns an integer value, to be used with testing offset multiplication"""
    return request.param


@pytest.fixture(
    params=[
        pd.tseries.offsets.Week(),
        pytest.param(
            pd.tseries.offsets.Week(n=4),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.Week(n=4, normalize=True, weekday=5),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.Week(n=2, normalize=False, weekday=0),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.Week(n=-1, normalize=False), marks=pytest.mark.slow
        ),
    ]
)
def week_value(request):
    return request.param


@pytest.mark.slow
def test_week_offset(week_value, memory_leak_check):
    def test_impl():
        return week_value

    check_func(test_impl, ())


@pytest.mark.slow
def test_week_boxing(week_value, memory_leak_check):
    """
    Test boxing and unboxing of pd.tseries.offsets.Week()
    """

    def test_impl(me_obj):
        return me_obj

    check_func(test_impl, (week_value,))


@pytest.mark.slow
def test_week_constructor(memory_leak_check):
    def test_impl1():
        return pd.tseries.offsets.Week()

    def test_impl2():
        return pd.tseries.offsets.Week(n=4, weekday=2)

    def test_impl3():
        return pd.tseries.offsets.Week(n=-2)

    check_func(test_impl1, ())
    check_func(test_impl2, ())
    check_func(test_impl3, ())


def test_week_add_datetime(week_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=30, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (week_value, datetime_val))
    check_func(test_impl, (datetime_val, week_value))


def test_week_add_timestamp(week_value, memory_leak_check):
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
    check_func(test_impl, (week_value, timestamp_val))
    check_func(test_impl, (timestamp_val, week_value))


def test_week_sub_datetime(week_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=30, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (datetime_val, week_value))


@pytest.mark.smoke
def test_week_sub_timestamp(week_value, memory_leak_check):
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
    check_func(test_impl, (timestamp_val, week_value))


@pytest.mark.slow
def test_week_neg(week_value, memory_leak_check):
    def test_impl(me):
        return -me

    check_func(test_impl, (week_value,))


@pytest.mark.slow
def test_week_mul_int(memory_leak_check, week_value, offset_multiplier):

    # Objects won't match exactly, so test mul by checking that addition in Python
    # has the same result
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

    def test_mul(a, b):
        return (a * b) + timestamp_val

    check_func(test_mul, (offset_multiplier, week_value))
    check_func(test_mul, (week_value, offset_multiplier))


@pytest.fixture(
    params=[
        pd.tseries.offsets.MonthBegin(),
        pytest.param(
            pd.tseries.offsets.MonthBegin(n=4, normalize=False),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.tseries.offsets.MonthBegin(n=4, normalize=True),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.tseries.offsets.MonthBegin(n=-2), marks=pytest.mark.slow),
        pytest.param(
            pd.tseries.offsets.MonthBegin(n=-1, normalize=True), marks=pytest.mark.slow
        ),
        pytest.param(
            pd.tseries.offsets.MonthBegin(n=0),
            marks=pytest.mark.slow,
        ),
    ]
)
def month_begin_value(request):
    return request.param


@pytest.mark.slow
def test_constant_lowering_month_begin(month_begin_value, memory_leak_check):
    def test_impl():
        return month_begin_value

    check_func(test_impl, ())


@pytest.mark.slow
def test_month_end_boxing(month_begin_value, memory_leak_check):
    """
    Test boxing and unboxing of pd.tseries.offsets.MonthBegin()
    """

    def test_impl(mb_obj):
        return mb_obj

    check_func(test_impl, (month_begin_value,))


@pytest.mark.slow
def test_month_begin_constructor(memory_leak_check):
    def test_impl1():
        return pd.tseries.offsets.MonthBegin()

    def test_impl2():
        return pd.tseries.offsets.MonthBegin(n=4, normalize=True)

    def test_impl3():
        return pd.tseries.offsets.MonthBegin(-2)

    check_func(test_impl1, ())
    check_func(test_impl2, ())
    check_func(test_impl3, ())


@pytest.mark.slow
def test_month_begin_n(month_begin_value, memory_leak_check):
    def test_impl(mb_obj):
        return mb_obj.n

    check_func(test_impl, (month_begin_value,))


@pytest.mark.slow
def test_month_begin_normalize(month_begin_value, memory_leak_check):
    def test_impl(mb_obj):
        return mb_obj.normalize

    check_func(test_impl, (month_begin_value,))


def test_month_begin_add_datetime(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=30, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (month_begin_value, datetime_val))
    check_func(test_impl, (datetime_val, month_begin_value))


@pytest.mark.slow
def test_month_begin_add_datetime_boundary(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=1, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (month_begin_value, datetime_val))
    check_func(test_impl, (datetime_val, month_begin_value))


def test_month_begin_add_timestamp(month_begin_value, memory_leak_check):
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
    check_func(test_impl, (month_begin_value, timestamp_val))
    check_func(test_impl, (timestamp_val, month_begin_value))


@pytest.mark.slow
def test_month_begin_add_timestamp_boundary(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    timestamp_val = pd.Timestamp(
        year=2020,
        month=10,
        day=1,
        hour=22,
        minute=12,
        second=45,
        microsecond=99320,
        nanosecond=891,
    )
    check_func(test_impl, (month_begin_value, timestamp_val))
    check_func(test_impl, (timestamp_val, month_begin_value))


def test_month_begin_add_date_timestamp(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=30,
    )
    check_func(test_impl, (month_begin_value, date_val))
    check_func(test_impl, (date_val, month_begin_value))


@pytest.mark.slow
def test_month_begin_add_date_boundary(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=1,
    )
    check_func(test_impl, (month_begin_value, date_val))
    check_func(test_impl, (date_val, month_begin_value))


def test_month_begin_add_series(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    S = pd.Series(pd.date_range(start="2018-04-24", end="2020-04-29", periods=5))
    check_func(test_impl, (month_begin_value, S))
    check_func(test_impl, (S, month_begin_value))


def test_month_begin_sub_datetime(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=30, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (datetime_val, month_begin_value))


@pytest.mark.slow
def test_month_begin_sub_datetime_boundary(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=1, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (datetime_val, month_begin_value))


@pytest.mark.smoke
def test_month_begin_sub_timestamp(month_begin_value, memory_leak_check):
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
    check_func(test_impl, (timestamp_val, month_begin_value))


@pytest.mark.slow
def test_month_begin_sub_timestamp_boundary(month_begin_value, memory_leak_check):
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
    check_func(test_impl, (timestamp_val, month_begin_value))


def test_month_begin_sub_date_timestamp(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=30,
    )
    check_func(test_impl, (date_val, month_begin_value))


@pytest.mark.slow
def test_month_begin_sub_date_boundary(month_begin_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=1,
    )
    check_func(test_impl, (date_val, month_begin_value))


def test_month_begin_sub_series(month_begin_value, memory_leak_check):
    def test_impl(S, val):
        return S - val

    S = pd.Series(pd.date_range(start="2018-04-24", end="2020-04-29", periods=5))
    check_func(test_impl, (S, month_begin_value))


@pytest.mark.slow
def test_month_begin_neg(month_begin_value, memory_leak_check):
    def test_impl(mb):
        return -mb

    check_func(test_impl, (month_begin_value,))


@pytest.mark.slow
def test_month_begin_mul_int(month_begin_value, offset_multiplier, memory_leak_check):

    # Objects won't match exactly, so test mul by checking that addition in Python
    # has the same result
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

    def test_mul(a, b):
        return (a * b) + timestamp_val

    check_func(test_mul, (offset_multiplier, month_begin_value))
    check_func(test_mul, (month_begin_value, offset_multiplier))


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
        pytest.param(
            pd.tseries.offsets.MonthEnd(n=0),
            marks=pytest.mark.slow,
        ),
    ]
)
def month_end_value(request):
    return request.param


@pytest.mark.slow
def test_constant_lowering_month_end(month_end_value, memory_leak_check):
    def test_impl():
        return month_end_value

    check_func(test_impl, ())


@pytest.mark.slow
def test_month_end_boxing(month_end_value, memory_leak_check):
    """
    Test boxing and unboxing of pd.tseries.offsets.MonthEnd()
    """

    def test_impl(me_obj):
        return me_obj

    check_func(test_impl, (month_end_value,))


@pytest.mark.slow
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


@pytest.mark.smoke
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


@pytest.mark.slow
def test_month_end_neg(month_end_value, memory_leak_check):
    def test_impl(me):
        return -me

    check_func(test_impl, (month_end_value,))


@pytest.mark.slow
def test_month_end_mul_int(month_end_value, offset_multiplier, memory_leak_check):

    # Objects won't match exactly, so test mul by checking that addition in Python
    # has the same result
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

    def test_mul(a, b):
        return (a * b) + timestamp_val

    check_func(test_mul, (offset_multiplier, month_end_value))
    check_func(test_mul, (month_end_value, offset_multiplier))


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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_date_offset_constructor(memory_leak_check):
    # Objects won't match exactly, so test boxing by checking that addition in Python
    # has the same result
    def test_impl1(n, normalize, years, year):
        return pd.tseries.offsets.DateOffset(n, normalize, years=years, year=year)

    def test_impl2(years, year):
        return pd.tseries.offsets.DateOffset(years=years, year=year)

    def test_impl3(n, nanosecond, nanoseconds):
        return pd.tseries.offsets.DateOffset(
            n, nanosecond=nanosecond, nanoseconds=nanoseconds
        )

    def test_impl4(normalize, year):
        return pd.tseries.offsets.DateOffset(normalize=normalize, year=year)

    def test_impl5(n, years):
        return pd.tseries.offsets.DateOffset(n, years=years)

    py_outputs = []
    bodo_outputs = []
    py_outputs.append(test_impl1(1, False, 5, 2000))
    bodo_outputs.append(bodo.jit(test_impl1)(1, False, 5, 2000))
    py_outputs.append(test_impl2(-1, 2018))
    bodo_outputs.append(bodo.jit(test_impl2)(-1, 2018))
    py_outputs.append(test_impl3(0, 969, -922))
    bodo_outputs.append(bodo.jit(test_impl3)(0, 969, -922))
    py_outputs.append(test_impl4(True, 2020))
    bodo_outputs.append(bodo.jit(test_impl4)(True, 2020))
    py_outputs.append(test_impl5(2, -4))
    bodo_outputs.append(bodo.jit(test_impl5)(2, -4))
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
    for i in range(len(py_outputs)):
        assert timestamp_val + py_outputs[i] == timestamp_val + bodo_outputs[i]


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


def test_date_offset_add_datetime(date_offset_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=30, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (date_offset_value, datetime_val))
    check_func(test_impl, (datetime_val, date_offset_value))


def test_date_offset_add_date(date_offset_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=31,
    )
    check_func(test_impl, (date_offset_value, date_val))
    check_func(test_impl, (date_val, date_offset_value))


def test_date_offset_add_series(date_offset_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 + val2

    S = pd.Series(pd.date_range(start="2018-04-24", end="2020-04-29", periods=5))
    check_func(test_impl, (date_offset_value, S))
    check_func(test_impl, (S, date_offset_value))


@pytest.mark.smoke
def test_date_offset_sub_timestamp(date_offset_value, memory_leak_check):
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

    if pandas_version == (1, 4) and (
        hasattr(date_offset_value, "nanoseconds") or hasattr(date_offset_value, "n")
    ):
        # TODO: fix nanosecond handling in pandas 1.4 [BE-2598]
        pass
    else:
        check_func(test_impl, (timestamp_val, date_offset_value))


def test_date_offset_sub_datetime(date_offset_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    datetime_val = datetime.datetime(
        year=2020, month=10, day=30, hour=22, minute=12, second=45, microsecond=99320
    )
    check_func(test_impl, (datetime_val, date_offset_value))


def test_date_offset_sub_date(date_offset_value, memory_leak_check):
    def test_impl(val1, val2):
        return val1 - val2

    date_val = datetime.date(
        year=2020,
        month=10,
        day=31,
    )
    check_func(test_impl, (date_val, date_offset_value))


def test_date_offset_sub_series(date_offset_value, memory_leak_check):
    def test_impl(S, val):
        return S - val

    S = pd.Series(pd.date_range(start="2018-04-24", end="2020-04-29", periods=5))
    check_func(test_impl, (S, date_offset_value))


@pytest.mark.slow
def test_date_offset_neg(date_offset_value, memory_leak_check):
    # Objects won't match exactly, so test boxing by checking that addition in Python
    # has the same result
    def test_impl(do_obj):
        return -do_obj

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


def test_date_offset_mul_int(memory_leak_check, offset_multiplier, date_offset_value):

    # Objects won't match exactly, so test mul by checking that addition in Python
    # has the same result
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

    def test_mul(a, b):
        return (a * b) + timestamp_val

    check_func(test_mul, (offset_multiplier, date_offset_value))
    check_func(test_mul, (date_offset_value, offset_multiplier))
