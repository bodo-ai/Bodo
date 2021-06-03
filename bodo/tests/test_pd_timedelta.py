# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
    Test File for pd_timedelta types. Covers basic functionality of boxing, unboxing,
    lowering, fields and methods.
"""
import datetime

import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pd.Timedelta(232142),
        pytest.param(
            pd.Timedelta(
                days=1,
                weeks=10,
                hours=5,
                minutes=3,
                microseconds=121,
                milliseconds=787,
                seconds=2,
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(pd.Timedelta(-1000010000100), marks=pytest.mark.slow),
    ]
)
def timedelta_value(request):
    return request.param


@pytest.mark.slow
def test_constant_lowering(timedelta_value, memory_leak_check):
    def test_impl():
        return timedelta_value

    check_func(test_impl, ())


@pytest.mark.slow
def test_timedelta_boxing(timedelta_value, memory_leak_check):
    """
    Test boxing and unboxing of pd.Timedelta
    """

    def test_impl(td_obj):
        return td_obj

    check_func(test_impl, (timedelta_value,))


@pytest.mark.slow
def test_constructor(memory_leak_check):
    def test_impl1():
        return pd.Timedelta(232142)

    def test_impl2():
        return pd.Timedelta(
            days=1,
            weeks=10,
            hours=5,
            minutes=3,
            microseconds=121,
            milliseconds=787,
            seconds=2,
        )

    check_func(test_impl1, ())
    check_func(test_impl2, ())


def test_value(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.value

    check_func(test_impl, (timedelta_value,))


def test_delta(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.delta

    check_func(test_impl, (timedelta_value,))


def test_days(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.days

    check_func(test_impl, (timedelta_value,))


def test_seconds(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.seconds

    check_func(test_impl, (timedelta_value,))


def test_microseconds(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.microseconds

    check_func(test_impl, (timedelta_value,))


def test_nanoseconds(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.nanoseconds

    check_func(test_impl, (timedelta_value,))


def test_components(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.components

    check_func(test_impl, (timedelta_value,))


def test_to_numpy(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.to_numpy()

    check_func(test_impl, (timedelta_value,))


def test_to_timedelta64(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.to_timedelta64()

    check_func(test_impl, (timedelta_value,))


def test_to_pytimedelta(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.to_pytimedelta()

    check_func(test_impl, (timedelta_value,))


def test_total_seconds(timedelta_value, memory_leak_check):
    def test_impl(td):
        return td.total_seconds()

    check_func(test_impl, (timedelta_value,))


@pytest.mark.slow
def test_hash(memory_leak_check):
    td1 = pd.Timedelta(1)
    td2 = pd.Timedelta(2)
    td3 = pd.Timedelta(1)

    def impl(td1, td2, td3):
        d = dict()
        d[td1] = 1
        d[td2] = 2
        d[td3] = 3
        return d

    check_func(impl, (td1, td2, td3), dist_test=False)


@pytest.fixture(
    params=[
        (
            pd.Timedelta(232142),
            pd.Timedelta(
                days=1,
                weeks=10,
                hours=5,
                minutes=3,
                microseconds=121,
                milliseconds=787,
                seconds=2,
            ),
        ),
        (
            pd.Timedelta(232142),
            datetime.timedelta(
                days=1,
                weeks=10,
                hours=5,
                minutes=3,
                microseconds=121,
                milliseconds=787,
                seconds=2,
            ),
        ),
        (
            pd.Timedelta(232142),
            datetime.datetime(
                year=2015,
                month=7,
                day=1,
                hour=5,
                minute=3,
                microsecond=121,
                second=2,
            ),
        ),
        (
            pd.Timedelta(232142),
            pd.Timestamp(
                year=2018,
                month=4,
                day=1,
                hour=5,
                minute=3,
                microsecond=12100,
                second=2,
                nanosecond=42,
            ),
        ),
    ],
)
def binary_params(request):
    return request.param


@pytest.fixture(
    params=[
        (
            pd.Timestamp(
                year=2018,
                month=4,
                day=1,
                hour=5,
                minute=3,
                microsecond=12100,
                second=2,
                nanosecond=42,
            ),
            pd.Timedelta(232142),
        ),
        (
            datetime.timedelta(
                days=1,
                weeks=10,
                hours=5,
                minutes=3,
                microseconds=121,
                milliseconds=787,
                seconds=2,
            ),
            pd.Timestamp(
                year=2018,
                month=4,
                day=1,
                hour=5,
                minute=3,
                microsecond=12100,
                second=2,
                nanosecond=42,
            ),
        ),
    ]
)
def add_params(request):
    return request.param


def test_pd_timedelta_add(binary_params, memory_leak_check):
    def test_impl(a, b):
        return a + b

    val1 = binary_params[0]
    val2 = binary_params[1]

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))


def test_timestamp_timedelta_add(add_params, memory_leak_check):
    def test_impl(a, b):
        return a + b

    val1 = add_params[0]
    val2 = add_params[1]

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))


def test_pd_timedelta_sub(binary_params, memory_leak_check):
    def test_impl(a, b):
        return a - b

    val1 = binary_params[0]
    val2 = binary_params[1]

    check_func(test_impl, (val2, val1))


def test_pd_timedelta_mult(timedelta_value, memory_leak_check):
    def test_impl(a, b):
        return a * b

    check_func(test_impl, (timedelta_value, 3))
    check_func(test_impl, (-4, timedelta_value))


@pytest.mark.slow
def test_pd_timedelta_mult_int_literal(timedelta_value, memory_leak_check):
    def test_impl1(val):
        return 3 * val

    def test_impl2(val):
        return val * -4

    check_func(test_impl1, (timedelta_value,))
    check_func(test_impl2, (timedelta_value,))


def test_pd_timedelta_floordiv_int(timedelta_value, memory_leak_check):
    def test_impl(a, b):
        return a // b

    check_func(test_impl, (timedelta_value, 3))
    check_func(test_impl, (timedelta_value, -4))


@pytest.mark.slow
def test_pd_timedelta_floordiv_int_literal(timedelta_value, memory_leak_check):
    def test_impl1(val):
        return val // 3

    def test_impl2(val):
        return val // -4

    check_func(test_impl1, (timedelta_value,))
    check_func(test_impl2, (timedelta_value,))


def test_pd_timedelta_floordiv_tds(memory_leak_check):
    def test_impl(a, b):
        return a // b

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))

    val1 = val1 * -1

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))


def test_pd_timedelta_truediv_int(timedelta_value, memory_leak_check):
    def test_impl(a, b):
        return a / b

    check_func(test_impl, (timedelta_value, 3))
    check_func(test_impl, (timedelta_value, -4))


@pytest.mark.slow
def test_pd_timedelta_truediv_int_literal(timedelta_value, memory_leak_check):
    def test_impl1(val):
        return val / 3

    def test_impl2(val):
        return val / -4

    check_func(test_impl1, (timedelta_value,))
    check_func(test_impl2, (timedelta_value,))


def test_pd_timedelta_truediv_tds(memory_leak_check):
    def test_impl(a, b):
        return a / b

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))

    val1 = val1 * -1

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))


def test_pd_timedelta_mod(memory_leak_check):
    def test_impl(a, b):
        return a % b

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))

    val1 = val1 * -1

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))


def test_pd_timedelta_eq(memory_leak_check):
    def test_impl(a, b):
        return a == b

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val1, val1))
    check_func(test_impl, (val1, -val1))


def test_pd_timedelta_ne(memory_leak_check):
    def test_impl(a, b):
        return a != b

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val1, val1))
    check_func(test_impl, (val1, -val1))


def test_pd_timedelta_le(memory_leak_check):
    def test_impl(a, b):
        return a <= b

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val1, val1))
    check_func(test_impl, (val1, -val1))


def test_pd_timedelta_lt(memory_leak_check):
    def test_impl(a, b):
        return a < b

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val1, val1))
    check_func(test_impl, (val1, -val1))


def test_pd_timedelta_ge(memory_leak_check):
    def test_impl(a, b):
        return a >= b

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val1, val1))
    check_func(test_impl, (val1, -val1))


def test_pd_timedelta_gt(memory_leak_check):
    def test_impl(a, b):
        return a > b

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val1, val1))
    check_func(test_impl, (val1, -val1))


@pytest.mark.slow
def test_pd_timedelta_neg(timedelta_value, memory_leak_check):
    def test_impl(a):
        return -a

    check_func(test_impl, (timedelta_value,))


def test_pd_timedelta_pos(timedelta_value, memory_leak_check):
    def test_impl(a):
        return +a

    check_func(test_impl, (timedelta_value,))


def test_pd_timedelta_divmod(memory_leak_check):
    def test_impl(a, b):
        return divmod(a, b)

    val1 = pd.Timedelta(-1000010000100)
    val2 = pd.Timedelta(
        days=1,
        weeks=10,
        hours=5,
        minutes=3,
        microseconds=121,
        milliseconds=787,
        seconds=2,
    )

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))

    val1 = val1 * -1

    check_func(test_impl, (val1, val2))
    check_func(test_impl, (val2, val1))


def test_pd_timedelta_abs(timedelta_value, memory_leak_check):
    def test_impl(a):
        return abs(a)

    check_func(test_impl, (timedelta_value,))


def test_pd_timedelta_range():
    def test1():
        return pd.timedelta_range(start="1 day", periods=4)

    def test2():
        return pd.timedelta_range(start="1 day", periods=4, closed="right")

    def test3():
        return pd.timedelta_range(start="1 day", end="2 days", freq="6H")

    def test4():
        return pd.timedelta_range(start="1 day", end="5 days", periods=4)

    tests = [test1, test2, test3, test4]
    for test in tests:
        actual = bodo.jit(test)()
        expected = test()
        pd.testing.assert_index_equal(actual, expected)
