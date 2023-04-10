# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL date/time functions
"""
import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.libs.bodosql_datetime_array_kernels import (
    standardize_snowflake_date_time_part_compile_time,
)
from bodo.tests.timezone_common import (
    generate_date_trunc_date_func,
    generate_date_trunc_func,
    generate_date_trunc_time_func,
)
from bodo.tests.utils import (
    bodosql_use_date_type,
    check_func,
    nanoseconds_to_other_time_units,
)


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


@pytest.mark.parametrize(
    "interval_input",
    [
        pytest.param(pd.Timedelta(seconds=90), id="timedelta-scalar"),
        pytest.param(
            pd.Series(
                [
                    pd.Timedelta(days=1),
                    None,
                    pd.Timedelta(seconds=-42),
                    pd.Timedelta(microseconds=15),
                    pd.Timedelta(days=-1, minutes=15),
                ]
            ).values,
            id="timedelta-vector",
        ),
    ],
)
def test_interval_add_interval_to_time(interval_input, memory_leak_check):
    """
    Tests support for interval_add_interval on various Interval types.
    """

    def impl(arr0, arr1):
        return pd.Series(bodo.libs.bodosql_array_kernels.add_interval(arr0, arr1))

    time_input = pd.Series(
        [
            bodo.Time(12, 30, 0, nanosecond=1),
            bodo.Time(0, 0, 1),
            bodo.Time(0, 0, 0),
            bodo.Time(12, 0, 0),
            bodo.Time(10, 50, 45, microsecond=500),
        ]
    )

    if isinstance(interval_input, (pd.Timedelta)):
        answer = pd.Series(
            [
                bodo.Time(12, 31, 30, nanosecond=1),
                bodo.Time(0, 1, 31),
                bodo.Time(0, 1, 30),
                bodo.Time(12, 1, 30),
                bodo.Time(10, 52, 15, microsecond=500),
            ]
        )
    else:
        answer = pd.Series(
            [
                bodo.Time(12, 30, 0, nanosecond=1),
                None,
                bodo.Time(23, 59, 18),
                bodo.Time(12, 0, 0, microsecond=15),
                bodo.Time(11, 5, 45, microsecond=500),
            ]
        )

    check_func(
        impl,
        (time_input, interval_input),
        py_output=answer,
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


@pytest.mark.parametrize(
    "args, answers",
    [
        pytest.param(
            (
                pd.Series([1, 1, 2, 2, -1, -1, 16, 16, -50, -50]),
                pd.Series([datetime.date(2018, 1, 1), None] * 5),
            ),
            {
                "years": pd.Series(
                    [
                        datetime.date(2019, 1, 1),
                        None,
                        datetime.date(2020, 1, 1),
                        None,
                        datetime.date(2017, 1, 1),
                        None,
                        datetime.date(2034, 1, 1),
                        None,
                        datetime.date(1968, 1, 1),
                        None,
                    ]
                ),
                "quarters": pd.Series(
                    [
                        datetime.date(2018, 4, 1),
                        None,
                        datetime.date(2018, 7, 1),
                        None,
                        datetime.date(2017, 10, 1),
                        None,
                        datetime.date(2022, 1, 1),
                        None,
                        datetime.date(2005, 7, 1),
                        None,
                    ]
                ),
                "months": pd.Series(
                    [
                        datetime.date(2018, 2, 1),
                        None,
                        datetime.date(2018, 3, 1),
                        None,
                        datetime.date(2017, 12, 1),
                        None,
                        datetime.date(2019, 5, 1),
                        None,
                        datetime.date(2013, 11, 1),
                        None,
                    ]
                ),
                "weeks": pd.Series(
                    [
                        datetime.date(2018, 1, 8),
                        None,
                        datetime.date(2018, 1, 15),
                        None,
                        datetime.date(2017, 12, 25),
                        None,
                        datetime.date(2018, 4, 23),
                        None,
                        datetime.date(2017, 1, 16),
                        None,
                    ]
                ),
                "days": pd.Series(
                    [
                        datetime.date(2018, 1, 2),
                        None,
                        datetime.date(2018, 1, 3),
                        None,
                        datetime.date(2017, 12, 31),
                        None,
                        datetime.date(2018, 1, 17),
                        None,
                        datetime.date(2017, 11, 12),
                        None,
                    ]
                ),
                "hours": pd.Series(
                    [
                        pd.Timestamp("2018-01-01 01:00:00"),
                        None,
                        pd.Timestamp("2018-01-01 02:00:00"),
                        None,
                        pd.Timestamp("2017-12-31 23:00:00"),
                        None,
                        pd.Timestamp("2018-01-01 16:00:00"),
                        None,
                        pd.Timestamp("2017-12-29 22:00:00"),
                        None,
                    ]
                ),
                "minutes": pd.Series(
                    [
                        pd.Timestamp("2018-01-01 00:01:00"),
                        None,
                        pd.Timestamp("2018-01-01 00:02:00"),
                        None,
                        pd.Timestamp("2017-12-31 23:59:00"),
                        None,
                        pd.Timestamp("2018-01-01 00:16:00"),
                        None,
                        pd.Timestamp("2017-12-31 23:10:00"),
                        None,
                    ]
                ),
                "seconds": pd.Series(
                    [
                        pd.Timestamp("2018-01-01 00:00:01"),
                        None,
                        pd.Timestamp("2018-01-01 00:00:02"),
                        None,
                        pd.Timestamp("2017-12-31 23:59:59"),
                        None,
                        pd.Timestamp("2018-01-01 00:00:16"),
                        None,
                        pd.Timestamp("2017-12-31 23:59:10"),
                        None,
                    ]
                ),
                "milliseconds": pd.Series(
                    [
                        pd.Timestamp("2018-01-01 00:00:00.001"),
                        None,
                        pd.Timestamp("2018-01-01 00:00:00.002"),
                        None,
                        pd.Timestamp("2017-12-31 23:59:59.999"),
                        None,
                        pd.Timestamp("2018-01-01 00:00:00.016"),
                        None,
                        pd.Timestamp("2017-12-31 23:59:59.950"),
                        None,
                    ]
                ),
                "microseconds": pd.Series(
                    [
                        pd.Timestamp("2018-01-01 00:00:00.000001"),
                        None,
                        pd.Timestamp("2018-01-01 00:00:00.000002"),
                        None,
                        pd.Timestamp("2017-12-31 23:59:59.999999"),
                        None,
                        pd.Timestamp("2018-01-01 00:00:00.000016"),
                        None,
                        pd.Timestamp("2017-12-31 23:59:59.999950"),
                        None,
                    ]
                ),
                "nanoseconds": pd.Series(
                    [
                        pd.Timestamp("2018-01-01 00:00:00.000000001"),
                        None,
                        pd.Timestamp("2018-01-01 00:00:00.000000002"),
                        None,
                        pd.Timestamp("2017-12-31 23:59:59.999999999"),
                        None,
                        pd.Timestamp("2018-01-01 00:00:00.000000016"),
                        None,
                        pd.Timestamp("2017-12-31 23:59:59.999999950"),
                        None,
                    ]
                ),
            },
            id="all_vector",
        ),
        pytest.param(
            (
                100,
                pd.Series(pd.date_range("1999-12-20", "1999-12-30", 11).date),
            ),
            {
                "years": pd.Series(pd.date_range("2099-12-20", "2099-12-30", 11).date),
                "quarters": pd.Series(
                    pd.date_range("2024-12-20", "2024-12-30", 11).date
                ),
                "months": pd.Series(pd.date_range("2008-04-20", "2008-04-30", 11).date),
                "weeks": pd.Series(pd.date_range("2001-11-19", "2001-11-29", 11).date),
                "days": pd.Series(pd.date_range("2000-03-29", "2000-04-08", 11).date),
                "hours": pd.Series(
                    pd.date_range("1999-12-24 04:00:00", "2000-01-03 04:00:00", 11)
                ),
                "minutes": pd.Series(
                    pd.date_range("1999-12-20 01:40:00", "1999-12-30 01:40:00", 11)
                ),
                "seconds": pd.Series(
                    pd.date_range("1999-12-20 00:01:40", "1999-12-30 00:01:40", 11)
                ),
                "milliseconds": pd.Series(
                    pd.date_range(
                        "1999-12-20 00:00:00.100", "1999-12-30 00:00:00.100", 11
                    )
                ),
                "microseconds": pd.Series(
                    pd.date_range(
                        "1999-12-20 00:00:00.000100", "1999-12-30 00:00:00.000100", 11
                    )
                ),
                "nanoseconds": pd.Series(
                    pd.date_range(
                        "1999-12-20 00:00:00.000000100",
                        "1999-12-30 00:00:00.000000100",
                        11,
                    )
                ),
            },
            id="scalar_vector",
        ),
        pytest.param(
            (300, datetime.date(1776, 7, 4)),
            {
                "years": datetime.date(2076, 7, 4),
                "quarters": datetime.date(1851, 7, 4),
                "months": datetime.date(1801, 7, 4),
                "weeks": datetime.date(1782, 4, 4),
                "days": datetime.date(1777, 4, 30),
                "hours": pd.Timestamp("1776-07-16 12:00:00"),
                "minutes": pd.Timestamp("1776-07-04 05:00:00"),
                "seconds": pd.Timestamp("1776-07-04 00:05:00"),
                "milliseconds": pd.Timestamp("1776-07-04 00:00:00.300"),
                "microseconds": pd.Timestamp("1776-07-04 00:00:00.000300"),
                "nanoseconds": pd.Timestamp("1776-07-04 00:00:00.000000300"),
            },
            id="all_scalar",
        ),
        pytest.param(
            (None, datetime.date(1776, 7, 4)),
            {
                "years": None,
                "quarters": None,
                "months": None,
                "weeks": None,
                "days": None,
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
        "years",
        "quarters",
        "months",
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ],
)
def test_add_interval_date(unit, args, answers, memory_leak_check):
    """
    Tests the add_interval_xxx kernels with datetime.date input.
    If the time unit is larger than or equal to day, the function will return a
    datetime.date object. If the time unit is smaller than or equal to hour, the
    function will transform the date to pd.Timestamp and calculate with pd.Timedelta.
    """
    if any(isinstance(arg, pd.Series) for arg in args):
        fn_str = f"lambda amount, start_dt: pd.Series(bodo.libs.bodosql_array_kernels.add_interval_{unit}(amount, start_dt))"
    else:
        fn_str = f"lambda amount, start_dt: bodo.libs.bodosql_array_kernels.add_interval_{unit}(amount, start_dt)"
    impl = eval(fn_str)

    with bodosql_use_date_type():
        check_func(
            impl,
            args,
            py_output=answers[unit],
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


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(
                [
                    datetime.date(2017, 3, 26),
                    datetime.date(2000, 12, 31),
                    datetime.date(2003, 9, 6),
                    datetime.date(2023, 3, 6),
                    datetime.date(2020, 6, 26),
                ]
                + [None, None]
                + [
                    datetime.date(2024, 1, 1),
                    datetime.date(1996, 4, 25),
                    datetime.date(2022, 11, 17),
                    datetime.date(1917, 7, 29),
                    datetime.date(2007, 10, 14),
                ]
            ),
            id="vector",
        ),
        pytest.param(datetime.date(2025, 5, 3), id="scalar"),
    ],
)
def datetime_dates_scalar_vector(request):
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


def test_dayname_date(datetime_dates_scalar_vector, memory_leak_check):
    """
    test dayname kernel works for datetime.date input
    """

    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.dayname(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(datetime_dates_scalar_vector, datetime.date):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.dayname(arr)

    # Simulates DAYNAME on a single row
    def dayname_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            dows = (
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            )
            return dows[elem.weekday()]

    dayname_answer = vectorized_sol(
        (datetime_dates_scalar_vector,), dayname_scalar_fn, None
    )
    check_func(
        impl,
        (datetime_dates_scalar_vector,),
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


def test_get_weekofyear(dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.get_weekofyear(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(dates_scalar_vector, pd.Timestamp):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.get_weekofyear(arr)

    # Simulates get_weekofyear on a single row
    def get_weekofyear_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            elem = pd.Timestamp(elem)
            return elem.weekofyear

    answer = vectorized_sol((dates_scalar_vector,), get_weekofyear_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=answer,
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


@pytest.mark.parametrize(
    "dt, date_part",
    [
        pytest.param(
            pd.Series(pd.date_range("2019-01-01", "2020-01-01", periods=20)),
            "month",
            id="timestamp-vector",
        ),
        pytest.param(
            pd.Timestamp("2019-01-01 12:34:56"),
            "year",
            id="timestamp-scalar",
        ),
        pytest.param(
            pd.Series(pd.date_range("2019-01-01", "2020-01-01", periods=20).date),
            "week",
            id="date-vector",
        ),
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
                    None,
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
            "year",
            id="timestamp-tz-vector",
        ),
        pytest.param(
            pd.Timestamp("2022-3-1", tz="Poland"),
            "quarter",
            id="timestamp-tz-scalar",
        ),
    ],
)
def test_last_day(dt, date_part, memory_leak_check):
    if isinstance(dt, pd.Series):
        fn_str = f"lambda date_or_time_expr: pd.Series(bodo.libs.bodosql_array_kernels.last_day_{date_part}(date_or_time_expr))"
    else:
        fn_str = f"lambda date_or_time_expr: bodo.libs.bodosql_array_kernels.last_day_{date_part}(date_or_time_expr)"
    impl = eval(fn_str)

    last_day_answer = vectorized_sol(
        (dt, date_part),
        last_day_scalar_fn,
        None,
    )
    with bodosql_use_date_type():
        check_func(
            impl,
            (dt,),
            py_output=last_day_answer,
        )


def last_day_scalar_fn(elem, unit):
    """
    Simulates LAST_DAY on a single row
    """
    if pd.isna(elem) or pd.isna(unit):
        return None
    else:
        if unit == "year":
            return datetime.date(elem.year, 12, 31)
        elif unit == "quarter":
            y = elem.year
            m = ((elem.month - 1) // 3 + 1) * 3
            d = bodo.hiframes.pd_offsets_ext.get_days_in_month(y, m)
            return datetime.date(y, m, d)
        elif unit == "month":
            y = elem.year
            m = elem.month
            d = bodo.hiframes.pd_offsets_ext.get_days_in_month(y, m)
            return datetime.date(y, m, d)
        elif unit == "week":
            return (pd.Timestamp(elem) + pd.Timedelta(days=(6 - elem.weekday()))).date()


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


def test_monthname_date(datetime_dates_scalar_vector, memory_leak_check):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.monthname(arr))

    # avoid pd.Series() conversion for scalar output
    if isinstance(datetime_dates_scalar_vector, datetime.date):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.monthname(arr)

    mons = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    # Simulates MONTHNAME on a single row
    def monthname_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return mons[elem.month - 1]

    monthname_answer = vectorized_sol(
        (datetime_dates_scalar_vector,), monthname_scalar_fn, None
    )
    with bodosql_use_date_type():
        check_func(
            impl,
            (datetime_dates_scalar_vector,),
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
    "dt, dow_str",
    [
        pytest.param(
            pd.Series(pd.date_range("2018-01-01", "2019-01-01", periods=20)),
            pd.Series(["su"] * 20),
            id="timestamp-vector",
        ),
        pytest.param(
            pd.Series(pd.date_range("2019-01-01", "2020-01-01", periods=21).date),
            pd.Series(["mo", "tu", "we", "th", "fr", "sa", "su"] * 3),
            id="date-vector",
        ),
    ],
)
def test_next_previous_day(dt, dow_str, memory_leak_check):
    def next_impl(arr0, arr1):
        return pd.Series(bodo.libs.bodosql_array_kernels.next_day(arr0, arr1))

    def prev_impl(arr0, arr1):
        return pd.Series(bodo.libs.bodosql_array_kernels.previous_day(arr0, arr1))

    dow_map = {"mo": 0, "tu": 1, "we": 2, "th": 3, "fr": 4, "sa": 5, "su": 6}
    # Simulates next/previous_day on a single row
    def next_prev_day_scalar_fn(is_prev=False):
        mlt = -1 if is_prev else 1

        def impl(dt, day):
            if pd.isna(dt) or pd.isna(day):
                return None
            if isinstance(dt, pd.Timestamp):
                return dt.date() + mlt * pd.Timedelta(
                    days=7 - ((mlt * (dt.dayofweek - dow_map[day])) % 7)
                )
            else:
                return dt + mlt * pd.Timedelta(
                    days=7 - ((mlt * (dt.weekday() - dow_map[day])) % 7)
                )

        return impl

    next_day_answer = pd.Series(
        [next_prev_day_scalar_fn()(dt[i], dow_str[i]) for i in range(len(dt))]
    )
    print(next_day_answer)
    with bodosql_use_date_type():
        check_func(
            next_impl,
            (
                dt,
                dow_str,
            ),
            py_output=next_day_answer,
            reset_index=True,
        )
    previous_day_answer = pd.Series(
        [next_prev_day_scalar_fn(True)(dt[i], dow_str[i]) for i in range(len(dt))]
    )
    with bodosql_use_date_type():
        check_func(
            prev_impl,
            (
                dt,
                dow_str,
            ),
            py_output=previous_day_answer,
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
def test_date_trunc(datetime_part_strings, ts_input, memory_leak_check):
    """
    Tests date_trunc array kernel on various timestamp inputs, testing all the different code paths
    in the generated kernel.
    """

    def impl(datetime_part_strings, arr):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.date_trunc(datetime_part_strings, arr)
        )

    # avoid pd.Series() conversion for scalar output
    if isinstance(ts_input, pd.Timestamp):
        impl = lambda datetime_part_strings, arr: bodo.libs.bodosql_array_kernels.date_trunc(
            datetime_part_strings, arr
        )

    # Simulates date_trunc on a single row
    def date_trunc_scalar_fn(datetime_part_strings, ts_input):
        return generate_date_trunc_func(datetime_part_strings)(ts_input)

    answer = vectorized_sol(
        (datetime_part_strings, ts_input), date_trunc_scalar_fn, None
    )
    check_func(
        impl,
        (datetime_part_strings, ts_input),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "time_input",
    [
        pytest.param(
            bodo.Time(14, 24, 35, 523, 98, 13),
            id="scalar-time",
        ),
        pytest.param(
            pd.Series(
                [
                    bodo.Time(19, 53, 26, 901, 8, 79),
                    bodo.Time(0, 20, 43, 365, 128, 74),
                    bodo.Time(23, 16, 6, 25, 77, 32),
                    None,
                ]
            ).values,
            id="vector-time",
        ),
    ],
)
def test_date_trunc_time(datetime_part_strings, time_input, memory_leak_check):
    """
    Tests date_trunc array kernel on various bodo.Time inputs, testing all the different code paths
    in the generated kernel.
    """

    def impl(datetime_part_strings, arr):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.date_trunc(datetime_part_strings, arr)
        )

    # avoid pd.Series() conversion for scalar output
    if isinstance(time_input, bodo.Time):
        impl = lambda datetime_part_strings, arr: bodo.libs.bodosql_array_kernels.date_trunc(
            datetime_part_strings, arr
        )

    # Simulates date_trunc on a single row
    def date_trunc_time_scalar_fn(datetime_part_strings, time_input):
        return generate_date_trunc_time_func(datetime_part_strings)(time_input)

    answer = vectorized_sol(
        (datetime_part_strings, time_input), date_trunc_time_scalar_fn, None
    )
    check_func(
        impl,
        (datetime_part_strings, time_input),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "date_input",
    [
        pytest.param(
            datetime.date(2007, 10, 14),
            id="scalar-date",
        ),
        pytest.param(
            pd.Series(
                [
                    datetime.date(2025, 5, 3),
                    datetime.date(1987, 3, 15),
                    datetime.date(2117, 8, 29),
                    None,
                ]
            ).values,
            id="vector-date",
        ),
    ],
)
def test_date_trunc_date(day_part_strings, date_input, memory_leak_check):
    """
    Tests date_trunc array kernel on various bodo.Time inputs, testing all the different code paths
    in the generated kernel.
    """

    def impl(day_part_strings, arr):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.date_trunc(day_part_strings, arr)
        )

    # avoid pd.Series() conversion for scalar output
    if isinstance(date_input, datetime.date):
        impl = lambda day_part_strings, arr: bodo.libs.bodosql_array_kernels.date_trunc(
            day_part_strings, arr
        )

    # Simulates date_trunc on a single row
    def date_trunc_date_scalar_fn(datetime_part_strings, date_input):
        return generate_date_trunc_date_func(datetime_part_strings)(date_input)

    answer = vectorized_sol(
        (day_part_strings, date_input), date_trunc_date_scalar_fn, None
    )
    with bodosql_use_date_type():
        check_func(
            impl,
            (day_part_strings, date_input),
            py_output=answer,
            check_dtype=False,
            reset_index=True,
        )


@pytest.mark.parametrize(
    "interval_input",
    [
        pytest.param(pd.DateOffset(years=1, months=-4), id="date-offset-scalar"),
        pytest.param(pd.Timedelta(seconds=42), id="timedelta-scalar"),
        pytest.param(
            pd.Series(
                [
                    pd.Timedelta(days=1),
                    None,
                    pd.Timedelta(seconds=-42),
                    pd.Timedelta(microseconds=15),
                    pd.Timedelta(days=-1, minutes=15),
                ]
                * 6
            ).values,
            id="timedelta-vector",
        ),
    ],
)
def test_interval_multiply(interval_input, memory_leak_check):
    """
    Tests interval_multiply with the different interval types and
    on arrays.
    """
    int_value = 6
    int_array = pd.array([1, None, -4, 5, 0] * 6, dtype="Int64")

    def impl(interval, integer):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.interval_multiply(interval, integer)
        )

    scalar_integer_impl = impl
    # If we have a scalar interval then the scalar integer outputs a scalar.
    if isinstance(interval_input, (pd.Timedelta, pd.DateOffset)):
        scalar_integer_impl = (
            lambda interval, integer: bodo.libs.bodosql_array_kernels.interval_multiply(
                interval, integer
            )
        )

    def interval_scalar_fn(interval, integer):
        if pd.isna(interval) or pd.isna(integer):
            return None
        else:
            if isinstance(interval, np.timedelta64):
                interval = pd.Timedelta(interval)
            return interval * integer

    scalar_integer_answer = vectorized_sol(
        (interval_input, int_value), interval_scalar_fn, None
    )
    check_func(
        scalar_integer_impl,
        (interval_input, int_value),
        py_output=scalar_integer_answer,
        check_dtype=False,
        reset_index=True,
    )
    if not isinstance(interval_input, pd.DateOffset):
        # DateOffset can't be used with an array input.
        vector_integer_answer = vectorized_sol(
            (interval_input, int_array), interval_scalar_fn, None
        )
        check_func(
            impl,
            (interval_input, int_array),
            py_output=vector_integer_answer,
            check_dtype=False,
            reset_index=True,
        )


@pytest.mark.parametrize(
    "interval_input",
    [
        pytest.param(pd.Timedelta(seconds=42), id="timedelta-scalar"),
        pytest.param(
            pd.Series(
                [
                    pd.Timedelta(days=1),
                    None,
                    pd.Timedelta(seconds=-42),
                    pd.Timedelta(microseconds=15),
                    pd.Timedelta(days=-1, minutes=15),
                ]
                * 6
            ).values,
            id="timedelta-vector",
        ),
    ],
)
def test_interval_add_interval(interval_input, memory_leak_check):
    """
    Tests support for interval_add_interval on various Interval types.
    """

    def impl(arr0, arr1):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.interval_add_interval(arr0, arr1)
        )

    is_scalar = isinstance(interval_input, (pd.Timedelta, pd.DateOffset))

    # If we have a scalar interval then the scalar integer outputs a scalar.
    if is_scalar:
        impl = lambda arr0, arr1: bodo.libs.bodosql_array_kernels.interval_add_interval(
            arr0, arr1
        )

    def interval_scalar_fn(arg0, arg1):
        if pd.isna(arg0) or pd.isna(arg1):
            return None
        else:
            if isinstance(arg0, np.timedelta64):
                arg0 = pd.Timedelta(arg0)
            if isinstance(arg1, np.timedelta64):
                arg1 = pd.Timedelta(arg1)
            return arg0 + arg1

    answer = vectorized_sol((interval_input, interval_input), interval_scalar_fn, None)
    check_func(
        impl,
        (interval_input, interval_input),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )
    if not is_scalar:
        # testing adding an array with scalars
        scalar_value = pd.Timedelta(interval_input[0])
        answer = vectorized_sol(
            (interval_input, scalar_value), interval_scalar_fn, None
        )
        check_func(
            impl,
            (interval_input, scalar_value),
            py_output=answer,
            check_dtype=False,
            reset_index=True,
        )
        check_func(
            impl,
            (scalar_value, interval_input),
            py_output=answer,
            check_dtype=False,
            reset_index=True,
        )


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param(pd.Timestamp("2022-12-25 4:40:45"), id="scalar"),
        pytest.param(
            pd.Series(
                [
                    "2023-1-1",
                    None,
                    "2023-1-10",
                    "2020-11-21",
                    "2022-12-31",
                ]
                * 6
            ).values,
            id="vector",
        ),
    ],
)
def test_create_date(arg, memory_leak_check):
    """
    Tests create_date with array and scalar data.
    """

    def impl(arg):
        return pd.Series(bodo.libs.bodosql_array_kernels.create_date(arg))

    # Scalar isn't wrapped in a series.
    if isinstance(arg, (pd.Timestamp, str)):
        impl = lambda arg: bodo.libs.bodosql_array_kernels.create_date(arg)

    def days_scalar_fn(arg):
        if pd.isna(arg):
            return None
        else:
            return pd.Timestamp(arg).normalize()

    answer = vectorized_sol((arg,), days_scalar_fn, None)

    check_func(
        impl,
        (arg,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param(pd.Timestamp("2022-12-25 4:40:45"), id="scalar"),
        pytest.param(
            pd.Series(
                [
                    "2023-1-1 00:00:00",
                    None,
                    "2023-1-10 12:00:01",
                    "2020-11-21",
                    "2022-12-31 23:11:11.433",
                ]
                * 6
            ).values,
            id="vector",
        ),
    ],
)
def test_create_timestamp(arg, memory_leak_check):
    """
    Tests create_timestamp with array and scalar data.
    """

    def impl(arg):
        return pd.Series(bodo.libs.bodosql_array_kernels.create_timestamp(arg))

    # Scalar isn't wrapped in a series.
    if isinstance(arg, (pd.Timestamp, str)):
        impl = lambda arg: bodo.libs.bodosql_array_kernels.create_timestamp(arg)

    def days_scalar_fn(arg):
        if pd.isna(arg):
            return None
        else:
            return pd.Timestamp(arg)

    answer = vectorized_sol((arg,), days_scalar_fn, None)

    check_func(
        impl,
        (arg,),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


def diff_fn(unit, arg0, arg1):
    if pd.isna(arg0) or pd.isna(arg1):
        return None
    else:
        datetime_part = standardize_snowflake_date_time_part_compile_time(unit)(unit)
        if isinstance(arg0, bodo.Time):
            return nanoseconds_to_other_time_units(
                arg1.value, datetime_part
            ) - nanoseconds_to_other_time_units(arg0.value, datetime_part)
        A = pd.Timestamp(arg0)
        B = pd.Timestamp(arg1)

        if datetime_part == "year":
            return B.year - A.year
        elif datetime_part == "quarter":
            yr_diff = (B.year - A.year) * 4
            q_diff = B.quarter - A.quarter
            return yr_diff + q_diff
        elif datetime_part == "month":
            yr_diff = (B.year - A.year) * 12
            m_diff = B.month - A.month
            return yr_diff + m_diff
        elif datetime_part == "week":
            yr_diff = bodo.libs.bodosql_array_kernels.get_iso_weeks_between_years(
                A.year, B.year
            )
            wk_diff = B.week - A.week
            return yr_diff + wk_diff
        elif datetime_part == "day":
            return (B - A).days
        else:
            return nanoseconds_to_other_time_units(
                B.value, datetime_part
            ) - nanoseconds_to_other_time_units(A.value, datetime_part)


@pytest.fixture(params=["scalar", "vector", "null"])
def construct_timestamp_data(request):
    if request.param == "null":
        return 2015, None, None, None, 100, 0, None, None
    if request.param == "scalar":
        return (2018, -1, 600, 30, -30, 15, 0, "2019-06-24 05:30:15")
    year = pd.Series([2025, 2020, 2022, 2020, 2027, 2020], dtype=pd.Int64Dtype())
    month = pd.Series([7, None, 1, 53, -6, 0], dtype=pd.Int64Dtype())
    day = pd.Series([4, 12, 100, 0, 20, 0], dtype=pd.Int64Dtype())
    hour = pd.Series([0, None, 0, 12, 2882, 1], dtype=pd.Int64Dtype())
    minute = pd.Series([0, 0, 0, 45, 5, 2], dtype=pd.Int64Dtype())
    second = pd.Series([0, None, 0, 59, 0, 3], dtype=pd.Int64Dtype())
    nanosecond = pd.Series([0, 13, 0, 1234, 68719476736, 0], dtype=pd.Int64Dtype())
    answer = pd.Series(
        [
            "2025-7-4",
            None,
            "2022-4-10",
            "2024-4-30 12:45:59.000001234",
            "2026-10-18 02:06:08.719476736",
            "2019-11-30 01:02:03",
        ]
    )
    return year, month, day, hour, minute, second, nanosecond, answer


@pytest.mark.parametrize(
    "has_time_zone",
    [
        pytest.param(False, id="tz_naive"),
        pytest.param(True, id="tz_aware"),
    ],
)
def test_construct_timestamp(
    construct_timestamp_data, has_time_zone, memory_leak_check
):
    """
    Tests construct_timestamp with array and scalar data, with and without timezones.
    """

    def impl_naive(year, month, day, hour, minute, second, nanosecond):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.construct_timestamp(
                year, month, day, hour, minute, second, nanosecond, None
            )
        )

    def impl_aware(year, month, day, hour, minute, second, nanosecond):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.construct_timestamp(
                year, month, day, hour, minute, second, nanosecond, "US/Eastern"
            )
        )

    if has_time_zone:
        impl = impl_aware
    else:
        impl = impl_naive

    (
        year,
        month,
        day,
        hour,
        minute,
        second,
        nanosecond,
        answer,
    ) = construct_timestamp_data
    args = (year, month, day, hour, minute, second, nanosecond)
    if not any(isinstance(arg, pd.Series) for arg in args):
        if has_time_zone:
            impl = lambda year, month, day, hour, minute, second, nanosecond: bodo.libs.bodosql_array_kernels.construct_timestamp(
                year,
                month,
                day,
                hour,
                minute,
                second,
                nanosecond,
                numba.literally("US/Eastern"),
            )
        else:
            impl = lambda year, month, day, hour, minute, second, nanosecond: bodo.libs.bodosql_array_kernels.construct_timestamp(
                year, month, day, hour, minute, second, nanosecond, None
            )

    tz = "US/Eastern" if has_time_zone else None

    def construct_tz_scalar_fn(arg0, arg1):
        if pd.isna(arg0):
            return None
        else:
            return pd.Timestamp(arg0).tz_localize(arg1)

    check_func(
        impl,
        args,
        py_output=vectorized_sol((answer, tz), construct_tz_scalar_fn, None),
        check_dtype=False,
        reset_index=True,
    )


@pytest.fixture(params=["scalar", "vector", "null"])
def construct_date_data(request):
    if request.param == "null":
        return 2015, None, None, None
    if request.param == "scalar":
        return 2018, -1, 600, datetime.date(2019, 6, 24)
    year = pd.Series([2025, 2020, 2022, 2020, 2027, 2020], dtype=pd.Int64Dtype())
    month = pd.Series([7, None, 1, 53, -6, 0], dtype=pd.Int64Dtype())
    day = pd.Series([4, 12, 100, 0, 20, 0], dtype=pd.Int64Dtype())
    answer = pd.Series(
        [
            datetime.date(2025, 7, 4),
            None,
            datetime.date(2022, 4, 10),
            datetime.date(2024, 4, 30),
            datetime.date(2026, 10, 18),
            datetime.date(2019, 11, 30),
        ]
    )
    return year, month, day, answer


def test_date_from_parts(construct_date_data, memory_leak_check):
    """
    Tests date_from_parts with array and scalar data.
    """

    def impl(year, month, day):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.date_from_parts(year, month, day)
        )

    (
        year,
        month,
        day,
        answer,
    ) = construct_date_data
    args = (year, month, day)

    if not any(isinstance(arg, pd.Series) for arg in args):
        impl = lambda year, month, day: bodo.libs.bodosql_array_kernels.date_from_parts(
            year, month, day
        )

    def construct_date_scalar_fn(arg0, arg1, arg2):
        if pd.isna(arg0) or pd.isna(arg1) or pd.isna(arg2):
            return None
        else:
            months, month_overflow = 1 + ((arg1 - 1) % 12), (arg1 - 1) // 12
            date = datetime.date(arg0 + month_overflow, months, 1)
            date = date + datetime.timedelta(days=int(arg2) - 1)
            return date

    answer = vectorized_sol((year, month, day), construct_date_scalar_fn, None)

    with bodosql_use_date_type():
        check_func(
            impl,
            args,
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
                    py_output=(a1, a2, a3, a4, a5, a6, a7, a8),
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


@pytest.mark.slow
def test_option_get_weekofyear(memory_leak_check):
    """
    Tests get_weekofyear array kernel on optional input
    """

    def impl(arg, flag):
        arg0 = arg if flag else None
        return bodo.libs.bodosql_array_kernels.get_weekofyear(arg0)

    arg = pd.Timestamp("2022-11-6 12:40:45", tz="US/Pacific")

    for flag in [True, False]:
        answer = arg.weekofyear if flag else None
        check_func(
            impl,
            (arg, flag),
            py_output=answer,
        )


@pytest.mark.slow
def test_option_interval_multiply(memory_leak_check):
    """
    Tests interval_multiply array kernel on optional inputs.
    """

    def impl(interval, integer, flag0, flag1):
        arg0 = interval if flag0 else None
        arg1 = integer if flag1 else None
        return bodo.libs.bodosql_array_kernels.interval_multiply(arg0, arg1)

    arg0 = pd.Timedelta(days=14, seconds=12)
    arg1 = -14

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = arg0 * arg1 if flag0 and flag1 else None
            check_func(
                impl,
                (arg0, arg1, flag0, flag1),
                py_output=answer,
            )


@pytest.mark.slow
def test_interval_add_interval_optional(memory_leak_check):
    """
    Tests interval_add_interval with optional data.
    """

    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.interval_add_interval(arg0, arg1)

    A = pd.Timedelta(days=17)
    B = pd.Timedelta(seconds=-4)
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = A + B if flag0 and flag1 else None
            check_func(
                impl,
                (A, B, flag0, flag1),
                py_output=answer,
            )


@pytest.mark.slow
def test_create_date_optional(memory_leak_check):
    """
    Tests create_date with optional data.
    """

    def impl(arg, flag):
        arg0 = arg if flag else None
        return bodo.libs.bodosql_array_kernels.create_date(arg0)

    arg = "2022-11-1"
    for flag in [True, False]:
        answer = pd.Timestamp(arg) if flag else None
        check_func(
            impl,
            (arg, flag),
            py_output=answer,
        )


@pytest.mark.slow
def test_create_timestamp_optional(memory_leak_check):
    """
    Tests create_timestamp with optional data.
    """

    def impl(arg, flag):
        arg0 = arg if flag else None
        return bodo.libs.bodosql_array_kernels.create_timestamp(arg0)

    arg = "2022-11-1 12:32:31"
    for flag in [True, False]:
        answer = pd.Timestamp(arg) if flag else None
        check_func(
            impl,
            (arg, flag),
            py_output=answer,
        )


@pytest.mark.parametrize(
    "expr, format_str, answer",
    [
        pytest.param(
            pd.Series(
                [
                    datetime.date(2017, 6, 15),
                    datetime.date(1971, 2, 2),
                    None,
                    datetime.date(2022, 11, 25),
                    datetime.date(2001, 9, 30),
                ]
                * 4
            ),
            "%Y",
            pd.Series(["2017", "1971", None, "2022", "2001"] * 4),
            id="date-vector",
        ),
        pytest.param(
            datetime.date(2017, 6, 15), "%M %d %Y", "June 15 2017", id="date-scalar"
        ),
        pytest.param(
            pd.Series(
                [
                    pd.Timestamp("2015-3-14 00:00:01"),
                    pd.Timestamp("2015-3-19 00:00:30"),
                    pd.Timestamp("2015-3-13 23:59:50"),
                    None,
                    pd.Timestamp("2015-3-6 10:14:39"),
                ]
                * 4
            ),
            "%d %S",
            pd.Series(["14 01", "19 30", "13 50", None, "06 39"] * 4),
            id="timestamp-vector",
        ),
        pytest.param(
            pd.Timestamp("2022-11-6 12:40:45"),
            "%M %d %Y %H:%i:%S",
            "Nov 06 2022 12:40:45",
            id="timestamp-scalar",
        ),
    ],
)
def test_date_format(expr, format_str, answer, memory_leak_check):
    """
    Tests date_format kernel
    """

    def impl(expr, format_str):
        return pd.Series(bodo.libs.bodosql_array_kernels.date_format(expr, format_str))

    if isinstance(expr, (datetime.date, pd.Timestamp)):
        return lambda expr, format: bodo.libs.bodosql_array_kernels.date_format(
            expr, format_str
        )

    check_func(
        impl,
        (expr, format_str),
        py_output=answer,
    )
