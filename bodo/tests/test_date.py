# Copyright (C) 2022 Bodo Inc. All rights reserved.

import calendar
import datetime
import random

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func, generate_comparison_ops_func
from bodo.utils.typing import BodoError


# ------------------------- Test datetime OPs ------------------------- #
def test_datetime_date_series_min_max(memory_leak_check):
    """
    Test Series.min() and Series.max() with a datetime.date Series.
    """
    np.random.seed(1)
    date_arr = pd.date_range(start="2/1/2021", end="3/8/2021").date
    np.random.shuffle(date_arr)
    S = pd.Series(date_arr)

    def impl_min(S):
        return S.min()

    def impl_max(S):
        return S.max()

    check_func(impl_min, (S,))
    check_func(impl_max, (S,))


def test_datetime_date_series_min_max_none(memory_leak_check):
    """
    Test Series.min() and Series.max() with a datetime.date Series
    and a None entry. This isn't supported in Pandas but should work
    in Bodo.
    """
    np.random.seed(1)
    date_arr = pd.date_range(start="2/1/2021", end="3/8/2021").date
    np.random.shuffle(date_arr)
    S = pd.Series(date_arr)
    S.iat[10] = None

    def impl_min(S):
        return S.min()

    def impl_max(S):
        return S.max()

    # Comparison with None isn't supported in Python, so provide
    # py_output
    py_output = S.dropna().min()
    check_func(impl_min, (S,), py_output=py_output)
    py_output = S.dropna().max()
    check_func(impl_max, (S,), py_output=py_output)


def test_datetime_operations(memory_leak_check):
    """
    Test operations of datetime module objects in Bodo
    """

    def test_add(a, b):
        return a + b

    def test_sub(a, b):
        return a - b

    def test_mul(a, b):
        return a * b

    def test_floordiv(a, b):
        return a // b

    def test_truediv(a, b):
        return a / b

    def test_mod(a, b):
        return a % b

    def test_neg(a):
        return -a

    def test_pos(a):
        return +a

    def test_divmod(a, b):
        return divmod(a, b)

    def test_min(a, b):
        return min(a, b)

    def test_max(a, b):
        return max(a, b)

    def test_abs(a):
        return abs(a)

    # Test timedelta
    dt_obj1 = datetime.timedelta(7, 7, 7)
    dt_obj2 = datetime.timedelta(2, 2, 2)
    check_func(test_add, (dt_obj1, dt_obj2))
    check_func(test_sub, (dt_obj1, dt_obj2))
    check_func(test_mul, (dt_obj1, 5))
    check_func(test_mul, (5, dt_obj1))
    check_func(test_floordiv, (dt_obj1, dt_obj2))
    check_func(test_floordiv, (dt_obj1, 2))
    check_func(test_truediv, (dt_obj1, dt_obj2))
    check_func(test_truediv, (dt_obj1, 2))
    check_func(test_mod, (dt_obj1, dt_obj2))
    check_func(test_neg, (dt_obj1,))
    check_func(test_pos, (dt_obj1,))
    check_func(test_divmod, (dt_obj1, dt_obj2))

    # Test date
    date = datetime.date(2020, 1, 4)
    date2 = datetime.date(1999, 5, 2)
    td = datetime.timedelta(1, 2, 1)
    check_func(test_add, (date, td))
    check_func(test_add, (td, date))
    check_func(test_sub, (date, td))
    check_func(test_sub, (date, date2))
    check_func(test_min, (date, date2))
    check_func(test_min, (date2, date))
    check_func(test_max, (date, date2))
    check_func(test_max, (date2, date))

    # Test datetime
    dt = datetime.datetime(2020, 1, 20, 10, 20, 30, 40)
    dt2 = datetime.datetime(2019, 3, 8, 7, 12, 15, 20)
    td = datetime.timedelta(7, 7, 7)
    check_func(test_add, (dt, td))
    check_func(test_add, (td, dt))
    check_func(test_sub, (dt, td))
    check_func(test_sub, (dt, dt2))

    # Test series(dt64)
    Srange = pd.DatetimeIndex(
        [
            "2018-04-24 00:00:00",
            None,
            "2018-04-26 12:00:00",
            "2018-04-27 18:00:00",
            "2018-04-29 00:00:00",
        ],
        dtype="datetime64[ns]",
        freq=None,
    )
    Srange2 = pd.DatetimeIndex(
        [
            "2018-04-20 00:00:00",
            "2018-04-21 06:00:00",
            "2018-04-22 12:00:00",
            None,
            "2018-04-25 00:00:00",
        ],
        dtype="datetime64[ns]",
        freq=None,
    )
    S = pd.Series(Srange)
    S2 = pd.Series(Srange2)
    timestamp = pd.to_datetime("2018-04-24")
    dt_dt = datetime.datetime(2001, 1, 1)
    dt_td = datetime.timedelta(1, 1, 1)
    check_func(test_sub, (S, timestamp))
    check_func(test_sub, (S, dt_dt))
    check_func(test_sub, (S, dt_td))
    check_func(test_sub, (timestamp, S))
    check_func(test_sub, (dt_dt, S))
    check_func(test_add, (S, dt_td))
    check_func(test_add, (dt_td, S))
    check_func(test_sub, (S, S2))

    # Test series(timedelta64)
    tdS = pd.Series(
        (
            datetime.timedelta(6, 6, 6),
            datetime.timedelta(5, 5, 5),
            datetime.timedelta(4, 4, 4),
            None,
            datetime.timedelta(2, 2, 2),
        )
    )
    check_func(test_sub, (tdS, dt_td))
    check_func(test_sub, (dt_td, tdS))
    check_func(test_sub, (S, tdS))
    check_func(test_add, (S, tdS))
    check_func(test_add, (tdS, S))

    # Test Timestamp
    t1 = pd.Timestamp(1993, 12, 1)
    t2 = pd.Timestamp(2013, 1, 13)
    check_func(test_min, (t1, t2))
    check_func(test_max, (t2, t1))
    check_func(test_sub, (t2, t1))


def test_dt64_sub_output(memory_leak_check):
    """make sure output of Series(dt64) subtraction is Series(timedelta64) and supports
    dt.days properly (from forecast code issue)
    """

    def impl(S1, S2):
        return (S2 - S1).dt.days

    S1 = pd.to_datetime(
        pd.Series(
            [
                "2018-04-24 00:00:00",
                None,
                "2018-04-26 12:00:00",
                "2018-04-27 18:00:00",
                "2018-04-29 00:00:00",
            ]
        )
    )
    S2 = pd.to_datetime(
        pd.Series(
            [
                "2018-04-20 00:00:00",
                "2018-04-21 06:00:00",
                "2018-04-22 12:00:00",
                None,
                "2018-04-25 00:00:00",
            ]
        )
    )
    check_func(impl, (S1, S2), check_dtype=False)


@pytest.mark.slow
def test_datetime_timedelta_coerce(memory_leak_check):
    ts = datetime.timedelta(7, 7, 7)

    def f(df):
        df["ts"] = ts
        return df

    df1 = pd.DataFrame({"a": [1, 2, 3]})
    check_func(f, (df1,))


@pytest.mark.slow
def test_datetime_datetime_coerce(memory_leak_check):
    ts = datetime.datetime(2020, 1, 20, 10, 20, 30, 40)

    def f(df):
        df["ts"] = ts
        return df

    df1 = pd.DataFrame({"a": [1, 2, 3]})
    check_func(f, (df1,))


@pytest.mark.slow
def test_datetime_date_coerce(memory_leak_check):
    ts = datetime.date(2003, 1, 1)

    def f(df):
        df["ts"] = ts
        return df

    df1 = pd.DataFrame({"a": [1, 2, 3, 0, 1, 2, 5, 9]})
    check_func(f, (df1,))


@pytest.mark.slow
def test_datetime_date_constant_lowering(memory_leak_check):
    date = datetime.date(2004, 1, 1)

    def f():
        return date

    bodo_f = bodo.jit(f)
    val_ret = bodo_f()
    assert val_ret == date


@pytest.mark.slow
def test_timedelta_constant_lowering(memory_leak_check):
    timedelta = datetime.timedelta(3, 3, 3)

    def f():
        return timedelta

    bodo_f = bodo.jit(f)
    val_ret = bodo_f()
    assert val_ret == timedelta


@pytest.mark.slow
def test_datetime_datetime_constant_lowering(memory_leak_check):
    ts = datetime.datetime.now()

    def f():
        return ts

    bodo_f = bodo.jit(f)
    val_ret = bodo_f()
    assert val_ret == ts


@pytest.mark.slow
def test_timestamp_to_date(memory_leak_check):
    def test_impl(ts_val):
        return pd.to_datetime(ts_val)

    ts_val = pd.Timestamp("1998-12-01")
    check_func(test_impl, (ts_val,))


@pytest.mark.slow
def test_to_datetime_none(memory_leak_check):
    def test_impl(ts_val):
        return pd.to_datetime(ts_val)

    check_func(test_impl, (None,))


@pytest.mark.slow
def test_to_timedelta_none(memory_leak_check):
    def test_impl(ts_val):
        return pd.to_timedelta(ts_val)

    check_func(test_impl, (None,))


@pytest.mark.slow
def test_datetime_date_hash(memory_leak_check):
    date1 = datetime.date(2004, 1, 1)
    date2 = datetime.date(2004, 1, 2)
    date3 = datetime.date(2004, 1, 1)

    def impl(date1, date2, date3):
        d = dict()
        d[date1] = 1
        d[date2] = 2
        d[date3] = 3
        return d

    check_func(impl, (date1, date2, date3), dist_test=False)


@pytest.mark.slow
def test_datetime_timedelta_hash(memory_leak_check):
    td1 = datetime.timedelta(1)
    td2 = datetime.timedelta(2)
    td3 = datetime.timedelta(1)

    def impl(td1, td2, td3):
        d = dict()
        d[td1] = 1
        d[td2] = 2
        d[td3] = 3
        return d

    check_func(impl, (td1, td2, td3), dist_test=False)


@pytest.mark.slow
def test_timestamp_constant_lowering(memory_leak_check):
    t = pd.Timestamp("2012-06-18")

    def f():
        return t

    bodo_f = bodo.jit(f)
    val_ret = bodo_f()
    assert val_ret == t


@pytest.mark.slow
def test_timestamp_weekday(memory_leak_check):
    def test_impl(ts):
        return ts.weekday()

    for i in range(1, 21):
        ts = pd.Timestamp("2012-06-" + ("0" + str(i) if i < 10 else str(i)))
        check_func(test_impl, (ts,))


@pytest.mark.slow
def test_timestamp_day_name(memory_leak_check):
    def test_impl(ts):
        return ts.day_name()

    for i in range(1, 21):
        ts = pd.Timestamp("2021-06-" + ("0" + str(i) if i < 10 else str(i)))
        check_func(test_impl, (ts,))


@pytest.mark.slow
def test_timestamp_month_name(memory_leak_check):
    def test_impl(ts):
        return ts.month_name()

    for i in range(1, 12):
        ts = pd.Timestamp(f"2021-{'0' + str(i) if i < 10 else str(i)}-27")
        check_func(test_impl, (ts,))


@pytest.mark.slow
def test_timestamp_strftime(memory_leak_check):
    def test_impl(ts, fmt_string):
        return ts.strftime(fmt_string)

    ts = pd.Timestamp("2012-06-18")

    check_func(test_impl, (ts, "%m/%d/%Y, %H:%M:%S"))
    check_func(test_impl, (ts, "%Y"))


def test_timestamp_dt_strftime(memory_leak_check):
    def test_impl(S, fmt_string):
        return S.dt.strftime(fmt_string)

    S = pd.date_range("2020-01-03", "2020-02-04").to_series()

    check_func(test_impl, (S, "%m/%d/%Y, %H:%M:%S"))
    check_func(test_impl, (S, "%Y"))


def test_sub_add_timestamp_timedelta(memory_leak_check):
    def f_sub(date, timedelta):
        return date - timedelta

    def f_add(date, timedelta):
        return date + timedelta

    date = pd.Timestamp("2017-10-14")
    timedelta = datetime.timedelta(3, 3, 3)
    check_func(f_sub, (date, timedelta))
    check_func(f_add, (date, timedelta))


def test_datetime_comparisons_scalar(is_slow_run, memory_leak_check):
    """
    Test comparison operators of datetime module objects in Bodo for single values
    """

    def test_eq(a, b):
        return a == b

    def test_ne(a, b):
        return a != b

    def test_le(a, b):
        return a <= b

    def test_lt(a, b):
        return a < b

    def test_ge(a, b):
        return a >= b

    def test_gt(a, b):
        return a > b

    # Test timedelta
    dt_obj1 = datetime.timedelta(7, 7, 7)
    dt_obj2 = datetime.timedelta(2, 2, 2)
    check_func(test_eq, (dt_obj1, dt_obj2))
    if is_slow_run:
        check_func(test_ne, (dt_obj1, dt_obj2))
        check_func(test_le, (dt_obj1, dt_obj2))
        check_func(test_lt, (dt_obj1, dt_obj2))
        check_func(test_ge, (dt_obj1, dt_obj2))
        check_func(test_gt, (dt_obj1, dt_obj2))

    # test date
    date = datetime.date(2020, 1, 4)
    date2 = datetime.date(2020, 3, 1)
    check_func(test_eq, (date, date2))
    if is_slow_run:
        check_func(test_ne, (date, date2))
        check_func(test_le, (date, date2))
        check_func(test_lt, (date, date2))
        check_func(test_ge, (date, date2))
        check_func(test_gt, (date, date2))

    # compare timestamp and date
    t = pd.Timestamp("2020-03-01")
    check_func(test_eq, (t, date2))
    if is_slow_run:
        check_func(test_ne, (date2, t))
        check_func(test_le, (date, t))
        check_func(test_lt, (t, date2))
        check_func(test_ge, (date, t))
        check_func(test_gt, (t, date2))

    # datetime.datetime comparisons
    dt = datetime.datetime(2020, 1, 4, 10, 40, 55, 11)
    dt2 = datetime.datetime(2020, 1, 4, 11, 22, 12, 33)
    check_func(test_eq, (dt, dt2))
    if is_slow_run:
        check_func(test_ne, (dt, dt2))
        check_func(test_le, (dt, dt2))
        check_func(test_lt, (dt, dt2))
        check_func(test_ge, (dt, dt2))
        check_func(test_gt, (dt, dt2))


def test_function_to_datetime_string_array(memory_leak_check):
    """Test pd.to_datetime for an array of strings"""

    def f(S, formatting):
        return pd.to_datetime(S, format=formatting)

    S = pd.Series(["2019-01-01", "2019-01-02"] * 3)
    formatting = "%Y-%d-%m"
    check_func(f, (S.values, formatting))

    check_func(f, (S, formatting))


def test_function_to_datetime_scalar(memory_leak_check):
    def f(str1, formating):
        return pd.to_datetime(str1, format=formating)

    str1 = "2016-01-06"
    formating = "%Y-%d-%m"
    check_func(f, (str1, formating))


def test_function_to_datetime_array_int(memory_leak_check):
    """Test pd.to_datetime for an array of int"""

    def f(S):
        return pd.to_datetime(S, unit="D", origin=pd.Timestamp("1960-01-01"))

    S = pd.Series([1, 2, 3]).values
    check_func(f, (S,))


def test_function_to_datetime_infer_datetime_format(memory_leak_check):
    """Test unix to_datetime"""

    def f(S):
        return pd.to_datetime(S, infer_datetime_format=True)

    S = pd.Series(["3/11/2000", "3/12/2000", "3/13/2000"] * 1000)
    check_func(f, (S,))


@pytest.mark.slow
def test_function_to_datetime_unix_epoch(memory_leak_check):
    """Test unix to_datetime"""

    def f1():
        return pd.to_datetime(1490195805, unit="s")

    def f2(timein):
        return pd.to_datetime(timein, unit="s")

    def f3():
        return pd.to_datetime(1490195805433502912, unit="ns")

    timein = 1490195805
    check_func(f1, ())
    check_func(f2, (timein,))
    check_func(f3, ())


def test_datetime_comparisons_datetime_datetime(is_slow_run, memory_leak_check):
    """
    Test comparison operators of datetime module objects in Bodo.
    Comparison between array of datetime and datetime
    """

    def test_eq(a, b):
        return a == b

    def test_ne(a, b):
        return a != b

    def test_le(a, b):
        return a <= b

    def test_lt(a, b):
        return a < b

    def test_ge(a, b):
        return a >= b

    def test_gt(a, b):
        return a > b

    # test series(dt64) scalar cmp
    t = np.datetime64("2018-04-27").astype("datetime64[ns]")
    Srange = pd.DatetimeIndex(
        [
            "2018-04-24 00:00:00",
            None,
            "2018-04-26 12:00:00",
            "2018-04-27 18:00:00",
            None,
        ],
        dtype="datetime64[ns]",
        freq=None,
    )
    S = pd.Series(Srange)
    check_func(test_ge, (S, t))
    if is_slow_run:
        return
    check_func(test_ge, (t, S))
    check_func(test_gt, (t, S))
    check_func(test_gt, (S, t))
    check_func(test_le, (t, S))
    check_func(test_le, (S, t))
    check_func(test_lt, (t, S))
    check_func(test_lt, (S, t))
    check_func(test_ne, (t, S))
    check_func(test_ne, (S, t))
    check_func(test_eq, (t, S))
    check_func(test_eq, (S, t))


def test_datetime_comparisons_datetime_string(is_slow_run, memory_leak_check):
    """
    Test comparison operators of datetime module objects in Bodo.
    Comparison between datetime and string
    """

    def test_eq(a, b):
        return a == b

    def test_ne(a, b):
        return a != b

    def test_le(a, b):
        return a <= b

    def test_lt(a, b):
        return a < b

    def test_ge(a, b):
        return a >= b

    def test_gt(a, b):
        return a > b

    Srange = pd.DatetimeIndex(
        [
            "2018-04-24 00:00:00",
            None,
            "2018-04-26 12:00:00",
            "2018-04-27 18:00:00",
            None,
        ],
        dtype="datetime64[ns]",
        freq=None,
    )
    S = pd.Series(Srange)
    t_str = "2018-04-27"
    check_func(test_ge, (t_str, S))
    if is_slow_run:
        return
    check_func(test_ge, (S, t_str))
    check_func(test_gt, (t_str, S))
    check_func(test_gt, (S, t_str))
    check_func(test_le, (t_str, S))
    check_func(test_le, (S, t_str))
    check_func(test_lt, (t_str, S))
    check_func(test_lt, (S, t_str))
    check_func(test_ne, (t_str, S))
    check_func(test_ne, (S, t_str))
    check_func(test_eq, (t_str, S))
    check_func(test_eq, (S, t_str))


def test_datetime_comparisons_timedelta(is_slow_run, memory_leak_check):
    """
    Test comparison operators of datetime module objects in Bodo.
    Comparison between Series(timedelta64) and timedelta scalar
    """

    def test_eq(a, b):
        return a == b

    def test_ne(a, b):
        return a != b

    def test_le(a, b):
        return a <= b

    def test_lt(a, b):
        return a < b

    def test_ge(a, b):
        return a >= b

    def test_gt(a, b):
        return a > b

    # Test with TimeDelta
    TimeDeltas = pd.Series(
        [
            datetime.timedelta(1, 0, 0),
            datetime.timedelta(0, 7200, 0),
            datetime.timedelta(2, 0, 0),
            datetime.timedelta(-1, 3600, 0),
            None,
        ],
        dtype="timedelta64[ns]",
    )
    t_timedelta = datetime.timedelta(0, 7200, 0)
    check_func(test_ge, (t_timedelta, TimeDeltas))
    if is_slow_run:
        return
    check_func(test_ge, (TimeDeltas, t_timedelta))
    check_func(test_gt, (t_timedelta, TimeDeltas))
    check_func(test_gt, (TimeDeltas, t_timedelta))
    check_func(test_le, (t_timedelta, TimeDeltas))
    check_func(test_le, (TimeDeltas, t_timedelta))
    check_func(test_lt, (t_timedelta, TimeDeltas))
    check_func(test_lt, (TimeDeltas, t_timedelta))
    check_func(test_ne, (t_timedelta, TimeDeltas))
    check_func(test_ne, (TimeDeltas, t_timedelta))
    check_func(test_eq, (t_timedelta, TimeDeltas))
    check_func(test_eq, (TimeDeltas, t_timedelta))


def test_datetime_comparisons_datetime_list(is_slow_run, memory_leak_check):
    """
    Test comparison operators of datetime module objects in Bodo.
    Comparison between lists of datetime
    """

    def test_eq(a, b):
        return a == b

    def test_ne(a, b):
        return a != b

    def test_le(a, b):
        return a <= b

    def test_lt(a, b):
        return a < b

    def test_ge(a, b):
        return a >= b

    def test_gt(a, b):
        return a > b

    # test series(dt64) cmp
    Srange1 = pd.DatetimeIndex(
        [
            "2018-04-24 00:00:00",
            None,
            "2018-04-26 12:00:00",
            "2018-04-27 18:00:00",
            "2018-04-29 00:00:00",
        ],
        dtype="datetime64[ns]",
        freq=None,
    )
    Srange2 = pd.DatetimeIndex(
        [
            "2018-04-24 00:00:00",
            "2018-04-25 06:00:00",
            "2018-04-26 12:00:00",
            None,
            "2018-04-29 00:00:00",
        ],
        dtype="datetime64[ns]",
        freq=None,
    )
    S1 = pd.Series(Srange1)
    S2 = pd.Series(Srange2)
    check_func(test_ge, (S1, S2))
    if is_slow_run:
        return
    check_func(test_gt, (S1, S2))
    check_func(test_le, (S1, S2))
    check_func(test_lt, (S1, S2))
    check_func(test_ne, (S1, S2))
    check_func(test_eq, (S1, S2))


def test_datetime_comparisons_date(is_slow_run, memory_leak_check):
    """
    Test comparison operators of datetime module objects in Bodo.
    Comparison between list of dates and a scalar of date
    """

    def test_eq(a, b):
        return a == b

    def test_ne(a, b):
        return a != b

    def test_le(a, b):
        return a <= b

    def test_lt(a, b):
        return a < b

    def test_ge(a, b):
        return a >= b

    def test_gt(a, b):
        return a > b

    # test series(datetime.date)

    Srange = pd.DatetimeIndex(
        [
            "2018-04-24 00:00:00",
            None,
            "2018-04-26 12:00:00",
            "2018-04-27 18:00:00",
            "2018-04-28 00:00:00",
        ],
        dtype="datetime64[ns]",
        freq=None,
    )
    S = pd.Series(Srange.date)
    t = datetime.date(2018, 4, 27)
    check_func(test_ge, (S, t))
    if is_slow_run:
        return
    check_func(test_ge, (t, S))
    check_func(test_gt, (t, S))
    check_func(test_gt, (S, t))
    check_func(test_le, (t, S))
    check_func(test_le, (S, t))
    check_func(test_lt, (t, S))
    check_func(test_lt, (S, t))
    check_func(test_ne, (t, S))
    check_func(test_ne, (S, t))
    check_func(test_eq, (t, S))
    check_func(test_eq, (S, t))


def test_date_isin(memory_leak_check):
    """test Series.isin with datetime.date values"""

    def test_isin(S, vals):
        return S.isin(vals)

    S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    v = [
        datetime.date(2018, 1, 24),
        datetime.date(2011, 1, 3),
        datetime.date(2018, 4, 27),
    ]
    check_func(test_isin, (S, v))


@pytest.mark.slow
def test_datetime_boxing(memory_leak_check):
    """
    Test boxing and unboxing of datetime module object in Bodo
    """

    def test_impl(dt_obj):
        return dt_obj

    # Test timedelta
    td = datetime.timedelta(34535, 34959834, 948583858)
    check_func(test_impl, (td,))

    # Test date
    d = datetime.date(2020, 1, 8)
    check_func(test_impl, (d,))

    # Test datetime
    dt = datetime.datetime(2020, 1, 4, 13, 44, 33, 22)
    check_func(test_impl, (dt,))

    # test series(datetime.date)
    S = pd.Series(pd.date_range("2017-01-03", "2017-01-17").date)
    S[10] = None
    check_func(test_impl, (S,))


# ------------------------- Test datetime.timedelta ------------------------- #
@pytest.mark.slow
def test_datetime_timedelta_construct(memory_leak_check):
    """
    Test construction of datetime.timedelta object in Bodo
    """

    def test_impl():
        dt_obj = datetime.timedelta(34535, 34959834, 948583858)
        return dt_obj

    check_func(test_impl, ())


def test_datetime_timedelta_getattr(memory_leak_check):
    """
    Test getting attributes from datetime.timedelta object in Bodo
    """

    def test_days(dt_obj):
        return dt_obj.days

    def test_seconds(dt_obj):
        return dt_obj.seconds

    def test_microseconds(dt_obj):
        return dt_obj.microseconds

    dt_obj = datetime.timedelta(2, 55, 34)
    check_func(test_days, (dt_obj,))
    check_func(test_seconds, (dt_obj,))
    check_func(test_microseconds, (dt_obj,))


def test_datetime_timedelta_total_seconds(memory_leak_check):
    """
    Test total_seconds method of datetime.timedelta object in Bodo
    """

    def test_impl(dt_obj):
        return dt_obj.total_seconds()

    dt_obj = datetime.timedelta(1, 1, 1)
    check_func(test_impl, (dt_obj,))


# ------------------------- Test datetime.date ------------------------- #
@pytest.mark.slow
def test_datetime_date_construct(memory_leak_check):
    """
    Test construction of datetime.date object in Bodo
    """

    def test_impl():
        dt_obj = datetime.date(2020, 1, 4)
        return dt_obj

    check_func(test_impl, ())


def test_datetime_date_today(memory_leak_check):
    """
    Test datetime.date.today() classmethod
    """

    def test_impl():
        return datetime.date.today()

    assert bodo.jit(test_impl)() == test_impl()


def test_datetime_date_relative_import_today(memory_leak_check):
    """
    Test datetime.date.today() classmethod from a relative import
    """
    from datetime import date

    def test_impl():
        return date.today()

    assert bodo.jit(test_impl)() == test_impl()


def test_datetime_date_replace(memory_leak_check):
    def impl(date):
        return date.replace(year=1991, month=2, day=20)

    date = datetime.date(2020, 12, 1)
    check_func(impl, (date,))


def test_datetime_date_error(memory_leak_check):
    message = "year must be an integer"
    date = datetime.date(314, 1, 5)
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: date.replace(year=314.159))()
    message = "month must be an integer"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: date.replace(month=1.5))()
    message = "day must be an integer"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: date.replace(day=5.9))()


def test_datetime_date_fromordinal(memory_leak_check):
    """
    Test datetime.date.fromordinal() classmethod
    """

    def test_impl(n):
        return datetime.date.fromordinal(n)

    date = datetime.date(2013, 10, 5)
    n = date.toordinal()
    assert bodo.jit(test_impl)(n) == test_impl(n)


def test_datetime_date_methods(memory_leak_check):
    """
    Test methods of datetime.date object in Bodo
    """

    def test_weekday(date):
        return date.weekday()

    def test_toordinal(date):
        return date.toordinal()

    date = datetime.date(2013, 10, 5)
    check_func(test_weekday, (date,))
    check_func(test_toordinal, (date,))


def test_datetime_date_strftime(memory_leak_check):
    def impl(date, fmt_string):
        return date.strftime(fmt_string)

    date = datetime.date(2012, 6, 18)
    check_func(impl, (date, "%m/%d/%Y, %H:%M:%S"))
    check_func(impl, (date, "%Y"))


@pytest.fixture(
    params=[
        # Float values
        pd.Series([1.5, 100.42, 214213.242, 242112.341, 3435423.543, np.nan] * 2),
        pd.Series(
            [1.5, 100.42, 214213.242, 242112.341, 3435423.543, np.nan] * 2,
            dtype=np.float32,
        ),
        # Integer values
        pd.Series([1, 6, 1, -4, 11, 12] * 2, dtype="int8"),
        pd.Series([1, 6, 1, 4, 11, 12] * 2, dtype="uint8"),
        pd.Series([1, 6, 1, -30000, 11, 12] * 2, dtype="int16"),
        pd.Series([1, 6, 1, 40000, 11, 12] * 2, dtype="uint16"),
        pd.Series([1, 6000000, 1, -40000, 11, 12] * 2, dtype="int32"),
        pd.Series([1, 6000000, 1, 40000, 11, 12] * 2, dtype="uint32"),
        pd.Series([1, 6000000, 1, -40000, 11, 12] * 2, dtype="int64"),
        pd.Series([1, 6000000, 1, 40000, 11, 12] * 2, dtype="uint64"),
        # Nullable values
        pd.Series([1, 6, 1, -4, 11, None] * 2, dtype="Int8"),
        pd.Series([1, 6, 1, 4, 11, None] * 2, dtype="UInt8"),
        pd.Series([1, 6, 1, -30000, 11, None] * 2, dtype="Int16"),
        pd.Series([1, 6, 1, 40000, 11, None] * 2, dtype="UInt16"),
        pd.Series([1, 6000000, 1, -40000, 11, None] * 2, dtype="Int32"),
        pd.Series([1, 6000000, 1, 40000, 11, None] * 2, dtype="UInt32"),
        pd.Series([1, 6000000, 1, -40000, 11, None] * 2, dtype="Int64"),
        pd.Series([1, 6000000, 1, 40000, 11, None] * 2, dtype="UInt64"),
        # Boolean values
        pd.Series([False, True, False, True, True, None] * 2, dtype="boolean"),
        # Datetime64
        pd.Series(np.append([None], pd.date_range("20200101", periods=11))).astype(
            "datetime64[ns]"
        ),
        # Timedelta64
        pd.Series(
            np.append(
                [None],
                pd.timedelta_range(
                    start="12 days", end="12 days 3 hours 2 seconds", periods=11
                ),
            )
        ).astype("timedelta64[ns]"),
    ]
)
def datetime_convertable_series(request):
    return request.param


@pytest.mark.slow
def test_dt64_astype(datetime_convertable_series, memory_leak_check):
    """
    Test legal astype conversions to datetime64[ns]
    """

    def impl1(S):
        return S.astype(np.dtype("datetime64[ns]"))

    def impl2(S):
        return S.astype("datetime64[ns]")

    # Pandas can't handle null values in Nullable Boolean arrays.
    if isinstance(datetime_convertable_series.dtype, pd.BooleanDtype):
        py_output = datetime_convertable_series.astype("Int8").astype("datetime64[ns]")
    # Conversion from td64 -> dt64 is supported in Numpy but not pandas
    elif datetime_convertable_series.dtype == np.dtype("timedelta64[ns]"):
        py_output = pd.Series(
            datetime_convertable_series.values.astype("datetime64[ns]"),
            index=datetime_convertable_series.index,
        )
    else:
        py_output = None

    check_func(impl1, (datetime_convertable_series,), py_output=py_output)
    check_func(impl2, (datetime_convertable_series,), py_output=py_output)


@pytest.mark.slow
def test_dt64_str_astype(memory_leak_check):
    """
    Test legal astype conversions from string to datetime64[ns]
    """

    def impl1(S):
        return S.astype(np.dtype("datetime64[ns]"))

    def impl2(S):
        return S.astype("datetime64[ns]")

    S = pd.Series(["3/11/2000", "nan", "3/13/2000", "5/31/2021 03:23:53.231"] * 3)

    check_func(impl1, (S,))
    check_func(impl2, (S,))


@pytest.mark.slow
def test_dt64_date_astype(memory_leak_check):
    def impl1(S):
        return S.astype(np.dtype("datetime64[ns]"))

    def impl2(S):
        return S.astype("datetime64[ns]")

    S = pd.Series(np.append([None], pd.date_range("20200101", periods=11).date))
    check_func(impl1, (S,))
    check_func(impl2, (S,))


@pytest.mark.slow
def test_td64_astype(datetime_convertable_series, memory_leak_check):
    """
    Test legal astype conversions to timedelta[ns]
    """

    def impl1(S):
        return S.astype(np.dtype("timedelta64[ns]"))

    def impl2(S):
        return S.astype("timedelta64[ns]")

    # Pandas can't handle null values, so we can't properly test them.
    datetime_convertable_series = datetime_convertable_series.dropna()

    # Conversion from dt64 -> td64 is supported in Numpy but not pandas
    if datetime_convertable_series.dtype == np.dtype("datetime64[ns]"):
        py_output = pd.Series(
            datetime_convertable_series.values.astype("timedelta64[ns]"),
            index=datetime_convertable_series.index,
        )
    else:
        py_output = None

    check_func(impl1, (datetime_convertable_series,), py_output=py_output)
    check_func(impl2, (datetime_convertable_series,), py_output=py_output)


@pytest.mark.slow
def test_td64_str_astype(memory_leak_check):
    """
    Test legal astype conversions from string to timedelta64[ns]
    """

    def impl1(S):
        return S.astype(np.dtype("timedelta64[ns]"))

    def impl2(S):
        return S.astype("timedelta64[ns]")

    # TD64 seems to only work at converting strings of integers
    # without units.
    S = pd.Series(["2133421", "155", "32141", "-1542542"] * 3)

    check_func(impl1, (S,))
    check_func(impl2, (S,))


@pytest.mark.slow
def test_td64_max(memory_leak_check):
    def test_impl(S):
        return S.max()

    S = pd.Series(
        [
            np.timedelta64(10, "Y"),
            np.timedelta64(9, "M"),
            np.timedelta64(8, "W"),
        ]
        * 4
    )
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_td64_min(memory_leak_check):
    def test_impl(S):
        return S.min()

    S = pd.Series(
        [
            np.timedelta64(10, "Y"),
            np.timedelta64(9, "M"),
            np.timedelta64(8, "W"),
        ]
        * 4
    )
    check_func(test_impl, (S,))


def test_datetime_date_array_dtype(memory_leak_check):
    """
    Test using a datetime_data_array type as the type for a
    DataFrame.astype.
    """

    def test_impl(arr):
        return arr.dtype

    arr = pd.date_range(start="1998-04-24", end="1998-04-29", periods=5).date
    check_func(test_impl, (arr,))


@pytest.mark.slow
def test_dt64_astype_int64(memory_leak_check):
    """
    Test for astype from a dt64 Series to various numeric types.
    """

    def test_impl1(S):
        return S.astype(np.int64)

    def test_impl2(S):
        return S.astype(np.int64, copy=False)

    S = pd.Series(
        [
            np.datetime64("2007-01-01T03:30"),
            np.datetime64("2020-12-01T13:56:03.172"),
            np.datetime64("2021-03-03"),
        ]
        * 4
    )
    check_func(test_impl1, (S,))
    check_func(test_impl2, (S,))


@pytest.mark.slow
def test_dt64_astype_str(memory_leak_check):
    """
    Test for astype from a dt64 Series to string type.
    """

    # TODO: add tests for datetimes with nanosecond values if/when
    # str(timestamp) gets support for nonsecond values

    def test_impl1(S):
        return S.astype(str)

    def test_impl2(S):
        return S.astype(str, copy=False)

    def test_scalar_impl(dt_val):
        return str(dt_val)

    # There is also a minor bug in the current version of str(timestamp):
    #
    # bodo.jit(lambda: str(pd.Timestamp("2020-12-01 13:56:03.172")) )() == '2020-12-01 13:56:03'
    # str(pd.Timestamp("2020-12-01 13:56:03.172")) ==                      '2020-12-01 13:56:03.172000'
    #
    # we floor to second to avoid this issue.
    S = pd.Series(
        [
            np.datetime64("2007-01-01T03:30"),
            np.datetime64("2020-12-01T13:56:03.172"),
            np.datetime64("2021-03-03"),
        ]
        * 4
    ).dt.floor("s")
    arr = S.values

    # array implementation is currently failing, see BE-1388
    check_func(test_impl1, (S,))
    # check_func(test_impl1, (arr,))
    check_func(test_impl2, (S,))
    # check_func(test_impl2, (arr,))
    check_func(test_scalar_impl, (S[0],))


@pytest.mark.slow
def test_td64_astype_int64(memory_leak_check):
    """
    Test for astype from a td64 Series to various numeric types.
    """

    def test_impl1(S):
        return S.astype(np.int64)

    def test_impl2(S):
        return S.astype(np.int64, copy=False)

    S = pd.Series(
        [
            np.timedelta64(10, "Y"),
            np.timedelta64(9, "M"),
            np.timedelta64(8, "W"),
        ]
        * 4
    )
    check_func(test_impl1, (S,))
    check_func(test_impl2, (S,))


# ------------------------- Test datetime.datetime ------------------------- #
@pytest.mark.slow
def test_datetime_datetime_construct(memory_leak_check):
    """
    Test construction of datetime.datetime object in Bodo
    """

    def test_impl():
        dt_obj = datetime.datetime(2020, 1, 4, 13, 44, 33, 22)
        return dt_obj

    check_func(test_impl, ())


def test_datetime_datetime_getattr(memory_leak_check):
    """
    Test getting attributes from datetime.datetime object in Bodo
    """

    def test_year(dt_obj):
        return dt_obj.year

    def test_month(dt_obj):
        return dt_obj.month

    def test_day(dt_obj):
        return dt_obj.day

    def test_hour(dt_obj):
        return dt_obj.hour

    def test_minute(dt_obj):
        return dt_obj.minute

    def test_second(dt_obj):
        return dt_obj.second

    def test_microsecond(dt_obj):
        return dt_obj.microsecond

    dt_obj = datetime.datetime(2020, 1, 4, 13, 44, 33, 22)
    check_func(test_year, (dt_obj,))
    check_func(test_month, (dt_obj,))
    check_func(test_day, (dt_obj,))
    check_func(test_hour, (dt_obj,))
    check_func(test_minute, (dt_obj,))
    check_func(test_second, (dt_obj,))
    check_func(test_microsecond, (dt_obj,))


def test_datetime_datetime_methods(memory_leak_check):
    """
    Test methods of datetime.date object in Bodo
    """

    def test_weekday(dt):
        return dt.weekday()

    def test_toordinal(dt):
        return dt.toordinal()

    def test_date(dt):
        return dt.date()

    dt = datetime.datetime(2020, 1, 8, 11, 1, 30, 40)
    check_func(test_weekday, (dt,))
    check_func(test_toordinal, (dt,))
    check_func(test_date, (dt,))


def test_datetime_datetime_now(memory_leak_check):
    """
    Test datetime.datetime classmethod 'now'
    """

    def test_now():
        return datetime.datetime.now()

    # cannot test whether two results are exactly same because they are different in
    # microseconds due to the run time.
    b = bodo.jit(test_now)()
    p = test_now()
    assert (p - b) < datetime.timedelta(seconds=5)


def test_datetime_datetime_today(memory_leak_check):
    """
    Test datetime.datetime.today() classmethod
    """

    def test_today():
        return datetime.datetime.today()

    b = bodo.jit(test_today)()
    p = test_today()

    assert (p - b) < datetime.timedelta(minutes=1)


def test_timestamp_now(memory_leak_check):
    """
    Test pd.Timestamp classmethod 'now'
    """

    def test_now():
        return pd.Timestamp.now()

    # cannot test whether two results are exactly same because they are different in
    # microseconds due to the run time.
    b = bodo.jit(test_now)()
    p = test_now()
    assert (p - b) < pd.Timedelta(seconds=5)


def test_datetime_datetime_strptime(memory_leak_check):
    """
    Test datetime.datetime classmethod 'strptime'
    """

    def test_strptime(datetime_str, dtformat):
        return datetime.datetime.strptime(datetime_str, dtformat)

    datetime_str = "2020-01-08"
    dtformat = "%Y-%m-%d"
    check_func(test_strptime, (datetime_str, dtformat))


@pytest.mark.parametrize(
    "date",
    [
        pytest.param(datetime.date(1800, 1, 1), marks=pytest.mark.slow),
        pytest.param(datetime.date(1800, 6, 1), marks=pytest.mark.slow),
        pytest.param(datetime.date(1800, 10, 1), marks=pytest.mark.slow),
        pytest.param(datetime.date(1800, 1, 14), marks=pytest.mark.slow),
        pytest.param(datetime.date(1800, 6, 14), marks=pytest.mark.slow),
        pytest.param(datetime.date(1997, 1, 14), marks=pytest.mark.slow),
        pytest.param(datetime.date(2015, 1, 1), marks=pytest.mark.slow),
        pytest.param(datetime.date(2015, 6, 1), marks=pytest.mark.slow),
        pytest.param(datetime.date(2015, 10, 1), marks=pytest.mark.slow),
        pytest.param(datetime.date(2019, 1, 1), marks=pytest.mark.slow),
        pytest.param(datetime.date(2019, 6, 1), marks=pytest.mark.slow),
        pytest.param(datetime.date(2019, 10, 1), marks=pytest.mark.slow),
        pytest.param(datetime.date(2020, 1, 28), marks=pytest.mark.slow),
        pytest.param(datetime.date(2020, 6, 28), marks=pytest.mark.slow),
        datetime.date(2020, 10, 28),
        pytest.param(datetime.date(2025, 1, 28), marks=pytest.mark.slow),
        pytest.param(datetime.date(2025, 6, 28), marks=pytest.mark.slow),
        pytest.param(datetime.date(2025, 10, 28), marks=pytest.mark.slow),
    ],
)
def test_datetime_date_isocalendar(date, memory_leak_check):
    """Test datetime.date's isocalendar() method"""

    def test_impl(date):
        return date.isocalendar()

    check_func(test_impl, (date,))


# -------------------------  Test series.dt  -------------------------- #


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(pd.date_range(start="2019-01-24", end="2019-01-29", periods=5)),
            marks=pytest.mark.slow,
        ),
        # Test Series.dt.year for values less than 2000 (issue #343)
        pd.Series(pd.date_range(start="1998-04-24", end="1998-04-29", periods=5)),
        pd.Series(pd.date_range(start="5/20/2015", periods=5, freq="10N")),
        pytest.param(
            pd.Series(pd.date_range(start="1/1/2000", periods=5, freq="4Y")),
            marks=pytest.mark.slow,
        ),
    ]
)
def series_value(request):
    return request.param


@pytest.mark.parametrize("date_fields", bodo.hiframes.pd_timestamp_ext.date_fields)
def test_dt_extract(series_value, date_fields, memory_leak_check):
    """Test Series.dt extraction"""
    func_text = "def impl(S, date_fields):\n"
    func_text += "  return S.dt.{}\n".format(date_fields)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]
    check_func(impl, (series_value, date_fields), check_dtype=False)


@pytest.mark.parametrize("date_methods", bodo.hiframes.pd_timestamp_ext.date_methods)
def test_dt_date_methods(series_value, date_methods, memory_leak_check):
    """Test Series.dt datetime methods"""
    func_text = "def impl(S, date_methods):\n"
    func_text += "  return S.dt.{}()\n".format(date_methods)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    check_func(impl, (series_value, date_methods))


def test_dt_extract_date(series_value, memory_leak_check):
    """Test Series.dt.date extraction"""

    def impl(S):
        return S.dt.date

    check_func(impl, (series_value,))


# Copy the series info for isocalendar but remove the dates broken in pandas
@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(pd.date_range(start="2019-01-24", end="2019-01-29", periods=5)),
            marks=pytest.mark.slow,
        ),
        # Test Series.dt.year for values less than 2000 (issue #343)
        pd.Series(pd.date_range(start="1998-04-24", end="1998-04-29", periods=5)),
        pytest.param(
            pd.Series(pd.date_range(start="5/20/2015", periods=5, freq="10N")),
            marks=pytest.mark.slow,
        ),
    ]
)
def series_value_no_bad_dates(request):
    return request.param


def test_dt_isocalendar(series_value_no_bad_dates, memory_leak_check):
    """Test Series.dt.isocalendar()"""

    def impl(S):
        return S.dt.isocalendar()

    check_func(impl, (series_value_no_bad_dates,))


def test_dt_ceil_timestamp_min(series_value_no_bad_dates, memory_leak_check):
    def impl(S, freq):
        return S.dt.ceil(freq)

    freq = "min"
    check_func(impl, (series_value_no_bad_dates, freq))


@pytest.mark.slow
def test_dt_ceil_timestamp_others(series_value_no_bad_dates, memory_leak_check):
    """Test Series.dt.ceil()"""

    def impl(S, freq):
        return S.dt.ceil(freq)

    freqs = ["D", "H", "T", "S", "ms", "L", "U", "us", "N"]
    for freq in freqs:
        check_func(impl, (series_value_no_bad_dates, freq))


def test_dt_floor_timestamp_min(series_value_no_bad_dates, memory_leak_check):
    def impl(S, freq):
        return S.dt.floor(freq)

    freq = "min"
    check_func(impl, (series_value_no_bad_dates, freq))


@pytest.mark.slow
def test_dt_floor_timestamp_others(series_value_no_bad_dates, memory_leak_check):
    """Test Series.dt.floor()"""

    def impl(S, freq):
        return S.dt.floor(freq)

    freqs = ["D", "H", "T", "S", "ms", "L", "U", "us", "N"]
    for freq in freqs:
        check_func(impl, (series_value_no_bad_dates, freq))


@pytest.mark.slow
def test_timestamp_floor_edgecase(memory_leak_check):
    """Test Timestamp.floor() on a case that causes errors with np.remainder"""

    def impl(ts, freq):
        return ts.floor(freq)

    ts = pd.Timestamp("2010-01-01 11:24:02")
    freqs = ["D", "H", "T", "S", "ms", "L", "U", "us", "N"]
    for freq in freqs:
        check_func(impl, (ts, freq))


@pytest.mark.slow
def test_timestamp_ceil_edgecase(memory_leak_check):
    """Test Timestamp.ceil() on a case that causes errors with np.remainder"""

    def impl(ts, freq):
        return ts.ceil(freq)

    ts = pd.Timestamp("2010-01-01 11:24:02")
    freqs = ["D", "H", "T", "S", "ms", "L", "U", "us", "N"]
    for freq in freqs:
        check_func(impl, (ts, freq))


@pytest.mark.slow
def test_timestamp_round_edgecase(memory_leak_check):
    """Test Timestamp.round() on a case that causes errors with np.remainder"""

    def impl(ts, freq):
        return ts.round(freq)

    ts = pd.Timestamp("2010-01-01 11:24:02")
    freqs = ["D", "H", "T", "S", "ms", "L", "U", "us", "N"]
    for freq in freqs:
        check_func(impl, (ts, freq))


def test_dt_round_timestamp_min(series_value_no_bad_dates, memory_leak_check):
    def impl(S, freq):
        return S.dt.round(freq)

    freq = "min"
    check_func(impl, (series_value_no_bad_dates, freq))


@pytest.mark.slow
def test_dt_round_timestamp_others(series_value_no_bad_dates, memory_leak_check):
    """Test Series.dt.round()"""

    def impl(S, freq):
        return S.dt.round(freq)

    freqs = ["D", "H", "T", "S", "ms", "L", "U", "us", "N"]
    for freq in freqs:
        check_func(impl, (series_value_no_bad_dates, freq))


@pytest.mark.parametrize(
    "timedelta_fields", bodo.hiframes.pd_timestamp_ext.timedelta_fields
)
def test_dt_timedelta_fields(timedelta_fields, memory_leak_check):
    """Test Series.dt for timedelta64 fields"""
    func_text = "def impl(S, date_fields):\n"
    func_text += "  return S.dt.{}\n".format(timedelta_fields)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    S = pd.timedelta_range("1s", "1d", freq="s").to_series()
    check_func(impl, (S, timedelta_fields), check_dtype=False)


@pytest.mark.parametrize(
    "timedelta_methods", bodo.hiframes.pd_timestamp_ext.timedelta_methods
)
def test_dt_timedelta_methods(timedelta_methods, memory_leak_check):
    """Test Series.dt for timedelta64 methods"""
    func_text = "def impl(S, timedelta_methods):\n"
    func_text += "  return S.dt.{}()\n".format(timedelta_methods)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    S = pd.timedelta_range("1s", "1d", freq="s").to_series()
    check_func(impl, (S, timedelta_methods))


def test_series_dt64_timestamp_cmp(memory_leak_check):
    """Test Series.dt comparison with pandas.timestamp scalar"""

    def test_impl(S, t):
        return S == t

    def test_impl2(S):
        return S == "2018-04-24"

    def test_impl3(S):
        return "2018-04-24" == S

    def test_impl4(S, t):
        return S[S == t]

    S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    timestamp = pd.to_datetime("2018-04-24")
    t_string = "2018-04-24"

    # compare series(dt64) with a timestamp and a string
    check_func(test_impl, (S, timestamp))
    check_func(test_impl, (S, t_string))
    check_func(test_impl, (timestamp, S))
    check_func(test_impl, (t_string, S))

    # compare series(dt64) with a string constant
    check_func(test_impl2, (S,))
    check_func(test_impl3, (S,))

    # test filter
    check_func(test_impl4, (S, timestamp))
    check_func(test_impl4, (S, t_string))


@pytest.mark.smoke
def test_series_dt_getitem(memory_leak_check):
    """Test getitem of series(dt64)"""

    def test_impl(S):
        return S[0]

    S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    check_func(test_impl, (S,))
    # TODO: test datetime.date array when #522 is closed
    # S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5).date)
    # check_func(test_impl, (S,))


# -------------------------  series.dt errorchecking  -------------------------- #


def test_series_dt_type(memory_leak_check):
    """
    Test dt is called on series of type dt64
    """

    def impl(S):
        return S.dt.year

    S = pd.Series([" bbCD ", "ABC", " mCDm ", np.nan, "abcffcc", "", "A"])

    with pytest.raises(
        BodoError, match="Can only use .dt accessor with datetimelike values."
    ):
        bodo.jit(impl)(S)


# -----------------------------  Timestamp Test  ------------------------------ #


@pytest.mark.slow
def test_timestamp_constructors(memory_leak_check):
    """Test pd.Timestamp's different types of constructors"""

    def test_constructor_kw():
        # Test constructor with year/month/day passed as keyword arguments
        return pd.Timestamp(year=1998, month=2, day=3)

    def test_constructor_pos():
        # Test constructor with year/month/day passed as positional arguments
        return pd.Timestamp(1998, 2, 3)

    def test_constructor_input(dt):
        ts = pd.Timestamp(dt)
        return ts

    check_func(test_constructor_kw, ())
    check_func(test_constructor_pos, ())

    dt_d = datetime.date(2020, 2, 6)
    dt_dt = datetime.datetime(2017, 4, 26, 4, 55, 23, 32)

    check_func(test_constructor_input, (dt_d,))
    check_func(test_constructor_input, (dt_dt,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "time_num",
    [
        10210420,
        10210420.0,
    ],
)
def test_timestamp_number(time_num, memory_leak_check):
    def test_impl(time_num):
        return pd.Timestamp(time_num)

    check_func(test_impl, (time_num,))


@pytest.mark.slow
def test_timestamp_unit_constructor_error(memory_leak_check):
    def test_impl(time_int, unit):
        return pd.Timestamp(time_int, unit=unit)

    unit = "s"
    with pytest.raises(
        BodoError, match=r"pandas\.Timedelta\(\): unit argument must be a constant str"
    ):
        bodo.jit(test_impl)(10210420, unit)


def test_timestamp_dt64_ops(cmp_op, memory_leak_check):
    """
    Tests timestamp equality operators compared to dt64.
    """
    func = generate_comparison_ops_func(cmp_op)
    val1 = pd.Timestamp("2021-02-24")
    val2 = pd.Timestamp("2019-11-15")
    check_func(func, (val1, val1.to_numpy()))
    check_func(func, (val1.to_numpy(), val2))
    check_func(func, (val2, val1.to_numpy()))


def test_timedelta_td64_ops(cmp_op, memory_leak_check):
    """
    Tests timedelta equality operators compared to td64.
    """
    func = generate_comparison_ops_func(cmp_op)
    val1 = pd.Timedelta(days=1, seconds=42)
    val2 = pd.Timedelta(weeks=-2, days=11, microseconds=42)
    check_func(func, (val1, val1.to_numpy()))
    check_func(func, (val1.to_numpy(), val2))
    check_func(func, (val2, val1.to_numpy()))


@pytest.mark.slow
def test_timestamp_number_constructor_units(memory_leak_check):
    def test_impl1(time_val):
        return pd.Timestamp(time_val, unit="ns")

    def test_impl2(time_val):
        return pd.Timestamp(time_val, unit="us")

    def test_impl3(time_val):
        return pd.Timestamp(time_val, unit="ms")

    def test_impl4(time_val):
        return pd.Timestamp(time_val, unit="s")

    def test_impl5(time_val):
        return pd.Timestamp(time_val, unit="m")

    def test_impl6(time_val):
        return pd.Timestamp(time_val, unit="h")

    def test_impl7(time_val):
        return pd.Timestamp(time_val, unit="D")

    check_func(test_impl1, (10210420,))
    check_func(test_impl2, (10210420,))
    check_func(test_impl3, (10210420,))
    check_func(test_impl4, (10210420,))
    check_func(test_impl5, (10210420,))
    check_func(test_impl6, (102420,))
    check_func(test_impl7, (1020,))
    check_func(test_impl1, (10210420.0,))
    check_func(test_impl2, (1021042.20,))
    check_func(test_impl3, (1021032.420,))
    check_func(test_impl4, (102121.0420,))
    check_func(test_impl5, (121210.210420,))
    check_func(test_impl6, (1024.202020,))
    check_func(test_impl7, (1020.678,))


def test_pd_to_datetime(memory_leak_check):
    """Test pd.to_datetime on Bodo"""

    def test_scalar():
        return pd.to_datetime("2020-1-12")

    def test_input(input):
        return pd.to_datetime(input)

    date_str = "2020-1-12"
    check_func(test_scalar, ())
    check_func(test_input, (date_str,))

    date_arr = pd.Series(
        np.append(
            pd.date_range("2017-10-01", "2017-10-10").date,
            [None, datetime.date(2019, 3, 3)],
        )
    )
    check_func(test_input, (date_arr,))

    # input is already Series(dt64)
    S = pd.to_datetime(date_arr)
    check_func(test_input, (S,))

    # categorical string input
    S = pd.Series(
        ["2017-03-03", "2017-04-04", "2017-04-04", None, "2018-01-01", "2017-03-03"]
    ).astype("category")
    check_func(test_input, (S,))

    # TODO: Support following inputs
    # df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
    # date_str_arr = np.array(['1991-1-1', '1992-1-1', '1993-1-1'])
    # date_str_arr = ['1991-1-1', '1992-1-1', '1993-1-1']


def test_pd_to_timedelta(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(a):
        return pd.to_timedelta(a, "D")

    S = pd.Series([1.0, 2.2, np.nan, 4.2], [3, 1, 0, -2], name="AA")
    check_func(impl, (S,))


@pytest.mark.slow
def test_pd_to_timedelta_td64(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(S):
        return pd.to_timedelta(S)

    S = pd.timedelta_range(start="1 day", end="2 days", periods=100).to_series()
    check_func(impl, (S,))


@pytest.mark.slow
def test_pd_to_timedelta_datetime_td_arr(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(a):
        return pd.to_timedelta(a)

    arr = np.append(
        datetime.timedelta(days=5, seconds=4, weeks=4),
        [None, datetime.timedelta(microseconds=100000001213131, hours=5)] * 10,
    )
    check_func(impl, (arr,))


@pytest.mark.slow
def test_pd_to_timedelta_float_arr(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(a):
        return pd.to_timedelta(a, "D")

    arr = np.array([1.0, 2.2, np.nan, 4.2] * 5)
    check_func(impl, (arr,))


@pytest.mark.slow
def test_datetime_date_datetime_equality(memory_leak_check):
    """Tests == and != between datetime.date and datetime.datetime.
    These are never equal, even if they evaluate to the same value.
    We throw a warning because this is likely a user bug."""

    def impl1(d1, d2):
        return d1 == d2

    def impl2(d1, d2):
        return d1 != d2

    date_val1 = datetime.date(2021, 8, 31)
    date_val2 = datetime.date(2021, 9, 1)
    datetime_val = datetime.datetime(2021, 8, 31)

    # TODO: Check for the warning. This appears in python/ipython
    # but isn't showing up with pytest.

    # Test equals
    check_func(impl1, (date_val1, datetime_val))
    check_func(impl1, (datetime_val, date_val1))
    check_func(impl1, (date_val2, datetime_val))

    # Test not equals
    check_func(impl2, (date_val1, datetime_val))
    check_func(impl2, (datetime_val, date_val1))
    check_func(impl2, (date_val2, datetime_val))


@pytest.mark.slow
def test_pd_to_timedelta_int_arr(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(a):
        return pd.to_timedelta(a, "U")

    arr1 = pd.arrays.IntegerArray(
        np.array([115, 314, 0, 410214, 15] * 5, np.int64),
        np.array([False, False, False, False, True] * 5),
    )

    arr2 = np.array([115, 314, 0, 410214, 15] * 5)
    check_func(impl, (arr1,))
    check_func(impl, (arr2,))


@pytest.mark.slow
def test_pd_to_timedelta_string_arr(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(a):
        return pd.to_timedelta(a)

    arr1 = np.array(
        [
            "1 days 06:05:01.00003",
            "-2 days 23:15:31.4",
            "00:15:00.150015001",
            "00:00:00.000000001",
        ]
        * 5
    )
    arr2 = pd.array(
        [
            "1 days 06:05:01.00003",
            "-2 days 23:15:31.4",
            "00:15:00.150015001",
            "00:00:00.000000001",
        ]
        * 5
    )
    check_func(impl, (arr1,))
    check_func(impl, (arr2,))


@pytest.mark.slow
def test_pd_to_timedelta_float_scalar(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(a):
        return pd.to_timedelta(a, "D")

    check_func(impl, (2.2,))


@pytest.mark.slow
def test_pd_to_timedelta_int_scalar(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(a):
        return pd.to_timedelta(a, "L")

    check_func(impl, (100,))


@pytest.mark.slow
def test_pd_to_timedelta_str(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(string):
        return pd.to_timedelta(string)

    string = "1 days 06:05:01.00003"
    check_func(impl, (string,))


@pytest.mark.slow
def test_pd_to_timedelta_td(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(td):
        return pd.to_timedelta(td)

    td = pd.Timedelta(days=7, hours=7, nanoseconds=7)
    check_func(impl, (td,))


@pytest.mark.slow
def test_pd_to_timedelta_datetime(memory_leak_check):
    """Test pd.to_timedelta()"""

    def impl(td):
        return pd.to_timedelta(td)

    td = datetime.timedelta(days=7, hours=7, seconds=7, microseconds=90110)
    check_func(impl, (td,))


def test_extract(memory_leak_check):
    """Test extracting an attribute of timestamp"""

    def test_impl(s):
        return s.month

    ts = pd.Timestamp(datetime.datetime(2017, 4, 26).isoformat())
    check_func(test_impl, (ts,))


def test_timestamp_isoformat(memory_leak_check):
    def test_impl(ts):
        return ts.isoformat()

    def test_impl_sep(ts):
        return ts.isoformat(sep=" ")

    ts = pd.Timestamp(1513393355, unit="s")
    check_func(test_impl, (ts,))
    check_func(test_impl_sep, (ts,))


def test_timestamp_date(memory_leak_check):
    """Test timestamp's date() method"""

    def test_impl(s):
        return s.date()

    ts = pd.Timestamp(datetime.datetime(2017, 4, 26).isoformat())
    check_func(test_impl, (ts,))


@pytest.mark.parametrize(
    "ts",
    [
        pytest.param(pd.Timestamp(1800, 1, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1800, 6, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1800, 10, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1800, 1, 14), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1800, 6, 14), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1800, 10, 14), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1800, 1, 28), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1800, 6, 28), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1800, 10, 28), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1920, 1, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1920, 6, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1920, 10, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1920, 6, 28), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1920, 10, 28), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1952, 6, 14), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1952, 10, 14), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1997, 1, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1997, 6, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1997, 10, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(1997, 1, 14), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2015, 1, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2015, 6, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2015, 10, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2019, 1, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2019, 6, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2019, 10, 1), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2020, 1, 28), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2020, 6, 28), marks=pytest.mark.slow),
        pd.Timestamp(2020, 10, 28),
        pytest.param(pd.Timestamp(2025, 1, 28), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2025, 6, 28), marks=pytest.mark.slow),
        pytest.param(pd.Timestamp(2025, 10, 28), marks=pytest.mark.slow),
    ],
)
def test_timestamp_isocalendar(ts, memory_leak_check):
    """Test timestamp's isocalendar() method"""

    def test_impl(ts):
        return ts.isocalendar()

    check_func(test_impl, (ts,))


def test_dt_components(memory_leak_check):
    def impl(S):
        return S.dt.components

    S = pd.timedelta_range(start="1 day", end="2 days", periods=100).to_series()

    check_func(impl, (S,))


def test_dt_ceil_timedelta_min(memory_leak_check):
    def impl(S, freq):
        return S.dt.ceil(freq)

    S = pd.timedelta_range(start="1 day", end="2 days", periods=100).to_series()

    freq = "min"
    check_func(impl, (S, freq))


@pytest.mark.slow
def test_dt_ceil_timedelta_others(memory_leak_check):
    """Test Series.dt.ceil()"""

    def impl(S, freq):
        return S.dt.ceil(freq)

    S = pd.timedelta_range(start="1 day", end="2 days", periods=100).to_series()
    freqs = ["D", "H", "T", "S", "ms", "L", "U", "us", "N"]
    for freq in freqs:
        check_func(impl, (S, freq))


def test_dt_floor_timedelta_min(memory_leak_check):
    def impl(S, freq):
        return S.dt.floor(freq)

    S = pd.timedelta_range(start="1 day", end="2 days", periods=100).to_series()

    freq = "min"
    check_func(impl, (S, freq))


@pytest.mark.slow
def test_dt_floor_timedelta_others(memory_leak_check):
    """Test Series.dt.floor()"""

    def impl(S, freq):
        return S.dt.floor(freq)

    S = pd.timedelta_range(start="1 day", end="2 days", periods=100).to_series()
    freqs = ["D", "H", "T", "S", "ms", "L", "U", "us", "N"]
    for freq in freqs:
        check_func(impl, (S, freq))


def test_dt_round_timedelta_min(memory_leak_check):
    def impl(S, freq):
        return S.dt.round(freq)

    S = pd.timedelta_range(start="1 day", end="2 days", periods=100).to_series()

    freq = "min"
    check_func(impl, (S, freq))


@pytest.mark.slow
def test_dt_round_timedelta_others(memory_leak_check):
    """Test Series.dt.round()"""

    def impl(S, freq):
        return S.dt.round(freq)

    S = pd.timedelta_range(start="1 day", end="2 days", periods=100).to_series()
    freqs = ["D", "H", "T", "S", "ms", "L", "U", "us", "N"]
    for freq in freqs:
        check_func(impl, (S, freq))


# ------------------------- DatetimeIndex Testing  -------------------------- #


def _gen_str_date_df():
    rows = 10
    data = []
    random.seed(5)
    for row in range(rows):
        data.append(
            datetime.datetime(
                2017, random.randint(1, 12), random.randint(1, 28)
            ).isoformat()
        )
    return pd.DataFrame({"str_date": data})


dt_df = _gen_str_date_df()
dt_ser = pd.Series(pd.date_range(start="1998-04-24", end="1998-04-29", periods=10))


def test_datetime_index_ctor(memory_leak_check):
    """Test pd.DatetimeIndex constructors"""

    def test_impl_pos(S):
        return pd.DatetimeIndex(S)

    def test_impl_kw(S):
        return pd.DatetimeIndex(data=S)

    check_func(test_impl_pos, (dt_ser,))
    check_func(test_impl_kw, (dt_ser,))


def test_ts_map(memory_leak_check):
    def test_impl(A):
        return A.map(lambda x: x.hour)

    check_func(test_impl, (dt_ser,))


@pytest.mark.skip(reason="pending proper datetime.date() support")
def test_ts_map_date(memory_leak_check):
    def test_impl(A):
        return A.map(lambda x: x.date())[0]

    bodo_func = bodo.jit(test_impl)
    # TODO: Test after issue #530 is closed
    assert bodo_func(dt_ser) == test_impl(dt_ser)


@pytest.mark.slow
def test_ts_map_date2(memory_leak_check):
    def test_impl(df):
        return df.apply(lambda row: row.dt_ind.date(), axis=1)[0]

    bodo_func = bodo.jit(test_impl)
    dt_df["dt_ind"] = pd.DatetimeIndex(dt_df["str_date"])
    np.testing.assert_array_equal(bodo_func(dt_df), test_impl(dt_df))
    # TODO: Use check_func when #522 is closed
    # check_func(test_impl, (dt_df,))


@pytest.mark.skip(reason="pending proper datetime.date() support")
def test_ts_map_date_set(memory_leak_check):
    def test_impl(df):
        df["hpat_date"] = df.dt_ind.map(lambda x: x.date())
        return

    bodo_func = bodo.jit(test_impl)
    dt_df["dt_ind"] = pd.DatetimeIndex(dt_df["str_date"])
    bodo_func(dt_df)
    dt_df["pd_date"] = dt_df.dt_ind.map(lambda x: x.date())
    # TODO: Test after issue #530 is closed
    np.testing.assert_array_equal(dt_df["hpat_date"], dt_df["pd_date"])


def test_datetime_index_set(memory_leak_check):
    def test_impl(df):
        df["bodo"] = pd.DatetimeIndex(df["str_date"]).values
        return

    bodo_func = bodo.jit(test_impl)
    bodo_func(dt_df)
    dt_df["std"] = pd.DatetimeIndex(dt_df["str_date"])
    allequal = dt_df["std"].equals(dt_df["bodo"])
    assert allequal == True


def test_datetimeindex_str_comp(memory_leak_check):
    def test_impl(df):
        return (df.A >= "2011-10-23").values

    def test_impl2(df):
        return ("2011-10-23" <= df.A).values

    df = pd.DataFrame({"A": pd.DatetimeIndex(["2015-01-03", "2010-10-11"])})
    check_func(test_impl, (df,))
    check_func(test_impl2, (df,))


def test_datetimeindex_df(memory_leak_check):
    def test_impl(df):
        df = pd.DataFrame({"A": pd.DatetimeIndex(df["str_date"])})
        return df.A

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_array_equal(bodo_func(dt_df), test_impl(dt_df))
    # TODO: Use check_func when #522 is closed
    # check_func(test_impl, (dt_df,))


@pytest.mark.slow
def test_datetime_date_array_len(memory_leak_check):
    def test_impl(A):
        return len(A)

    A = np.array([datetime.date(2012, 1, 1), datetime.date(2011, 3, 3)] * 3)
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_datetime_timedelta_array_len(memory_leak_check):
    def test_impl(A):
        return len(A)

    A = np.array(
        [datetime.timedelta(34), datetime.timedelta(microseconds=43000000)] * 3
    )
    check_func(test_impl, (A,))


def test_calendar_month_abbbr():
    def impl():
        return calendar.month_abbr

    bodo_impl = bodo.jit(impl)
    abbrs = list(bodo_impl())
    assert abbrs == list(calendar.month_abbr)


@pytest.mark.parametrize(
    "dateindex",
    [
        pd.date_range("2004-11-12", periods=11),
        pytest.param(pd.date_range("2019-01-14", periods=11), marks=pytest.mark.slow),
        pytest.param(pd.date_range("2030-01-1", periods=11), marks=pytest.mark.slow),
    ],
)
def test_datetime_index_isocalendar(dateindex, memory_leak_check):
    def test_impl(I):
        return I.isocalendar()

    check_func(test_impl, (dateindex,))


@pytest.mark.parametrize(
    "timestamp",
    [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp(42, unit="s"),
    ],
)
def test_timestamp_datetime_conversion(timestamp, memory_leak_check):
    def test_impl(ts):
        return np.datetime64(ts, "ns")

    check_func(test_impl, (timestamp,))


def test_timestamp_datetime_conversion_error(memory_leak_check):
    timestamp = pd.Timestamp(42, unit="s")

    def test_impl(ts):
        return np.datetime64(ts, "s")

    with pytest.raises(BodoError, match="ns"):
        bodo.jit(test_impl)(timestamp)


@pytest.mark.parametrize(
    "timestamp",
    [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp(42, unit="s"),
    ],
)
def test_timestamp_to_datetime64(timestamp, memory_leak_check):
    def test_impl(ts):
        return ts.to_datetime64()

    check_func(test_impl, (timestamp,))


@pytest.mark.parametrize(
    "datetime",
    [
        np.datetime64("2020-01-01"),
        np.datetime64(42, "s"),
    ],
)
def test_datetime_timestamp_conversion(datetime, memory_leak_check):
    def test_impl(dt):
        return pd.Timestamp(dt)

    check_func(test_impl, (datetime,))


@pytest.mark.parametrize(
    "datetime",
    [
        np.datetime64("2019-01-01", "ns"),
        np.datetime64("2020-01-01", "ns"),
        np.datetime64("2030-01-01", "ns"),
    ],
)
@pytest.mark.parametrize(
    "timestamp",
    [
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2030-01-01"),
    ],
)
@pytest.mark.parametrize(
    "comparison_impl",
    [
        pytest.param(lambda a, b: a == b, id="eq"),
        pytest.param(lambda a, b: a != b, id="ne"),
        pytest.param(lambda a, b: a < b, id="lt"),
        pytest.param(lambda a, b: a <= b, id="le"),
        pytest.param(lambda a, b: a > b, id="gt"),
        pytest.param(lambda a, b: a >= b, id="ge"),
    ],
)
def test_datetime_compare_pd_timestamp(
    datetime, timestamp, comparison_impl, memory_leak_check
):
    check_func(comparison_impl, (datetime, timestamp))
    check_func(comparison_impl, (timestamp, datetime))
