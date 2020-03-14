# Copyright (C) 2019 Bodo Inc. All rights reserved.
import unittest
import pandas as pd
import numpy as np
from math import sqrt
import numba
import bodo
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    count_parfor_OneD_Vars,
    count_array_OneD_Vars,
    dist_IR_contains,
    check_func,
)
import datetime
import random
import pytest
from bodo.utils.typing import BodoError


# ------------------------- Test datetime OPs ------------------------- #
def test_datetime_operations():
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
    S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    S2 = pd.Series(pd.date_range(start="2018-04-20", end="2018-04-25", periods=5))
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
            datetime.timedelta(3, 3, 3),
            datetime.timedelta(2, 2, 2),
        )
    )
    check_func(test_sub, (tdS, dt_td))
    check_func(test_sub, (dt_td, tdS))
    check_func(test_sub, (S, tdS))
    check_func(test_add, (S, tdS))
    check_func(test_add, (tdS, S))


def test_datetime_comparisons():
    """
    Test comparison operators of datetime module objects in Bodo
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
    check_func(test_ne, (dt_obj1, dt_obj2))
    check_func(test_le, (dt_obj1, dt_obj2))
    check_func(test_lt, (dt_obj1, dt_obj2))
    check_func(test_ge, (dt_obj1, dt_obj2))
    check_func(test_gt, (dt_obj1, dt_obj2))

    # test date
    date = datetime.date(2020, 1, 4)
    date2 = datetime.date(2020, 3, 1)
    check_func(test_eq, (date, date2))
    check_func(test_ne, (date, date2))
    check_func(test_le, (date, date2))
    check_func(test_lt, (date, date2))
    check_func(test_ge, (date, date2))
    check_func(test_gt, (date, date2))

    # datetime.datetime comparisons
    dt = datetime.datetime(2020, 1, 4, 10, 40, 55, 11)
    dt2 = datetime.datetime(2020, 1, 4, 11, 22, 12, 33)
    check_func(test_eq, (dt, dt2))
    check_func(test_ne, (dt, dt2))
    check_func(test_le, (dt, dt2))
    check_func(test_lt, (dt, dt2))
    check_func(test_ge, (dt, dt2))
    check_func(test_gt, (dt, dt2))

    # test series(dt64) scalar cmp
    t = np.datetime64("2018-04-27").astype("datetime64[ns]")
    S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    check_func(test_ge, (S, t))
    check_func(test_ge, (t, S))

    # test series(dt64) cmp
    S1 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    S2 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    check_func(test_ge, (S1, S2))

    # test series(datetime.date)
    S = pd.Series(pd.date_range("2017-01-03", "2017-01-07").date)
    t = datetime.date(2017, 1, 4)
    check_func(test_ge, (S, t))


def test_datetime_boxing():
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
def test_datetime_timedelta_construct():
    """
    Test construction of datetime.timedelta object in Bodo
    """

    def test_impl():
        dt_obj = datetime.timedelta(34535, 34959834, 948583858)
        return dt_obj

    check_func(test_impl, ())


def test_datetime_timedelta_getattr():
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


def test_datetime_timedelta_total_seconds():
    """
    Test total_seconds method of datetime.timedelta object in Bodo
    """

    def test_impl(dt_obj):
        return dt_obj.total_seconds()

    dt_obj = datetime.timedelta(1, 1, 1)
    check_func(test_impl, (dt_obj,))


# ------------------------- Test datetime.date ------------------------- #
def test_datetime_date_construct():
    """
    Test construction of datetime.date object in Bodo
    """

    def test_impl():
        dt_obj = datetime.date(2020, 1, 4)
        return dt_obj

    check_func(test_impl, ())


def test_datetime_date_today():
    """
    Test datetime.date.today() classmethod
    """

    def test_impl():
        return datetime.date.today()

    assert bodo.jit(test_impl)() == test_impl()


def test_datetime_date_fromordinal():
    """
    Test datetime.date.fromordinal() classmethod
    """

    def test_impl(n):
        return datetime.date.fromordinal(n)

    date = datetime.date(2013, 10, 5)
    n = date.toordinal()
    assert bodo.jit(test_impl)(n) == test_impl(n)


def test_datetime_date_methods():
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


# ------------------------- Test datetime.datetime ------------------------- #
def test_datetime_datetime_construct():
    """
    Test construction of datetime.datetime object in Bodo
    """

    def test_impl():
        dt_obj = datetime.datetime(2020, 1, 4, 13, 44, 33, 22)
        return dt_obj

    check_func(test_impl, ())


def test_datetime_datetime_getattr():
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


def test_datetime_datetime_methods():
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


def test_datetime_datetime_now():
    """
    Test datetime.datetime classmethod 'now'
    """

    def test_now():
        return datetime.datetime.now()

    dt = datetime.datetime(2020, 1, 8, 11, 1, 30, 40)
    # cannot test whether two results are exactly same because they are different in
    # microseconds due to the run time.
    b = bodo.jit(test_now)()
    p = test_now()
    assert (p - b) < datetime.timedelta(seconds=5)


def test_datetime_datetime_strptime():
    """
    Test datetime.datetime classmethod 'strptime'
    """

    def test_strptime(datetime_str, dtformat):
        return datetime.datetime.strptime(datetime_str, dtformat)

    datetime_str = "2020-01-08"
    dtformat = "%Y-%m-%d"
    check_func(test_strptime, (datetime_str, dtformat))


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
def test_dt_extract(series_value, date_fields):
    """Test Series.dt extraction
    """
    func_text = "def impl(S, date_fields):\n"
    func_text += "  return S.dt.{}\n".format(date_fields)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    check_func(impl, (series_value, date_fields))


def test_dt_extract_date(series_value):
    """Test Series.dt.date extraction
    """

    def impl(S):
        return S.dt.date

    check_func(impl, (series_value,))


@pytest.mark.parametrize("timedelta_fields", bodo.hiframes.pd_timestamp_ext.timedelta_fields)
def test_dt_timedelta_fileds(timedelta_fields):
    """Test Series.dt for timedelta64 fields
    """
    func_text = "def impl(S, date_fields):\n"
    func_text += "  return S.dt.{}\n".format(timedelta_fields)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]

    S = pd.timedelta_range("1s", "1d", freq="s").to_series()
    check_func(impl, (S, timedelta_fields))


def test_series_dt64_timestamp_cmp():
    """Test Series.dt comparison with pandas.timestamp scalar
    """

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


def test_series_dt_getitem():
    """ Test getitem of series(dt64)
    """

    def test_impl(S):
        return S[0]

    S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    check_func(test_impl, (S,))
    # TODO: test datetime.date array when #522 is closed
    # S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5).date)
    # check_func(test_impl, (S,))


# -------------------------  series.dt errorchecking  -------------------------- #


def test_series_dt_type():
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


def test_timestamp_constructors():
    """ Test pd.Timestamp's different types of constructors
    """

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


def test_pd_to_datetime():
    """Test pd.to_datetime on Bodo
    """

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

    # TODO: Support following inputs
    # df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
    # date_str_arr = np.array(['1991-1-1', '1992-1-1', '1993-1-1'])
    # date_str_arr = ['1991-1-1', '1992-1-1', '1993-1-1']


def test_pd_to_timedelta():
    """Test pd.to_timedelta()
    """

    def impl(a):
        return pd.to_timedelta(a, "D")

    S = pd.Series([1.0, 2.2, np.nan, 4.2], [3, 1, 0, -2], name="AA")
    check_func(impl, (S,))


def test_extract():
    """ Test extracting an attribute of timestamp
    """

    def test_impl(s):
        return s.month

    ts = pd.Timestamp(datetime.datetime(2017, 4, 26).isoformat())
    check_func(test_impl, (ts,))


def test_timestamp_date():
    """ Test timestamp's date() method
    """

    def test_impl(s):
        return s.date()

    ts = pd.Timestamp(datetime.datetime(2017, 4, 26).isoformat())
    check_func(test_impl, (ts,))


# ------------------------- DatetimeIndex Testing  -------------------------- #


def _gen_str_date_df():
    rows = 10
    data = []
    for row in range(rows):
        data.append(
            datetime.datetime(
                2017, random.randint(1, 12), random.randint(1, 28)
            ).isoformat()
        )
    return pd.DataFrame({"str_date": data})


dt_df = _gen_str_date_df()
dt_ser = pd.Series(pd.date_range(start="1998-04-24", end="1998-04-29", periods=10))


def test_datetime_index_ctor():
    """ Test pd.DatetimeIndex constructors
    """

    def test_impl_pos(S):
        return pd.DatetimeIndex(S)

    def test_impl_kw(S):
        return pd.DatetimeIndex(data=S)

    check_func(test_impl_pos, (dt_ser,))
    check_func(test_impl_kw, (dt_ser,))


def test_ts_map():
    def test_impl(A):
        return A.map(lambda x: x.hour)

    check_func(test_impl, (dt_ser,))


@pytest.mark.skip(reason="pending proper datetime.date() support")
def test_ts_map_date():
    def test_impl(A):
        return A.map(lambda x: x.date())[0]

    bodo_func = bodo.jit(test_impl)
    # TODO: Test after issue #530 is closed
    assert bodo_func(dt_ser) == test_impl(dt_ser)


def test_ts_map_date2():
    def test_impl(df):
        return df.apply(lambda row: row.dt_ind.date(), axis=1)[0]

    bodo_func = bodo.jit(test_impl)
    dt_df["dt_ind"] = pd.DatetimeIndex(dt_df["str_date"])
    np.testing.assert_array_equal(bodo_func(dt_df), test_impl(dt_df))
    # TODO: Use check_func when #522 is closed
    # check_func(test_impl, (dt_df,))


@pytest.mark.skip(reason="pending proper datetime.date() support")
def test_ts_map_date_set():
    def test_impl(df):
        df["hpat_date"] = df.dt_ind.map(lambda x: x.date())
        return

    bodo_func = bodo.jit(test_impl)
    dt_df["dt_ind"] = pd.DatetimeIndex(dt_df["str_date"])
    bodo_func(dt_df)
    dt_df["pd_date"] = dt_df.dt_ind.map(lambda x: x.date())
    # TODO: Test after issue #530 is closed
    np.testing.assert_array_equal(dt_df["hpat_date"], dt_df["pd_date"])


def test_datetime_index_set():
    def test_impl(df):
        df["bodo"] = pd.DatetimeIndex(df["str_date"]).values
        return

    bodo_func = bodo.jit(test_impl)
    bodo_func(dt_df)
    dt_df["std"] = pd.DatetimeIndex(dt_df["str_date"])
    allequal = dt_df["std"].equals(dt_df["bodo"])
    assert allequal == True


def test_datetimeindex_str_comp():
    def test_impl(df):
        return (df.A >= "2011-10-23").values

    def test_impl2(df):
        return ("2011-10-23" <= df.A).values

    df = pd.DataFrame({"A": pd.DatetimeIndex(["2015-01-03", "2010-10-11"])})
    check_func(test_impl, (df,))
    check_func(test_impl2, (df,))


def test_datetimeindex_df():
    def test_impl(df):
        df = pd.DataFrame({"A": pd.DatetimeIndex(df["str_date"])})
        return df.A

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_array_equal(bodo_func(dt_df), test_impl(dt_df))
    # TODO: Use check_func when #522 is closed
    # check_func(test_impl, (dt_df,))
