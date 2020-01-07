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


# ------------------------- Test datetime.timedelta ------------------------- #
def test_datetime_timedelta_construct():
    """
    Test construction of datetime.timedelta object in Bodo
    """

    def test_impl():
        dt_obj = datetime.timedelta(34535, 34959834, 948583858)
        return dt_obj

    check_func(test_impl, ())


def test_datetime_timedelta_boxing():
    """
    Test boxing and unboxing of datetime.timedelta object in Bodo
    """

    def test_impl(dt_obj):
        return dt_obj

    dt_obj = datetime.timedelta(34535, 34959834, 948583858)
    check_func(test_impl, (dt_obj,))


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


def test_datetime_timedelta_operations():
    """
    Test operations of datetime.timedelta objects in Bodo
    """

    def test_add(dt_obj1, dt_obj2):
        return dt_obj1 + dt_obj2

    def test_sub(dt_obj1, dt_obj2):
        return dt_obj1 - dt_obj2

    def test_mul(dt_obj1, dt_obj2):
        return dt_obj1 * dt_obj2

    def test_floordiv(dt_obj1, dt_obj2):
        return dt_obj1 // dt_obj2

    def test_truediv(dt_obj1, dt_obj2):
        return dt_obj1 / dt_obj2

    def test_mod(dt_obj1, dt_obj2):
        return dt_obj1 % dt_obj2

    def test_eq(dt_obj1, dt_obj2):
        return dt_obj1 == dt_obj2

    def test_ne(dt_obj1, dt_obj2):
        return dt_obj1 != dt_obj2

    def test_le(dt_obj1, dt_obj2):
        return dt_obj1 <= dt_obj2

    def test_lt(dt_obj1, dt_obj2):
        return dt_obj1 < dt_obj2

    def test_ge(dt_obj1, dt_obj2):
        return dt_obj1 >= dt_obj2

    def test_gt(dt_obj1, dt_obj2):
        return dt_obj1 > dt_obj2

    def test_neg(dt_obj1):
        return -dt_obj1

    def test_pos(dt_obj1):
        return +dt_obj1

    def test_divmod(dt_obj1, dt_obj2):
        return divmod(dt_obj1, dt_obj2)

    def test_abs(dt_obj1):
        return abs(dt_obj1)

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
    check_func(test_eq, (dt_obj1, dt_obj2))
    check_func(test_ne, (dt_obj1, dt_obj2))
    check_func(test_le, (dt_obj1, dt_obj2))
    check_func(test_lt, (dt_obj1, dt_obj2))
    check_func(test_ge, (dt_obj1, dt_obj2))
    check_func(test_gt, (dt_obj1, dt_obj2))
    check_func(test_neg, (dt_obj1,))
    check_func(test_pos, (dt_obj1,))
    check_func(test_divmod, (dt_obj1, dt_obj2))


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


def test_datetime_date_operations():
    """
    Test operations of datetime.date and timedelta object in Bodo
    """

    def test_add(a, b):
        return a + b

    def test_sub(a, b):
        return a - b

    date = datetime.date(2020, 1, 4)
    date2 = datetime.date(1999, 5, 2)
    timedelta = datetime.timedelta(1, 2, 1)
    check_func(test_add, (date, timedelta))
    check_func(test_add, (timedelta, date))
    check_func(test_sub, (date, timedelta))
    check_func(test_sub, (date, date2))


def test_datetime_date_comparisons():
    """
    Test comparison operators of datetime.date object in Bodo
    """

    def test_eq(date, date2):
        return date == date2

    def test_le(date, date2):
        return date <= date2

    def test_lt(date, date2):
        return date < date2

    def test_ge(date, date2):
        return date >= date2

    def test_gt(date, date2):
        return date > date2

    date = datetime.date(2020, 1, 4)
    date2 = datetime.date(2020, 3, 1)
    check_func(test_eq, (date, date2))
    check_func(test_le, (date, date2))
    check_func(test_lt, (date, date2))
    check_func(test_ge, (date, date2))
    check_func(test_gt, (date, date2))


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


# ---------------------------------------------------------------------------- #


def test_series_dt64_scalar_cmp():
    t = np.datetime64("2018-04-27").astype("datetime64[ns]")

    def test_impl(S):
        return S >= t

    S = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    check_func(test_impl, (S,))


def test_series_dt64_cmp():
    def test_impl(S1, S2):
        return S1 >= S2

    S1 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    S2 = pd.Series(pd.date_range(start="2018-04-24", end="2018-04-29", periods=5))
    S2.values[3] = np.datetime64("2018-05-03").astype("datetime64[ns]")
    check_func(test_impl, (S1, S2))


def test_dt_year_before_2000():
    """Test Series.dt.year for values less than 2000 (issue #343)
    """

    def test_impl(S):
        return S.dt.year

    S = pd.Series(pd.date_range(start="1998-04-24", end="1998-04-29", periods=5))
    check_func(test_impl, (S,))


################################## Timestamp tests ###################################


def test_timestamp_constructor_kw():
    """Test pd.Timestamp() constructor with year/month/day passed as keyword arguments
    """

    def test_impl():
        return pd.Timestamp(year=1998, month=2, day=3)

    assert bodo.jit(test_impl)() == test_impl()


def test_timestamp_constructor_pos():
    """Test pd.Timestamp() constructor with year/month/day passed as positional
    arguments
    """

    def test_impl():
        return pd.Timestamp(1998, 2, 3)

    assert bodo.jit(test_impl)() == test_impl()


class TestDate(unittest.TestCase):
    def test_datetime_index(self):
        def test_impl(df):
            return pd.DatetimeIndex(df["str_date"]).values

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetime_index_kw(self):
        def test_impl(df):
            return pd.DatetimeIndex(data=df["str_date"]).values

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetime_arg(self):
        def test_impl(A):
            return A

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df["str_date"]).to_series()
        np.testing.assert_array_equal(bodo_func(A), test_impl(A))

    def test_datetime_getitem(self):
        def test_impl(A):
            return A[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df["str_date"]).to_series()
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_ts_map(self):
        def test_impl(A):
            return A.map(lambda x: x.hour)

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df["str_date"]).to_series()
        np.testing.assert_array_equal(bodo_func(A), test_impl(A))

    @unittest.skip("pending proper datatime.date() support")
    def test_ts_map_date(self):
        def test_impl(A):
            return A.map(lambda x: x.date())[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df["str_date"]).to_series()
        np.testing.assert_array_equal(bodo_func(A), test_impl(A))

    def test_ts_map_date2(self):
        def test_impl(df):
            return df.apply(lambda row: row.dt_ind.date(), axis=1)[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        df["dt_ind"] = pd.DatetimeIndex(df["str_date"])
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    @unittest.skip("pending proper datatime.date() support")
    def test_ts_map_date_set(self):
        def test_impl(df):
            df["hpat_date"] = df.dt_ind.map(lambda x: x.date())

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        df["dt_ind"] = pd.DatetimeIndex(df["str_date"])
        bodo_func(df)
        df["pd_date"] = df.dt_ind.map(lambda x: x.date())
        np.testing.assert_array_equal(df["hpat_date"], df["pd_date"])

    def test_date_series_unbox(self):
        def test_impl(A):
            return A[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df["str_date"]).to_series().map(lambda x: x.date())
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_date_series_unbox2(self):
        def test_impl(A):
            return A[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        # TODO: index of date values
        A = (
            pd.DatetimeIndex(df["str_date"])
            .map(lambda x: x.date())
            .to_series()
            .reset_index(drop=True)
        )
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_datetime_index_set(self):
        def test_impl(df):
            df["bodo"] = pd.DatetimeIndex(df["str_date"]).values

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        bodo_func(df)
        df["std"] = pd.DatetimeIndex(df["str_date"])
        allequal = df["std"].equals(df["bodo"])
        self.assertTrue(allequal)

    @unittest.skip("pending proper datatime.datetime() support")
    def test_timestamp(self):
        def test_impl():
            dt = datetime.datetime(2017, 4, 26)
            ts = pd.Timestamp(dt)
            return (
                ts.day
                + ts.hour
                + ts.microsecond
                + ts.month
                + ts.nanosecond
                + ts.second
                + ts.year
            )

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_extract(self):
        def test_impl(s):
            return s.month

        bodo_func = bodo.jit(test_impl)
        ts = pd.Timestamp(datetime.datetime(2017, 4, 26).isoformat())
        month = bodo_func(ts)
        self.assertEqual(month, 4)

    def test_timestamp_date(self):
        def test_impl(s):
            return s.date()

        bodo_func = bodo.jit(test_impl)
        ts = pd.Timestamp(datetime.datetime(2017, 4, 26).isoformat())
        self.assertEqual(bodo_func(ts), test_impl(ts))

    def test_datetimeindex_str_comp(self):
        def test_impl(df):
            return (df.A >= "2011-10-23").values

        df = pd.DataFrame({"A": pd.DatetimeIndex(["2015-01-03", "2010-10-11"])})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetimeindex_str_comp2(self):
        def test_impl(df):
            return ("2011-10-23" <= df.A).values

        df = pd.DataFrame({"A": pd.DatetimeIndex(["2015-01-03", "2010-10-11"])})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetime_index_df(self):
        def test_impl(df):
            df = pd.DataFrame({"A": pd.DatetimeIndex(df["str_date"])})
            return df.A

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetime_series_dt_date(self):
        def test_impl(A):
            return A.dt.date

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df["str_date"]).to_series()
        # TODO: fix index and name
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def test_datetime_series_dt_year(self):
        def test_impl(A):
            return A.dt.year

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df["str_date"]).to_series()
        # TODO: fix index and name
        pd.testing.assert_series_equal(bodo_func(A), test_impl(A))

    def _gen_str_date_df(self):
        rows = 10
        data = []
        for row in range(rows):
            data.append(
                datetime.datetime(
                    2017, random.randint(1, 12), random.randint(1, 28)
                ).isoformat()
            )
        return pd.DataFrame({"str_date": data})


if __name__ == "__main__":
    unittest.main()
