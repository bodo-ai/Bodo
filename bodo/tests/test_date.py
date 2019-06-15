import unittest
import pandas as pd
import numpy as np
from math import sqrt
import numba
import bodo
from bodo.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                            count_parfor_OneDs, count_array_OneDs,
                            count_parfor_OneD_Vars, count_array_OneD_Vars,
                            dist_IR_contains)
from datetime import datetime
import random


class TestDate(unittest.TestCase):

    def test_datetime_index(self):
        def test_impl(df):
            return pd.DatetimeIndex(df['str_date']).values

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetime_index_kw(self):
        def test_impl(df):
            return pd.DatetimeIndex(data=df['str_date']).values

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetime_arg(self):
        def test_impl(A):
            return A

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        np.testing.assert_array_equal(bodo_func(A), test_impl(A))

    def test_datetime_getitem(self):
        def test_impl(A):
            return A[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_ts_map(self):
        def test_impl(A):
            return A.map(lambda x: x.hour)

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        np.testing.assert_array_equal(bodo_func(A), test_impl(A))

    @unittest.skip("pending proper datatime.date() support")
    def test_ts_map_date(self):
        def test_impl(A):
            return A.map(lambda x: x.date())[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        np.testing.assert_array_equal(bodo_func(A), test_impl(A))

    def test_ts_map_date2(self):
        def test_impl(df):
            return df.apply(lambda row: row.dt_ind.date(), axis=1)[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        df['dt_ind'] = pd.DatetimeIndex(df['str_date'])
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    @unittest.skip("pending proper datatime.date() support")
    def test_ts_map_date_set(self):
        def test_impl(df):
            df['hpat_date'] = df.dt_ind.map(lambda x: x.date())

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        df['dt_ind'] = pd.DatetimeIndex(df['str_date'])
        bodo_func(df)
        df['pd_date'] = df.dt_ind.map(lambda x: x.date())
        np.testing.assert_array_equal(df['hpat_date'], df['pd_date'])

    def test_date_series_unbox(self):
        def test_impl(A):
            return A[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series().map(lambda x: x.date())
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_date_series_unbox2(self):
        def test_impl(A):
            return A[0]

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        # TODO: index of date values
        A = pd.DatetimeIndex(df['str_date']).map(
            lambda x: x.date()).to_series().reset_index(drop=True)
        self.assertEqual(bodo_func(A), test_impl(A))

    def test_datetime_index_set(self):
        def test_impl(df):
            df['bodo'] = pd.DatetimeIndex(df['str_date']).values

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        bodo_func(df)
        df['std'] = pd.DatetimeIndex(df['str_date'])
        allequal = (df['std'].equals(df['bodo']))
        self.assertTrue(allequal)

    def test_timestamp(self):
        def test_impl():
            dt = datetime(2017, 4, 26)
            ts = pd.Timestamp(dt)
            return ts.day + ts.hour + ts.microsecond + ts.month + ts.nanosecond + ts.second + ts.year

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_extract(self):
        def test_impl(s):
            return s.month

        bodo_func = bodo.jit(test_impl)
        ts = pd.Timestamp(datetime(2017, 4, 26).isoformat())
        month = bodo_func(ts)
        self.assertEqual(month, 4)

    def test_timestamp_date(self):
        def test_impl(s):
            return s.date()

        bodo_func = bodo.jit(test_impl)
        ts = pd.Timestamp(datetime(2017, 4, 26).isoformat())
        self.assertEqual(bodo_func(ts), test_impl(ts))

    def test_datetimeindex_str_comp(self):
        def test_impl(df):
            return (df.A >= '2011-10-23').values

        df = pd.DataFrame({'A': pd.DatetimeIndex(['2015-01-03', '2010-10-11'])})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetimeindex_str_comp2(self):
        def test_impl(df):
            return ('2011-10-23' <= df.A).values

        df = pd.DataFrame({'A': pd.DatetimeIndex(['2015-01-03', '2010-10-11'])})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetime_index_df(self):
        def test_impl(df):
            df = pd.DataFrame({'A': pd.DatetimeIndex(df['str_date'])})
            return df.A

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_datetime_series_dt_date(self):
        def test_impl(A):
            return A.dt.date

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        # TODO: fix index and name
        pd.testing.assert_series_equal(
            bodo_func(A), test_impl(A).reset_index(drop=True),
            check_names=False)

    def test_datetime_series_dt_year(self):
        def test_impl(A):
            return A.dt.year

        bodo_func = bodo.jit(test_impl)
        df = self._gen_str_date_df()
        A = pd.DatetimeIndex(df['str_date']).to_series()
        # TODO: fix index and name
        pd.testing.assert_series_equal(
            bodo_func(A), test_impl(A).reset_index(drop=True),
            check_names=False)

    def _gen_str_date_df(self):
        rows = 10
        data = []
        for row in range(rows):
            data.append(datetime(2017, random.randint(1,12), random.randint(1,28)).isoformat())
        return pd.DataFrame({'str_date' : data})

if __name__ == "__main__":
    unittest.main()
