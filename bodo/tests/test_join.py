"""Test join operations like df.merge(), df.join(), pd.merge_asof() ...
"""
import unittest
import os
import pandas as pd
import numpy as np
import random
import string
import pyarrow.parquet as pq
from pandas.api.types import CategoricalDtype
import numba
import bodo
from bodo.libs.str_arr_ext import StringArray
from bodo.tests.test_utils import (count_array_REPs, count_parfor_REPs,
                            count_parfor_OneDs, count_array_OneDs, dist_IR_contains,
                            get_start_end)
import pytest


@pytest.mark.parametrize('df1', [pd.DataFrame({'A': [1, 11, 3]}),
    pd.DataFrame({'A': [1, 11, 3], 'B': [4, 5, 1]}),
    pd.DataFrame({'A': [1, 11, 3], 'B': [4, 5, 1], 'C': [-1, 3, 4]})])
@pytest.mark.parametrize('df2', [pd.DataFrame({'A': [-1, 1, 3]}),
    pd.DataFrame({'A': [-1, 1, 3], 'B': [-1, 0, 1]}),
    pd.DataFrame({'A': [-1, 1, 3], 'B': [-1, 0, 1], 'C': [-11, 0, 4]})])
def test_merge_common_cols(df1, df2):
    # test merge() based on common columns when key columns not provided
    def impl(df1, df2):
        return df1.merge(df2)

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(df1, df2), impl(df1, df2))


@pytest.mark.parametrize('df1', [
    pd.DataFrame({'A': [1, 11, 3], 'B': [4, 5, 1]}),
    pd.DataFrame({'A': [1, 11, 3], 'B': [4, 5, 1], 'C': [-1, 3, 4]})])
@pytest.mark.parametrize('df2', [
    pd.DataFrame({'A': [-1, 1, 3], 'B': [-1, 0, 1]}),
    pd.DataFrame({'A': [-1, 1, 3], 'B': [-1, 0, 1], 'C': [-11, 0, 4]})])
def test_merge_suffix(df1, df2):
    # test cases that have name overlaps, require adding suffix to column names
    def impl1(df1, df2):
        return df1.merge(df2, on='A')

    bodo_func = bodo.jit(impl1)
    pd.testing.assert_frame_equal(bodo_func(df1, df2), impl1(df1, df2))

    def impl2(df1, df2):
        return df1.merge(df2, on=['B', 'A'])

    bodo_func = bodo.jit(impl2)
    pd.testing.assert_frame_equal(bodo_func(df1, df2), impl2(df1, df2))


@pytest.mark.parametrize('df1', [
    pd.DataFrame({'A': [1, 11, 3], 'B': [4, 5, 1]}, index=[1, 4, 3]),
    pd.DataFrame({'A': [1, 11, 3], 'B': [4, 5, 1], 'C': [-1, 3, 4]},
    index=[1, 4, 3])])
@pytest.mark.parametrize('df2', [
    pd.DataFrame({'A': [-1, 1, 3], 'B': [-1, 0, 1]}, index=[-1, 1, 3]),
    pd.DataFrame({'A': [-1, 1, 3], 'B': [-1, 0, 1], 'C': [-11, 0, 4]},
    index=[-1, 1, 3])])
def test_merge_index(df1, df2):
    # test using index for join
    def impl1(df1, df2):
        return df1.merge(df2, left_index=True, right_index=True)

    bodo_func = bodo.jit(impl1)
    pd.testing.assert_frame_equal(bodo_func(df1, df2), impl1(df1, df2))

    # pandas duplicates key column if one side is using index
    # TODO: replicate pandas behavior
    # def impl2(df1, df2):
    #     return df1.merge(df2, left_on='A', right_index=True)

    # bodo_func = bodo.jit(impl2)
    # print(bodo_func(df1, df2), impl2(df1, df2))
    # pd.testing.assert_frame_equal(bodo_func(df1, df2), impl2(df1, df2))

    # def impl3(df1, df2):
    #     return df1.merge(df2, left_index=True, right_on='A')

    # bodo_func = bodo.jit(impl3)
    # pd.testing.assert_frame_equal(bodo_func(df1, df2), impl3(df1, df2))


@pytest.mark.parametrize('df1', [
    pd.DataFrame({'A': [1, 11, 3], 'B': [4, 5, 1]}, index=[1, 4, 3]),
    pd.DataFrame({'A': [1, 11, 3], 'B': [4, 5, 1], 'C': [-1, 3, 4]},
    index=[1, 4, 3])])
@pytest.mark.parametrize('df2', [
    pd.DataFrame({'D': [-1., 1., 3.]}, index=[-1, 1, 3]),
    pd.DataFrame({'D': [-1., 1., 3.], 'E': [-1., 0., 1.]},
    index=[-1, 1, 3])])
def test_join_call(df1, df2):
    def impl1(df1, df2):
        return df1.join(df2)

    bodo_func = bodo.jit(impl1)
    pd.testing.assert_frame_equal(bodo_func(df1, df2), impl1(df1, df2))

    def impl2(df1, df2):
        return df1.join(df2, on='A')

    bodo_func = bodo.jit(impl2)
    pd.testing.assert_frame_equal(bodo_func(df1, df2), impl2(df1, df2))


def test_merge_asof_parallel1(datapath):
    fname1 = datapath('asof1.pq')
    fname2 = datapath('asof2.pq')
    def impl():
        df1 = pd.read_parquet(fname1)
        df2 = pd.read_parquet(fname2)
        df3 = pd.merge_asof(df1, df2, on='time')
        return (df3.A.sum(), df3.time.max(), df3.B.sum())

    bodo_func = bodo.jit(impl)
    assert bodo_func() == impl()


class TestJoin(unittest.TestCase):
    def test_join1(self):
        def test_impl(n):
            df1 = pd.DataFrame({'key1': np.arange(n)+3, 'A': np.arange(n)+1.0})
            df2 = pd.DataFrame({'key2': 2*np.arange(n)+1, 'B': n+np.arange(n)+1.0})
            df3 = pd.merge(df1, df2, left_on='key1', right_on='key2')
            return df3.B.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        n = 11111
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_join1_seq(self):
        def test_impl(df1, df2):
            df3 = df1.merge(df2, left_on='key1', right_on='key2')
            return df3

        bodo_func = bodo.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'key1': np.arange(n)+3, 'A': np.arange(n)+1.0})
        df2 = pd.DataFrame({'key2': 2*np.arange(n)+1, 'B': n+np.arange(n)+1.0})
        pd.testing.assert_frame_equal(bodo_func(df1, df2), test_impl(df1, df2))
        n = 11111
        df1 = pd.DataFrame({'key1': np.arange(n)+3, 'A': np.arange(n)+1.0})
        df2 = pd.DataFrame({'key2': 2*np.arange(n)+1, 'B': n+np.arange(n)+1.0})
        pd.testing.assert_frame_equal(bodo_func(df1, df2), test_impl(df1, df2))

    def test_join1_seq_str(self):
        def test_impl():
            df1 = pd.DataFrame({'key1': ['foo', 'bar', 'baz']})
            df2 = pd.DataFrame({'key2': ['baz', 'bar', 'baz'], 'B': ['b', 'zzz', 'ss']})
            df3 = pd.merge(df1, df2, left_on='key1', right_on='key2')
            return df3.B

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(set(bodo_func()), set(test_impl()))

    def test_join1_seq_str_na(self):
        # test setting NA in string data column
        def test_impl():
            df1 = pd.DataFrame({'key1': ['foo', 'bar', 'baz']})
            df2 = pd.DataFrame({'key2': ['baz', 'bar', 'baz'], 'B': ['b', 'zzz', 'ss']})
            df3 = df1.merge(df2, left_on='key1', right_on='key2', how='left')
            return df3.B

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(set(bodo_func()), set(test_impl()))

    def test_join_mutil_seq1(self):
        def test_impl(df1, df2):
            return df1.merge(df2, on=['A', 'B'])

        bodo_func = bodo.jit(test_impl)
        df1 = pd.DataFrame({'A': [3,1,1,3,4],
                            'B': [1,2,3,2,3],
                            'C': [7,8,9,4,5]})

        df2 = pd.DataFrame({'A': [2,1,4,4,3],
                            'B': [1,3,2,3,2],
                            'D': [1,2,3,4,8]})

        pd.testing.assert_frame_equal(bodo_func(df1, df2), test_impl(df1, df2))

    def test_join_mutil_parallel1(self):
        def test_impl(A1, B1, C1, A2, B2, D2):
            df1 = pd.DataFrame({'A': A1, 'B': B1, 'C': C1})
            df2 = pd.DataFrame({'A': A2, 'B': B2, 'D': D2})
            df3 = df1.merge(df2, on=['A', 'B'])
            return df3.C.sum() + df3.D.sum()

        bodo_func = bodo.jit(locals={
            'A1:input': 'distributed',
            'B1:input': 'distributed',
            'C1:input': 'distributed',
            'A2:input': 'distributed',
            'B2:input': 'distributed',
            'D2:input': 'distributed',})(test_impl)
        df1 = pd.DataFrame({'A': [3,1,1,3,4],
                            'B': [1,2,3,2,3],
                            'C': [7,8,9,4,5]})

        df2 = pd.DataFrame({'A': [2,1,4,4,3],
                            'B': [1,3,2,3,2],
                            'D': [1,2,3,4,8]})

        start, end = get_start_end(len(df1))
        h_A1 = df1.A.values[start:end]
        h_B1 = df1.B.values[start:end]
        h_C1 = df1.C.values[start:end]
        h_A2 = df2.A.values[start:end]
        h_B2 = df2.B.values[start:end]
        h_D2 = df2.D.values[start:end]
        p_A1 = df1.A.values
        p_B1 = df1.B.values
        p_C1 = df1.C.values
        p_A2 = df2.A.values
        p_B2 = df2.B.values
        p_D2 = df2.D.values
        h_res = bodo_func(h_A1, h_B1, h_C1, h_A2, h_B2, h_D2)
        p_res = test_impl(p_A1, p_B1, p_C1, p_A2, p_B2, p_D2)
        self.assertEqual(h_res, p_res)

    def test_join_left_parallel1(self):
        """
        """
        def test_impl(A1, B1, C1, A2, B2, D2):
            df1 = pd.DataFrame({'A': A1, 'B': B1, 'C': C1})
            df2 = pd.DataFrame({'A': A2, 'B': B2, 'D': D2})
            # TODO: const tuple
            # df3 = df1.merge(df2, on=('A', 'B'))
            df3 = df1.merge(df2, on=['A', 'B'])
            return df3.C.sum() + df3.D.sum()

        bodo_func = bodo.jit(locals={
            'A1:input': 'distributed',
            'B1:input': 'distributed',
            'C1:input': 'distributed',})(test_impl)
        df1 = pd.DataFrame({'A': [3,1,1,3,4],
                            'B': [1,2,3,2,3],
                            'C': [7,8,9,4,5]})

        df2 = pd.DataFrame({'A': [2,1,4,4,3],
                            'B': [1,3,2,3,2],
                            'D': [1,2,3,4,8]})

        start, end = get_start_end(len(df1))
        h_A1 = df1.A.values[start:end]
        h_B1 = df1.B.values[start:end]
        h_C1 = df1.C.values[start:end]
        h_A2 = df2.A.values
        h_B2 = df2.B.values
        h_D2 = df2.D.values
        p_A1 = df1.A.values
        p_B1 = df1.B.values
        p_C1 = df1.C.values
        p_A2 = df2.A.values
        p_B2 = df2.B.values
        p_D2 = df2.D.values
        h_res = bodo_func(h_A1, h_B1, h_C1, h_A2, h_B2, h_D2)
        p_res = test_impl(p_A1, p_B1, p_C1, p_A2, p_B2, p_D2)
        self.assertEqual(h_res, p_res)
        self.assertEqual(count_array_OneDs(), 3)

    def test_join_datetime_seq1(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, on='time')

        bodo_func = bodo.jit(test_impl)
        df1 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-03', '2017-01-06', '2017-02-21']), 'B': [4, 5, 6]})
        df2 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-01', '2017-01-06', '2017-01-03']), 'A': [7, 8, 9]})
        pd.testing.assert_frame_equal(bodo_func(df1, df2), test_impl(df1, df2))

    def test_join_datetime_parallel1(self):
        def test_impl(df1, df2):
            df3 = pd.merge(df1, df2, on='time')
            return (df3.A.sum(), df3.time.max(), df3.B.sum())

        bodo_func = bodo.jit(distributed=['df1', 'df2'])(test_impl)
        df1 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-03', '2017-01-06', '2017-02-21']), 'B': [4, 5, 6]})
        df2 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-01', '2017-01-06', '2017-01-03']), 'A': [7, 8, 9]})
        start1, end1 = get_start_end(len(df1))
        start2, end2 = get_start_end(len(df2))
        self.assertEqual(
            bodo_func(df1.iloc[start1:end1], df2.iloc[start2:end2]),
            test_impl(df1, df2))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_merge_asof_seq1(self):
        def test_impl(df1, df2):
            return pd.merge_asof(df1, df2, on='time')

        bodo_func = bodo.jit(test_impl)
        df1 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-03', '2017-01-06', '2017-02-21']), 'B': [4, 5, 6]})
        df2 = pd.DataFrame(
            {'time': pd.DatetimeIndex(
                ['2017-01-01', '2017-01-02', '2017-01-04', '2017-02-23',
                '2017-02-25']), 'A': [2,3,7,8,9]})
        pd.testing.assert_frame_equal(bodo_func(df1, df2), test_impl(df1, df2))

    def test_join_left_seq1(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, how='left', on='key')

        bodo_func = bodo.jit(test_impl)
        df1 = pd.DataFrame(
            {'key': [2,3,5,1,2,8], 'A': np.array([4,6,3,9,9,-1], np.float)})
        df2 = pd.DataFrame(
            {'key': [1,2,9,3,2], 'B': np.array([1,7,2,6,5], np.float)})
        h_res = bodo_func(df1, df2)
        res = test_impl(df1, df2)
        np.testing.assert_array_equal(h_res.key.values, res.key.values)
        # converting arrays to sets since order of values can be different
        self.assertEqual(set(h_res.A.values), set(res.A.values))
        self.assertEqual(
            set(h_res.B.dropna().values), set(res.B.dropna().values))

    def test_join_left_seq2(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, how='left', on='key')

        bodo_func = bodo.jit(test_impl)
        # test left run where a key is repeated on left but not right side
        df1 = pd.DataFrame(
            {'key': [2,3,5,3,2,8], 'A': np.array([4,6,3,9,9,-1], np.float)})
        df2 = pd.DataFrame(
            {'key': [1,2,9,3,10], 'B': np.array([1,7,2,6,5], np.float)})
        h_res = bodo_func(df1, df2)
        res = test_impl(df1, df2)
        np.testing.assert_array_equal(h_res.key.values, res.key.values)
        # converting arrays to sets since order of values can be different
        self.assertEqual(set(h_res.A.values), set(res.A.values))
        self.assertEqual(
            set(h_res.B.dropna().values), set(res.B.dropna().values))

    def test_join_right_seq1(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, how='right', on='key')

        bodo_func = bodo.jit(test_impl)
        df1 = pd.DataFrame(
            {'key': [2,3,5,1,2,8], 'A': np.array([4,6,3,9,9,-1], np.float)})
        df2 = pd.DataFrame(
            {'key': [1,2,9,3,2], 'B': np.array([1,7,2,6,5], np.float)})
        h_res = bodo_func(df1, df2)
        res = test_impl(df1, df2)
        self.assertEqual(set(h_res.key.values), set(res.key.values))
        # converting arrays to sets since order of values can be different
        self.assertEqual(set(h_res.B.values), set(res.B.values))
        self.assertEqual(
            set(h_res.A.dropna().values), set(res.A.dropna().values))

    def test_join_outer_seq1(self):
        def test_impl(df1, df2):
            return pd.merge(df1, df2, how='outer', on='key')

        bodo_func = bodo.jit(test_impl)
        df1 = pd.DataFrame(
            {'key': [2,3,5,1,2,8], 'A': np.array([4,6,3,9,9,-1], np.float)})
        df2 = pd.DataFrame(
            {'key': [1,2,9,3,2], 'B': np.array([1,7,2,6,5], np.float)})
        h_res = bodo_func(df1, df2)
        res = test_impl(df1, df2)
        self.assertEqual(set(h_res.key.values), set(res.key.values))
        # converting arrays to sets since order of values can be different
        self.assertEqual(
            set(h_res.B.dropna().values), set(res.B.dropna().values))
        self.assertEqual(
            set(h_res.A.dropna().values), set(res.A.dropna().values))

    def test_join1_seq_key_change1(self):
        # make sure const list typing doesn't replace const key values
        def test_impl(df1, df2, df3, df4):
            o1 = df1.merge(df2, on=['A'])
            o2 = df3.merge(df4, on=['B'])
            return o1, o2

        bodo_func = bodo.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.arange(n)+3, 'AA': np.arange(n)+1.0})
        df2 = pd.DataFrame({'A': 2*np.arange(n)+1, 'AAA': n+np.arange(n)+1.0})
        df3 = pd.DataFrame({'B': 2*np.arange(n)+1, 'BB': n+np.arange(n)+1.0})
        df4 = pd.DataFrame({'B': 2*np.arange(n)+1, 'BBB': n+np.arange(n)+1.0})
        pd.testing.assert_frame_equal(bodo_func(df1, df2, df3, df4)[1], test_impl(df1, df2, df3, df4)[1])

    def test_join_cat1(self):
        fname = os.path.join('bodo', 'tests', 'data', 'csv_data_cat1.csv')
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1':np.int, 'C2': ct_dtype, 'C3':str}
            df1 = pd.read_csv(fname,
                names=['C1', 'C2', 'C3'],
                dtype=dtypes,
            )
            n = len(df1)
            df2 = pd.DataFrame({'C1': 2*np.arange(n)+1, 'AAA': n+np.arange(n)+1.0})
            df3 = df1.merge(df2, on='C1')
            return df3

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_join_cat2(self):
        # test setting NaN in categorical array
        fname = os.path.join('bodo', 'tests', 'data', 'csv_data_cat1.csv')
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1':np.int, 'C2': ct_dtype, 'C3':str}
            df1 = pd.read_csv(fname,
                names=['C1', 'C2', 'C3'],
                dtype=dtypes,
            )
            n = len(df1)
            df2 = pd.DataFrame({'C1': 2*np.arange(n)+1, 'AAA': n+np.arange(n)+1.0})
            df3 = df1.merge(df2, on='C1', how='right')
            return df3

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(
            bodo_func().sort_values('C1').reset_index(drop=True),
            test_impl().sort_values('C1').reset_index(drop=True))

    def test_join_cat_parallel1(self):
        # TODO: cat as keys
        fname = os.path.join('bodo', 'tests', 'data', 'csv_data_cat1.csv')
        def test_impl():
            ct_dtype = CategoricalDtype(['A', 'B', 'C'])
            dtypes = {'C1':np.int, 'C2': ct_dtype, 'C3':str}
            df1 = pd.read_csv(fname,
                names=['C1', 'C2', 'C3'],
                dtype=dtypes,
            )
            n = len(df1)
            df2 = pd.DataFrame({'C1': 2*np.arange(n)+1, 'AAA': n+np.arange(n)+1.0})
            df3 = df1.merge(df2, on='C1')
            return df3

        bodo_func = bodo.jit(distributed=['df3'])(test_impl)
        # TODO: check results
        self.assertTrue((bodo_func().columns == test_impl().columns).all())


if __name__ == "__main__":
    unittest.main()
