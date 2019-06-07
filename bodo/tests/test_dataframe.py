import unittest
import random
import string
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.test_utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_start_end)



# TODO: other possible df types like Categorical, dt64, td64, ...
@pytest.fixture(params = [
    # int and float columns
    pd.DataFrame({'A': [1, 8, 4, 11], 'B': [1.1, np.nan, 4.2, 3.1],
        'C': [True, False, False, True]}),
    # uint8, float32 dtypes
    pd.DataFrame({'A': np.array([1, 8, 4, 0], dtype=np.uint8),
        'B': np.array([1.1, np.nan, 4.2, 3.1], dtype=np.float32)}),
    # string and int columns, float index
    pd.DataFrame({'A': ['AA', np.nan, '', 'D'], 'B': [1, 8, 4, -1]},
        [1.1, -2.1, 7.1, 0.1]),
    # range index
    pd.DataFrame({'A': [1, 8, 4, 1], 'B': ['A', 'B', 'CG', 'ACDE']},
        range(1, 9, 2)),
    # int index
    pd.DataFrame({'A': [1, 8, 4, 1], 'B': ['A', 'B', 'CG', 'ACDE']},
        [3, 7, 9, 3]),
    # string index
    pd.DataFrame({'A': [1, 2, 3, -1]}, ['A', 'BA', '', 'DD']),
    # datetime column
    pd.DataFrame({'A': pd.date_range(
        start='2018-04-24', end='2018-04-28', periods=4)}),
    # datetime index
    pd.DataFrame({'A': [3, 5, 1, -1]},
              pd.date_range(start='2018-04-24', end='2018-04-28', periods=4)),
    # TODO: timedelta
])
def df_value(request):
    return request.param


@pytest.fixture(params = [
    # int
    pd.DataFrame({'A': [1, 8, 4, 11]}),
    # int and float columns
    pd.DataFrame({'A': [1, 8, 4, 11], 'B': [1.1, np.nan, 4.2, 3.1]}),
    # uint8, float32 dtypes
    pd.DataFrame({'A': np.array([1, 8, 4, 0], dtype=np.uint8),
        'B': np.array([1.1, np.nan, 4.2, 3.1], dtype=np.float32)}),
    # int column, float index
    pd.DataFrame({'A': [1, 8, 4, -1]},
        [1.1, -2.1, 7.1, 0.1]),
    # range index
    pd.DataFrame({'A': [1, 8, 4, 1]}, range(1, 9, 2)),
    # datetime column
    pd.DataFrame({'A': pd.date_range(
        start='2018-04-24', end='2018-04-28', periods=4)}),
    # datetime index
    pd.DataFrame({'A': [3, 5, 1, -1]},
              pd.date_range(start='2018-04-24', end='2018-04-28', periods=4)),
    # TODO: timedelta
])
def numeric_df_value(request):
    return request.param


def test_unbox_df(df_value):
    # just unbox
    def impl(df_arg):
        return True

    bodo_func = bodo.jit(impl)
    assert bodo_func(df_value)

    # unbox and box
    def impl2(df_arg):
        return df_arg

    bodo_func = bodo.jit(impl2)
    pd.testing.assert_frame_equal(bodo_func(df_value), impl2(df_value))

    # unbox and return Series data with index
    # (previous test can box Index unintentionally)
    def impl3(df_arg):
        return df_arg.A

    bodo_func = bodo.jit(impl3)
    pd.testing.assert_series_equal(bodo_func(df_value), impl3(df_value))


def test_df_index(df_value):
    def impl(df):
        return df.index

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(df_value), impl(df_value))


def test_df_index_non():
    # test None index created inside the function
    def impl():
        df = pd.DataFrame({'A': [2, 3, 1]})
        return df.index

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(), impl())


def test_df_columns(df_value):
    def impl(df):
        return df.columns

    bodo_func = bodo.jit(impl)
    pd.testing.assert_index_equal(bodo_func(df_value), impl(df_value))


def test_df_values(numeric_df_value):
    def impl(df):
        return df.values

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(
        bodo_func(numeric_df_value), impl(numeric_df_value))


def test_df_get_values(numeric_df_value):
    def impl(df):
        return df.get_values()

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(
        bodo_func(numeric_df_value), impl(numeric_df_value))


def test_df_ndim(df_value):
    def impl(df):
        return df.ndim

    bodo_func = bodo.jit(impl)
    assert bodo_func(df_value) == impl(df_value)


def test_df_size(df_value):
    def impl(df):
        return df.size

    bodo_func = bodo.jit(impl)
    assert bodo_func(df_value) == impl(df_value)


def test_df_shape(df_value):
    def impl(df):
        return df.shape

    bodo_func = bodo.jit(impl)
    assert bodo_func(df_value) == impl(df_value)


# TODO: empty df: pd.DataFrame()
@pytest.mark.parametrize('df', [pd.DataFrame({'A': [1, 3]}),
    pd.DataFrame({'A': []})])
def test_df_empty(df):
    def impl(df):
        return df.empty

    bodo_func = bodo.jit(impl)
    assert bodo_func(df) == impl(df)


def test_df_astype_num(numeric_df_value):
    # not supported for dt64
    if any(d == np.dtype('datetime64[ns]') for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.astype(np.float32)

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(
        bodo_func(numeric_df_value), impl(numeric_df_value))


def test_df_astype_str(numeric_df_value):
    # not supported for dt64
    if any(d == np.dtype('datetime64[ns]') for d in numeric_df_value.dtypes):
        return

    # XXX str(float) not consistent with Python yet
    if any(d == np.dtype('float64') or d == np.dtype('float32')
                                             for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.astype(str)

    bodo_func = bodo.jit(impl)
    np.testing.assert_array_equal(
        bodo_func(numeric_df_value), impl(numeric_df_value))


def test_df_copy_deep(df_value):
    def impl(df):
        return df.copy()

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(df_value), impl(df_value))


def test_df_copy_shallow(df_value):
    def impl(df):
        return df.copy(deep=False)

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(df_value), impl(df_value))


def test_df_isna(df_value):
    # TODO: test dt64 NAT, categorical, etc.
    def impl(df):
        return df.isna()

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(df_value), impl(df_value))


def test_df_notna(df_value):
    # TODO: test dt64 NAT, categorical, etc.
    def impl(df):
        return df.notna()

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(df_value), impl(df_value))


def test_df_head(df_value):
    def impl(df):
        return df.head(3)

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(df_value), impl(df_value))


def test_df_tail(df_value):
    def impl(df):
        return df.tail(3)

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(df_value), impl(df_value))


@pytest.mark.parametrize('other', [
    pd.DataFrame({'A': np.arange(4), 'C': np.arange(4)**2}),
    [2, 3, 4]
])
def test_df_isin(other):
    # TODO: more tests, other data types
    # TODO: Series and dictionary values cases
    def impl(df, other):
        return df.isin(other)

    bodo_func = bodo.jit(impl)
    df = pd.DataFrame({'A': np.arange(4), 'B': np.arange(4)**2})
    pd.testing.assert_frame_equal(bodo_func(df, other), impl(df, other))


def test_df_abs(numeric_df_value):
    # not supported for dt64
    if any(d == np.dtype('datetime64[ns]') for d in numeric_df_value.dtypes):
        return

    def impl(df):
        return df.abs()

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(
        bodo_func(numeric_df_value), impl(numeric_df_value))


def test_df_corr(df_value):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.corr()

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(df_value), impl(df_value))


def test_df_corr_parallel():
    def impl(n):
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        return df.corr()

    bodo_func = bodo.jit(impl)
    n = 11
    pd.testing.assert_frame_equal(bodo_func(n), impl(n))
    assert count_array_OneDs() >= 3
    assert count_parfor_OneDs() >= 1


def test_df_cov(df_value):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.cov()

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(df_value), impl(df_value))


def test_df_count(df_value):
    def impl(df):
        return df.count()

    bodo_func = bodo.jit(impl)
    pd.testing.assert_series_equal(
        bodo_func(df_value), impl(df_value))


def test_df_prod(df_value):
    # empty dataframe output not supported yet
    if len(df_value._get_numeric_data().columns) == 0:
        return

    def impl(df):
        return df.prod()

    bodo_func = bodo.jit(impl)
    pd.testing.assert_series_equal(
        bodo_func(df_value), impl(df_value))


############################# old tests ###############################



@bodo.jit
def inner_get_column(df):
    # df2 = df[['A', 'C']]
    # df2['D'] = np.ones(3)
    return df.A


COL_IND = 0


class TestDataFrame(unittest.TestCase):
    def test_create1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_create_kws1(self):
        def test_impl(n):
            df = pd.DataFrame(data={'A': np.ones(n), 'B': np.random.ranf(n)})
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_create_dtype1(self):
        def test_impl(n):
            df = pd.DataFrame(data={'A': np.ones(n), 'B': np.random.ranf(n)},
                              dtype=np.int8)
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_create_column1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)},
                              columns=['B'])
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_create_column2(self):
        # column arg uses list('AB')
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)},
                              columns=list('AB'))
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_create_range_index1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.zeros(n), 'B': np.ones(n)},
                              index=range(0, n), columns=['A', 'B'])
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_create_ndarray1(self):
        def test_impl(n):
            # TODO: fix in Numba
            # data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            data = np.arange(9).reshape(3, 3)
            df = pd.DataFrame(data, columns=['a', 'b', 'c'])
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_create_ndarray_copy1(self):
        def test_impl(data):
            # TODO: fix in Numba
            # data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            df = pd.DataFrame(data, columns=['a', 'b', 'c'], copy=True)
            data[0] = 6
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        data = np.arange(9).reshape(3, 3)
        pd.testing.assert_frame_equal(
            bodo_func(data.copy()), test_impl(data.copy()))

    def test_create_empty_column1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)},
                              columns=['B', 'C'])
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        # bodo uses np.nan for empty columns currently but Pandas uses objects
        df1 = bodo_func(n)
        df2 = test_impl(n)
        df2['C'] = df2.C.astype(np.float64)
        pd.testing.assert_frame_equal(df1, df2)

    def test_create_cond1(self):
        def test_impl(A, B, c):
            if c:
                df = pd.DataFrame({'A': A})
            else:
                df = pd.DataFrame({'A': B})
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        A = np.ones(n)
        B = np.arange(n) + 1.0
        c = 0
        pd.testing.assert_series_equal(bodo_func(A, B, c), test_impl(A, B, c))
        c = 2
        pd.testing.assert_series_equal(bodo_func(A, B, c), test_impl(A, B, c))

    def test_unbox1(self):
        def test_impl(df):
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.random.ranf(n)})
        pd.testing.assert_series_equal(bodo_func(df), test_impl(df))

    @unittest.skip("needs properly refcounted dataframes")
    def test_unbox2(self):
        def test_impl(df, cond):
            n = len(df)
            if cond:
                df['A'] = np.arange(n) + 2.0
            return df.A

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        pd.testing.assert_series_equal(bodo_func(df.copy(), True), test_impl(df.copy(), True))
        pd.testing.assert_series_equal(bodo_func(df.copy(), False), test_impl(df.copy(), False))


    def test_box1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_box2(self):
        def test_impl():
            df = pd.DataFrame({'A': [1,2,3], 'B': ['a', 'bb', 'ccc']})
            return df

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    @unittest.skip("pending df filter support")
    def test_box3(self):
        def test_impl(df):
            df = df[df.A != 'dd']
            return df

        bodo_func = bodo.jit(test_impl)
        df = pd.DataFrame({'A': ['aa', 'bb', 'cc']})
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_box_dist_return(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df

        bodo_func = bodo.jit(distributed={'df'})(test_impl)
        n = 11
        hres, res = bodo_func(n), test_impl(n)
        self.assertEqual(count_array_OneDs(), 3)
        self.assertEqual(count_parfor_OneDs(), 2)
        dist_sum = bodo.jit(
            lambda a: bodo.libs.distributed_api.dist_reduce(
                a, np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)))
        dist_sum(1)  # run to compile
        np.testing.assert_allclose(dist_sum(hres.A.sum()), res.A.sum())
        np.testing.assert_allclose(dist_sum(hres.B.sum()), res.B.sum())

    def test_len1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.random.ranf(n)})
            return len(df)

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_shape1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.random.ranf(n)})
            return df.shape

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_column_getitem1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
            Ac = df['A'].values
            return Ac.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)
        self.assertEqual(count_parfor_OneDs(), 1)

    def test_column_list_getitem1(self):
        def test_impl(df):
            return df[['A', 'C']]

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame(
            {'A': np.arange(n), 'B': np.ones(n), 'C': np.random.ranf(n)})
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_filter1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+n, 'B': np.arange(n)**2})
            df1 = df[df.A > .5]
            return df1.B.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_filter2(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+n, 'B': np.arange(n)**2})
            df1 = df.loc[df.A > .5]
            return np.sum(df1.B)

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_filter3(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+n, 'B': np.arange(n)**2})
            df1 = df.iloc[(df.A > .5).values]
            return np.sum(df1.B)

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_iloc1(self):
        def test_impl(df, n):
            return df.iloc[1:n].B.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(bodo_func(df, n), test_impl(df, n))

    def test_iloc2(self):
        def test_impl(df, n):
            return df.iloc[np.array([1,4,9])].B.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(bodo_func(df, n), test_impl(df, n))

    def test_iloc3(self):
        def test_impl(df):
            return df.iloc[:,1].values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    @unittest.skip("TODO: support A[[1,2,3]] in Numba")
    def test_iloc4(self):
        def test_impl(df, n):
            return df.iloc[[1,4,9]].B.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(bodo_func(df, n), test_impl(df, n))

    def test_iloc5(self):
        # test iloc with global value
        def test_impl(df):
            return df.iloc[:,COL_IND].values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_loc1(self):
        def test_impl(df):
            return df.loc[:,'B'].values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_iat1(self):
        def test_impl(n):
            df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n)+n})
            return df.iat[3, 1]
        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_iat2(self):
        def test_impl(df):
            return df.iat[3, 1]
        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n)+n})
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_iat3(self):
        def test_impl(df, n):
            return df.iat[n-1, 1]
        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n)+n})
        self.assertEqual(bodo_func(df, n), test_impl(df, n))

    def test_iat_set1(self):
        def test_impl(df, n):
            df.iat[n-1, 1] = n**2
            return df.A  # return the column to check column aliasing
        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n)+n})
        df2 = df.copy()
        pd.testing.assert_series_equal(bodo_func(df, n), test_impl(df2, n))

    def test_iat_set2(self):
        def test_impl(df, n):
            df.iat[n-1, 1] = n**2
            return df  # check df aliasing/boxing
        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'B': np.ones(n), 'A': np.arange(n)+n})
        df2 = df.copy()
        pd.testing.assert_frame_equal(bodo_func(df, n), test_impl(df2, n))

    def test_set_column1(self):
        # set existing column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)+3.0})
            df['A'] = np.arange(n)
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_set_column_reflect4(self):
        # set existing column
        def test_impl(df, n):
            df['A'] = np.arange(n)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)+3.0})
        df2 = df1.copy()
        bodo_func(df1, n)
        test_impl(df2, n)
        pd.testing.assert_frame_equal(df1, df2)

    def test_set_column_new_type1(self):
        # set existing column with a new type
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)+3.0})
            df['A'] = np.arange(n)
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_set_column2(self):
        # create new column
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)+1.0})
            df['C'] = np.arange(n)
            return df

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_set_column_reflect3(self):
        # create new column
        def test_impl(df, n):
            df['C'] = np.arange(n)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df1 = pd.DataFrame({'A': np.ones(n, np.int64), 'B': np.arange(n)+3.0})
        df2 = df1.copy()
        bodo_func(df1, n)
        test_impl(df2, n)
        pd.testing.assert_frame_equal(df1, df2)

    def test_set_column_bool1(self):
        def test_impl(df):
            df['C'] = df['A'][df['B']]

        bodo_func = bodo.jit(test_impl)
        df = pd.DataFrame({'A': [1,2,3], 'B': [True, False, True]})
        df2 = df.copy()
        test_impl(df2)
        bodo_func(df)
        pd.testing.assert_series_equal(df.C, df2.C)

    def test_set_column_reflect1(self):
        def test_impl(df, arr):
            df['C'] = arr
            return df.C.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        bodo_func(df, arr)
        self.assertIn('C', df)
        np.testing.assert_almost_equal(df.C.values, arr)

    def test_set_column_reflect2(self):
        def test_impl(df, arr):
            df['C'] = arr
            return df.C.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        arr = np.random.ranf(n)
        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})
        df2 = df.copy()
        np.testing.assert_almost_equal(bodo_func(df, arr), test_impl(df2, arr))

    def test_df_values1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        np.testing.assert_array_equal(bodo_func(n), test_impl(n))

    def test_df_values2(self):
        def test_impl(df):
            return df.values

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
        np.testing.assert_array_equal(bodo_func(df), test_impl(df))

    def test_df_values_parallel1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.values.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        np.testing.assert_array_equal(bodo_func(n), test_impl(n))
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_df_apply(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            B = df.apply(lambda r: r.A + r.B, axis=1)
            return df.B.sum()

        n = 121
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(n), test_impl(n))

    def test_df_apply_branch(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)})
            B = df.apply(lambda r: r.A < 10 and r.B > 20, axis=1)
            return df.B.sum()

        n = 121
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(n), test_impl(n))

    def test_df_describe(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(0, n, 1, np.float32),
                               'B': np.arange(n)})
            #df.A[0:1] = np.nan
            return df.describe()

        bodo_func = bodo.jit(test_impl)
        n = 1001
        bodo_func(n)
        # XXX: test actual output
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_sort_values(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.B.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({'A': np.random.ranf(n), 'B': np.arange(n), 'C': np.random.ranf(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(df.copy()), test_impl(df))

    def test_sort_values_copy(self):
        def test_impl(df):
            df2 = df.sort_values('A')
            return df2.B.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({'A': np.random.ranf(n), 'B': np.arange(n), 'C': np.random.ranf(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(df.copy()), test_impl(df))

    def test_sort_values_single_col(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.A.values

        n = 1211
        np.random.seed(2)
        df = pd.DataFrame({'A': np.random.ranf(n)})
        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(df.copy()), test_impl(df))

    def test_sort_values_single_col_str(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.A.values

        n = 1211
        random.seed(2)
        str_vals = []

        for _ in range(n):
            k = random.randint(1, 30)
            val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals.append(val)
        df = pd.DataFrame({'A': str_vals})
        bodo_func = bodo.jit(test_impl)
        self.assertTrue((bodo_func(df.copy()) == test_impl(df)).all())

    def test_sort_values_str(self):
        def test_impl(df):
            df.sort_values('A', inplace=True)
            return df.B.values

        n = 1211
        random.seed(2)
        str_vals = []
        str_vals2 = []

        for i in range(n):
            k = random.randint(1, 30)
            val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals.append(val)
            val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))
            str_vals2.append(val)

        df = pd.DataFrame({'A': str_vals, 'B': str_vals2})
        # use mergesort for stability, in str generation equal keys are more probable
        sorted_df = df.sort_values('A', inplace=False, kind='mergesort')
        bodo_func = bodo.jit(test_impl)
        self.assertTrue((bodo_func(df) == sorted_df.B.values).all())

    def test_sort_parallel_single_col(self):
        # TODO: better parallel sort test
        def test_impl():
            df = pd.read_parquet('bodo/tests/data/kde.parquet')
            df.sort_values('points', inplace=True)
            res = df.points.values
            return res

        bodo_func = bodo.jit(locals={'res:return': 'distributed'})(test_impl)

        save_min_samples = bodo.ir.sort.MIN_SAMPLES
        try:
            bodo.ir.sort.MIN_SAMPLES = 10
            res = bodo_func()
            self.assertTrue((np.diff(res)>=0).all())
        finally:
            bodo.ir.sort.MIN_SAMPLES = save_min_samples  # restore global val

    def test_sort_parallel(self):
        # TODO: better parallel sort test
        def test_impl():
            df = pd.read_parquet('bodo/tests/data/kde.parquet')
            df['A'] = df.points.astype(np.float64)
            df.sort_values('points', inplace=True)
            res = df.A.values
            return res

        bodo_func = bodo.jit(locals={'res:return': 'distributed'})(test_impl)

        save_min_samples = bodo.ir.sort.MIN_SAMPLES
        try:
            bodo.ir.sort.MIN_SAMPLES = 10
            res = bodo_func()
            self.assertTrue((np.diff(res)>=0).all())
        finally:
            bodo.ir.sort.MIN_SAMPLES = save_min_samples  # restore global val

    def test_itertuples(self):
        def test_impl(df):
            res = 0.0
            for r in df.itertuples():
                res += r[1]
            return res

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.ones(n, np.int64)})
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_itertuples_str(self):
        def test_impl(df):
            res = ""
            for r in df.itertuples():
                res += r[1]
            return res

        bodo_func = bodo.jit(test_impl)
        n = 3
        df = pd.DataFrame({'A': ['aa', 'bb', 'cc'], 'B': np.ones(n, np.int64)})
        self.assertEqual(bodo_func(df), test_impl(df))

    def test_itertuples_order(self):
        def test_impl(n):
            res = 0.0
            df = pd.DataFrame({'B': np.arange(n), 'A': np.ones(n, np.int64)})
            for r in df.itertuples():
                res += r[1]
            return res

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_itertuples_analysis(self):
        """tests array analysis handling of generated tuples, shapes going
        through blocks and getting used in an array dimension
        """
        def test_impl(n):
            res = 0
            df = pd.DataFrame({'B': np.arange(n), 'A': np.ones(n, np.int64)})
            for r in df.itertuples():
                if r[1] == 2:
                    A = np.ones(r[1])
                    res += len(A)
            return res

        bodo_func = bodo.jit(test_impl)
        n = 11
        self.assertEqual(bodo_func(n), test_impl(n))

    def test_df_head1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.ones(n), 'B': np.arange(n)})
            return df.head(3)

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_pct_change1(self):
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.arange(n)+1},
                              np.arange(n)+1.3)
            return df.pct_change(3)

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_frame_equal(bodo_func(n), test_impl(n))

    def test_mean1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.arange(n)+1})
            return df.mean()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_std1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.arange(n)+1})
            return df.std()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_var1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.arange(n)+1})
            return df.var()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_max1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.arange(n)+1})
            return df.max()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_min1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.arange(n)+1})
            return df.min()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_sum1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.arange(n)+1})
            return df.sum()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_prod1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.arange(n)+1})
            return df.prod()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_count1(self):
        # TODO: non-numeric columns should be ignored automatically
        def test_impl(n):
            df = pd.DataFrame({'A': np.arange(n)+1.0, 'B': np.arange(n)+1})
            return df.count()

        bodo_func = bodo.jit(test_impl)
        n = 11
        pd.testing.assert_series_equal(bodo_func(n), test_impl(n))

    def test_df_fillna1(self):
        def test_impl(df):
            return df.fillna(5.0)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_fillna_str1(self):
        def test_impl(df):
            return df.fillna("dd")

        df = pd.DataFrame({'A': ['aa', 'b', None, 'ccc']})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_fillna_inplace1(self):
        def test_impl(A):
            A.fillna(11.0, inplace=True)
            return A

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        df2 = df.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df2))

    def test_df_reset_index1(self):
        def test_impl(df):
            return df.reset_index(drop=True)

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_reset_index_inplace1(self):
        def test_impl():
            df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
            df.reset_index(drop=True, inplace=True)
            return df

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_df_dropna1(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        bodo_func = bodo.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = bodo_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_dropna2(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0]})
        bodo_func = bodo.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = bodo_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_dropna_inplace1(self):
        # TODO: fix error when no df is returned
        def test_impl(df):
            df.dropna(inplace=True)
            return df

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        df2 = df.copy()
        bodo_func = bodo.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = bodo_func(df2)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_dropna_str1(self):
        def test_impl(df):
            return df.dropna()

        df = pd.DataFrame({'A': [1.0, 2.0, 4.0, 1.0], 'B': ['aa', 'b', None, 'ccc']})
        bodo_func = bodo.jit(test_impl)
        out = test_impl(df).reset_index(drop=True)
        h_out = bodo_func(df)
        pd.testing.assert_frame_equal(out, h_out)

    def test_df_drop1(self):
        def test_impl(df):
            return df.drop(columns=['A'])

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_drop_inplace2(self):
        # test droping after setting the column
        def test_impl(df):
            df2 = df[['A', 'B']]
            df2['D'] = np.ones(3)
            df2.drop(columns=['D'], inplace=True)
            return df2

        df = pd.DataFrame({'A': [1,2,3], 'B': [2,3,4]})
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_df_drop_inplace1(self):
        def test_impl(df):
            df.drop('A', axis=1, inplace=True)
            return df

        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 1.0], 'B': [4, 5, 6, 7]})
        df2 = df.copy()
        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df2))

    def test_isin_df1(self):
        def test_impl(df, df2):
            return df.isin(df2)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2 = pd.DataFrame({'A': np.arange(n), 'C': np.arange(n)**2})
        df2.A[n//2:] = n
        pd.testing.assert_frame_equal(bodo_func(df, df2), test_impl(df, df2))

    @unittest.skip("needs dict typing in Numba")
    def test_isin_dict1(self):
        def test_impl(df):
            vals = {'A': [2,3,4], 'C': [4,5,6]}
            return df.isin(vals)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_isin_list1(self):
        def test_impl(df):
            vals = [2,3,4]
            return df.isin(vals)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(bodo_func(df), test_impl(df))

    def test_append1(self):
        def test_impl(df, df2):
            return df.append(df2, ignore_index=True)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2 = pd.DataFrame({'A': np.arange(n), 'C': np.arange(n)**2})
        df2.A[n//2:] = n
        pd.testing.assert_frame_equal(bodo_func(df, df2), test_impl(df, df2))

    def test_append2(self):
        def test_impl(df, df2, df3):
            return df.append([df2, df3], ignore_index=True)

        bodo_func = bodo.jit(test_impl)
        n = 11
        df = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2 = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        df2.A[n//2:] = n
        df3 = pd.DataFrame({'A': np.arange(n), 'B': np.arange(n)**2})
        pd.testing.assert_frame_equal(
            bodo_func(df, df2, df3), test_impl(df, df2, df3))

    def test_concat_columns1(self):
        def test_impl(S1, S2):
            return pd.concat([S1, S2], axis=1)

        bodo_func = bodo.jit(test_impl)
        S1 = pd.Series([4, 5])
        S2 = pd.Series([6., 7.])
        # TODO: support int as column name
        pd.testing.assert_frame_equal(
            bodo_func(S1, S2),
            test_impl(S1, S2).rename(columns={0:'0', 1:'1'}))

    def test_var_rename(self):
        # tests df variable replacement in untyped_pass where inlining
        # can cause extra assignments and definition handling errors
        # TODO: inline freevar
        def test_impl():
            df = pd.DataFrame({'A': [1,2,3], 'B': [2,3,4]})
            # TODO: df['C'] = [5,6,7]
            df['C'] = np.ones(3)
            return inner_get_column(df)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_series_equal(bodo_func(), test_impl(), check_names=False)


if __name__ == "__main__":
    unittest.main()
