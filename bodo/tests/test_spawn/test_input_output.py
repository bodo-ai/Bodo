"""Test that inputs and outputs are supported by spawn mode"""

import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.range import RangeIndex

import bodo
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries
from bodo.tests.dataframe_common import df_value  # noqa
from bodo.tests.series_common import series_val  # noqa
from bodo.tests.utils import (
    _get_dist_arg,
    _test_equal_guard,
    check_func,
    pytest_spawn_mode,
)

pytestmark = pytest_spawn_mode


def test_distributed_input_scalar():
    def test(i):
        return i

    check_func(test, (42,), only_spawn=True)


def test_distributed_input_array():
    def test(A):
        s = A.sum()
        return s

    A = np.ones(1000, dtype=np.int64)
    check_func(test, (A,), only_spawn=True)


def test_distributed_scalar_output():
    def test():
        return 1

    check_func(test, (), only_spawn=True)


def test_distributed_input_output_df(df_value):
    if (
        not isinstance(df_value.index, RangeIndex)
        and type(df_value.index) is not pd.Index
        and type(df_value.index) is not pd.IntervalIndex
        and type(df_value.index) is not pd.MultiIndex
    ):
        pytest.skip("BSE-4099: Support all pandas index types in lazy wrappers")

    def test(df):
        return df

    check_func(test, (df_value,), only_spawn=True, check_pandas_types=False)


def test_distributed_input_output_series(series_val):
    if (
        not isinstance(series_val.index, RangeIndex)
        and type(series_val.index) is not pd.Index
        and type(series_val.index) is not pd.IntervalIndex
        and type(series_val.index) is not pd.MultiIndex
    ):
        pytest.skip("BSE-4099: Support all pandas index types in lazy wrappers")

    def test(A):
        return A

    check_func(test, (series_val,), only_spawn=True, check_pandas_types=False)


def test_spawn_distributed():
    @bodo.jit(distributed={"A"}, spawn=True)
    def test(A):
        s = A.sum()
        return s

    A = pd.Series(np.ones(1000, dtype=np.int64))
    assert test(_get_dist_arg(A)) == 1000


def test_distributed_small_output():
    """Test that inputs and distributed outputs are supported by spawn mode even when the output is smaller than a typical head"""

    def test(A):
        return A + 1

    A = pd.Series(np.random.randn(3))
    check_func(test, (A,), only_spawn=True, check_pandas_types=False)


def test_distributed_0_len_output():
    """Test that inputs and distributed outputs are supported by spawn mode even when the output is length 0"""

    def test(A):
        return A[0:0]

    check_func(
        test,
        (pd.DataFrame({"A": [1] * 10}),),
        py_output=pd.DataFrame({"A": []}, dtype=np.int64),
        only_spawn=True,
        check_pandas_types=False,
    )


def test_scalar_tuple_return():
    """Make sure returning tuple without distributed data works"""

    def test(df):
        return df.shape

    check_func(
        test,
        (pd.DataFrame({"A": [1, 5, 8]}),),
        only_spawn=True,
    )


def test_bodo_dataframe_arg_doesnt_collect():
    """
    Test that passing a BodoDataFrame as an argument to a function doesn't collect the data
    """

    @bodo.jit(spawn=True)
    def impl(df):
        return df

    df = pd.DataFrame({"A": [1, 2, 3] * 100})
    bodo_df = impl(df)
    assert isinstance(bodo_df, BodoDataFrame)
    new_bodo_df = impl(bodo_df)
    # Test that using it as an arg doesn't collect the result
    assert bodo_df._mgr._md_result_id is not None
    _test_equal_guard(bodo_df, df, check_pandas_types=False)
    _test_equal_guard(new_bodo_df, df, check_pandas_types=False)


def test_bodo_dataframe_arg_updates():
    """
    Test that passing a BodoDataFrame as an argument to a function updates the data returned when collected
    """

    @bodo.jit(spawn=True)
    def _get_bodo_df(df):
        return df

    @bodo.jit(spawn=True)
    def impl(df, kwarg=None):
        df["A"] += 3
        if kwarg is not None:
            kwarg["A"] += 3
        return 1

    df = pd.DataFrame({"A": [1, 2, 3] * 100})
    arg_bodo_df = _get_bodo_df(df)
    kwarg_bodo_df = _get_bodo_df(df)
    assert isinstance(arg_bodo_df, BodoDataFrame)
    assert isinstance(kwarg_bodo_df, BodoDataFrame)
    impl(arg_bodo_df, kwarg=kwarg_bodo_df)

    _test_equal_guard(arg_bodo_df.head(), df.head() + 3, check_pandas_types=False)
    _test_equal_guard(kwarg_bodo_df.head(), df.head() + 3, check_pandas_types=False)
    # Check that the BodoDataFrameas are still lazy
    assert arg_bodo_df._lazy
    assert kwarg_bodo_df._lazy

    _test_equal_guard(arg_bodo_df, df + 3, check_pandas_types=False)
    _test_equal_guard(kwarg_bodo_df, df + 3, check_pandas_types=False)


def test_bodo_dataframe_arg_collected():
    """
    Test that passing a BodoDataFrame that's been collected as an argument to a function works
    """

    @bodo.jit(spawn=True)
    def impl(df):
        return df

    df = pd.DataFrame({"A": [1, 2, 3] * 100})
    bodo_df = impl(df)
    assert isinstance(bodo_df, BodoDataFrame)
    # Force collection
    bodo_df["A"].sum()
    new_bodo_df = impl(bodo_df)
    assert isinstance(new_bodo_df, BodoDataFrame)
    _test_equal_guard(new_bodo_df, df, check_pandas_types=False)


def test_bodo_dataframe_arg_modified_on_spawner():
    """
    Test that passing a BodoDataFrame that's been modified works as expected
    """

    @bodo.jit(spawn=True)
    def impl(df):
        return df

    df = pd.DataFrame({"A": [1, 2, 3] * 100})
    bodo_df = impl(df)
    assert isinstance(bodo_df, BodoDataFrame)
    bodo_df.loc[0, "A"] = 0
    new_bodo_df = impl(bodo_df)
    assert isinstance(new_bodo_df, BodoDataFrame)
    _test_equal_guard(new_bodo_df, bodo_df, check_pandas_types=False)


def test_bodo_series_arg_doesnt_collect():
    """
    Test that passing a BodoSeries as an argument to a function doesn't collect the data
    """

    @bodo.jit(spawn=True)
    def impl(series):
        return series

    series = pd.Series([1, 2, 3] * 100)
    bodo_series = impl(series)
    assert isinstance(bodo_series, BodoSeries)
    new_bodo_series = impl(bodo_series)
    # Test that using it as an arg doesn't collect the result
    assert bodo_series._mgr._md_result_id is not None
    _test_equal_guard(bodo_series, series, check_pandas_types=False)
    _test_equal_guard(new_bodo_series, series, check_pandas_types=False)


def test_bodo_series_arg_updates():
    """
    Test that passing a BodoSeries as an argument to a function updates the data returned when collected
    """

    @bodo.jit(spawn=True)
    def _get_bodo_series(series):
        return series

    @bodo.jit(spawn=True)
    def impl(series, kwarg=None):
        series += 3
        if kwarg is not None:
            kwarg += 3
        return 1

    series = pd.Series([1, 2, 3] * 100)
    arg_bodo_series = _get_bodo_series(series)
    kwarg_bodo_series = _get_bodo_series(series)
    assert isinstance(arg_bodo_series, BodoSeries)
    assert isinstance(kwarg_bodo_series, BodoSeries)
    impl(arg_bodo_series, kwarg=kwarg_bodo_series)

    _test_equal_guard(
        arg_bodo_series.head(), series.head() + 3, check_pandas_types=False
    )
    _test_equal_guard(
        kwarg_bodo_series.head(), series.head() + 3, check_pandas_types=False
    )
    # Check that the BodoSeries are still lazy
    assert arg_bodo_series._lazy
    assert kwarg_bodo_series._lazy

    _test_equal_guard(arg_bodo_series, series + 3, check_pandas_types=False)
    _test_equal_guard(kwarg_bodo_series, series + 3, check_pandas_types=False)


def test_bodo_series_arg_collected():
    """
    Test that passing a BodoSeries that's been collected as an argument to a function works
    """

    @bodo.jit(spawn=True)
    def impl(series):
        return series

    series = pd.Series([1, 2, 3] * 100)
    bodo_series = impl(series)
    assert isinstance(bodo_series, BodoSeries)
    # Force collection
    bodo_series.sum()
    new_bodo_series = impl(bodo_series)
    assert isinstance(new_bodo_series, BodoSeries)
    _test_equal_guard(new_bodo_series, series, check_pandas_types=False)


def test_bodo_series_arg_modified_on_spawner():
    """
    Test that passing a BodoSeries that's been modified works as expected
    """

    @bodo.jit(spawn=True)
    def impl(series):
        return series

    series = pd.Series([1, 2, 3] * 100)
    bodo_series = impl(series)
    assert isinstance(bodo_series, BodoSeries)
    bodo_series[0] = 0
    new_bodo_series = impl(bodo_series)
    assert isinstance(new_bodo_series, BodoSeries)
    _test_equal_guard(new_bodo_series, bodo_series, check_pandas_types=False)
