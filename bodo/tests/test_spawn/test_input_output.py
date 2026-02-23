"""Test that inputs and outputs are supported by spawn mode"""

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.series import BodoSeries
from bodo.tests.dataframe_common import df_value_params
from bodo.tests.series_common import series_val_params
from bodo.tests.utils import (
    _test_equal_guard,
    check_func,
    pytest_spawn_mode,
)

pytestmark = pytest_spawn_mode


@pytest.fixture(
    params=df_value_params
    + [
        pytest.param(
            pd.DataFrame({"A": list(range(100))}, pd.interval_range(0, 100)),
            id="interval_index",
            marks=[
                pytest.mark.slow,
                pytest.mark.skip("support IntervalIndex in non-jit gather/scatter"),
            ],
        ),
        pytest.param(
            pd.DataFrame(
                {"A": list(range(100))},
                pd.MultiIndex.from_tuples([(1, 2), (3, 4)] * 50, names=["AA", "BB"]),
            ),
            id="multi_index",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            pd.DataFrame(
                {"A": ["A"] * 100},
                index=pd.CategoricalIndex(["A", "B", "C", "D", "E"] * 20),
            ),
            id="categorical_index",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            pd.DataFrame(
                {"A": ["A"] * 100},
                index=pd.PeriodIndex.from_fields(
                    year=[2020, 2021] * 50,
                    month=[1, 6] * 50,
                    day=[1, 30] * 50,
                    freq="D",
                ),
            ),
            id="period_index",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            pd.DataFrame(
                {"A": ["A"] * 100},
                index=pd.DatetimeIndex(pd.date_range("2020-01-01", periods=100)),
            ),
            id="datetime_index",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            pd.DataFrame(
                {"A": ["A"] * 100},
                index=pd.TimedeltaIndex(
                    pd.timedelta_range("1 days", periods=100, unit="ns")
                ),
            ),
            id="timedelta_index",
            marks=[pytest.mark.slow],
        ),
    ]
)
def df_value(request):
    return request.param


@pytest.fixture(
    params=series_val_params
    + [
        pytest.param(
            pd.Series(list(range(100)), pd.interval_range(0, 100)),
            id="interval_index",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            pd.Series(
                list(range(100)),
                pd.MultiIndex.from_tuples([(1, 2), (3, 4)] * 50, names=["AA", "BB"]),
            ),
            id="multi_index",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            pd.Series(
                ["A"] * 100,
                index=pd.CategoricalIndex(["A", "B", "C", "D", "E"] * 20),
            ),
            id="categorical_index",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            pd.Series(
                [1, 2, 3, -1, 4] * 20,
                index=pd.PeriodIndex(
                    ["2018-01", "2018-02", "2018-03", "2018-04", "2018-05"] * 20,
                    freq="M",
                ),
            ),
            id="period_index",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            pd.Series(
                [1, 2, 3, -1, 4] * 20,
                index=pd.DatetimeIndex(pd.date_range("2018-01-01", periods=100)).astype(
                    "datetime64[ns]"
                ),
            ),
            id="datetime_index",
            marks=[pytest.mark.slow],
        ),
        pytest.param(
            pd.Series(
                [1, 2, 3, -1, 4] * 20,
                index=pd.TimedeltaIndex(
                    pd.timedelta_range("1 days", periods=100, unit="ns")
                ).astype("timedelta64[ns]"),
            ),
            id="timedelta_index",
            marks=[pytest.mark.slow],
        ),
    ]
)
def series_val(request):
    return request.param


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
    def test(df):
        return df

    # Convert non-string column names to strings
    if not all(isinstance(col, str) for col in df_value.columns):
        df_value = df_value.copy()
        df_value.columns = df_value.columns.astype(str)

    check_func(test, (df_value,), only_spawn=True, check_pandas_types=False)


def test_distributed_input_output_series(series_val):
    def test(A):
        return A

    check_func(test, (series_val,), only_spawn=True, check_pandas_types=False)


def test_spawn_distributed():
    @bodo.jit(distributed={"A"}, spawn=True)
    def test(A):
        s = A.sum()
        return s

    A = pd.Series(np.ones(1000, dtype=np.int64))
    assert test(A) == 1000


@pytest.fixture(
    params=[
        pd.RangeIndex(100, -100, -5, name="ABC"),
        pd.Index([3, 4, 1, 7, 0]),
        pd.MultiIndex.from_arrays(
            [
                np.arange(5),
                pd.date_range("2001-10-15", periods=5).astype("datetime64[ns]"),
            ],
            names=["AA", None],
        ),
        lambda: bodo.hiframes.table.Table((np.arange(6),)),
        pd.Categorical([1, 4, 5, 1, 4]),
        pd.arrays.IntervalArray(
            [
                pd.Interval(0, 1),
                pd.Interval(0, 3),
                pd.Interval(4, 6),
                pd.Interval(0, 3),
                pd.Interval(4, 6),
            ]
        ),
    ],
    ids=["range_index", "index", "multi_index", "table", "categorical", "interval"],
)
def other_dist_arg(request):
    """Lazily evaluate some arguments to avoid importing the compiler at test
    collection time"""
    import bodo.decorators  # noqa

    val = request.param

    return val() if callable(val) else val


def test_distributed_others(other_dist_arg):
    """Test less common distributable arguments and return values (Index, ...)"""

    def test(other_dist_arg):
        return other_dist_arg

    check_func(test, (other_dist_arg,), only_spawn=True, check_pandas_types=False)


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


@pytest.mark.skip(
    reason="TODO: support updating kwargs other than Numpy arrays in the compiler"
)
def test_tuple_bodo_dataframe_arg_updates():
    """
    Test that passing a BodoDataFrame in a tuple as an argument to a function updates the data returned when collected
    """

    @bodo.jit(spawn=True)
    def _get_bodo_df(df):
        return df

    @bodo.jit(spawn=True)
    def impl(arg, kwarg=None):
        arg[0]["A"] += 3
        kwarg[0]["A"] += 3
        return arg[1]

    df = pd.DataFrame({"A": [1, 2, 3] * 100})
    arg_bodo_df = _get_bodo_df(df)
    kwarg_bodo_df = _get_bodo_df(df)
    assert isinstance(arg_bodo_df, BodoDataFrame)
    assert isinstance(kwarg_bodo_df, BodoDataFrame)
    assert impl((arg_bodo_df, 2), kwarg=(kwarg_bodo_df,)) == 2
    _test_equal_guard(arg_bodo_df.head(), df.head() + 3, check_pandas_types=False)
    _test_equal_guard(kwarg_bodo_df.head(), df.head() + 3, check_pandas_types=False)

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


def test_tuple_bodo_series_arg_doesnt_collect():
    """
    Test that passing a BodoSeries as an argument to a function doesn't collect the data
    """

    @bodo.jit(spawn=True)
    def impl(x):
        return x

    series = pd.Series([1, 2, 3] * 100)
    bodo_series = impl(series)
    assert isinstance(bodo_series, BodoSeries)
    new_bodo_series, _ = impl((bodo_series, 1))
    # Test that using it as an arg doesn't collect the result
    assert bodo_series._lazy
    _test_equal_guard(bodo_series, series, check_pandas_types=False)
    _test_equal_guard(new_bodo_series, series, check_pandas_types=False)


@pytest.mark.skip(
    reason="TODO: support updating args other than Numpy arrays in the compiler"
)
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
