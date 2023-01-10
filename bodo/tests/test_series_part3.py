import re
from string import ascii_lowercase

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.series_common import numeric_series_val  # noqa
from bodo.tests.test_parfor_optimizations import _check_num_parfors
from bodo.tests.utils import ParforTestPipeline, check_func
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_series_empty_dtype(memory_leak_check):
    """
    Checks creating an empty series with only a
    DataType works as expected. This is used in
    BodoSQL to create optimized out empty DataFrames.
    This test suite checks all types BodoSQL might create.
    """

    def test_impl1():
        return pd.Series(dtype=str)

    def test_impl2():
        return pd.Series(dtype="boolean")

    def test_impl3():
        return pd.Series(dtype=pd.Int8Dtype())

    def test_impl4():
        return pd.Series(dtype=pd.Int16Dtype())

    def test_impl5():
        return pd.Series(dtype=pd.Int32Dtype())

    def test_impl6():
        return pd.Series(dtype=pd.Int64Dtype())

    def test_impl7():
        return pd.Series(dtype=np.float32)

    def test_impl8():
        return pd.Series(dtype=np.float64)

    def test_impl9():
        return pd.Series(dtype=np.dtype("datetime64[ns]"))

    def test_impl10():
        return pd.Series(dtype=np.dtype("timedelta64[ns]"))

    # Not needed by BodoSQL but seems relevant for coverage
    def test_impl11():
        return pd.Series(dtype=np.uint32)

    check_func(test_impl1, (), reset_index=True)
    check_func(test_impl2, (), reset_index=True)
    check_func(test_impl3, (), reset_index=True)
    check_func(test_impl4, (), reset_index=True)
    check_func(test_impl5, (), reset_index=True)
    check_func(test_impl6, (), reset_index=True)
    check_func(test_impl7, (), reset_index=True)
    check_func(test_impl8, (), reset_index=True)
    check_func(test_impl9, (), reset_index=True)
    check_func(test_impl10, (), reset_index=True)
    check_func(test_impl11, (), reset_index=True)


@pytest.mark.parametrize(
    "type_name",
    [
        "Int8",
        "UInt8",
        "Int16",
        "UInt16",
        "Int32",
        "UInt32",
        "Int64",
        "UInt64",
    ],
)
def test_str_nullable_astype(type_name, memory_leak_check):
    """
    Checks that casting from a String Series to a
    Nullable Integer works as expected.
    """
    # Generate the test code becuase the typename
    # must be a string constant
    def test_impl(S):
        return S.astype(type_name)

    # Avoid negative numbers to prevent undefined behavior
    # for some types.
    S = pd.Series(["0", "1", None, "123", "43", "32", None, "97"])
    # Panda's doesn't support this conversion, so generate
    # an expected output.
    py_output = pd.Series([0, 1, None, 123, 43, 32, None, 97], dtype=type_name)
    check_func(test_impl, (S,), py_output=py_output)


@pytest.mark.slow
def test_empty_dataframe(memory_leak_check):
    """
    Checks creating an empty DataFrame using
    Series values with only a  DataType works
    as expected. This is used in BodoSQL to
    create optimized out empty DataFrames.
    """

    def test_impl():
        return pd.DataFrame(
            {
                "A": pd.Series(dtype=str),
                "B": pd.Series(dtype=pd.Int32Dtype()),
                "C": pd.Series(dtype="boolean"),
                "D": pd.Series(dtype=np.float64),
                "F": pd.Series(dtype=np.dtype("datetime64[ns]")),
            }
        )

    check_func(test_impl, (), reset_index=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    "val",
    [
        pd.Timestamp("2021-05-11"),
        pd.Timedelta(days=13, seconds=-20),
        np.int8(1),
        np.uint8(1),
        np.int16(1),
        np.uint16(1),
        np.int32(1),
        np.uint32(1),
        np.int64(1),
        np.uint64(1),
        np.float32(1.5),
        np.float64(1.0),
        "Bears",
        True,
    ],
)
def test_scalar_series(val, memory_leak_check):
    def test_impl():
        return pd.Series(val, pd.RangeIndex(0, 100, 1))

    check_func(test_impl, (), reset_index=True)


def test_or_null(memory_leak_check):
    """
    Checks or null behavior inside Series
    """

    def test_impl(S1, S2):
        return S1 | S2

    S1 = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
    S2 = pd.Series([True, False, None] * 3, dtype="boolean")

    check_func(test_impl, (S1, S2))


@pytest.mark.slow
def test_or_numpy(memory_leak_check):
    """
    Checks or null behavior inside Series with a boolean array
    """

    def test_impl(val1, val2):
        return val1 | val2

    S1 = pd.Series([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")
    S2 = pd.Series(np.array([True, False] * 3))

    check_func(test_impl, (S1, S2))
    check_func(test_impl, (S2, S1))


@pytest.mark.slow
def test_or_null_arr(memory_leak_check):
    """
    Checks or null behavior inside Series with a boolean array
    """

    def test_impl(val1, val2):
        return val1 | val2

    S = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
    arr = pd.array([True, False, None] * 3, dtype="boolean")

    check_func(test_impl, (S, arr))
    check_func(test_impl, (arr, S))


@pytest.mark.slow
def test_or_numpy_array(memory_leak_check):
    """
    Checks or null behavior inside Series with a boolean array
    """

    def test_impl(val1, val2):
        return val1 | val2

    S = pd.Series([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")
    arr = np.array([True, False] * 3)

    check_func(test_impl, (S, arr))
    check_func(test_impl, (arr, S))


@pytest.mark.slow
def test_or_scalar(memory_leak_check):
    """
    Checks or null behavior inside Series with a boolean array
    """

    def test_impl(val1, val2):
        return val1 | val2

    S = pd.Series([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")

    check_func(test_impl, (S, True))
    check_func(test_impl, (S, False))
    check_func(test_impl, (True, S))
    check_func(test_impl, (False, S))


def test_and_null(memory_leak_check):
    """
    Checks and null behavior inside two Series with
    nullable boolean arrays.
    """

    def test_impl(S1, S2):
        return S1 & S2

    S1 = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
    S2 = pd.Series([True, False, None] * 3, dtype="boolean")

    check_func(test_impl, (S1, S2))


def test_neg(memory_leak_check):
    """
    make sure we can negate a series
    """

    def test_impl(S):
        return -S

    S = pd.Series(np.arange(10))
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_and_numpy(memory_leak_check):
    """
    Checks and null behavior inside a Series with a nullable
    boolean array and a numpy boolean array.
    """

    def test_impl(val1, val2):
        return val1 & val2

    S1 = pd.Series([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")
    S2 = pd.Series(np.array([True, False] * 3))

    check_func(test_impl, (S1, S2))
    check_func(test_impl, (S2, S1))


@pytest.mark.slow
def test_and_null_arr(memory_leak_check):
    """
    Checks and null behavior inside a Series with a boolean array
    and a nullable boolean array.
    """

    def test_impl(val1, val2):
        return val1 & val2

    S = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
    arr = pd.array([True, False, None] * 3, dtype="boolean")

    check_func(test_impl, (S, arr))
    check_func(test_impl, (arr, S))


@pytest.mark.slow
def test_and_numpy_array(memory_leak_check):
    """
    Checks and null behavior inside Series with a nullable boolean array
    and a numpy array.
    """

    def test_impl(val1, val2):
        return val1 & val2

    S = pd.Series([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")
    arr = np.array([True, False] * 3)

    check_func(test_impl, (S, arr))
    check_func(test_impl, (arr, S))


@pytest.mark.slow
def test_and_scalar(memory_leak_check):
    """
    Checks and null behavior inside Series with a
    nullable boolean array and a scalar
    """

    def test_impl(val1, val2):
        return val1 & val2

    S = pd.Series([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")

    check_func(test_impl, (S, True))
    check_func(test_impl, (S, False))
    check_func(test_impl, (True, S))
    check_func(test_impl, (False, S))


def test_timestamp_ge(memory_leak_check):
    """
    Tests for comparing a Timestamp value with a dt64 series.
    This ensure that operators that don't commute handle Timestamp
    on the lhs and rhs.
    """

    def test_impl(val1, val2):
        return val1 >= val2

    S = pd.Series(
        [
            pd.Timestamp(2021, 11, 21),
            pd.Timestamp(2022, 1, 12),
            pd.Timestamp(2021, 3, 3),
        ]
        * 4
    )
    ts = pd.Timestamp(2021, 8, 16)
    check_func(test_impl, (S, ts))
    check_func(test_impl, (ts, S))


def test_series_apply_method_str(memory_leak_check):
    """
    Test running series.apply with a string literal that
    matches a Series method.
    """

    def impl1(S):
        # Test a Series method
        return S.apply("nunique")

    def impl2(S):
        # Test a Series method that conflicts with a numpy function
        return S.apply("sum")

    def impl3(S):
        # Test a Series method that returns a Series
        return S.apply("abs")

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))
    # Used for abs test
    S[0] = -150

    check_func(impl1, (S,))
    check_func(impl2, (S,))
    check_func(impl3, (S,))


def test_series_apply_numpy_str(memory_leak_check):
    """
    Test running series.apply with a string literal that
    matches a Numpy function.
    """

    def impl1(S):
        return S.apply("sin")

    def impl2(S):
        return S.apply("log")

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))

    check_func(impl1, (S,))
    check_func(impl2, (S,))


@pytest.mark.slow
def test_series_apply_no_func(memory_leak_check):
    """
    Test running series.apply with a string literal that
    doesn't match a method or Numpy function raises an
    Exception.
    """

    def impl1(S):
        # This function doesn't exist in Numpy or as a
        # Series method.
        return S.apply("concat")

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(S)


@pytest.mark.slow
def test_series_apply_pandas_unsupported_method(memory_leak_check):
    """
    Test running series.apply with a string literal that
    matches an unsupported Series method raises an appropriate
    exception.
    """

    def impl1(S):
        return S.apply("argmin")

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(S)


@pytest.mark.slow
def test_series_apply_numpy_unsupported_ufunc(memory_leak_check):
    """
    Test running series.apply with a string literal that
    matches an unsupported ufunc raises an appropriate
    exception.
    """

    def impl1(S):
        return S.apply("frexp")

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(S)


@pytest.mark.slow
def test_series_apply_pandas_unsupported_type(memory_leak_check):
    """
    Test running series.apply with a string literal that
    matches a method but has an unsupported type
    raises an appropriate exception.
    """

    def impl1(S):
        # Mean is unsupported for string types
        return S.apply("mean")

    S = pd.Series(["abc", "342", "41"] * 100)
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(S)


@pytest.mark.slow
def test_series_apply_numpy_unsupported_type(memory_leak_check):
    """
    Test running series.apply with a string literal that
    matches a Numpy ufunc but has an unsupported type
    raises an appropriate exception.
    """

    def impl1(S):
        # radians is unsupported for string types
        return S.apply("radians")

    S = pd.Series(["abc", "342", "41"] * 100)
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(S)


def test_series_ufunc(memory_leak_check):
    """
    Test running series.apply with a numpy ufunc.
    """

    def test_impl1(S):
        return S.apply(np.radians)

    def test_impl2(S):
        return S.apply(np.abs)

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))
    # Used for abs test
    S[0] = -150

    check_func(test_impl1, (S,))
    check_func(test_impl2, (S,))


@pytest.mark.slow
def test_series_corr(memory_leak_check):
    """
    Test running series.corr().
    """

    def test_impl(S1, S2):
        return S1.corr(S2)

    S1 = pd.Series(list(np.arange(100)) + list(np.arange(100)))
    S2 = pd.Series(np.arange(200))

    check_func(test_impl, (S1, S2))


@pytest.mark.slow
def test_series_cov(memory_leak_check):
    """
    Test running series.cov().
    """

    def test_impl(S1, S2):
        return S1.cov(S2)

    S1 = pd.Series(list(np.arange(100)) + list(np.arange(100)))
    S2 = pd.Series(np.arange(200))

    check_func(test_impl, (S1, S2))


@pytest.mark.slow
def test_series_apply_numpy_unsupported_ufunc_function(memory_leak_check):
    """
    Test running series.apply with a np.ufunc that
    matches an unsupported ufunc raises an appropriate
    exception.
    """

    def impl1(S):
        return S.apply(np.frexp)

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))
    with pytest.raises(
        BodoError, match="Unsupported Numpy ufunc encountered in JIT code"
    ):
        bodo.jit(impl1)(S)


@pytest.mark.slow
def test_series_apply_numpy_ufunc_unsupported_type(memory_leak_check):
    """
    Test running series.apply with a np.ufunc that
    has an unsupported type raises an appropriate
    exception.
    """

    def impl1(S):
        # radians is unsupported for string types
        return S.apply(np.radians)

    S = pd.Series(["abc", "342", "41"] * 100)
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(S)


@pytest.mark.slow
def test_series_apply_numpy_literal_non_ufunc(memory_leak_check):
    """
    Test running series.apply with a string literal that
    matches a Numpy function but not a ufunc raises an
    appropriate exception.
    """

    def impl1(S):
        return S.apply("nansum")

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(S)


@pytest.mark.slow
def test_series_apply_numpy_func_non_ufunc(memory_leak_check):
    """
    Test running series.apply with a numpy function
    but not a ufunc raises an appropriate exception.
    """

    def impl1(S):
        return S.apply(np.nansum)

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(S)


@pytest.mark.slow
def test_series_is_none(memory_leak_check):
    """
    Test that series is None can compile and keep the series distributed.
    """

    def impl1(S):
        return S is None

    def impl2(S):
        return S is not None

    S = pd.Series(np.arange(100))
    check_func(impl1, (S,))
    check_func(impl2, (S,))


@pytest.mark.slow
def test_astype_str_null(memory_leak_check):
    """
    Checks that astype(str) converts Null values to strings
    """

    def impl(S):
        return S.astype(str)

    S = pd.Series([1, 2, 4, None, 7] * 10, dtype="Int64")
    check_func(impl, (S,))

    S = pd.Series([pd.Timestamp(2021, 5, 4, 1), None] * 25)
    check_func(impl, (S,))


def test_astype_str_keep_null(memory_leak_check):
    """
    Checks that astype(str) keeps null values null when _bodo_nan_to_str=False
    """

    def impl(S):
        return S.astype(str, _bodo_nan_to_str=False)

    S = pd.Series([1, 2, 4, None, 7] * 10, dtype="Int64")
    # This is a Bodo specific arg so use py_output
    py_output = S.astype(str)
    py_output[py_output == "<NA>"] = None
    check_func(impl, (S,), py_output=py_output)

    S = pd.Series([pd.Timestamp(2021, 5, 4, 1), None] * 25)
    # This is a Bodo specific arg so use py_output
    py_output = S.astype(str)
    py_output[py_output == "NaT"] = None
    check_func(impl, (S,), py_output=py_output)


@pytest.mark.slow
def test_series_apply_multi_freevar(memory_leak_check):
    """
    Bug revealed by starschema in BodoSQL. Ensures that if you
    have multiple freevars all of them are properly removed.
    """

    def impl(S, freevar1, freevar2):
        return S.apply(lambda x: x + freevar1 + freevar2)

    S = pd.Series(np.arange(100))
    check_func(impl, (S, 2, 5))


@pytest.mark.slow
def test_series_categorical_astype_str(memory_leak_check):
    """
    Tests support for Series.astype(str, _bodo_nan_to_str=False)
    Needed for BodoSQL.
    """

    def impl(S):
        return S.astype(str, _bodo_nan_to_str=False)

    S = pd.Series(pd.Categorical([1, 2, 4, None, 5] * 5))
    py_output = S.astype(str)
    py_output[py_output == "nan"] = None
    check_func(impl, (S,), py_output=py_output)


def test_heterogeneous_series_box(memory_leak_check):
    """
    Tests support for boxing a Heterogeneeous Series.
    """

    def g(row):
        return 0

    def impl(df):
        def f(row):
            with bodo.objmode(res="int64"):
                res = g(row)
            return res

        return df.apply(f, axis=1)

    df = pd.DataFrame(
        {
            "A": pd.Series([1, None, None] * 5, dtype="Int64"),
            "B": pd.Series([None, "a", "c"] * 5),
        }
    )
    check_func(impl, (df,))


@pytest.mark.parametrize("dtype", ["Int64", "Int32", np.int32, np.int64])
def test_astype_int32(dtype, memory_leak_check):
    """
    Tests conversion of pd.Int32 series to np.int32.
    """

    def impl(A):
        return A.astype(np.int32)

    A = pd.Series([1, 2, 3, 4, 5], dtype=dtype)
    check_func(impl, (A,))


def test_astype_nocopy(memory_leak_check):
    """
    Tests to make sure conversion of pd.Int32 series to pd.Int32 does no conversion.
    """

    def impl(A):
        return A.astype("Int32", copy=False)

    A = pd.Series([1, 2, 3, 4, 5], dtype="Int32")
    check_func(impl, (A,))

    # ensure no parfor is created
    bodo_func = bodo.jit(pipeline_class=ParforTestPipeline)(impl)
    bodo_func(A)
    _check_num_parfors(bodo_func, 0)


S = pd.Series([1, 2, 3], name="ABC")


def test_constant_lowering(memory_leak_check):
    """
    Tests constant lowering for Series values
    """

    def impl():
        return S

    check_func(impl, (), only_seq=True)


@pytest.mark.parametrize("offset", ("15D", pd.DateOffset(days=15), "0D"))
def test_series_first_last(offset):
    """
    Test Series.first() and Series.last() with string and DateOffset offsets (for Series with DateTimeIndex)
    """

    def impl_first(S):
        return S.first(offset)

    def impl_last(S):
        return S.last(offset)

    n = 30
    i = pd.date_range("2018-04-09", periods=n, freq="2D")
    ts = pd.Series(np.arange(n), index=i)

    # Pandas functionality is ostensibly incorrect, see:
    # https://github.com/pandas-dev/pandas/blob/v1.4.0/pandas/core/generic.py#L8401-L8463
    if isinstance(offset, pd.DateOffset):
        end_date = end = ts.index[0] + offset
        # Tick-like, e.g. 3 weeks
        if isinstance(offset, pd._libs.tslibs.Tick) and end_date in ts.index:
            end = ts.index.searchsorted(end_date, side="left")
            py_output = ts.iloc[:end]
        else:
            py_output = ts.loc[:end]
    else:
        py_output = None

    check_func(impl_first, (ts,), py_output=py_output)
    check_func(impl_last, (ts,))


def test_empty_series_first_last():
    """
    Test Series.first() and Series.last() with an empty series.
    """

    def impl_first(S):
        return S.first("5D")

    def impl_last(S):
        return S.last("5D")

    n = 10
    ts = pd.Series(
        np.arange(n), index=pd.date_range("2018-04-09", periods=n, freq="1D")
    )
    empty_ts = ts[ts > n]

    check_func(impl_first, (empty_ts,))
    check_func(impl_last, (empty_ts,))


def _convert_helper(df, typ):
    # NOTE: first value in each column must not be NA
    res = df.fillna(method="ffill").apply(lambda r: r.astype(typ), axis=1)
    res[pd.isna(df)] = np.nan
    res.replace(np.nan, pd.NA)
    return res


@pytest.mark.parametrize(
    "to_type",
    (
        "str",
        # TODO: support non-nullable boolean
        "bool",
        "boolean",
        "int",
        "float",
        "float32",
        "float64",
        "Int8",
        "UInt8",
        "Int16",
        "UInt16",
        "Int32",
        "UInt32",
        "Int64",
        "UInt64",
        # TODO: support non-nullable types
        # below equivalent to np.int* or np.float*
        # "int8",
        # "int16",
        # "int32",
        # "int64",
        # "uint8",
        # "uint16",
        # "uint32",
        # "uint64",
        "datetime64[ns]",
        "timedelta64[ns]",
    ),
)
def test_heterogeneous_series_df_apply_astype(to_type):
    """
    Tests astype on heterogenous series produced by df.apply
    i.e. type conversion for Series of tuples
    """

    def test_impl(df):
        res = df.apply(lambda row: row.astype(to_type), axis=1)
        return res

    df_str = pd.DataFrame(
        {
            "A": pd.array([1, 0, 3] * 2, dtype="Int32"),
            "B": [4.3, 2.4, 1.2] * 2,
            "C": ["a", None, "c"] * 2,
            "D": [True, True, False] * 2,
        }
    )
    df_int = pd.DataFrame(
        {
            "A": pd.array([1, 295, 3] * 2, dtype="Int32"),
            "B": pd.Series([4, None, 7] * 2, dtype="Int8"),
            "C": ["1", None, "4"] * 2,
        }
    )
    df_float = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3] * 2, dtype="Int32"),
            "B": pd.array([4.3, None, 1.2] * 2, dtype="float32"),
            "C": pd.Series([4, None, 7] * 2, dtype="Int8"),
            "D": ["1.2", None, "4.5"] * 2,
        }
    )
    df_dt = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3] * 2, dtype="Int64"),
            "B": pd.array([1, 2, 3] * 2, dtype="UInt64"),
            "C": pd.date_range("01-01-2022", periods=6),
            "D": ["01-01-2021", "01-02-2021", "01-03-2022"] * 2,
        }
    )
    df_td = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3] * 2, dtype="Int64"),
            "B": pd.array([1, 2, 3] * 2, dtype="UInt64"),
            "C": pd.timedelta_range("1 second", periods=6),
            # String columns don't work properly in Pandas
        }
    )
    if to_type in ("str", "bool", "boolean"):
        if to_type == "str":

            def _str_helper(x):
                if pd.isna(x):
                    return pd.NA
                elif type(x) == float:
                    return "{:.6f}".format(x)
                else:
                    return str(x)

            exp_output = df_str.applymap(_str_helper)
        elif to_type in ("bool", "boolean"):
            exp_output = df_str.applymap(lambda x: bool(x) if not pd.isna(x) else pd.NA)
        check_func(test_impl, (df_str,), check_dtype=False, py_output=exp_output)
    elif "int" in to_type or "Int" in to_type:
        int_func = {
            "int": int,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
        }
        exp_output = df_int.applymap(
            lambda x: int_func[to_type.lower()](x) if not pd.isna(x) else pd.NA
        )
        check_func(test_impl, (df_int,), py_output=exp_output)
    elif "float" in to_type:
        float_func = {
            "float": float,
            "float32": np.float32,
            "float64": np.float64,
        }
        exp_output = df_float.applymap(
            lambda x: float_func[to_type](x)
            if not pd.isna(x)
            else float_func[to_type](np.nan)
        )
        check_func(test_impl, (df_float,), py_output=exp_output)
    elif to_type == "datetime64[ns]":
        # astype behavior fails when iterating through rows in pandas
        py_output = df_dt.astype(to_type)
        check_func(test_impl, (df_dt,), py_output=py_output)
    elif to_type == "timedelta64[ns]":
        check_func(test_impl, (df_td,))


def test_heterogeneous_series_df_apply_astype_classes():
    """
    Tests astype on heterogenous series produced by df.apply
    i.e. type conversion for Series of tuples
    """

    def test_impl_str(df):
        return df.apply(lambda row: row.astype(str), axis=1)

    def test_impl_bool(df):
        return df.apply(lambda row: row.astype(bool), axis=1)

    def test_impl_int(df):
        return df.apply(lambda row: row.astype(int), axis=1)

    def test_impl_float(df):
        return df.apply(lambda row: row.astype("float"), axis=1)

    df_str = pd.DataFrame(
        {
            "A": pd.array([1, 0, 3] * 2, dtype="Int32"),
            "B": [4.3, 2.4, 1.2] * 2,
            "C": ["a", None, "c"] * 2,
            "D": [True, True, False] * 2,
        }
    )
    df_int = pd.DataFrame(
        {
            "A": pd.array([1, 295, 3] * 2, dtype="Int32"),
            "B": pd.Series([4, None, 7] * 2, dtype="Int8"),
            "C": ["1", None, "4"] * 2,
        }
    )
    df_float = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3] * 2, dtype="Int32"),
            "B": pd.array([4.3, np.nan, 1.2] * 2, dtype="float"),
            "C": pd.Series([4, None, 7] * 2, dtype="Int8"),
            "D": ["1.2", None, "4.5"] * 2,
        }
    )

    def _str_helper(x):
        if pd.isna(x):
            return pd.NA
        elif type(x) == float:
            return "{:.6f}".format(x)
        else:
            return str(x)

    str_output = df_str.applymap(_str_helper)
    # TODO: nullable bool
    bool_output = df_str.applymap(lambda x: bool(x) if not pd.isna(x) else pd.NA)
    int_output = df_int.applymap(lambda x: int(x) if not pd.isna(x) else pd.NA)
    float_output = df_float.applymap(lambda x: float(x) if not pd.isna(x) else pd.NA)

    check_func(test_impl_str, (df_str,), py_output=str_output)
    check_func(test_impl_bool, (df_str,), py_output=bool_output)
    check_func(test_impl_int, (df_int,), py_output=int_output)
    # TODO: error with float32
    # check_func(test_impl_float, (df_float,), py_output=float_output)


@pytest.mark.parametrize(
    "S_val",
    [
        pd.Series([1, 2, 1, 3, 6, 4, 2, 3, 5, 1, 9]),
        pd.Series(["a", "b", "c", "a", "b", "d", "e", "c"]),
        pd.Series(
            ["4 h", "1 h", "1 h", "2 h", "2 h", "3 h", "5 h"], dtype="timedelta64[ns]"
        ),
        pd.Series(["a", None, "c", "a", None, "d", None, "c"]),
    ],
)
def test_series_duplicated(S_val, memory_leak_check):
    """
    Tests for Series.duplicated()
    """

    def test_impl(S):
        return S.duplicated()

    check_func(test_impl, (S_val,), sort_output=True, reset_index=True)


@pytest.mark.parametrize(
    "arr",
    [
        [None] * 5 + [None, None, 2, 3, None] + [None] * 5,
        [None, None, 2, 3, None] + [None, None, 2, 3, None] + [None] * 5,
        np.arange(15),
        [None] * 15,
    ],
)
@pytest.mark.parametrize(
    "idx",
    [
        pd.Index(list(ascii_lowercase[:15])),
        pd.Index(np.arange(100, 115)),
        pd.date_range("01-01-2022", periods=15, freq="D"),
    ],
)
def test_series_first_last_valid_index(arr, idx):
    """
    Tests for Series.first_valid_index() and Series.last_valid_index()
    """

    def test_first(S):
        return S.first_valid_index()

    def test_last(S):
        return S.last_valid_index()

    def f(S):
        idx_val = S.first_valid_index()
        return idx_val.date

    s = pd.Series(arr, index=idx)
    check_func(test_last, (s,), check_typing_issues=False)
    check_func(test_first, (s,), check_typing_issues=False)


def test_series_rename_axis():
    """
    Tests Series.rename_axis() support.
    """

    def test_impl(S, name):
        return S.rename_axis(name)

    s = pd.Series(np.arange(100))
    check_func(
        test_impl,
        (
            s,
            "name",
        ),
    )
    check_func(
        test_impl,
        (
            s,
            10,
        ),
    )

    with pytest.raises(
        BodoError,
        match=re.escape(
            "Series.rename_axis(): 'mapper' is required and must be a scalar type."
        ),
    ):
        bodo.jit(test_impl)(s, ["a", "b"])


def test_unique(memory_leak_check):
    """
    Tests for pd.unique() on 1-d array and Series
    """

    def test_impl(vals):
        return pd.unique(vals)

    s = pd.Series([1, 2, 3, 4, 2, 3, 5] * 5)
    a = np.array([1, 2, 3, 4, 2, 3, 5] * 5)
    check_func(test_impl, (s,), sort_output=True, is_out_distributed=False)
    check_func(test_impl, (a,), sort_output=True, is_out_distributed=False)
    with pytest.raises(
        BodoError,
        match=re.escape("pd.unique(): 'values' must be either a Series or a 1-d array"),
    ):
        bodo.jit(test_impl)([1, 2, 3, 4, 2, 3, 5] * 5)


@pytest.mark.parametrize(
    "S",
    [
        pd.Series(np.arange(100)),
        pd.Series(pd.date_range("02-20-2022", freq="3D1H", periods=30)),
    ],
)
def test_dist_iat(S, memory_leak_check):
    """
    Tests for [BE-2838], replicated return values from
    Series.iat.
    """

    def impl(S):
        return S.iat[1]

    check_func(impl, (S,))


def test_iat_control_flow(memory_leak_check):
    """
    Tests for [BE-2838], replicated return values from
    Series.iat with control flow.
    """

    def impl(S1, S2, flag):
        if flag:
            s_iat = S1.iat
        else:
            s_iat = S2.iat
        return s_iat[1]

    S1 = pd.Series(np.arange(100))
    S2 = pd.Series(np.arange(100, 200))

    check_func(impl, (S1, S2, False))


@pytest.mark.parametrize(
    "S",
    [
        pd.Series([2, 1, 2, 4, 8, 4]),
        pd.Series([None, "b", "a", "b", "c", "â", "c"]),
        pd.Series([np.nan, 2, 1, 2, 4.2, 8, 4.2, np.nan]),
        pd.Series([None, True, False, False, True, True, None]),
    ],
)
@pytest.mark.parametrize(
    "method",
    ["average", "min", "max", "first", "dense"],
)
@pytest.mark.parametrize(
    "na_option",
    ["keep", "top", "bottom"],
)
@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize("pct", [True, False])
def test_series_rank(S, method, na_option, ascending, pct, memory_leak_check):
    """
    Tests for Series.rank
    """

    def impl(S):
        return S.rank(method=method, na_option=na_option, ascending=ascending, pct=pct)

    if method == "first" and not ascending:
        with pytest.raises(
            BodoError,
            match=re.escape(
                "Series.rank(): method='first' with ascending=False is currently unsupported."
            ),
        ):
            bodo.jit(impl)(S)
    else:
        check_func(impl, (S,), dist_test=False)


@pytest.mark.slow()
def test_np_mod(numeric_series_val):
    """tests that np.mod is properly supported"""

    # np.mod not supported for datetime types
    if isinstance(numeric_series_val.iloc[0], pd.Timestamp):
        pytest.skip("np.mod not supported for datetime types")

    def impl(arg1, arg2):
        return np.mod(arg1, arg2)

    check_func(impl, (numeric_series_val, numeric_series_val), check_dtype=False)
    check_func(impl, (10, numeric_series_val), check_dtype=False)

@pytest.mark.tz_aware
def test_tz_aware_series_getitem(memory_leak_check):
    def impl_iat(S):
        return S.iat[2]

    def impl_iloc(S):
        return S.iloc[2]

    def impl_loc(S):
        return S.loc[2]

    def impl_regular(S):
        return S[2]

    S = pd.Series(pd.array([pd.Timestamp("2000-01-01", tz="US/Pacific")] * 10))
    check_func(impl_iat, (S,))
    check_func(impl_iloc, (S,))
    check_func(impl_loc, (S,))
    check_func(impl_regular, (S,))
