import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


def test_dataframe_apply_method_str(memory_leak_check):
    """
    Test running DataFrame.apply with a string literal that
    matches a DataFrame method. Note by default all of these
    will run with axis=0 if the argument exists.

    """

    def impl1(df):
        # Test a DataFrame method that returns a Series without axis=1.
        return df.apply("nunique")

    def impl2(df):
        # Test a DataFrame method that conflicts with a numpy function
        return df.apply("sum", axis=1)

    def impl3(df):
        # Test a DataFrame method that returns a DataFrame
        return df.apply("abs")

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )

    check_func(impl1, (df,), is_out_distributed=False)
    check_func(impl2, (df,))
    check_func(impl3, (df,))


@pytest.mark.skip("[BE-1198] Support numpy ufuncs on DataFrames")
def test_dataframe_apply_numpy_str(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches a Numpy function.
    """

    def impl1(df):
        return df.apply("sin")

    def impl2(df):
        # Test with axis=1 (unused)
        return df.apply("log", axis=1)

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )

    check_func(impl1, (df,))
    check_func(impl2, (df,))


@pytest.mark.slow
def test_dataframe_apply_no_func(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    doesn't match a method or Numpy function raises an
    Exception.
    """

    def impl1(df):
        # This function doesn't exist in Numpy or as a
        # DataFrame method.
        return df.apply("concat", axis=1)

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_pandas_unsupported_method(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches an unsupported DataFrame method raises an appropriate
    exception.
    """

    def impl1(df):
        return df.apply("argmin", axis=1)

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_numpy_unsupported_ufunc(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches an unsupported ufunc raises an appropriate
    exception.
    """

    def impl1(df):
        return df.apply("cbrt", axis=1)

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_pandas_unsupported_type(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches a method but has an unsupported type
    raises an appropriate exception.
    """

    def impl1(df):
        # Mean is unsupported for string types
        return df.apply("mean", axis=1)

    df = pd.DataFrame(
        {
            "A": ["ABC", "21", "231", "21dwcp"] * 25,
            "B": ["feq", "3243412rfe", "fonie wqw   ", "3c", "r32r23fc"] * 20,
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_pandas_unsupported_axis(memory_leak_check):
    """
    Test running dataframe.apply with a method using
    axis=1 when Bodo doesn't support axis=1 yet.
    """

    def impl1(df):
        # nunique is unsupported for axis=1
        return df.apply("nunique", axis=1)

    df = pd.DataFrame(
        {
            "A": ["ABC", "21", "231", "21dwcp"] * 25,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


@pytest.mark.slow
def test_dataframe_apply_numpy_unsupported_type(memory_leak_check):
    """
    Test running dataframe.apply with a string literal that
    matches a Numpy ufunc but has an unsupported type
    raises an appropriate exception.
    """

    def impl1(df):
        # radians is unsupported for string types
        return df.apply("radians", axis=1)

    df = pd.DataFrame(
        {
            "A": ["ABC", "21", "231", "21dwcp"] * 25,
            "B": -np.arange(100),
        }
    )
    with pytest.raises(BodoError, match="user-defined function not supported"):
        bodo.jit(impl1)(df)


def test_dataframe_optional_scalar(memory_leak_check):
    """
    Test calling pd.DataFrame with a scalar that is an optional type.
    """

    def impl(table1):
        df1 = pd.DataFrame({"A": table1["A"], "$f3": table1["A"] == np.int32(1)})
        S0 = df1["A"][df1["$f3"]]
        df2 = pd.DataFrame(
            {"col1_sum_a": S0.sum() if len(S0) > 0 else None},
            index=pd.RangeIndex(0, 1, 1),
        )
        return df2

    df = pd.DataFrame({"A": [1, 2, 3] * 4})

    # Pandas can avoid nullable so the types don't match
    check_func(impl, (df,), check_dtype=False)
