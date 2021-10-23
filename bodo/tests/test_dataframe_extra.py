import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

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


def test_dataframe_is_none(memory_leak_check):
    """
    Test that dataframe is None can compile and keep the dataframe distributed.
    """

    def impl1(df):
        return df is None

    def impl2(df):
        return df is not None

    df = pd.DataFrame(
        {
            "A": np.arange(100) % 10,
            "B": -np.arange(100),
        }
    )
    check_func(impl1, (df,))
    check_func(impl2, (df,))
    # Test that none still works
    check_func(impl1, (None,))
    check_func(impl2, (None,))


@pytest.mark.slow
def test_describe_many_columns(memory_leak_check):
    """
    Runs df.describe on a dataframe with 25 columns.
    If df.describe is inlined, this will take more than
    1 hour.
    If df.describe is not inlined as expected, this should take
    closer to 12 seconds. To avoid failures due to small variations,
    we will check this tests take no longer than 1 minute.
    """

    def impl(df):
        return df.describe()

    df = pd.DataFrame(
        columns=np.arange(25), data=np.arange(1000 * 25).reshape(1000, 25)
    )
    import time

    t0 = time.time()
    check_func(impl, (df,), is_out_distributed=False)
    compilation_time = time.time() - t0
    # Determine the max compilation time on any rank to avoid hangs.
    comm = MPI.COMM_WORLD
    compilation_time = comm.allreduce(compilation_time, op=MPI.MAX)
    assert (
        compilation_time < 60
    ), "df.describe() took too long to compile. Possible regression?"


@pytest.mark.parametrize(
    "dt_like_series",
    [
        pd.Series([pd.Timestamp(2021, 4, 3), None] * 10),
        pd.Series([pd.Timedelta(days=-2), None] * 10),
    ],
)
def test_optional_fusion(memory_leak_check, dt_like_series):
    """
    Checks that pd.DataFrame can be used on multiple series
    operations with optional types. This triggers a parfor fusion
    that keeps values as optional types when merging functions
    (see BE-1396)
    """

    def impl(S):
        return pd.DataFrame(
            {"A": S.apply(lambda x: (None if (pd.isna(x)) else x)).isna()}
        )

    check_func(impl, (dt_like_series,))


def test_astype_str_null(memory_leak_check):
    """
    Checks that astype(str) converts Null values to strings
    """

    def impl(df):
        return df.astype(str)

    df = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 4, None, 7] * 10, dtype="Int64"),
            "B": [pd.Timestamp(2021, 5, 4, 1), None] * 25,
        }
    )
    check_func(impl, (df,))


def test_astype_str_keep_null(memory_leak_check):
    """
    Checks that astype(str) keeps null values null when _bodo_nan_to_str=False
    """

    def impl(S):
        return S.astype(str, _bodo_nan_to_str=False)

    df = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 4, None, 7] * 10, dtype="Int64"),
            "B": [pd.Timestamp(2021, 5, 4, 1), None] * 25,
        }
    )
    # This is a Bodo specific arg so use py_output
    py_output = df.astype(str)
    py_output["A"][py_output["A"] == "<NA>"] = None
    py_output["B"][py_output["B"] == "NaT"] = None
    check_func(impl, (df,), py_output=py_output)


def test_categorical_astype(memory_leak_check):
    """
    Test that astype with categorical columns to the underlying
    elem type works as expected. Needed for BodoSQL.
    """

    def impl(categorical_table):
        categorical_table = pd.DataFrame(
            {
                "A": categorical_table["A"].astype("str"),
                "B": categorical_table["B"].astype("Int64"),
                "C": categorical_table["C"].astype("UInt64"),
                "D": categorical_table["D"].astype("float64"),
                "E": categorical_table["E"].astype("datetime64[ns]"),
                "F": categorical_table["F"].astype("timedelta64[ns]"),
                "G": categorical_table["G"].astype("boolean"),
            }
        )
        return categorical_table

    df = pd.DataFrame(
        {
            # String category
            "A": pd.Categorical(["anve", "Er2"] * 5),
            # int64
            "B": pd.Categorical([5, -32] * 5),
            # uint64
            "C": pd.Categorical(pd.array([5, 2] * 5, "uint64")),
            # float64
            "D": pd.Categorical([1.1, 2.7] * 5),
            # dt64
            "E": pd.Categorical(
                [pd.Timestamp(2021, 4, 5), pd.Timestamp(2021, 4, 4)] * 5
            ),
            # td64
            "F": pd.Categorical([pd.Timedelta(2), pd.Timedelta(seconds=-4)] * 5),
            # boolean
            "G": pd.Categorical([True, False] * 5),
        }
    )

    check_func(impl, (df,))
