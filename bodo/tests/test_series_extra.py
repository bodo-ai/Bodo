import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
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
        return S.apply("cbrt")

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
def test_series_apply_numpy_unsupported_ufunc_function(memory_leak_check):
    """
    Test running series.apply with a np.ufunc that
    matches an unsupported ufunc raises an appropriate
    exception.
    """

    def impl1(S):
        return S.apply(np.cbrt)

    S = pd.Series(list(np.arange(100) + list(np.arange(100))))
    with pytest.raises(BodoError, match="user-defined function not supported"):
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
