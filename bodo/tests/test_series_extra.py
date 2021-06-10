import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func


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

    # Not need by BodoSQL but seems relevant for coverage
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
