"""Test that inputs and outputs are supported by spawn mode"""

import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.range import RangeIndex

import bodo
from bodo.tests.dataframe_common import df_value  # noqa
from bodo.tests.series_common import series_val  # noqa
from bodo.tests.utils import _get_dist_arg, check_func, pytest_spawn_mode

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
    ):
        pytest.skip("BSE-4099: Support all pandas index types in lazy wrappers")

    def test(df):
        return df

    check_func(test, (df_value,), only_spawn=True, check_pandas_types=False)


def test_distributed_input_output_series(series_val):
    if (
        not isinstance(series_val.index, RangeIndex)
        and type(series_val.index) is not pd.Index
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
