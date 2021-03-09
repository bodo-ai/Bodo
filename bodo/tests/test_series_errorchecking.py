import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


def test_isin(memory_leak_check):
    """
    tests error for 'values' argument of Series.isin()
    """

    def impl(S, values):
        return S.isin(values)

    S = pd.Series([3, 2, np.nan, 2, 7], [3, 4, 2, 1, 0], name="A")
    with pytest.raises(BodoError, match="parameter should be a set or a list"):
        bodo.jit(impl)(S, "3")


@pytest.mark.slow
def test_series_dt_not_supported(memory_leak_check):
    """
    tests error for unsupported Series.dt methods
    """

    def impl():
        rng = pd.date_range("20/02/2020", periods=5, freq="M")
        ts = pd.Series(np.random.randn(len(rng)), index=rng)
        ps = ts.to_period()
        return ps

    with pytest.raises(BodoError, match="not supported"):
        bodo.jit(impl)()


@pytest.mark.slow
def test_series_head_errors(memory_leak_check):
    def impl():
        S = pd.Series(np.random.randn(10))
        return S.head(5.0)

    with pytest.raises(BodoError, match="Series.head.*: 'n' must be an Integer"):
        bodo.jit(impl)()


@pytest.mark.slow
def test_series_tail_errors(memory_leak_check):
    def impl():
        S = pd.Series(np.random.randn(10))
        return S.tail(5.0)

    with pytest.raises(BodoError, match="Series.tail.*: 'n' must be an Integer"):
        bodo.jit(impl)()


@pytest.mark.slow
def test_series_rename_none(memory_leak_check):
    S = pd.Series(np.random.randn(10))

    def impl(S):
        return S.rename(None)

    with pytest.raises(BodoError, match="Series.rename.* 'index' can only be a string"):
        bodo.jit(impl)(S)


@pytest.mark.slow
def test_series_take_errors(memory_leak_check):
    S = pd.Series(np.random.randn(10))

    def impl1(S):
        return S.take([1.0, 2.0])

    def impl2(S):
        return S.take(1)

    err_msg = "Series.take.* 'indices' must be an array-like and contain integers."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(S)

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(S)


# TODO: Mark as slow after CI passes
def test_series_map_runtime_categorical(memory_leak_check):
    """
    Tests that a UDF with categories that aren't known at
    compile time throws a reasonable error.
    """
    # TODO: Modify test once input -> output categories are supported.

    S = pd.Series(["A", "bbr", "wf", "cewf", "Eqcq", "qw"])

    def impl(S):
        cat_series = S.astype("category")
        return cat_series.map(lambda x: x)

    with pytest.raises(
        BodoError,
        match="UDFs or Groupbys that return Categorical values must have categories known at compile time.",
    ):
        bodo.jit(impl)(S)


def test_series_dropna_axis_error(memory_leak_check):
    S = pd.Series([0, np.nan, 1])

    match = "axis parameter only supports default value 0"
    with pytest.raises(BodoError, match=match):
        bodo.jit(lambda: S.dropna(axis=1))()


def test_series_dropna_inplace_error(memory_leak_check):
    S = pd.Series([0, np.nan, 1])

    match = "inplace parameter only supports default value False"
    with pytest.raises(BodoError, match=match):
        bodo.jit(lambda: S.dropna(inplace=True))()
