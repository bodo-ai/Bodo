from decimal import Decimal

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


@pytest.mark.slow
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


@pytest.mark.slow
def test_series_groupby_args(memory_leak_check):
    """ Test Series.groupby with all unsupported and wrong arguments"""

    def test_impl_by_level(S):
        return S.groupby(by=["a", "b", "a", "b"], level=0).mean()

    def test_impl_no_by_no_level(S):
        return S.groupby().mean()

    def test_axis(S):
        return S.groupby(axis=1).mean()

    def test_as_index(S):
        return S.groupby(as_index=False).mean()

    def test_group_keys(S):
        return S.groupby(group_keys=False).mean()

    def test_observed(S):
        return S.groupby(observed=True).mean()

    def test_dropna(S):
        return S.groupby(dropna=False).mean()

    # deprecated since 1.1.0
    def test_squeeze(S):
        return S.groupby(squeeze=True).mean()

    S = pd.Series([390.0, 350.0, 30.0, 20.0])

    with pytest.raises(BodoError, match="Series.groupby.* argument should be None if"):
        bodo.jit(test_impl_by_level)(S)

    with pytest.raises(BodoError, match="You have to supply one of 'by' and 'level'"):
        bodo.jit(test_impl_no_by_no_level)(S)

    with pytest.raises(BodoError, match="only valid with DataFrame"):
        bodo.jit(test_as_index)(S)

    with pytest.raises(BodoError, match="parameter only supports default"):
        bodo.jit(test_axis)(S)
        bodo.jit(test_group_keys)(S)
        bodo.jit(test_observed)(S)
        bodo.jit(test_dropna)(S)
        bodo.jit(test_squeeze)(S)


@pytest.mark.slow
def test_series_groupby_by_arg_unsupported_types(memory_leak_check):
    """ Test Series.groupby by argument with Bodo Types that it doesn't currently support"""

    def test_by_type(S, byS):
        return S.groupby(byS).max()

    with pytest.raises(BodoError, match="not supported yet"):
        S = pd.Series([390.0, 350.0, 30.0, 20.0, 5.5])
        byS = pd.Series(
            [
                Decimal("1.6"),
                Decimal("-0.2"),
                Decimal("44.2"),
                np.nan,
                Decimal("0"),
            ]
        )
        bodo.jit(test_by_type)(S, byS)
        byS = pd.Series([1, 8, 4, 10, 3], dtype="Int32")
        bodo.jit(test_by_type)(S, byS)
        byS = pd.Series(pd.Categorical([1, 2, 5, 1, 2], ordered=True))
        bodo.jit(test_by_type)(S, byS)


# TODO: Mark as slow after CI passes
def test_series_idxmax_unordered_cat(memory_leak_check):
    def impl(S):
        return S.idxmax()

    S = pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=False))

    match = "Series.idxmax.*: only ordered categoricals are possible"
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl)(S)


# TODO: Mark as slow after CI passes
def test_series_idxmin_unordered_cat(memory_leak_check):
    def impl(S):
        return S.idxmin()

    S = pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=False))

    match = "Series.idxmin.*: only ordered categoricals are possible"
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl)(S)
