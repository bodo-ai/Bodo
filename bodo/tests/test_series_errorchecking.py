import re
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


@pytest.mark.slow
def test_dropna_axis_error(memory_leak_check):
    """test axis argument error checking in Series.dropna()"""
    S = pd.Series([0, np.nan, 1])
    match = "axis parameter only supports default value 0"
    with pytest.raises(BodoError, match=match):
        bodo.jit(lambda: S.dropna(axis=1))()


@pytest.mark.slow
def test_dropna_inplace_error(memory_leak_check):
    """test inplace argument error checking in Series.dropna()"""
    S = pd.Series([0, np.nan, 1])
    match = "inplace parameter only supports default value False"
    with pytest.raises(BodoError, match=match):
        bodo.jit(lambda: S.dropna(inplace=True))()


@pytest.mark.slow
def test_replace_inplace_error(memory_leak_check):
    """test inplace argument error checking in Series.replace()"""
    S = pd.Series([1, 2, 3])
    message = "inplace parameter only supports default value False"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: S.replace(1, 10, inplace=True))()


@pytest.mark.slow
def test_replace_limit_error(memory_leak_check):
    """test limit argument error checking in Series.replace()"""
    S = pd.Series([1, 2, 3])
    message = "limit parameter only supports default value None"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: S.replace(1, 10, limit=3))()


@pytest.mark.slow
def test_replace_regex_error(memory_leak_check):
    """test regex argument error checking in Series.replace()"""
    S = pd.Series([1, 2, 3])
    message = "regex parameter only supports default value False"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: S.replace(1, 10, regex=True))()


@pytest.mark.slow
def test_replace_method_error(memory_leak_check):
    """test method argument error checking in Series.replace()"""
    S = pd.Series([1, 2, 3])
    message = "method parameter only supports default value pad"
    with pytest.raises(BodoError, match=message):
        bodo.jit(lambda: S.replace(1, 10, method=None))()


@pytest.mark.slow
def test_replace_dict_error():
    """test 'value' is None when 'to_replace' is a dictionary in
    Series.replace()"""

    def impl(series, to_replace, value):
        return series.replace(to_replace, value)

    series = pd.Series(np.arange(10))
    to_replace = dict(zip(np.arange(10), np.arange(10) * 10))
    value = 10
    message = "'value' must be None when 'to_replace' is a dictionary"
    with pytest.raises(BodoError, match=message):
        bodo.jit(impl)(series, to_replace, value)


@pytest.mark.slow
def test_replace_unequal_list_lengths():
    """test Series.replace() fails at runtime when 'to_replace' and
    'value' are lists of differing lengths"""

    def impl(series, to_replace, value):
        return series.replace(to_replace, value)

    series = pd.Series([1, 2, 3, 4] * 4)
    to_replace = [1, 2, 3]
    value = [9]
    message = "lengths must be the same"
    with pytest.raises(AssertionError, match=message):
        bodo.jit(impl)(series, to_replace, value)


@pytest.mark.slow
def test_series_groupby_args(memory_leak_check):
    """Test Series.groupby with all unsupported and wrong arguments"""

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
        return S.groupby(observed=False).mean()

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
    """Test Series.groupby by argument with Bodo Types that it doesn't currently support"""

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


@pytest.mark.slow
def test_series_idxmax_unordered_cat(memory_leak_check):
    def impl(S):
        return S.idxmax()

    S = pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=False))

    match = "Series.idxmax.*: only ordered categoricals are possible"
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl)(S)


@pytest.mark.slow
def test_series_idxmin_unordered_cat(memory_leak_check):
    def impl(S):
        return S.idxmin()

    S = pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=False))

    match = "Series.idxmin.*: only ordered categoricals are possible"
    with pytest.raises(BodoError, match=match):
        bodo.jit(impl)(S)


def test_series_groupby_max_min_cat_unordered(memory_leak_check):
    def test_impl1(S):
        return S.groupby(level=0).max()

    def test_impl2(S):
        return S.groupby(level=0).min()

    S = pd.Series(pd.Categorical([1, 2, 5, None, 2], ordered=False))
    with pytest.raises(
        BodoError,
        match="categorical column must be ordered in groupby built-in function",
    ):
        bodo.jit(test_impl1)(S)
    with pytest.raises(
        BodoError,
        match="categorical column must be ordered in groupby built-in function",
    ):
        bodo.jit(test_impl2)(S)


def test_series_first_last_invalid_offset():
    def test_impl1(S, off):
        return S.first(off)

    def test_impl2(S, off):
        return S.last(off)

    i = pd.date_range("2018-04-09", periods=10, freq="2D")
    ts = pd.Series(np.arange(10), index=i)
    off = 2
    with pytest.raises(
        BodoError,
        match=re.escape("Series.first(): 'offset' must be a string or a DateOffset"),
    ):
        bodo.jit(test_impl1)(ts, off)
    with pytest.raises(
        BodoError,
        match=re.escape("Series.last(): 'offset' must be a string or a DateOffset"),
    ):
        bodo.jit(test_impl2)(ts, off)


@pytest.mark.slow
def test_cmp_errors(memory_leak_check):
    def test_impl1(S, val):
        return S > val

    def test_impl2(S, val):
        return S == val

    def test_impl3(S, val):
        return S != val

    S = pd.Series([1, 2, 3, 1, 7, 8])
    val = "a"
    with pytest.raises(
        BodoError,
        match="series\\(int64, array\\(int64, 1d, C\\), RangeIndexType\\(none\\), none, REP\\) > unicode_type not supported",
    ):
        bodo.jit(test_impl1)(S, val)
    with pytest.raises(
        BodoError,
        match="series\\(int64, array\\(int64, 1d, C\\), RangeIndexType\\(none\\), none, REP\\) == unicode_type not supported",
    ):
        bodo.jit(test_impl2)(S, val)
    with pytest.raises(
        BodoError,
        match="series\\(int64, array\\(int64, 1d, C\\), RangeIndexType\\(none\\), none, REP\\) != unicode_type not supported",
    ):
        bodo.jit(test_impl3)(S, val)


@pytest.mark.slow
def test_and_or_typing_errors(memory_leak_check):
    """Currently, bodo doesn't allow and/or between int/boolean types. Checks that we raise a reasonable error"""

    def test_and(S1, S2):
        return S1 & S2

    def test_or(S1, S2):
        return S1 | S2

    int_s = pd.Series([1, 2, 3] * 3)
    bool_s = pd.Series([True, False, True] * 3, dtype="boolean")
    import re

    with pytest.raises(
        BodoError,
        match=re.escape(
            "series(int64, array(int64, 1d, C), RangeIndexType(none), none, REP) & series(bool, BooleanArrayType(), RangeIndexType(none), none, REP) not supported"
        ),
    ):
        bodo.jit(test_and)(int_s, bool_s)

    with pytest.raises(
        BodoError,
        match=re.escape(
            "series(int64, array(int64, 1d, C), RangeIndexType(none), none, REP) | series(bool, BooleanArrayType(), RangeIndexType(none), none, REP) not supported"
        ),
    ):
        bodo.jit(test_or)(int_s, bool_s)


@pytest.mark.slow
def test_astype_non_constant_string(memory_leak_check):
    """
    Checks that calling Series.astype(str_value) with a string that
    is not a compile time constant will produce a reasonable BodoError.
    """

    def impl(S, type_str):
        return S.astype(type_str[0])

    S = pd.Series([1, 2, 3, 4] * 10)
    type_str = ["uint64"]

    with pytest.raises(
        BodoError,
        match="Series.astype\\(\\): 'dtype' when passed as string must be a constant value",
    ):
        bodo.jit(impl)(S, type_str)


@pytest.mark.slow
def test_series_init_series_idx():
    """
    Checks that proper error is given when initializing a series with another Series value, and
    specifiying the index values (previously, this caused segfault)
    """

    @bodo.jit
    def impl():
        S = pd.Series([1, 2, 3], index=["A", "B", "C"])
        S2 = pd.Series(S, index=["A", "C"])
        return S2

    with pytest.raises(
        BodoError,
        match="pd.Series\\(\\) does not support index value when input data is a Series",
    ):
        impl()


@pytest.mark.slow
def test_np_select_series_cond(memory_leak_check):
    """tests np select returns the correct error when passed a bool series for cond"""
    np.random.seed(42)

    # For now, can only handle cond == bool ndarry.
    S = pd.Series([1, 2, 3])
    cond = pd.Series(np.random.randint(2, size=len(S)).astype(bool))

    @bodo.jit
    def impl(S, cond):
        choicelist = [S]
        condlist = [cond]
        return np.select(condlist, choicelist)

    err_msg_cat = r".*np.select\(\): 'condlist' argument must be list or tuple of boolean ndarrays. If passing a Series, please convert with pd.Series.to_numpy\(\).*"
    with pytest.raises(BodoError, match=err_msg_cat):
        impl(S, cond)


def test_series_init_dict_non_const_keys():
    """
    Tests the error message when initializing series with non constant keyed dicts.
    """

    @bodo.jit
    def test_impl():
        init_dict = dict()
        for i in range(10):
            init_dict[f"idx_{i}"] = 1
        return pd.Series(init_dict)

    with pytest.raises(
        BodoError,
        match="pd.Series\\(\\): When intializing series with a dictionary, it is required that the dict has constant keys",
    ):
        test_impl()


def test_series_init_dict_idx_kw():
    """
    Checks that proper error is given when initializing a series with dict and also supplying
    an index value.
    """

    @bodo.jit
    def impl_err1():
        S = pd.Series({"A": 2, "B": 3}, ["X", "Y"])
        return S

    @bodo.jit
    def impl_err2():
        S = pd.Series({"A": 2, "B": 3}, index=["X", "Y"])
        return S

    @bodo.jit
    def impl_good1():
        S = pd.Series({"A": 2, "B": 3}, None)
        return S

    @bodo.jit
    def impl_good2():
        S = pd.Series({"A": 2, "B": 3}, index=None)
        return S

    for cur_impl in [impl_err1, impl_err2]:
        with pytest.raises(
            BodoError,
            match="pd.Series\\(\\): Cannot specify index argument when initializing with a dictionary",
        ):
            cur_impl()

    impl_good1()
    impl_good2()


def test_heterogenous_series_unsupported_attr(memory_leak_check):
    """
    Checks that an unsupported attribute for a HeterogenousSeries
    raises a reasonable error message. A HeterogenousSeries
    is a generated when iterating across the rows of a DataFrame
    with different column types.
    """

    @bodo.jit
    def test_impl(df):
        return df.apply(lambda row: row.axes, axis=1)

    df = pd.DataFrame({"A": [1, 2, 3, 4] * 5, "B": ["a", "c", "Er2w", ""] * 5})
    err_msg = re.escape("HeterogeneousSeries.axes not supported yet")
    with pytest.raises(BodoError, match=err_msg):
        test_impl(df)


def test_heterogenous_series_unsupported_method(memory_leak_check):
    """
    Checks that an unsupported method for a HeterogenousSeries
    raises a reasonable error message. A HeterogenousSeries
    is a generated when iterating across the rows of a DataFrame
    with different column types.
    """

    @bodo.jit
    def test_impl(df):
        return df.apply(lambda row: row.any(), axis=1)

    df = pd.DataFrame({"A": [1, 2, 3, 4] * 5, "B": ["a", "c", "Er2w", ""] * 5})
    err_msg = re.escape("HeterogeneousSeries.any not supported yet")
    with pytest.raises(BodoError, match=err_msg):
        test_impl(df)
