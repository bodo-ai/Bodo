# Copyright (C) 2019 Bodo Inc. All rights reserved.
import pandas as pd
import numpy as np
import numba
import bodo
from bodo.utils.typing import BodoError
import pytest


# ------------------------------ df.groupby() ------------------------------ #


def test_groupby_supply_by():
    """
    Test groupby(): 'by' is supplied
    """

    def impl1(df):
        return df.groupby()

    def impl2(df):
        return df.groupby(by=None)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="'by' must be supplied"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="'by' must be supplied"):
        bodo.jit(impl2)(df)


def test_groupby_by_const_str_or_str_list():
    """
    Test groupby(): 'by' is of type const str or str list

    """

    def impl(df):
        return df.groupby(by=1)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError,
        match="'by' parameter only supports a constant column label or column labels",
    ):
        bodo.jit(impl)(df)


def test_groupby_by_labels():
    """
    Test groupby(): 'by' is a valid label or label lists
    """

    def impl(df):
        return df.groupby(by=["A", "D"])

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="invalid key .* for by"):
        bodo.jit(impl)(df)


def test_groupby_axis_default():
    """
    Test groupby(): 'axis' cannot be values other than integer value 0
    """

    def impl1(df):
        return df.groupby(by=["A"], axis=1).sum()

    def impl2(df):
        return df.groupby(by=["A"], axis="1").sum()

    df = pd.DataFrame({"A": [1, 2, 2], "C": [3, 1, 2]})
    with pytest.raises(
        BodoError, match="'axis' parameter only supports integer value 0"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="'axis' parameter only supports integer value 0"
    ):
        bodo.jit(impl2)(df)


def test_groupby_supply_level():
    """
    Test groupby(): 'level' cannot be supplied
    """

    def impl(df):
        return df.groupby(by=["A", "C"], level=2)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="'level' is not supported since MultiIndex is not supported."
    ):
        bodo.jit(impl)(df)


def test_groupby_as_index_bool():
    """
    Test groupby(): 'as_index' must be of type bool
    """

    def impl(df):
        return df.groupby(by=["A", "C"], as_index=2)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="'as_index' parameter must be of type bool"):
        bodo.jit(impl)(df)


def test_groupby_sort_default():
    """
    Test groupby(): 'sort' cannot have values other than boolean value False
    """

    def impl1(df):
        return df.groupby(by=["A", "C"], sort=1)

    def impl2(df):
        return df.groupby(by=["A", "C"], sort=True)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="'sort' parameter only supports default value False"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="'sort' parameter only supports default value False"
    ):
        bodo.jit(impl2)(df)


def test_groupby_group_keys_true():
    """
    Test groupby(): 'group_keys' cannot have values other than boolean value True
    """

    def impl1(df):
        return df.groupby(by=["A", "C"], group_keys=2)

    def impl2(df):
        return df.groupby(by=["A", "C"], group_keys=False)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="'group_keys' parameter only supports default value True"
    ):
        bodo.jit(impl1)(df)
    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="'group_keys' parameter only supports default value True"
    ):
        bodo.jit(impl2)(df)


def test_groupby_squeeze_false():
    """
    Test groupby(): 'squeeze' cannot have values other than boolean value False
    """

    def impl1(df):
        return df.groupby(by=["A", "C"], squeeze=0)

    def impl2(df):
        return df.groupby(by=["A", "C"], squeeze=True)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="'squeeze' parameter only supports default value False"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="'squeeze' parameter only supports default value False"
    ):
        bodo.jit(impl2)(df)


def test_groupby_observed_false():
    """
    Test groupby(): 'observed' cannot have values other than boolean value False
    """

    def impl1(df):
        return df.groupby(by=["A", "C"], observed=0)

    def impl2(df):
        return df.groupby(by=["A", "C"], observed=True)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="'observed' parameter only supports default value False"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="'observed' parameter only supports default value False"
    ):
        bodo.jit(impl2)(df)


# ------------------------------ Groupby._() ------------------------------ #


def test_groupby_agg_func():
    """
    Test Groupby.agg(): func must be specified
    """

    def impl(df):
        return df.groupby(by=["A"]).agg()

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="Must provide 'func'"):
        bodo.jit(impl)(df)


def test_groupby_agg_multi_funcs():
    """
    Test Groupby.agg(): when more than one functions are supplied, a column must be explictely selected
    """

    def impl(df):
        return df.groupby(by=["A"]).agg((lambda x: len(x), lambda x: len(x)))

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError,
        match="must select exactly one column when more than one functions supplied",
    ):
        bodo.jit(impl)(df)


def test_groupby_agg_func_input_type():
    """
    Test Groupby.agg(): error should be raised when user defined function cannot be applied
    """

    def impl(df):
        return df.groupby(by=["A"]).agg(lambda x: x.max() - x.min())

    df = pd.DataFrame({"A": [1, 2, 2], "B": [1, 2, 2], "C": ["aba", "aba", "aba"]})
    with pytest.raises(
        BodoError,
        match="column C .* unsupported/not a valid input type for user defined function",
    ):
        bodo.jit(impl)(df)


def test_groupby_agg_func_udf():
    """
    Test Groupby.agg(): error should be raised when 'func' is not a user defined function
    """

    def impl(df):
        return df.groupby(by=["A"]).agg(np.sum)

    df = pd.DataFrame({"A": [1, 2, 2], "B": [1, 2, 2], "C": ["aba", "aba", "aba"]})
    with pytest.raises(BodoError, match=".* 'func' must be user defined function"):
        bodo.jit(impl)(df)


def test_groupby_agg_funcs_udf():
    """
    Test Groupby.agg(): error should be raised when 'func' tuple contains non user defined functions
    """

    def impl(df):
        return df.groupby(by=["A"]).agg(np.sum, np.sum)

    df = pd.DataFrame({"A": [1, 2, 2], "B": [1, 2, 2], "C": ["aba", "aba", "aba"]})
    with pytest.raises(BodoError, match=".* 'func' must be user defined function"):
        bodo.jit(impl)(df)


def test_groupby_aggregate_func_required_parameter():
    """
    Test Groupby.aggregate(): func must be specified
    """

    def impl(df):
        return df.groupby(by=["A"]).aggregate()

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="Must provide 'func'"):
        bodo.jit(impl)(df)


def test_groupby_aggregate_multi_funcs():
    """
    Test Groupby.aggregate(): when more than one functions are supplied, a column must be explictely selected
    """

    def impl(df):
        return df.groupby(by=["A"]).aggregate((lambda x: len(x), lambda x: len(x)))

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError,
        match="must select exactly one column when more than one functions supplied",
    ):
        bodo.jit(impl)(df)


def test_groupby_aggregate_func_udf():
    """
    Test Groupby.aggregate(): error should be raised when 'func' is not a user defined function
    """

    def impl(df):
        return df.groupby(by=["A"]).aggregate(np.sum)

    df = pd.DataFrame({"A": [1, 2, 2], "B": [1, 2, 2], "C": ["aba", "aba", "aba"]})
    with pytest.raises(BodoError, match=".* 'func' must be user defined function"):
        bodo.jit(impl)(df)


def test_groupby_aggregate_funcs_udf():
    """
    Test Groupby.aggregate(): error should be raised when 'func' tuple contains non user defined functions
    """

    def impl(df):
        return df.groupby(by=["A"]).aggregate(np.sum, np.sum)

    df = pd.DataFrame({"A": [1, 2, 2], "B": [1, 2, 2], "C": ["aba", "aba", "aba"]})
    with pytest.raises(BodoError, match=".* 'func' must be user defined function"):
        bodo.jit(impl)(df)


def test_groupby_built_in_col_type():
    """
    Test Groupby.prod()
    and mean(), prod(), std(), sum(), var() should have same behaviors
    They all accept only integer, float, and boolean as column dtypes
    """

    def impl(df):
        return df.groupby(by=["A"]).prod()

    df = pd.DataFrame({"A": [1, 2, 2], "B": ["aba", "aba", "aba"]})
    with pytest.raises(
        BodoError,
        match="column type of .* is not supported in groupby built-in functions",
    ):
        bodo.jit(impl)(df)


def test_groupby_built_in_col_bool():
    """
    Test Groupby.max() does not accept Boolean columns
    and max(), min(), prod(), sum() should have same behaviors
    """

    def impl(df):
        return df.groupby(by=["A"]).max()

    df = pd.DataFrame({"A": [1, 2, 2], "B": [True, False, True]})
    with pytest.raises(
        BodoError, match="groupby built-in functions .* does not support boolean column"
    ):
        bodo.jit(impl)(df)


def test_groupby_built_in_col_str():
    """
    Test Groupby.max() does not accept string columns
    count() is supported
    """

    def impl(df):
        return df.groupby(by=["A"]).max()

    df = pd.DataFrame({"A": [1, 2, 2], "B": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError,
        match="column type of .* is not supported in groupby built-in functions",
    ):
        bodo.jit(impl)(df)


def test_groupby_cumsum_col_type():
    """
    Test Groupby.cumsum() only accepts integers and floats
    """

    def impl(df):
        return df.groupby(by=["A"]).cumsum()

    df = pd.DataFrame({"A": [1, 2, 2], "B": [True, False, True]})
    with pytest.raises(
        BodoError, match="only supports columns of types integer and float"
    ):
        bodo.jit(impl)(df)
