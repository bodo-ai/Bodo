# Copyright (C) 2022 Bodo Inc. All rights reserved.

import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError

# ------------------------------ df.groupby() ------------------------------ #


def test_groupby_supply_by(memory_leak_check):
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


def test_groupby_by_labels(memory_leak_check):
    """
    Test groupby(): 'by' is a valid label or label lists
    """

    def impl(df):
        return df.groupby(by=["A", "D"])

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="invalid key .* for 'by'"):
        bodo.jit(impl)(df)


def test_groupby_axis_default(memory_leak_check):
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


def test_groupby_sum_date(memory_leak_check):
    """dates are currently not supported"""

    def impl(df):
        return df.groupby("A").sum()

    df1 = pd.DataFrame({"A": [2, 1, 1], "B": pd.date_range("2019-1-3", "2019-1-5")})
    with pytest.raises(
        BodoError, match="is not supported in groupby built-in function"
    ):
        bodo.jit(impl)(df1)


def test_groupby_supply_level(memory_leak_check):
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


def test_groupby_as_index_bool(memory_leak_check):
    """
    Test groupby(): 'as_index' must be a constant bool
    """

    def impl(df):
        return df.groupby(by=["A", "C"], as_index=2)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="'as_index' parameter must be a constant bool"):
        bodo.jit(impl)(df)


def test_groupby_sort_default(memory_leak_check):
    """
    Test groupby(): 'sort' cannot have values other than boolean value False
    """

    def impl1(df):
        return df.groupby(by=["A", "C"], sort=1)

    def impl2(df):
        return df.groupby(by=["A", "C"], sort=True)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="sort parameter only supports default value False"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="sort parameter only supports default value False"
    ):
        bodo.jit(impl2)(df)


def test_groupby_group_keys_true(memory_leak_check):
    """
    Test groupby(): 'group_keys' cannot have values other than boolean value True
    """

    def impl1(df):
        return df.groupby(by=["A", "C"], group_keys=2)

    def impl2(df):
        return df.groupby(by=["A", "C"], group_keys=False)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="group_keys parameter only supports default value True"
    ):
        bodo.jit(impl1)(df)
    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="group_keys parameter only supports default value True"
    ):
        bodo.jit(impl2)(df)


def test_groupby_squeeze_false(memory_leak_check):
    """
    Test groupby(): 'squeeze' cannot have values other than boolean value False
    """

    def impl1(df):
        return df.groupby(by=["A", "C"], squeeze=1)

    def impl2(df):
        return df.groupby(by=["A", "C"], squeeze=True)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="squeeze parameter only supports default value False"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="squeeze parameter only supports default value False"
    ):
        bodo.jit(impl2)(df)


def test_groupby_observed_true(memory_leak_check):
    """
    Test groupby(): 'observed' cannot have values other than boolean value True
    """

    def impl1(df):
        return df.groupby(by=["A", "C"], observed=0)

    def impl2(df):
        return df.groupby(by=["A", "C"], observed=False)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError, match="observed parameter only supports default value True"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="observed parameter only supports default value True"
    ):
        bodo.jit(impl2)(df)


def test_groupby_dropna(memory_leak_check):
    """
    Test groupby(): 'dropna' cannot have values other than boolean values
    """

    def impl1(df):
        return df.groupby(by=["A", "C"], dropna=2)

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="parameter must be a constant bool"):
        bodo.jit(impl1)(df)


# ------------------------------ Groupby._() ------------------------------ #


def test_groupby_column_selection(memory_leak_check):
    """
    Test Groupby[]: selected column must exist in the Dataframe
    """

    def impl(df):
        return df.groupby(by=["A"])["B"]

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="selected column .* not found in dataframe"):
        bodo.jit(impl)(df)


def test_groupby_column_selection_attr(memory_leak_check):
    """
    Test Groupby.col: selected column must exist in the dataframe
    """

    def impl(df):
        return df.groupby(by=["A"]).B

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="groupby: invalid attribute"):
        bodo.jit(impl)(df)


def test_groupby_columns_selection(memory_leak_check):
    """
    Test Groupby[]: selceted column(s) must exist in the Dataframe
    """

    def impl(df):
        return df.groupby(by=["A"])["B", "C"]

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="selected column .* not found in dataframe"):
        bodo.jit(impl)(df)


def test_groupby_agg_func(memory_leak_check):
    """
    Test Groupby.agg(): func must be specified
    """

    def impl(df):
        return df.groupby(by=["A"]).agg()

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="Must provide 'func'"):
        bodo.jit(impl)(df)


def test_groupby_agg_multi_funcs(memory_leak_check):
    """
    Test Groupby.agg(): when more than one functions are supplied, a column must be explictely selected
    """

    def impl(df):
        return df.groupby(by=["A"]).agg((lambda x: len(x), lambda x: len(x)))

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError,
        match="must select exactly one column when more than one function is supplied",
    ):
        bodo.jit(impl)(df)


def test_groupby_agg_func_input_type(memory_leak_check):
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


def test_groupby_agg_func_udf(memory_leak_check):
    """
    Test Groupby.agg(): error should be raised when 'func' is not a user defined function
    """

    def impl(df):
        return df.groupby(by=["A"]).agg(np.sum)

    df = pd.DataFrame({"A": [1, 2, 2], "B": [1, 2, 2], "C": ["aba", "aba", "aba"]})
    with pytest.raises(BodoError, match=".* 'func' must be user defined function"):
        bodo.jit(impl)(df)


def test_groupby_agg_funcs_udf(memory_leak_check):
    """
    Test Groupby.agg(): error should be raised when 'func' tuple contains non user defined functions
    """

    def impl(df):
        return df.groupby(by=["A"]).B.agg((np.sum, np.sum))

    df = pd.DataFrame({"A": [1, 2, 2], "B": [1, 2, 2], "C": ["aba", "aba", "aba"]})
    with pytest.raises(BodoError, match=".* 'func' must be user defined function"):
        bodo.jit(impl)(df)


def test_groupby_aggregate_func_required_parameter(memory_leak_check):
    """
    Test Groupby.aggregate(): func must be specified
    """

    def impl(df):
        return df.groupby(by=["A"]).aggregate()

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="Must provide 'func'"):
        bodo.jit(impl)(df)


def test_groupby_aggregate_multi_funcs(memory_leak_check):
    """
    Test Groupby.aggregate(): when more than one functions are supplied, a column must be explictely selected
    """

    def impl(df):
        return df.groupby(by=["A"]).aggregate((lambda x: len(x), lambda x: len(x)))

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(
        BodoError,
        match="must select exactly one column when more than one function is supplied",
    ):
        bodo.jit(impl)(df)


def test_groupby_aggregate_func_udf(memory_leak_check):
    """
    Test Groupby.aggregate(): error should be raised when 'func' is not a user defined function
    """

    def impl(df):
        return df.groupby(by=["A"]).aggregate(np.sum)

    df = pd.DataFrame({"A": [1, 2, 2], "B": [1, 2, 2], "C": ["aba", "aba", "aba"]})
    with pytest.raises(BodoError, match=".* 'func' must be user defined function"):
        bodo.jit(impl)(df)


def test_groupby_aggregate_funcs_udf(memory_leak_check):
    """
    Test Groupby.aggregate(): error should be raised when 'func' tuple contains non user defined functions
    """

    def impl(df):
        return df.groupby(by=["A"]).B.aggregate((np.sum, np.sum))

    df = pd.DataFrame({"A": [1, 2, 2], "B": [1, 2, 2], "C": ["aba", "aba", "aba"]})
    with pytest.raises(BodoError, match=".* 'func' must be user defined function"):
        bodo.jit(impl)(df)


def test_groupby_apply_udf_non_numba_err(memory_leak_check):
    """
    Test Groupby.apply() with a UDF that raises a non-Numba error (i.e. no msg and loc
    attributes)
    """

    @bodo.jit
    def apply_func(df):
        with bodo.objmode(
            out='bodo.DataFrameType((bodo.float64[::1],), bodo.RangeIndexType(bodo.none), ("0",))'
        ):
            out = pd.Series([9, 8, 7, 6, 5])
        return out

    @bodo.jit(distributed=["df", "res"])
    def main_func(df):
        res = df.groupby("A").apply(apply_func)
        return res

    df = pd.DataFrame({"A": [1.0, 2, 3, 1.0, 5], "B": [4.0, 5, 6, 2, 1]})
    with pytest.raises(
        BodoError,
        match=r"The value must be a compile-time constant",
    ):
        main_func(df)


def test_groupby_built_in_col_type(memory_leak_check):
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
        match="column type of strings or list of strings is not supported in groupby built-in function prod",
    ):
        bodo.jit(impl)(df)


def test_groupby_cumsum_col_type(memory_leak_check):
    """
    Test Groupby.cumsum() only accepts integers and floats
    """

    def impl(df):
        return df.groupby(by=["A"]).cumsum()

    df = pd.DataFrame({"A": [1, 2, 2], "B": [True, False, True]})
    with pytest.raises(
        BodoError,
        match="Groupby.cumsum.* only supports columns of types integer, float, string or liststring",
    ):
        bodo.jit(impl)(df)


def test_groupby_median_type_check(memory_leak_check):
    """
    Test Groupby.median() testing the input type argument
    """

    def impl(df):
        return df.groupby("A")["B"].median()

    df1 = pd.DataFrame({"A": [1, 1, 1, 1], "B": ["a", "b", "c", "d"]})
    df2 = pd.DataFrame({"A": [1, 1, 1, 1], "B": [True, False, True, False]})
    with pytest.raises(
        BodoError,
        match="For median, only column of integer, float or Decimal type are allowed",
    ):
        bodo.jit(impl)(df1)
    with pytest.raises(
        BodoError,
        match="For median, only column of integer, float or Decimal type are allowed",
    ):
        bodo.jit(impl)(df2)


def test_groupby_cumsum_argument_check(memory_leak_check):
    """
    Test Groupby.cumsum() testing for skipna argument
    """

    def impl1(df):
        return df.groupby("A")["B"].cumsum(skipna=0)

    def impl2(df):
        return df.groupby("A")["B"].cumsum(wrongarg=True)

    df = pd.DataFrame({"A": [1, 1, 1, 1], "B": [1, 2, 3, 4]})
    with pytest.raises(
        BodoError, match="For cumsum argument of skipna should be a boolean"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="got an unexpected keyword argument 'wrongarg'"
    ):
        bodo.jit(impl2)(df)


def test_groupby_cumsum_argument_duplication_check(memory_leak_check):
    """
    Test Groupby.cumsum() testing for skipna argument
    """

    def impl(df):
        return df.groupby("A")["B"].agg(("cumsum", "cumsum"))

    df = pd.DataFrame({"A": [1, 1, 1, 1], "B": [1, 2, 3, 4]})
    with pytest.raises(
        BodoError, match="aggregate with duplication in output is not allowed"
    ):
        bodo.jit(impl)(df)


def test_groupby_cumprod_argument_check(memory_leak_check):
    """
    Test Groupby.cumprod() testing for skipna argument
    """

    def impl1(df):
        return df.groupby("A")["B"].cumprod(skipna=0)

    def impl2(df):
        return df.groupby("A")["B"].cumprod(wrongarg=True)

    df = pd.DataFrame({"A": [1, 1, 1, 1], "B": [1, 2, 3, 4]})
    with pytest.raises(
        BodoError, match="For cumprod argument of skipna should be a boolean"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="got an unexpected keyword argument 'wrongarg'"
    ):
        bodo.jit(impl2)(df)


def test_groupby_nunique_argument_check(memory_leak_check):
    """
    Test Groupby.nunique() testing for dropna argument
    """

    def impl1(df):
        return df.groupby("A")["B"].nunique(dropna=0)

    def impl2(df):
        return df.groupby("A")["B"].nunique(wrongarg=True)

    df = pd.DataFrame({"A": [1, 1, 1, 1], "B": [1, 2, 3, 4]})
    with pytest.raises(
        BodoError, match="argument of dropna to nunique should be a boolean"
    ):
        bodo.jit(impl1)(df)
    with pytest.raises(
        BodoError, match="got an unexpected keyword argument 'wrongarg'"
    ):
        bodo.jit(impl2)(df)


def test_groupby_datetimeoperation_checks(memory_leak_check):
    """
    Testing the operations which cannot be done for date / datetime / timedelta
    """

    def impl_sum(df):
        return df.groupby("A")["B"].sum()

    def impl_prod(df):
        return df.groupby("A")["B"].prod()

    def impl_cumsum(df):
        return df.groupby("A")["B"].cumsum()

    def impl_cumprod(df):
        return df.groupby("A")["B"].cumprod()

    siz = 10
    datetime_arr_1 = pd.date_range("1917-01-01", periods=siz)
    datetime_arr_2 = pd.date_range("2017-01-01", periods=siz)
    timedelta_arr = datetime_arr_1 - datetime_arr_2
    date_arr = datetime_arr_1.date
    df1_datetime = pd.DataFrame({"A": np.arange(siz), "B": datetime_arr_1})
    df1_date = pd.DataFrame({"A": np.arange(siz), "B": date_arr})
    df1_timedelta = pd.DataFrame({"A": np.arange(siz), "B": timedelta_arr})
    # Check for sums
    with pytest.raises(
        BodoError,
        match="column type of datetime64.* is not supported in groupby built-in function sum",
    ):
        bodo.jit(impl_sum)(df1_datetime)
    with pytest.raises(
        BodoError,
        match="column type of DatetimeDateType.* is not supported in groupby built-in function sum",
    ):
        bodo.jit(impl_sum)(df1_date)
    with pytest.raises(
        BodoError,
        match="column type of timedelta64.* is not supported in groupby built-in function sum",
    ):
        bodo.jit(impl_sum)(df1_timedelta)
    # checks for prod
    with pytest.raises(
        BodoError,
        match="column type of datetime64.* is not supported in groupby built-in function prod",
    ):
        bodo.jit(impl_prod)(df1_datetime)
    with pytest.raises(
        BodoError,
        match="column type of DatetimeDateType.* is not supported in groupby built-in function prod",
    ):
        bodo.jit(impl_prod)(df1_date)
    with pytest.raises(
        BodoError,
        match="column type of timedelta64.* is not supported in groupby built-in function prod",
    ):
        bodo.jit(impl_prod)(df1_timedelta)
    # checks for cumsum
    with pytest.raises(
        BodoError,
        match="Groupby.cumsum.* only supports columns of types integer, float, string or liststring",
    ):
        bodo.jit(impl_cumsum)(df1_datetime)
    with pytest.raises(
        BodoError,
        match="Groupby.cumsum.* only supports columns of types integer, float, string or liststring",
    ):
        bodo.jit(impl_cumsum)(df1_date)
    with pytest.raises(
        BodoError,
        match="Groupby.cumsum.* only supports columns of types integer, float, string or liststring",
    ):
        bodo.jit(impl_cumsum)(df1_timedelta)
    # checks for cumprod
    with pytest.raises(
        BodoError,
        match="Groupby.cumprod.* only supports columns of types integer and float",
    ):
        bodo.jit(impl_cumprod)(df1_datetime)
    with pytest.raises(
        BodoError,
        match="Groupby.cumprod.* only supports columns of types integer and float",
    ):
        bodo.jit(impl_cumprod)(df1_date)
    with pytest.raises(
        BodoError,
        match="Groupby.cumprod.* only supports columns of types integer and float",
    ):
        bodo.jit(impl_cumprod)(df1_timedelta)


@pytest.mark.slow
def test_groupby_unsupported_error_checking(memory_leak_check):
    """Test that a Bodo error is raised for unsupported
    groupby methods
    """

    def test_method(df):
        return df.groupby("a").sample(n=1, random_state=1)

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_method)(
            df=pd.DataFrame(
                {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}
            )
        )


# ------------------------------ df.groupby().func()------------------------------ #


@pytest.mark.slow
def test_first_args(memory_leak_check):
    """Test Groupby.first with arguments"""

    def impl_numeric_only(df):
        A = df.groupby("A").first(numeric_only=True)
        return A

    def impl_min_count(df):
        A = df.groupby("A").first(min_count=10)
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_min_count)(df_bool)


@pytest.mark.slow
def test_last_args(memory_leak_check):
    """Test Groupby.last with arguments"""

    def impl_numeric_only(df):
        A = df.groupby("A").last(numeric_only=True)
        return A

    def impl_min_count(df):
        A = df.groupby("A").last(min_count=10)
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_min_count)(df_bool)


@pytest.mark.slow
def test_first_last_unsupported_types(memory_leak_check):
    """Test first/last with unsupported Bodo types"""

    def impl1(df):
        A = df.groupby("A").first()
        return A

    def impl2(df):
        A = df.groupby("A").last()
        return A

    # [BE-416] Support with list
    df_list_int = pd.DataFrame(
        {"A": [2, 1, 1, 2], "B": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]])}
    )
    df_list_float = pd.DataFrame(
        {
            "A": [2, 1, 1, 2],
            "B": pd.Series([[1.1, 2.3], [3.3], [5.4, 4.54, 6.5], [-1.3, 3.0, 4.4]]),
        }
    )
    df_list_bool = pd.DataFrame(
        {
            "A": [2, 1, 1, 2],
            "B": pd.Series(
                [[True, False], [True], [True, False, True], [False, False, False]]
            ),
        }
    )
    err_msg = "not supported in groupby"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df_list_int)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df_list_int)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df_list_float)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df_list_float)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df_list_bool)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df_list_bool)


# ------------------------------ df.groupby().mean()/median()------------------------------ #


@pytest.mark.slow
def test_mean_args(memory_leak_check):
    """Test Groupby.mean with wrong arguments values"""

    # wrong keyword value test
    def impl_numeric_only(df):
        A = df.groupby("A").mean(numeric_only=False)
        return A

    # wrong arg value test
    def impl_numeric_only_arg(df):
        A = df.groupby("A").mean(False)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16],
            "B": [3, 5, 6, 5, 4, 4],
            "C": [3.4, 2.5, 9.6, 1.5, -4.3, 4.3],
        }
    )
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only_arg)(df)


@pytest.mark.slow
def test_median_args(memory_leak_check):
    """Test Groupby.median with wrong arguments values"""

    # wrong keyword value test
    def impl_numeric_only(df):
        A = df.groupby("A").median(numeric_only=False)
        return A

    # wrong arg value test
    def impl_numeric_only_arg(df):
        A = df.groupby("A").median(False)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16],
            "B": [3, 5, 6, 5, 4, 4],
            "C": [3.4, 2.5, 9.6, 1.5, -4.3, 4.3],
        }
    )
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only_arg)(df)


# ------------------------------ df.groupby().sum()/prod()------------------------------ #


@pytest.mark.slow
def test_sum_args(memory_leak_check):
    """Test Groupby.sum with arguments"""

    # wrong keyword value test
    def impl_numeric_only(df):
        A = df.groupby("A").sum(numeric_only=False)
        return A

    def impl_min_count(df):
        A = df.groupby("A").sum(min_count=10)
        return A

    # wrong args value test
    def impl_numeric_only_args(df):
        A = df.groupby("A").sum(False)
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_min_count)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only_args)(df_bool)


@pytest.mark.slow
def test_prod_args(memory_leak_check):
    """Test Groupby.prod with arguments"""

    # wrong keyword value
    def impl_numeric_only(df):
        A = df.groupby("A").prod(numeric_only=False)
        return A

    def impl_min_count(df):
        A = df.groupby("A").prod(min_count=10)
        return A

    # wrong args value test
    def impl_numeric_only_args(df):
        A = df.groupby("A").prod(False)
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )

    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_min_count)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only_args)(df_bool)


@pytest.mark.slow
@pytest.mark.parametrize(
    "df",
    [
        # None changes timedelta64[ns] to DatetimeTimeDeltaType() which we don't support
        pd.DataFrame(
            {
                "A": [1, 2, 3, 2, 1],
                "B": pd.concat(
                    [
                        pd.Series(pd.timedelta_range(start="1 day", periods=4)),
                        pd.Series(data=[None], index=[4]),
                    ]
                ),
            }
        ),
        # timedelta
        pd.DataFrame(
            {
                "A": [1, 2, 3, 2, 1],
                "B": pd.Series(pd.timedelta_range(start="1 day", periods=5)),
            }
        ),
        # [BE-416] Support with list
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]])}
        ),
        # Categorical
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16, 16],
                "B": pd.Categorical([1, 2, 5, 5, 3, 3], ordered=True),
            }
        ),
        # Timestamp
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16],
                "B": [
                    pd.Timestamp("20130101 09:00:00"),
                    pd.Timestamp("20130101 09:00:02"),
                    pd.Timestamp("20130101 09:00:03"),
                    pd.Timestamp("20130101 09:00:05"),
                    pd.Timestamp("20130101 09:00:06"),
                ],
            }
        ),
        # boolean Array
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16, 16, 1, 40],
                "B": [True, np.nan, False, True, np.nan, False, False, True],
                "C": [True, True, False, True, True, False, False, True],
            }
        ),
        # datetime
        pd.DataFrame(
            {"A": [2, 1, 1, 1, 2, 2, 1], "B": pd.date_range("2019-1-3", "2019-1-9")}
        ),
        # string
        pd.DataFrame(
            {
                "A": [3, 3, 5, 5, 4, 4, 3],
                "B": ["ccc", "a", "a", "aa", "ccc", "ggg", "a"],
            }
        ),
    ],
)
def test_mean_median_unsupported_types(df, memory_leak_check):
    """Test mean/median with their unsupported Bodo types"""

    def impl1(df):
        A = df.groupby("A").mean()
        return A

    def impl2(df):
        A = df.groupby("A").median()
        return A

    err_msg = "not supported in groupby"
    if isinstance(df["B"][0], bool):
        err_msg = "does not support boolean column"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df)

    # median has specific message but not for DatetimeTimeDeltaType
    err_msg_median = (
        "For median, only column of integer, float or Decimal type are allowed"
    )
    with pytest.raises(BodoError) as excinfo:
        bodo.jit(impl2)(df)
    assert err_msg in str(excinfo.value) or err_msg_median in str(excinfo.value)


@pytest.mark.slow
@pytest.mark.parametrize(
    "df",
    [
        # None changes timedelta64[ns] to DatetimeTimeDeltaType() which we don't support
        pd.DataFrame(
            {
                "A": [1, 2, 3, 2, 1],
                "B": pd.concat(
                    [
                        pd.Series(pd.timedelta_range(start="1 day", periods=4)),
                        pd.Series(data=[None], index=[4]),
                    ]
                ),
            }
        ),
        # Decimal
        pd.DataFrame(
            {
                "A": [2, 1, 1, 2, 2],
                "B": pd.Series(
                    [
                        Decimal("1.6"),
                        Decimal("-0.2"),
                        Decimal("44.2"),
                        np.nan,
                        Decimal("0"),
                    ]
                ),
            }
        ),
        # timedelta
        pd.DataFrame(
            {
                "A": [1, 2, 3, 2, 1],
                "B": pd.Series(pd.timedelta_range(start="1 day", periods=5)),
            }
        ),
        # [BE-416] Support with list
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]])}
        ),
        # Categorical
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16, 16],
                "B": pd.Categorical([1, 2, 5, 5, 3, 3], ordered=True),
            }
        ),
        # Timestamp
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16],
                "B": [
                    pd.Timestamp("20130101 09:00:00"),
                    pd.Timestamp("20130101 09:00:02"),
                    pd.Timestamp("20130101 09:00:03"),
                    pd.Timestamp("20130101 09:00:05"),
                    pd.Timestamp("20130101 09:00:06"),
                ],
            }
        ),
    ],
)
def test_sum_prod_unsupported_types(df, memory_leak_check):
    """Test sum/prod with their unsupported Bodo types"""

    def impl1(df):
        A = df.groupby("A").sum()
        return A

    def impl2(df):
        A = df.groupby("A").prod()
        return A

    err_msg = "not supported in groupby"

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df)

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df)


# ------------------------------ df.groupby().min()/max()------------------------------ #


@pytest.mark.slow
def test_min_args(memory_leak_check):
    """Test Groupby.min with arguments"""

    # wrong keyword value test
    def impl_numeric_only(df):
        A = df.groupby("A").min(numeric_only=True)
        return A

    def impl_min_count(df):
        A = df.groupby("A").min(min_count=10)
        return A

    # wrong arg value test
    def impl_numeric_only_arg(df):
        A = df.groupby("A").min(True)
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )

    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_min_count)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only_arg)(df_bool)


@pytest.mark.slow
def test_max_args(memory_leak_check):
    """Test Groupby.max with arguments"""

    # wrong keyword value test
    def impl_numeric_only(df):
        A = df.groupby("A").max(numeric_only=True)
        return A

    def impl_min_count(df):
        A = df.groupby("A").max(min_count=10)
        return A

    # wrong arg value test
    def impl_numeric_only_arg(df):
        A = df.groupby("A").max(True)
        return A

    df_bool = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, np.nan, False, True, np.nan, False, False, True],
            "C": [True, True, False, True, True, False, False, False],
        }
    )
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_min_count)(df_bool)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_numeric_only_arg)(df_bool)


@pytest.mark.slow
def test_min_max_unsupported_types(memory_leak_check):
    """Test min/max with unsupported Bodo types"""

    def impl1(df):
        A = df.groupby("A").min()
        return A

    def impl2(df):
        A = df.groupby("A").max()
        return A

    # [BE-416] Support with list
    df_list_int = pd.DataFrame(
        {"A": [2, 1, 1, 2], "B": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]])}
    )
    err_msg = "not supported in groupby"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df_list_int)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df_list_int)

    # [BE-416] Support with tuples
    df_tuples = pd.DataFrame(
        {"A": [2, 1, 1, 2], "B": pd.Series([(1, 2), (3,), (5, 4, 6), (-1, 3, 4)])}
    )
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df_tuples)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df_tuples)


# ------------------------------ df.groupby().var()/std()------------------------------ #


@pytest.mark.slow
def test_var_args(memory_leak_check):
    """Test Groupby.var with arguments"""

    # wrong keyword value test
    def impl_ddof(df):
        A = df.groupby("A").var(ddof=10)
        return A

    # wrong arg value test
    def impl_ddof_arg(df):
        A = df.groupby("A").var(10)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [1.1, 2.2, 1.1, 3.3, 4.4, 3.2, 45.6, 10.0],
        }
    )

    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_ddof)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_ddof_arg)(df)


@pytest.mark.slow
def test_std_args(memory_leak_check):
    """Test Groupby.std with arguments"""

    # wrong keyword value test
    def impl_ddof(df):
        A = df.groupby("A").std(ddof=10)
        return A

    # wrong arg value test
    def impl_ddof_arg(df):
        A = df.groupby("A").std(10)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [1.1, 2.2, 1.1, 3.3, 4.4, 3.2, 45.6, 10.0],
        }
    )

    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_ddof)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_ddof_arg)(df)


# ------------------------------ df.groupby().idxmin()/idxmax()------------------------------ #


@pytest.mark.slow
def test_idxmin_args(memory_leak_check):
    """Test Groupby.idxmin with arguments"""

    # wrong keyword value test
    def impl_axis(df):
        A = df.groupby("A").idxmin(axis=1)
        return A

    # wrong arg value test
    def impl_axis_arg(df):
        A = df.groupby("A").idxmin(1)
        return A

    def impl_skipna(df):
        A = df.groupby("A").idxmin(skipna=False)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [1.1, 2.2, 1.1, 3.3, 4.4, 3.2, 45.6, 10.0],
        }
    )

    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_axis_arg)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_axis)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_skipna)(df)


@pytest.mark.slow
def test_idxmax_args(memory_leak_check):
    """Test Groupby.idxmax with arguments"""

    # wrong keyword value test
    def impl_axis(df):
        A = df.groupby("A").idxmax(axis=1)
        return A

    # wrong arg value test
    def impl_axis_arg(df):
        A = df.groupby("A").idxmax(1)
        return A

    def impl_skipna(df):
        A = df.groupby("A").idxmax(skipna=False)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [1.1, 2.2, 1.1, 3.3, 4.4, 3.2, 45.6, 10.0],
        }
    )

    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_axis_arg)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_axis)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_skipna)(df)


@pytest.mark.slow
@pytest.mark.parametrize(
    "df",
    [
        # datetime
        pd.DataFrame(
            {
                "A": [2, 1, 1, 2, 3],
                "B": pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
            }
        ),
        # None changes timedelta64[ns] to DatetimeTimeDeltaType() which we don't support
        pd.DataFrame(
            {
                "A": [1, 2, 3, 2, 1],
                "B": pd.concat(
                    [
                        pd.Series(pd.timedelta_range(start="1 day", periods=4)),
                        pd.Series(data=[None], index=[4]),
                    ]
                ),
            }
        ),
        # timedelta
        pd.DataFrame(
            {
                "A": [1, 2, 3, 2, 1],
                "B": pd.Series(pd.timedelta_range(start="1 day", periods=5)),
            }
        ),
        # [BE-416] Support with list
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]])}
        ),
        # Tuple
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([(1, 2), (3), (5, 4, 6), (-1, 3, 4)])}
        ),
        # Categorical
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16, 16],
                "B": pd.Categorical([1, 2, 5, 5, 3, 3], ordered=True),
            }
        ),
        # Timestamp
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16],
                "B": [
                    pd.Timestamp("20130101 09:00:00"),
                    pd.Timestamp("20130101 09:00:02"),
                    pd.Timestamp("20130101 09:00:03"),
                    pd.Timestamp("20130101 09:00:05"),
                    pd.Timestamp("20130101 09:00:06"),
                ],
            }
        ),
        # boolean Array
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16, 16, 1, 40],
                "B": [True, np.nan, False, True, np.nan, False, False, True],
                "C": [True, True, False, True, True, False, False, True],
            }
        ),
        # string
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16, 16, 1, 40],
                "B": ["ab", "cd", "ef", "gh", "mm", "a", "abc", "x"],
            }
        ),
    ],
)
def test_var_std_unsupported_types(df, memory_leak_check):
    """Test var/std with their unsupported Bodo types"""

    def impl1(df):
        A = df.groupby("A").var()
        return A

    def impl2(df):
        A = df.groupby("A").std()
        return A

    err_msg = "not supported in groupby"
    if isinstance(df["B"][0], bool):
        err_msg = "does not support boolean column"

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df)

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df)


@pytest.mark.slow
@pytest.mark.parametrize(
    "df",
    [
        # Decimal
        pd.DataFrame(
            {
                "A": [2, 1, 1, 2, 2],
                "B": pd.Series(
                    [
                        Decimal("1.6"),
                        Decimal("-0.2"),
                        Decimal("44.2"),
                        np.nan,
                        Decimal("0"),
                    ]
                ),
            }
        ),
        # datetime
        pd.DataFrame(
            {
                "A": [2, 1, 1, 2, 3],
                "B": pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
            }
        ),
        # None changes timedelta64[ns] to DatetimeTimeDeltaType() which we don't support
        pd.DataFrame(
            {
                "A": [1, 2, 3, 2, 1],
                "B": pd.concat(
                    [
                        pd.Series(pd.timedelta_range(start="1 day", periods=4)),
                        pd.Series(data=[None], index=[4]),
                    ]
                ),
            }
        ),
        # timedelta
        pd.DataFrame(
            {
                "A": [1, 2, 3, 2, 1],
                "B": pd.Series(pd.timedelta_range(start="1 day", periods=5)),
            }
        ),
        # [BE-416] Support with list
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]])}
        ),
        # Tuple
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([(1, 2), (3), (5, 4, 6), (-1, 3, 4)])}
        ),
        # Categorical
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16, 16],
                "B": pd.Categorical([1, 2, 5, 5, 3, 3], ordered=True),
            }
        ),
        # Timestamp
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16],
                "B": [
                    pd.Timestamp("20130101 09:00:00"),
                    pd.Timestamp("20130101 09:00:02"),
                    pd.Timestamp("20130101 09:00:03"),
                    pd.Timestamp("20130101 09:00:05"),
                    pd.Timestamp("20130101 09:00:06"),
                ],
            }
        ),
        # string
        pd.DataFrame(
            {
                "A": [16, 1, 1, 1, 16, 16, 1, 40],
                "B": ["ab", "cd", "ef", "gh", "mm", "a", "abc", "x"],
            }
        ),
        # Zero columns
        pd.DataFrame({"A": [2, 1, 1, 1, 2, 2, 1]}),
    ],
)
def test_idxmin_idxmax_unsupported_types(df, memory_leak_check):
    """Test idxmin/idxmax with their unsupported Bodo types"""

    def impl1(df):
        A = df.groupby("A").idxmin()
        return A

    def impl2(df):
        A = df.groupby("A").idxmax()
        return A

    if len(df.columns) == 1:
        with pytest.raises(BodoError, match="No columns in output"):
            bodo.jit(impl1)(df)
        with pytest.raises(BodoError, match="No columns in output"):
            bodo.jit(impl2)(df)
        return
    err_msg = "not supported in groupby"
    if isinstance(df["B"][0], np.bool_):
        err_msg = "does not support boolean column"

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df)

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df)


# ------------------------------ df.groupby().cumulatives()------------------------------ #
@pytest.mark.slow
def test_cumultative_args(memory_leak_check):
    """Test Groupby.cumultatives (sum, prod, min, max) with arguments"""

    # keyword axis=1 cummin
    def impl_cummin_axis_kwargs(df):
        A = df.groupby("A").cummin(axis=1)
        return A

    # keyword axis=1 cummax
    def impl_cummax_axis_kwargs(df):
        A = df.groupby("A").cummax(axis=1)
        return A

    # keyword axis=1 cumsum
    def impl_cumsum_axis_kwargs(df):
        A = df.groupby("A").cumsum(axis=1)
        return A

    # keyword axis=1 cumprod
    def impl_cumprod_axis_kwargs(df):
        A = df.groupby("A").cumprod(axis=1)
        return A

    # args axis=1 cummin
    def impl_cummin_axis_args(df):
        A = df.groupby("A").cummin(1)
        return A

    # args axis=1 cummax
    def impl_cummax_axis_args(df):
        A = df.groupby("A").cummax(1)
        return A

    # args axis=1 cumsum
    def impl_cumsum_axis_args(df):
        A = df.groupby("A").cumsum(1)
        return A

    # args axis=1 cumprod
    def impl_cumprod_axis_args(df):
        A = df.groupby("A").cumprod(1)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [1.1, 2.2, 1.1, 3.3, 4.4, 3.2, 45.6, 10.0],
        }
    )
    err_msg = "axis parameter only supports default value 0"

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_cummin_axis_kwargs)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_cummax_axis_kwargs)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_cumsum_axis_kwargs)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_cumprod_axis_kwargs)(df)

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_cummin_axis_args)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_cummax_axis_args)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_cumsum_axis_args)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_cumprod_axis_args)(df)


@pytest.fixture(
    params=[
        # Decimal
        pytest.param(
            pd.DataFrame(
                {
                    "A": [2, 1, 1, 2, 2],
                    "B": pd.Series(
                        [
                            Decimal("1.6"),
                            Decimal("-0.2"),
                            Decimal("44.2"),
                            np.nan,
                            Decimal("0"),
                        ]
                    ),
                }
            ),
            marks=pytest.mark.slow,
        ),
        # datetime
        pytest.param(
            pd.DataFrame(
                {
                    "A": [2, 1, 1, 2, 3],
                    "B": pd.date_range(start="2018-04-24", end="2018-04-29", periods=5),
                }
            ),
            marks=pytest.mark.slow,
        ),
        # timedelta
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 3, 2, 1],
                    "B": pd.Series(pd.timedelta_range(start="1 day", periods=5)),
                }
            ),
            marks=pytest.mark.slow,
        ),
        # [BE-416] Support with list
        pytest.param(
            pd.DataFrame(
                {
                    "A": [2, 1, 1, 2],
                    "B": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]]),
                }
            ),
            marks=pytest.mark.slow,
        ),
        # Tuple
        pytest.param(
            pd.DataFrame(
                {
                    "A": [2, 1, 1, 2],
                    "B": pd.Series([(1, 2), (3), (5, 4, 6), (-1, 3, 4)]),
                }
            ),
            marks=pytest.mark.slow,
        ),
        # Categorical
        pytest.param(
            pd.DataFrame(
                {
                    "A": [16, 1, 1, 1, 16, 16],
                    "B": pd.Categorical([1, 2, 5, 5, 3, 3], ordered=True),
                }
            ),
            marks=pytest.mark.slow,
        ),
        # Timestamp
        pytest.param(
            pd.DataFrame(
                {
                    "A": [16, 1, 1, 1, 16],
                    "B": [
                        pd.Timestamp("20130101 09:00:00"),
                        pd.Timestamp("20130101 09:00:02"),
                        pd.Timestamp("20130101 09:00:03"),
                        pd.Timestamp("20130101 09:00:05"),
                        pd.Timestamp("20130101 09:00:06"),
                    ],
                }
            ),
            marks=pytest.mark.slow,
        ),
        # string
        pytest.param(
            pd.DataFrame(
                {
                    "A": [16, 1, 1, 1, 16, 16, 1, 40],
                    "B": ["ab", "cd", "ef", "gh", "mm", "a", "abc", "x"],
                }
            ),
            marks=pytest.mark.slow,
        ),
        # boolean
        pytest.param(
            pd.DataFrame(
                {
                    "A": [2, 1, 1, 1, 2, 2, 1],
                    "B": [True, True, False, True, True, False, False],
                }
            ),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_cumulatives_df(request):
    return request.param


def test_cumulatives_unsupported_types(test_cumulatives_df, memory_leak_check):
    """Test cummin, cummax, cumprod, and cumsum with their unsupported Bodo types"""

    def impl1(df):
        A = df.groupby("A").cummin()
        return A

    def impl2(df):
        A = df.groupby("A").cummax()
        return A

    def impl3(df):
        A = df.groupby("A").cumsum()
        return A

    def impl4(df):
        A = df.groupby("A").cumprod()
        return A

    err_msg = "only supports columns of types"

    # cummin/cummax supports datetime and timedelta
    if not isinstance(
        test_cumulatives_df["B"].values[0],
        (
            np.timedelta64,
            np.datetime64,
        ),
    ):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(impl1)(test_cumulatives_df)

        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(impl2)(test_cumulatives_df)

    # cumsum supports string and list of strings
    if not isinstance(test_cumulatives_df["B"][0], str):
        with pytest.raises(BodoError, match=err_msg):
            bodo.jit(impl3)(test_cumulatives_df)

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl4)(test_cumulatives_df)


# ------------------------------ df.groupby().count()/size()------------------------------ #
@pytest.mark.slow
def test_count_size_args(memory_leak_check):
    """Test Groupby.size/count with arguments"""

    # keyword axis=1
    def impl_size_kwargs(df):
        A = df.groupby("A").size(axis=1)
        return A

    # args
    def impl_size_args(df):
        A = df.groupby("A").size(1)
        return A

    # keyword axis=1
    def impl_count_kwargs(df):
        A = df.groupby("A").count(axis=1)
        return A

    # args
    def impl_count_args(df):
        A = df.groupby("A").count(1)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [1.1, 2.2, 1.1, 3.3, 4.4, 3.2, 45.6, 10.0],
        }
    )
    err_msg = "got an unexpected keyword argument."

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_size_kwargs)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_count_kwargs)(df)
    err_msg = "takes 1 positional argument but "
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_size_args)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_count_args)(df)


@pytest.mark.slow
@pytest.mark.parametrize(
    "df",
    [
        # [BE-416] Support with list
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]])}
        ),
        # Tuple
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([(1, 2), (3), (5, 4, 6), (-1, 3, 4)])}
        ),
    ],
)
def test_count_unsupported_types(df, memory_leak_check):
    """Test count with their unsupported Bodo types"""

    def impl1(df):
        A = df.groupby("A").count()
        return A

    err_msg = "not supported in groupby"

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df)


# ------------------------------ df.groupby().shift()/nunique()------------------------------ #


@pytest.mark.slow
def test_shift_args(memory_leak_check):
    """Test Groupby.shift with wrong arguments values"""

    # wrong freq keyword value test
    def impl_freq(df):
        A = df.groupby("A").shift(freq="D")
        return A

    # wrong freq arg value test
    def impl_freq_arg(df):
        A = df.groupby("A").shift(-1, "D")
        return A

    # wrong keyword value test
    def impl_axis(df):
        A = df.groupby("A").shift(axis=1)
        return A

    # wrong keyword value test
    def impl_fill_value(df):
        A = df.groupby("A").shift(fill_value=1)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16],
            "B": [3, 5, 6, 5, 4, 4],
        }
    )
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_freq)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_freq_arg)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_axis)(df)
    with pytest.raises(BodoError, match="parameter only supports default value"):
        bodo.jit(impl_fill_value)(df)


@pytest.mark.parametrize(
    "df",
    [
        # [BE-416] Support with list
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([[1, 2], [3], [5, 4, 6], [-1, 3, 4]])}
        ),
        # Tuple
        pd.DataFrame(
            {"A": [2, 1, 1, 2], "B": pd.Series([(1, 2), (3), (5, 4, 6), (-1, 3, 4)])}
        ),
        # Zero columns
        pd.DataFrame({"A": [2, 1, 1, 1, 2, 2, 1]}),
    ],
)
@pytest.mark.slow
def test_nunique_shift_unsupported_types(df, memory_leak_check):
    """Test nunique/shift with their unsupported Bodo types"""

    def impl_nunique(df):
        A = df.groupby("A").nunique()
        return A

    def impl_shift(df):
        A = df.groupby("A").shift()
        return A

    err_msg = "not supported in groupby"
    if len(df.columns) == 1:
        err_msg = "No columns in output"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_nunique)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_shift)(df)


@pytest.mark.slow
def test_shift_multi_index_error(memory_leak_check):
    """Make sure proper error is raised for MultiIndex input"""

    def impl(df):
        return df.groupby("A").shift()

    I = pd.MultiIndex.from_arrays([[5, 3, 1], ["a", "b", "c"]], names=("AI", "BI"))
    df = pd.DataFrame({"A": [1, 2, 1], "B": [3.1, 4.2, 2.2], "C": [3, 1, 5]}, index=I)
    with pytest.raises(
        BodoError,
        match="MultiIndex input not supported for groupby operations that use input Index",
    ):
        bodo.jit(impl)(df)


# ------------------------------ df.groupby().agg()------------------------------ #
@pytest.mark.slow
def test_agg_unsupported_types(test_cumulatives_df, memory_leak_check):
    """Test groupby.agg with unsupported types"""

    def impl1(df):
        A = df.groupby("A").agg(lambda x: x.sum())
        return A

    err_msg = "is unsupported/not a valid input type"

    if isinstance(test_cumulatives_df["B"][0], (np.bool_, pd.Timedelta)):
        return

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(test_cumulatives_df)


def test_cum_noncum_mix(memory_leak_check):
    """Tests that mixing cumulative and regular aggregations
    produces a reasonable error message."""

    def impl(df):
        return df.groupby("A")["B"].agg(("sum", "cumsum"))

    df = pd.DataFrame({"A": [1, 2, 3, 4] * 4, "B": [2, 3, 4, 5] * 4})
    with pytest.raises(
        BodoError,
        match="Cannot mix cumulative operations with other aggregation functions",
    ):
        bodo.jit(impl)(df)


# ------------------------------ df.groupby().transform()------------------------------ #
@pytest.mark.slow
def test_transform_args(memory_leak_check):
    """Test Groupby.transform with unsupported arguments"""

    # unsupported function
    def impl_func(df):
        A = df.groupby("A").transform("idxmin")
        return A

    # keyword engine
    def impl_engine(df):
        A = df.groupby("A").transform("sum", engine="cython")
        return A

    # keyword engine_kwargs
    def impl_engine_kwargs(df):
        A = df.groupby("A").transform("sum", engine_kwargs={"nopython": False})
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [1.1, 2.2, 1.1, 3.3, 4.4, 3.2, 45.6, 10.0],
        }
    )

    with pytest.raises(BodoError, match="unsupported transform function"):
        bodo.jit(impl_func)(df)

    err_msg = "parameter only supports default value"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_engine)(df)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl_engine_kwargs)(df)


@pytest.mark.slow
def test_transform_unsupported_type(memory_leak_check):
    """Test Groupby.transform('sum') with unsupported column type"""

    def impl(df):
        A = df.groupby("A").transform("sum")
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": pd.date_range("2018", "2019", periods=8),
        }
    )
    with pytest.raises(
        BodoError, match=re.escape("column type of datetime64[ns] is not supported by")
    ):
        bodo.jit(impl)(df)


# ------------------------------ df.groupby().head()------------------------------ #
@pytest.mark.slow
def test_head_negative_value(memory_leak_check):
    """Test Groupby.head() with -ve value"""

    def impl(df):
        A = df.groupby("A").head(-2)
        return A

    df = pd.DataFrame(
        {
            "A": [16, 1, 1, 1, 16, 16, 1, 40],
            "B": [True, False] * 4,
        }
    )
    with pytest.raises(BodoError, match="does not work with negative values."):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_groupby_agg_head(memory_leak_check):
    """
    Test Groupby.agg(): head cannot be mixed.
    """

    def impl(df):
        return df.groupby(by=["A"]).agg({"E": "sum", "C": "head"})

    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
    with pytest.raises(BodoError, match="head cannot be mixed"):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_unsupported_method(memory_leak_check):
    def impl(df):
        return df.groupby(by=["A"]).groups

    msg = re.escape("DataFrameGroupBy.groups not supported yet")
    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"]})
    with pytest.raises(BodoError, match=msg):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_unsupported_attr(memory_leak_check):
    def impl(df):
        return df.groupby(by=["A"]).get_group()

    msg = re.escape("DataFrameGroupBy.get_group not supported yet")
    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"]})
    with pytest.raises(BodoError, match=msg):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_unsupported_series_method(memory_leak_check):
    def impl(S):
        return S.groupby(level=0).nlargest()

    msg = re.escape("SeriesGroupBy.nlargest not supported yet")
    S = pd.Series([1, 2, 2])
    with pytest.raises(BodoError, match=msg):
        bodo.jit(impl)(S)


@pytest.mark.slow
def test_unsupported_series_attr(memory_leak_check):
    def impl(S):
        return S.groupby(level=0).is_monotonic_increasing

    msg = re.escape("SeriesGroupBy.is_monotonic_increasing not supported yet")
    S = pd.Series([1, 2, 2])
    with pytest.raises(BodoError, match=msg):
        bodo.jit(impl)(S)


@pytest.mark.slow
def test_unsupported_df_method(memory_leak_check):
    def impl(df):
        return df.groupby(by=["A"]).corrwith()

    msg = re.escape("DataFrameGroupBy.corrwith not supported yet")
    df = pd.DataFrame({"A": [1, 2, 2], "C": ["aa", "b", "c"]})
    with pytest.raises(BodoError, match=msg):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_ngroup_agg(memory_leak_check):
    """Test error message is shown when using ngroup inside agg.

    Args:
        memory_leak_check (fixture function): check memory leak in the test.
    """

    def impl(df):
        result = df.groupby("B").agg({"A": "ngroup", "C": "sum"})
        return result

    df = pd.DataFrame(
        {
            "A": [1, 3, 2, 1, 2, 3],
            "B": ["AA", "B", "XXX", "AA", "XXX", "B"],
            "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    with pytest.raises(
        BodoError, match="ngroup cannot be mixed with other groupby operations"
    ):
        bodo.jit(impl)(df)


@pytest.mark.slow
def test_ngroup_args(memory_leak_check):
    """Test error message is shown when using ngroup ascending argument with value other than default.

    Args:
        memory_leak_check (fixture function): check memory leak in the test.
    """

    # wrong type
    def impl1(df):
        result = df.groupby("B").ngroup(ascending=-1.2)
        return result

    # wrong value
    def impl2(df):
        result = df.groupby("B").ngroup(ascending=False)
        return result

    df = pd.DataFrame(
        {
            "A": [1, 3, 2, 1, 2, 3],
            "B": ["AA", "B", "XXX", "AA", "XXX", "B"],
            "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    err_msg = "ascending parameter only supports default value True"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(df)

    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(df)
