# Copyright (C) 2019 Bodo Inc.
"""Test df.query()
"""
import random
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


##################### df.query() ###############################


@pytest.mark.parametrize(
    "expr",
    [
        "`B B` > @a + 1 & 5 > index > 1",
        "(A == @a) | (C == 'AA')",
        "C in ['AA', 'C']",
        "C not in ['AA', 'C']",
        "C.str.contains('C')",
        "abs(A) > @a",
        "A in [1, 4]",
        "A not in [1, 4]",
    ],
)
def test_df_query_unicode_expr(expr):
    """Test DataFrame.query with unicode(non-constant) expr
    """
    def impl(df, expr, a):
        return df.query(expr)

    df = pd.DataFrame(
        {
            "A": [1, 8, 4, 11, -3],
            "B B": [1.1, np.nan, 4.2, 3.1, -1.3],
            "C": ["AA", "BBB", "C", "AA", "C"],
        },
        index=[3, 1, 2, 4, 5],
    )
    check_func(impl, (df, expr, 1))


def test_df_query_stringliteral_expr():
    """Test DataFrame.query with StringLiteral(constant) expr
    """
    def impl(df):
        return df.query("a > b")

    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(5, 2), columns=list("ab"))
    check_func(impl, (df,))


################### df.query() errorchecking ######################


def test_df_query_inplace_false():
    """
    Test df.query(): 'inplace' is not supported, false only
    """

    def impl1(df):
        return df.query("a > b", inplace=True)

    def impl2(df):
        return df.query("a > b", inplace="True")

    def impl3(df, inplace):
        return df.query("a > b", inplace = inplace)

    inplace = True
    df = pd.DataFrame({"A": [1, 2, 2], "B": [2, 2, 1]})
    with pytest.raises(BodoError, match="inplace argument not supported yet"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="inplace argument not supported yet"):
        bodo.jit(impl2)(df)
    with pytest.raises(BodoError, match="inplace argument not supported yet"):
        bodo.jit(impl3)(df, inplace)


def test_df_query_expr_str():
    """
    Test df.query(): 'expr' is of type string
    """

    def impl1(df):
        return df.query(1)

    def impl2(df):
        return df.query(True)

    def impl3(df, expr):
        return df.query(expr)

    expr1 = 1
    expr2 = True
    df = pd.DataFrame({"A": [1, 2, 2], "B": [2, 2, 1]})
    with pytest.raises(BodoError, match="expr argument should be a string"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="expr argument should be a string"):
        bodo.jit(impl2)(df)
    with pytest.raises(BodoError, match="expr argument should be a string"):
        bodo.jit(impl3)(df, expr1)
    with pytest.raises(BodoError, match="expr argument should be a string"):
        bodo.jit(impl3)(df, expr2)


def test_df_query_expr_non_empty_str():
    """
    Test df.query(): 'expr' is not an empty string
    """

    def impl1(df):
        return df.query("")
    
    def impl2(df, expr):
        return df.query(expr)

    expr = ""
    df = pd.DataFrame({"A": [1, 2, 2], "B": [2, 2, 1]})
    with pytest.raises(BodoError, match="expr argument cannot be an empty string"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="expr argument cannot be an empty string"):
        bodo.jit(impl2)(df, expr)


def test_df_query_multiline_expr():
    """
    Test df.query(): 'expr' cannot be multilined
    """

    def impl1(df):
        return df.query("a\nb")

    def impl2(df, expr):
        return df.query(expr)

    expr = "a\nb"
    df = pd.DataFrame({"A": [1, 2, 2], "B": [2, 2, 1]})
    with pytest.raises(BodoError, match="multiline expressions not supported"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="multiline expressions not supported"):
        bodo.jit(impl2)(df, expr)


def test_df_query_str_column():
    """
    Test df.query(): column.str.*, column must exist in dataframe
    """

    def impl1(df):
        return df.query("C.str.contains('1')")

    def impl2(df, expr):
        return df.query(expr)

    expr = "C.str.contains('1')"
    df = pd.DataFrame(
        {
            "A": ['1', '8', '4', '11', '-3']
        }
    )
    with pytest.raises(BodoError, match="column .* is not found in dataframe columns"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="column .* is not found in dataframe columns"):
        bodo.jit(impl2)(df, expr)


def test_df_query_series_dt():
    """
    Test df.query(): Series.dt is not supported in expr
    """

    def impl1(df):
        return df.query("A.dt.year == 2012")

    def impl2(df, expr):
        return df.query(expr)

    expr = "A.dt.year == 2012"
    df = pd.DataFrame({"A": pd.date_range("1/1/2012", periods=5)})
    with pytest.raises(BodoError, match="Series.dt is not supported"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="Series.dt is not supported"):
        bodo.jit(impl2)(df, expr)


def test_df_query_expr_bool():
    """
    Test df.query(): expression should evaluate to a 1D boolean array
    """

    def impl1(df):
        return df.query("C")

    def impl2(df, expr):
        return df.query(expr)

    expr = "A+3"
    # df = pd.DataFrame({"A": pd.date_range("1/1/2012", periods=5)})
    df = pd.DataFrame(
        {
            "A": [1, 8, 4, 11, -3],
            "B B": [1.1, np.nan, 4.2, 3.1, -1.3],
            "C": ["AA", "BBB", "C", "AA", "C"],
        },
        index=[3, 1, 2, 4, 5],
    )
    with pytest.raises(BodoError, match="expr does not evaluate to a 1D boolean array"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="expr does not evaluate to a 1D boolean array"):
        bodo.jit(impl2)(df, expr)


def test_df_query_undef_var():
    """
    Test df.query(): error when there is undefined variable
    """

    def impl1(df):
        return df.query("A > @a")

    def impl2(df):
        return df.query("B > 2")

    def impl3(df, expr):
        return df.query(expr)

    df = pd.DataFrame(
        {
            "A": ['1', '8', '4', '11', '-3']
        }
    )
    expr1 = "A > @a"
    expr2 = "B > 2"
    with pytest.raises(BodoError, match="undefined variable"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="undefined variable"):
        bodo.jit(impl2)(df)
    with pytest.raises(BodoError, match="undefined variable"):
        bodo.jit(impl3)(df, expr1)
    with pytest.raises(BodoError, match="undefined variable"):
        bodo.jit(impl3)(df, expr2)


def test_df_query_index_name():
    """
    Test df.query(): Refering to named index by name is not supported
    """

    def impl1(df):
        return df.query("index_name<3")

    def impl2(df, expr):
        return df.query(expr)

    expr = "index_name<3"
    df = pd.DataFrame(
            {
                "A": [True, False, True, False, True],
                "B": [False, True, True, False, False],
            }
        )
    df.index.name = "index_name"

    with pytest.raises(BodoError, match="Refering to named index .* by name is not supported"):
        bodo.jit(impl1)(df)
    with pytest.raises(BodoError, match="Refering to named index .* by name is not supported"):
        bodo.jit(impl2)(df, expr)
