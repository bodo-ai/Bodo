import pandas as pd
import numpy as np
import numba
import bodo
from bodo.utils.typing import BodoError
import pytest


df1 = pd.DataFrame({"A": [1, 2, 3], "C": ["aa", "b", "c"], "E": ["aa", "bb", "cc"]})
df2 = pd.DataFrame({"A": [1, 2, 5], "B": ["aa", "b", "c"], "C": ["aa", "bb", "cc"]})


# tests left is of type dataframe
def test_left_dataframe():
    def impl(df1):
        return pd.merge("abc", df1)

    with pytest.raises(BodoError, match="requires dataframe inputs"):
        bodo.jit(impl)(df1)


# tests right is of type dataframe
def test_right_dataframe():
    def impl(df1):
        return df1.merge("abc")

    with pytest.raises(BodoError, match="requires dataframe inputs"):
        bodo.jit(impl)(df1)


# tests how is of type str
def test_how_str():
    def impl(df1, df2):
        return df1.merge(df2, how=3)

    with pytest.raises(BodoError, match="how parameter must be of type str"):
        bodo.jit(impl)(df1, df2)


# tests how is one of ["left", "right", "outer", "inner"]
def test_how_invalid():
    def impl(df1, df2):
        return df1.merge(df2, how="break")

    with pytest.raises(BodoError, match="invalid key .* for how"):
        bodo.jit(impl)(df1, df2)


# tests invalid on key in left dataframe
def test_on_invalid_index_left():
    def impl(df1, df2):
        return df1.merge(df2, on=["A", "B"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests invalid on key in right dataframe
def test_on_invalid_index_right():
    def impl(df1, df2):
        return df1.merge(df2, on=["A", "E"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests invalid on key in both dataframes
def test_on_invalid_index_both():
    def impl(df1, df2):
        return df1.merge(df2, on=["A", "break"])

    with pytest.raises(BodoError, match="invalid key .* for on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests on without common cols
def test_on_no_comm_cols():
    df3 = pd.DataFrame(
        {"AA": [1, 2, 3], "CC": ["aa", "b", "c"], "EE": ["aa", "bb", "cc"]}
    )

    def impl(df1, df2):
        return df1.merge(df2, on=["A"])

    with pytest.raises(BodoError, match="No common columns to perform merge on"):
        bodo.jit(impl)(df1, df3)


# tests both on and left_on specified
def test_on_lefton():
    def impl(df1, df2):
        return df1.merge(df2, on=["A"], left_on=["C"])

    with pytest.raises(
        BodoError,
        match='Can only pass argument "on" OR "left_on" '
        'and "right_on", not a combination of both',
    ):
        bodo.jit(impl)(df1, df2)


# tests both on and lefton specified
def test_on_righton():
    def impl(df1, df2):
        return df1.merge(df2, on=["A"], right_on=["C"])

    with pytest.raises(
        BodoError,
        match='Can only pass argument "on" OR "left_on" '
        'and "right_on", not a combination of both',
    ):
        bodo.jit(impl)(df1, df2)


# tests only left_on specified
def test_lefton_only():
    def impl(df1, df2):
        return df1.merge(df2, left_on=["C"])

    with pytest.raises(BodoError, match="Must pass .*_on or .*_index=True"):
        bodo.jit(impl)(df1, df2)


# tests only right_on specified
def test_righton_only():
    def impl(df1, df2):
        return df1.merge(df2, right_on=["C"])

    with pytest.raises(BodoError, match="Must pass .*_on or .*_index=True"):
        bodo.jit(impl)(df1, df2)


# tests invalid left_on key
def test_lefton_invalid():
    def impl(df1, df2):
        return df1.merge(df2, left_on=["A", "B"], right_on=["A", "B"])

    with pytest.raises(BodoError, match="invalid key .* on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests invalid right_on key
def test_lefton_invalid():
    def impl(df1, df2):
        return df1.merge(df2, left_on=["A", "E"], right_on=["A", "E"])

    with pytest.raises(BodoError, match="invalid key .* on/left_on/right_on"):
        bodo.jit(impl)(df1, df2)


# tests unequal lengths of left_on and right_on
def test_lefton_invalid():
    def impl(df1, df2):
        return df1.merge(df2, left_on=["A"], right_on=["A", "B"])

    with pytest.raises(BodoError, match="len\(right_on\) must equal len\(left_on\)"):
        bodo.jit(impl)(df1, df2)


# tests left_index is of type bool
def test_leftindex_bool():
    def impl(df1, df2):
        return df1.merge(df2, left_index="A", right_index=True)

    with pytest.raises(BodoError, match="left_index parameter must be of type bool"):
        bodo.jit(impl)(df1, df2)


# tests right_index is of type bool
def test_rightindex_bool():
    def impl(df1, df2):
        return df1.merge(df2, left_index=True, right_index="B")

    with pytest.raises(BodoError, match="right_index parameter must be of type bool"):
        bodo.jit(impl)(df1, df2)


# tests only left_on specified
def test_leftindex_only():
    def impl(df1, df2):
        return df1.merge(df2, left_index=True)

    with pytest.raises(BodoError, match="Must pass .*_on or .*_index=True"):
        bodo.jit(impl)(df1, df2)


# tests only right_on specified
def test_rightindex_only():
    def impl(df1, df2):
        return df1.merge(df2, right_index=True)

    with pytest.raises(BodoError, match="Must pass .*_on or .*_index=True"):
        bodo.jit(impl)(df1, df2)


# tests right_index=True and len(left_on)!=1
def test_rightindex_lefton_len():
    def impl(df1, df2):
        return df1.merge(df2, right_index=True, left_on=["A", "C"])

    with pytest.raises(
        BodoError,
        match="len\(left_on\) must equal the number "
        'of levels in the index of "right", which is 1',
    ):
        bodo.jit(impl)(df1, df2)


# tests left_index=True and len(right_on)!=1
def test_leftindex_righton_len():
    def impl(df1, df2):
        return df1.merge(df2, left_index=True, right_on=["A", "C"])

    with pytest.raises(
        BodoError,
        match="len\(right_on\) must equal the number "
        'of levels in the index of "left", which is 1',
    ):
        bodo.jit(impl)(df1, df2)


# tests sort is of type bool
def test_sort_bool():
    def impl(df1, df2):
        return df1.merge(df2, sort="break")

    with pytest.raises(
        BodoError, match="sort parameter only supports default value False"
    ):
        bodo.jit(impl)(df1, df2)


# tests sort has default False
def test_sort():
    def impl(df1, df2):
        return df1.merge(df2, sort=True)

    with pytest.raises(
        BodoError, match="sort parameter only supports default value False"
    ):
        bodo.jit(impl)(df1, df2)


# tests suffixes cannot be specified
def test_suffixes():
    def impl(df1, df2):
        return df1.merge(df2, suffixes=("_x", "_y"))

    with pytest.raises(BodoError, match="suffixes parameter cannot be specified"):
        bodo.jit(impl)(df1, df2)


# tests copy is of type bool
def test_copy_bool():
    def impl(df1, df2):
        return df1.merge(df2, copy="break")

    with pytest.raises(
        BodoError, match="copy parameter only supports default value True"
    ):
        bodo.jit(impl)(df1, df2)


# tests copy has default True
def test_copy():
    def impl(df1, df2):
        return df1.merge(df2, copy=False)

    with pytest.raises(
        BodoError, match="copy parameter only supports default value True"
    ):
        bodo.jit(impl)(df1, df2)


# tests indicator is of type bool
def test_indicator_bool():
    def impl(df1, df2):
        return df1.merge(df2, indicator="break")

    with pytest.raises(
        BodoError, match="indicator parameter only supports default value False"
    ):
        bodo.jit(impl)(df1, df2)


# tests indicator has default False
def test_indicator():
    def impl(df1, df2):
        return df1.merge(df2, indicator=True)

    with pytest.raises(
        BodoError, match="indicator parameter only supports default value False"
    ):
        bodo.jit(impl)(df1, df2)


# tests validate has default None
def test_indicator_bool():
    def impl(df1, df2):
        return df1.merge(df2, validate=["one_to_one"])

    with pytest.raises(
        BodoError, match="validate parameter only supports default value None"
    ):
        bodo.jit(impl)(df1, df2)
