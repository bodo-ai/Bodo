"""
Tests dataframe library frontend (no triggering of execution).
"""

import pytest

import bodo.pandas as pd
from bodo.pandas.utils import BodoLibFallbackWarning


@pytest.mark.skip("disabled for release until merge is implemented")
def test_read_join_filter_proj(datapath):
    df1 = pd.read_parquet(datapath("dataframe_library/df1.parquet"))
    df2 = pd.read_parquet(datapath("dataframe_library/df2.parquet"))
    df3 = df1.merge(df2, on="A")
    df3 = df3[df3.A > 3]
    df3[["B", "C"]]


def test_df_getitem_fallback_warning():
    """Make sure DataFrame.__getitem__() raises a warning when falling back to Pandas."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3], "Int64"),
            "B": ["A1", "B1", "C1"],
        },
    )
    bdf = pd.from_pandas(df)
    with pytest.warns(BodoLibFallbackWarning):
        bdf[:]


def test_df_setitem_fallback_warning():
    """Make sure DataFrame.__setitem__() raises a warning when falling back to Pandas."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3], "Int64"),
        },
    )
    bdf = pd.from_pandas(df)
    with pytest.warns(BodoLibFallbackWarning):
        bdf[:] = 1


def test_df_apply_fallback_warning():
    """Make sure DataFrame.apply() raises a warning when falling back to Pandas."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3], "Int64"),
            "B": ["A1", "B1", "C1"],
        },
    )
    bdf = pd.from_pandas(df)
    with pytest.warns(BodoLibFallbackWarning):
        bdf.apply(lambda a: pd.Series([1, 2]), axis=1)


def test_merge_validation_checks():
    """Simple test for DataFrame merge validation."""
    import pandas as pds

    df1 = pds.DataFrame(
        {
            "A": pd.array([2, 2, 3], "Int64"),
            "B": ["a1", "b11", "c111"],
            "E": [1.1, 2.2, 3.3],
        },
    )
    df2 = pds.DataFrame(
        {
            "C": pd.array([2, 3, 8], "Int64"),
            "D": ["a1", "b222", "c33"],
        },
    )

    bdf1 = pd.from_pandas(df1)
    bdf2 = pd.from_pandas(df2)

    # Pandas merge should raise ValueError due to mismatched key lengths
    with pytest.raises(ValueError):
        df1.merge(df2, how="inner", left_on=["A", "B"], right_on=["C"])

    # BodoDataFrame merge should raise ValueError as well
    with pytest.raises(ValueError):
        bdf1.merge(bdf2, how="inner", left_on=["A", "B"], right_on=["C"])

    # bdf1.merge(bdf2, how="inner", left_on=["C"], right_on=["C"])
    df1.merge(df2, how="inner", left_on=["A"], right_on="C")
    bdf1.merge(bdf2, how="inner", left_on=["A"], right_on="C")

    # Number of elements mismatch, should raise ValueError
    with pytest.raises(ValueError):
        bdf1.merge(bdf2, how="inner", left_on=["A", "B"], right_on="C")

    # Validation checks should pass: tuple vs. tuple, tuple vs. string
    bdf1.merge(bdf2, how="inner", left_on=("A",), right_on=("C",))
    bdf1.merge(bdf2, how="inner", left_on=("A",), right_on="C")

    df3 = pds.DataFrame(
        {
            "cat": pd.array([2, 3, 8], "Int64"),
            "duck": ["a1", "b222", "c33"],
        },
    )

    bdf3 = pd.from_pandas(df3)

    # Pandas passes checks when column names are multi-character strings
    df1.merge(df3, how="inner", left_on=("A",), right_on="cat")

    # Bodo should pass checks when column names are multi-character strings
    bdf1.merge(bdf3, how="inner", left_on=("A",), right_on="cat")
    bdf1.merge(bdf3, how="inner", left_on=("A",), right_on=["cat"])

    # Check should fail if only one of left_on and right_on is provided
    with pytest.raises(ValueError):
        bdf1.merge(bdf2, how="inner", left_on=["A", "B"], right_on=None)

    # Check should fail if invalid key contained in on
    with pytest.raises(KeyError):
        bdf1.merge(bdf2, how="inner", left_on=["A", "C"], right_on=["C", "D"])

    # TODO[BSE-4810]: support "on" argument, which requires removing extra copy of
    # key columns with the same names from output
    # With "on" argument support, below test cases should work

    # bdf1.merge(bdf3, how="inner")
    # with pytest.raises(ValueError):
    #     bdf1.merge(bdf2, how="inner", on=["A", "C"])
    # bdf1.merge(bdf1, how="inner", on=["A", "B", "E"])
