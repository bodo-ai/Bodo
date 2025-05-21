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


def test_df_apply_bad_dtype_fallback_warning():
    """Make sure DataFrame.apply() raises a warning when falling back to Pandas.
    In cases where it could not infer the dtype properly.
    """
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3] * 50, "Int64"),
            "B": ["A1", "B1", "C1"] * 50,
        },
    )
    bdf = pd.from_pandas(df)
    # All None Case
    with pytest.warns(BodoLibFallbackWarning):
        bdf.apply(lambda a: None, axis=1)

    # Unsupported arrow types
    with pytest.warns(BodoLibFallbackWarning):
        bdf.apply(lambda a: (1, "a"), axis=1)
