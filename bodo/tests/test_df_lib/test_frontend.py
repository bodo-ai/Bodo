"""
Tests dataframe library frontend (no triggering of execution).
"""

import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.pandas.utils import BodoLibFallbackWarning


def test_read_join_filter_proj(datapath):
    df1 = bd.read_parquet(datapath("dataframe_library/df1.parquet"))
    df2 = bd.read_parquet(datapath("dataframe_library/df2.parquet"))
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
    bdf = bd.from_pandas(df)
    with pytest.warns(BodoLibFallbackWarning):
        bdf[:]

    # Non-bodo Series
    bdf = bd.from_pandas(df)
    with pytest.warns(BodoLibFallbackWarning):
        bdf[df.A > 1]

    # Different expr source plan
    bdf = bd.from_pandas(df)
    bdf2 = bd.from_pandas(df)
    S = bdf2.A > 0
    with pytest.warns(BodoLibFallbackWarning):
        bdf[S]


def test_df_setitem_fallback_warning():
    """Make sure DataFrame.__setitem__() raises a warning when falling back to Pandas."""
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3], "Int64"),
        },
    )
    bdf = bd.from_pandas(df)
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
    bdf = bd.from_pandas(df)
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
    bdf = bd.from_pandas(df)
    # All None Case
    with pytest.warns(BodoLibFallbackWarning):
        bdf.apply(lambda a: None, axis=1)

    # Unsupported arrow types
    with pytest.warns(BodoLibFallbackWarning):
        bdf.apply(lambda a: (1, "a"), axis=1)


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

    bdf1 = bd.from_pandas(df1)
    bdf2 = bd.from_pandas(df2)

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

    bdf3 = bd.from_pandas(df3)

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

    # Check should pass since merging with self
    bdf1.merge(bdf1, how="inner")

    # Check should fail since merging with other, without on-args
    with pytest.raises(ValueError):
        bdf1.merge(bdf2, how="inner")

    # Check should fail since on-arg isn't a subset of the common columns
    with pytest.raises(KeyError):
        bdf1.merge(bdf2, how="inner", on=["A", "C"])

    # Check should pass since merging with self
    bdf1.merge(bdf1, how="inner", on=["A", "B", "E"])


def test_fallback_warning_on_unknown_attribute():
    """Test that accessing unsupported attributes raises a fallback warning."""
    import warnings

    import pandas as pds

    df = pds.DataFrame({"A": [1, 2, 3]})
    bdf = bd.from_pandas(df)

    # Trigger __getattribute__ fallback by accessing an unsupported attribute
    with pytest.warns(
        BodoLibFallbackWarning, match="not implemented.*Falling back to Pandas"
    ):
        _ = bdf.A.pop(0)

    # Internal attributes should not raise a warning
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _ = bdf.A._get_axis_number(0)
    assert not record, f"Shouldn't raise warnings for internal attributes: {record[0]}"

    # Known attributes should not raise a warning
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _ = bdf.A.dtype
    assert not record, f"Shouldn't raise warnings for this attribute: {record[0]}"


def test_single_fallback_warning_emitted():
    """Test that only the initial fallback warning is emitted when falling back to Pandas."""
    import warnings

    import pandas as pds

    df = pds.DataFrame({"A": ["a", "b", "a", "c"]})
    bdf = bd.from_pandas(df)

    # pop(0) will fall back and may internally call other methods, but should only raise initial warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _ = bdf.A.pop(0)

    fallback_warnings = [
        w for w in record if issubclass(w.category, BodoLibFallbackWarning)
    ]

    assert len(fallback_warnings) == 1, (
        f"Expected 1 fallback warning, got {len(fallback_warnings)}:\n{fallback_warnings}"
    )

    warning_msg = str(fallback_warnings[0].message)
    assert (
        "pop" in warning_msg
        and "not implemented" in warning_msg
        and "Falling back to Pandas" in warning_msg
    ), f"Unexpected warning message: {warning_msg}"
