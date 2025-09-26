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

    # Arithmetic expression arguments with different source plans
    bdf = bd.from_pandas(df)
    bdf2 = bd.from_pandas(pd.DataFrame({"A": [1, 2, 3]}))
    with pytest.warns(BodoLibFallbackWarning):
        (bdf.A + bdf2.A)

    # Comparison expression arguments with different source plans
    bdf = bd.from_pandas(df)
    bdf2 = bd.from_pandas(pd.DataFrame({"A": [1, 2, 3]}))
    with pytest.warns(BodoLibFallbackWarning):
        (bdf.A == bdf2.A)

    # Conjunction expression arguments with different source plans
    bdf = bd.from_pandas(df)
    bdf2 = bd.from_pandas(pd.DataFrame({"A": [1, 2, 3]}))
    with pytest.warns(BodoLibFallbackWarning):
        ((bdf.A == 1) & (bdf2.A == 1))


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


@pytest.mark.jit_dependency
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


@pytest.mark.jit_dependency
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


def test_execution_counter():
    """Test execution counter, simulate application cases of execution counter to unit tests."""

    from bodo.pandas.plan import PlanExecutionCounter, assert_executed_plan_count

    df = bd.DataFrame({"A": ["2", "3"]})
    PlanExecutionCounter.reset()
    assert PlanExecutionCounter.get() == 0, "Execution plan counter not reset properly."

    plans = []

    with assert_executed_plan_count(0):
        for _ in range(5):
            plans.append(df.A.str.lower())

    with assert_executed_plan_count(5):
        for plan in plans:
            assert plan.is_lazy_plan()
            plan.execute_plan()

    try:
        with assert_executed_plan_count(1):
            pass
    except AssertionError as e:
        assert (
            str(e) == "Expected 1 plan executions, but got 0"
        )  # Created an assertion but not the expected error message.
    else:
        assert False  # Shouldn't have created an assertion but didn't.


def test_nested_cte():
    """final uses two versions of A and A use two versions of B.
    So, the B CTE is nested inside the A CTE.
    """
    C = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "val": [10, 20, 30, 40, 50, 60]})
    C = bd.DataFrame(C)

    # So CTE won't just be based on a dataframe.
    B = C[C["val"] < 100]
    B1 = B[B["id"] < 6]
    B2 = B[B["id"] > 1]

    A = bd.merge(B1, B2, on="id")
    A_left = A[A["val_x"] < 50]
    A_right = A[A["val_y"] > 0]

    final = bd.merge(A_left, A_right, left_on="val_x", right_on="val_y")

    generated_ctes = final._plan.get_cte_count()
    assert generated_ctes == 2


def test_non_nested_cte():
    """final uses two versions of A but A only uses one versions of B.
    B is used outside of A so only one CTE can be formed.
    """
    C = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "val": [10, 20, 30, 40, 50, 60]})
    C = bd.DataFrame(C)

    B = C[C["val"] < 100]
    A = B[B["val"] > 0]
    A_left = A[A["val"] < 40]
    A_right = A[A["val"] > 10]
    B_extra = B[B["val"] < 80]
    tmp = bd.merge(A_left, A_right, on="id")
    final = bd.merge(tmp, B_extra, on="id")

    generated_ctes = final._plan.get_cte_count()
    assert generated_ctes == 1
