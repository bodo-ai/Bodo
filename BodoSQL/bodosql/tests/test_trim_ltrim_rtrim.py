"""
Test correctness of SQL TRIM, LTRIM and RTRIM functions on BodoSQL
"""

import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.slow
def test_trim(trim_df, memory_leak_check):
    """
    Tests TRIM with and without optional characters (scalars/columns)
    """

    queries = (
        "SELECT TRIM(A) FROM table1",
        "SELECT TRIM(B, '*') FROM table1",
        "SELECT TRIM(C, 'asd') FROM table1",
        "SELECT TRIM(C, LEFT(C, 1)) FROM table1",
    )
    answers = (
        pd.DataFrame({0: trim_df["TABLE1"]["A"].str.strip()}),
        pd.DataFrame({0: trim_df["TABLE1"]["B"].str.strip("*")}),
        pd.DataFrame({0: trim_df["TABLE1"]["C"].str.strip("asd")}),
        pd.DataFrame(
            {0: pd.Series(["sdafzcvdf", "sasdaads", None, "cvxcbxasd", "akjhkjhs"] * 4)}
        ),
    )

    for query, answer in zip(queries, answers):
        check_query(
            query,
            trim_df,
            spark=None,
            check_dtype=False,
            check_names=False,
            expected_output=answer,
        )


@pytest.mark.slow
def test_ltrim(trim_df, memory_leak_check):
    """
    Tests LTRIM with and without optional characters (scalars/columns)
    """

    queries = (
        "SELECT LTRIM(A) FROM table1",
        "SELECT LTRIM(B, '*') FROM table1",
        "SELECT LTRIM(C, 'asd') FROM table1",
        "SELECT LTRIM(C, LEFT(C, 1)) FROM table1",
    )
    answers = (
        pd.DataFrame({0: trim_df["TABLE1"]["A"].str.lstrip()}),
        pd.DataFrame({0: trim_df["TABLE1"]["B"].str.lstrip("*")}),
        pd.DataFrame({0: trim_df["TABLE1"]["C"].str.lstrip("asd")}),
        pd.DataFrame({0: trim_df["TABLE1"]["C"].str[1:]}),
    )

    for query, answer in zip(queries, answers):
        check_query(
            query,
            trim_df,
            spark=None,
            check_dtype=False,
            check_names=False,
            expected_output=answer,
        )


@pytest.mark.slow
def test_rtrim(trim_df, memory_leak_check):
    """
    Tests RTRIM with and without optional characters (scalars/columns)
    """
    queries = (
        "SELECT RTRIM(A) FROM table1",
        "SELECT RTRIM(B, '*') FROM table1",
        "SELECT RTRIM(C, 'asd') FROM table1",
        "SELECT RTRIM(C, RIGHT(C, 1)) FROM table1",
    )
    answers = (
        pd.DataFrame({0: trim_df["TABLE1"]["A"].str.rstrip()}),
        pd.DataFrame({0: trim_df["TABLE1"]["B"].str.rstrip("*")}),
        pd.DataFrame({0: trim_df["TABLE1"]["C"].str.rstrip("asd")}),
        pd.DataFrame({0: trim_df["TABLE1"]["C"].str[:-1]}),
    )

    for query, answer in zip(queries, answers):
        check_query(
            query,
            trim_df,
            spark=None,
            check_dtype=False,
            check_names=False,
            expected_output=answer,
        )
