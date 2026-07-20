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
@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT TRIM(A) FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["A"].str.strip()}),
            id="one_arg",
            marks=pytest.mark.bodosql_cpp,
        ),
        pytest.param(
            "SELECT TRIM(B, '*') FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["B"].str.strip("*")}),
            id="two_args1",
            marks=pytest.mark.bodosql_cpp,
        ),
        pytest.param(
            "SELECT TRIM(C, 'asd') FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["C"].str.strip("asd")}),
            id="two_args2",
            marks=pytest.mark.bodosql_cpp,
        ),
        pytest.param(
            "SELECT TRIM(C, LEFT(C, 1)) FROM table1",
            lambda df: pd.DataFrame(
                {
                    0: pd.Series(
                        ["sdafzcvdf", "sasdaads", None, "cvxcbxasd", "akjhkjhs"] * 4
                    )
                }
            ),
            id="nonconstant_trim_chars",
        ),
    ],
)
def test_trim(trim_df, query, answer, memory_leak_check):
    """
    Tests TRIM with and without optional characters (scalars/columns)
    """
    check_query(
        query,
        trim_df,
        spark=None,
        check_dtype=False,
        check_names=False,
        expected_output=answer(trim_df),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT LTRIM(A) FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["A"].str.lstrip()}),
            id="one_arg",
            marks=pytest.mark.bodosql_cpp,
        ),
        pytest.param(
            "SELECT LTRIM(B, '*') FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["B"].str.lstrip("*")}),
            id="two_args1",
            marks=pytest.mark.bodosql_cpp,
        ),
        pytest.param(
            "SELECT LTRIM(C, 'asd') FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["C"].str.lstrip("asd")}),
            id="two_args2",
            marks=pytest.mark.bodosql_cpp,
        ),
        pytest.param(
            "SELECT LTRIM(C, LEFT(C, 1)) FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["C"].str[1:]}),
            id="nonconstant_trim_chars",
        ),
    ],
)
def test_ltrim(trim_df, query, answer, memory_leak_check):
    """
    Tests LTRIM with and without optional characters (scalars/columns)
    """
    check_query(
        query,
        trim_df,
        spark=None,
        check_dtype=False,
        check_names=False,
        expected_output=answer(trim_df),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT RTRIM(A) FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["A"].str.rstrip()}),
            id="one_arg",
            marks=pytest.mark.bodosql_cpp,
        ),
        pytest.param(
            "SELECT RTRIM(B, '*') FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["B"].str.rstrip("*")}),
            id="two_args1",
            marks=pytest.mark.bodosql_cpp,
        ),
        pytest.param(
            "SELECT RTRIM(C, 'asd') FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["C"].str.rstrip("asd")}),
            id="two_args2",
            marks=pytest.mark.bodosql_cpp,
        ),
        pytest.param(
            "SELECT RTRIM(C, RIGHT(C, 1)) FROM table1",
            lambda df: pd.DataFrame({0: df["TABLE1"]["C"].str[:-1]}),
            id="nonconstant_trim_chars",
        ),
    ],
)
def test_rtrim(trim_df, query, answer, memory_leak_check):
    """
    Tests RTRIM with and without optional characters (scalars/columns)
    """
    check_query(
        query,
        trim_df,
        spark=None,
        check_dtype=False,
        check_names=False,
        expected_output=answer(trim_df),
    )
