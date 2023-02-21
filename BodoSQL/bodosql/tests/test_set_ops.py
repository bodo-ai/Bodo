# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL set like operations, namely UNION, INTERSECT, and EXCEPT
"""
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

from bodo.tests.timezone_common import representative_tz  # noqa


@pytest.fixture(
    params=[
        "UNION",
        pytest.param("UNION DISTINCT", marks=pytest.mark.slow),
        "UNION ALL",
    ]
)
def union_cmds(request):
    return request.param


@pytest.fixture(
    params=[
        "INTERSECT",
        pytest.param("INTERSECT DISTINCT", marks=pytest.mark.slow),
        pytest.param(
            "INTERSECT ALL", marks=pytest.mark.skip("[BE-4134] Support INTERSECT ALL")
        ),
    ]
)
def intersect_cmds(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("EXCEPT", marks=pytest.mark.skip("[BS-379] Except not supported")),
        pytest.param("MINUS", marks=pytest.mark.skip("[BS-379] Except not supported")),
    ]
)
def except_cmds(request):
    return request.param


@pytest.fixture
def null_set_df():
    int_data = {
        "A": [1, 2, 3, None] * 3,
        "B": [4, None, 2, 3] * 3,
        "C": [None, 1, 2, 3] * 3,
    }
    return {"table1": pd.DataFrame(data=int_data, dtype="Int64")}


def make_tz_aware_df(tz):
    """
    Make a dataframe of tz-aware timestamps with the given timezone.
    """
    # Note: B's and A's will overlap.
    tz_aware_data = {
        "A": list(pd.date_range(start="1/1/2022", freq="4D7H", periods=30, tz=tz))
        + [None] * 4,
        "B": [None] * 14
        + list(pd.date_range(start="1/1/2022", freq="12D21H", periods=20, tz=tz)),
    }
    return pd.DataFrame(data=tz_aware_data)


def test_union_numeric_cols(basic_df, union_cmds, spark_info, memory_leak_check):
    """
    Test that UNION [ALL/DISTINCT] works for a single numeric column
    """
    query = f"SELECT A FROM table1 {union_cmds} SELECT B FROM table1 {union_cmds} SELECT C FROM table1"
    check_query(query, basic_df, spark_info, check_dtype=False)


def test_union_nullable_numeric_many_cols(
    null_set_df, union_cmds, spark_info, memory_leak_check
):
    """
    Test that UNION [ALL/DISTINCT] works for many nullable numeric columns
    """
    query = f"SELECT A, B, C FROM table1 {union_cmds} SELECT C, C, B FROM table1"
    check_query(query, null_set_df, spark_info, check_dtype=False)


def test_union_null_literals(memory_leak_check):
    """
    Test for [BE-4320], checks that union works for various null literals
    """
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4] * 3,
            "B": [1.1, 2.7, 3.4, 110.3] * 3,
            "C": ["A", None, "c", "recall"] * 3,
        }
    )
    ctx = {"table1": df}
    expected_output = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 3, 4, None], dtype="Int64"),
            "B": pd.Series([1.1, 2.7, 3.4, 110.3, None], dtype="Float64"),
            "C": ["A", None, "c", "recall", None],
        }
    )
    query = (
        "(SELECT A, B, C from table1) UNION (select null as A, null as B, null as C)"
    )
    check_query(query, ctx, None, expected_output=expected_output, check_dtype=False)


def test_union_string_cols(
    bodosql_string_types, union_cmds, spark_info, memory_leak_check
):
    """
    Test that UNION [ALL/DISTINCT] works for string columns
    """
    query = f"SELECT A FROM table1 {union_cmds} SELECT B FROM table1"
    check_query(query, bodosql_string_types, spark_info, check_dtype=False)


def test_union_string_restrictions(
    bodosql_string_types, union_cmds, spark_info, memory_leak_check
):
    """
    Test that UNION [ALL/DISTINCT] works for string columns when used with restrictions
    """
    query = (
        f"(SELECT A, B, C FROM table1 WHERE LENGTH(A) = 5) {union_cmds} "
        f"(SELECT CASE WHEN LENGTH(B) > 4 THEN B END, C, A FROM table1 ORDER BY C) {union_cmds} "
        f"(SELECT B, C, A FROM table1 WHERE LENGTH(A) = 5 AND LENGTH(B) > 3 ORDER BY B LIMIT 2) {union_cmds} "
        f"(SELECT C, B, A FROM table1)"
    )
    check_query(query, bodosql_string_types, spark_info, check_dtype=False)


def test_union_tz_aware_cols(union_cmds, representative_tz, memory_leak_check):
    """
    Tests that UNION [ALL/DISTINCT] works for tz_aware columns
    """
    df = make_tz_aware_df(representative_tz)
    py_output = pd.DataFrame({"A": pd.concat((df["A"], df["B"]))})
    if "ALL" not in union_cmds:  # Drop duplicates for UNION, UNION DISTINCT
        py_output = py_output.drop_duplicates()
    ctx = {"table1": df}
    query = f"SELECT A FROM table1 {union_cmds} SELECT B FROM table1"
    check_query(query, ctx, None, expected_output=py_output)


@pytest.mark.slow
@pytest.mark.parametrize(
    "query,unknown1,unknown2",
    [
        # [BS-381] These queries are not valid for spark
        ("SELECT 1,2 UNION SELECT 1,2", [1], [2]),
        ("SELECT 1,2 UNION SELECT 1,3", [1, 1], [2, 3]),
        ("SELECT 1,2 UNION ALL SELECT 1,2", [1, 1], [2, 2]),
    ],
)
def test_union_scalars(query, unknown1, unknown2, memory_leak_check):
    """
    Tests that UNION [ALL/DISTINCT] works for scalars
    """
    py_output = pd.DataFrame({"unknown1": unknown1, "unknown2": unknown2})
    check_query(
        query,
        dict(),
        None,
        check_names=False,
        expected_output=py_output,
        check_dtype=False,
    )


def test_intersect_numeric_cols(
    basic_df, intersect_cmds, spark_info, memory_leak_check
):
    """
    Test that INTERSECT [ALL/DISTINCT] works for a single numeric column
    """
    query = f"SELECT A FROM table1 {intersect_cmds} SELECT B FROM table1 {intersect_cmds} SELECT C FROM table1"
    check_query(query, basic_df, spark_info, check_dtype=False)


@pytest.mark.skip("[BE-4414] Intersect fails with many columns")
def test_intersect_nullable_numeric_many_cols(
    null_set_df, intersect_cmds, spark_info, memory_leak_check
):
    """
    Test that INTERSECT [ALL/DISTINCT] works for many nullable numeric columns
    """
    query = f"SELECT A, B, C FROM table1 {intersect_cmds} SELECT C, C, B FROM table1"
    check_query(query, null_set_df, spark_info, check_dtype=False)


def test_intersect_string_cols(
    bodosql_string_types, intersect_cmds, spark_info, memory_leak_check
):
    """
    Test that INTERSECT [ALL/DISTINCT] works for string columns
    """
    query = f"SELECT A FROM table1 {intersect_cmds} SELECT B FROM table1"
    check_query(query, bodosql_string_types, spark_info, check_dtype=False)


def test_intersect_string_restrictions(
    bodosql_string_types, intersect_cmds, spark_info, memory_leak_check
):
    """
    Test that INTERSECT [ALL/DISTINCT] works for string columns when used with restrictions
    """
    query = (
        f"(SELECT CASE WHEN LENGTH(B) > 4 THEN B END, C, A FROM table1 ORDER BY C) {intersect_cmds} "
        f"(SELECT B, C, A FROM table1 WHERE LENGTH(A) = 5 AND LENGTH(B) > 3 ORDER BY B LIMIT 2) {intersect_cmds} "
        f"(SELECT B, C, A FROM table1)"
    )
    check_query(
        query, bodosql_string_types, spark_info, check_names=False, check_dtype=False
    )


def test_intersect_tz_aware_cols(intersect_cmds, representative_tz, memory_leak_check):
    """
    Test that INTERSECT [ALL/DISTINCT] works for tz_aware columns
    """
    df = make_tz_aware_df(representative_tz)
    py_output = df[["A"]].merge(df[["B"]].rename(columns={"B": "A"}), on="A")

    # If there is 1 NA in the output it should exist.
    # TODO: Double check intersect null behavior with Snowflake.
    na_entries = py_output["A"].isna()
    # If there is 1 NA in the output it should exist.
    append_na = py_output["A"].isna().any()
    py_output = py_output[~na_entries]
    if append_na:
        py_output = pd.DataFrame({"A": list(py_output["A"]) + [None]})

    ctx = {"table1": df}
    query = f"SELECT A FROM table1 {intersect_cmds} SELECT B FROM table1"
    check_query(query, ctx, None, expected_output=py_output)


@pytest.mark.slow
@pytest.mark.parametrize(
    "query,unknown1,unknown2",
    [
        # [BS-381] These queries are not valid for spark
        ("SELECT 1,2 INTERSECT SELECT 2,3", [], []),
        ("SELECT 1,2 INTERSECT SELECT 1,2", [1], [2]),
        pytest.param(
            "SELECT 1,2 INTERSECT ALL SELECT 1,2",
            [1, 1],
            [2, 2],
            marks=pytest.mark.skip("[BE-4134] Support intersect all"),
        ),
    ],
)
def test_intersect_scalars(query, unknown1, unknown2, memory_leak_check):
    """
    Test that INTERSECT [ALL/DISTINCT] works for scalars
    """
    py_output = pd.DataFrame({"unknown1": unknown1, "unknown2": unknown2})
    check_query(
        query,
        dict(),
        None,
        check_names=False,
        expected_output=py_output,
        check_dtype=False,
    )


def test_except_cols(basic_df, except_cmds, spark_info, memory_leak_check):
    """
    Test that EXCEPT/MINUS works for columns
    """
    query = f"SELECT A, B FROM table1 {except_cmds} SELECT A, C FROM table1"
    check_query(query, basic_df, spark_info, only_python=True)


def test_except_string_cols(
    bodosql_string_types, except_cmds, spark_info, memory_leak_check
):
    """
    Test that EXCEPT/MINUS works for string columns
    """
    query = f"SELECT A, B FROM table1 {except_cmds} SELECT A, C FROM table1"
    check_query(query, bodosql_string_types, spark_info, only_python=True)


def test_except_string_restrictions(
    bodosql_string_types, except_cmds, spark_info, memory_leak_check
):
    """
    Test that EXCEPT/MINUS works for string columns when used with restrictions
    """
    query = (
        f"(SELECT CASE WHEN LENGTH(B) > 4 THEN B END, C, A FROM table1 ORDER BY C) {except_cmds} "
        f"(SELECT B, C, A FROM table1 WHERE LENGTH(A) = 5 AND LENGTH(B) > 3 ORDER BY B LIMIT 2) "
    )
    check_query(query, bodosql_string_types, spark_info, check_dtype=False)


def test_except_tz_aware_cols(except_cmds, representative_tz, memory_leak_check):
    """
    Tests that EXCEPT/MINUS works for tz_aware columns
    """
    df = make_tz_aware_df(representative_tz)
    py_output = pd.DataFrame({"A": list(set(df["A"]).difference(set(df["B"])))})
    ctx = {"table1": df}
    query = f"SELECT A FROM table1 {except_cmds} SELECT B FROM table1"
    check_query(query, ctx, None, expected_output=py_output)


@pytest.mark.slow
@pytest.mark.skip("[BS-379] Except not supported")
@pytest.mark.parametrize(
    "query,unknown1,unknown2",
    [
        # [BS-381] These queries are not valid for spark
        ("SELECT 1,2 EXCEPT SELECT 2,3", [1], [2]),
        ("SELECT 1,2 EXCEPT SELECT 1,2", [], []),
    ],
)
def test_except_scalars(query, unknown1, unknown2, memory_leak_check):
    """
    Tests that EXCEPT works for scalars
    """
    py_output = pd.DataFrame({"unknown1": unknown1, "unknown2": unknown2})
    check_query(
        query,
        dict(),
        None,
        check_names=False,
        expected_output=py_output,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-379] Except not supported")
def test_set_ops_join(join_dataframes, spark_info, memory_leak_check):
    """
    Integration test for UNION, INTERSECT, and EXCEPT on joined tables.
    The column order should be the same across join types.
    """
    query = (
        f"SELECT * FROM table1.A OUTER JOIN table2.A ON table1.A = table2.A EXCEPT "
        f"SELECT * FROM table1.A LEFT JOIN table2.A ON table1.A = table2.A UNION "
        f"SELECT * FROM table1.A RIGHT JOIN table2.A ON table1.A = table2.A INTERSECT "
        f"SELECT * FROM table1.A OUTER JOIN table2.A ON table1.A = table2.A"
    )
    check_query(query, join_dataframes, spark_info, check_dtype=False)
