"""
Test correctness of SQL case queries involving NULL on BodoSQL
"""

import pandas as pd
import pytest

from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_null_then(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """tests a case statement that always evaluates to true propagates
    null values correctly."""
    query = "Select Case WHEN (A IS NULL OR A <> 25) THEN B ELSE C END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_null_then_handled(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """tests a case statement that always evaluates to true handles null values
    properly when then checks for null."""
    query = "Select Case WHEN (A IS NULL OR A <> 25) THEN B IS NULL ELSE C IS NOT NULL END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_null_then_multicolumn(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """tests a case statement that always evaluates to true propagates
    null values correctly using multiple columns."""
    query = "Select Case WHEN (A IS NULL OR A <> 25) THEN A + B ELSE C END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_null_else(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """tests a case statement that always evaluates to false propagates
    null values correctly."""
    query = "Select Case WHEN (A IS NOT NULL AND A = 25) THEN B ELSE C END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_null_else_handled(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """tests a case statement that always evaluates to false handles null values
    properly when then checks for null."""
    query = "Select Case WHEN (A IS NOT NULL AND A = 25) THEN B IS NOT NULL ELSE C IS NULL END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_null_else_multicolumn(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """tests a case statement that always evaluates to false propagates
    null values correctly using multiple columns."""
    query = (
        "Select Case WHEN (A IS NOT NULL AND A = 25) THEN B ELSE C - A END FROM table1"
    )
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_null_when(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """tests a case statement where when may have null values is handled properly."""
    query = "Select Case WHEN A > 2 THEN B ELSE C END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_null_when_multicolumn(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """tests a case statement where when may have null values is handled properly using multiple columns."""
    query = "Select Case WHEN A + B > 6 THEN B ELSE C END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_null_when_handled_multicolumn(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """tests a case statement where null is handled properly."""
    query1 = "Select Case WHEN A IS NOT NULL THEN B ELSE C END FROM table1"
    check_query(
        query1,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    query2 = "Select Case WHEN A IS NULL THEN B ELSE C END FROM table1"
    check_query(
        query2,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_null_when_handled_multicolumn(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """tests that a case where null is handled properly using multiple columns."""
    query = "Select Case WHEN (A + B) IS NOT NULL THEN B ELSE C END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_null_when_or(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """tests a case statement where when may have null values using or is handled properly."""
    query = "Select Case WHEN A > 2 OR B < 6 THEN B ELSE C END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_null_when_and(bodosql_nullable_numeric_types, spark_info, memory_leak_check):
    """tests a case statement where when may have null values using or is handled properly."""
    query = "Select Case WHEN A > 2 AND B < 6 THEN B ELSE C END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_null_when_and_shortcircuit(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """tests a case statement where when may have null values using or is handled properly."""
    query = "Select Case WHEN A is NULL AND B <= 6 THEN B ELSE C END FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "scalar_values",
    [
        pytest.param(
            (True, False),
            id="bools",
        ),
        pytest.param(("'hello'", "'world'"), id="strings"),
        pytest.param(
            ("TIMESTAMP 2022-06-09", "TIMESTAMP 2022-06-17"),
            marks=pytest.mark.skip(
                "need to support pd.to_datetime(None) in the engine"
            ),
            id="timestamps",
        ),
        pytest.param(
            ("INTERVAL '1' year", "INTERVAL '3' months", pd.Timedelta(minutes=10)),
            marks=pytest.mark.skip(
                "need to support pd.to_timedelta(None) in the engine"
            ),
            id="timedeltas",
        ),
        pytest.param((0, 1), marks=pytest.mark.slow, id="ints"),
    ],
)
def test_null_scalars_cast(basic_df, spark_info, scalar_values, memory_leak_check):
    """Tests an issue found wherein we Calcite generates code that casts scalar nulls in CASE stmts.
    We test this on each type, to insure that the scalar conversion library functions all work.
    (see sqlTypenameToPandasTypename in utils.java)
    """

    query = f"select CASE WHEN A>0 THEN (CASE WHEN B = 1 THEN {scalar_values[0]} END) ELSE {scalar_values[1]} END from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_null_literal_then_else(major_types_nullable, memory_leak_check):
    """
    Tests passing a null literal to a case statement for each major type.
    """
    query = """
        select
            CASE when COND_COL THEN NULL ELSE C END as THEN_COL,
            CASE when COND_COL THEN B ELSE NULL END as ELSE_COL
        FROM
            table1
    """
    then_list = []
    else_list = []
    orig_df = major_types_nullable["TABLE1"]
    for i, val in enumerate(orig_df["COND_COL"]):
        if val is None or pd.isnull(val):
            then_list.append(None)
            else_list.append(orig_df.loc[i, "B"])
        else:
            then_list.append(orig_df.loc[i, "C"])
            else_list.append(None)
    if pd.api.types.is_integer_dtype(orig_df.dtypes["A"]):
        dtype = "Int64"
    else:
        dtype = None
    py_output = pd.DataFrame(
        {
            "THEN_COL": pd.Series(then_list, dtype=dtype),
            "ELSE_COL": pd.Series(else_list, dtype=dtype),
        }
    )
    check_query(
        query,
        major_types_nullable,
        None,
        expected_output=py_output,
        check_dtype=False,
    )


def test_null_literal_cond(major_types_nullable, memory_leak_check):
    """
    Tests comparing IS_NULL and IS_NOT_NULL for each major type.
    """
    query = """
        select
            A,
            B,
            C,
            CASE when A IS NULL THEN B ELSE C END as NULL_COL,
            CASE when A IS NOT NULL THEN B ELSE C END as NOT_NULL_COL
        FROM
            table1
    """
    null_list = []
    not_null_list = []
    orig_df = major_types_nullable["TABLE1"]
    for i, val in enumerate(orig_df["A"]):
        if val is None or pd.isnull(val):
            null_list.append(orig_df.loc[i, "B"])
            not_null_list.append(orig_df.loc[i, "C"])
        else:
            null_list.append(orig_df.loc[i, "C"])
            not_null_list.append(orig_df.loc[i, "B"])
    if pd.api.types.is_integer_dtype(orig_df.dtypes["A"]):
        dtype = "Int64"
    else:
        dtype = None
    py_output = pd.DataFrame(
        {
            "A": orig_df["A"],
            "B": orig_df["B"],
            "C": orig_df["C"],
            "NULL_COL": pd.Series(null_list, dtype=dtype),
            "NOT_NULL_COL": pd.Series(not_null_list, dtype=dtype),
        }
    )
    check_query(
        query,
        major_types_nullable,
        None,
        expected_output=py_output,
        check_dtype=False,
    )
