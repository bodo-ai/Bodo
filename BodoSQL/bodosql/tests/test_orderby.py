# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL queries containing orderby on BodoSQL
"""
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

from bodo.tests.timezone_common import representative_tz  # noqa


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": [1] * 12,
                    "B": [False, None, True, False] * 3,
                }
            )
        },
        {"table1": pd.DataFrame({"A": ["a"] * 12, "B": [1, 2, 3, 4, 5, 6] * 2})},
        {
            "table1": pd.DataFrame(
                {"A": [0] * 12, "B": ["a", "aa", "aaa", "ab", "b", "hello"] * 2}
            )
        },
    ]
)
def col_a_identical_tables(request):
    """
    Group of tables with identical column A, and varied column B, used for testing groupby
    """
    return request.param


@pytest.mark.slow
def test_orderby_numeric_scalar(bodosql_numeric_types, spark_info, memory_leak_check):
    """tests that orderby works with scalar values in the Select statment"""
    query = "SELECT A, 1, 2, 3, 4 as Y FROM table1 ORDER BY Y"
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        sort_output=False,
    )


@pytest.mark.slow
def test_orderby_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests orderby works in the simple case for numeric types
    """
    query = """
        SELECT
            A, B, C
        FROM
            table1
        ORDER BY
            A
        """
    query2 = """
        SELECT
            A, B, C
        FROM
            table1
        ORDER BY
            A DESC
        """
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, sort_output=False
    )
    check_query(
        query2, bodosql_numeric_types, spark_info, check_dtype=False, sort_output=False
    )


def test_orderby_nullable_numeric(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """
    Tests orderby works in the simple case for nullable numeric types
    """
    query = """
        SELECT
            A, B, C
        FROM
            table1
        ORDER BY
            A
        """
    query2 = """
        SELECT
            A, B, C
        FROM
            table1
        ORDER BY
            A DESC
        """
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_dtype=False,
        sort_output=False,
    )
    check_query(
        query2,
        bodosql_nullable_numeric_types,
        spark_info,
        check_dtype=False,
        sort_output=False,
    )


def test_orderby_bool(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Tests orderby works in the simple case for boolean types
    """
    query = f"""
        SELECT
             A, B, C
        FROM
            table1
        ORDER BY
            A, B, C
        """
    query2 = f"""
        SELECT
             A, B, C
        FROM
            table1
        ORDER BY
            A, B, C DESC
        """
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        sort_output=False,
        convert_columns_bool=["A", "B", "C"],
    )
    check_query(
        query2,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        sort_output=False,
        convert_columns_bool=["A", "B", "C"],
    )


@pytest.mark.slow
def test_orderby_str(bodosql_string_types, spark_info, memory_leak_check):
    """
    Tests orderby works in the simple case for string types
    Note: We include A to resolve ties.
    """
    query = f"""
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            B, A
        """
    query2 = f"""
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            B DESC, A
        """
    check_query(query, bodosql_string_types, spark_info, sort_output=False)
    check_query(query2, bodosql_string_types, spark_info, sort_output=False)


def test_orderby_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """
    Tests orderby works in the simple case for binary types
    """
    query = f"""
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            B, A DESC
        """
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        convert_columns_bytearray=["A", "B", "C"],
        sort_output=False,
    )


def test_orderby_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """
    Tests orderby works in the simple case for datetime types
    """
    query1 = f"""
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            A, B, C
        """
    query2 = f"""
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            A, B, C DESC
        """
    check_query(query1, bodosql_datetime_types, spark_info, sort_output=False)
    check_query(query2, bodosql_datetime_types, spark_info, sort_output=False)


def test_orderby_interval(bodosql_interval_types, spark_info, memory_leak_check):
    """
    Tests orderby works in the simple case for timedelta types
    """
    query1 = f"""
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            A
        """
    query2 = f"""
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        ORDER BY
            A DESC
        """
    check_query(
        query1,
        bodosql_interval_types,
        spark_info,
        convert_columns_timedelta=["A", "B", "C"],
    )
    check_query(
        query2,
        bodosql_interval_types,
        spark_info,
        convert_columns_timedelta=["A", "B", "C"],
    )


@pytest.mark.slow
def test_distinct_orderby(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests orderby and distinct work together as intended
    """
    query = f"""
        SELECT
            distinct A, B
        FROM
            table1
        ORDER BY
            A
        """
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, sort_output=False
    )


@pytest.mark.slow
def test_orderby_multiple_cols(col_a_identical_tables, spark_info, memory_leak_check):
    """
    checks that orderby works correctly when sorting by multiple columns
    """
    query = f"""
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A, B
        """
    check_query(
        query, col_a_identical_tables, spark_info, check_dtype=False, sort_output=False
    )


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {"A": [1, 2, None] * 8, "B": [1, 2, 3, None] * 6}, dtype="Int64"
            )
        },
    ]
)
def null_ordering_table(request):
    """
    Tables where null ordering impacts the sort output.
    """
    return request.param


def test_orderby_nulls_defaults(null_ordering_table, spark_info, memory_leak_check):
    """
    checks that order by null ordering matches the spark
    defaults
    """
    query = f"""
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A, B
        """
    check_query(
        query, null_ordering_table, spark_info, check_dtype=False, sort_output=False
    )


@pytest.mark.slow
def test_orderby_nulls_defaults_asc(null_ordering_table, spark_info, memory_leak_check):
    """
    checks that order by null ordering matches the spark
    defaults for ASC
    """
    query = f"""
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A ASC, B
        """
    check_query(
        query, null_ordering_table, spark_info, check_dtype=False, sort_output=False
    )


@pytest.mark.slow
def test_orderby_nulls_defaults_desc(
    null_ordering_table, spark_info, memory_leak_check
):
    """
    checks that order by null ordering matches the spark
    defaults for DESC
    """
    query = f"""
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A DESC, B
    """
    check_query(
        query, null_ordering_table, spark_info, check_dtype=False, sort_output=False
    )


@pytest.mark.slow
def test_orderby_nulls_first(null_ordering_table, spark_info, memory_leak_check):
    """
    checks that order by null ordering matches with nulls first
    """
    query = f"""
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A nulls first, B nulls first
    """
    check_query(
        query,
        null_ordering_table,
        spark_info,
        check_dtype=False,
        sort_output=False,
        convert_float_nan=True,
    )


@pytest.mark.slow
def test_orderby_nulls_last(null_ordering_table, spark_info, memory_leak_check):
    """
    checks that order by null ordering matches with nulls last
    """
    query = f"""
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A nulls last, B nulls last
    """
    check_query(
        query,
        null_ordering_table,
        spark_info,
        check_dtype=False,
        sort_output=False,
        convert_float_nan=True,
    )


def test_orderby_nulls_first_last(null_ordering_table, spark_info, memory_leak_check):
    """
    checks that order by null ordering matches with nulls first and last
    """
    query = f"""
        SELECT
            A, B
        FROM
            table1
        ORDER BY
            A nulls first, B nulls last
    """
    check_query(
        query,
        null_ordering_table,
        spark_info,
        check_dtype=False,
        sort_output=False,
        convert_float_nan=True,
    )


def test_orderby_tz_aware(representative_tz, memory_leak_check):
    """
    Test various ORDER BY operations on tz-aware data
    """
    df = pd.DataFrame(
        {
            "A": [
                pd.Timestamp("2022-1-1", tz=representative_tz),
                pd.Timestamp("2022-1-3", tz=representative_tz),
                None,
                pd.Timestamp("2023-11-13", tz=representative_tz),
                pd.Timestamp("2021-11-4", tz=representative_tz),
            ]
            * 2,
            "B": [
                None,
                pd.Timestamp("2021-1-1", tz=representative_tz),
                pd.Timestamp("2021-1-2", tz=representative_tz),
                pd.Timestamp("2024-1-1", tz=representative_tz),
                pd.Timestamp("2024-1-3", tz=representative_tz),
                pd.Timestamp("2019-1-1", tz=representative_tz),
                pd.Timestamp("2023-1-3", tz=representative_tz),
                pd.Timestamp("2015-1-1", tz=representative_tz),
                pd.Timestamp("2011-1-3", tz=representative_tz),
                None,
            ],
            # This is just a Data Column
            "C": pd.date_range(
                "2022/1/1", freq="13T", periods=10, tz=representative_tz
            ),
        }
    )
    ctx = {"table1": df}
    # NOTE: Pandas doesn't support using different NULLS FIRST or NULLS LAST
    # per column, so we will manually convert NULL to a different type.
    large_timestamp = pd.Timestamp("2050-1-1", tz=representative_tz)
    small_timestamp = pd.Timestamp("1970-1-1", tz=representative_tz)

    query1 = f"""
        Select *
        FROM table1
        ORDER BY
            A DESC nulls last,
            B ASC nulls first
    """
    py_output = df.fillna(small_timestamp)
    py_output = py_output.sort_values(["A", "B"], ascending=[False, True])
    # Reset small_timestamp to None
    col_a_nas = py_output["A"] == small_timestamp
    col_b_nas = py_output["B"] == small_timestamp
    py_output["A"][col_a_nas] = None
    py_output["B"][col_b_nas] = None
    check_query(query1, ctx, None, sort_output=False, expected_output=py_output)

    query2 = f"""
        Select *
        FROM table1
        ORDER BY
            A ASC nulls last,
            B DESC nulls first
    """
    py_output = df.fillna(large_timestamp)
    py_output = py_output.sort_values(["A", "B"], ascending=[True, False])
    # Reset small_timestamp to None
    col_a_nas = py_output["A"] == large_timestamp
    col_b_nas = py_output["B"] == large_timestamp
    py_output["A"][col_a_nas] = None
    py_output["B"][col_b_nas] = None
    check_query(query2, ctx, None, sort_output=False, expected_output=py_output)
