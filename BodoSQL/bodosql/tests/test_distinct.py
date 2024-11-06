# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL queries containing distinct on BodoSQL
"""

import pandas as pd
import pytest

from bodosql.tests.utils import check_query


def test_distinct_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests distinct works in the simple case for numeric types
    """
    query = """
        SELECT
            DISTINCT A
        FROM
            table1
        """
    check_query(query, bodosql_numeric_types, spark_info, check_dtype=False)


@pytest.mark.slow
def test_distinct_numeric_scalars(basic_df, spark_info, memory_leak_check):
    """Tests that distinct works in the case for scalars (in that it should have no effect)"""
    query = """
        SELECT
            DISTINCT 1 as A, 1 as B, 1 as C, 3 as D
        FROM
            table1
        """
    check_query(query, basic_df, spark_info)


def test_distinct_bool(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Tests distinct works in the simple case for boolean types
    """
    query = """
        SELECT
            DISTINCT A
        FROM
            table1
        """
    check_query(query, bodosql_boolean_types, spark_info, check_dtype=False)


def test_distinct_str(bodosql_string_types, spark_info, memory_leak_check):
    """
    Tests distinct works in the simple case for string types
    """
    query = """
        SELECT
            DISTINCT A
        FROM
            table1
        """
    check_query(query, bodosql_string_types, spark_info)


def test_distinct_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """
    Tests distinct works in the simple case for binary types
    """
    query = """
        SELECT
            DISTINCT A
        FROM
            table1
        """
    check_query(query, bodosql_binary_types, spark_info)


def test_distinct_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """
    Tests distinct works in the simple case for datetime
    """
    query = """
        SELECT
            DISTINCT A
        FROM
            table1
        """
    check_query(query, bodosql_datetime_types, spark_info, use_duckdb=True)


def test_distinct_interval(bodosql_interval_types, memory_leak_check):
    """
    Tests distinct works in the simple case for timedelta
    """
    query = """
        SELECT
            DISTINCT A
        FROM
            table1
        """
    check_query(
        query,
        bodosql_interval_types,
        None,
        expected_output=pd.DataFrame(
            {"A": bodosql_interval_types["TABLE1"]["A"].unique()}
        ),
    )


@pytest.mark.slow
def test_distinct_within_table(join_dataframes, spark_info, memory_leak_check):
    """
    Tests distinct works in the case where we are selecting multiple columns from the same table
    """

    if any(
        isinstance(
            x,
            (
                pd.core.arrays.integer.IntegerDtype,
                pd.Float32Dtype,
                pd.Float64Dtype,
            ),
        )
        for x in join_dataframes["TABLE1"].dtypes
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
        for colname in join_dataframes["TABLE1"].columns
    ):
        convert_columns_bytearray = ["A", "B", "C"]
    else:
        convert_columns_bytearray = None
    query = """
        SELECT
            DISTINCT A, B, C
        FROM
            table1
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_distinct_where_numeric(
    bodosql_numeric_types,
    comparison_ops,
    spark_info,
    # memory_leak_check Failing memory leak check on nightly, see [BS-534]
):
    """
    Test that distinct works with where restrictions
    """
    query = f"""
        SELECT
            DISTINCT A, B
        FROM
            table1
        WHERE
            A {comparison_ops} 1
        """
    check_query(query, bodosql_numeric_types, spark_info, check_dtype=False)


def test_distinct_where_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Test that distinct works with where restrictions for booleans
    """
    query = """
        SELECT
            DISTINCT A, B
        FROM
            table1
        WHERE
            A = TRUE
        """
    check_query(query, bodosql_boolean_types, spark_info, check_dtype=False)


@pytest.fixture(
    params=[
        # Testing null vs. non-null comparisons:
        # https://docs.snowflake.com/en/sql-reference/functions/is-distinct-from#usage-notes
        pd.DataFrame(
            {
                "A": [None, None, "x", "x"] * 3,
                "B": [None, "x", None, "x"] * 3,
            }
        ),
        pd.DataFrame(
            {
                "A": [None, None, "x", "x"] * 3,
                "B": "x",
            }
        ),
    ]
)
def is_distinct_from_null_dfs(request):
    return request.param


def test_is_distinct_from_nulls(
    is_distinct_from_null_dfs, spark_info, memory_leak_check
):
    """
    Test that IS DISTINCT FROM works with null columns/scalars
    """
    query = "SELECT A IS DISTINCT FROM B FROM table1"
    ctx = {"TABLE1": is_distinct_from_null_dfs}
    check_query(query, ctx, spark_info, check_dtype=False, check_names=False)


def test_is_distinct_from_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Test IS DISTINCT FROM for numeric types
    """
    query = "SELECT A IS DISTINCT FROM B FROM table1"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


def test_is_distinct_from_numeric_scalars(
    bodosql_numeric_types, spark_info, memory_leak_check
):
    """
    Test IS DISTINCT FROM for numeric scalar types
    """
    query = "SELECT A IS DISTINCT FROM 1 FROM table1"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


def test_is_not_distinct_from_datetime(
    bodosql_datetime_types, spark_info, memory_leak_check
):
    """
    Test IS NOT DISTINCT FROM for datetime types
    """
    query = "SELECT A IS NOT DISTINCT FROM B FROM table1"
    check_query(
        query, bodosql_datetime_types, spark_info, check_dtype=False, check_names=False
    )


def test_is_not_distinct_from_date(bodosql_date_types, spark_info, memory_leak_check):
    """
    Test IS NOT DISTINCT FROM for date types
    """
    query = "SELECT A IS NOT DISTINCT FROM B FROM table1"
    check_query(
        query, bodosql_date_types, spark_info, check_dtype=False, check_names=False
    )


def test_is_distinct_from_interval(bodosql_interval_types, memory_leak_check):
    """
    Test IS DISTINCT FROM for interval types
    """
    query = "SELECT A IS DISTINCT FROM B FROM table1"
    df = bodosql_interval_types["TABLE1"]
    check_query(
        query,
        bodosql_interval_types,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"A": df.A != df.B}),
    )


def test_is_distinct_from_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Test IS DISTINCT FROM for boolean types
    """
    query = "SELECT A IS DISTINCT FROM B FROM table1"
    check_query(
        query, bodosql_boolean_types, spark_info, check_dtype=False, check_names=False
    )


def test_is_distinct_from_string(bodosql_string_types, spark_info, memory_leak_check):
    """
    Test IS DISTINCT FROM for string types
    """
    query = "SELECT A IS DISTINCT FROM B FROM table1"
    check_query(
        query, bodosql_string_types, spark_info, check_dtype=False, check_names=False
    )


def test_is_distinct_from_nullable_numeric(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """
    Test IS DISTINCT FROM for nullable_numeric types
    """
    query = "SELECT A IS DISTINCT FROM B FROM table1"
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_is_distinct_from_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """
    Test IS DISTINCT FROM for binary types
    """
    query = "SELECT A IS DISTINCT FROM B FROM table1"
    check_query(
        query, bodosql_binary_types, spark_info, check_dtype=False, check_names=False
    )


def test_is_distinct_from_where_string(
    bodosql_string_types, spark_info, memory_leak_check
):
    """
    Test IS DISTINCT FROM in the WHERE condition for string types
    """
    query = "SELECT DISTINCT A, B FROM table1 WHERE A IS DISTINCT FROM B"
    check_query(
        query, bodosql_string_types, spark_info, check_dtype=False, check_names=False
    )


def test_is_distinct_from_case_nullable_numeric(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """
    Test IS DISTINCT FROM in a CASE expression for nullable numeric types
    """
    query = (
        "SELECT CASE WHEN A IS DISTINCT FROM B "
        "  THEN A IS DISTINCT FROM C "
        "  ELSE B IS NOT DISTINCT FROM C END "
        "FROM table1"
    )
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_distinct_null(basic_df, spark_info, memory_leak_check):
    """
    Test DISTINCT with a null array
    """
    query = "SELECT DISTINCT A, NULL as B from table1"
    check_query(query, basic_df, spark_info, check_dtype=False)
