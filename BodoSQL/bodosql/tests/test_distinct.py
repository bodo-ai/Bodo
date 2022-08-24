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
    query = f"""
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
    query = f"""
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
    query = f"""
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
    query = f"""
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
    query = f"""
        SELECT
            DISTINCT A
        FROM
            table1
        """
    check_query(query, bodosql_datetime_types, spark_info)


def test_distinct_interval(bodosql_interval_types, spark_info, memory_leak_check):
    """
    Tests distinct works in the simple case for timedelta
    """
    query = f"""
        SELECT
            DISTINCT A
        FROM
            table1
        """
    check_query(
        query, bodosql_interval_types, spark_info, convert_columns_timedelta=["A"]
    )


@pytest.mark.slow
def test_distinct_within_table(join_dataframes, spark_info):
    """
    Tests distinct works in the case where we are selecting multiple columns from the same table
    """
    # TODO: Enable memory leak check
    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["A", "B", "C"]
    else:
        convert_columns_bytearray = None
    query = f"""
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
    query = f"""
        SELECT
            DISTINCT A, B
        FROM
            table1
        WHERE
            A = TRUE
        """
    check_query(query, bodosql_boolean_types, spark_info, check_dtype=False)
