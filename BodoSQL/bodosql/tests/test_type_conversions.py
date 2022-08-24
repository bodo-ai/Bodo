# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL casting
"""
import pandas as pd
import pytest
from bodosql.tests.utils import bodo_version_older, check_query


def test_simple_cast(basic_df, spark_info, memory_leak_check):
    """
    Checks that integer casting of constants behaves as expected
    """
    query = "SELECT CAST(1.0 AS integer)"
    check_query(query, basic_df, spark_info, check_names=False)


@pytest.mark.slow
def test_float_to_int_cast(
    basic_df, numeric_values, sql_numeric_typestrings, spark_info, memory_leak_check
):
    """
    Checks that numeric casting of constants behaves as expected
    """
    # SparkSQL outputs Decimal type as object type
    if sql_numeric_typestrings == "DECIMAL":
        check_dtype = False
    else:
        check_dtype = True
    # Spark converts this to 0, but Bodo uses Decimal and Double interchangably
    if sql_numeric_typestrings == "DECIMAL" and numeric_values == 0.001:
        return
    query = f"SELECT CAST({numeric_values} AS {sql_numeric_typestrings})"
    check_query(query, basic_df, spark_info, check_names=False, check_dtype=check_dtype)


def test_numeric_column_casting(
    bodosql_numeric_types, sql_numeric_typestrings, spark_info, memory_leak_check
):
    """
    Checks that casting numeric columns behaves as expected
    """
    query = f"""
    SELECT
        CAST(A AS {sql_numeric_typestrings})
    FROM
        table1
    """
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.skip("Cast from dt64 to other numeric types not supported in Pandas")
def test_datetime_numeric_column_casting(
    bodosql_datetime_types, sql_numeric_typestrings, spark_info, memory_leak_check
):
    """
    Checks that casting datetime to numeric columns behaves as expected
    """
    query = f"""
    SELECT
        CAST(A AS {sql_numeric_typestrings})
    FROM
        table1
    """
    check_query(query, bodosql_datetime_types, spark_info, check_dtype=False)


@pytest.mark.skip("[BS-151] Cast Not supported in our MySQL Dialect")
def test_interval_numeric_column_casting(
    bodosql_interval_types, sql_numeric_typestrings, spark_info, memory_leak_check
):
    """
    Checks that casting interval to numeric columns behaves as expected
    """
    query = f"""
    SELECT
        CAST(A AS {sql_numeric_typestrings})
    FROM
        table1
    """
    check_query(query, bodosql_interval_types, spark_info, check_dtype=False)


@pytest.mark.skipif(
    bodo_version_older(2021, 7, 1),
    reason="Requires next mini-release for engine changes to support casting",
)
def test_varchar_to_numeric_cast(
    sql_numeric_typestrings, spark_info, memory_leak_check
):
    """
    Checks that casting strings to numeric values behaves as expected
    """
    query = f"""
    SELECT
        CAST(A AS {sql_numeric_typestrings})
    FROM
        table1
    """
    str_data = {
        "A": ["1", "2", "3"] * 4,
    }
    ctx = {"table1": pd.DataFrame(str_data)}
    check_query(query, ctx, spark_info, check_dtype=False, check_names=False)


@pytest.mark.skipif(
    bodo_version_older(2021, 9, 2),
    reason="Requires next mini-release for engine changes to support Bodo argument to astype",
)
def test_numeric_to_varchar_nullable(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """
    Checks that casting strings to numeric values behaves as expected
    """
    query = f"""
    SELECT
        CAST(A AS VARCHAR) as col
    FROM
        table1
    """
    spark_query = f"""
    SELECT
        CAST(A AS STRING) as col
    FROM
        table1
    """
    check_query(
        query,
        bodosql_nullable_numeric_types,
        spark_info,
        equivalent_spark_query=spark_query,
    )
