# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Test correctness of SQL aggregation operations without groupby on BodoSQL
"""
import numpy as np
import pytest

from bodosql.tests.utils import check_query


def test_agg_numeric(
    bodosql_numeric_types, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test agg func calls in queries"""

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B), {numeric_agg_builtin_funcs}(C) from table1"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_aliasing_agg_numeric(
    bodosql_numeric_types, numeric_agg_builtin_funcs, spark_info, memory_leak_check
):
    """test aliasing of aggregations in queries"""

    # bitwise aggregate function only valid on integers
    if numeric_agg_builtin_funcs in {"BIT_XOR", "BIT_OR", "BIT_AND"}:
        if not np.issubdtype(bodosql_numeric_types["table1"]["A"].dtype, np.integer):
            return

    query = f"select {numeric_agg_builtin_funcs}(B) as testCol from table1"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_repeat_columns(basic_df, spark_info, memory_leak_check):
    """
    Tests that a column that won't produce a conflicting name
    even if it performs the same operation.
    """
    query = "Select sum(A), sum(A) as alias from table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_count_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """test various count queries on numeric data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_count_nullable_numeric(
    bodosql_nullable_numeric_types, spark_info, memory_leak_check
):
    """test various count queries on nullable numeric data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_nullable_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_count_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """test various count queries on Timestamp data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_datetime_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_count_interval(bodosql_interval_types, spark_info, memory_leak_check):
    """test various count queries on Timedelta data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_interval_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_interval_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_count_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """test various count queries on boolean data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_count_string(bodosql_string_types, spark_info, memory_leak_check):
    """test various count queries on string data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_string_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_count_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """test various count queries on string data."""
    check_query(
        "SELECT COUNT(Distinct B) FROM table1",
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) FROM table1",
        bodosql_binary_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_count_numeric_alias(bodosql_numeric_types, spark_info, memory_leak_check):
    """test various count queries on numeric data with aliases."""
    check_query(
        "SELECT COUNT(Distinct B) as alias FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
    check_query(
        "SELECT COUNT(*) as alias FROM table1",
        bodosql_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-81]")
def test_max_string(bodosql_string_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on string types
    """
    query = """
        SELECT
            max(A)
        FROM
            table1
        """
    check_query(query, bodosql_string_types, spark_info, check_names=False)


def test_max_datetime_types(bodosql_datetime_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on datetime types
    """
    query = """
        SELECT
            max(A)
        FROM
            table1
        """
    check_query(query, bodosql_datetime_types, spark_info, check_names=False)


@pytest.mark.slow
def test_max_interval_types(bodosql_interval_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on timedelta types
    """
    query = """
        SELECT
            max(A) as output
        FROM
            table1
        """
    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_names=False,
        convert_columns_timedelta=["output"],
    )


@pytest.mark.slow
def test_max_literal(basic_df, spark_info, memory_leak_check):
    """tests that max works on a scalar value"""
    # This query does not get optimized, by manual check
    query = "Select A, scalar_max from table1, (Select Max(1) as scalar_max)"

    check_query(
        query,
        basic_df,
        spark_info,
        # Max outputs a nullable output by default to handle empty Series values
        check_dtype=False,
    )


def test_having(bodosql_numeric_types, comparison_ops, spark_info, memory_leak_check):
    """
    Tests having with a constant
    """
    query = f"""
        SELECT
           MAX(A)
        FROM
            table1
        HAVING
            max(B) {comparison_ops} 1
        """
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_max_bool(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Simple test to ensure that max is working on boolean types
    """
    query = """
        SELECT
            max(A)
        FROM
            table1
        """
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_having_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """
    Tests having with a constant
    """
    query = f"""
        SELECT
           MAX(A)
        FROM
            table1
        HAVING
            max(B) <> True
        """
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
