"""
Test correctness of SQL casting
"""

import pandas as pd
import pytest

from bodosql.tests.conftest import fixture_value_not_in, mark_bodosql_cpp_if
from bodosql.tests.utils import check_query


@pytest.mark.bodosql_cpp
def test_simple_cast(basic_df, memory_leak_check):
    """
    Checks that integer casting of constants behaves as expected
    """
    query = "SELECT CAST(1.0 AS integer)"
    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        use_duckdb=True,
    )


@mark_bodosql_cpp_if(fixture_value_not_in("sql_numeric_typestrings", {"DECIMAL"}))
@pytest.mark.slow
def test_float_to_int_cast(
    basic_df, numeric_values, sql_numeric_typestrings, memory_leak_check
):
    """
    Checks that numeric casting of constants behaves as expected
    """
    # Spark converts this to 0, but Bodo uses Decimal and Double interchangeably
    if sql_numeric_typestrings == "DECIMAL" and numeric_values == 0.001:
        return
    query = f"SELECT CAST({numeric_values} AS {sql_numeric_typestrings})"
    # check_dtype=False since Bodo returns nullable columns by default
    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        use_duckdb=True,
    )


@mark_bodosql_cpp_if(fixture_value_not_in("sql_numeric_typestrings", {"DECIMAL"}))
def test_numeric_column_casting(
    bodosql_numeric_types, sql_numeric_typestrings, memory_leak_check
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
        query,
        bodosql_numeric_types,
        None,
        check_dtype=False,
        check_names=False,
        use_duckdb=True,
    )


@pytest.mark.skip("Cast from dt64 to other numeric types not supported in Pandas")
def test_datetime_numeric_column_casting(
    bodosql_datetime_types, sql_numeric_typestrings, memory_leak_check
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
    check_query(
        query,
        bodosql_datetime_types,
        None,
        check_dtype=False,
        use_duckdb=True,
    )


@pytest.mark.skip("[BS-151] Cast Not supported in our MySQL Dialect")
def test_interval_numeric_column_casting(
    bodosql_interval_types, sql_numeric_typestrings, memory_leak_check
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
    check_query(
        query,
        bodosql_interval_types,
        None,
        check_dtype=False,
        use_duckdb=True,
    )


@mark_bodosql_cpp_if(fixture_value_not_in("sql_numeric_typestrings", {"DECIMAL"}))
@pytest.mark.slow
def test_varchar_to_numeric_cast(sql_numeric_typestrings, memory_leak_check):
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
    ctx = {"TABLE1": pd.DataFrame(str_data)}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        use_duckdb=True,
    )


@pytest.mark.bodosql_cpp
def test_numeric_to_varchar_nullable(bodosql_nullable_numeric_types, memory_leak_check):
    """
    Checks that casting numeric values to strings behaves as expected
    """
    query = """
    SELECT
        CAST(A AS VARCHAR) as col
    FROM
        table1
    """
    check_query(
        query,
        bodosql_nullable_numeric_types,
        None,
        expected_output=pd.DataFrame(
            {
                "COL": bodosql_nullable_numeric_types["TABLE1"]["A"]
                .astype(str)
                .replace("<NA>", None)
                .replace("<NA", None)
            }
        ),
    )
