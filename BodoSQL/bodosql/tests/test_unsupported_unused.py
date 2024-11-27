"""
Tests that check passing an unsupported type to BodoSQL in various
contexts where the column can be pruned by Bodo
"""

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.utils import (
    check_func,
    get_snowflake_connection_string,
    pytest_snowflake,
)

pytestmark = pytest_snowflake


@pytest.mark.skip("[BSE-1637] Updated unused columns for semi-structured data")
def test_snowflake_read_sql_unused(memory_leak_check):
    """
    Tests pruning Snowflake columns when passing the DataFrame
    with pd.read_sql().
    """

    def test_impl(query, conn):
        df = pd.read_sql(query, conn)
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select usedcol from table1")

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # Add arrow_number_to_decimal to create a decimal array
    conn = f"{conn}&arrow_number_to_decimal=True"
    # Note: This table was manually created inside Snowflake. The
    # relevant columns are
    # USEDCOL: NUMBER(38, 0)
    # BADCOL: NUMBER(25,20)

    # Note that we have to use decimal scale >= 18 for badcol, due to
    # a conflicting pr that uses arrow_number_to_decimal
    # see: https://bodo.atlassian.net/browse/BE-4168
    query = "SELECT * FROM UNSUPPORTED2"
    py_output = pd.read_sql(query, conn)[["usedcol"]]
    # Warnings are only raised on rank 0
    if bodo.get_rank() == 0:
        with pytest.warns(bodosql.utils.BodoSQLWarning):
            check_func(
                test_impl,
                (query, conn),
                py_output=py_output,
                check_dtype=False,
                reset_index=True,
                sort_output=True,
            )
    else:
        check_func(
            test_impl,
            (query, conn),
            py_output=py_output,
            check_dtype=False,
            reset_index=True,
            sort_output=True,
        )


@pytest.mark.skip("[BSE-1637] Updated unused columns for semi-structured data")
def test_snowflake_table_path_unused(memory_leak_check):
    """
    Tests pruning Snowflake columns when passing the DataFrame
    with tablePath.
    """

    def test_impl(conn):
        bc = bodosql.BodoSQLContext(
            {"TABLE1": bodosql.TablePath("UNSUPPORTED2", "sql", conn_str=conn)}
        )
        return bc.sql("select usedcol from table1")

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # Add arrow_number_to_decimal to create a decimal array
    conn = f"{conn}&arrow_number_to_decimal=True"
    query = "SELECT * FROM UNSUPPORTED2"
    py_output = pd.read_sql(query, conn)[["usedcol"]]
    # Warnings are only raised on rank 0
    if bodo.get_rank() == 0:
        with pytest.warns(bodosql.utils.BodoSQLWarning):
            check_func(
                test_impl,
                (conn,),
                py_output=py_output,
                check_dtype=False,
                reset_index=True,
                sort_output=True,
            )
    else:
        check_func(
            test_impl,
            (conn,),
            py_output=py_output,
            check_dtype=False,
            reset_index=True,
            sort_output=True,
        )


@pytest.mark.skip("[BSE-1637] Updated unused columns for semi-structured data")
def test_snowflake_table_path_unused_subquery(memory_leak_check):
    """
    Tests pruning Snowflake columns when passing the DataFrame
    with tablePath. This tests a subquery to check for optimizations.
    """

    def test_impl(conn):
        bc = bodosql.BodoSQLContext(
            {"TABLE1": bodosql.TablePath("UNSUPPORTED2", "sql", conn_str=conn)}
        )
        return bc.sql("select usedcol from (select * from table1 where usedcol = 1)")

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT * FROM UNSUPPORTED2"
    py_output = pd.read_sql(query, conn)[["usedcol"]]
    py_output = py_output[py_output.usedcol == 1]
    # Warnings are only raised on rank 0
    if bodo.get_rank() == 0:
        with pytest.warns(bodosql.utils.BodoSQLWarning):
            check_func(
                test_impl,
                (conn,),
                py_output=py_output,
                check_dtype=False,
                reset_index=True,
                sort_output=True,
            )
    else:
        check_func(
            test_impl,
            (conn,),
            py_output=py_output,
            check_dtype=False,
            reset_index=True,
            sort_output=True,
        )
