# Copyright (C) 2022 Bodo Inc. All rights reserved.

"""
Tests that check passing an unsupported type to BodoSQL in various
contexts where the column can be pruned by Bodo
"""
import os

import bodosql
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func, get_snowflake_connection_string


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_read_sql_unused(memory_leak_check):
    """
    Tests pruning Snowflake columns when passing the DataFrame
    with pd.read_sql().
    """

    def test_impl(query, conn):
        df = pd.read_sql(query, conn)
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select usedcol from table1")

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # Add arrow_number_to_decimal to create a decimal array
    conn = f"{conn}&arrow_number_to_decimal=True"
    # Note: This table was manually created inside Snowflake. The
    # relevant columns are
    # USEDCOL: NUMBER(38, 0)
    # BADCOL: NUMBER(25, 1)
    query = "SELECT * FROM UNSUPPORTED"
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


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_table_path_unused(memory_leak_check):
    """
    Tests pruning Snowflake columns when passing the DataFrame
    with tablePath.
    """

    def test_impl(conn):
        bc = bodosql.BodoSQLContext(
            {"table1": bodosql.TablePath("UNSUPPORTED", "sql", conn_str=conn)}
        )
        return bc.sql("select usedcol from table1")

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # Add arrow_number_to_decimal to create a decimal array
    conn = f"{conn}&arrow_number_to_decimal=True"
    query = "SELECT * FROM UNSUPPORTED"
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


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_table_path_unused_subquery(memory_leak_check):
    """
    Tests pruning Snowflake columns when passing the DataFrame
    with tablePath. This tests a subquery to check for optimizations.
    """

    def test_impl(conn):
        bc = bodosql.BodoSQLContext(
            {"table1": bodosql.TablePath("UNSUPPORTED", "sql", conn_str=conn)}
        )
        return bc.sql("select usedcol from (select * from table1 where usedcol = 1)")

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # Add arrow_number_to_decimal to create a decimal array
    conn = f"{conn}&arrow_number_to_decimal=True"
    query = "SELECT * FROM UNSUPPORTED"
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
