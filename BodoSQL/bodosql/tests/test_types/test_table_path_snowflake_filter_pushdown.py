# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests filter pushdown with a Snowflake SQL TablePath.
"""
import io
import os

import bodosql
import pandas as pd
import pytest

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func, get_snowflake_connection_string


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_simple_filter_pushdown(memory_leak_check):
    def impl1(table_name):
        bc = bodosql.BodoSQLContext(
            {"sql_table": bodosql.TablePath(table_name, "sql", conn_str=conn_str)},
        )
        return bc.sql(
            "select L_SUPPKEY from sql_table WHERE L_ORDERKEY > 10 AND 2 >= L_LINENUMBER ORDER BY L_SUPPKEY LIMIT 70",
        )

    def impl2(table_name):
        bc = bodosql.BodoSQLContext(
            {"sql_table": bodosql.TablePath(table_name, "sql", conn_str=conn_str)},
        )
        return bc.sql(
            "SELECT l_suppkey from sql_table where l_shipmode LIKE 'AIR%%' OR l_orderkey = 32",
        )

    table_name = "lineitem"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn_str = get_snowflake_connection_string(db, schema)
    py_output = pd.read_sql(
        f"select L_SUPPKEY from {table_name} ORDER BY L_SUPPKEY LIMIT 70", conn_str
    )
    # Names won't match because BodoSQL keep names uppercase.
    py_output.columns = ["L_SUPPKEY"]
    # Set check_dtype=False because BodoSQL loads columns with the actual Snowflake type
    check_func(
        impl1,
        (table_name,),
        py_output=py_output,
        reset_index=True,
        check_dtype=False,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl1)(table_name)
        # Check for pruned columns
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    py_output = pd.read_sql(
        f"SELECT l_suppkey from {table_name} where l_shipmode LIKE 'AIR%%' OR l_orderkey = 32",
        conn_str,
    )
    # Set check_dtype=False because BodoSQL loads columns with the actual Snowflake type
    check_func(
        impl2,
        (table_name,),
        py_output=py_output,
        reset_index=True,
        check_dtype=False,
        sort_output=True,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl2)(table_name)
        # Check for pruned columns
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")
