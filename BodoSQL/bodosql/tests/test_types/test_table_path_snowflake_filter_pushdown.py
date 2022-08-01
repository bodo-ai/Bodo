# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests filter pushdown with a Snowflake SQL TablePath.
"""
import io
import os

import bodo
import pandas as pd
import pytest
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func

import bodosql
from bodosql.tests.utils import (
    bodo_version_older,
    get_snowflake_connection_string,
)


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ or bodo_version_older(2022, 2, 1),
    reason="requires Azure Pipelines",
)
def test_simple_filter_pushdown(memory_leak_check):
    def impl(table_name):
        bc = bodosql.BodoSQLContext(
            {"sql_table": bodosql.TablePath(table_name, "sql", conn_str=conn_str)},
        )
        return bc.sql(
            "select L_SUPPKEY from sql_table WHERE L_ORDERKEY > 10 AND 2 >= L_LINENUMBER ORDER BY L_SUPPKEY LIMIT 70",
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
        impl,
        (table_name,),
        py_output=py_output,
        reset_index=True,
        check_dtype=False,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(table_name)
        # Check for pruned columns
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")
