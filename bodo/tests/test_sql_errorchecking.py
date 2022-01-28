# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests I/O error checking for SQL
"""
# TODO: Move error checking tests from test_sql to here.

import os

import pandas as pd
import pytest

import bodo

from .test_sql import (
    get_snowflake_connection_string,
    sql_user_pass_and_hostname,
)


@pytest.mark.slow
def test_read_sql_error_sqlalchemy(memory_leak_check):
    """This test for incorrect credentials and SQL sentence with sqlalchemy"""

    def test_impl_sql_err():
        sql_request = "select * from invalid"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    with pytest.raises(RuntimeError, match="Error executing query"):
        bodo.jit(test_impl_sql_err)()

    def test_impl_credentials_err():
        sql_request = "select * from employess"
        conn = "mysql+pymysql://unknown_user/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    with pytest.raises(RuntimeError, match="Error executing query"):
        bodo.jit(test_impl_credentials_err)()


@pytest.mark.slow
@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_read_sql_error_snowflake(memory_leak_check):
    """This test for incorrect credentials and SQL sentence with snowflake"""

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    def test_impl_sql_err(conn):
        sql_request = "select * from invalid"
        frame = pd.read_sql(sql_request, conn)
        return frame

    with pytest.raises(RuntimeError, match="Error executing query"):
        bodo.jit(test_impl_sql_err)(conn)

    def test_impl_credentials_err(conn):
        sql_request = "select * from LINEITEM LIMIT 10"
        frame = pd.read_sql(sql_request, conn)
        return frame

    account = "bodopartner.us-east-1"
    warehouse = "DEMO_WH"
    conn = f"snowflake://unknown:wrong@{account}/{db}/{schema}?warehouse={warehouse}"
    with pytest.raises(RuntimeError, match="Error executing query"):
        bodo.jit(test_impl_credentials_err)(conn)
