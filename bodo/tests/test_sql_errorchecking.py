# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests I/O error checking for SQL
"""
# TODO: Move error checking tests from test_sql to here.

import os
import random
import re
import string

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    get_snowflake_connection_string,
    oracle_user_pass_and_hostname,
    sql_user_pass_and_hostname,
)
from bodo.utils.typing import BodoError, BodoWarning


@pytest.mark.slow
def test_read_sql_error_sqlalchemy(memory_leak_check):
    """This test for incorrect credentials and SQL sentence with sqlalchemy"""

    def test_impl_sql_err():
        sql_request = "select * from invalid"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    with pytest.raises(RuntimeError, match="Table 'employees.invalid' doesn't exist"):
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


@pytest.mark.slow
@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_to_sql_schema_warning(memory_leak_check):
    """[BE-2117] This test for not using schema with snowflake"""

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    def impl(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
    tablename = "schema_warning_table"
    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="schema argument is recommended"):
            bodo.jit(impl)(df, tablename, conn)
    else:
        bodo.jit(impl)(df, tablename, conn)


@pytest.mark.slow
def test_unsupported_query(memory_leak_check):
    """Test error checking for unsupported queries"""

    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"

    def impl(conn):
        sql_request = "CREATE TABLE test(id int, fname varchar(255))"
        frame = pd.read_sql(sql_request, conn)
        return frame

    with pytest.raises(BodoError, match="query is not supported"):
        bodo.jit(impl)(conn)


def test_to_sql_oracle(memory_leak_check):
    """This test that runtime error message for Oracle with string > 4000
    is displayed"""

    def test_impl_write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, index=False, if_exists="replace")

    np.random.seed(5)
    random.seed(5)
    len_list = 1
    letters = string.ascii_letters
    list_string = [
        "".join(random.choice(letters) for i in range(4002)) for _ in range(len_list)
    ]
    df_in = pd.DataFrame(
        {
            "AB": list_string,
        }
    )
    table_name = "to_sql_table"
    df_input = df_in
    conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"
    with pytest.raises(ValueError, match=re.escape("error in to_sql() operation")):
        bodo.jit(test_impl_write_sql)(df_in, table_name, conn)
