# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests I/O error checking for SQL"""
# TODO: Move error checking tests from test_sql to here.

import random
import re
import string

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    oracle_user_pass_and_hostname,
    sql_user_pass_and_hostname,
)
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_read_sql_error_sqlalchemy():
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
def test_unsupported_query():
    """Test error checking for unsupported queries"""

    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"

    def impl(conn):
        sql_request = "CREATE TABLE test(id int, fname varchar(255))"
        frame = pd.read_sql(sql_request, conn)
        return frame

    with pytest.raises(BodoError, match="query is not supported"):
        bodo.jit(impl)(conn)


def test_to_sql_oracle():
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
    conn = "oracle+cx_oracle://" + oracle_user_pass_and_hostname + "/ORACLE"
    with pytest.raises(ValueError, match=re.escape("error in to_sql() operation")):
        bodo.jit(test_impl_write_sql)(df_in, table_name, conn)
