# Copyright (C) 2019 Bodo Inc. All rights reserved.
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
import pymysql
import pandas as pd

import operator
import pandas as pd
import numpy as np
import random
import pytest

import numba
import bodo
from bodo.tests.utils import check_func, get_start_end

np.random.seed(5)
random.seed(5)


def test_write_sql_aws():
    """This test for a write down on a SQL database"""

    def test_impl(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace")

    def test_specific_dataframe(test_impl, is_distributed, df_in):
        table_name = "test_table_ABCD"
        conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
        bodo_impl = bodo.jit(all_args_distributed_block=is_distributed)(test_impl)
        if is_distributed:
            start, end = get_start_end(len(df_in))
            df_input = df_in.iloc[start:end]
        else:
            df_input = df_in
        bodo_impl(df_input, table_name, conn)
        if bodo.get_rank() == 0:
            df_load = pd.read_sql("select * from " + table_name, conn)
            # The writing does not preserve the order a priori
            l_cols = df_in.columns.to_list()
            df_in_sort = df_in.sort_values(l_cols).reset_index(drop=True)
            df_load_sort = df_load[l_cols].sort_values(l_cols).reset_index(drop=True)
            pd.testing.assert_frame_equal(df_load_sort, df_in_sort)

    len_list = 20
    list_int = list(np.random.randint(1, 10, len_list))
    list_double = [
        4.0 if random.randint(1, 3) == 1 else np.nan for _ in range(len_list)
    ]
    list_datetime = pd.date_range("2001-01-01", periods=len_list)
    df1 = pd.DataFrame({"A": list_int, "B": list_double, "C": list_datetime})
    test_specific_dataframe(test_impl, False, df1)
    test_specific_dataframe(test_impl, True, df1)


def test_sql_if_exists_fail_errorchecking():
    """This test with the option if_exists="fail" (which is the default)
    The database must alredy exist (which should be ok if above test is done)
    It will fail because the database is already present."""

    def test_impl(df, table_name, conn):
        df.to_sql(table_name, conn)

    n_pes = bodo.libs.distributed_api.get_size()
    len_list = 20
    list_int = list(np.random.randint(1, 10, len_list))
    list_double = [
        4.0 if random.randint(1, 3) == 1 else np.nan for _ in range(len_list)
    ]
    list_datetime = pd.date_range("2001-01-01", periods=len_list)
    df1 = pd.DataFrame({"A": list_int, "B": list_double, "C": list_datetime})
    table_name = "test_table_ABCD"
    conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
    bodo_impl = bodo.jit(all_args_distributed_block=True)(test_impl)
    #    with pytest.raises(ValueError, match="Table .* already exists"):
    with pytest.raises(ValueError, match="error in to_sql.* operation"):
        bodo_impl(df1, table_name, conn)


def test_sql_hardcoded_aws():
    """This test for an hardcoded request and connection"""

    def test_impl():
        sql_request = "select * from employees"
        conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl, ())


def test_read_sql_hardcoded_time_offset_aws():
    """This test does not pass because the type of dates is not supported"""

    def test_impl():
        sql_request = "select * from employees limit 1000 offset 4000"
        conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl, ())


def test_read_sql_hardcoded_twocol_aws():
    """Selecting two columns without dates"""

    def test_impl():
        sql_request = "select first_name,last_name from employees"
        conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    bodo_impl = bodo.jit(test_impl)
    frame = bodo_impl()


def test_sql_argument_passing():
    """Test passing SQL query and connection as arguments
    """

    def test_impl(sql_request, conn):
        df = pd.read_sql(sql_request, conn)
        return df

    sql_request = "select * from employees"
    conn = "mysql+pymysql://admin:Bodosql2#@bodosqldb.cutkvh6do3qv.us-east-1.rds.amazonaws.com/employees"
    check_func(test_impl, (sql_request, conn))
