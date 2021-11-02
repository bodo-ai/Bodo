# Copyright (C) 2019 Bodo Inc. All rights reserved.
import datetime
import os
import random

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import (
    SeriesOptTestPipeline,
    _check_for_io_reader_filters,
    check_func,
    get_start_end,
    _check_connector_columns,
)

sql_user_pass_and_hostname = (
    "user:pass@localhost"
)


def test_write_sql_aws(memory_leak_check):
    """This test for a write down on a SQL database"""

    def test_impl_write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace")

    def test_specific_dataframe(test_impl, is_distributed, df_in):
        table_name = "test_table_ABCD"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        bodo_impl = bodo.jit(all_args_distributed_block=is_distributed)(test_impl)
        if is_distributed:
            start, end = get_start_end(len(df_in))
            df_input = df_in.iloc[start:end]
        else:
            df_input = df_in
        bodo_impl(df_input, table_name, conn)
        bodo.barrier()
        if bodo.get_rank() == 0:
            df_load = pd.read_sql("select * from " + table_name, conn)
            # The writing does not preserve the order a priori
            l_cols = df_in.columns.to_list()
            df_in_sort = df_in.sort_values(l_cols).reset_index(drop=True)
            df_load_sort = df_load[l_cols].sort_values(l_cols).reset_index(drop=True)
            pd.testing.assert_frame_equal(df_load_sort, df_in_sort)

    np.random.seed(5)
    random.seed(5)
    len_list = 20
    list_int = list(np.random.randint(1, 10, len_list))
    list_double = [
        4.0 if random.randint(1, 3) == 1 else np.nan for _ in range(len_list)
    ]
    list_datetime = pd.date_range("2001-01-01", periods=len_list)
    df1 = pd.DataFrame({"A": list_int, "B": list_double, "C": list_datetime})
    test_specific_dataframe(test_impl_write_sql, False, df1)
    test_specific_dataframe(test_impl_write_sql, True, df1)


# TODO: Add memory_leak_check when bug is resolved.
def test_sql_if_exists_fail_errorchecking():
    """This test with the option if_exists="fail" (which is the default)
    The database must alredy exist (which should be ok if above test is done)
    It will fail because the database is already present."""

    def test_impl_fails(df, table_name, conn):
        df.to_sql(table_name, conn)

    np.random.seed(5)
    random.seed(5)
    n_pes = bodo.libs.distributed_api.get_size()
    len_list = 20
    list_int = list(np.random.randint(1, 10, len_list))
    list_double = [
        4.0 if random.randint(1, 3) == 1 else np.nan for _ in range(len_list)
    ]
    list_datetime = pd.date_range("2001-01-01", periods=len_list)
    df1 = pd.DataFrame({"A": list_int, "B": list_double, "C": list_datetime})
    table_name = "test_table_ABCD"
    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
    bodo_impl = bodo.jit(all_args_distributed_block=True)(test_impl_fails)
    #    with pytest.raises(ValueError, match="Table .* already exists"):
    with pytest.raises(ValueError, match="error in to_sql.* operation"):
        bodo_impl(df1, table_name, conn)


def test_sql_hardcoded_aws(memory_leak_check):
    """This test for an hardcoded request and connection"""

    def test_impl_hardcoded():
        sql_request = "select * from employees"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl_hardcoded, ())


def test_read_sql_hardcoded_time_offset_aws(memory_leak_check):
    """This test does not pass because the type of dates is not supported"""

    def test_impl_offset():
        sql_request = "select * from employees limit 1000 offset 4000"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl_offset, ())


def test_read_sql_hardcoded_limit_aws(memory_leak_check):
    def test_impl_limit():
        sql_request = "select * from employees limit 1000"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl_limit, ())


def test_sql_limit_inference():
    f = bodo.ir.sql_ext.req_limit
    # Simple query
    sql_request = "select * from employees limit 1000"
    assert f(sql_request) == 1000
    # White space
    sql_request = "select * from employees limit 1000   "
    assert f(sql_request) == 1000
    # White space
    sql_request = "select * from employees limit 1000 \n\n\n\n  "
    assert f(sql_request) == 1000

    # Check that we do not match with an offset
    sql_request = "select * from employees limit 1, 1000"
    assert f(sql_request) is None
    sql_request = "select * from employees limit 1000 offset 1"
    assert f(sql_request) is None

    # Check that we select the right limit in a nested query
    sql_request = "with table1 as (select * from employees limit 1000) select * from table1 limit 500"
    assert f(sql_request) == 500

    # Check that we don't select an inner limit
    sql_request = "select * from employees, (select table1.A, table2.B from table1 FULL OUTER join table2 on table1.A = table2.A limit 1000)"
    assert f(sql_request) is None


def test_limit_inferrence_small_table(memory_leak_check):
    """
    Checks that a query where the limit is much larger than the size
    of the table succeeds. We create a very small table and then set
    the limit to be much greater than the table size.
    """

    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
    table_name = "test_small_table"

    df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
    # Create the table once.
    if bodo.get_rank() == 0:
        df.to_sql(table_name, conn, if_exists="replace")
    bodo.barrier()

    def test_impl_limit():
        """
        Test receiving the table with limit 1000 while there are only
        10 rows.
        """
        sql_request = "select A from test_small_table limit 1000"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl_limit, ())


def test_sql_single_column(memory_leak_check):
    """
    Test that loading using a single column has a correct result.
    This can break if dead column elimination is applied incorrectly.
    """

    def write_sql(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace")

    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
    table_name = "test_small_table"

    df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5, "C": [31, 247] * 5})
    # Create the table once.
    if bodo.get_rank() == 0:
        write_sql(df, table_name, conn)
    bodo.barrier()

    def test_impl1():
        sql_request = "select A, B, C from test_small_table"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame["B"]

    def test_impl2():
        sql_request = "select * from test_small_table"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame["B"]

    check_func(test_impl1, (), check_dtype=False)
    check_func(test_impl2, (), check_dtype=False)


def test_read_sql_hardcoded_twocol_aws(memory_leak_check):
    """Selecting two columns without dates"""

    def test_impl_hardcoded_twocol():
        sql_request = "select first_name,last_name from employees"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_func(test_impl_hardcoded_twocol, ())


def test_sql_argument_passing(memory_leak_check):
    """Test passing SQL query and connection as arguments"""

    def test_impl_arg_passing(sql_request, conn):
        df = pd.read_sql(sql_request, conn)
        return df

    sql_request = "select * from employees"
    conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
    check_func(test_impl_arg_passing, (sql_request, conn))


# We only run snowflake tests on Azure Pipelines because the Snowflake account credentials
# are stored there (to avoid failing on AWS or our local machines)
def get_snowflake_connection_string(db, schema):
    """
    Generates a common snowflake connection string. Some details (how to determine
    username and password) seem unlikely to change, whereas as some tests could require
    other details (db and schema) to change.
    """
    username = os.environ["SF_USER"]
    password = os.environ["SF_PASSWORD"]
    account = "bodopartner.us-east-1"
    warehouse = "DEMO_WH"
    conn = f"snowflake://{username}:{password}@{account}/{db}/{schema}?warehouse={warehouse}"
    return conn


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake(memory_leak_check):
    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    check_func(impl, (query, conn))


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_single_column(memory_leak_check):
    """
    Test that loading using a single column from snowflake has a correct result
    that reduces the number of columns that need loading.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df["l_suppkey"]

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(query, conn)
    _check_connector_columns(bodo_func, ["l_suppkey"], bodo.ir.sql_ext.SqlReader)


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_count(memory_leak_check):
    """
    Test that using a sql function without an alias doesn't cause issues with
    dead column elimination.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        # TODO: Pandas loads count(*) as COUNT(*) but we can't detect this difference
        # and load it as count(*)
        df.columns = [x.lower() for x in df.columns]
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT L_ORDERKEY, count(*) FROM LINEITEM GROUP BY L_ORDERKEY ORDER BY L_ORDERKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_filter_pushdown(memory_leak_check):
    """
    Test that filter pushdown works properly with a variety of data types.
    """

    def impl_integer(query, conn, int_val):
        df = pd.read_sql(query, conn)
        df = df[(df["l_orderkey"] > 10) & (int_val >= df["l_linenumber"])]
        return df["l_suppkey"]

    def impl_string(query, conn, str_val):
        df = pd.read_sql(query, conn)
        df = df[(df["l_linestatus"] != str_val) | (df["l_shipmode"] == "FOB")]
        return df["l_suppkey"]

    def impl_date(query, conn, date_val):
        df = pd.read_sql(query, conn)
        df = df[date_val > df["l_shipdate"]]
        return df["l_suppkey"]

    def impl_timestamp(query, conn, ts_val):
        df = pd.read_sql(query, conn)
        # Note when comparing to date Pandas will truncate the timestamp.
        # This comparison is deprecated in general.
        df = df[ts_val <= df["l_shipdate"]]
        return df["l_suppkey"]

    def impl_mixed(query, conn, int_val, str_val, date_val, ts_val):
        """
        Test a query with mixed parameter types.
        """
        df = pd.read_sql(query, conn)
        df = df[
            ((df["l_linenumber"] <= int_val) | (date_val > df["l_shipdate"]))
            | ((ts_val <= df["l_shipdate"]) & (df["l_linestatus"] != str_val))
        ]
        return df["l_suppkey"]

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    # Reset index because Pandas applies the filter later

    int_val = 2
    check_func(
        impl_integer, (query, conn, int_val), check_dtype=False, reset_index=True
    )
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl_integer)
    bodo_func(query, conn, int_val)
    _check_for_io_reader_filters(bodo_func, bodo.ir.sql_ext.SqlReader)
    _check_connector_columns(
        bodo_func,
        [
            "l_suppkey",
        ],
        bodo.ir.sql_ext.SqlReader
    )

    str_val = "O"
    check_func(impl_string, (query, conn, str_val), check_dtype=False, reset_index=True)
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl_string)
    bodo_func(query, conn, str_val)
    _check_for_io_reader_filters(bodo_func, bodo.ir.sql_ext.SqlReader)
    _check_connector_columns(bodo_func, ["l_suppkey"], bodo.ir.sql_ext.SqlReader)

    date_val = datetime.date(1996, 4, 12)
    check_func(impl_date, (query, conn, date_val), check_dtype=False, reset_index=True)
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl_date)
    bodo_func(query, conn, date_val)
    _check_for_io_reader_filters(bodo_func, bodo.ir.sql_ext.SqlReader)
    _check_connector_columns(bodo_func, ["l_suppkey"], bodo.ir.sql_ext.SqlReader)

    ts_val = pd.Timestamp(year=1997, month=4, day=12)
    check_func(
        impl_timestamp, (query, conn, ts_val), check_dtype=False, reset_index=True
    )
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl_timestamp)
    bodo_func(query, conn, ts_val)
    _check_for_io_reader_filters(bodo_func, bodo.ir.sql_ext.SqlReader)
    _check_connector_columns(bodo_func, ["l_suppkey"], bodo.ir.sql_ext.SqlReader)

    check_func(
        impl_mixed,
        (query, conn, int_val, str_val, date_val, ts_val),
        check_dtype=False,
        reset_index=True,
    )
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl_mixed)
    bodo_func(query, conn, int_val, str_val, date_val, ts_val)
    _check_for_io_reader_filters(bodo_func, bodo.ir.sql_ext.SqlReader)
    _check_connector_columns(bodo_func, ["l_suppkey"], bodo.ir.sql_ext.SqlReader)
