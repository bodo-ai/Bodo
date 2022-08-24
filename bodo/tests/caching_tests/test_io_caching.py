# Copyright (C) 2022 Bodo Inc. All rights reserved.
import os

import pandas as pd
import pytest
from caching_tests_common import fn_distribution  # noqa

import bodo
from bodo.tests.utils import (
    check_caching,
    get_snowflake_connection_string,
    sql_user_pass_and_hostname,
)


@bodo.jit
def read_pq_func1(fname):
    """
    Helper func to test caching with a
    read in another function
    """
    return pd.read_parquet(fname)


@bodo.jit
def read_pq_func2(fname):
    """
    Helper func to test caching with a
    read in another function
    """
    return pd.read_parquet(fname)


@pytest.mark.smoke
def test_read_csv_cache(fn_distribution, is_cached, datapath, memory_leak_check):
    """
    test read_csv with cache=True
    """
    fname = datapath("csv_data1.csv")

    def impl():
        df = pd.read_csv(fname, names=["A", "B", "C", "D"], compression=None)
        return df.C

    check_caching(impl, (), is_cached, fn_distribution)


@pytest.mark.smoke
def test_read_parquet_cache(fn_distribution, is_cached, datapath, memory_leak_check):
    """
    test read_parquet with cache=True
    """

    def impl(fname):
        return pd.read_parquet(fname)

    fname = datapath("groupby3.pq")
    check_caching(impl, (fname,), is_cached, fn_distribution)


def test_read_parquet_cache_fname_arg(
    fn_distribution, is_cached, datapath, memory_leak_check
):
    """
    test read_parquet with cache=True and passing different file name as
    argument to the Bodo function
    """

    def impl(fname):
        return pd.read_parquet(fname)

    if is_cached:
        fname = datapath("int_nulls_multi.pq")
    else:
        fname = datapath("int_nulls_single.pq")

    py_out = impl(fname)
    check_caching(impl, (fname,), is_cached, fn_distribution, py_output=py_out)


def test_read_parquet_cache_fname_arg2(
    fn_distribution, is_cached, datapath, memory_leak_check
):
    """
    test read_parquet with cache=True and passing different file name as
    argument to the Bodo function where the read is in separate function.
    """

    def impl(fname):
        df1 = read_pq_func1(fname)
        df2 = read_pq_func2(fname)
        print(df1.info())
        print(df2.info())
        return df1

    if is_cached:
        fname = datapath("int_nulls_multi.pq")
    else:
        fname = datapath("int_nulls_single.pq")

    # Use py_output because we use the jit decorator directly
    py_out = pd.read_parquet(fname)
    check_caching(impl, (fname,), is_cached, fn_distribution, py_output=py_out)


def test_read_parquet_cache_fname_arg_list_files(
    fn_distribution, is_cached, datapath, memory_leak_check
):
    """
    test read_parquet with cache=True and passing different list of files as
    argument to the Bodo function
    """

    def impl(fpaths):
        return pd.read_parquet(fpaths)

    if not is_cached:
        fpaths = [datapath("example.parquet"), datapath("example2.parquet")]
    else:
        fpaths = [datapath("example2.parquet"), datapath("example.parquet")]
    py_output_part1 = pd.read_parquet(fpaths[0])
    py_output_part2 = pd.read_parquet(fpaths[1])
    py_out = pd.concat([py_output_part1, py_output_part2])
    check_caching(impl, (fpaths,), is_cached, fn_distribution, py_output=py_out)


def test_read_csv_cache_fname_arg(
    fn_distribution, is_cached, datapath, memory_leak_check
):
    """
    test read_csv with cache=True and passing different file name as
    argument to the Bodo function
    """

    def impl(fname):
        return pd.read_csv(fname)

    def impl2(fname):
        return pd.read_csv(fname)

    fname1 = datapath("example.csv")
    fname2 = datapath("example_multi.csv")  # directory of csv files

    py_out = impl(fname1)
    check_caching(impl, (fname1,), is_cached, fn_distribution, py_output=py_out)
    check_caching(impl2, (fname2,), is_cached, fn_distribution, py_output=py_out)


def test_read_json_cache_fname_arg(
    fn_distribution, is_cached, datapath, memory_leak_check
):
    """
    test read_json with cache=True and passing different file name as
    argument to the Bodo function
    """

    def impl(fname):
        return pd.read_json(fname, orient="records", lines=True)

    def impl2(fname):
        return pd.read_json(fname, orient="records", lines=True)

    fname1 = datapath("example.json")
    fname2 = datapath("example_single.json")  # directory with one json file

    py_out = impl(fname1)
    check_caching(impl, (fname1,), is_cached, fn_distribution, py_output=py_out)
    check_caching(impl2, (fname2,), is_cached, fn_distribution, py_output=py_out)


def test_cache_sql_hardcoded_aws(fn_distribution, is_cached, memory_leak_check):
    """This test caching a hardcoded request and connection"""

    def test_impl_hardcoded():
        sql_request = "select * from employees"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame

    check_caching(test_impl_hardcoded, (), is_cached, fn_distribution)


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_cache_sql_snowflake(fn_distribution, is_cached, memory_leak_check):
    """
    Tests that caching works on Snowflake queries
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    check_caching(impl, (query, conn), is_cached, fn_distribution)
