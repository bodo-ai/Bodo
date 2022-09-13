# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests various components of the SnowflakeCatalog type both inside and outside a
direct BodoSQLContext.
"""

import os

import bodosql
import pandas as pd
import pytest

from bodo.tests.utils import check_func, get_snowflake_connection_string


@pytest.fixture(
    params=[
        bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
        ),
        bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            {"role": "USERADMIN"},
        ),
    ]
)
def dummy_snowflake_catalogs(request):
    """
    List of table paths that should be suppported.
    None of these actually point to valid data
    """
    return request.param


def test_snowflake_catalog_lower_constant(dummy_snowflake_catalogs, memory_leak_check):
    """
    Test lowering a constant snowflake catalog.
    """

    def impl():
        return dummy_snowflake_catalogs

    check_func(impl, ())


def test_snowflake_catalog_boxing(dummy_snowflake_catalogs, memory_leak_check):
    """
    Test boxing and unboxing a table path type.
    """

    def impl(snowflake_catalog):
        return snowflake_catalog

    check_func(impl, (dummy_snowflake_catalogs,))


def test_snowflake_catalog_constructor(memory_leak_check):
    """
    Test using the table path constructor from JIT.
    """

    def impl1():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
        )

    def impl2():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            # Explicitly pass None
            None,
        )

    def impl3():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            {"role": "USERADMIN"},
        )

    def impl4():
        return bodosql.SnowflakeCatalog(
            "myusername",
            "mypassword",
            "myaccount",
            "mywarehouse",
            "mydatabase",
            # Try passing an empty dictionary
            {},
        )

    check_func(impl1, ())
    check_func(impl2, ())
    check_func(impl3, ())
    # Note: Empty dictionary passed via args or literal map not supported yet.
    # [BE-3455]
    # check_func(impl4, ())


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_catalog_read(memory_leak_check):
    def impl(bc):
        return bc.sql("SELECT r_name FROM TPCH_SF1.REGION ORDER BY r_name")

    bodo_connect_params = {"query_tag": "folder=folder1+ folder2&"}
    catalog = bodosql.SnowflakeCatalog(
        os.environ["SF_USER"],
        os.environ["SF_PASSWORD"],
        "bodopartner.us-east-1",
        "DEMO_WH",
        "SNOWFLAKE_SAMPLE_DATA",
        connection_params=bodo_connect_params,
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)
    py_output = pd.read_sql(
        "Select r_name from REGION ORDER BY r_name",
        get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1"),
    )
    check_func(impl, (bc,), py_output=py_output)


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_catalog_read_tpch(memory_leak_check):
    tpch_query = """select
                      n_name,
                      sum(l_extendedprice * (1 - l_discount)) as revenue
                    from
                      tpch_sf1.customer,
                      tpch_sf1.orders,
                      tpch_sf1.lineitem,
                      tpch_sf1.supplier,
                      tpch_sf1.nation,
                      tpch_sf1.region
                    where
                      c_custkey = o_custkey
                      and l_orderkey = o_orderkey
                      and l_suppkey = s_suppkey
                      and c_nationkey = s_nationkey
                      and s_nationkey = n_nationkey
                      and n_regionkey = r_regionkey
                      and r_name = 'ASIA'
                      and o_orderdate >= '1994-01-01'
                      and o_orderdate < '1995-01-01'
                    group by
                      n_name
                    order by
                      revenue desc
    """

    def impl(bc):
        return bc.sql(tpch_query)

    catalog = bodosql.SnowflakeCatalog(
        os.environ["SF_USER"],
        os.environ["SF_PASSWORD"],
        "bodopartner.us-east-1",
        "DEMO_WH",
        "SNOWFLAKE_SAMPLE_DATA",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)
    py_output = pd.read_sql(
        tpch_query,
        get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1"),
    )

    check_func(impl, (bc,), py_output=py_output, reset_index=True)
