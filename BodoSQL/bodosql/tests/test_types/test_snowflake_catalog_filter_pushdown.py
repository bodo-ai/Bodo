# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests filter pushdown with a Snowflake Catalog object.
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
from bodo.tests.utils import (
    DistTestPipeline,
    check_func,
    get_snowflake_connection_string,
)


@pytest.fixture(
    params=[
        pytest.param(
            "select L_SUPPKEY from TPCH_SF1.LINEITEM WHERE L_ORDERKEY > 10 AND L_SHIPMODE LIKE 'AIR%%' ORDER BY L_SUPPKEY LIMIT 70",
            marks=pytest.mark.slow,
        ),
    ]
)
def simple_queries(request):
    return request.param


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_catalog_simple_filter_pushdown(memory_leak_check):
    def impl(bc, query):
        return bc.sql(query)

    def test_impl(bc, query, conn_str, col_name):
        py_output = pd.read_sql(query, conn_str)
        py_output.columns = py_output.columns.str.upper()

        check_func(
            impl,
            (
                bc,
                query,
            ),
            py_output=py_output,
            reset_index=True,
            check_dtype=False,
        )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(impl)(bc, query)
            # Check for pruned columns
            check_logger_msg(stream, f"Columns loaded ['{col_name}']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")

    bc = bodosql.BodoSQLContext(
        catalog=bodosql.SnowflakeCatalog(
            os.environ["SF_USER"],
            os.environ["SF_PASSWORD"],
            "bodopartner.us-east-1",
            "DEMO_WH",
            "SNOWFLAKE_SAMPLE_DATA",
        )
    )

    query = "select L_SUPPKEY from TPCH_SF1.LINEITEM WHERE L_ORDERKEY > 10 AND L_SHIPMODE LIKE 'AIR%%' ORDER BY L_SUPPKEY LIMIT 70"

    conn_str = get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1")

    test_impl(bc, query, conn_str, "l_suppkey")


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_catalog_zero_columns_pruning(memory_leak_check):
    """
    Test loading just a length from a table in a Snowflake Catalog.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(
        catalog=bodosql.SnowflakeCatalog(
            os.environ["SF_USER"],
            os.environ["SF_PASSWORD"],
            "bodopartner.us-east-1",
            "DEMO_WH",
            "SNOWFLAKE_SAMPLE_DATA",
        )
    )

    query = "SELECT COUNT(*) as cnt from TPCH_SF1.LINEITEM"

    py_output = pd.read_sql(
        query,
        get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1"),
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl)(bc, query)
        check_logger_msg(stream, "Columns loaded []")

    check_func(impl, (bc, query), py_output=py_output, is_out_distributed=False)


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_catalog_limit_pushdown(memory_leak_check):
    """
    Test limit pushdown with loading from a table from a Snowflake catalog.
    """

    def impl(bc, query):
        return bc.sql(query)

    # Note: We don't yet support casting with limit pushdown.
    py_output = pd.read_sql(
        f"select mycol from PUBLIC.BODOSQL_ALL_SUPPORTED WHERE mycol = 'A' LIMIT 5",
        get_snowflake_connection_string("TEST_DB", "PUBLIC"),
    )

    bc = bodosql.BodoSQLContext(
        catalog=bodosql.SnowflakeCatalog(
            os.environ["SF_USER"],
            os.environ["SF_PASSWORD"],
            "bodopartner.us-east-1",
            "DEMO_WH",
            "TEST_DB",
        )
    )

    query = "select mycol from PUBLIC.BODOSQL_ALL_SUPPORTED WHERE mycol = 'A' LIMIT 5"
    check_func(impl, (bc, query), py_output=py_output)

    # make sure filter + limit pushdown worked
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
        bodo_func(bc, query)
        fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
        assert hasattr(fir, "meta_head_only_info")
        assert fir.meta_head_only_info[0] is not None
        check_logger_msg(stream, "Columns loaded ['mycol']")
        check_logger_msg(stream, "Filter pushdown successfully performed")
