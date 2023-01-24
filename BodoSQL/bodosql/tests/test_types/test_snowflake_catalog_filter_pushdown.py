# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests filter pushdown with a Snowflake Catalog object.
"""
import io
import os

import bodosql
import pandas as pd
import pytest
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    test_db_snowflake_catalog,
)

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    DistTestPipeline,
    check_func,
    create_snowflake_table,
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
            os.environ["SF_USERNAME"],
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
            os.environ["SF_USERNAME"],
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
def test_snowflake_catalog_just_limit_pushdown(memory_leak_check):
    """
    Test limit pushdown with loading from a table from a Snowflake catalog.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(
        catalog=bodosql.SnowflakeCatalog(
            os.environ["SF_USERNAME"],
            os.environ["SF_PASSWORD"],
            "bodopartner.us-east-1",
            "DEMO_WH",
            "TEST_DB",
        )
    )

    query = "select * from PUBLIC.BODOSQL_ALL_SUPPORTED LIMIT 5"

    # make sure limit pushdown worked. Note we don't test correctness because it
    # is undefined.
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
        bodo_func(bc, query)
        fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
        assert hasattr(fir, "meta_head_only_info")
        assert fir.meta_head_only_info[0] is not None
        check_logger_msg(stream, "Columns loaded ['mycol', 'mycol2']")
        check_logger_msg(stream, "Constant limit detected, reading at most 5 rows")


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
            os.environ["SF_USERNAME"],
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


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_like_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that queries with like perform filter pushdown for all the
    cases with the optimized paths. This is tested both with and without
    escapes.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["afewf", "b%bf", "hello", "happy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        # equality test
        query1 = f"Select a from {table_name} where a like 'hello'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a == "hello"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # startswith test
        query2 = f"Select a from {table_name} where a like 'he%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.startswith("he")]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # endswith test
        query3 = f"Select a from {table_name} where a like '%lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.endswith("lo")]
            check_func(
                impl,
                (bc, query3),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # contains test
        query4 = f"Select a from {table_name} where a like '%e%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.contains("e")]
            check_func(
                impl,
                (bc, query4),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # Equality with escape test
        query5 = f"Select a from {table_name} where a like 'b^%bf' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a == "b%bf"]
            check_func(
                impl,
                (bc, query5),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # startswith with escape test
        query6 = f"Select a from {table_name} where a like 'b^%%' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.startswith("b%")]
            check_func(
                impl,
                (bc, query6),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # endswith with escape test
        query7 = f"Select a from {table_name} where a like '%^%bf' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.endswith("%bf")]
            check_func(
                impl,
                (bc, query7),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # contains with escape test
        query8 = f"Select a from {table_name} where a like '%^%%' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.contains("%")]
            check_func(
                impl,
                (bc, query8),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # Always true test
        query9 = f"Select a from {table_name} where a like '%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output
            check_func(
                impl,
                (bc, query9),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_ilike_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that queries with ilike perform filter pushdown for all the
    cases with the optimized paths. This is tested both with and without
    escapes.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["AFewf", "b%Bf", "hELlO", "HAPPy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        # equality test
        query1 = f"Select a from {table_name} where a ilike 'hello'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.lower() == "hello"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # startswith test
        query2 = f"Select a from {table_name} where a ilike 'He%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.lower().str.startswith("he")]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # endswith test
        query3 = f"Select a from {table_name} where a ilike '%lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.lower().str.endswith("lo")]
            check_func(
                impl,
                (bc, query3),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # contains test
        query4 = f"Select a from {table_name} where a ilike '%e%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.lower().str.contains("e")]
            check_func(
                impl,
                (bc, query4),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # Equality with escape test
        query5 = f"Select a from {table_name} where a ilike 'b^%bf' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.lower() == "b%bf"]
            check_func(
                impl,
                (bc, query5),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # startswith with escape test
        query6 = f"Select a from {table_name} where a ilike 'B^%%' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.lower().str.startswith("b%")]
            check_func(
                impl,
                (bc, query6),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # endswith with escape test
        query7 = f"Select a from {table_name} where a ilike '%^%bf' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.lower().str.endswith("%bf")]
            check_func(
                impl,
                (bc, query7),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # contains with escape test
        query8 = f"Select a from {table_name} where a ilike '%^%b%' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.lower().str.contains("%b")]
            check_func(
                impl,
                (bc, query8),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_like_regex_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that queries with LIKE that perform filter pushdown where the pattern
    cannot be simplified to avoid regular expression matching. This filter pushdown path
    requires pushing the original LIKE into Snowflake as opposed to a simpler expression.
    This is tested both with and without escapes.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame({"A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10})

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_regex_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        # equality test
        query1 = f"Select a from {table_name} where a like 'hE_lO'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a == "hELlO"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query2 = f"Select a from {table_name} where a like 'b^%_f' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a == "b%Bf"]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_ilike_regex_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that queries with ILIKE that perform filter pushdown where the pattern
    cannot be simplified to avoid regular expression matching. This filter pushdown path
    requires pushing the original ILIKE into Snowflake as opposed to a simpler expression.
    This is tested both with and without escapes.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame({"A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10})

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "ilike_regex_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        # equality test
        query1 = f"Select a from {table_name} where a ilike 'he_lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a == "hELlO"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query2 = f"Select a from {table_name} where a ilike 'b^%_F' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a == "b%Bf"]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "Columns loaded ['a']")
