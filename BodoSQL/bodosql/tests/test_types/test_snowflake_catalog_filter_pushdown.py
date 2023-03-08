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
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    DistTestPipeline,
    check_func,
    create_snowflake_table,
    get_snowflake_connection_string,
    pytest_snowflake,
)

pytestmark = pytest_snowflake


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


def test_snowflake_catalog_coalesce_pushdown(memory_leak_check):
    """
    Test filter pushdown with with coalesce on a column.
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
            connection_params={"schema": "PUBLIC"},
        )
    )

    query = "select * from BODOSQL_ALL_SUPPORTED where coalesce(mycol2, current_date()) > '2022-01-01'"

    py_output = pd.read_sql(
        query,
        get_snowflake_connection_string("TEST_DB", "PUBLIC"),
    )

    # make sure filter pushdown worked
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl, (bc, query), py_output=py_output, sort_output=True, reset_index=True
        )
        check_logger_msg(stream, "Columns loaded ['mycol', 'mycol2']")
        check_logger_msg(stream, "Filter pushdown successfully performed")
        check_logger_msg(stream, r"WHERE  ( ( coalesce(\"MYCOL2\", {f0}) > {f1} ) )")


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
@pytest.mark.parametrize(
    "func_args",
    [
        pytest.param(("lower", "lower"), id="lower"),
        pytest.param(("upper", "upper"), id="upper"),
        pytest.param(
            ("initcap", "capitalize"),
            id="capitalize",
            marks=pytest.mark.skip(
                "[BE-4445] Additional arg in filter pushdown not supported"
            ),
        ),
    ],
)
def test_snowflake_case_conversion_filter_pushdown(
    test_db_snowflake_catalog, func_args, memory_leak_check
):
    """
    Test upper, lower, initcap support in Snowflake filter pushdown
    """
    sql_func, pd_func_name = func_args
    test_str_val = getattr("macedonia", pd_func_name)()

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]

    def impl(bc, query):
        return bc.sql(query)

    df = pd.DataFrame(
        {
            "A": [
                "macedonia",
                "MACEDONIA",
                "MaCedOnIA",
                None,
                "not_macedonia",
                "NOT_MACEDONIA",
                None,
            ]
            * 10,
        }
    )

    with create_snowflake_table(
        df, f"{sql_func}_pushdown_table", db, schema
    ) as table_name:
        query = f"select a from {table_name} where {sql_func}(a) = '{test_str_val}'"

        # make sure filter pushdown worked
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = df[getattr(df.A.str, pd_func_name)() == test_str_val]
            expected_output.columns = [x.lower() for x in expected_output.columns]
            check_func(
                impl,
                (bc, query),
                py_output=expected_output,
                sort_output=True,
                reset_index=True,
                check_dtype=False,
            )
            check_logger_msg(stream, "Columns loaded ['a']")
            check_logger_msg(stream, "Filter pushdown successfully performed")
            check_logger_msg(stream, f"WHERE  ( ( {sql_func}" r"(\"A\") = {f1} ) )")


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_coalesce_lower_pushdown(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test filter pushdown with coalesce and lower on a column.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]

    def impl(bc, query):
        return bc.sql(query)

    new_df = pd.DataFrame(
        {
            "A": ["LoWeR", "LOWER", "lower", None, "lover", None] * 10,
            "B": [100, 1000, 10000, 11, 1321, 10] * 10,
        }
    )

    expected_output = pd.DataFrame({"b": [11, 10] * 10})

    with create_snowflake_table(
        new_df, "coalesce_lower_pushdown_table", db, schema
    ) as table_name:
        query = f"select b from {table_name} where coalesce(lower(a), 'macedonia') = 'macedonia'"

        # make sure filter pushdown worked
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_func(
                impl,
                (bc, query),
                py_output=expected_output,
                sort_output=True,
                reset_index=True,
                check_dtype=False,
            )
            check_logger_msg(stream, "Columns loaded ['b']")
            check_logger_msg(stream, "Filter pushdown successfully performed")
            check_logger_msg(
                stream, r"WHERE  ( ( coalesce(lower(\"A\"), {f1}) = {f2} ) )"
            )


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_upper_coalesce_pushdown(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test filter pushdown with upper and coalesce on a column.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]

    def impl(bc, query):
        return bc.sql(query)

    new_df = pd.DataFrame(
        {
            "A": ["MaCedOniA", "UPPER", "macedonia", None, "MACEDONIA", None] * 10,
            "B": [100, 1000, 10000, 11, 1321, 10] * 10,
        }
    )

    expected_output = pd.DataFrame({"b": [100, 10000, 11, 1321, 10] * 10})

    with create_snowflake_table(
        new_df, "upper_coalesce_pushdown_table", db, schema
    ) as table_name:
        query = f"select b from {table_name} where upper(coalesce(a, 'macedonia')) = 'MACEDONIA'"

        # make sure filter pushdown worked
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_func(
                impl,
                (bc, query),
                py_output=expected_output,
                sort_output=True,
                reset_index=True,
                check_dtype=False,
            )
            check_logger_msg(stream, "Columns loaded ['b']")
            check_logger_msg(stream, "Filter pushdown successfully performed")
            check_logger_msg(
                stream, r"WHERE  ( ( upper(coalesce(\"A\", {f0})) = {f2} ) )"
            )


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_catalog_coalesce_not_pushdown(memory_leak_check):
    """
    Make sure coalesce with two column input is not pushed down since not supported yet
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
            connection_params={"schema": "PUBLIC"},
        )
    )

    query = "select l_commitdate from TPCH_SF1.LINEITEM where coalesce(l_commitdate, l_shipdate) > '1998-10-29'"

    py_output = pd.read_sql(
        query,
        get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "PUBLIC"),
    )

    # make sure filter pushdown was not performed
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl, (bc, query), py_output=py_output, sort_output=True, reset_index=True
        )
        check_logger_no_msg(stream, "Filter pushdown successfully performed")


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
        # Note the only column need is pruned by the planner due to the filter.
        check_logger_msg(stream, "Columns loaded []")
        check_logger_msg(stream, "Filter pushdown successfully performed")


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
        new_df, "ilike_pushdown_table", db, schema
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


def test_snowflake_like_non_constant_pushdown(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that queries with like and non-constant patterns support filter
    pushdown. This is tested both with and without escapes.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame({"A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10})

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_non_constant_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        # equality test
        query1 = f"Select a from {table_name} where a like 'hE_l' || 'O'"
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
        query2 = (
            f"Select a from {table_name} where a like 'b' || '^%_f' escape upper('^')"
        )
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


def test_snowflake_ilike_non_constant_pushdown(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that queries with ilike and non-constant patterns support filter
    pushdown. This is tested both with and without escapes.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame({"A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10})

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "ilike_non_constant_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        # equality test
        query1 = f"Select a from {table_name} where a ilike 'he' || '_lo'"
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
        query2 = (
            f"Select a from {table_name} where a ilike 'b^' || '%_F' escape lower('^')"
        )
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
def test_snowflake_column_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that queries with WHERE using a boolean column supports
    filter pushdown.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {
            "A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10,
            "B": [True, False, False, True] * 10,
            "C": [True, True, True, False] * 10,
        }
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(new_df, "where_column_table", db, schema) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        query1 = f"Select a from {table_name} where b"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.b][["a"]]
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
        query2 = f"Select a from {table_name} where b or c"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.b | py_output.c][["a"]]
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
        query3 = f"Select a from {table_name} where b and c"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.b & py_output.c][["a"]]
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


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_not_column_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that queries with WHERE using NOT with a boolean column support
    filter pushdown. This also tests AND/OR support.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {
            "A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10,
            "B": [True, False, False, True] * 10,
            "C": [True, True, True, False] * 10,
        }
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "where_not_column_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        query1 = f"Select a from {table_name} where not b"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.b][["a"]]
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
        query2 = f"Select a from {table_name} where not(b or c)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.b | py_output.c)][["a"]]
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
        query3 = f"Select a from {table_name} where not(b and c)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.b & py_output.c)][["a"]]
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
        query4 = f"Select a from {table_name} where (not b) or c"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.b | py_output.c][["a"]]
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
        query5 = f"Select a from {table_name} where (not b) and c"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.b & py_output.c][["a"]]
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


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_not_comparison_pushdown(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests filter pushdown queries using NOT with various comparison
    operators.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {
            "A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10,
            "B": [1, 2, 3, 4] * 10,
            "C": [True, True, True, True, False] * 8,
        }
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "where_not_compare_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        query1 = f"Select a from {table_name} where not (b <> 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.b != 3)][["a"]]
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
        query2 = f"Select a from {table_name} where not (b <= 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.b <= 3)][["a"]]
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
        query3 = f"Select a from {table_name} where not (b < 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.b < 3)][["a"]]
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
        query4 = f"Select a from {table_name} where not (b >= 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.b >= 3)][["a"]]
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
        query5 = f"Select a from {table_name} where not (b > 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.b > 3)][["a"]]
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
        query6 = f"Select a from {table_name} where NOT((b > 3 OR b < 2) AND C)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[
                ~(((py_output.b > 3) | (py_output.b < 2)) & py_output.c)
            ][["a"]]
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


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_not_is_null_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests filter pushdown queries using NOT with IS NULL
    operators.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {
            "A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10,
            "B": pd.Series([1, None, 3, None] * 10, dtype="Int64"),
        }
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "where_not_is_null_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        query1 = f"Select a from {table_name} where not (b is NULL)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.b.isna())][["a"]]
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
        query2 = f"Select a from {table_name} where not (b is not NULL)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.b.notna())][["a"]]
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
def test_snowflake_not_in_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests filter pushdown queries using NOT IN.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {
            "A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10,
            "B": pd.Series([1, 2, 3, None] * 10, dtype="Int64"),
        }
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(new_df, "where_not_in_table", db, schema) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        query = f"Select a from {table_name} where b not in (1, 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[
                (~py_output.b.isin([1, 3])) & py_output.b.notna()
            ][["a"]]
            check_func(
                impl,
                (bc, query),
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
def test_snowflake_not_like_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that queries with not like perform filter pushdown for all the
    cases with the optimized paths.
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
        new_df, "not_like_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        # equality test
        query1 = f"Select a from {table_name} where a not like 'hello'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a != "hello"]
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
        query2 = f"Select a from {table_name} where a not like 'he%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.a.str.startswith("he")]
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
        query3 = f"Select a from {table_name} where a not like '%lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.a.str.endswith("lo")]
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
        query4 = f"Select a from {table_name} where a not like '%e%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.a.str.contains("e")]
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


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_not_ilike_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that queries with not ilike perform filter pushdown for all the
    cases with the optimized paths.
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
        new_df, "not_ilike_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        # equality test
        query1 = f"Select a from {table_name} where a not ilike 'hello'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a.str.lower() != "hello"]
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
        query2 = f"Select a from {table_name} where a not ilike 'He%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.a.str.lower().str.startswith("he")]
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
        query3 = f"Select a from {table_name} where a not ilike '%lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.a.str.lower().str.endswith("lo")]
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
        query4 = f"Select a from {table_name} where a not ilike '%e%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.a.str.lower().str.contains("e")]
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


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_not_like_regex_pushdown(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that queries with NOT LIKE that perform filter pushdown where the pattern
    cannot be simplified to avoid regular expression matching.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame({"A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10})

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "not_like_regex_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        # equality test
        query = f"Select a from {table_name} where a not like 'hE_lO'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.a != "hELlO"]
            check_func(
                impl,
                (bc, query),
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
def test_snowflake_length_filter_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that queries with all aliases of LENGTH work with
    filter pushdown.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame({"A": ["AFewf", "bf", None, "cool", "alter"] * 10})

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "length_pushdown_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        expected_output = py_output[py_output.a.str.len() == 4]
        # equality test
        for func_name in ("CHAR_LENGTH", "LENGTH", "LEN", "CHARACTER_LENGTH"):
            query = f"Select a from {table_name} where {func_name}(A) = 4"
            stream = io.StringIO()
            logger = create_string_io_logger(stream)
            with set_logging_stream(logger, 1):
                check_func(
                    impl,
                    (bc, query),
                    py_output=expected_output,
                    reset_index=True,
                    sort_output=True,
                )
                check_logger_msg(
                    stream,
                    "Filter pushdown successfully performed. Moving filter step:",
                )
                check_logger_msg(stream, "Columns loaded ['a']")


@pytest.mark.parametrize(
    "func_args",
    [
        pytest.param(
            ("LTRIM", "lstrip"),
            id="LTRIM",
        ),
        pytest.param(
            ("RTRIM", "rstrip"),
            id="RTRIM",
        ),
        pytest.param(
            ("TRIM", "strip"),
            id="TRIM",
        ),
    ],
)
@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
@pytest.mark.skip("[BE-4445] Extra arg in filter pushdown not supported")
def test_snowflake_trim_filter_pushdown(
    func_args, test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that queries with all variations of TRIM work with
    filter pushdown (no optional chars).
    """
    sql_func_name, pd_func_name = func_args

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    df = pd.DataFrame(
        {
            "L_SHIPMODE": [
                "   SHIP       ",
                "  ship ",
                None,
                "qwerSHIP   ",
                "asdfSHIPSdfasdfasdf",
            ]
            * 10,
            "L_ORDERKEY": [1230, 4100, None, 9999, 0] * 10,
        }
    )

    with create_snowflake_table(df, "trim_pushdown_table", db, schema) as table_name:
        shipmode_col = getattr(df.L_SHIPMODE.str, pd_func_name)()
        expected_output = df[shipmode_col == "SHIP"][["L_ORDERKEY"]]
        expected_output.columns = [x.lower() for x in expected_output.columns]

        test_query = f"""
            select l_orderkey from {table_name}
            WHERE {sql_func_name}({table_name}.l_shipmode) = 'SHIP'
        """
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_func(
                impl,
                (bc, test_query),
                py_output=expected_output,
                is_out_distributed=False,
                reset_index=True,
                sort_output=True,
                # Pandas output is non-nullable
                check_dtype=False,
            )
            check_logger_msg(stream, "Filter pushdown successfully performed.")
            check_logger_msg(stream, "Columns loaded ['l_orderkey']")


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ,
    reason="requires Azure Pipelines",
)
def test_snowflake_reverse_filter_pushdown(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that queries with REVERSE work with
    filter pushdown.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    df = pd.DataFrame(
        {
            "L_SHIPMODE": [
                "   SHIP       ",
                "  ship ",
                " blah",
                "qwerSHIP   ",
                "asdfSHIPSdfasdfasdf",
            ]
            * 10,
            "L_ORDERKEY": [1230, 4100, None, 9999, 0] * 10,
        }
    )

    with create_snowflake_table(df, "reverse_pushdown_table", db, schema) as table_name:
        # Load the whole table in pandas.
        shipmode_col = df.L_SHIPMODE.apply(lambda x: x[::-1])
        expected_output = df[shipmode_col == "PIHS"][["L_ORDERKEY"]]
        expected_output.columns = [x.lower() for x in expected_output.columns]

        test_query = f"""
            select l_orderkey from {table_name}
            WHERE REVERSE({table_name}.l_shipmode) = 'PIHS'
        """
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_func(
                impl,
                (bc, test_query),
                py_output=expected_output,
                is_out_distributed=False,
                reset_index=True,
                sort_output=True,
                # Pandas output is non-nullable
                check_dtype=False,
            )
            check_logger_msg(stream, "Filter pushdown successfully performed.")
            check_logger_msg(stream, "Columns loaded ['l_orderkey']")
