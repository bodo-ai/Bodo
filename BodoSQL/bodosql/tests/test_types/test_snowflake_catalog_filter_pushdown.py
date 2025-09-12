"""
Tests filter pushdown with a Snowflake Catalog object.
"""

import io
import os
import re

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.libs.dict_arr_ext import is_dict_encoded
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    create_snowflake_table,
    get_snowflake_connection_string,
    pytest_snowflake,
)
from bodo.tests.utils_jit import DistTestPipeline
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    test_db_snowflake_catalog,
)
from bodosql.utils import levenshteinDistance

pytestmark = pytest_snowflake


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
            check_logger_msg(
                stream,
                'WHERE "L_ORDERKEY" > 10 AND STARTSWITH("L_SHIPMODE", $$AIR$$)',
            )

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
    with set_logging_stream(logger, 2):
        bodo.jit(pipeline_class=DistTestPipeline)(impl)(bc, query)
        check_logger_msg(stream, "Columns loaded ['mycol', 'mycol2']")
        # This should be included in the pushed down SQL query if it
        # succeeds.
        check_logger_msg(stream, "FETCH NEXT 5 ROWS ONLY")


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
    py_output.columns = py_output.columns.str.upper()

    # make sure filter pushdown worked
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl, (bc, query), py_output=py_output, sort_output=True, reset_index=True
        )
        check_logger_msg(stream, "Columns loaded ['mycol', 'mycol2']")
        # Pushdown happens in the planner. Check the timer message instead.
        check_logger_msg(
            stream,
            'FROM "TEST_DB"."PUBLIC"."BODOSQL_ALL_SUPPORTED" WHERE COALESCE("MYCOL2", CURRENT_DATE) > $$2022-01-01$$',
        )


@pytest.mark.parametrize(
    "func_args",
    [
        pytest.param(("lower", "lower"), id="lower"),
        pytest.param(("upper", "upper"), id="upper"),
    ],
)
def test_no_arg_string_transform_functions(
    test_db_snowflake_catalog, func_args, memory_leak_check
):
    """
    Test no-arg string transform function support in Snowflake filter pushdown
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
            expected_output.columns = expected_output.columns.str.upper()
            check_func(
                impl,
                (bc, query),
                py_output=expected_output,
                sort_output=True,
                reset_index=True,
                check_dtype=False,
            )
            check_logger_msg(stream, "Columns loaded ['a']")
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE {sql_func.upper()}("A") = $${test_str_val}$$',
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

    expected_output = pd.DataFrame({"B": [11, 10] * 10})

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
            # Pushdown happens in the planner. Check the timer message instead.
            # Note this optimized in the planner
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE LOWER("A") = $$macedonia$$ OR "A" IS NULL',
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

    expected_output = pd.DataFrame({"B": [100, 10000, 11, 1321, 10] * 10})

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
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE UPPER(COALESCE("A", $$macedonia$$)) = $$MACEDONIA$$',
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

    query = "select c_name from TPCH_SF1.CUSTOMER where coalesce(C_MKTSEGMENT, C_COMMENT) = 'hello world'"

    py_output = pd.read_sql(
        query,
        get_snowflake_connection_string("SNOWFLAKE_SAMPLE_DATA", "PUBLIC"),
    )
    py_output.columns = py_output.columns.str.upper()

    # make sure filter pushdown was not performed
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl, (bc, query), py_output=py_output, sort_output=True, reset_index=True
        )
        # Pushdown happens in the planner. Check the timer message instead.
        check_logger_msg(
            stream,
            'COALESCE("C_MKTSEGMENT", "C_COMMENT") = $$hello world$$',
        )


def test_snowflake_catalog_limit_pushdown(memory_leak_check):
    """
    Test limit pushdown with loading from a table from a Snowflake catalog.
    """

    def impl(bc, query):
        return bc.sql(query)

    # Note: We don't yet support casting with limit pushdown.
    py_output = pd.read_sql(
        "select mycol from PUBLIC.BODOSQL_ALL_SUPPORTED WHERE mycol = 'A' LIMIT 5",
        get_snowflake_connection_string("TEST_DB", "PUBLIC"),
    )
    py_output.columns = py_output.columns.str.upper()

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
    # make sure filter + limit pushdown worked
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(impl, (bc, query), py_output=py_output)

        # The planner doesn't fully prune the filter column with the volcano
        # planner due to the metadata constraints on the RelSubset. This may
        # change as we increase the number of HEP steps.
        check_logger_msg(stream, "Columns loaded ['mycol']")

        # This should be included in the pushed down SQL query if it
        # succeeds.
        check_logger_msg(stream, "FETCH NEXT 5 ROWS ONLY")


"""
Tests that queries with like perform filter pushdown for all the
cases with the optimized paths. This is tested both with and without
escapes.
"""


def test_snowflake_like_pushdown_equality(test_db_snowflake_catalog, memory_leak_check):
    """Specific test where "like" condition is just an equality check"""

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["afewf", "b%bf", "hello", "happy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_equality_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # equality test
        query1 = f"Select a from {table_name} where a like 'hello'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "hello"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )

            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$hello$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_like_pushdown_startswith(
    test_db_snowflake_catalog, memory_leak_check
):
    """Specific test where "like" condition is equivalent to a startswith"""

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["afewf", "b%bf", "hello", "happy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_startswith_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # startswith test
        query2 = f"Select a from {table_name} where a like 'he%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.startswith("he")]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$he%$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_like_pushdown_endswith(test_db_snowflake_catalog, memory_leak_check):
    """Specific test where "like" condition is equivalent to an endswith"""

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["afewf", "b%bf", "hello", "happy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_endswith_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # endswith test
        query3 = f"Select a from {table_name} where a like '%lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.endswith("lo")]
            check_func(
                impl,
                (bc, query3),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$%lo$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_like_pushdown_contains(test_db_snowflake_catalog, memory_leak_check):
    """Specific test where "like" condition is equivalent to a contains"""

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["afewf", "b%bf", "hello", "happy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_contains_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # contains test
        query4 = f"Select a from {table_name} where a like '%e%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.contains("e")]
            check_func(
                impl,
                (bc, query4),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$%e%$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_like_pushdown_equality_escape(
    test_db_snowflake_catalog, memory_leak_check
):
    """Specific test where "like" condition is just an equality check (with a escape)"""

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["afewf", "b%bf", "hello", "happy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_equality_escape_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # Equality with escape test
        query5 = f"Select a from {table_name} where a like 'b^%bf' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "b%bf"]
            check_func(
                impl,
                (bc, query5),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$b^%bf$$ ESCAPE $$^$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_like_pushdown_startswith_escape(
    test_db_snowflake_catalog, memory_leak_check
):
    """Specific test where "like" condition is just an startswith check (with a escape)"""

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["afewf", "b%bf", "hello", "happy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_startswith_escape_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # startswith with escape test
        query6 = f"Select a from {table_name} where a like 'b^%%' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.startswith("b%")]
            check_func(
                impl,
                (bc, query6),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$b^%%$$ ESCAPE $$^$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_like_pushdown_endswith_escape(
    test_db_snowflake_catalog, memory_leak_check
):
    """Specific test where "like" condition is just an endswith check (with a escape)"""

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["afewf", "b%bf", "hello", "happy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_endswith_escape_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # endswith with escape test
        query7 = f"Select a from {table_name} where a like '%^%bf' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.endswith("%bf")]
            check_func(
                impl,
                (bc, query7),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$%^%bf$$ ESCAPE $$^$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_like_pushdown_contains_escape(
    test_db_snowflake_catalog, memory_leak_check
):
    """Specific test where "like" condition is just an contains check (with a escape)"""

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame(
        {"A": ["afewf", "b%bf", "hello", "happy", "hel", "llo", "b%L", "ab%%bf"] * 10}
    )

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_pushdown_contains_escape_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # contains with escape test
        query8 = f"Select a from {table_name} where a like '%^%%' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.contains("%")]
            check_func(
                impl,
                (bc, query8),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$%^%%$$ ESCAPE $$^$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


@pytest.mark.skip("Not properly taking optimized path")
def test_snowflake_like_always_true_optimized(
    test_db_snowflake_catalog, memory_leak_check
):
    """Specific test where "like" condition is always true"""

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
        py_output.columns = py_output.columns.str.upper()
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
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" IS NOT NULL',
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
        py_output.columns = py_output.columns.str.upper()
        # equality test
        query1 = f"Select a from {table_name} where a ilike 'hello'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.lower() == "hello"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$hello$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # startswith test
        query2 = f"Select a from {table_name} where a ilike 'He%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.lower().str.startswith("he")]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$He%$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # endswith test
        query3 = f"Select a from {table_name} where a ilike '%lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.lower().str.endswith("lo")]
            check_func(
                impl,
                (bc, query3),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$%lo$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # contains test
        query4 = f"Select a from {table_name} where a ilike '%e%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.lower().str.contains("e")]
            check_func(
                impl,
                (bc, query4),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$%e%$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # Equality with escape test
        query5 = f"Select a from {table_name} where a ilike 'b^%bf' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.lower() == "b%bf"]
            check_func(
                impl,
                (bc, query5),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$b^%bf$$ ESCAPE $$^$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # startswith with escape test
        query6 = f"Select a from {table_name} where a ilike 'B^%%' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.lower().str.startswith("b%")]
            check_func(
                impl,
                (bc, query6),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$B^%%$$ ESCAPE $$^$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # endswith with escape test
        query7 = f"Select a from {table_name} where a ilike '%^%bf' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.lower().str.endswith("%bf")]
            check_func(
                impl,
                (bc, query7),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$%^%bf$$ ESCAPE $$^$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # contains with escape test
        query8 = f"Select a from {table_name} where a ilike '%^%b%' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.lower().str.contains("%b")]
            check_func(
                impl,
                (bc, query8),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$%^%b%$$ ESCAPE $$^$$',
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
        py_output.columns = py_output.columns.str.upper()
        # equality test
        query1 = f"Select a from {table_name} where a like 'hE_lO'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "hELlO"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$hE_lO$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query2 = f"Select a from {table_name} where a like 'b^%_f' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "b%Bf"]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$b^%_f$$ ESCAPE $$^$$',
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
        py_output.columns = py_output.columns.str.upper()
        # equality test
        query1 = f"Select a from {table_name} where a ilike 'he_lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "hELlO"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$he_lo$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query2 = f"Select a from {table_name} where a ilike 'b^%_F' escape '^'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "b%Bf"]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$b^%_F$$ ESCAPE $$^$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


"""
    Tests that queries with like and non-constant patterns support filter
    pushdown. This is tested both with and without escapes.
"""


def test_snowflake_like_non_constant_pushdown_no_escape(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test without escape.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame({"A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10})

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_non_constant_pushdown_without_escape_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # equality test
        query1 = f"Select a from {table_name} where a like 'hE_l' || 'O'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "hELlO"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$hE_l$$ || $$O$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_like_non_constant_pushdown_with_escape(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Test with escape.
    """
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame({"A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10})

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "like_non_constant_pushdown_with_escape_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        query2 = (
            f"Select a from {table_name} where a like 'b' || '^%_f' escape upper('^')"
        )
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "b%Bf"]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" LIKE $$b^%_f$$ ESCAPE $$^$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


"""
Tests that queries with ilike and non-constant patterns support filter
pushdown. This is tested both with and without escapes.
"""


def test_snowflake_ilike_non_constant_pushdown_no_escape(
    test_db_snowflake_catalog, memory_leak_check
):
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    new_df = pd.DataFrame({"A": ["AFewf", "b%Bf", "hELlO", "HAPPy"] * 10})

    def impl(bc, query):
        return bc.sql(query)

    with create_snowflake_table(
        new_df, "ilike_non_constant_pushdown_no_escape_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        # equality test
        query1 = f"Select a from {table_name} where a ilike 'he' || '_lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "hELlO"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$he$$ || $$_lo$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_ilike_non_constant_pushdown_escape(
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
        new_df, "ilike_non_constant_pushdown_with_escape_table", db, schema
    ) as table_name:
        # Load the whole table in pandas.
        conn_str = get_snowflake_connection_string(db, schema)
        py_output = pd.read_sql(f"select * from {table_name}", conn_str)
        py_output.columns = py_output.columns.str.upper()
        query2 = (
            f"Select a from {table_name} where a ilike 'b^' || '%_F' escape lower('^')"
        )
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A == "b%Bf"]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" ILIKE $$b^%_F$$ ESCAPE $$^$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


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
        py_output.columns = py_output.columns.str.upper()
        query1 = f"Select a from {table_name} where b"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.B][["A"]]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                'WHERE "B"',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query2 = f"Select a from {table_name} where b or c"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.B | py_output.C][["A"]]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "B" OR "C"',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query3 = f"Select a from {table_name} where b and c"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.B & py_output.C][["A"]]
            check_func(
                impl,
                (bc, query3),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "B" AND "C"',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


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
        py_output.columns = py_output.columns.str.upper()
        query1 = f"Select a from {table_name} where not b"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.B][["A"]]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                'WHERE NOT "B"',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query2 = f"Select a from {table_name} where not(b or c)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.B | py_output.C)][["A"]]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE NOT "B" AND NOT "C"',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query3 = f"Select a from {table_name} where not(b and c)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.B & py_output.C)][["A"]]
            check_func(
                impl,
                (bc, query3),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE NOT "B" OR NOT "C"',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query4 = f"Select a from {table_name} where (not b) or c"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.B | py_output.C][["A"]]
            check_func(
                impl,
                (bc, query4),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "C" OR NOT "B"',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query5 = f"Select a from {table_name} where (not b) and c"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.B & py_output.C][["A"]]
            check_func(
                impl,
                (bc, query5),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "C" AND NOT "B"',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


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
        py_output.columns = py_output.columns.str.upper()
        query1 = f"Select a from {table_name} where not (b <> 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.B != 3)][["A"]]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream,
                'WHERE "B" = 3',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query2 = f"Select a from {table_name} where not (b <= 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.B <= 3)][["A"]]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                'WHERE "B" > 3',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query3 = f"Select a from {table_name} where not (b < 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.B < 3)][["A"]]
            check_func(
                impl,
                (bc, query3),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                'WHERE "B" >= 3',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query4 = f"Select a from {table_name} where not (b >= 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.B >= 3)][["A"]]
            check_func(
                impl,
                (bc, query4),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                'WHERE "B" < 3',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query5 = f"Select a from {table_name} where not (b > 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.B > 3)][["A"]]
            check_func(
                impl,
                (bc, query5),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                'WHERE "B" <= 3',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query6 = f"Select a from {table_name} where NOT((b > 3 OR b < 2) AND C)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[
                ~(((py_output.B > 3) | (py_output.B < 2)) & py_output.C)
            ][["A"]]
            check_func(
                impl,
                (bc, query6),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE NOT "C" OR "B" >= 2 AND "B" <= 3',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


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
        py_output.columns = py_output.columns.str.upper()
        query1 = f"Select a from {table_name} where not (b is NULL)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.B.isna())][["A"]]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "B" IS NOT NULL',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        query2 = f"Select a from {table_name} where not (b is not NULL)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~(py_output.B.notna())][["A"]]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "B" IS NULL',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


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
        py_output.columns = py_output.columns.str.upper()
        query = f"Select a from {table_name} where b not in (1, 3)"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[
                (~py_output.B.isin([1, 3])) & py_output.B.notna()
            ][["A"]]
            check_func(
                impl,
                (bc, query),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "B" <> 1 AND "B" <> 3',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


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
        py_output.columns = py_output.columns.str.upper()
        # equality test
        query1 = f"Select a from {table_name} where a not like 'hello'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A != "hello"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" NOT LIKE $$hello$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # startswith test
        query2 = f"Select a from {table_name} where a not like 'he%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.A.str.startswith("he")]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" NOT LIKE $$he%$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # endswith test
        query3 = f"Select a from {table_name} where a not like '%lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.A.str.endswith("lo")]
            check_func(
                impl,
                (bc, query3),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" NOT LIKE $$%lo$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # contains test
        query4 = f"Select a from {table_name} where a not like '%e%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.A.str.contains("e")]
            check_func(
                impl,
                (bc, query4),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" NOT LIKE $$%e%$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


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
        py_output.columns = py_output.columns.str.upper()
        # equality test
        query1 = f"Select a from {table_name} where a not ilike 'hello'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A.str.lower() != "hello"]
            check_func(
                impl,
                (bc, query1),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" NOT ILIKE $$hello$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # startswith test
        query2 = f"Select a from {table_name} where a not ilike 'He%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.A.str.lower().str.startswith("he")]
            check_func(
                impl,
                (bc, query2),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" NOT ILIKE $$He%$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # endswith test
        query3 = f"Select a from {table_name} where a not ilike '%lo'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.A.str.lower().str.endswith("lo")]
            check_func(
                impl,
                (bc, query3),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" NOT ILIKE $$%lo$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")
        # contains test
        query4 = f"Select a from {table_name} where a not ilike '%e%'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[~py_output.A.str.lower().str.contains("e")]
            check_func(
                impl,
                (bc, query4),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" NOT ILIKE $$%e%$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


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
        py_output.columns = py_output.columns.str.upper()
        # equality test
        query = f"Select a from {table_name} where a not like 'hE_lO'"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            expected_output = py_output[py_output.A != "hELlO"]
            check_func(
                impl,
                (bc, query),
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
            )
            # Filter pushdown is handled by the planner. Check the timer message instead.
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "A" NOT LIKE $$hE_lO$$',
            )
            check_logger_msg(stream, "Columns loaded ['a']")


def test_snowflake_length_filter_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that the aliases of LENGTH which are valid in snowflake are pushed
    directly into the snowflake query filter pushdown
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
        py_output.columns = py_output.columns.str.upper()
        expected_output = py_output[py_output.A.str.len() == 4]

        # Test for the aliases of LENGTH which are valid in snowflake
        for func_name in ("LENGTH", "LEN", "CHAR_LENGTH", "CHARACTER_LENGTH"):
            query = f"Select a from {table_name} where {func_name}(A) = 4"
            if func_name in ("CHAR_LENGTH", "CHARACTER_LENGTH", "LEN"):
                pushdown_name = "LENGTH"
            else:
                pushdown_name = func_name
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
                    f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE {pushdown_name}("A") = 4',
                )
                check_logger_msg(stream, "Columns loaded ['a']")


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
            check_logger_msg(
                stream,
                f'FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE REVERSE("L_SHIPMODE") = $$PIHS$$',
            )

            check_logger_msg(stream, "Columns loaded ['l_orderkey']")


num_to_pd_func = {
    "abs": np.abs,
    "sign": np.sign,
    "ceil": np.ceil,
    "floor": np.floor,
}


@pytest.mark.parametrize(
    "func_args",
    [
        pytest.param(("abs", "INT_COL", 3), id="abs_int"),
        pytest.param(("abs", "FLOAT_COL", 3.2), id="abs_float"),
        pytest.param(("sign", "INT_COL", 0), id="sign_int"),
        pytest.param(("sign", "FLOAT_COL", -1), id="sign_float"),
    ],
)
def test_no_arg_numeric_functions(
    test_db_snowflake_catalog, func_args, memory_leak_check
):
    """
    Test no-arg numeric function support in Snowflake filter pushdown
    """
    sql_func, num_col, test_val = func_args
    pd_func = num_to_pd_func[sql_func]
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    table_name = "NUMERIC_DATA"

    def impl(bc, query):
        return bc.sql(query)

    conn_str = get_snowflake_connection_string(db, schema)
    df = pd.read_sql(f"select {num_col} from {table_name}", conn_str)
    df.columns = df.columns.str.upper()
    expected_output = df[pd_func(df[num_col]) == test_val]
    query = (
        f"select {num_col} from {table_name} where {sql_func}({num_col}) = '{test_val}'"
    )

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

        check_logger_msg(stream, f"Columns loaded ['{num_col.lower()}']")
        check_logger_msg(
            stream,
            f"""FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE {sql_func.upper()}("{num_col.upper()}") = {test_val}""",
        )


@pytest.mark.parametrize(
    "func_args",
    [
        pytest.param(("ceil", "FLOAT_COL", -4), id="ceil_float"),
        pytest.param(("floor", "FLOAT_COL", 3), id="floor_float"),
    ],
)
def test_optional_arg_numeric_functions(
    test_db_snowflake_catalog, func_args, memory_leak_check
):
    """
    Test numeric function support in Snowflake filter pushdown
    that take an optional argument but the argument isn't provided.
    """
    sql_func, num_col, test_val = func_args
    pd_func = num_to_pd_func[sql_func]
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    table_name = "NUMERIC_DATA"

    def impl(bc, query):
        return bc.sql(query)

    conn_str = get_snowflake_connection_string(db, schema)
    df = pd.read_sql(f"select {num_col} from {table_name}", conn_str)
    df.columns = df.columns.str.upper()
    expected_output = df[pd_func(df[num_col]) == test_val]
    query = (
        f"select {num_col} from {table_name} where {sql_func}({num_col}) = '{test_val}'"
    )

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

        check_logger_msg(stream, f"Columns loaded ['{num_col.lower()}']")
        check_logger_msg(
            stream,
            f"""FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE {sql_func.upper()}("{num_col.upper()}") = {test_val}""",
        )


@pytest.mark.parametrize(
    "test_cases",
    [
        pytest.param(("hour", True, 0), id="hour"),
        pytest.param(("minute", True, 30), id="minute"),
        pytest.param(("second", True, 59), id="second"),
        pytest.param(("year", False, 2022), id="year"),
        pytest.param(("day", False, 1), id="day"),
        pytest.param(("dayofweek", False, 6), id="dayofweek"),
        pytest.param(("dayofyear", False, 15), id="dayofyear"),
        pytest.param(("week", False, 2), id="week"),
        pytest.param(("weekofyear", False, 22), id="weekofyear"),
        pytest.param(("yearofweek", False, 10), id="yearofweek"),
        pytest.param(("yearofweekiso", False, 30), id="yearofweekiso"),
        pytest.param(("month", False, 6), id="month"),
        pytest.param(("quarter", False, 0), id="quarter"),
    ],
)
def test_no_arg_datetime_functions(
    test_db_snowflake_catalog, test_cases, memory_leak_check
):
    """
    Test no-arg datetime function support in Snowflake filter pushdown
    """
    sql_func, is_time_related, test_val = test_cases
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]
    table_name = "DATETIME_RELATED_DATA"
    test_cols = ["timestamp_col"] if is_time_related else ["date_col", "timestamp_col"]

    def impl(bc, query):
        return bc.sql(query)

    conn_str = get_snowflake_connection_string(db, schema)
    df = pd.read_sql(f"select * from {table_name}", conn_str)

    for test_col in test_cols:
        datetime_test_col = df[test_col]
        if test_col != "timestamp_col":
            datetime_test_col = pd.to_datetime(datetime_test_col)

        if sql_func == "dayofweek":
            # this is technically incorrect and does not 100% emulate Snowflake
            # DAYOFWEEK behavior, but suffices for this test
            computed_col = (datetime_test_col.dt.dayofweek + 1) % 7
        elif sql_func in ("week", "weekofyear"):
            # this is technically incorrect and does not 100% emulate Snowflake
            # WEEKOFYEAR behavior, but suffices for this test
            computed_col = datetime_test_col.dt.isocalendar().week
        elif sql_func == "yearofweek":
            # this is technically incorrect and does not 100% emulate Snowflake
            # YEAROFWEEK behavior, but suffices for this test
            computed_col = datetime_test_col.dt.isocalendar().year
        elif sql_func == "yearofweekiso":
            computed_col = datetime_test_col.dt.isocalendar().year
        else:
            computed_col = getattr(datetime_test_col.dt, sql_func)

        # Match expected datetime.date output type
        if test_col != "timestamp_col":
            datetime_test_col = datetime_test_col.dt.date

        expected_output = datetime_test_col[computed_col == test_val].to_frame()
        expected_output.columns = expected_output.columns.str.upper()

        query = f"select {test_col} from {table_name} where {sql_func}({test_col}) = '{test_val}'"

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

            check_logger_msg(stream, f"Columns loaded ['{test_col}']")
            check_logger_msg(
                stream,
                f"""FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE {sql_func.upper()}("{test_col.upper()}") = {test_val}""",
            )


string_transform_func = {
    "trim": lambda x: x.strip(),
    "trim-opt-char": lambda x: x.strip("h"),
    "ltrim": lambda x: x.lstrip(),
    "ltrim-opt-char": lambda x: x.lstrip("h"),
    "rtrim": lambda x: x.rstrip(),
    "rtrim-opt-char": lambda x: x.rstrip("h"),
    "initcap": lambda x: x.title(),
    "initcap-opt-char": lambda x: ",".join([t.title() for t in x.split(",")]),
    "lpad-opt-char": lambda x: x.rjust(4, "0"),
    "rpad-opt-char": lambda x: x.ljust(7, "r"),
    "repeat": lambda x: x * 3,
    "strtok": lambda x: x.split()[0] if x else None,
    "strtok-opt-delimiter": lambda x: x.split("h")[0] if x else None,
    "strtok-opt-delimiter-partNr": lambda x: x.split("h")[2]
    if len(list(x.split("h"))) >= 3
    else None,
    "translate": lambda x: x.translate(str.maketrans("abcdefgh", "ABCDEFGH")),
    "concat": lambda x: x + "bodo",
    "concat-binop": lambda x: x + "bodo",
}


@pytest.mark.parametrize(
    "test_cases",
    [
        pytest.param(("LTRIM(A)", "LTRIM", "hi"), id="ltrim"),
        pytest.param(("LTRIM(A, 'h')", "LTRIM", "i"), id="ltrim-opt-char"),
        pytest.param(("RTRIM(A)", "RTRIM", "hi"), id="rtrim"),
        pytest.param(("RTRIM(A, 'h')", "RTRIM", "i"), id="rtrim-opt-char"),
        pytest.param(("INITCAP(A)", "INITCAP", "Hello"), id="initcap"),
        pytest.param(
            ("INITCAP(A, ',')", "INITCAP", "Hi,Hello"),
            id="initcap-opt-char",
        ),
        pytest.param(("LPAD(A, 4, '0')", "LPAD", "00hi"), id="lpad-opt-char"),
        pytest.param(("RPAD(A, 7, 'r')", "RPAD", "hellorr"), id="rpad-opt-char"),
        pytest.param(("REPEAT(A, 3)", "REPEAT", "hihihi"), id="repeat"),
        pytest.param(("STRTOK(A)", "STRTOK", "hello"), id="strtok"),
        pytest.param(("STRTOK(A, 'h')", "STRTOK", "h"), id="strtok-opt-delimiter"),
        pytest.param(
            ("STRTOK(A, 'h', 2)", "STRTOK", "ello"), id="strtok-opt-delimiter-partNr"
        ),
        pytest.param(
            ("TRANSLATE(A, 'abcdefgh', 'ABCDEFGH')", "TRANSLATE", "Hello"),
            id="translate",
        ),
        pytest.param(
            ("CONCAT(A, 'bodo')", "CONCAT", "hibodo"),
            id="concat",
            marks=pytest.mark.skip("Number of args in the IR for concat_ws is wrong"),
        ),
        pytest.param(
            ("A || 'bodo'", "CONCAT", "hibodo"),
            id="concat-binop",
            marks=pytest.mark.skip("Number of args in the IR for concat_ws is wrong"),
        ),
        pytest.param(("TRIM(A)", "TRIM", "hi"), id="trim"),
        pytest.param(("TRIM(A, 'h')", "TRIM", "i"), id="trim-opt-char"),
    ],
)
def test_with_arg_string_transform_functions(
    test_db_snowflake_catalog, test_cases, request, memory_leak_check
):
    """Checks that filter pushdown works for the functions that we handle within the planner"""
    filter_predicate, sql_func, answer = test_cases
    sql_filter = f"{filter_predicate} = '{answer}'"

    table_name = "STRING_DATA"
    conn_str = get_snowflake_connection_string(
        test_db_snowflake_catalog.database,
        test_db_snowflake_catalog.connection_params["schema"],
    )
    df = pd.read_sql(f"select a from {table_name}", conn_str)
    df.columns = df.columns.str.upper()

    test_case = request.node.callspec.id[request.node.callspec.id.find("-") + 1 :]
    pd_func = string_transform_func[test_case]

    expected_output = df[df["A"].apply(pd_func) == answer].dropna()

    query = f"select a from {table_name} where {sql_filter}"

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    def impl(bc, query):
        return bc.sql(query)

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

        check_logger_msg(stream, "Columns loaded ['a']")
        # Each function may have different number of args,
        # so we will just check that the SQL function got logged.
        check_logger_msg(
            stream,
            f"""FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE {sql_func.upper()}(""",
        )


# Dictionary mapping test case to lambda function
# used to generate expected output for
# test_nonregex_string_match_functions
nonregex_string_match_func = {
    "left": lambda x: x[:3],
    "right": lambda x: x[-4:],
    "position": lambda x: "Hi, bye bodo.ai".find(x) + 1,
    "position-alt": lambda x: "Hi, bye bodo.ai".find(x) + 1,
    "replace": lambda x: x.replace("bodo", "snowflake"),
    "replace-empty": lambda x: x.replace("hi", ""),
    "substr": lambda x: x[3:7],
    "substring": lambda x: x[:2],
    "charindex": lambda x: "snowflake hi, hello".find(x) + 1,
    "charindex-start-pos": lambda x: "snowflake hi, hello".find(x, 10) + 1,
    "editdistance-no-max": lambda x: levenshteinDistance(x, "hi, hello"),
    "editdistance-with-max": lambda x: levenshteinDistance(x, "hi, hello", 5),
}


@pytest.mark.parametrize(
    "test_cases",
    [
        pytest.param(("LEFT(A, 3)", "LEFT", "hel"), id="left"),
        pytest.param(("RIGHT(A, 4)", "RIGHT", "ello"), id="right"),
        pytest.param(("POSITION(A, 'Hi, bye bodo.ai')", "CHARINDEX", 5), id="position"),
        pytest.param(
            ("POSITION(A in 'Hi, bye bodo.ai')", "CHARINDEX", 5), id="position-alt"
        ),
        pytest.param(
            ("REPLACE(A, 'bodo', 'snowflake')", "REPLACE", "hi snowflake"), id="replace"
        ),
        pytest.param(
            ("REPLACE(A, 'hi')", "REPLACE", "hi snowflake"),
            id="replace-empty",
            marks=pytest.mark.skip("[BE-4599] REPLACE with 2 args not supported yet"),
        ),
        pytest.param(("SUBSTR(A, 4, 4)", "SUBSTRING", "bodo"), id="substr"),
        pytest.param(("SUBSTRING(A, 1, 2)", "SUBSTRING", "hi"), id="substring"),
        pytest.param(
            ("CHARINDEX(A, 'snowflake hi, hello')", "CHARINDEX", 11), id="charindex"
        ),
        pytest.param(
            ("CHARINDEX(A, 'snowflake hi, hello', 11)", "CHARINDEX", 11),
            id="charindex-start-pos",
        ),
        pytest.param(
            ("EDITDISTANCE(A, 'hi, hello')", "EDITDISTANCE", 7),
            id="editdistance-no-max",
        ),
        pytest.param(
            ("EDITDISTANCE(A, 'hi, hello', 5)", "EDITDISTANCE", 5),
            id="editdistance-with-max",
        ),
    ],
)
def test_nonregex_string_match_functions(
    test_db_snowflake_catalog, test_cases, request, memory_leak_check
):
    filter_predicate, sql_func, answer = test_cases

    if isinstance(answer, str):
        sql_filter = f"{filter_predicate} = '{answer}'"
    else:
        sql_filter = f"{filter_predicate} = {answer}"

    table_name = "STRING_DATA"
    conn_str = get_snowflake_connection_string(
        test_db_snowflake_catalog.database,
        test_db_snowflake_catalog.connection_params["schema"],
    )
    df = pd.read_sql(f"select a from {table_name}", conn_str)
    df.columns = df.columns.str.upper()

    test_case = request.node.callspec.id[request.node.callspec.id.find("-") + 1 :]
    pd_func = nonregex_string_match_func[test_case]

    expected_output = df[df["A"].apply(pd_func) == answer].dropna()

    query = f"select a from {table_name} where {sql_filter}"

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    def impl(bc, query):
        return bc.sql(query)

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

        check_logger_msg(stream, "Columns loaded ['a']")

        check_logger_msg(
            stream,
            f"""FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE {sql_func.upper()}(""",
        )


@pytest.mark.parametrize(
    "test_cases",
    [
        pytest.param(
            (r"REGEXP_LIKE(A, '.*\w+-\w+.*')", r".*\w+-\w+.*", True, "REGEXP_LIKE"),
            id="regexp_like",
        ),
        pytest.param(
            (
                r"REGEXP_LIKE(A, '.*\w+-\w+.*', 'c')",
                r".*\w+-\w+.*",
                True,
                "REGEXP_LIKE",
            ),
            id="regexp_like_opt_arg",
        ),
        pytest.param(
            ("RLIKE(A, 'bodo[sS].*')", "bodo[sS].*", True, "REGEXP_LIKE"),
            id="regexp_like_alias_1",
        ),
        pytest.param(
            ("RLIKE(A, 'bodo[sS].*', 'c')", "bodo[sS].*", True, "REGEXP_LIKE"),
            id="regexp_like_alias_1_opt_args",
        ),
        pytest.param(
            ("A REGEXP 'hi* [bB].*'", "hi* [bB].*", True, "REGEXP_LIKE"),
            id="regexp_like_alias_2",
        ),
        pytest.param(
            ("REGEXP_INSTR(A, '\\W+') = 3", "\\W+", 3, "REGEXP_INSTR"),
            id="regexp_instr",
        ),
        pytest.param(
            ("REGEXP_INSTR(A, '\\W+', 1, 1, 0, 'i', 0) = 3", "\\W+", 3, "REGEXP_INSTR"),
            id="regexp_instr_opt_args",
        ),
        pytest.param(
            ("REGEXP_SUBSTR(A, 'h\\w+') = 'hi'", "h\\w+", "hi", "REGEXP_SUBSTR"),
            id="regexp_substr",
        ),
        pytest.param(
            (
                "REGEXP_SUBSTR(A, 'h\\w+', 1, 1, 'c', 0) = 'hi'",
                "h\\w+",
                "hi",
                "REGEXP_SUBSTR",
            ),
            id="regexp_substr_opt_args",
        ),
        pytest.param(
            ("REGEXP_COUNT(A, '\\w+') = 3", "\\w+", 3, "REGEXP_COUNT"),
            id="regexp_count",
        ),
        pytest.param(
            ("REGEXP_COUNT(A, '\\w+', 1, 'c') = 3", "\\w+", 3, "REGEXP_COUNT"),
            id="regexp_count_opt_args",
        ),
        pytest.param(
            (
                "REGEXP_REPLACE(A, '\\w+', 'snowflake') = 'snowflake'",
                "\\w+",
                "snowflake",
                "REGEXP_REPLACE",
            ),
            id="regexp_replace",
        ),
        pytest.param(
            (
                "REGEXP_REPLACE(A, '\\w+', 'snowflake', 1, 0, 'c') = 'snowflake'",
                "\\w+",
                "snowflake",
                "REGEXP_REPLACE",
            ),
            id="regexp_replace_opt_args",
        ),
    ],
)
def test_regex_string_match_functions(
    test_db_snowflake_catalog, test_cases, request, memory_leak_check
):
    sql_filter, pat, answer, sql_function = test_cases

    table_name = "STRING_DATA"
    conn_str = get_snowflake_connection_string(
        test_db_snowflake_catalog.database,
        test_db_snowflake_catalog.connection_params["schema"],
    )
    df = pd.read_sql(f"select a from {table_name}", conn_str)
    df.columns = df.columns.str.upper()

    if "regexp_instr" in request.node.name:
        func = (
            lambda x: re.search(pat, x).start() + 1
            if re.search(pat, x) is not None
            else None
        )
    elif "regexp_substr" in request.node.name:
        func = (
            lambda x: re.search(pat, x).group()
            if re.search(pat, x) is not None
            else None
        )
    elif "regexp_count" in request.node.name:
        func = lambda x: len(re.findall(pat, x))
    elif "regexp_replace" in request.node.name:
        func = lambda x: re.sub(pat, "snowflake", x)
    elif "regexp_like" in request.node.name:
        func = lambda x: re.match(pat, x) is not None
    else:
        raise ValueError("Unknown Test Name")

    expected_output = df[df["A"].apply(func) == answer].dropna()

    query = f"select a from {table_name} where {sql_filter}"

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    def impl(bc, query):
        return bc.sql(query)

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

        check_logger_msg(stream, "Columns loaded ['a']")
        if sql_function == "RLIKE":
            # RLIKE has a different syntax then the others, so we use a different check
            check_logger_msg(
                stream,
                "RLIKE",
            )
        else:
            check_logger_msg(
                stream,
                f"""FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE {sql_function}(""",
            )


def test_snowflake_coalesce_constant_date_string_filter_pushdown(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that queries of the form COALESCE(date, constanst_string) <COMPARISON OP> constant_string
    can be pushed down to Snowflake.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    db = test_db_snowflake_catalog.database
    schema = test_db_snowflake_catalog.connection_params["schema"]

    df = pd.DataFrame(
        {
            "L_COMMITDATE": pd.Series(
                ["2023-01-20", "2023-01-21", "2023-01-19", None, "2023-06-20"] * 10,
                dtype="datetime64[ns]",
            ),
            "L_QUANTITY": pd.arrays.IntegerArray(
                np.array([1, -3, 2, 3, 10] * 10, np.int8),
                np.array([False, True, True, False, False] * 10),
            ),
        }
    )

    with create_snowflake_table(df, "simple_date_table", db, schema) as table_name:
        expected_output = pd.DataFrame(
            {
                "L_QUANTITY": pd.arrays.IntegerArray(
                    np.array([1, -3, 3, 10] * 10, np.int8),
                    np.array([False, True, False, False] * 10),
                ),
            }
        )

        # Explicit type specification is needed on the constant, otherwise casting ends up blocking filter pushdown
        query = f"select l_quantity from {table_name} where coalesce(L_COMMITDATE, TIMESTAMP '2023-06-20') >= TIMESTAMP '2023-01-20'"

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_func(
                impl,
                (bc, query),
                py_output=expected_output,
                is_out_distributed=False,
                reset_index=True,
                sort_output=True,
                # Pandas output is non-nullable
                check_dtype=False,
            )
            # Pushdown happens in the planner. Check the timer message instead.
            # Note this is optimized in the planner because the second value is always True.
            check_logger_msg(
                stream,
                f"""FROM "TEST_DB"."PUBLIC"."{table_name.upper()}" WHERE "L_COMMITDATE" IS NULL OR "L_COMMITDATE" >= TIMESTAMP '2023-01-20 00:00:00.000000000'""",
            )


@pytest.mark.parametrize(
    "sql_func, expected_output",
    [
        pytest.param(
            "LEAST",
            pd.DataFrame({"FLOAT_COL": [3.2, 3.1, 1.26, 2.99, 2.6]}),
            id="least",
        ),
        pytest.param(
            "GREATEST",
            pd.DataFrame({"FLOAT_COL": [-3.2, -4.51, -4.49, -5.25, 0, 0]}),
            id="greatest",
        ),
    ],
)
def test_least_greatest(
    test_db_snowflake_catalog, sql_func, expected_output, memory_leak_check
):
    query = f"select float_col from numeric_data where {sql_func}(float_col, 1.0) = 1.0"

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    def impl(bc, query):
        return bc.sql(query)

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

        check_logger_msg(stream, "Columns loaded ['float_col']")
        # Pushdown happens in the planner. Check the timer message instead.
        check_logger_msg(
            stream,
            f'WHERE {sql_func.upper()}("FLOAT_COL", 1.0) = 1.0',
        )


def test_column_pruning_pushdown_dict_encoding(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests for that column pruning does not interfere with out ability to do dictionairy encoding.
    """

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # KEATON_TESTING_TABLE_STRING_DUPLICATE_WHERE_FILTER_COL_EQUAL_2, as it's name suggests,
    # is completly duplicate for the column my_col where filter_col = 2, and completely unique
    # for all other groups.
    def impl1(bc):
        df1 = bc.sql(
            "SELECT MY_COL FROM KEATON_TESTING_TABLE_STRING_DUPLICATE_WHERE_FILTER_COL_EQUAL_2 WHERE FILTER_COL = 2"
        )
        # Check that the string columns are NOT dict encoded
        is_dict1_encoded = is_dict_encoded(df1["MY_COL"])
        return is_dict1_encoded

    def impl2(bc):
        df2 = bc.sql(
            "SELECT MY_COL FROM KEATON_TESTING_TABLE_STRING_DUPLICATE_WHERE_FILTER_COL_EQUAL_2 WHERE FILTER_COL <> 2"
        )
        is_dict2_encoded = is_dict_encoded(df2["MY_COL"])
        return is_dict2_encoded

    # In both cases the column isn't dict encoded because we don't depend on the filter.
    check_func(impl1, (bc,), py_output=False, check_dtype=False, reset_index=True)
    check_func(impl2, (bc,), py_output=False, check_dtype=False, reset_index=True)


def test_projection_pushdown_rename(test_db_snowflake_catalog, memory_leak_check):
    """Makes sure that renaming columns in a projection pushdown works as expected"""

    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)

    # KEATON_TESTING_TABLE_STRING_DUPLICATE_WHERE_FILTER_COL_EQUAL_2, as it's name suggests,
    # is completly duplicate for the column my_col where filter_col = 2, and completely unique
    # for all other groups.
    def impl1(bc):
        df1 = bc.sql(
            "SELECT MY_COL AS I_AM_NOT_A_REAL_COLUMN FROM KEATON_TESTING_TABLE_STRING_DUPLICATE_WHERE_FILTER_COL_EQUAL_2 WHERE FILTER_COL = 2"
        )
        # Check that the string columns are NOT dict encoded
        is_dict1_encoded = is_dict_encoded(df1["I_AM_NOT_A_REAL_COLUMN"])
        return is_dict1_encoded

    def impl2(bc):
        df2 = bc.sql(
            "SELECT MY_COL AS FILTER_COL FROM KEATON_TESTING_TABLE_STRING_DUPLICATE_WHERE_FILTER_COL_EQUAL_2 WHERE FILTER_COL <> 2"
        )
        is_dict2_encoded = is_dict_encoded(df2["FILTER_COL"])
        return is_dict2_encoded

    # In both cases the column isn't dict encoded because we don't depend on the filter.
    check_func(impl1, (bc,), py_output=False, check_dtype=False, reset_index=True)
    check_func(impl2, (bc,), py_output=False, check_dtype=False, reset_index=True)


def test_separate_database_read(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests reading from a table with a different database than the default
    in a manner that includes filter pushdown. This is primarily intended to
    ensure the filter pushdown code generation can load the original table.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION WHERE R_COMMENT IS NOT NULL"
    py_output = pd.read_sql(
        query,
        get_snowflake_connection_string(
            test_db_snowflake_catalog.database,
            test_db_snowflake_catalog.connection_params["schema"],
        ),
    )
    py_output.columns = py_output.columns.str.upper()
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    # make sure filter pushdown worked
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            check_dtype=False,
            sort_output=True,
            reset_index=True,
        )

        # Pushdown happens in the planner. Check the timer message instead.
        check_logger_msg(
            stream,
            'WHERE "R_COMMENT" IS NOT NULL',
        )


def test_inline_view_filter_pushdown(test_db_snowflake_catalog, memory_leak_check):
    """
    Tests that the a view defined in the same database and schema as every
    table it uses is properly inlined and filter pushdown sill works as expected.

    If you need to recreate this in snowflake, these are the relevant defintions:

    create table nick_base_table as (select length(r_comment) as A from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION)

    create view basic_view as select * from nick_base_table
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from basic_view where A > 1"
    py_output = pd.read_sql(
        query,
        get_snowflake_connection_string(
            test_db_snowflake_catalog.database,
            test_db_snowflake_catalog.connection_params["schema"],
        ),
    )
    py_output.columns = py_output.columns.str.upper()
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    # Ensure we always inline views
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            check_dtype=False,
            sort_output=True,
            reset_index=True,
        )
        # Verify that NICK_BASE_TABLE is found in the logger message so the
        # view was inlined.
        check_logger_msg(stream, "NICK_BASE_TABLE")
        # Pushdown happens in the planner. Check the timer message instead.
        check_logger_msg(
            stream,
            'WHERE "A" > 1',
        )


def test_inline_filter_view_filter_pushdown(
    test_db_snowflake_catalog, memory_leak_check
):
    """
    Tests that the a view defined to contain a filter still has its filter pushed
    down once it has been inlined.

    If you need to recreate this in snowflake, these are the relevant defintions:

    create table nick_base_table as (select length(r_comment) as A from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION)

    create view filter_view as select * from nick_base_table where A > 1
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select * from filter_view"
    py_output = pd.read_sql(
        query,
        get_snowflake_connection_string(
            test_db_snowflake_catalog.database,
            test_db_snowflake_catalog.connection_params["schema"],
        ),
    )
    py_output.columns = py_output.columns.str.upper()
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    # Ensure we always inline views
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            check_dtype=False,
            sort_output=True,
            reset_index=True,
        )
        # Verify that NICK_BASE_TABLE is found in the logger message so the
        # view was inlined.
        check_logger_msg(stream, "NICK_BASE_TABLE")
        # Pushdown happens in the planner. Check the timer message instead.
        check_logger_msg(
            stream,
            'WHERE "A" > 1',
        )
