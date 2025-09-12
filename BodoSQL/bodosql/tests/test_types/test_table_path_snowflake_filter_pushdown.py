"""
Tests filter pushdown with a Snowflake SQL TablePath.
"""

import io

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    get_snowflake_connection_string,
    pytest_snowflake,
)
from bodo.tests.utils_jit import DistTestPipeline

pytestmark = pytest_snowflake


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
@pytest.mark.skipif(
    bodo.bodosql_use_streaming_plan,
    reason="Limit Pushdown with Streaming & TablePath Snowflake Tables is Not Supported",
)
def test_simple_filter_pushdown(memory_leak_check):
    def impl1(table_name):
        bc = bodosql.BodoSQLContext(
            {"SQL_TABLE": bodosql.TablePath(table_name, "sql", conn_str=conn_str)},
        )
        return bc.sql(
            "select L_SUPPKEY from sql_table WHERE L_ORDERKEY > 10 AND 2 >= L_LINENUMBER ORDER BY L_SUPPKEY LIMIT 70",
        )

    def impl2(table_name):
        bc = bodosql.BodoSQLContext(
            {"SQL_TABLE": bodosql.TablePath(table_name, "sql", conn_str=conn_str)},
        )
        return bc.sql(
            "SELECT l_suppkey from sql_table where l_shipmode LIKE 'AIR%%' OR l_orderkey = 32",
        )

    table_name = "LINEITEM"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn_str = get_snowflake_connection_string(db, schema)
    py_output = pd.read_sql(
        f"select L_SUPPKEY from {table_name} ORDER BY L_SUPPKEY LIMIT 70", conn_str
    )
    # Names won't match because BodoSQL keep names uppercase.
    py_output.columns = ["L_SUPPKEY"]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        # Set check_dtype=False because BodoSQL loads columns with the actual Snowflake type
        check_func(
            impl1,
            (table_name,),
            py_output=py_output,
            reset_index=True,
            check_dtype=False,
        )

        # Check for pruned columns
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    py_output = pd.read_sql(
        f"SELECT l_suppkey from {table_name} where l_shipmode LIKE 'AIR%%' OR l_orderkey = 32",
        conn_str,
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        # Set check_dtype=False because BodoSQL loads columns with the actual Snowflake type
        check_func(
            impl2,
            (table_name,),
            py_output=py_output,
            reset_index=True,
            check_dtype=False,
            sort_output=True,
        )

        # Check for pruned columns
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")


def test_zero_columns_pruning(memory_leak_check):
    """
    Test loading just a length from a Snowflake table.
    """

    def impl(table_name, conn_str):
        bc = bodosql.BodoSQLContext(
            {"SQL_TABLE": bodosql.TablePath(table_name, "sql", conn_str=conn_str)},
        )
        return bc.sql("SELECT COUNT(*) as cnt from sql_table")

    table_name = "LINEITEM"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn_str = get_snowflake_connection_string(db, schema)
    py_output = pd.read_sql(
        f"select COUNT(*) as cnt from {table_name} ORDER BY L_SUPPKEY LIMIT 70",
        conn_str,
    )
    py_output.columns = py_output.columns.str.upper()

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (table_name, conn_str),
            py_output=py_output,
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Columns loaded []")


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
@pytest.mark.skipif(
    bodo.bodosql_use_streaming_plan,
    reason="Limit Pushdown with Streaming & TablePath Snowflake Tables is Not Supported",
)
def test_snowflake_limit_pushdown(memory_leak_check):
    """
    Test limit pushdown with loading from a Snowflake table.
    """

    def impl(table_name, conn_str):
        bc = bodosql.BodoSQLContext(
            {"SQL_TABLE": bodosql.TablePath(table_name, "sql", conn_str=conn_str)},
        )
        return bc.sql(
            "select mycol from sql_table WHERE mycol = 'A' LIMIT 5",
        )

    # Note: We don't yet support casting with limit pushdown.
    table_name = "BODOSQL_ALL_SUPPORTED"
    db = "TEST_DB"
    schema = "PUBLIC"
    conn_str = get_snowflake_connection_string(db, schema)
    py_output = pd.read_sql(
        f"select mycol from {table_name} WHERE mycol = 'A' LIMIT 5",
        conn_str,
    )
    check_func(impl, (table_name, conn_str), py_output=py_output)

    # make sure filter + limit pushdown worked
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
        bodo_func(table_name, conn_str)
        fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
        assert hasattr(fir, "meta_head_only_info")
        assert fir.meta_head_only_info[0] is not None
        # Note the only column need is pruned by the planner due to the filter.
        check_logger_msg(stream, "Columns loaded []")
        check_logger_msg(stream, "Filter pushdown successfully performed")


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
@pytest.mark.skipif(
    bodo.bodosql_use_streaming_plan,
    reason="In Pushdown with Streaming & TablePath Snowflake Tables is Not Supported",
)
def test_snowflake_in_pushdown(memory_leak_check):
    """
    Test in pushdown with loading from a Snowflake table.
    """

    def impl(table_name, conn_str):
        bc = bodosql.BodoSQLContext(
            {"SQL_TABLE": bodosql.TablePath(table_name, "sql", conn_str=conn_str)},
        )
        return bc.sql(
            "select mycol from sql_table WHERE mycol in ('A', 'B', 'C')",
        )

    # Note: We don't yet support casting with limit pushdown.
    table_name = "BODOSQL_ALL_SUPPORTED"
    db = "TEST_DB"
    schema = "PUBLIC"
    conn_str = get_snowflake_connection_string(db, schema)
    py_output = pd.read_sql(
        f"select mycol from {table_name} WHERE mycol in ('A', 'B', 'C')",
        conn_str,
    )

    # make sure filter + limit pushdown worked
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (table_name, conn_str),
            py_output=py_output,
            reset_index=True,
            check_dtype=False,
        )

        check_logger_msg(stream, "Columns loaded ['mycol']")
        check_logger_msg(stream, "Filter pushdown successfully performed")
