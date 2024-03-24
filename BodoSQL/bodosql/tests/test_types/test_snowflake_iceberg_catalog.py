import os
from io import StringIO

import pandas as pd
import pytest
from mpi4py import MPI

import bodo
import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    create_snowflake_iceberg_table,
    drop_snowflake_table,
    gen_unique_table_id,
    get_snowflake_connection_string,
    pytest_snowflake,
    temp_env_override,
)
from bodosql.tests.test_types.test_snowflake_catalog import assert_tables_equal

pytestmark = [pytest.mark.iceberg] + pytest_snowflake


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_basic_read(memory_leak_check):
    """
    Test reading an entire Iceberg table from Snowflake in SQL
    """
    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
        iceberg_volume="exvol",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "A": ["ally", "bob", "cassie", "david", pd.NA],
            "B": [10.5, -124.0, 11.11, 456.2, -8e2],
            "C": [True, pd.NA, False, pd.NA, pd.NA],
        }
    )

    query = "SELECT A, B, C FROM BODOSQL_ICEBERG_READ_TEST"
    check_func(
        impl,
        (bc, query),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_column_pruning(memory_leak_check):
    """
    Test reading an Iceberg table from Snowflake in SQL
    where columns are pruned and reordered
    """

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
        iceberg_volume="exvol",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "B": [10.5, 124.0, 11.11, 456.2, 8e2],
            "A": ["ally", "bob", "cassie", "david", pd.NA],
        }
    )

    query = "SELECT ABS(B) as B, A FROM BODOSQL_ICEBERG_READ_TEST"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(stream, "Columns loaded ['A', 'B']")


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_filter_pushdown(memory_leak_check):
    """
    Test reading an Iceberg table from Snowflake with filter pushdown
    """

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
        iceberg_volume="exvol",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "B": [10.5, 11.11, 456.2],
            "A": ["ally", "cassie", "david"],
        }
    )

    # A IS NOT NULL can be pushed down to iceberg
    # LIKE A '%a%' cannot be pushed down to iceberg
    query = "SELECT ABS(B) as B, A FROM BODOSQL_ICEBERG_READ_TEST WHERE B > 0 AND A LIKE '%a%'"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(stream, "Columns loaded ['A', 'B']")
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nbic.FilterExpr('AND', [bic.FilterExpr('>', [bic.ColumnRef('B'), bic.Scalar(f0)]), bic.FilterExpr('IS_NOT_NULL', [bic.ColumnRef('A')])])",
        )


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_filter_pushdown_col_not_read(memory_leak_check):
    """
    Test reading a Iceberg table with BodoSQL filter pushdown
    where a column used in the filter is not read in
    """

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
        iceberg_volume="exvol",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame(
        {
            "A": ["ally", "cassie", "david"],
        }
    )

    # A IS NOT NULL can be pushed down to iceberg
    # LIKE A '%a%' cannot be pushed down to iceberg
    query = "SELECT A FROM BODOSQL_ICEBERG_READ_TEST WHERE B > 0 AND A LIKE '%a%'"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(stream, "Columns loaded ['A']")
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nFilterExpr('AND', [FilterExpr('>', [ColumnRef('B'), Scalar(f0)]), FilterExpr('IS_NOT_NULL', [ColumnRef('A')])])",
        )


def test_snowflake_catalog_iceberg_write(memory_leak_check):
    """tests that writing tables using Iceberg works"""

    # Create a catalog with iceberg_volume specified
    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "E2E_TESTS_DB",
        connection_params={"schema": "PUBLIC"},
        iceberg_volume="exvol",
    )
    db = "E2E_TESTS_DB"
    schema = "PUBLIC"

    bc = bodosql.BodoSQLContext(catalog=catalog)
    # TODO[BSE-2665]: Support and test Snowflake Iceberg write for all types
    in_df = pd.DataFrame({"A": ["abc", "df"] * 100})
    bc = bc.add_or_replace_view("TABLE1", in_df)

    def impl(bc, query):
        bc.sql(query)
        # Return an arbitrary value. This is just to enable a py_output
        # so we can reuse the check_func infrastructure.
        return 5

    comm = MPI.COMM_WORLD
    table_name = None
    if bodo.get_rank() == 0:
        table_name = gen_unique_table_id("bodosql_catalog_write_iceberg_table")
    table_name = comm.bcast(table_name)

    success_query = f"CREATE OR REPLACE TABLE {schema}.{table_name} AS Select A from __bodolocal__.table1"
    exception_occurred_in_test_body = False
    try:
        # Only test with only_1D=True so we only insert into the table once.
        check_func(impl, (bc, success_query), only_1D=True, py_output=5)

        output_df = None
        # Load the data from snowflake on rank 0 and then broadcast all ranks. This is
        # to reduce the demand on Snowflake.
        if bodo.get_rank() == 0:
            conn_str = get_snowflake_connection_string(db, schema)
            output_df = pd.read_sql(f"select * from {table_name}", conn_str)
            output_df.columns = output_df.columns.str.upper()

        output_df = comm.bcast(output_df)
        assert_tables_equal(output_df, in_df)

        # Make sure the table is a managed Iceberg table
        output_table_type = None
        if bodo.get_rank() == 0:
            output_table_type = pd.read_sql(
                f"SHOW ICEBERG TABLES LIKE '{table_name}' in SCHEMA IDENTIFIER('{db}.{schema}')",
                conn_str,  # type: ignore
            )["iceberg_table_type"][0]

        output_table_type = comm.bcast(output_table_type)
        assert (
            output_table_type == "MANAGED"
        ), f"Table type is not as expected. Expected MANAGED but found {output_table_type}"

    except Exception as e:
        # In the case that another exception ocurred within the body of the try,
        # We may not have created a table to drop.
        # because of this, we call drop_snowflake_table in a try/except, to avoid
        # masking the original exception
        exception_occurred_in_test_body = True
        raise e
    finally:
        if exception_occurred_in_test_body:
            try:
                drop_snowflake_table(table_name, db, schema)
            except:
                pass
        else:
            drop_snowflake_table(table_name, db, schema)


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_limit_pushdown(memory_leak_check):
    """
    Test reading an Iceberg from Snowflake with limit pushdown.
    Since the planner has access to length statistics, we need to actually
    reduce the amount of data being read to test limit pushdown.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
        iceberg_volume="exvol",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [2]})

    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM BODOSQL_ICEBERG_READ_TEST LIMIT 2)"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Constant limit detected, reading at most 2 rows")


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_limit_filter_pushdown(memory_leak_check):
    """
    Test reading an Iceberg from Snowflake with limit + filter pushdown.
    Since the planner has access to length statistics, we need to actually
    reduce the amount of data being read to test limit pushdown.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
        iceberg_volume="exvol",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [1]})

    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM BODOSQL_ICEBERG_READ_TEST where B > 200 LIMIT 2)"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Constant limit detected, reading at most 2 rows")
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nFilterExpr('>', [ColumnRef('B'), Scalar(f0)])",
        )


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_multi_limit_pushdown(memory_leak_check):
    """
    Verify multiple limits are still simplified even though Iceberg trees
    only support a single limit.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
        iceberg_volume="exvol",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [1]})
    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM (SELECT * FROM BODOSQL_ICEBERG_READ_TEST LIMIT 2) LIMIT 1)"
    print(bc.generate_plan(query))
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        # The planner should simplify the two limits into a single limit
        check_logger_msg(stream, "Constant limit detected, reading at most 1 rows")


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_limit_filter_limit_pushdown(memory_leak_check):
    """
    Test reading an Iceberg from Snowflake with limit pushdown the first limit
    is pushed down, but an additional limit + filter after the first isn't pushed down.
    Since the planner has access to length statistics, we need to actually
    reduce the amount of data being read to test limit pushdown.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
        iceberg_volume="exvol",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [2]})

    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM (SELECT * FROM BODOSQL_ICEBERG_READ_TEST LIMIT 4) where B > 11 LIMIT 2)"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Constant limit detected, reading at most 4 rows")
        # Verify no limit pushdown as its after the limit
        check_logger_msg(
            stream, "Iceberg Filter Pushed Down:\nFilterExpr('ALWAYS_TRUE', [])"
        )
        # Verify we don't push down the second limit
        check_logger_no_msg(stream, "Constant limit detected, reading at most 2 rows")


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_filter_limit_filter_pushdown(memory_leak_check):
    """
    Test reading an Iceberg from Snowflake with filter + limit pushdown, any
    filter before the limit is pushed down with the limit, but not after.
    Since the planner has access to length statistics, we need to actually
    reduce the amount of data being read to test limit pushdown.

    As a result, since this is no longer order we will instead compute summary
    statistics and check that the number of rows read is identical
    """

    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        "TEST_DB",
        connection_params={"schema": "PUBLIC", "role": "ACCOUNTADMIN"},
        iceberg_volume="exvol",
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    py_out = pd.DataFrame({"OUTPUT": [1]})

    query = "SELECT COUNT(*) AS OUTPUT FROM (SELECT * FROM (SELECT * FROM BODOSQL_ICEBERG_READ_TEST where B > 11 LIMIT 4) where A <> 'david')"
    stream = StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, query),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
            # We have a scalar output.
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Constant limit detected, reading at most 4 rows")
        # Verify no limit pushdown as its after the limit
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nFilterExpr('>', [ColumnRef('B'), Scalar(f0)])",
        )


@temp_env_override({"AWS_REGION": "us-east-1"})
def test_dynamic_scalar_filter_pushdown(memory_leak_check):
    """
    Test that a dynamically generated filter can be pushed down to Iceberg.
    """
    database = "TEST_DB"
    schema = "PUBLIC"
    iceberg_volume = "exvol"
    catalog = bodosql.SnowflakeCatalog(
        os.environ.get("SF_USERNAME", ""),
        os.environ.get("SF_PASSWORD", ""),
        "bodopartner.us-east-1",
        "DEMO_WH",
        database,
        connection_params={"schema": schema},
        iceberg_volume=iceberg_volume,
    )
    bc = bodosql.BodoSQLContext(catalog=catalog)

    def impl(bc, query):
        return bc.sql(query)

    current_date = pd.Timestamp.now().date()
    # Use a large delta so we don't have to worry about the current date changing
    offsets = [-30, -20, -10, 10, 20, 30]
    column = [current_date + pd.Timedelta(days=offset) for offset in offsets]
    input_df = pd.DataFrame({"A": column})
    py_output = pd.DataFrame({"A": [x for x in column if x <= current_date]})
    with create_snowflake_iceberg_table(
        input_df, "current_date_table", database, schema, iceberg_volume
    ) as table_name:
        query = f"SELECT * FROM {table_name} WHERE A <= CURRENT_DATE"
        stream = StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_func(
                impl,
                (bc, query),
                py_output=py_output,
                sort_output=True,
                reset_index=True,
            )
            # Verify filter pushdown
            check_logger_msg(
                stream,
                "Iceberg Filter Pushed Down:\nFilterExpr('<=', [ColumnRef('A'), Scalar(f0)])",
            )
