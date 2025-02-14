"""Tests merge into operations for various filter pushdown expectations.
This includes testing that the target table is only filtered by files
and that the source table can still have proper filtering.
"""

import io

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.conftest import iceberg_database, iceberg_table_conn  # noqa
from bodo.tests.iceberg_database_helpers import spark_reader
from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLES_MAP,
)
from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table
from bodo.tests.tracing_utils import TracingContextManager
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func
from bodosql import BodoSQLContext, TablePath

pytestmark = [
    pytest.mark.iceberg,
    pytest.mark.skip(
        reason="[BSE-952] Current MERGE INTO Algorithm Designed for Non-Streaming and Doesn't Support Filter Pushdown w/ Streaming. This also doesn't work with the current volcano planner.",
    ),
]


@pytest.mark.slow
def test_filter_pushdown_target(iceberg_database, iceberg_table_conn):
    """
    Test that merge into with only the target table loaded from Iceberg
    demonstrates filter pushdown that only filters the files being loaded.
    """
    # TODO: Add memory leak check

    # Create the merge into function and read the result.
    def impl(bc):
        # Execute the merge into
        bc.sql(
            """
            Merge INTO target_table t
            USING source_table s
            ON t.A = s.A AND t.B < 4
            WHEN MATCHED THEN
                DELETE
        """
        )
        # Read the result
        return bc.sql("select * from target_table")

    # Create the Iceberg Table
    table_name = "MERGE_INTO_NUMERIC_TABLE1"
    expected_output, sql_schema = SIMPLE_TABLES_MAP["NUMERIC_TABLE"]
    if bodo.get_rank() == 0:
        create_iceberg_table(
            expected_output,
            sql_schema,
            table_name,
        )
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    # Generate the expected output
    a_vals = {-1, -7, -17, 1, 2, 3}
    # This filter matches the merge into condition
    filter = (expected_output.B < 4) & expected_output.A.isin(a_vals)
    expected_output = expected_output[~filter]

    # Create the BodoSQL context.
    bc = BodoSQLContext(
        {
            "TARGET_TABLE": TablePath(
                table_name, "sql", conn_str=conn, db_schema=db_schema
            ),
            "SOURCE_TABLE": pd.DataFrame({"A": [-1, -7, -17, 1, 2, 3]}),
        }
    )

    # Check filter pushdown when running the function
    tracing_info = TracingContextManager()
    with tracing_info:
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # We only test with 1DVar because merge into will alter the table contents.
            check_func(
                impl,
                (bc,),
                only_1DVar=True,
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
                check_dtype=False,
            )
            # The logger messages will occur twice, once for merge into and once for
            # the read. However, the filter pushdown must refer to the merge into.
            check_logger_msg(stream, "Filter pushdown successfully performed")

    # Load the tracing results
    dnf_filters = tracing_info.get_event_attribute(
        "get_iceberg_file_list", "g_dnf_filter", 0
    )
    expr_filters = tracing_info.get_event_attribute(
        "get_row_counts", "g_expr_filters", 0
    )
    # Verify we have dnf filters.
    assert dnf_filters != "None", "No DNF filters were pushed"
    # Verify we don't have expr filters
    assert expr_filters == "None", "Expr filters were pushed unexpectedly"


@pytest.mark.slow
def test_filter_pushdown_target_and_source(iceberg_database, iceberg_table_conn):
    """
    Test that merge into with the target table loaded from Iceberg
    and the source table being loaded from Iceberg results in full
    filter pushdown on the source table and file filter pushdown
    on the target table.
    """
    # TODO: Add memory leak check

    # Create the merge into function and read the result.
    def impl(bc):
        # Execute the merge into
        bc.sql(
            """
            Merge INTO target_table t
            USING source_table s
            ON t.A = s.A AND t.B < 4 AND s.A >= 2
            WHEN MATCHED THEN
                DELETE
        """
        )
        # Read the result
        return bc.sql("select * from target_table")

    # Select the iceberg table
    table_name = "MERGE_INTO_NUMERIC_TABLE2"
    # Create the table
    expected_output, sql_schema = SIMPLE_TABLES_MAP["NUMERIC_TABLE"]
    if bodo.get_rank() == 0:
        create_iceberg_table(
            expected_output,
            sql_schema,
            table_name,
        )
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    target_table_path = TablePath(table_name, "sql", conn_str=conn, db_schema=db_schema)

    # Generate the expected output
    a_vals = {2, 3}
    # This filter matches the merge into condition
    filter = (expected_output.B < 4) & expected_output.A.isin(a_vals)
    expected_output = expected_output[~filter]

    # open connection and create source table
    table_name = "SOURCE_TABLE_MERGE_INTO_PUSHDOWN"
    sql_schema = [("A", "bigint", False)]
    source_df = pd.DataFrame({"A": [-1, -7, -17, 1, 2, 3]})
    if bodo.get_rank() == 0:
        create_iceberg_table(
            source_df,
            sql_schema,
            table_name,
        )
    # Wait for the table to update in all ranks.
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    source_table_path = TablePath(table_name, "sql", conn_str=conn, db_schema=db_schema)

    # Create the BodoSQL context.
    bc = BodoSQLContext(
        {
            "TARGET_TABLE": target_table_path,
            "SOURCE_TABLE": source_table_path,
        }
    )

    # Check filter pushdown when running the function
    tracing_info = TracingContextManager()
    with tracing_info:
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # We only test with 1DVar because merge into will alter the table contents.
            check_func(
                impl,
                (bc,),
                only_1DVar=True,
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
                check_dtype=False,
            )
            # The logger messages will occur twice, once for merge into and once for
            # the read. However, the filter pushdown must refer to the merge into.
            check_logger_msg(stream, "Filter pushdown successfully performed")

    # Load the tracing results
    dnf_filters_target = tracing_info.get_event_attribute(
        "get_iceberg_file_list", "g_dnf_filter", 1
    )
    expr_filters_target = tracing_info.get_event_attribute(
        "get_row_counts", "g_expr_filters", 1
    )
    # Verify we have dnf filters.
    assert dnf_filters_target != "None", "No DNF filters were pushed"
    # Verify we don't have expr filters
    assert expr_filters_target == "None", "Expr filters were pushed unexpectedly"

    dnf_filters_source = tracing_info.get_event_attribute(
        "get_iceberg_file_list", "g_dnf_filter", 0
    )
    expr_filters_source = tracing_info.get_event_attribute(
        "get_row_counts", "g_expr_filters", 0
    )
    # Verify we have dnf filters.
    assert dnf_filters_source != "None", "No DNF filters were pushed"
    # Verify we don't have expr filters
    assert expr_filters_source != "None", "No Expr filters were pushed"


@pytest.mark.slow
def test_filter_pushdown_self_merge(iceberg_database, iceberg_table_conn):
    """
    Test that merge into with the target table and source table
    as the same table still results in loading Iceberg results
    with full filter pushdown on the source table and file filter
    pushdown on the target table. This does imply that two separate IO
    steps are required, which is likely acceptable.
    """
    # TODO: Add memory leak check

    # Create the merge into function and read the result.
    def impl(bc):
        # Execute the merge into
        bc.sql(
            """
            Merge INTO target_table t
            USING target_table s
            ON t.A = s.A AND s.B < 4 and t.B < 7
            WHEN MATCHED THEN
                DELETE
        """
        )
        # Read the result
        return bc.sql("select * from target_table")

    # open connection and create source table
    db_schema, warehouse_loc = iceberg_database()
    table_name = "MERGE_INTO_UNIQUE_TARGET"
    sql_schema = [("A", "bigint", False), ("B", "bigint", False)]
    source_df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10)})
    if bodo.get_rank() == 0:
        create_iceberg_table(
            source_df,
            sql_schema,
            table_name,
        )
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    target_table_path = TablePath(table_name, "sql", conn_str=conn, db_schema=db_schema)

    # Read the table from Spark.
    expected_output = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)

    # Generate the expected output
    # This filter matches the merge into condition
    filter = expected_output.B < 4
    expected_output = expected_output[~filter]

    # Create the BodoSQL context.
    bc = BodoSQLContext(
        {
            "TARGET_TABLE": target_table_path,
        }
    )

    # Check filter pushdown when running the function
    tracing_info = TracingContextManager()
    with tracing_info:
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # We only test with 1DVar because merge into will alter the table contents.
            check_func(
                impl,
                (bc,),
                only_1DVar=True,
                py_output=expected_output,
                reset_index=True,
                sort_output=True,
                check_dtype=False,
            )
            # The logger messages will occur twice, once for merge into and once for
            # the read. However, the filter pushdown must refer to the merge into.
            check_logger_msg(stream, "Filter pushdown successfully performed")

    # Load the tracing results
    dnf_filters_target = tracing_info.get_event_attribute(
        "get_iceberg_file_list", "g_dnf_filter", 1
    )
    expr_filters_target = tracing_info.get_event_attribute(
        "get_row_counts", "g_expr_filters", 1
    )
    # Verify we have dnf filters.
    assert dnf_filters_target != "None", "No DNF filters were pushed"
    # Verify we don't have expr filters
    assert expr_filters_target == "None", "Expr filters were pushed unexpectedly"
    dnf_filters_source = tracing_info.get_event_attribute(
        "get_iceberg_file_list", "g_dnf_filter", 0
    )
    expr_filters_source = tracing_info.get_event_attribute(
        "get_row_counts", "g_expr_filters", 0
    )
    # Verify we have dnf filters.
    assert dnf_filters_source != "None", "No DNF filters were pushed"
    # Verify we don't have expr filters
    assert expr_filters_source != "None", "No Expr filters were pushed"
