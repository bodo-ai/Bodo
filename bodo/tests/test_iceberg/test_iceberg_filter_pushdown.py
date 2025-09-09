"""
Basic E2E tests for each type of filter pushdown on Iceberg tables.
"""

from __future__ import annotations

import datetime
import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pyiceberg.expressions as pie
import pytest
from pyiceberg.expressions.literals import TimeLiteral

import bodo
from bodo.tests.iceberg_database_helpers.utils import (
    PartitionField,
    create_iceberg_table,
)
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func, pytest_mark_one_rank, run_rank0

pytestmark = pytest.mark.iceberg


def _write_iceberg_table(input_df: pd.DataFrame, warehouse: str, table_id: str):
    from bodo.io.iceberg.catalog.dir import DirCatalog

    catalog = DirCatalog("write_catalog", warehouse=warehouse)
    table = catalog.create_table(table_id, pa.Schema.from_pandas(input_df))
    table.append(pa.table(input_df))


@pytest_mark_one_rank
def test_filter_pushdown_time_direct(iceberg_database, iceberg_table_conn):
    """
    Test that directly calls the filter pushdown functions to work around the time comparison
    issue (see test_filter_pushdown_time)
    """
    from bodo.io.iceberg.catalog import conn_str_to_catalog

    table_name = "filter_pushdown_time_table"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    input_df = pd.DataFrame(
        {
            "ID": np.arange(10),
            "time_col": [
                pa.scalar(datetime.time(i, i), type=pa.time64("us")) for i in range(10)
            ],
        }
    )

    table_id = f"{db_schema}.{table_name}"
    _write_iceberg_table(input_df, warehouse_loc, table_id)
    filter_expr = pie.NotEqualTo(
        "time_col", TimeLiteral((10 * 60 + 10) * 60 * 1_000_000)
    )

    from bodo.io.iceberg import get_iceberg_file_list_parallel

    catalog = conn_str_to_catalog(conn)

    get_iceberg_file_list_parallel(catalog, table_id, filter_expr)


@pytest.mark.skip(
    "Time array comparison operators not supported: https://bodo.atlassian.net/browse/BSE-3061"
)
def test_filter_pushdown_time(iceberg_database, iceberg_table_conn):
    table_name = "filter_pushdown_time_table_2"
    input_df = pd.DataFrame(
        {"ID": np.arange(10), "time_col": [bodo.types.Time(10, 10, precision=9)] * 10}
    )

    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    # Based on the documentation here: https://spark.apache.org/docs/latest/sql-ref-datatypes.html
    # Spark SQL does not support time data type, so for now, we just handle the write with Bodo itself.
    _write_iceberg_table(input_df, table_name, conn, db_schema)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df.time_col == bodo.types.Time(10, 10, precision=6)]
        return df

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=input_df,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, f"Columns loaded {list(input_df.columns)}")
        check_logger_msg(stream, "Filter pushdown successfully performed")


def test_filter_pushdown_binary(iceberg_database, iceberg_table_conn):
    """Simple test that with a filter that selects all of the data, which is stored in
    a single file"""
    table_name = "filter_pushdown_bytes_table"
    input_df = pd.DataFrame({"ID": np.arange(10), "bytes_col": [b"todo"] * 10})

    @run_rank0
    def setup():
        create_iceberg_table(
            input_df,
            [("ID", "int", False), ("bytes_col", "binary", False)],
            table_name,
        )

    setup()
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df.bytes_col == b"todo"]
        return df

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=input_df,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, f"Columns loaded {list(input_df.columns)}")
        check_logger_msg(stream, "Filter pushdown successfully performed")


@pytest.mark.slow
def test_filter_pushdown_binary_complex(iceberg_database, iceberg_table_conn):
    """
    More complex test with two filters that test row filtering, and file level
    filtering respectively.
    """
    table_name = "filter_pushdown_bytes_table_2"
    input_df = pd.DataFrame(
        {
            "ID": np.arange(10),
            "bytes_col_row_filter": [b"hello", b"world"] * 5,
            "bytes_col_file_filter": [b"0" for i in range(5)]
            + [b"1" for i in range(5)],
        }
    )

    @run_rank0
    def setup():
        create_iceberg_table(
            input_df,
            [
                ("ID", "int", False),
                ("bytes_col_row_filter", "binary", False),
                ("bytes_col_file_filter", "binary", False),
            ],
            table_name,
            par_spec=[PartitionField("bytes_col_file_filter", "identity", -1)],
        )

    setup()
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df.bytes_col_file_filter != b"1"]
        df = df[df.bytes_col_row_filter == b"hello"]
        return df

    output_df = input_df[input_df.bytes_col_file_filter != b"1"]
    output_df = output_df[output_df.bytes_col_row_filter == b"hello"]

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=output_df,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, f"Columns loaded {list(input_df.columns)}")
        check_logger_msg(stream, "Filter pushdown successfully performed")
        check_logger_msg(stream, "Total number of files is 2. Reading 1 files:")
        # I'm not sure why iceberg represents the directory like this,
        # when partitioning bytes objects, but it seems to be consistent
        check_logger_msg(stream, "bytes_col_file_filter=MA%3D%3D")


def test_filter_pushdown_logging_msg(iceberg_database, iceberg_table_conn):
    """Simple test to make sure that the logged messages for iceberg filter pushdown are correct"""

    ten_partition_table_name = "ten_partition_table"
    many_partition_table_name = "many_partition_table"
    input_df = pd.DataFrame(
        {
            "ID": np.arange(1000),
            "partition_col_1": list(range(10)) * 100,
            "partition_col_2": list(range(100)) * 10,
        }
    )

    @run_rank0
    def setup():
        create_iceberg_table(
            input_df,
            [
                ("ID", "int", False),
                ("partition_col_1", "int", False),
                ("partition_col_2", "int", False),
            ],
            ten_partition_table_name,
            par_spec=[PartitionField("partition_col_1", "identity", -1)],
        )

        create_iceberg_table(
            input_df,
            [
                ("ID", "int", False),
                ("partition_col_1", "int", False),
                ("partition_col_2", "int", False),
            ],
            many_partition_table_name,
            par_spec=[
                PartitionField("partition_col_1", "identity", -1),
                PartitionField("partition_col_2", "identity", -1),
            ],
        )

    setup()
    db_schema, warehouse_loc = iceberg_database()
    ten_parition_conn = iceberg_table_conn(
        ten_partition_table_name, db_schema, warehouse_loc, check_exists=False
    )
    many_partition_conn = iceberg_table_conn(
        many_partition_table_name, db_schema, warehouse_loc, check_exists=False
    )

    def impl_filter_none(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df.partition_col_1 != -1]
        return df

    def impl_filter_col_1(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df.partition_col_1 == 0]
        return df

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit(impl_filter_none)(
            ten_partition_table_name, ten_parition_conn, db_schema
        )
        check_logger_msg(stream, "Total number of files is 10. Reading 10 files:")
        for i in range(10):
            check_logger_msg(stream, f"partition_col_1={i}")

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit(impl_filter_none)(
            many_partition_table_name, many_partition_conn, db_schema
        )
        check_logger_msg(stream, "Total number of files is 100. Reading 100 files:")
        check_logger_msg(stream, "partition_col_1=")
        check_logger_msg(stream, "... and 90 more")

    # Check that we don't list the files with log level 1
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_filter_none)(
            many_partition_table_name, many_partition_conn, db_schema
        )
        check_logger_msg(stream, "Total number of files is 100. Reading 100 files.")
        check_logger_no_msg(stream, "partition_col_1=")

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit(impl_filter_col_1)(
            ten_partition_table_name, ten_parition_conn, db_schema
        )
        check_logger_msg(stream, "Total number of files is 10. Reading 1 files:")
        check_logger_msg(stream, "partition_col_1=0")

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit(impl_filter_col_1)(
            many_partition_table_name, many_partition_conn, db_schema
        )
        check_logger_msg(stream, "Total number of files is 100. Reading 10 files:")
        for i in range(10):
            check_logger_msg(stream, f"partition_col_1=0/partition_col_2={i * 10}")
