# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Basic E2E tests for each type of filter pushdown on Iceberg tables.
"""
import io

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.iceberg_database_helpers.utils import (
    PartitionField,
    create_iceberg_table,
)
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func, run_rank0

pytestmark = pytest.mark.iceberg


@pytest.mark.skip("Unable to use spark to write time data type")
def test_filter_pushdown_time(iceberg_database, iceberg_table_conn):
    table_name = "filter_pushdown_time_table"
    # Based on the documentation here: https://spark.apache.org/docs/latest/sql-ref-datatypes.html
    # Spark SQL does not support time data type, so we will have to write a custom script to handle
    # time.
    pass


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
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, f"Columns loaded {list(input_df.columns)}")
        check_logger_msg(stream, "Filter pushdown successfully performed")
        check_logger_msg(stream, "Files selected for read:")
        # I'm not sure why iceberg represents the directory like this,
        # when partitioning bytes objects, but it seems to be consistent
        check_logger_msg(stream, "bytes_col_file_filter=MA%3D%3D")
