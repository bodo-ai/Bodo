# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests for reading from Snowflake using streaming APIs
"""

import io

import numpy as np
import pandas as pd
import pytest

import bodo
import bodo.io.snowflake
import bodosql
from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next
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

pytestmark = pytest_snowflake


def test_streaming_read_filter(memory_leak_check):
    """
    Tests batched Snowflake reads with filter pushdown
    """

    def impl(conn):
        total_len = 0

        reader = pd.read_sql(
            "SELECT * FROM LINEITEM",
            conn,
            _bodo_chunksize=4000,
            _bodo_read_as_table=True,
        )  # type: ignore
        is_last_global = False
        while not is_last_global:
            T1, is_last = read_arrow_next(reader, True)
            T2 = T1[
                bodosql.kernels.equal(bodo.hiframes.table.get_table_data(T1, 14), "AIR")
            ]
            total_len += len(T2)
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )

        arrow_reader_del(reader)
        return total_len

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (conn,), py_output=858104)
        check_logger_msg(stream, "Filter pushdown successfully performed")
        check_logger_msg(stream, "Columns loaded []")


def test_streaming_read_agg(memory_leak_check):
    """
    Test a simple use of batched Snowflake reads by
    getting the max of a column
    """

    def impl(conn):
        total_max = 0

        reader = pd.read_sql(
            "SELECT * FROM LINEITEM",
            conn,
            _bodo_chunksize=4000,
            _bodo_read_as_table=True,
        )  # type: ignore
        is_last_global = False
        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            filtered_table = table[(bodo.hiframes.table.get_table_data(table, 0) > 10)]
            # Perform more compute in between to see caching speedup
            local_max = pd.Series(
                bodo.hiframes.table.get_table_data(filtered_table, 1)
            ).max()
            total_max = max(total_max, local_max)
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )

        arrow_reader_del(reader)
        return total_max

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (conn,), py_output=200000)
        check_logger_msg(stream, "Filter pushdown successfully performed")
        check_logger_msg(stream, "Columns loaded ['l_partkey']")


def test_batched_read_only_len(memory_leak_check):
    """
    Test shape pushdown with batched Snowflake reads
    """

    def impl(conn):
        total_len = 0

        reader = pd.read_sql("SELECT * FROM LINEITEM", conn, _bodo_chunksize=4000)  # type: ignore
        while True:
            table, is_last = read_arrow_next(reader, True)
            total_len += len(table)

            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            if is_last_global:
                break

        arrow_reader_del(reader)
        return total_len

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (conn,), py_output=6001215)
        check_logger_msg(stream, "Columns loaded []")


def test_batched_read_limit_pushdown_query(memory_leak_check):
    """
    Test shape pushdown with batched Snowflake reads
    """

    def impl(conn):
        total_sum = 0

        reader = pd.read_sql(
            "SELECT * FROM LINEITEM ORDER BY L_PARTKEY LIMIT 100",
            conn,
            _bodo_chunksize=4000,
        )  # type: ignore
        while True:
            table, is_last = read_arrow_next(reader, True)
            total_sum += pd.Series(bodo.hiframes.table.get_table_data(table, 1)).sum()

            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            if is_last_global:
                break

        arrow_reader_del(reader)
        return total_sum

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    # Python Computation
    py_sum = pd.read_sql(
        "SELECT L_PARTKEY FROM LINEITEM ORDER BY L_PARTKEY LIMIT 100", conn
    )["l_partkey"].sum()

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (conn,), py_output=py_sum)
        check_logger_msg(stream, "Columns loaded ['l_partkey']")


def test_batched_read_dict_encoding(memory_leak_check):
    """
    Test that batched SF read works with dictionary encoding.
    """

    def impl(conn):
        total_length = 0
        is_last_global = False

        reader = pd.read_sql(
            "SELECT l_shipmode FROM LINEITEM", conn, _bodo_chunksize=4000
        )  # type: ignore
        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            total_length += (
                pd.Series(bodo.hiframes.table.get_table_data(table, 0)).str.len().sum()
            )
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
        arrow_reader_del(reader)
        return total_length

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (conn,), py_output=25717034)
        check_logger_msg(stream, "Columns ['l_shipmode'] using dictionary encoding")
    print(stream.getvalue())


@pytest.mark.slow
def test_batched_read_produce_output(memory_leak_check):
    """
    Test that no output is produced if produce_output=False
    """

    @bodo.jit
    def impl(conn):
        reader = pd.read_sql("SELECT * FROM LINEITEM", conn, _bodo_chunksize=4000)  # type: ignore
        is_last = False
        _temp1 = 0
        out_tables = []
        output_when_not_request_input = False
        while not is_last:
            produce_output = _temp1 != 0
            table, is_last = read_arrow_next(reader, produce_output)
            output_when_not_request_input = output_when_not_request_input or (
                _temp1 == 0 and len(table) != 0
            )
            _temp1 += 1
            is_last = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )
            out_tables.append(table)
        arrow_reader_del(reader)
        return output_when_not_request_input

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # Ensure that the output is empty if produce_output is False
    assert not impl(conn)
