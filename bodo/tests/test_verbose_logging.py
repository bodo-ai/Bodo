# Copyright (C) 2022 Bodo Inc. All rights reserved.

"""
    Tests the logging done with the verbose JIT flag.
    This set of tests is focused on log input, NOT
    the correctness of the JIT code, which is presumed
    to be tested elsewhere.

    All logs are written solely on rank 0, and tests are
    written accordingly.
"""
import io

import pandas as pd

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import sql_user_pass_and_hostname


def test_json_column_pruning(datapath, memory_leak_check):
    """
    Tests that column pruning is logged with json.
    """

    @bodo.jit
    def test_impl(fname):
        df = pd.read_json(fname, orient="records", lines=True)
        return df.two

    fname_file = datapath("example.json")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        test_impl(fname_file)
        check_logger_msg(stream, "Columns loaded ['two']")


def test_csv_column_pruning(datapath, memory_leak_check):
    """
    Tests that column pruning is logged with csv.
    """

    @bodo.jit
    def test_impl(fname):
        df = pd.read_csv(fname, names=["A", "B", "C", "D"])
        return df.A

    fname_file = datapath("csv_data1.csv")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        test_impl(fname_file)
        check_logger_msg(stream, "Columns loaded ['A']")


def test_csv_iterator_column_pruning(datapath, memory_leak_check):
    """
    Tests that column pruning is logged with CSVIterator using
    chunksize. The CSVIterator doesn't actually prune
    columns, so this just tests that the columns loaded
    are in the output.
    """

    @bodo.jit
    def test_impl(fname):
        total_len = 0
        for df in pd.read_csv(fname, names=["A", "B", "C", "D"], chunksize=7):
            total_len += len(df)
        return total_len

    fname_file = datapath("csv_data1.csv")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        test_impl(fname_file)
        check_logger_msg(stream, "Columns loaded ['A', 'B', 'C', 'D']")


def test_sql_column_pruning(memory_leak_check):
    """
    Tests that column pruning is logged with sql.
    """

    @bodo.jit
    def test_impl():
        sql_request = "select * from employees limit 100"
        conn = "mysql+pymysql://" + sql_user_pass_and_hostname + "/employees"
        frame = pd.read_sql(sql_request, conn)
        return frame.gender

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        test_impl()
        check_logger_msg(stream, "Columns loaded ['gender']")


def test_pq_column_pruning_filter_pushdown(datapath, memory_leak_check):
    """
    Tests that column pruning and filter pushdown is
    logged with parquet.
    """

    @bodo.jit
    def test_impl(fname):
        df = pd.read_parquet(fname)
        df = df[df.one > 1]
        return df.four

    fname_file = datapath("example.parquet")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        test_impl(fname_file)
        # Check filter pushdown succeeded
        check_logger_msg(stream, "Filter pushdown successfully performed")
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['four']")


def test_pq_logging_closure(datapath, memory_leak_check):
    """
    Tests that column pruning and filter pushdown is
    logged with parquet + a closure.
    """

    @bodo.jit
    def test_impl(fname):
        def f():
            df = pd.read_parquet(fname)
            df = df[df.one > 1]
            return df.four

        return f()

    fname_file = datapath("example.parquet")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        test_impl(fname_file)
        # Check filter pushdown succeeded
        check_logger_msg(stream, "Filter pushdown successfully performed")
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['four']")


def test_pq_logging_multifunction(datapath, memory_leak_check):
    """
    Tests that column pruning and filter pushdown is
    logged with parquet + an extra function call.
    """

    @bodo.jit
    def f(fname):
        df = pd.read_parquet(fname)
        df = df[df.one > 1]
        return df.four

    @bodo.jit
    def test_impl(fname):
        return f(fname)

    fname_file = datapath("example.parquet")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        test_impl(fname_file)
        # Check filter pushdown succeeded
        check_logger_msg(stream, "Filter pushdown successfully performed")
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['four']")


def test_pq_logging_multifunction_inlining(datapath, memory_leak_check):
    """
    Tests that column pruning and filter pushdown is
    logged with parquet + an extra inlined function call.
    """

    @bodo.jit(inline="always")
    def f(fname):
        df = pd.read_parquet(fname)
        return df

    @bodo.jit
    def test_impl(fname):
        df = f(fname)
        df = df[df.one > 1]
        return df.four

    fname_file = datapath("example.parquet")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        test_impl(fname_file)
        # Check filter pushdown succeeded
        check_logger_msg(stream, "Filter pushdown successfully performed")
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['four']")
