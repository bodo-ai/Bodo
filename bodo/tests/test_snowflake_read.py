# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests for reading from Snowflake using Python APIs
"""
import datetime
import io
import json
import os
import urllib

import pandas as pd
import pytest

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func, get_snowflake_connection_string
from bodo.utils.typing import BodoWarning


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake(memory_leak_check):
    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    check_func(impl, (query, conn))


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_bodo_read_as_dict(memory_leak_check):
    """
    Test reading string columns as dictionary-encoded from Snowflake
    """

    @bodo.jit
    def test_impl0(query, conn):
        return pd.read_sql(query, conn)

    @bodo.jit
    def test_impl1(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_shipmode"])

    @bodo.jit
    def test_impl2(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_shipinstruct"])

    @bodo.jit
    def test_impl3(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_comment"])

    @bodo.jit
    def test_impl4(query, conn):
        return pd.read_sql(
            query, conn, _bodo_read_as_dict=["l_shipmode", "l_shipinstruct"]
        )

    @bodo.jit
    def test_impl5(query, conn):
        return pd.read_sql(
            query, conn, _bodo_read_as_dict=["l_comment", "l_shipinstruct"]
        )

    @bodo.jit
    def test_impl6(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_comment", "l_shipmode"])

    @bodo.jit
    def test_impl7(query, conn):
        return pd.read_sql(
            query,
            conn,
            _bodo_read_as_dict=["l_shipmode", "l_comment", "l_shipinstruct"],
        )

    # 'l_suppkey' shouldn't be read as dictionary encoded since it's not a string column

    @bodo.jit
    def test_impl8(query, conn):
        return pd.read_sql(
            query,
            conn,
            _bodo_read_as_dict=[
                "l_shipmode",
                "l_comment",
                "l_shipinstruct",
                "l_suppkey",
            ],
        )

    @bodo.jit
    def test_impl9(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_suppkey"])

    @bodo.jit
    def test_impl10(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_suppkey", "l_comment"])

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    # l_shipmode, l_shipinstruct should be dictionary encoded by default
    # l_comment could be specified by the user to be dictionary encoded
    # l_suppkey is not of type string and could not be dictionary encoded
    query = "SELECT l_shipmode, l_shipinstruct, l_comment, l_suppkey FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 3000"
    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 1):
        test_impl0(query, conn)
        check_logger_msg(
            stream, "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding"
        )
    with set_logging_stream(logger, 1):
        test_impl1(query, conn)
        check_logger_msg(
            stream, "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding"
        )

    with set_logging_stream(logger, 1):
        test_impl2(query, conn)
        check_logger_msg(
            stream, "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding"
        )

    with set_logging_stream(logger, 1):
        test_impl3(query, conn)
        check_logger_msg(
            stream,
            "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
        )

    with set_logging_stream(logger, 1):
        test_impl4(query, conn)
        check_logger_msg(
            stream, "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding"
        )

    with set_logging_stream(logger, 1):
        test_impl5(query, conn)
        check_logger_msg(
            stream,
            "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
        )

    with set_logging_stream(logger, 1):
        test_impl6(query, conn)
        check_logger_msg(
            stream,
            "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
        )

    with set_logging_stream(logger, 1):
        test_impl7(query, conn)
        check_logger_msg(
            stream,
            "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
        )

    with set_logging_stream(logger, 1):
        if bodo.get_rank() == 0:  # warning is thrown only on rank 0
            with pytest.warns(
                BodoWarning,
                match="The following columns are not of datatype string and hence cannot be read with dictionary encoding: {'l_suppkey'}",
            ):
                test_impl8(query, conn)
        else:
            test_impl8(query, conn)
        # we combine the two tests because otherwise caching would cause problems for logger.stream.
        check_logger_msg(
            stream,
            "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
        )

    with set_logging_stream(logger, 1):
        if bodo.get_rank() == 0:
            with pytest.warns(
                BodoWarning,
                match="The following columns are not of datatype string and hence cannot be read with dictionary encoding: {'l_suppkey'}",
            ):
                test_impl9(query, conn)
        else:
            test_impl9(query, conn)
        check_logger_msg(
            stream, "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding"
        )

    with set_logging_stream(logger, 1):
        if bodo.get_rank() == 0:  # warning is thrown only on rank 0
            with pytest.warns(
                BodoWarning,
                match="The following columns are not of datatype string and hence cannot be read with dictionary encoding: {'l_suppkey'}",
            ):
                test_impl10(query, conn)
        else:
            test_impl10(query, conn)
        check_logger_msg(
            stream,
            "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
        )


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requres Azure Pipelines")
@pytest.mark.parametrize("enable_dict_encoding", [True, False])
def test_sql_snowflake_dict_encoding_enabled(memory_leak_check, enable_dict_encoding):
    """
    Test that SF_READ_AUTO_DICT_ENCODE_ENABLED works as expected.
    """
    import bodo.io.snowflake

    # need to sort the output to make sure pandas and Bodo get the same rows
    # l_shipmode, l_shipinstruct should be dictionary encoded based on Snowflake
    # probe query.
    # l_comment could be specified by the user to be dictionary encoded
    # l_suppkey is not of type string and could not be dictionary encoded
    query = "SELECT l_shipmode, l_shipinstruct, l_comment, l_suppkey FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 3000"
    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    @bodo.jit
    def test_impl(query, conn):
        return pd.read_sql(query, conn)

    @bodo.jit
    def test_impl_with_forced_dict_encode(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_comment"])

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    orig_SF_READ_AUTO_DICT_ENCODE_ENABLED = (
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED
    )
    bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = enable_dict_encoding

    try:
        if enable_dict_encoding:
            # check that dictionary encoding works
            with set_logging_stream(logger, 1):
                test_impl(query, conn)
                check_logger_msg(
                    stream,
                    "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
                )

            # Verify that _bodo_read_as_dict still works as expected
            with set_logging_stream(logger, 1):
                test_impl_with_forced_dict_encode(query, conn)
                check_logger_msg(
                    stream,
                    "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
                )
        else:
            # check that dictionary encoding is disabled
            with set_logging_stream(logger, 1):
                test_impl(query, conn)
                check_logger_no_msg(
                    stream,
                    "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
                )

            # Verify that _bodo_read_as_dict still works as expected
            with set_logging_stream(logger, 1):
                test_impl_with_forced_dict_encode(query, conn)
                check_logger_msg(
                    stream,
                    "Columns ['l_comment'] using dictionary encoding",
                )

    finally:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = (
            orig_SF_READ_AUTO_DICT_ENCODE_ENABLED
        )


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_nonascii(memory_leak_check):
    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM NONASCII_T1"
    check_func(impl, (query, conn), reset_index=True, sort_output=True)


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_single_column(memory_leak_check):
    """
    Test that loading using a single column from snowflake has a correct result
    that reduces the number of columns that need loading.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df["l_suppkey"]

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_use_index(memory_leak_check):
    """
    Tests loading using index_col with pd.read_sql from snowflake
    has a correct result and only loads the columns that need loading.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn, index_col="l_partkey")
        # Returns l_suppkey and the index
        return df["l_suppkey"]

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey', 'l_partkey']")


# TODO: Re-add this test once [BE-2758] is resolved
# @pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
@pytest.mark.skip(reason="Outdated index returned by pandas")
def test_sql_snowflake_use_index_dead_table(memory_leak_check):
    """
    Tests loading using index_col with pd.read_sql from snowflake
    where all columns are dead.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn, index_col="l_partkey")
        # Returns just the index
        return df.index

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_partkey']")


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_no_index_dead_table(memory_leak_check):
    """
    Tests loading with pd.read_sql from snowflake
    where all columns are dead. This should load
    0 columns.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        # Returns just the index
        return df.index

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned. l_orderkey is determined
        # by testing and we just need to confirm it loads a single column.
        check_logger_msg(stream, "Columns loaded []")


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_use_index_dead_index(memory_leak_check):
    """
    Tests loading using index_col with pd.read_sql from snowflake
    where the index is dead.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn, index_col="l_partkey")
        # Returns just l_suppkey array
        return df["l_suppkey"].values

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_count(memory_leak_check):
    """
    Test that using a sql function without an alias doesn't cause issues with
    dead column elimination.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        # TODO: Pandas loads count(*) as COUNT(*) but we can't detect this difference
        # and load it as count(*)
        df.columns = [x.lower() for x in df.columns]
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT L_ORDERKEY, count(*) FROM LINEITEM GROUP BY L_ORDERKEY ORDER BY L_ORDERKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_filter_pushdown(memory_leak_check):
    """
    Test that filter pushdown works properly with a variety of data types.
    """

    def impl_integer(query, conn, int_val):
        df = pd.read_sql(query, conn)
        df = df[(df["l_orderkey"] > 10) & (int_val >= df["l_linenumber"])]
        return df["l_suppkey"]

    def impl_string(query, conn, str_val):
        df = pd.read_sql(query, conn)
        df = df[(df["l_linestatus"] != str_val) | (df["l_shipmode"] == "FOB")]
        return df["l_suppkey"]

    def impl_date(query, conn, date_val):
        df = pd.read_sql(query, conn)
        df = df[date_val > df["l_shipdate"]]
        return df["l_suppkey"]

    def impl_timestamp(query, conn, ts_val):
        df = pd.read_sql(query, conn)
        # Note when comparing to date Pandas will truncate the timestamp.
        # This comparison is deprecated in general.
        df = df[ts_val <= df["l_shipdate"]]
        return df["l_suppkey"]

    def impl_mixed(query, conn, int_val, str_val, date_val, ts_val):
        """
        Test a query with mixed parameter types.
        """
        df = pd.read_sql(query, conn)
        df = df[
            ((df["l_linenumber"] <= int_val) | (date_val > df["l_shipdate"]))
            | ((ts_val <= df["l_shipdate"]) & (df["l_linestatus"] != str_val))
        ]
        return df["l_suppkey"]

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    # Reset index because Pandas applies the filter later

    int_val = 2
    check_func(
        impl_integer, (query, conn, int_val), check_dtype=False, reset_index=True
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_integer)(query, conn, int_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    str_val = "O"
    check_func(impl_string, (query, conn, str_val), check_dtype=False, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_string)(query, conn, str_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    date_val = datetime.date(1996, 4, 12)
    check_func(impl_date, (query, conn, date_val), check_dtype=False, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_date)(query, conn, date_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    ts_val = pd.Timestamp(year=1997, month=4, day=12)
    check_func(
        impl_timestamp, (query, conn, ts_val), check_dtype=False, reset_index=True
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_timestamp)(query, conn, ts_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    check_func(
        impl_mixed,
        (query, conn, int_val, str_val, date_val, ts_val),
        check_dtype=False,
        reset_index=True,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_mixed)(query, conn, int_val, str_val, date_val, ts_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_na_pushdown(memory_leak_check):
    """
    Test that filter pushdown with isna/notna/isnull/notnull works in snowflake.
    """

    def impl_or_isna(query, conn):
        df = pd.read_sql(query, conn)
        df = df[(df["l_orderkey"] > 10) | (df["l_linenumber"].isna())]
        return df["l_suppkey"]

    def impl_and_notna(query, conn):
        df = pd.read_sql(query, conn)
        df = df[(df["l_orderkey"] > 10) & (df["l_linenumber"].notna())]
        return df["l_suppkey"]

    def impl_or_isnull(query, conn):
        df = pd.read_sql(query, conn)
        df = df[(df["l_orderkey"] > 10) | (df["l_linenumber"].isnull())]
        return df["l_suppkey"]

    def impl_and_notnull(query, conn):
        df = pd.read_sql(query, conn)
        df = df[(df["l_orderkey"] > 10) & (df["l_linenumber"].notnull())]
        return df["l_suppkey"]

    def impl_just_nona(query, conn):
        df = pd.read_sql(query, conn)
        df = df[(df["l_linenumber"].notna())]
        return df["l_suppkey"]

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    check_func(impl_or_isna, (query, conn), check_dtype=False, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_or_isna)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    check_func(impl_and_notna, (query, conn), check_dtype=False, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_and_notna)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    check_func(impl_or_isnull, (query, conn), check_dtype=False, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_or_isnull)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    check_func(impl_and_notnull, (query, conn), check_dtype=False, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_and_notnull)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    check_func(impl_just_nona, (query, conn), check_dtype=False, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_just_nona)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_isin_pushdown(memory_leak_check):
    """
    Test that filter pushdown with isna/notna/isnull/notnull works in snowflake.
    """

    def impl_isin(query, conn, isin_list):
        df = pd.read_sql(query, conn)
        df = df[df["l_orderkey"].isin(isin_list)]
        return df["l_suppkey"]

    def impl_isin_or(query, conn, isin_list):
        df = pd.read_sql(query, conn)
        df = df[(df["l_shipmode"] == "FOB") | df["l_orderkey"].isin(isin_list)]
        return df["l_suppkey"]

    def impl_isin_and(query, conn, isin_list):
        df = pd.read_sql(query, conn)
        df = df[(df["l_shipmode"] == "FOB") & df["l_orderkey"].isin(isin_list)]
        return df["l_suppkey"]

    isin_list = [32, 35]
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    check_func(
        impl_isin,
        (query, conn, isin_list),
        check_dtype=False,
        reset_index=True,
        use_dict_encoded_strings=False,
    )
    # TODO: BE-3404: Support `pandas.Series.isin` for dictionary-encoded arrays
    prev_criterion = bodo.io.snowflake.SF_READ_DICT_ENCODE_CRITERION
    bodo.io.snowflake.SF_READ_DICT_ENCODE_CRITERION = -1
    try:
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(impl_isin)(query, conn, isin_list)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")

        check_func(
            impl_isin_or, (query, conn, isin_list), check_dtype=False, reset_index=True
        )
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(impl_isin_or)(query, conn, isin_list)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")

        check_func(
            impl_isin_and, (query, conn, isin_list), check_dtype=False, reset_index=True
        )
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(impl_isin_and)(query, conn, isin_list)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")
    finally:
        bodo.io.snowflake.SF_READ_DICT_ENCODE_CRITERION = prev_criterion


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_startswith_endswith_pushdown(memory_leak_check):
    """
    Test that filter pushdown with startswith/endswith works in snowflake.
    """

    def impl_startswith(query, conn, starts_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.startswith(starts_val)]
        return df["l_suppkey"]

    def impl_startswith_or(query, conn, starts_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.startswith(starts_val) | (df["l_orderkey"] == 32)]
        return df["l_suppkey"]

    def impl_startswith_and(query, conn, starts_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.startswith(starts_val) & (df["l_orderkey"] == 32)]
        return df["l_suppkey"]

    def impl_endswith(query, conn, ends_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.endswith(ends_val)]
        return df["l_suppkey"]

    def impl_endswith_or(query, conn, ends_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.endswith(ends_val) | (df["l_orderkey"] == 32)]
        return df["l_suppkey"]

    def impl_endswith_and(query, conn, ends_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.endswith(ends_val) & (df["l_orderkey"] == 32)]
        return df["l_suppkey"]

    starts_val = "AIR"
    ends_val = "AIL"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    for func in (impl_startswith, impl_startswith_or, impl_startswith_and):
        check_func(func, (query, conn, starts_val), check_dtype=False, reset_index=True)
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(func)(query, conn, starts_val)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")

    for func in (impl_endswith, impl_endswith_or, impl_endswith_and):
        check_func(func, (query, conn, ends_val), check_dtype=False, reset_index=True)
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(func)(query, conn, ends_val)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_sql_snowflake_json_url(memory_leak_check):
    """
    Check running a snowflake query with a dictionary for connection parameters
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    username = os.environ["SF_USER"]
    password = os.environ["SF_PASSWORD"]
    account = "bodopartner.us-east-1"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    connection_params = {
        "warehouse": "DEMO_WH",
        "session_parameters": json.dumps({"JSON_INDENT": 0}),
        "paramstyle": "pyformat",
        "insecure_mode": True,
    }
    conn = f"snowflake://{username}:{password}@{account}/{db}/{schema}?{urllib.parse.urlencode(connection_params)}"
    # session_parameters bug exists in sqlalchemy/snowflake connector
    del connection_params["session_parameters"]
    pandas_conn = f"snowflake://{username}:{password}@{account}/{db}/{schema}?{urllib.parse.urlencode(connection_params)}"
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    check_func(impl, (query, conn), py_output=impl(query, pandas_conn))


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_timezones(memory_leak_check):
    """
    Tests trying to read Arrow timestamp columns with
    timezones using Bodo + Snowflake succeeds.

    Note: tz_test was manually created in our snowflake account.
    """

    def test_impl1(query, conn_str):
        """
        read_sql that should succeed
        and filters out tz columns.
        """
        df = pd.read_sql(query, conn_str)
        return df.b

    def test_impl2(query, conn_str):
        """
        Read parquet loading a single tz column.
        """
        df = pd.read_sql(query, conn_str)
        return df.a

    def test_impl3(query, conn_str):
        """
        Read parquet loading t columns.
        """
        df = pd.read_sql(query, conn_str)
        return df

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    full_query = f"select * from tz_test"
    partial_query = f"select B, C from tz_test"
    # Loading just the non-tz columns should suceed.
    check_func(test_impl1, (full_query, conn), check_dtype=False)
    check_func(test_impl1, (partial_query, conn), check_dtype=False)
    check_func(test_impl2, (full_query, conn), check_dtype=False)
    check_func(test_impl3, (full_query, conn), check_dtype=False)
    check_func(test_impl3, (partial_query, conn), check_dtype=False)


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_empty_typing(memory_leak_check):
    """
    Tests support for read_sql when typing a query returns an empty DataFrame.
    """

    def test_impl(query, conn):
        return pd.read_sql(query, conn)

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT L_ORDERKEY FROM LINEITEM WHERE L_ORDERKEY IN (10, 11)"
    check_func(test_impl, (query, conn))


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_empty_filter(memory_leak_check):
    """
    Tests support for read_sql when a query returns an empty DataFrame via filter pushdown.
    """

    def test_impl(query, conn):
        df = pd.read_sql(query, conn)
        df = df[df["l_orderkey"] == 10]
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT L_ORDERKEY FROM LINEITEM"
    check_func(test_impl, (query, conn), check_dtype=False)


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_dead_node(memory_leak_check):
    """
    Tests when read_sql should be eliminated from the code.
    """

    def test_impl(query, conn):
        # This query should be optimized out.
        df = pd.read_sql(query, conn)
        return 1

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT L_ORDERKEY FROM LINEITEM"
    check_func(test_impl, (query, conn))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(test_impl)(query, conn)
        # Check that no Columns loaded message occurred,
        # so the whole node was deleted.
        check_logger_no_msg(stream, "Columns loaded")


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_zero_cols(memory_leak_check):
    """
    Tests when read_sql's table should load 0 columns.
    """

    def test_impl(query, conn):
        df = pd.read_sql(query, conn)
        return len(df)

    def test_impl_index(query, conn):
        # Test only loading an index
        df = pd.read_sql(query, conn, index_col="l_orderkey")
        return len(df), df.index.min()

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # We only load 1 column because Pandas is loading all of the data.
    query = "SELECT L_ORDERKEY FROM LINEITEM"
    check_func(
        test_impl,
        (
            query,
            conn,
        ),
        only_seq=True,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(test_impl)(query, conn)
        # Check that no columns were loaded.
        check_logger_msg(stream, "Columns loaded []")

    check_func(test_impl_index, (query, conn))
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(test_impl_index)(query, conn)
        # Check that we only load the index.
        check_logger_msg(stream, "Columns loaded ['l_orderkey']")
