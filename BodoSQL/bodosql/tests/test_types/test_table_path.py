# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests various components of the TablePath type both inside and outside a
direct BodoSQLContext.
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
    TypeInferenceTestPipeline,
    check_func,
    get_snowflake_connection_string,
    pytest_mark_snowflake,
)
from bodosql.tests.utils import _check_query_equal, check_num_parquet_readers


@pytest.fixture(
    params=[
        bodosql.TablePath("myfile.pq", "parquet"),
        bodosql.TablePath("example.pq", "pq"),
        bodosql.TablePath("caps.pq", "PARQUET"),
        bodosql.TablePath("my_table", "SQL", conn_str="test_str"),
        bodosql.TablePath("my_other_table", "sql", conn_str="dummy_str"),
    ]
)
def dummy_table_paths(request):
    """
    List of table paths that should be supported.
    None of these actually point to valid data
    """
    return request.param


@pytest.fixture(
    params=[
        "sample-parquet-data/no_index.pq",
        "sample-parquet-data/numeric_index.pq",
        "sample-parquet-data/string_index.pq",
    ]
)
def parquet_filepaths(request, datapath):
    """
    List of files used to load data.
    """
    return datapath(request.param)


@pytest.mark.parquet
@pytest.mark.slow
def test_table_path_lower_constant(dummy_table_paths, memory_leak_check):
    """
    Test lowering a constant table path.
    """

    def impl():
        return dummy_table_paths

    check_func(impl, ())


@pytest.mark.parquet
@pytest.mark.slow
def test_table_path_boxing(dummy_table_paths, memory_leak_check):
    """
    Test boxing and unboxing a table path type.
    """

    def impl(table_path):
        return table_path

    check_func(impl, (dummy_table_paths,))


@pytest.mark.parametrize("reorder_io", [True, False, None])
@pytest.mark.parquet
@pytest.mark.slow
def test_table_path_pq_constructor(reorder_io, memory_leak_check):
    """
    Test using the table path constructor from JIT.
    """

    def impl():
        return bodosql.TablePath("caps.pq", "parquet", reorder_io=reorder_io)

    check_func(impl, ())


@pytest.mark.parametrize("reorder_io", [True, False, None])
@pytest.mark.slow
@pytest.mark.parquet
def test_table_path_pq_bodosqlContext_python(
    reorder_io, parquet_filepaths, memory_leak_check
):
    """
    Test using the table path constructor inside a BodoSQLContext that is in Python.
    """

    def impl(filename):
        bc = bodosql.BodoSQLContext(
            {
                "PARQUET_TABLE": bodosql.TablePath(
                    filename, "parquet", reorder_io=reorder_io
                )
            }
        )
        return bc.sql("select * from parquet_table")

    filename = parquet_filepaths
    py_output = pd.read_parquet(filename)
    bodosql_output = impl(filename)
    _check_query_equal(
        bodosql_output,
        py_output,
        True,
        # BodoSQL loads columns as nullable
        False,
        False,
        False,
        "Reading a parquet file from BodoSQLContext doesn't match reading from Pandas",
        False,
    )


@pytest.mark.parquet
@pytest.mark.slow
@pytest.mark.parametrize("reorder_io", [True, False, None])
def test_table_path_pq_bodosqlContext_jit(
    reorder_io, parquet_filepaths, memory_leak_check
):
    """
    Test using the table path constructor inside a BodoSQLContext in JIT.
    """

    def impl(filename):
        bc = bodosql.BodoSQLContext(
            {
                "PARQUET_TABLE": bodosql.TablePath(
                    filename, "parquet", reorder_io=reorder_io
                )
            }
        )
        return bc.sql("select * from parquet_table")

    filename = parquet_filepaths

    # Should SQL still produce index columns?
    # Should it include an index column within the data?
    py_output = pd.read_parquet(filename).reset_index(drop=True)
    check_func(impl, (filename,), py_output=py_output)


@pytest_mark_snowflake
def test_table_path_sql_bodosqlContext_python(memory_leak_check):
    def impl(table_name, conn_str):
        bc = bodosql.BodoSQLContext(
            {"SQL_TABLE": bodosql.TablePath(table_name, "sql", conn_str=conn_str)}
        )
        return bc.sql(
            "select L_SUPPKEY from SQL_TABLE ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
        )

    table_name = "LINEITEM"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn_str = get_snowflake_connection_string(db, schema)
    py_output = pd.read_sql(
        f"select L_SUPPKEY from {table_name} ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70",
        conn_str,
    )
    bodosql_output = impl(table_name, conn_str)
    _check_query_equal(
        bodosql_output,
        py_output,
        # Names won't match because BodoSQL keep names uppercase.
        False,
        # BodoSQL loads columns with the actual Snowflake type
        False,
        False,
        False,
        "Reading a Snowflake Table from BodoSQLContext doesn't match reading from Pandas",
        False,
    )


@pytest_mark_snowflake
def test_table_path_sql_bodosqlContext_jit(memory_leak_check):
    def impl(table_name):
        bc = bodosql.BodoSQLContext(
            {"SQL_TABLE": bodosql.TablePath(table_name, "sql", conn_str=conn_str)}
        )
        return bc.sql("select L_SUPPKEY from sql_table ORDER BY L_SUPPKEY LIMIT 70")

    table_name = "LINEITEM"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn_str = get_snowflake_connection_string(db, schema)
    py_output = pd.read_sql(
        f"select L_SUPPKEY from {table_name} ORDER BY L_SUPPKEY LIMIT 70", conn_str
    )
    py_output.columns = py_output.columns.str.upper()
    # Set check_dtype=False because BodoSQL loads columns with the actual Snowflake type
    check_func(
        impl,
        (table_name,),
        py_output=py_output,
        reset_index=True,
        check_dtype=False,
    )


@pytest.mark.parquet
@pytest.mark.slow
def test_table_path_avoid_unused_table_jit(
    parquet_filepaths, datapath, memory_leak_check
):
    """
    Tests avoiding an unused table in JIT. We check the IR after BodoTypeInference
    pass to ensure there is only 1 ReadParquet IR Node.
    """

    def impl(f1, f2):
        bc = bodosql.BodoSQLContext(
            {
                "PARQUET_TABLE": bodosql.TablePath(f1, "parquet"),
                "UNUSED_TABLE": bodosql.TablePath(f2, "parquet"),
            }
        )
        return bc.sql("select * from parquet_table")

    f1 = parquet_filepaths
    f2 = datapath("tpcxbb-test-data") + "/web_sales"

    py_output = pd.read_parquet(f1).reset_index(drop=True)
    check_func(impl, (f1, f2), py_output=py_output)
    bodo_func = bodo.jit(impl, pipeline_class=TypeInferenceTestPipeline)
    bodo_func(f1, f2)
    check_num_parquet_readers(bodo_func, 1)


@pytest.mark.parquet
@pytest.mark.slow
@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
def test_table_path_avoid_unused_table_python(
    parquet_filepaths, datapath, memory_leak_check
):
    """
    Tests avoiding an unused table also works in Python.
    """

    def impl(f1, f2):
        bc = bodosql.BodoSQLContext(
            {
                "PARQUET_TABLE": bodosql.TablePath(f1, "parquet"),
                "UNUSED_TABLE": bodosql.TablePath(f2, "parquet"),
            }
        )
        return bc.sql("select * from parquet_table")

    f1 = parquet_filepaths
    f2 = datapath("tpcxbb-test-data") + "/web_sales"

    py_output = pd.read_parquet(f1)
    bodosql_output = impl(f1, f2)
    _check_query_equal(
        bodosql_output,
        py_output,
        True,
        # BodoSQL loads columns as nullable
        False,
        False,
        False,
        "Reading a parquet file from BodoSQLContext doesn't match reading from Pandas",
        False,
    )
    # TODO: Check the IR from Python.


@pytest.mark.parquet
@pytest.mark.slow
@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
def test_table_path_categorical_unused_table_jit(datapath, memory_leak_check):
    """
    Tests loading a partitioned parquet table in JIT.
    """

    def impl(f1, f2):
        bc = bodosql.BodoSQLContext(
            {
                "PARQUET_TABLE": bodosql.TablePath(f1, "parquet"),
                "UNUSED_TABLE": bodosql.TablePath(f2, "parquet"),
            }
        )
        return bc.sql("select * from parquet_table")

    f1 = datapath("sample-parquet-data/partitioned")
    f2 = datapath("tpcxbb-test-data/web_sales")

    py_output = pd.read_parquet(f1)
    py_output["part"] = py_output["part"].astype(str)
    check_func(impl, (f1, f2), py_output=py_output)
    bodo_func = bodo.jit(impl, pipeline_class=TypeInferenceTestPipeline)
    bodo_func(f1, f2)
    check_num_parquet_readers(bodo_func, 1)


@pytest.mark.parquet
@pytest.mark.slow
@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
def test_table_path_categorical_unused_table_python(datapath, memory_leak_check):
    """
    Tests loading a partitioned parquet table in Python.
    """

    def impl(f1, f2):
        bc = bodosql.BodoSQLContext(
            {
                "PARQUET_TABLE": bodosql.TablePath(f1, "parquet"),
                "UNUSED_TABLE": bodosql.TablePath(f2, "parquet"),
            }
        )
        return bc.sql("select * from parquet_table")

    f1 = datapath("sample-parquet-data") + "/partitioned"
    f2 = datapath("tpcxbb-test-data") + "/web_sales"

    py_output = pd.read_parquet(f1)
    py_output["part"] = py_output["part"].astype(str)
    bodosql_output = impl(f1, f2)
    _check_query_equal(
        bodosql_output,
        py_output,
        True,
        # BodoSQL loads columns as nullable
        False,
        False,
        False,
        "Reading a parquet file from BodoSQLContext doesn't match reading from Pandas",
        False,
    )
    # TODO: Check the IR from Python.


@pytest.mark.parquet
@pytest.mark.slow
@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
def test_table_path_timing_debug_message(datapath, memory_leak_check):
    """
    Tests that loading a table using TablePath with bodo.set_verbose_level(1)
    automatically adds a debug message about IO.
    """

    @bodo.jit
    def impl(f1):
        bc = bodosql.BodoSQLContext(
            {
                "PARQUET_TABLE": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("select * from parquet_table")

    f1 = datapath("sample-parquet-data/partitioned")

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        impl(f1)
        check_logger_msg(stream, "Execution time for reading table parquet_table")


@pytest.mark.parquet
@pytest.mark.slow
def test_parquet_row_count_estimation(datapath, memory_leak_check):
    """
    Tests that loading a parquet table using TablePath produces a row count estimation.
    """

    f1 = datapath("sample-parquet-data/partitioned")
    bc = bodosql.BodoSQLContext(
        {
            "PARQUET_TABLE": bodosql.TablePath(f1, "parquet"),
        }
    )

    assert bc.estimated_row_counts == [10]


# TODO: Add a test with multiple tables to check reorder_io works as expected.
