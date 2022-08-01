# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Tests various components of the TablePath type both inside and outside a
direct BodoSQLContext.
"""
import os

import bodo
import pandas as pd
import pytest
from bodo.tests.utils import TypeInferenceTestPipeline, check_func

import bodosql
from bodosql.tests.utils import (
    _check_query_equal,
    bodo_version_older,
    check_num_parquet_readers,
    get_snowflake_connection_string,
)


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
    List of table paths that should be suppported.
    None of these actually point to valid data
    """
    return request.param


@pytest.fixture(
    params=[
        "bodosql/tests/data/sample-parquet-data/no_index.pq",
        "bodosql/tests/data/sample-parquet-data/numeric_index.pq",
        "bodosql/tests/data/sample-parquet-data/string_index.pq",
    ]
)
def parquet_filepaths(request):
    """
    List of files used to load data.
    """
    return request.param


@pytest.mark.slow
def test_table_path_lower_constant(dummy_table_paths, memory_leak_check):
    """
    Test lowering a constant table path.
    """

    def impl():
        return dummy_table_paths

    check_func(impl, ())


@pytest.mark.slow
def test_table_path_boxing(dummy_table_paths, memory_leak_check):
    """
    Test boxing and unboxing a table path type.
    """

    def impl(table_path):
        return table_path

    check_func(impl, (dummy_table_paths,))


@pytest.mark.parametrize("reorder_io", [True, False, None])
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
def test_table_path_pq_bodosqlContext_python(
    reorder_io, parquet_filepaths, datapath, memory_leak_check
):
    """
    Test using the table path constructor inside a BodoSQLContext that is in Python.
    """

    def impl(filename):
        bc = bodosql.BodoSQLContext(
            {
                "parquet_table": bodosql.TablePath(
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


@pytest.mark.slow
@pytest.mark.parametrize("reorder_io", [True, False, None])
def test_table_path_pq_bodosqlContext_jit(
    reorder_io, parquet_filepaths, datapath, memory_leak_check
):
    """
    Test using the table path constructor inside a BodoSQLContext in JIT.
    """

    def impl(filename):
        bc = bodosql.BodoSQLContext(
            {
                "parquet_table": bodosql.TablePath(
                    filename, "parquet", reorder_io=reorder_io
                )
            }
        )
        return bc.sql("select * from parquet_table")

    filename = parquet_filepaths

    py_output = pd.read_parquet(filename)
    check_func(impl, (filename,), py_output=py_output)


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ or bodo_version_older(2022, 2, 1),
    reason="requires Azure Pipelines",
)
def test_table_path_sql_bodosqlContext_python(memory_leak_check):
    def impl(table_name, conn_str):
        bc = bodosql.BodoSQLContext(
            {"sql_table": bodosql.TablePath(table_name, "sql", conn_str=conn_str)}
        )
        return bc.sql(
            "select L_SUPPKEY from sql_table ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
        )

    table_name = "lineitem"
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


@pytest.mark.skipif(
    "AGENT_NAME" not in os.environ or bodo_version_older(2022, 2, 1),
    reason="requires Azure Pipelines",
)
def test_table_path_sql_bodosqlContext_jit(memory_leak_check):
    def impl(table_name):
        bc = bodosql.BodoSQLContext(
            {"sql_table": bodosql.TablePath(table_name, "sql", conn_str=conn_str)}
        )
        return bc.sql("select L_SUPPKEY from sql_table ORDER BY L_SUPPKEY LIMIT 70")

    table_name = "lineitem"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn_str = get_snowflake_connection_string(db, schema)
    py_output = pd.read_sql(
        f"select L_SUPPKEY from {table_name} ORDER BY L_SUPPKEY LIMIT 70", conn_str
    )
    # Names won't match because BodoSQL keep names uppercase.
    py_output.columns = ["L_SUPPKEY"]
    # Set check_dtype=False because BodoSQL loads columns with the actual Snowflake type
    check_func(
        impl,
        (table_name,),
        py_output=py_output,
        reset_index=True,
        check_dtype=False,
    )


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
                "parquet_table": bodosql.TablePath(f1, "parquet"),
                "unused_table": bodosql.TablePath(f2, "parquet"),
            }
        )
        return bc.sql("select * from parquet_table")

    f1 = parquet_filepaths
    f2 = datapath("tpcxbb-test-data") + "/web_sales"

    py_output = pd.read_parquet(f1)
    check_func(impl, (f1, f2), py_output=py_output)
    bodo_func = bodo.jit(impl, pipeline_class=TypeInferenceTestPipeline)
    bodo_func(f1, f2)
    check_num_parquet_readers(bodo_func, 1)


@pytest.mark.slow
def test_table_path_avoid_unused_table_python(
    parquet_filepaths, datapath, memory_leak_check
):
    """
    Tests avoiding an unused table also works in Python.
    """

    def impl(f1, f2):
        bc = bodosql.BodoSQLContext(
            {
                "parquet_table": bodosql.TablePath(f1, "parquet"),
                "unused_table": bodosql.TablePath(f2, "parquet"),
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


@pytest.mark.slow
def test_table_path_categorical_unused_table_jit(datapath):
    # TODO: Add memory leak check
    """
    Tests loading a paritioned parquet table in JIT.
    """

    def impl(f1, f2):
        bc = bodosql.BodoSQLContext(
            {
                "parquet_table": bodosql.TablePath(f1, "parquet"),
                "unused_table": bodosql.TablePath(f2, "parquet"),
            }
        )
        return bc.sql("select * from parquet_table")

    f1 = "bodosql/tests/data/sample-parquet-data/partitioned"
    f2 = datapath("tpcxbb-test-data") + "/web_sales"

    py_output = pd.read_parquet(f1)
    py_output["part"] = py_output["part"].astype(str)
    check_func(impl, (f1, f2), py_output=py_output)
    bodo_func = bodo.jit(impl, pipeline_class=TypeInferenceTestPipeline)
    bodo_func(f1, f2)
    check_num_parquet_readers(bodo_func, 1)


@pytest.mark.slow
def test_table_path_categorical_unused_table_python(datapath):
    # TODO: Add memory leak check
    """
    Tests loading a paritioned parquet table in Python.
    """

    def impl(f1, f2):
        bc = bodosql.BodoSQLContext(
            {
                "parquet_table": bodosql.TablePath(f1, "parquet"),
                "unused_table": bodosql.TablePath(f2, "parquet"),
            }
        )
        return bc.sql("select * from parquet_table")

    f1 = "bodosql/tests/data/sample-parquet-data/partitioned"
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


# TODO: Add a test with muliple tables to check reorder_io works as expected.
