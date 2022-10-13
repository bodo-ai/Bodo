# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests that TablePath can work when supplied iceberg tables.

Mostly reuses tests/fixtures from the engine, but using bodosql's TablePath
"""
import io

import bodosql
import pandas as pd
import pytest

import bodo
from bodo.tests.conftest import (  # pragma: no cover
    iceberg_database,
    iceberg_table_conn,
)
from bodo.tests.iceberg_database_helpers import spark_reader
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func, sync_dtypes


@pytest.mark.parametrize(
    "table_name",
    [
        # TODO: BE-2831 Reading maps from parquet not supported yet
        pytest.param(
            "simple_map_table",
            marks=pytest.mark.skip(reason="Need to support reading maps from parquet."),
        ),
        pytest.param(
            "simple_numeric_table",
            marks=pytest.mark.skip("bodosql does not support decimal types"),
        ),
        "simple_string_table",
        "partitions_dt_table",
        # TODO: The results of Bodo and Spark implemenation are different from original
        # but only in check_func
        pytest.param("simple_dt_tsz_table", marks=pytest.mark.slow),
        pytest.param(
            "simple_list_table",
            marks=pytest.mark.skip("bodosql does not support list array types"),
        ),
        pytest.param("simple_bool_binary_table"),
        pytest.param(
            "simple_struct_table",
            marks=pytest.mark.skip("bodosql does not support struct types"),
        ),
    ],
)
def test_simple_table_read(
    memory_leak_check, iceberg_database, iceberg_table_conn, table_name
):
    """
    Test simple read operation on test tables
    """
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        bc = bodosql.BodoSQLContext(
            {
                "iceberg_tbl": bodosql.TablePath(
                    table_name, "sql", conn_str=conn, db_schema=db_schema
                )
            }
        )
        df = bc.sql("SELECT * FROM iceberg_tbl")
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)

    if table_name == "simple_bool_binary_table":
        # Bodo outputs binary data as bytes while Spark does bytearray (which Bodo doesn't support),
        # so we convert Spark output.
        # This has been copied from BodoSQL. See `convert_spark_bytearray`
        # in `BodoSQL/bodosql/tests/utils.py`.
        py_out[["C"]] = py_out[["C"]].apply(
            lambda x: [bytes(y) if isinstance(y, bytearray) else y for y in x],
            axis=1,
            result_type="expand",
        )
    elif table_name == "simple_struct_table":
        # Needs special handling since PySpark returns nested structs as tuples.
        # Convert columns with nested structs from tuples to dictionaries with correct keys
        py_out["A"] = py_out["A"].map(lambda x: {"a": x[0], "b": x[1]})
        py_out["B"] = py_out["B"].map(lambda x: {"a": x[0], "b": x[1], "c": x[2]})

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=(
            table_name != "simple_list_table"
        ),  # No sorting in this case because lists are not hashable
        reset_index=True,
    )


def test_column_pruning(memory_leak_check, iceberg_database, iceberg_table_conn):
    """
    Test simple read operation on test table simple_string_table
    with column pruning.
    """

    table_name = "simple_string_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema, bodo_read_as_dict):
        bc = bodosql.BodoSQLContext(
            {
                "iceberg_tbl": bodosql.TablePath(
                    table_name,
                    "sql",
                    conn_str=conn,
                    db_schema=db_schema,
                    bodo_read_as_dict=bodo_read_as_dict,
                )
            }
        )
        df = bc.sql("SELECT A, C FROM iceberg_tbl")
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    py_out = py_out[["A", "C"]]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    res = None
    bodo_read_as_dict = ["A"]
    with set_logging_stream(logger, 1):
        res = bodo.jit()(impl)(table_name, conn, db_schema, bodo_read_as_dict)
        check_logger_msg(stream, "Columns loaded ['A', 'C']")
        check_logger_msg(
            stream, f"Columns {bodo_read_as_dict} using dictionary encoding"
        )

    py_out = sync_dtypes(py_out, res.dtypes.values.tolist())
    check_func(impl, (table_name, conn, db_schema, bodo_read_as_dict), py_output=py_out)


def test_zero_columns_pruning(memory_leak_check, iceberg_database, iceberg_table_conn):
    """
    Test loading just a length from iceberg tables.
    """

    table_name = "simple_string_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        bc = bodosql.BodoSQLContext(
            {
                "iceberg_tbl": bodosql.TablePath(
                    table_name, "sql", conn_str=conn, db_schema=db_schema
                )
            }
        )
        df = bc.sql("SELECT COUNT(*) as cnt FROM iceberg_tbl")
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    py_out = pd.DataFrame({"cnt": len(py_out)}, index=pd.RangeIndex(0, 1, 1))

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, "Columns loaded []")

    check_func(
        impl, (table_name, conn, db_schema), py_output=py_out, is_out_distributed=False
    )


def test_tablepath_dict_encoding(
    memory_leak_check, iceberg_database, iceberg_table_conn
):
    """
    Test simple read operation on test table simple_string_table
    with column pruning and multiple columns being dictionary encoded.
    """

    table_name = "simple_string_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema, bodo_read_as_dict):
        bc = bodosql.BodoSQLContext(
            {
                "iceberg_tbl": bodosql.TablePath(
                    table_name,
                    "sql",
                    conn_str=conn,
                    db_schema=db_schema,
                    bodo_read_as_dict=bodo_read_as_dict,
                )
            }
        )
        df = bc.sql("SELECT A, C FROM iceberg_tbl")
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    py_out = py_out[["A", "C"]]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    res = None
    bodo_read_as_dict = ["A", "C"]
    with set_logging_stream(logger, 1):
        res = bodo.jit()(impl)(table_name, conn, db_schema, bodo_read_as_dict)
        check_logger_msg(
            stream, f"Columns {bodo_read_as_dict} using dictionary encoding"
        )

    py_out = sync_dtypes(py_out, res.dtypes.values.tolist())
    check_func(impl, (table_name, conn, db_schema, bodo_read_as_dict), py_output=py_out)
