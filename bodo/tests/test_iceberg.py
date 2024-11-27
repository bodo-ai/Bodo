import glob
import io
import os
import re
import struct
import traceback
from datetime import date, datetime
from pathlib import Path

import bodo_iceberg_connector
import mmh3
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyspark.sql.types as spark_types
import pytest
import pytz
from numba.core import types

import bodo
from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next
from bodo.io.iceberg import ICEBERG_FIELD_ID_MD_KEY
from bodo.mpi4py import MPI
from bodo.tests.iceberg_database_helpers import schema_evolution_eg_table, spark_reader
from bodo.tests.iceberg_database_helpers.part_sort_table import (
    BASE_NAME as PART_SORT_TABLE_BASE_NAME,
)
from bodo.tests.iceberg_database_helpers.part_sort_table import (
    PARTITION_SPEC as PART_SORT_TABLE_PARTITION_SPEC,
)
from bodo.tests.iceberg_database_helpers.part_sort_table import (
    SORT_ORDER as PART_SORT_TABLE_SORT_ORDER,
)
from bodo.tests.iceberg_database_helpers.partition_tables import (
    PARTITION_MAP,
    part_table_name,
)
from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLES_MAP,
)
from bodo.tests.iceberg_database_helpers.sort_tables import (
    SORT_MAP,
    sort_table_name,
)
from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    PartitionField,
    SortField,
    create_iceberg_table,
    get_spark,
)
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    ColumnDelTestPipeline,
    DistTestPipeline,
    _gather_output,
    _get_dist_arg,
    _test_equal,
    _test_equal_guard,
    cast_dt64_to_ns,
    check_func,
    convert_non_pandas_columns,
    gen_nonascii_list,
    reduce_sum,
    run_rank0,
    sync_dtypes,
)
from bodo.utils.testing import ensure_clean2
from bodo.utils.typing import BodoError

pytestmark = pytest.mark.iceberg

WRITE_TABLES = [
    "BOOL_BINARY_TABLE",
    "DT_TSZ_TABLE",
    "TZ_AWARE_TABLE",
    "DTYPE_LIST_TABLE",
    "NUMERIC_TABLE",
    "STRING_TABLE",
    "LIST_TABLE",
    "STRUCT_TABLE",
    "OPTIONAL_TABLE",
    # TODO Needs investigation.
    pytest.param(
        "MAP_TABLE",
        marks=pytest.mark.skip(
            reason="Results in runtime error that's consistent with to_parquet."
        ),
    ),
    pytest.param(
        "DECIMALS_TABLE",
        marks=pytest.mark.skip(
            reason="We don't support custom precisions and scale at the moment."
        ),
    ),
    pytest.param(
        "DECIMALS_LIST_TABLE",
        marks=pytest.mark.skip(
            reason="We don't support custom precisions and scale at the moment."
        ),
    ),
    "DICT_ENCODED_STRING_TABLE",
]


@pytest.fixture(params=WRITE_TABLES)
def simple_dataframe(request):
    return (
        request.param,
        f"SIMPLE_{request.param}",
        SIMPLE_TABLES_MAP[f"SIMPLE_{request.param}"][0],
    )


def _run_spark(query):
    """Run query with Spark and return the results in a Pandas DataFrame"""
    if bodo.get_rank() == 0:
        spark = get_spark()
        py_out = spark.sql(query)
        py_out = py_out.toPandas()
    else:
        py_out = None

    comm = MPI.COMM_WORLD
    py_out = comm.bcast(py_out, root=0)
    return py_out


@pytest.mark.slow
@pytest.mark.parametrize(
    "table_name",
    [
        # TODO: BE-2831 Reading maps from parquet not supported yet
        pytest.param(
            "SIMPLE_MAP_TABLE",
            marks=pytest.mark.skip(reason="Need to support reading maps from parquet."),
        ),
        "SIMPLE_STRING_TABLE",
        "PARTITIONS_DT_TABLE",
        "SIMPLE_DT_TSZ_TABLE",
        "SIMPLE_DECIMALS_TABLE",
    ],
)
def test_simple_table_read(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    """
    Test simple read operation on test tables
    """

    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "table_name",
    [
        # TODO: BE-2831 Reading maps from parquet not supported yet
        pytest.param(
            "SIMPLE_MAP_TABLE",
            marks=pytest.mark.skip(reason="Need to support reading maps from parquet."),
        ),
        "SIMPLE_STRING_TABLE",
        "PARTITIONS_DT_TABLE",
        "SIMPLE_DT_TSZ_TABLE",
        "SIMPLE_DECIMALS_TABLE",
    ],
)
def test_read_zero_cols(iceberg_database, iceberg_table_conn, table_name):
    """
    Test that computing just a length in Iceberg loads 0 columns.
    """
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return len(df)

    py_out = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=len(py_out),
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(impl)
        bodo_func(table_name, conn, db_schema)

        check_logger_msg(stream, "Columns loaded []")


@pytest.mark.slow
def test_simple_tz_aware_table_read(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    """
    Test simple read operation on SIMPLE_TZ_AWARE_TABLE.
    Needs to be separate since there's a type mismatch between
    original and what's read from Iceberg (written by Spark).
    When Spark reads it and converts it to Pandas, the datatype
    is:
    A    datetime64[ns]
    B    datetime64[ns]
    but when Bodo reads it, it's:
    A    datetime64[ns, UTC]
    B    datetime64[ns, UTC]
    which causes the mismatch.
    """

    table_name = "SIMPLE_TZ_AWARE_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )


@pytest.mark.slow
def test_simple_numeric_table_read(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    """
    Test simple read operation on test table SIMPLE_NUMERIC_TABLE
    with column pruning.
    """

    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)
    res: pd.DataFrame = bodo.jit()(impl)(table_name, conn, db_schema)
    py_out = sync_dtypes(py_out, res.dtypes.values.tolist())
    py_out["E"] = py_out["E"].astype("Int32")
    py_out["F"] = py_out["F"].astype("Int64")
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        convert_to_nullable_float=False,
    )


@pytest.mark.timeout(600)
@pytest.mark.slow
@pytest.mark.parametrize(
    "table_name", ["SIMPLE_LIST_TABLE", "SIMPLE_DECIMALS_LIST_TABLE"]
)
def test_simple_list_table_read(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    memory_leak_check,
):
    """
    Test reading SIMPLE_LIST_TABLE which consists of columns of lists.
    Need to compare Bodo and PySpark results without sorting them.
    """
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        reset_index=True,
        # No sorting because lists are not hashable
    )


@pytest.mark.slow
def test_simple_bool_binary_table_read(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    """
    Test reading SIMPLE_BOOL_BINARY_TABLE which consists of boolean
    and binary types (bytes). Needs special handling to compare
    with PySpark.
    """
    table_name = "SIMPLE_BOOL_BINARY_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)
    # Bodo outputs binary data as bytes while Spark does bytearray (which Bodo doesn't support),
    # so we convert Spark output.
    # This has been copied from BodoSQL. See `convert_spark_bytearray`
    # in `BodoSQL/bodosql/tests/utils.py`.
    py_out[["C"]] = py_out[["C"]].apply(
        lambda x: [bytes(y) if isinstance(y, bytearray) else y for y in x],
        axis=1,
        result_type="expand",
    )
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.slow
def test_simple_struct_table_read(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    """
    Test reading SIMPLE_STRUCT_TABLE which consists of columns of structs.
    Needs special handling since PySpark returns nested structs as tuples.
    """
    table_name = "SIMPLE_STRUCT_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    # Convert columns with nested structs from tuples to dictionaries with correct keys
    py_out = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)
    py_out["A"] = py_out["A"].map(lambda x: {"a": x["a"], "b": x["b"]})
    py_out["B"] = py_out["B"].map(lambda x: {"a": x["a"], "b": x["b"], "c": x["c"]})

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        reset_index=True,
    )


@pytest.mark.slow
def test_column_pruning(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Test simple read operation on test table SIMPLE_NUMERIC_TABLE
    with column pruning.
    """

    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[["A", "D"]]
        return df

    py_out = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)
    py_out = py_out[["A", "D"]]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    res = None
    with set_logging_stream(logger, 1):
        res = bodo.jit()(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, "Columns loaded ['A', 'D']")

    py_out = sync_dtypes(py_out, res.dtypes.values.tolist())
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        convert_to_nullable_float=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize("dict_encode_in_bodo", [True, False])
def test_dict_encoded_string_arrays(
    iceberg_database, iceberg_table_conn, dict_encode_in_bodo
):
    """
    Test reading string arrays as dictionary-encoded when specified by the user or
    determined from properties of table data.

    dict_encode_in_bodo: Whether the dict-encoding should be performed by
        Bodo instead of Arrow. This is used to test the read behavior
        on Snowflake-managed Iceberg tables where we do the dict-encoding
        ourselves after reading the columns as string arrays because of
        Arrow gaps in being able to read files written by Snowflake
        as dict-encoded columns directly.
    """

    table_name = "SIMPLE_DICT_ENCODED_STRING"

    db_schema, warehouse_loc = iceberg_database(table_name)
    spark = get_spark()

    # Write a simple dataset with strings (repetitive/non-repetitive) and non-strings
    df = pd.DataFrame(
        {
            # non-string
            "A": np.arange(2000) + 1.1,
            # should be dictionary encoded
            "B": ["awe", "awv2"] * 1000,
            # should not be dictionary encoded
            "C": [str(i) for i in range(2000)],
            # non-string column
            "D": np.arange(2000) + 3,
            # should be dictionary encoded
            "E": ["r32"] * 2000,
            # non-string column
            "F": np.arange(2000),
        }
    )
    sql_schema = [
        ("A", "double", True),
        ("B", "string", False),
        ("C", "string", True),
        ("D", "long", False),
        ("E", "string", True),
        ("F", "long", True),
    ]
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    # read all columns and determine dict-encoding automatically
    def impl1(table_name, conn, db_schema):
        df = pd.read_sql_table(
            table_name, conn, db_schema, _bodo_dict_encode_in_bodo=dict_encode_in_bodo
        )
        return df

    check_func(impl1, (table_name, conn, db_schema), py_output=df)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl1)(table_name, conn, db_schema)
        check_logger_msg(stream, "Columns ['B', 'E'] using dictionary encoding")

    # test dead column elimination with dict-encoded columns
    def impl2(table_name, conn, db_schema):
        df = pd.read_sql_table(
            table_name, conn, db_schema, _bodo_dict_encode_in_bodo=dict_encode_in_bodo
        )
        return df[["B", "D"]]

    check_func(impl2, (table_name, conn, db_schema), py_output=df[["B", "D"]])

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl2)(table_name, conn, db_schema)
        check_logger_msg(stream, "Columns ['B'] using dictionary encoding")

    # test _bodo_read_as_dict (force non-dict to dict)
    def impl3(table_name, conn, db_schema):
        df = pd.read_sql_table(
            table_name,
            conn,
            db_schema,
            _bodo_read_as_dict=["C", "E"],
            _bodo_dict_encode_in_bodo=dict_encode_in_bodo,
        )  # type: ignore
        return df[["B", "C", "D", "E"]]

    check_func(impl3, (table_name, conn, db_schema), py_output=df[["B", "C", "D", "E"]])

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl3)(table_name, conn, db_schema)
        check_logger_msg(stream, "Columns ['B', 'C', 'E'] using dictionary encoding")

    # error checking _bodo_read_as_dict
    with pytest.raises(BodoError, match=r"must be a constant list of column names"):

        def impl4(table_name, conn, db_schema):
            df = pd.read_sql_table(
                table_name,
                conn,
                db_schema,
                _bodo_read_as_dict=True,
                _bodo_dict_encode_in_bodo=dict_encode_in_bodo,
            )  # type: ignore
            return df

        bodo.jit(impl4)(table_name, conn, db_schema)

    with pytest.raises(BodoError, match=r"_bodo_read_as_dict is not in table columns"):

        def impl5(table_name, conn, db_schema):
            df = pd.read_sql_table(
                table_name,
                conn,
                db_schema,
                _bodo_read_as_dict=["H"],
                _bodo_dict_encode_in_bodo=dict_encode_in_bodo,
            )  # type: ignore
            return df

        bodo.jit(impl5)(table_name, conn, db_schema)

    with pytest.raises(BodoError, match=r"is not a string column"):

        def impl6(table_name, conn, db_schema):
            df = pd.read_sql_table(
                table_name,
                conn,
                db_schema,
                _bodo_read_as_dict=["D"],
                _bodo_dict_encode_in_bodo=dict_encode_in_bodo,
            )  # type: ignore
            return df

        bodo.jit(impl6)(table_name, conn, db_schema)

    # make sure dict-encoding detection works even if there is schema evolution
    # test both column renaming and type changes since checked separately

    # create a new table since CachingCatalog inside Bodo can't see schema changes done
    # by Spark code below
    suffix = "1" if dict_encode_in_bodo else "2"
    table_name = f"SIMPLE_DICT_ENCODED_STRING2_{suffix}"
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    # rename B to B2
    if bodo.get_rank() == 0:
        spark.sql(
            f"ALTER TABLE hadoop_prod.{db_schema}.{table_name} RENAME COLUMN B TO B2"
        )
        spark_schema = spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.DoubleType(), True),
                spark_types.StructField("B", spark_types.StringType(), False),
                spark_types.StructField("C", spark_types.StringType(), True),
                spark_types.StructField("D", spark_types.LongType(), False),
                spark_types.StructField("E", spark_types.StringType(), True),
                spark_types.StructField("F", spark_types.LongType(), True),
            ]
        )
        sdf = spark.createDataFrame(df, schema=spark_schema)
        sdf.withColumnRenamed("B", "B2").writeTo(
            f"hadoop_prod.{db_schema}.{table_name}"
        ).append()

        # change type of C from string to int
        spark.sql(f"ALTER TABLE hadoop_prod.{db_schema}.{table_name} DROP COLUMN C")
        spark.sql(
            f"ALTER TABLE hadoop_prod.{db_schema}.{table_name} ADD COLUMN C bigint AFTER B2"
        )
    bodo.barrier()
    df = df.rename(columns={"B": "B2"})
    df["C"] = 123
    df["F"] = df.F + 10000
    if bodo.get_rank() == 0:
        spark_schema = spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.DoubleType(), True),
                spark_types.StructField("B2", spark_types.StringType(), False),
                spark_types.StructField("C", spark_types.LongType(), True),
                spark_types.StructField("D", spark_types.LongType(), False),
                spark_types.StructField("E", spark_types.StringType(), True),
                spark_types.StructField("F", spark_types.LongType(), True),
            ]
        )
        sdf = spark.createDataFrame(df, schema=spark_schema)
        sdf.writeTo(f"hadoop_prod.{db_schema}.{table_name}").append()
    bodo.barrier()

    def impl7(table_name, conn, db_schema):
        df = pd.read_sql_table(
            table_name, conn, db_schema, _bodo_dict_encode_in_bodo=dict_encode_in_bodo
        )
        df = df[df.F >= 10000]
        return df

    check_func(impl7, (table_name, conn, db_schema), py_output=df)


@pytest.mark.parametrize("dict_encode_in_bodo", [True, False])
def test_dict_encoded_string_arrays_streaming_read(
    iceberg_database,
    iceberg_table_conn,
    dict_encode_in_bodo,
):
    """
    Similar to the previous test, but using the streaming code path.
    The error-checking code paths are common between streaming and non-streaming,
    so we skip them here.
    """
    table_name = "SIMPLE_DICT_ENCODED_STRING_STREAMING"

    db_schema, warehouse_loc = iceberg_database(table_name)

    # Write a simple dataset with strings (repetitive/non-repetitive) and non-strings
    df = pd.DataFrame(
        {
            # non-string
            "A": np.arange(2000) + 1.1,
            # should be dictionary encoded
            "B": ["awe", "awv2"] * 1000,
            # should not be dictionary encoded
            "C": [str(i) for i in range(2000)],
            # non-string column
            "D": np.arange(2000) + 3,
            # should be dictionary encoded
            "E": ["r32"] * 2000,
            # non-string column
            "F": np.arange(2000),
        }
    )
    sql_schema = [
        ("A", "double", True),
        ("B", "string", False),
        ("C", "string", True),
        ("D", "long", False),
        ("E", "string", True),
        ("F", "long", True),
    ]

    @run_rank0
    def setup():
        spark = get_spark()
        create_iceberg_table(df, sql_schema, table_name, spark)

    setup()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B", "C", "D", "E", "F"))

    def impl1(table_name, conn, db_schema):
        is_last_global = False
        reader = pd.read_sql_table(
            table_name,
            conn,
            db_schema,
            _bodo_dict_encode_in_bodo=dict_encode_in_bodo,
            # TODO Investigate the failure when we set this to 64 (including on develop)
            _bodo_chunksize=4000,
        )  # type: ignore
        __bodo_streaming_batches_table_builder_1 = (
            bodo.libs.table_builder.init_table_builder_state(-1)
        )
        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            bodo.libs.table_builder.table_builder_append(
                __bodo_streaming_batches_table_builder_1, table
            )
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )

        arrow_reader_del(reader)
        out_table = bodo.libs.table_builder.table_builder_finalize(
            __bodo_streaming_batches_table_builder_1
        )
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl1,
            (table_name, conn, db_schema),
            py_output=df,
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(stream, "Columns ['B', 'E'] using dictionary encoding")

    # test _bodo_read_as_dict (force non-dict to dict)
    col_meta = bodo.utils.typing.ColNamesMetaType(("B", "C", "D", "E"))

    def impl2(table_name, conn, db_schema):
        is_last_global = False
        reader = pd.read_sql_table(
            table_name,
            conn,
            db_schema,
            _bodo_dict_encode_in_bodo=dict_encode_in_bodo,
            _bodo_chunksize=4000,
            _bodo_read_as_dict=["C", "E"],
            _bodo_columns=["B", "C", "D", "E"],
        )  # type: ignore
        __bodo_streaming_batches_table_builder_1 = (
            bodo.libs.table_builder.init_table_builder_state(-1)
        )
        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            bodo.libs.table_builder.table_builder_append(
                __bodo_streaming_batches_table_builder_1, table
            )
            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )

        arrow_reader_del(reader)
        out_table = bodo.libs.table_builder.table_builder_finalize(
            __bodo_streaming_batches_table_builder_1
        )
        index_var = bodo.hiframes.pd_index_ext.init_range_index(
            0, len(out_table), 1, None
        )
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (out_table,), index_var, col_meta
        )
        return df

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl2,
            (table_name, conn, db_schema),
            py_output=df[["B", "C", "D", "E"]],
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(stream, "Columns ['B', 'C', 'E'] using dictionary encoding")


@pytest.mark.slow
def test_dict_encoding_sync_determination(iceberg_database, iceberg_table_conn):
    """
    Test that columns with dictionary encoding are determined
    in a deterministic fashion across all ranks. This test is
    only useful when run on multiple ranks.
    We saw that there can be bugs when e.g. the list of string
    columns passed to determine_str_as_dict_columns is not
    ordered the same on all ranks.
    For more context, see https://bodo.atlassian.net/browse/BE-3679
    This was fixed in https://github.com/bodo-ai/Bodo/pull/4356.
    The probability of invoking the failure is high when the
    number of columns is higher, which is why we are creating
    a table with 100 string columns: 50 which should be dictionary
    encoded, and 50 which shouldn't.
    This is not guaranteed to work, but provides at least some
    protection against regressions.
    """

    table_name = "TEST_DICT_ENCODING_SYNC_DETERMINATION"

    db_schema, warehouse_loc = iceberg_database()
    spark = get_spark()

    # For convenience name them the columns differently so
    # we can check just the name during validation.
    dict_enc_columns = [f"A{i}" for i in range(1, 51)]
    reg_str_columns = [f"B{i}" for i in range(1, 51)]

    cols = {c: ["awe", "awv2"] * 1000 for c in dict_enc_columns}
    cols.update({c: [str(i) for i in range(2000)] for c in reg_str_columns})

    df = pd.DataFrame(cols)
    sql_schema = [(c, "string", False) for c in dict_enc_columns] + [
        (c, "string", True) for c in reg_str_columns
    ]
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    # ColumnDelTestPipeline preserves the typemap which is what we need.
    @bodo.jit(pipeline_class=ColumnDelTestPipeline)
    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    impl(table_name, conn, db_schema)
    typemap = impl.overloads[impl.signatures[0]].metadata["preserved_typemap"]

    # Validate that all columns are typed correctly.
    passed = 1
    try:
        for col_name, col_type in zip(typemap["df"].columns, typemap["df"].data):
            if col_name.startswith("A"):
                assert isinstance(col_type, bodo.libs.dict_arr_ext.DictionaryArrayType)
            elif col_name.startswith("B"):
                assert isinstance(col_type, bodo.libs.str_arr_ext.StringArrayType)
            else:
                raise ValueError(
                    f"Expected a column starting with A or B, got {col_name} instead."
                )
    except Exception:
        passed = 0
    passed = reduce_sum(passed)
    assert passed == bodo.get_size(), "Datatype validation failed on one or more ranks."


@pytest.mark.slow
def test_disable_dict_detection(iceberg_database, iceberg_table_conn):
    """
    Test reading string arrays as dictionary-encoded when specified by the user or
    determined from properties of table data.
    """

    table_name = "DONT_DICT_ENCODE_TABLE"

    db_schema, warehouse_loc = iceberg_database(table_name)
    spark = get_spark()

    # Write a simple dataset with strings (repetitive/non-repetitive) and non-strings
    df = pd.DataFrame(
        {
            # non-string
            "A": np.arange(2000) + 1.1,
            # should be dictionary encoded
            "B": ["awe", "awv2"] * 1000,
            # should not be dictionary encoded
            "C": [str(i) for i in range(2000)],
            # non-string column
            "D": np.arange(2000) + 3,
            # should be dictionary encoded
            "E": ["r32"] * 2000,
            # non-string column
            "F": np.arange(2000),
        }
    )
    sql_schema = [
        ("A", "double", True),
        ("B", "string", False),
        ("C", "string", True),
        ("D", "long", False),
        ("E", "string", True),
        ("F", "long", True),
    ]
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    # Ensure that dict encoding is not reported during compilation for any columns other than B.
    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(
            table_name,
            conn,
            db_schema,
            _bodo_read_as_dict=["B"],
            _bodo_detect_dict_cols=False,
        )  # type: ignore
        return df

    table_name_type = numba.types.literal(table_name)
    db_schema_type = numba.types.literal(db_schema)
    conn_type = numba.types.literal(conn)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit((table_name_type, conn_type, db_schema_type))(impl)
        check_logger_msg(stream, "Columns ['B'] using dictionary encoding")


@pytest.mark.slow
def test_no_files_after_filter_pushdown(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test the use case where Iceberg filters out all files
    based on the provided filters. We need to load an empty
    DataFrame with the right schema in this case.
    """

    table_name = "FILTER_PUSHDOWN_TEST_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df["TY"].isna()]
        return df

    py_out = _run_spark(
        f"""
        select * from hadoop_prod.{db_schema}.{table_name}
        where TY IS NULL;
        """
    )
    assert (
        py_out.shape[0] == 0
    ), f"Expected DataFrame to be empty, found {py_out.shape[0]} rows instead."

    check_func(impl, (table_name, conn, db_schema), py_output=py_out)


@pytest.mark.slow
def test_snapshot_id(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Test that the bodo_iceberg_connector correctly loads the latest snapshot id.
    """
    comm = MPI.COMM_WORLD
    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    # Format the connection string since we don't go through pd.read_sql_table
    conn = bodo.io.iceberg.format_iceberg_conn(conn)
    spark = get_spark()
    snapshot_id = None
    spark_snapshot_id = None
    if bodo.get_rank() == 0:
        snapshot_id = bodo_iceberg_connector.bodo_connector_get_current_snapshot_id(
            conn, db_schema, table_name
        )
        py_out = spark.sql(
            f"""
        select snapshot_id from hadoop_prod.{db_schema}.{table_name}.history order by made_current_at DESC
        """
        )
        py_out = py_out.toPandas()
        spark_snapshot_id = py_out.iloc[0, 0]
    snapshot_id, spark_snapshot_id = comm.bcast((snapshot_id, spark_snapshot_id))
    assert (
        snapshot_id == spark_snapshot_id
    ), "Bodo loaded snapshot id doesn't match spark"


@pytest.mark.slow
def test_read_merge_into_cow_row_id_col(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that reading from an Iceberg table in MERGE INTO COW mode
    returns a DataFrame with an additional row id column
    """

    comm = MPI.COMM_WORLD
    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    # Get Relevant Info from Spark
    spark_out = None
    out_len = -1
    err = None
    if bodo.get_rank() == 0:
        try:
            spark_out, out_len, _ = spark_reader.read_iceberg_table(
                table_name, db_schema
            )
        except Exception as e:
            err = e
    spark_out, out_len, err = comm.bcast((spark_out, out_len, err))
    if isinstance(err, Exception):
        raise err

    # _BODO_ROW_ID is always loaded in MERGE INTO COW Mode, see iceberg_ext.py
    # Since Iceberg output is unordered, not guarantee that the row id values
    # are assigned to the same row. Thus, need to check them separately
    check_func(
        lambda name, conn, db: pd.read_sql_table(name, conn, db, _bodo_merge_into=True)[
            0
        ]["_BODO_ROW_ID"],  # type: ignore
        (table_name, conn, db_schema),
        py_output=np.arange(out_len),
    )

    check_func(
        lambda name, conn, db: pd.read_sql_table(name, conn, db, _bodo_merge_into=True)[
            0
        ][["B", "E", "A"]],  # type: ignore
        (table_name, conn, db_schema),
        py_output=spark_out[["B", "E", "A"]],
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )


@pytest.mark.slow
def test_filter_pushdown_partitions(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that simple date based partitions can be read as expected.
    """

    table_name = "PARTITIONS_DT_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df["A"] <= date(2018, 12, 12)]
        return df

    py_out = _run_spark(
        f"""
        select * from hadoop_prod.{db_schema}.{table_name}
        where A <= '2018-12-12';
        """
    )

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.slow
def test_filter_pushdown_file_filters(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that simple filter pushdown works inside the parquet file.
    """

    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df.B == 2]
        return df

    py_out = _run_spark(
        f"""
    select * from hadoop_prod.{db_schema}.{table_name}
    WHERE B = 2
    """
    )

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, "Columns loaded ['A', 'B', 'C', 'D', 'E', 'F']")
        check_logger_msg(stream, "Filter pushdown successfully performed")


@pytest.mark.slow
def test_filter_pushdown_merge_into(iceberg_database, iceberg_table_conn):
    """
    Test that passing _bodo_merge_into still has filter pushdown succeed
    but doesn't filter files.
    """

    comm = MPI.COMM_WORLD
    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def f(df, file_list):
        # Sort the filenames so the order is consistent with Spark
        return sorted(file_list)

    def impl1(table_name, conn, db_schema):
        # Just return df because sort_output, reset_index don't work when
        # returning tuples.
        df, _, _ = pd.read_sql_table(table_name, conn, db_schema, _bodo_merge_into=True)  # type: ignore
        df = df[df.B == 2]
        return df.drop(columns=["_BODO_ROW_ID"])

    file_list_type = types.List(types.unicode_type)

    def impl2(table_name, conn, db_schema):
        df, file_list, snapshot_id = pd.read_sql_table(
            table_name, conn, db_schema, _bodo_merge_into=True
        )  # type: ignore
        df = df[df.B == 2]
        # Force use of df since we won't return it and still need
        # to load data.
        with bodo.objmode(sort_list=file_list_type):
            sort_list = f(df, file_list)
        return (sort_list, snapshot_id)

    table_output = None
    err = None
    if bodo.get_rank() == 0:
        try:
            spark = get_spark()
            # Load the table output
            table_output = spark.sql(
                f"""SELECT * FROM hadoop_prod.{db_schema}.{table_name}
                WHERE B = 2
                """
            ).toPandas()
        except Exception as e:
            err = e
    table_output, err = comm.bcast((table_output, err))
    if isinstance(err, Exception):
        raise err

    check_func(
        impl1,
        (table_name, conn, db_schema),
        py_output=table_output,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )

    # Check filter pushdown
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl1)(table_name, conn, db_schema)
        check_logger_msg(
            stream, "Columns loaded ['A', 'B', 'C', 'D', 'E', 'F', '_BODO_ROW_ID']"
        )
        check_logger_msg(stream, "Filter pushdown successfully performed")
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nbic.FilterExpr('==', [bic.ColumnRef('B'), bic.Scalar(f0)])",
        )
        check_logger_no_msg(stream, "Arrow filters pushed down:\nNone")

    files_set = None
    spark_snapshot_id = None
    if bodo.get_rank() == 0:
        try:
            # Check the files list + snapshot id
            # Load the file list
            files_frame = spark.sql(
                f"""
                select file_path from hadoop_prod.{db_schema}.{table_name}.files
                """
            )
            files_frame = files_frame.toPandas()
            # Convert to a set because Bodo will only return unique file names
            files_set = set(files_frame["file_path"])
            # We use a sorted list for easier comparison
            files_set = sorted(files_set)
            # Load the snapshot id
            snapshot_frame = spark.sql(
                f"""
                select snapshot_id from hadoop_prod.{db_schema}.{table_name}.history where parent_id is NULL
                """
            )
            snapshot_frame = snapshot_frame.toPandas()
            spark_snapshot_id = snapshot_frame.iloc[0, 0]
        except Exception as e:
            err = e

    files_set, spark_snapshot_id, err = comm.bcast((files_set, spark_snapshot_id, err))
    if isinstance(err, Exception):
        raise err

    check_func(
        impl2,
        (table_name, conn, db_schema),
        py_output=(files_set, spark_snapshot_id),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )


def _check_for_sql_read_head_only(bodo_func, head_size):
    """Make sure head-only SQL read optimization is recognized"""
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert hasattr(fir, "meta_head_only_info")
    assert fir.meta_head_only_info[0] == head_size


@pytest.mark.slow
def test_limit_pushdown(iceberg_database, iceberg_table_conn, memory_leak_check):
    """Test that Limit Pushdown is successfully enabled"""
    table_name = "SIMPLE_STRING_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl():
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df.head(5)  # type: ignore

    py_out = _run_spark(f"select * from hadoop_prod.{db_schema}.{table_name} LIMIT 5;")

    check_func(
        impl,
        (),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )

    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
    bodo_func()
    _check_for_sql_read_head_only(bodo_func, 5)


@pytest.mark.slow
@pytest.mark.skip("[BE-3212] Fix Java failures on CI")
def test_iceberg_invalid_table(iceberg_database, iceberg_table_conn):
    """Tests error raised when a nonexistent Iceberg table is provided."""

    table_name = "NO_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df["A"].sum()

    with pytest.raises(BodoError, match="No such Iceberg table found"):
        bodo.jit(impl)(table_name, conn, db_schema)


@pytest.mark.slow
def test_iceberg_invalid_path(iceberg_database, iceberg_table_conn):
    """Tests error raised when invalid path is provided."""

    table_name = "FILTER_PUSHDOWN_TEST_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    db_schema += "not"

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df["A"].sum()

    with pytest.raises(BodoError, match="No such Iceberg table found"):
        bodo.jit(impl)(table_name, conn, db_schema)


@pytest.mark.slow
def test_write_existing_fail(
    iceberg_database,
    iceberg_table_conn,
    simple_dataframe,
):
    """Test that writing to an existing table when if_exists='fail' errors"""
    base_name, table_name, df = simple_dataframe
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="fail")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "DICT_ENCODED_STRING_TABLE":
        bodo.hiframes.boxing._use_dict_str_type = True

    try:
        # TODO Uncomment after adding replicated write support.
        # err = None
        # if bodo.get_rank() == 0:
        #     try:
        #         with pytest.raises(BodoError, match="already exists"):
        #             bodo.jit(replicated=["df"])(impl)(df, table_name, conn, db_schema)
        #     except Exception as e:
        #         err = e
        # err = comm.bcast(err)
        # if isinstance(err, Exception):
        #     raise err

        with pytest.raises(ValueError, match="already exists"):
            bodo.jit(distributed=["df"])(impl)(
                _get_dist_arg(df), table_name, conn, db_schema
            )
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type


@pytest.mark.timeout(1000)
@pytest.mark.slow
@pytest.mark.parametrize("read_behavior", ["spark", "bodo"])
def test_basic_write_replace(
    iceberg_database,
    iceberg_table_conn,
    simple_dataframe,
    read_behavior,
    memory_leak_check,
):
    """Test basic Iceberg table replace on Spark table"""

    comm = MPI.COMM_WORLD
    base_name, table_name, df = simple_dataframe
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="replace")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "DICT_ENCODED_STRING_TABLE":
        bodo.hiframes.boxing._use_dict_str_type = True

    try:
        # Write using Bodo
        bodo.jit(distributed=["df"])(impl)(
            _get_dist_arg(df), table_name, conn, db_schema
        )
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
    # Read using PySpark or Bodo, and then check that it's what's expected

    if table_name == "SIMPLE_STRUCT_TABLE" and read_behavior == "spark":
        # There's an issue where Spark is unable to read structs that we
        # write through Iceberg. It's able to read the parquet file
        # when using `spark.read.format("parquet").load(fname)`
        # and the Iceberg metadata that we write looks correct,
        # so it seems like a Spark issue, but needs further investigation.
        # We're able to read the table using Bodo though.
        # TODO Open issue
        return

    if read_behavior == "spark":
        py_out = spark_reader.read_iceberg_table_single_rank(table_name, db_schema)
        # Spark reads the column in micro-seconds instead of nano-seconds like
        # we do, so we need to convert it to match Bodo.
        if base_name == "DT_TSZ_TABLE":
            py_out["B"] = py_out["B"].astype("datetime64[ns]")
    else:
        assert (
            read_behavior == "bodo"
        ), "Read Behavior can only be either `spark` or `bodo`"
        py_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
        py_out = _gather_output(py_out)

    # Uncomment if we get Spark to be able to read this table (see comment above)
    # if table_name == "SIMPLE_STRUCT_TABLE":
    #     py_out["A"] = py_out["A"].map(lambda x: {"a": x["a"], "b": x["b"]})
    #     py_out["B"] = py_out["B"].map(lambda x: {"a": x["a"], "b": x["b"], "c": x["c"]})

    comm = MPI.COMM_WORLD
    passed = None
    if comm.Get_rank() == 0:
        passed = _test_equal_guard(df, py_out, sort_output=False, check_dtype=False)
        table_cmt = (
            get_spark()
            .sql(f"DESCRIBE TABLE EXTENDED hadoop_prod.{db_schema}.{table_name}")
            .filter("col_name = 'Comment'")
            .select("data_type")
            .head()
        )
        assert (
            table_cmt is None
        ), "Expected table comment to be None, but actual comment is not None"
    passed = comm.bcast(passed)
    assert passed == 1

    # TODO Uncomment after adding replicated write support.
    # Test replicated -- only run on rank0, and synchronize errors to avoid hangs
    # if behavior == "create":
    #     table_name = f"{table_name}_repl"
    #
    # err = None
    # # Start with 1, it'll become 0 on rank 0 if it fails
    # passed = 1
    # if bodo.get_rank() == 0:
    #     try:
    #         bodo.jit(replicated=["df"])(impl)(df, table_name, conn, db_schema)
    #         # Read using PySpark, and then check that it's what's expected
    #         passed = _test_equal_guard(
    #             orig_df,
    #             py_out,
    #             sort_output=False,
    #             check_names=True,
    #             check_dtype=False,
    #         )
    #     except Exception as e:
    #         err = e
    # err = comm.bcast(err)
    # if isinstance(err, Exception):
    #     raise err
    # n_passed = reduce_sum(passed)
    # assert n_passed == n_pes)


@pytest.mark.timeout(1000)
@pytest.mark.slow
@pytest.mark.parametrize("behavior", ["create", "append"])
@pytest.mark.parametrize("initial_write", ["bodo", "spark"])
def test_basic_write_new_append(
    iceberg_database,
    iceberg_table_conn,
    simple_dataframe,
    behavior,
    initial_write,
    memory_leak_check,
):
    """
    Test basic Iceberg table write + append on new table
    (append to table written by Bodo)
    """

    comm = MPI.COMM_WORLD
    n_pes = comm.Get_size()
    base_name, table_name, df = simple_dataframe

    if (
        table_name == "SIMPLE_LIST_TABLE"
        and initial_write == "spark"
        and behavior == "append"
    ):
        pytest.skip(
            reason="During unboxing of Series with lists, we always assume int64 (vs int32) and float64 (vs float32), which doesn't match original schema written by Spark."
        )

    # We want to use completely new table for each test
    db_schema, warehouse_loc = iceberg_database(table_name)
    table_name += f"_{behavior}_{initial_write}"
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def create_impl(df, table_name, conn, db_schema):
        df.to_sql(
            table_name,
            conn,
            db_schema,
            if_exists="append",
        )

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "DICT_ENCODED_STRING_TABLE":
        bodo.hiframes.boxing._use_dict_str_type = True

    try:
        impl = bodo.jit(distributed=["df"])(create_impl)

        if initial_write == "bodo" or behavior == "create":
            # Write using Bodo
            impl(_get_dist_arg(df), table_name, conn, db_schema)
        elif initial_write == "spark" and behavior == "append":
            spark = get_spark()
            # Write using Spark on rank 0
            if bodo.get_rank() == 0:
                _, sql_schema = SIMPLE_TABLES_MAP[f"SIMPLE_{base_name}"]
                create_iceberg_table(df, sql_schema, table_name, spark)
            bodo.barrier()
        elif initial_write == "spark" and behavior == "create":
            # Nothing to test here.
            return
        else:
            raise ValueError(
                f"Got unsupported values: initial_write: {initial_write} and behavior: {behavior}."
            )

        if behavior == "append":
            # Append using Bodo
            impl(_get_dist_arg(df), table_name, conn, db_schema)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type

    expected_df = (
        pd.concat([df, df]).reset_index(drop=True) if behavior == "append" else df
    )
    not_hashable = False
    for i in range(len(expected_df.columns)):
        # Technically bytes are hashable but Spark uses bytearray and that's not.
        if isinstance(expected_df.iloc[0, i], (list, dict, set, bytearray, bytes)):
            not_hashable = True
            break
    if not_hashable:
        expected_df = convert_non_pandas_columns(expected_df)

    # Read using Bodo and PySpark, and then check that it's what's expected
    bodo_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
    bodo_out = _gather_output(bodo_out)
    passed = None
    comm = MPI.COMM_WORLD
    if bodo.get_rank() == 0:
        if base_name == "DT_TSZ_TABLE":
            expected_df["B"] = expected_df["B"].fillna(
                pd.Timestamp(1970, 1, 1, tz="UTC")
            )
            bodo_out["B"] = bodo_out["B"].fillna(
                pd.Timestamp(year=1970, month=1, day=1, tz="UTC")
            )

        if not_hashable:
            bodo_out = convert_non_pandas_columns(bodo_out)

        table_cmt = (
            get_spark()
            .sql(f"DESCRIBE TABLE EXTENDED hadoop_prod.{db_schema}.{table_name}")
            .filter("col_name = 'Comment'")
            .select("data_type")
            .head()
        )
        assert (
            table_cmt is None
        ), "Expected table comment to be None, but actual comment is not None"

        passed = _test_equal_guard(
            bodo_out,
            expected_df,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1

    if table_name.startswith("SIMPLE_STRUCT_TABLE"):
        # There's an issue where Spark is unable to read structs that we
        # write through Iceberg. It's able to read the parquet file
        # when using `spark.read.format("parquet").load(fname)`
        # and the Iceberg metadata that we write looks correct,
        # so it seems like a Spark issue, but needs further investigation.
        # We're able to read the table using Bodo though.
        # TODO Open issue
        return

    if initial_write == "spark" and behavior == "append" and bodo.get_rank() == 0:
        # We need to invalidate spark cache, because it doesn't realize
        # that the table has been modified.
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

    spark_passed = 1
    if bodo.get_rank() == 0:
        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)

        # Uncomment if we get Spark to be able to read this table (see comment above)
        # if table_name == "SIMPLE_STRUCT_TABLE":
        #     spark_out["A"] = spark_out["A"].map(lambda x: {"a": x["a"], "b": x["b"]})
        #     spark_out["B"] = spark_out["B"].map(
        #         lambda x: {"a": x["a"], "b": x["b"], "c": x["c"]}
        #     )

        # Spark doesn't handle null timestamps properly. It converts them to
        # 0 (i.e. epoch) instead of NaTs like Pandas does. This modifies both
        # dataframes to match Spark.
        if base_name == "DT_TSZ_TABLE":
            expected_df["B"] = expected_df["B"].fillna(
                pd.Timestamp(1970, 1, 1, tz="UTC")
            )
            spark_out["B"] = (
                spark_out["B"].astype("datetime64[ns]").dt.tz_localize("UTC")
            )
            spark_out["B"] = spark_out["B"].fillna(pd.Timestamp(1970, 1, 1, tz="UTC"))

        if not_hashable:
            spark_out = convert_non_pandas_columns(spark_out)

        spark_passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    spark_n = reduce_sum(spark_passed)
    assert spark_n == n_pes


def _verify_pq_schema_in_files(
    warehouse_loc: str,
    db_schema: str,
    table_name: str,
    data_files_before_write: set[str],
    expected_schema: pa.Schema,
):
    """
    Helper function to validate that the parquet
    files written by Bodo have the expected schema,
    including the metadata fields.
    """
    passed = 1
    err = "Parquet field schema metadata validation failed. See error on rank 0"
    if bodo.get_rank() == 0:
        try:
            # Get List of files (only the ones written after Bodo write)
            data_files = glob.glob(
                os.path.join(warehouse_loc, db_schema, table_name, "data", "*.parquet")
            )
            data_files = list(set(data_files) - data_files_before_write)
            assert all(os.path.isfile(file) for file in data_files)
            # Verify that all the parquet files have the correct metadata:
            for data_file in data_files:
                pq_file = pq.ParquetFile(data_file)
                file_schema = pq_file.schema.to_arrow_schema()
                assert expected_schema.equals(
                    file_schema, check_metadata=True
                ), data_file
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err


def _setup_test_iceberg_field_ids_in_pq_schema(
    warehouse_loc: str,
    db_schema: str,
    table_name: str,
    mode: str,
    pa_schema: pa.Schema,
    df: pd.DataFrame,
    sql_schema: list[tuple[str, str, bool]],
) -> tuple[set[str], pa.Schema]:
    """
    Helper function for test_iceberg_field_ids_in_pq_schema. This is
    used for testing both the streaming and non-streaming versions.
    """
    passed = 1
    err = "Setup step failed. See error on rank 0"
    if bodo.get_rank() == 0:
        try:
            if mode == "replace":
                # Create a dummy table with a completely different schema using Spark.
                # This will verify that Bodo doesn't use the existing schema when
                # writing new files.
                df_, sql_schema_ = SIMPLE_TABLES_MAP["SIMPLE_BOOL_BINARY_TABLE"]
                create_iceberg_table(df_, sql_schema_, table_name)
            elif mode == "append":
                # Write a set of files with Spark first.
                create_iceberg_table(df, sql_schema, table_name)
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    data_files_before_write = set()
    if bodo.get_rank() == 0 and mode in ("append", "replace"):
        data_files_before_write = set(
            glob.glob(
                os.path.join(warehouse_loc, db_schema, table_name, "data", "*.parquet")
            )
        )
    # Construct the expected schema object:
    passed = 1
    err = "Constructing expected schema failed. See error on rank 0"
    expected_schema = None
    if bodo.get_rank() == 0:
        try:
            expected_schema = bodo.io.iceberg.add_iceberg_field_id_md_to_pa_schema(
                pa_schema
            )
            iceberg_schema_str = bodo_iceberg_connector.pyarrow_to_iceberg_schema_str(
                expected_schema
            )
            expected_schema = expected_schema.with_metadata(
                {"iceberg.schema": iceberg_schema_str}
            )
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    return data_files_before_write, expected_schema


ICEBERG_FIELD_IDS_IN_PQ_SCHEMA_TEST_PARAMS: list[str, pa.Schema] = [
    pytest.param(
        "NUMERIC_TABLE",
        pa.schema(
            [
                pa.field("A", pa.int32(), False),
                pa.field("B", pa.int64(), False),
                pa.field("C", pa.float32(), False),
                pa.field("D", pa.float64(), False),
                pa.field("E", pa.int32(), True),
                pa.field("F", pa.int64(), True),
            ]
        ),
        id="simple_numeric",
    ),
    pytest.param(
        "STRUCT_TABLE",
        pa.schema(
            [
                pa.field(
                    "A",
                    pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.int64())]),
                ),
                pa.field(
                    "B",
                    pa.struct(
                        [
                            pa.field("a", pa.float64()),
                            pa.field("b", pa.int64()),
                            pa.field("c", pa.float64()),
                        ]
                    ),
                ),
            ]
        ),
        id="struct",
    ),
    pytest.param(
        "LIST_TABLE",
        pa.schema(
            [
                pa.field("A", pa.list_(pa.int64())),
                pa.field("B", pa.list_(pa.string())),
                pa.field("C", pa.list_(pa.int32())),
                pa.field("D", pa.list_(pa.float32())),
                pa.field("E", pa.list_(pa.float64())),
            ]
        ),
        id="list",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "base_name,pa_schema", ICEBERG_FIELD_IDS_IN_PQ_SCHEMA_TEST_PARAMS
)
@pytest.mark.parametrize("mode", ["create", "replace", "append"])
def test_iceberg_field_ids_in_pq_schema(
    base_name,
    pa_schema,
    mode,
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    """
    Test that the parquet files written by Bodo have the expected
    metadata. In particular, the fields' metadata should contain
    the Iceberg Field ID and the schema metadata should have an
    encoded JSON describing the Iceberg schema.
    """

    table_name = f"SIMPLE_{base_name}_pq_schema_test_{mode}"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    df, sql_schema = SIMPLE_TABLES_MAP[f"SIMPLE_{base_name}"]

    if_exists = "fail" if mode == "create" else mode

    if base_name == "LIST_TABLE" and mode == "append":
        pytest.skip(
            reason="During unboxing of Series with lists, we always assume int64 (vs int32) and float64 (vs float32), which doesn't match original schema written by Spark."
        )

    (
        data_files_before_write,
        expected_schema,
    ) = _setup_test_iceberg_field_ids_in_pq_schema(
        warehouse_loc, db_schema, table_name, mode, pa_schema, df, sql_schema
    )

    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists=if_exists)

    # Write using Bodo
    bodo.jit(distributed=["df"])(impl)(_get_dist_arg(df), table_name, conn, db_schema)
    bodo.barrier()

    _verify_pq_schema_in_files(
        warehouse_loc, db_schema, table_name, data_files_before_write, expected_schema
    )


@pytest.mark.slow
def test_iceberg_field_ids_in_pq_schema_append_to_schema_evolved_table(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that when appending to a table that has gone through
    several complicated schema evolutions, we still write
    the correct metadata in the parquet file. In particular,
    the fields' metadata should contain the Iceberg Field ID
    and the schema metadata should have an encoded JSON describing
    the Iceberg schema (with the correct schema id).
    """
    table_name = "schema_evolution_eg_table"

    # We want to use completely new table for each test
    table_name += "_append"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    # Write using Spark on rank 0
    passed = 1
    err = "Setup step failed. See error on rank 0"
    if bodo.get_rank() == 0:
        try:
            schema_evolution_eg_table.create_table(table_name=table_name)
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    df = pd.DataFrame(
        {
            "A": ["salad", "pizza", "pasta"] * 10,
            "D": [4352, 598237, 23480950, 21329, 423503] * 6,
            "TY": ["AB", "e3", "tyler"] * 10,
            "E": np.array([1.56, 2.72, 23.90, 100.84, 234.67, 19.00] * 5, np.float32),
            "F": np.array([56, 23, 43, 64234, 902, 943] * 5, np.int32),
        }
    )

    # Construct the expected schema object.
    # XXX TODO Investigate why it's large_string here.
    passed = 1
    err = "Constructing expected schema failed. See error on rank 0"
    expected_schema = None
    if bodo.get_rank() == 0:
        try:
            expected_schema = pa.schema(
                [
                    pa.field(
                        "A", pa.large_string(), metadata={ICEBERG_FIELD_ID_MD_KEY: "1"}
                    ),
                    pa.field("D", pa.int64(), metadata={ICEBERG_FIELD_ID_MD_KEY: "4"}),
                    pa.field(
                        "TY", pa.large_string(), metadata={ICEBERG_FIELD_ID_MD_KEY: "3"}
                    ),
                    pa.field(
                        "E", pa.float32(), metadata={ICEBERG_FIELD_ID_MD_KEY: "5"}
                    ),
                    pa.field("F", pa.int32(), metadata={ICEBERG_FIELD_ID_MD_KEY: "6"}),
                ]
            )
            iceberg_schema_str = bodo_iceberg_connector.pyarrow_to_iceberg_schema_str(
                expected_schema
            )
            # This will return a string with schema-id as 0. However, the latest
            # schema-id is 5, which is what Bodo is expected to write.
            start_idx = iceberg_schema_str.find('"schema-id":0')
            assert start_idx != -1, "schema id not in the schema string"
            iceberg_schema_str = iceberg_schema_str.replace(
                '"schema-id":0', '"schema-id":5', 1
            )
            expected_schema = expected_schema.with_metadata(
                {"iceberg.schema": iceberg_schema_str}
            )
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    data_files_before_write = set()
    if bodo.get_rank() == 0:
        data_files_before_write = set(
            glob.glob(
                os.path.join(warehouse_loc, db_schema, table_name, "data", "*.parquet")
            )
        )

    # Append using Bodo
    impl(_get_dist_arg(df), table_name, conn, db_schema)

    _verify_pq_schema_in_files(
        warehouse_loc, db_schema, table_name, data_files_before_write, expected_schema
    )


@pytest.mark.slow
def test_basic_write_runtime_cols_fail(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that Iceberg writes throw an error at compile-time when
    writing a DataFrame with runtime columns (created using a pivot)
    """

    table_name = "SIMPLE_NUMERIC_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    @bodo.jit
    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df_piv = pd.pivot_table(
            df, index=["A"], columns=["B"], values="C", aggfunc="sum"
        )
        return df_piv.to_sql(table_name, conn, db_schema, if_exists="replace")

    with pytest.raises(
        BodoError,
        match=r"DataFrame\.to_sql\(\) on DataFrames with columns determined at runtime is not yet supported\. Please return the DataFrame to regular Python to update typing information\.",
    ):
        impl(table_name, conn, db_schema)


@pytest.mark.slow
def test_basic_write_append_not_null_arrays(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that Iceberg appends can write non-nullable float and timestamp
    arrays to nullable float and timestamp columns in Iceberg. This is a
    special case since Bodo does not have nullable arrays for these types
    TODO: [BE-41] Update when nullable floating point arrays are supported
    """

    df = pd.DataFrame(
        {
            "A": pd.Series(
                [1.0, 2.0, np.nan, 3.0, 4.0, 5.0, None] * 10, dtype="object"
            ),
            "B": pd.Series(
                [1.0, 2.0, np.nan, 3.0, 4.0, 5.0, None] * 10, dtype="object"
            ),
            "C": pd.Series(
                [
                    pd.NaT,
                    None,
                    datetime(2019, 8, 21, 15, 23, 45, 0),
                    pd.NaT,
                    None,
                    datetime(2021, 1, 30, 7, 20, 30, 0),
                    pd.NaT,
                ]
                * 10,
                dtype="object",
            ),
        }
    )

    sql_schema = [("A", "float", True), ("B", "double", True), ("C", "timestamp", True)]

    table_name = "NULLABLE_TABLE_APPEND_SPARK"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    # Write using Spark on rank 0
    spark = get_spark()
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    # Cast to non-null types
    bodo_in_df = df.copy()
    bodo_in_df["A"] = bodo_in_df["A"].astype("float32")
    bodo_in_df["B"] = bodo_in_df["B"].astype("float64")
    bodo_in_df["C"] = bodo_in_df["C"].astype("datetime64[ns]")

    # Append using Bodo. Note that we can't check the output since Bodo and
    # Spark can not return nulls in float or datetime Pandas arrays. Thus,
    # we can only check that this does not fail.
    impl(_get_dist_arg(bodo_in_df), table_name, conn, db_schema)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,sql_schema,df,df_write",
    [
        pytest.param(
            "NULL",
            [("A", "int", True), ("B", "long", True)],
            pd.DataFrame(
                {
                    "A": pd.Series([1, 2, 3, 4, None] * 5, dtype="Int32"),
                    "B": pd.Series([1, 2, 3, 4, None] * 5, dtype="Int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series([6, 7, 8, 9, 10], dtype="int32"),
                    "B": pd.Series([6, 7, 8, 9, 10], dtype="int64"),
                }
            ),
            id="null",
        ),
    ],
)
def test_basic_write_upcasting(
    iceberg_database,
    iceberg_table_conn,
    name,
    sql_schema,
    df,
    df_write,
    memory_leak_check,
):
    """
    Test that Bodo is able to perform null upcasting when writing
    to Iceberg. This means writing non-nullable arrays to nullable columns.
    Note that we can only test this for ints right now since all other
    arrays types are nullable by default
    TODO: [BE-41] Update when nullable floating point arrays are supported
    """

    comm = MPI.COMM_WORLD
    n_pes = comm.Get_size()

    table_name = name + "_UPCASTING_TEST"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    # Write using Spark on rank 0
    spark = get_spark()
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    # Append using Bodo
    impl(_get_dist_arg(df_write), table_name, conn, db_schema)
    expected_df = pd.concat([df, df_write])

    # Read using Bodo and then check that it's what's expected
    bodo_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
    bodo_out = _gather_output(bodo_out)
    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            expected_df,
            bodo_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1

    spark_passed = 1
    if bodo.get_rank() == 0:
        # Read using Spark and then check that it's what's expected
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")
        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
        spark_passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    spark_n = reduce_sum(spark_passed)
    assert spark_n == n_pes


INT_MAX: int = np.iinfo(np.int32).max
INT_MIN: int = np.iinfo(np.int32).min
DOUBLE_MAX: float = np.finfo(np.float64).max
DOUBLE_MIN: float = np.finfo(np.float64).min
NULL_ERR = (
    "Iceberg Parquet Write: Column A contains nulls but is expected to be non-nullable"
)
TYPE_ERR = (
    "Iceberg Parquet Write: Column A is type int64 but is expected to be type int32"
)
OTHER_ERR = "See other ranks for runtime error"
DOWNCAST_INFO = [
    (
        "NULL",
        [("A", "int", False), ("B", "long", False)],
        pd.DataFrame(
            {
                "A": pd.Series([1, 2, 3, 4, 5] * 5, dtype="int32"),
                "B": pd.Series([1, 2, 3, 4, 5] * 5, dtype="int64"),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series([6, 7, 8, 9, 10], dtype="Int32"),
                "B": pd.Series([6, 7, 8, 9, 10], dtype="Int64"),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series([6, None, 8, None, 10], dtype="Int32"),
                "B": pd.Series([6, None, 8, None, 10], dtype="Int64"),
            }
        ),
        [NULL_ERR, OTHER_ERR],
    ),
    (
        "TYPE",
        [("A", "int", False), ("B", "float", False)],
        pd.DataFrame(
            {
                "A": pd.Series([1, 2, 3, 4, 5] * 5, dtype="int32"),
                "B": pd.Series([1, 2, 3, 4, 5] * 5, dtype="float32"),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series([6, 7, 8, 9, 10], dtype="int64"),
                "B": pd.Series([6, 7, 8, 9, 10], dtype="float64"),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series([INT_MAX + 1, INT_MIN - 1] * 3, dtype="int64"),
                "B": pd.Series([DOUBLE_MAX, DOUBLE_MIN] * 3, dtype="float64"),
            }
        ),
        [TYPE_ERR, OTHER_ERR],
    ),
    (
        "NULL_AND_TYPE",
        [("A", "int", False)],
        pd.DataFrame({"A": pd.Series([1, 2, 3, 4, 5] * 5, dtype="int32")}),
        pd.DataFrame({"A": pd.Series([6, 7, 8, 9, 10], dtype="Int64")}),
        pd.DataFrame({"A": pd.Series([INT_MAX + 1, None] * 3, dtype="Int64")}),
        [NULL_ERR, TYPE_ERR, OTHER_ERR],
    ),
]


@pytest.fixture(ids=lambda f: f[0], params=DOWNCAST_INFO)
def downcasting_table_info(request):
    return request.param


@pytest.mark.slow
def test_basic_write_downcasting_fail(
    iceberg_database,
    iceberg_table_conn,
    downcasting_table_info,
    # Tests that throw errors are known to cause memory leaks
    # memory_leak_check
):
    """
    Test that writing to an Iceberg table with incorrect types
    that would need to be downcasted fails.
    """

    id, sql_schema, df, df_write, _, _ = downcasting_table_info
    table_name = id + "_DOWNCASTING_FAIL_TEST"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    # Write using Spark on rank 0
    spark = get_spark()
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    with pytest.raises(
        BodoError,
        match="DataFrame schema needs to be an ordered subset of Iceberg table for append",
    ):
        impl(_get_dist_arg(df_write), table_name, conn, db_schema)


@pytest.mark.slow
def test_basic_write_downcasting(
    iceberg_database,
    iceberg_table_conn,
    downcasting_table_info,
    # Tests that throw errors are known to cause memory leaks
    # memory_leak_check
):
    """
    Test that writing to an Iceberg table while performing type
    and null downcasting works. This will test a situation that will
    succeed and a situation that wont. The failing cases occur when
    there is a null in the array or an overflow would occur.
    """

    comm = MPI.COMM_WORLD
    n_pes = comm.Get_size()

    id, sql_schema, df, df_write, df_fail, err_msgs = downcasting_table_info
    table_name = id + "_DOWNCASTING_TEST"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    # Write using Spark on rank 0
    spark = get_spark()
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(
            table_name,
            conn,
            db_schema,
            if_exists="append",
            _bodo_allow_downcasting=True,
        )

    # Append using Bodo
    impl(_get_dist_arg(df_write), table_name, conn, db_schema)
    expected_df = pd.concat([df, df_write])

    # Read using Bodo and then check that it's what's expected
    bodo_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
    bodo_out = _gather_output(bodo_out)
    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            expected_df,
            bodo_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1

    spark_passed = 1
    if bodo.get_rank() == 0:
        # Read using Spark and then check that it's what's expected
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")
        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
        spark_passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    spark_n = reduce_sum(spark_passed)
    assert spark_n == n_pes

    with pytest.raises(RuntimeError) as excinfo:
        impl(_get_dist_arg(df_fail), table_name, conn, db_schema)

    err_msg: str = excinfo.value.args[0]
    assert any(err_msg.startswith(msg) for msg in err_msgs)


@pytest.mark.slow
def test_basic_write_downcasting_copy(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that downcasting during Iceberg write does not affect the
    original dataframe by using it after the write step
    """

    _, sql_schema, df, df_write, _, _ = DOWNCAST_INFO[1]
    table_name = "DOWNCASTING_COPY_TABLE"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    # Write using Spark on rank 0
    spark = get_spark()
    if bodo.get_rank() == 0:
        create_iceberg_table(df, sql_schema, table_name, spark)
    bodo.barrier()

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(
            table_name,
            conn,
            db_schema,
            if_exists="append",
            _bodo_allow_downcasting=True,
        )
        return df

    # Append using Bodo and Get Output
    new_df = impl(_get_dist_arg(df_write), table_name, conn, db_schema)
    comm = MPI.COMM_WORLD
    passed = _test_equal_guard(_get_dist_arg(df_write), new_df)
    assert reduce_sum(passed) == comm.Get_size()


@pytest.mark.slow
def test_iceberg_write_error_checking(iceberg_database, iceberg_table_conn):
    """
    Tests for known errors thrown when writing an Iceberg table.
    """
    table_name = "SIMPLE_BOOL_BINARY_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    df = SIMPLE_TABLES_MAP[table_name][0]

    # Check that error is raised when schema is not provided
    def impl1(df, table_name, conn):
        df.to_sql(table_name, conn)

    with pytest.raises(
        ValueError,
        match="schema must be provided when writing to an Iceberg table",
    ):
        bodo.jit(distributed=["df"])(impl1)(df, table_name, conn)

    # Check that error is raised when chunksize is provided
    def impl2(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, chunksize=5)

    with pytest.raises(ValueError, match="chunksize not supported for Iceberg tables"):
        bodo.jit(distributed=["df"])(impl2)(df, table_name, conn, db_schema)

    # TODO Remove after adding replicated write support
    # Check that error is raise when trying to write a replicated dataframe
    # (unsupported for now)
    def impl3(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema)

    with pytest.raises(
        AssertionError, match="Iceberg Write only supported for distributed DataFrames"
    ):
        bodo.jit(replicated=["df"])(impl3)(df, table_name, conn, db_schema)


@pytest.mark.slow
def test_read_pq_write_iceberg(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Some compilation errors can only be observed when running multiple steps.
    This is to test one such common use case, which is reading a table
    from a parquet file and writing it as an Iceberg table.
    This unit test was added as part of https://github.com/bodo-ai/Bodo/pull/4145
    where an error for such use case was found.
    """

    # The exact table to use doesn't matter, so picking one at random.
    df = SIMPLE_TABLES_MAP["SIMPLE_NUMERIC_TABLE"][0]
    fname = "test_read_pq_write_iceberg_ds.pq"

    # Give it a unique name so there's no conflicts.
    table_name = "TEST_READ_PQ_WRITE_ICEBERG_TABLE"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(pq_fname, table_name, conn, db_schema):
        df = pd.read_parquet(pq_fname)
        df.to_sql(
            table_name,
            conn,
            db_schema,
            if_exists="replace",
            index=False,
        )

    with ensure_clean2(fname):
        if bodo.get_rank() == 0:
            df.to_parquet(fname)
        bodo.barrier()
        # We're just running to make sure that it executes,
        # not for correctness itself, since that is
        # already being tested by the other unit tests.
        bodo.jit(impl)(fname, table_name, conn, db_schema)


@pytest.mark.slow
def test_iceberg_missing_optional_column(iceberg_database, iceberg_table_conn):
    """
    Test support for adding a dataframe to an iceberg table where the dataframe
    is missing an optional column.
    The entire column should be filled with nulls instead of failing.
    """
    table_name = "SIMPLE_OPTIONAL_TABLE"
    write_table_name = f"{table_name}_WRITE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    if bodo.get_rank() == 0:
        spark = get_spark()
        spark.sql(
            f"CREATE TABLE hadoop_prod.{db_schema}.{write_table_name} AS SELECT * FROM hadoop_prod.{db_schema}.{table_name}"
        )
    bodo.barrier()

    # Test that a dataframe with a missing optional column can be appended
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append", index=False)

    df = pd.DataFrame(
        {
            "A": np.array([1, 2, 3, 4] * 25, dtype=np.int32),
        }
    )
    try:
        bodo.jit(distributed=["df"])(impl)(
            _get_dist_arg(df), write_table_name, conn, db_schema
        )

        # Read the columns with Spark and check that the missing column is filled
        # with nulls.
        spark_out = spark_reader.read_iceberg_table_single_rank(
            write_table_name, db_schema
        )

        assert (
            list(spark_out["B"]).count(None) == 100
        ), "Missing column not filled with nulls on spark read"

        # Read the columns with Bodo and check that the missing column is filled
        # with NAs.
        @bodo.jit
        def read_bodo(table_name, conn, db_schema):
            return pd.read_sql_table(table_name, conn, db_schema)

        bodo_out = read_bodo(write_table_name, conn, db_schema)
        assert (
            reduce_sum(bodo_out["B"].isna().sum()) == 100
        ), "Missing column not filled with nulls on Bodo read"
    finally:
        if bodo.get_rank() == 0:
            spark.sql(
                f"DROP TABLE IF EXISTS hadoop_prod.{db_schema}.{write_table_name}"
            )


@pytest.mark.slow
def test_iceberg_missing_optional_column_missing_error(
    iceberg_database, iceberg_table_conn
):
    """
    Test that the correct error is thrown when a dataframe is missing a required
    column.
    """
    table_name = "SIMPLE_OPTIONAL_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    # Test that a dataframe with a missing optional column can be appended
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append", index=False)

    df = pd.DataFrame(
        {
            "B": np.array([1, 2, 3, 4] * 25, dtype=np.int32),
        }
    )

    with pytest.raises(
        BodoError,
        match="DataFrame schema needs to be an ordered subset of Iceberg table for append",
    ):
        bodo.jit(distributed=["df"])(impl)(df, table_name, conn, db_schema)


@pytest.mark.slow
def test_iceberg_missing_optional_column_extra_error(
    iceberg_database, iceberg_table_conn
):
    """
    Test support for adding a dataframe to an iceberg table where the dataframe
    has an additional column that is not in the Iceberg table schema.
    """
    table_name = "SIMPLE_OPTIONAL_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    # Test that a dataframe with a missing optional column can be appended
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append", index=False)

    df = pd.DataFrame(
        {
            "A": np.array([1, 2, 3, 4] * 25, dtype=np.int32),
            "C": np.array([1, 2, 3, 4] * 25, dtype=np.int32),
        }
    )

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame schema needs to be an ordered subset of Iceberg table for append"
        ),
    ):
        bodo.jit(distributed=["df"])(impl)(df, table_name, conn, db_schema)


@pytest.mark.slow
def test_iceberg_missing_optional_column_incorrect_field_order(
    iceberg_database, iceberg_table_conn
):
    """
    Test that the correct error is thrown when a dataframe columns in incorrect order.
    """
    table_name = "SIMPLE_OPTIONAL_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    # Test that a dataframe with a missing optional column can be appended
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append", index=False)

    df = pd.DataFrame(
        {
            "B": np.array(["a", "b", "c", "d"] * 25),
            "A": np.array([1, 2, 3, 4] * 25, dtype=np.int32),
        }
    )

    with pytest.raises(
        BodoError,
        match=re.escape(
            "DataFrame schema needs to be an ordered subset of Iceberg table for append"
        ),
    ):
        bodo.jit(distributed=["df"])(impl)(df, table_name, conn, db_schema)


@pytest.mark.slow
def test_iceberg_middle_optional_column(iceberg_database, iceberg_table_conn):
    """
    Test support for adding a dataframe to an iceberg table where the dataframe
    is missing an optional column in the middle of the schema and in the middle
    of a struct.
    The entire column should be filled with nulls instead of failing.
    """
    table_name = "SIMPLE_OPTIONAL_TABLE_MIDDLE"
    write_table_name = f"{table_name}_WRITE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    if bodo.get_rank() == 0:
        spark = get_spark()
    else:
        spark = None

    @run_rank0
    def create_table():
        spark.sql(
            f"CREATE TABLE hadoop_prod.{db_schema}.{write_table_name} AS SELECT * FROM hadoop_prod.{db_schema}.{table_name}"
        )

    create_table()

    # Test that a dataframe with a missing optional column can be appended
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append", index=False)

    df = pd.DataFrame(
        {
            "A": np.array([1, 2, 3, 4] * 25, dtype=np.int32),
            "C": pd.Series(
                [{"f1": 10.321, "f3": 30.5}] * 100,
                dtype=pd.ArrowDtype(
                    pa.struct([("f1", pa.float64()), ("f3", pa.float64())])
                ),
            ),
            "D": np.array([1, 2, 3, 4] * 25, dtype=np.int64),
        }
    )
    try:
        bodo.jit(distributed=["df"])(impl)(
            _get_dist_arg(df), write_table_name, conn, db_schema
        )

        # Read the bodo output
        @bodo.jit
        def read_bodo(table_name, conn, db_schema):
            return pd.read_sql_table(table_name, conn, db_schema)

        bodo_out = read_bodo(write_table_name, conn, db_schema)
        bodo_out = _gather_output(bodo_out)

        # Read the columns with Spark and check that the missing column is filled
        # with nulls.
        @run_rank0
        def validate_output(bodo_out):
            spark_out, _, _ = spark_reader.read_iceberg_table(
                write_table_name, db_schema
            )
            assert (
                spark_out["B"].isna().sum() == 100
            ), "Missing column not filled with nulls on spark read"
            assert (
                spark_out["C"].map(lambda x: x["f2"]).isna().sum() == 100
            ), "Missing field not filled with nulls on spark read"
            assert (
                bodo_out["B"].isna().sum() == 100
            ), "Missing column not filled with nulls on Bodo read"
            assert (
                bodo_out["C"].map(lambda x: x["f2"]).isna().sum() == 100
            ), "Missing field not filled with nulls on Bodo read"
            bodo_out = convert_non_pandas_columns(bodo_out)
            spark_out = convert_non_pandas_columns(spark_out)
            assert _test_equal_guard(
                bodo_out, spark_out, reset_index=True, sort_output=True
            ), "Bodo and Spark outputs do not match"

        validate_output(bodo_out)
    finally:
        if bodo.get_rank() == 0:
            spark.sql(
                f"DROP TABLE IF EXISTS hadoop_prod.{db_schema}.{write_table_name}"
            )


def truncate_impl(x: pd.Series, W: int):
    """
    Apply the Iceberg truncate transform on Pandas series x.

    Args:
        x (pd.Series): Array to transform
        W (int): width for the truncate operation.

    Raises:
        NotImplementedError: when dtype of x is not string or integer

    Returns:
        Truncated array
    """
    if x.dtype in ["str", "object", "string[python]"]:
        return x.str.slice(stop=W)
    elif x.dtype in ["int32", "Int32", "int64", "Int64"]:
        return x - (((x % W) + W) % W)
    else:
        raise NotImplementedError(f"truncate_impl not implemented for {x.dtype}")


def null_scalar_wrapper(inner):
    return lambda x, y: None if x == "null" else inner(x, y)


def truncate_scalar_impl(x, W: int):
    if isinstance(x, str):
        return x[:W]
    elif isinstance(x, int):
        return x - (((x % W) + W) % W)
    else:
        raise NotImplementedError(f"truncate_scalar_impl not implemented for {type(x)}")


def bucket_scalar_impl(x, y: int) -> int | None:
    if x is None:
        return None
    if x is pd.NA:
        return x

    # Pandas 2.2.2 converts Int64 data to float in Series.apply for some reason.
    # We work around this bug by converting back to int.
    if isinstance(x, float):
        if np.isnan(x):
            return pd.NA
        x = int(x)

    if isinstance(x, int):
        res = mmh3.hash(struct.pack("<q", x))
    elif isinstance(x, (datetime, pd.Timestamp)):
        if pd.isna(x):
            return None
        res = mmh3.hash(
            struct.pack(
                "<q",
                int(
                    round(
                        (x - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
                        * 1e6
                    )
                ),
            )
        )
    elif isinstance(x, date):
        # Based on https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements,
        # dates should be hashed as int32 (after computing number of days since epoch),
        # however Spark, Iceberg-Python and example in Iceberg Spec seems to use
        # int64.
        res = mmh3.hash(struct.pack("<q", (x - date(1970, 1, 1)).days))
    elif isinstance(x, str):
        res = mmh3.hash(x)
    else:
        # TODO: Include Date/Time, Decimal, and UUID Types
        raise NotImplementedError(f"bucket_scalar_impl not implemented for {type(x)}")

    return (res & 2147483647) % y


def identity_scalar_impl(x: str, y):
    if x == "true":
        return True
    elif x == "false":
        return False
    else:
        return x


def month_scalar_impl(x: str, y):
    if "-" in x:
        parsed = x.split("-")
        year = int(parsed[0])
        month = int(parsed[1])
        return (year - 1970) * 12 + month - 1
    else:
        return int(x)


SCALAR_TRANSFORM_FUNC = {
    "years": null_scalar_wrapper(
        lambda x, _: int(x) - 1970 if int(x) > 1000 else int(x)
    ),
    "months": month_scalar_impl,
    "days": lambda x, _: (
        datetime.strptime(x, "%Y-%m-%d").date() - date(1970, 1, 1)
        if "-" in x
        else int(x)
    ),
    "hours": lambda x, _: (
        (datetime.strptime(x, "%Y-%m-%d-%H") - datetime(1970, 1, 1)).total_seconds()
        // 3600
        if "-" in x
        else int(x)
    ),
    "identity": null_scalar_wrapper(identity_scalar_impl),
    "truncate": truncate_scalar_impl,
    "bucket": lambda x, _: int(x),  # The scalar is already correct
}

ARRAY_TRANSFORM_FUNC = {
    "years": lambda df, _: df.apply(lambda x: None if pd.isna(x) else x.year - 1970),
    "months": lambda df, _: df.apply(
        lambda x: None if pd.isna(x) else (x.year - 1970) * 12 + x.month - 1
    ),
    "days": lambda df, _: df.apply(
        lambda x: (
            None
            if pd.isna(x)
            else (
                (x.date() if isinstance(x, (datetime, pd.Timestamp)) else x)
                - date(1970, 1, 1)
            ).days
        )
    ),
    "hours": lambda df, _: df.apply(
        lambda x: (
            None if pd.isna(x) else (x.date() - date(1970, 1, 1)).days * 24 + x.hour
        )
    ),
    "identity": lambda df, _: df,
    "truncate": truncate_impl,
    # Since the function can return pd.NA, cast to nullable integer array by default
    "bucket": lambda df, val: df.apply(lambda x: bucket_scalar_impl(x, val)).astype(
        "Int64"
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("base_name,part_spec", PARTITION_MAP)
def test_write_partitioned(
    iceberg_database,
    iceberg_table_conn,
    base_name: str,
    part_spec: list[PartitionField],
    memory_leak_check,
):
    """
    Tests that appending to a table with a defined partition spec works
    as expected, i.e. the generated files are partitioned based on the
    partitioned spec and the transform values are as expected.
    We then also read the table back using Spark and Bodo and validate
    that the contents are as expected.
    """
    table_name = part_table_name(base_name, part_spec)
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=True)
    comm = MPI.COMM_WORLD

    if "DECIMALS" in table_name:
        pytest.skip(reason="Bodo cannot write decimals other than (38,18).")
    if "STRUCT" in table_name:
        # TODO Add support for this.
        pytest.skip(
            reason="Bodo doesn't support writing tables where the partition column is a nested field."
        )

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "DICT_ENCODED_STRING_TABLE":
        bodo.hiframes.boxing._use_dict_str_type = True

    # TODO Add repl test when supported

    df = SIMPLE_TABLES_MAP[f"SIMPLE_{base_name}"][0]
    try:
        impl(_get_dist_arg(df), table_name, conn, db_schema)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type

    bodo.barrier()

    # Check that the files are correctly partitioned
    passed = 1
    err = "File partition validation failed. See error on rank 0."
    if bodo.get_rank() == 0:
        try:
            data_files: list[str] = glob.glob(
                os.path.join(warehouse_loc, db_schema, table_name, "data", "**.parquet")
            )
            for data_file in data_files:
                _test_file_part(data_file, part_spec)
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    # Read the table back using Spark and Bodo and validate that the
    # contents are as expected
    expected_df = pd.concat([df, df]).reset_index(drop=True)

    if base_name == "BOOL_BINARY_TABLE":
        # [BE-3585] Bodo write binary columns as string when partitioned,
        # so validating by reading the table back would fail.
        return

    # Spark doesn't handle null timestamps properly. It converts them to
    # 0 (i.e. epoch) instead of NaTs like Pandas does. This modifies expected
    # df to match Spark.
    if base_name == "DT_TSZ_TABLE":
        expected_df["B"] = expected_df["B"].fillna(pd.Timestamp(1970, 1, 1, tz="UTC"))

    # Validate Bodo read output:
    bodo_out = bodo.jit(distributed=["df"])(
        lambda: pd.read_sql_table(table_name, conn, db_schema)
    )()  # type: ignore

    # Spark can have inconsistent behavior when reading/writing null
    # timestamps, so we convert all NaTs to epoch for consistent
    # comparison
    if base_name == "DT_TSZ_TABLE":
        bodo_out["B"] = bodo_out["B"].fillna(
            pd.Timestamp(year=1970, month=1, day=1, tz="UTC")
        )
    bodo_out = _gather_output(bodo_out)

    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            expected_df,
            bodo_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo read output doesn't match expected output"

    passed = None
    if bodo.get_rank() == 0:
        spark = get_spark()
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")
        # Validate Spark read output:
        # We need to invalidate spark cache, because it doesn't realize
        # that the table has been modified.

        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema, spark)
        # Spark doesn't handle null timestamps consistently. It converts them to
        # 0 (i.e. epoch) instead of NaTs like Pandas does. This modifies the
        # dataframe to match Spark.
        if base_name == "DT_TSZ_TABLE":
            spark_out["B"] = (
                spark_out["B"].astype("datetime64[ns]").dt.tz_localize("UTC")
            )
            spark_out["B"] = spark_out["B"].fillna(pd.Timestamp(1970, 1, 1, tz="UTC"))
        passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    passed = comm.bcast(passed)
    assert passed == 1, "Spark read output doesn't match expected output"


@pytest.fixture(params=SORT_MAP, ids=lambda x: sort_table_name(x[0], x[1]))
def sort_cases(request):
    base_name, sort_order = request.param
    return (base_name, sort_order, sort_table_name(base_name, sort_order))


@pytest.mark.slow
def test_write_sorted(
    iceberg_database, iceberg_table_conn, sort_cases, memory_leak_check
):
    """
    Test that we can append to tables with a defined sort-order.
    We append rows to the table and then verify that all files
    for the table are sorted as expected.
    We then also read the table back using Spark and Bodo and validate
    that the contents are as expected.
    """
    base_name, sort_order, table_name = sort_cases
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=True)
    comm = MPI.COMM_WORLD

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "DICT_ENCODED_STRING_TABLE":
        bodo.hiframes.boxing._use_dict_str_type = True

    df = SIMPLE_TABLES_MAP[f"SIMPLE_{base_name}"][0]
    try:
        impl(_get_dist_arg(df), table_name, conn, db_schema)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
    bodo.barrier()

    # TODO Add repl test when supported

    # Validate that the files are sorted based on the sort order
    passed = 1
    err = "Sorted file validation failed. See error on rank 0"
    if bodo.get_rank() == 0:
        try:
            # Get List of Sorted Data Files
            data_files = glob.glob(
                os.path.join(warehouse_loc, db_schema, table_name, "data", "*.parquet")
            )
            assert all(os.path.isfile(file) for file in data_files)

            # Check Contents of Each Folder
            for data_file in data_files:
                _test_file_sorted(data_file, sort_order)
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    # Read the table back using Spark and Bodo and validate that the
    # contents are as expected
    expected_df = pd.concat([df, df]).reset_index(drop=True)
    # Spark can have inconsistent behavior when reading/writing null
    # timestamps, so we convert all NaTs to epoch for consistent
    # comparison
    if base_name == "DT_TSZ_TABLE":
        expected_df["B"] = expected_df["B"].fillna(pd.Timestamp(1970, 1, 1, tz="UTC"))

    if base_name == "BOOL_BINARY_TABLE":
        # [BE-3585] Bodo write binary columns as string when partitioned,
        # so validating by reading the table back would fail.
        return

    # Validate Bodo read output:
    bodo_out = bodo.jit(distributed=["df"])(
        lambda: pd.read_sql_table(table_name, conn, db_schema)
    )()  # type: ignore
    # Spark can have inconsistent behavior when reading/writing null
    # timestamps, so we convert all NaTs to epoch for consistent
    # comparison
    if base_name == "DT_TSZ_TABLE":
        bodo_out["B"] = bodo_out["B"].fillna(
            pd.Timestamp(year=1970, month=1, day=1, tz="UTC")
        )
    bodo_out = _gather_output(bodo_out)

    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            expected_df,
            bodo_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo read output doesn't match expected output"

    passed = None
    if bodo.get_rank() == 0:
        # Validate Spark read output:
        # We need to invalidate spark cache, because it doesn't realize
        # that the table has been modified.
        spark = get_spark()
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")
        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema, spark)
        # Spark doesn't handle null timestamps consistently. It sometimes converts them to
        # 0 (i.e. epoch) instead of NaTs like Pandas does. This modifies expected
        # df to match Spark.
        if base_name == "DT_TSZ_TABLE":
            spark_out["B"] = (
                spark_out["B"].astype("datetime64[ns]").dt.tz_localize("UTC")
            )
            spark_out["B"] = spark_out["B"].fillna(pd.Timestamp(1970, 1, 1, tz="UTC"))
        passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    passed = comm.bcast(passed)
    assert passed == 1, "Spark read output doesn't match expected output"


@pytest.mark.slow
@pytest.mark.parametrize("use_dict_encoding_boxing", [False, True])
def test_write_part_sort(
    iceberg_database, iceberg_table_conn, use_dict_encoding_boxing, memory_leak_check
):
    """
    Append to a table with both a partition spec and a sort order,
    and verify that the append was done correctly, i.e. validate
    that each file is correctly sorted and partitioned.
    Then read the table using Spark and Bodo and validate that the
    output is as expected.
    """
    table_name = f"PARTSORT_{PART_SORT_TABLE_BASE_NAME}"
    df, sql_schema = SIMPLE_TABLES_MAP[f"SIMPLE_{PART_SORT_TABLE_BASE_NAME}"]
    if use_dict_encoding_boxing:
        table_name += "_DICT_ENCODING"
        spark = get_spark()
        if bodo.get_rank() == 0:
            create_iceberg_table(
                df,
                sql_schema,
                table_name,
                spark,
                PART_SORT_TABLE_PARTITION_SPEC,
                PART_SORT_TABLE_SORT_ORDER,
            )
        bodo.barrier()
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=True)

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    bodo.hiframes.boxing._use_dict_str_type = use_dict_encoding_boxing

    try:
        impl(_get_dist_arg(df), table_name, conn, db_schema)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
    bodo.barrier()

    data_files = glob.glob(
        os.path.join(warehouse_loc, db_schema, table_name, "data", "**.parquet")
    )
    assert all(os.path.isfile(file) for file in data_files)

    passed = 1
    err = "Partition-Spec/Sort-Order validation of files failed. See error on rank 0"
    if bodo.get_rank() == 0:
        try:
            for data_file in data_files:
                _test_file_part(data_file, PART_SORT_TABLE_PARTITION_SPEC)
                _test_file_sorted(data_file, PART_SORT_TABLE_SORT_ORDER)
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    # Read the table back using Spark and Bodo and validate that the
    # contents are as expected
    expected_df = pd.concat([df, df]).reset_index(drop=True)

    passed = None
    comm = MPI.COMM_WORLD
    if bodo.get_rank() == 0:
        # Validate Spark read output:
        # We need to invalidate spark cache, because it doesn't realize
        # that the table has been modified.
        spark = get_spark()
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema, spark)
        passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    passed = comm.bcast(passed)
    assert passed == 1, "Spark read output doesn't match expected output"

    # Validate Bodo read output:
    bodo_out = bodo.jit(distributed=["df"])(
        lambda: pd.read_sql_table(table_name, conn, db_schema)
    )()  # type: ignore
    bodo_out = _gather_output(bodo_out)

    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            expected_df,
            bodo_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo read output doesn't match expected output"


def _test_file_part(file_name: str, part_spec: list[PartitionField]):
    # Construct Expected Partition Values
    before = True
    part_folders = [
        p.name
        for p in Path(file_name).parents
        if (before := before and not str(p).endswith("data"))
    ]
    part_values = [folder.split("=")[1] for folder in part_folders]
    expected_vals = [
        SCALAR_TRANSFORM_FUNC[trans](val, tval)
        for val, (_, trans, tval) in zip(part_values, part_spec)
    ]

    # Check if data adheres to partitioning
    df = pd.read_parquet(file_name)

    for (col, trans, tval), expected_val in zip(part_spec, expected_vals):
        trans_col = ARRAY_TRANSFORM_FUNC[trans](df[col], tval)

        if expected_val is None:
            assert (
                trans_col.isnull()
            ).all(), "Partition value does not equal the result after applying the transformation"
        else:
            expected_col = pd.Series([expected_val]).astype(trans_col.dtype)[0]
            assert (
                trans_col == expected_col
            ).all(), "Partition value does not equal the result after applying the transformation"


def _test_file_sorted(file_name: str, sort_order: list[SortField]):
    df = pd.read_parquet(file_name, dtype_backend="numpy_nullable")
    df = cast_dt64_to_ns(df)

    # Compute Transformed Columns
    new_cols: list[pd.Series] = [
        ARRAY_TRANSFORM_FUNC[trans](df[col], val)
        for (col, trans, val, _, _) in sort_order
    ]
    idxs = [str(x) for x in range(len(sort_order))]
    df_vals = pd.DataFrame(dict(zip(idxs, new_cols)))

    # Check if it's sorted
    ascending = [s.asc for s in sort_order]
    na_position = ["first" if s.nulls_first else "last" for s in sort_order]

    @bodo.jit(distributed=False)
    def bodo_sort(df):
        res = df.sort_values(
            by=idxs,
            ascending=ascending,
            na_position=na_position,
            ignore_index=False,
        )
        return res.reset_index(drop=True)

    sorted_vals = bodo_sort(df_vals)
    _test_equal(df_vals, sorted_vals, check_dtype=False)


@pytest.mark.slow
@pytest.mark.parametrize("use_dict_encoding_boxing", [False, True])
def test_write_part_sort_return_orig(
    iceberg_database,
    iceberg_table_conn,
    use_dict_encoding_boxing,
):
    """
    Test that when writing to a table with a defined partition-spec
    and sort order, performing the sort for the write doesn't modify
    the original dataframe.
    This tests that refcounts are handled correctly in the C++
    code. If there's an issue, this should segfault.
    """

    comm = MPI.COMM_WORLD
    table_name = "TEST_WRITE_SORTED_RETURN_ORIG_TABLE"
    df, sql_schema = SIMPLE_TABLES_MAP[f"SIMPLE_{PART_SORT_TABLE_BASE_NAME}"]
    if use_dict_encoding_boxing:
        table_name += "_DICT_ENCODING"

    if bodo.get_rank() == 0:
        spark = get_spark()
        create_iceberg_table(
            df,
            sql_schema,
            table_name,
            spark,
            PART_SORT_TABLE_PARTITION_SPEC,
            PART_SORT_TABLE_SORT_ORDER,
        )
    bodo.barrier()
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=True)

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(
            table_name,
            conn,
            db_schema,
            if_exists="append",
            index=False,
        )
        return df

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    bodo.hiframes.boxing._use_dict_str_type = use_dict_encoding_boxing

    try:
        out = impl(_get_dist_arg(df), table_name, conn, db_schema)
        out = _gather_output(out)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
    bodo.barrier()

    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            df,
            out,
            # Do not sort since that defeats the purpose
            sort_output=False,
            check_dtype=False,
            reset_index=False,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo function output doesn't match expected output"


@pytest.mark.slow
def test_merge_into_cow_write_api(
    iceberg_database,
    iceberg_table_conn,
):
    comm = MPI.COMM_WORLD
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()
    # Should always only run this test on rank O
    if bodo.get_rank() != 0:
        passed = comm.bcast(False)
        if not passed:
            raise Exception("Exception on Rank 0")
        return

    passed = True
    try:
        # Create a table to work off of
        table_name = "MERGE_INTO_COW_WRITE_API"
        if bodo.get_rank() == 0:
            df = pd.DataFrame({"A": [1, 2, 3, 4]})
            create_iceberg_table(df, [("A", "long", True)], table_name)

        conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
        conn = bodo.io.iceberg.format_iceberg_conn(conn)

        # Get relevant read info from connector
        snapshot_id = bodo_iceberg_connector.bodo_connector_get_current_snapshot_id(
            conn, db_schema, table_name
        )
        old_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        # Write new file into warehouse location
        new_fname = "new_file.parquet"
        df_new = pd.DataFrame({"A": [5, 6, 7, 8]})
        df_new.to_parquet(
            os.path.join(warehouse_loc, db_schema, table_name, "data", new_fname)
        )

        # Commit the MERGE INTO COW operation
        success = bodo_iceberg_connector.commit_merge_cow(
            conn,
            db_schema,
            table_name,
            warehouse_loc,
            old_fnames,
            [new_fname],
            [0],
            [{"rowCount": 4}],
            snapshot_id,
        )
        assert success, "MERGE INTO Commit Operation Failed"

        # See if the reported files to read is only the new file
        new_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        assert len(new_fnames) == 1 and new_fnames[0] == os.path.join(
            db_schema, table_name, "data", new_fname
        )

    except Exception as e:
        passed = False
        raise e
    finally:
        passed = comm.bcast(passed)


@pytest.mark.slow
def test_merge_into_cow_write_api_partitioned(
    iceberg_database,
    iceberg_table_conn,
):
    """
    Test the Iceberg Connectors MERGE INTO COW Write Operation
    with partitioned Iceberg tables
    """

    comm = MPI.COMM_WORLD
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()

    # Should always only run this test on rank O
    if bodo.get_rank() != 0:
        passed = comm.bcast(False)
        if not passed:
            raise Exception("Exception on Rank 0")
        return

    passed = True
    try:
        # Create a table to work off of
        table_name = "MERGE_INTO_COW_WRITE_API_PARTITIONED"
        df, sql_schema = SIMPLE_TABLES_MAP["SIMPLE_PRIMITIVES_TABLE"]
        create_iceberg_table(
            df,
            sql_schema,
            table_name,
            par_spec=[PartitionField("C", "identity", -1)],
        )

        conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
        conn = bodo.io.iceberg.format_iceberg_conn(conn)

        # Get relevant read info from connector
        snapshot_id = bodo_iceberg_connector.bodo_connector_get_current_snapshot_id(
            conn, db_schema, table_name
        )
        old_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        # Write new partitioned files into warehouse location
        new_paths = []
        record_counts = []

        # Write a new data file to the "warehouse" to use in the operation
        # Note, Bodo will essentially do this internally at Parquet Write
        for part_val, name in [(True, "true"), (False, "false"), (None, "null")]:
            sub_df = df[df["C"] == part_val]
            if len(sub_df) == 0:
                continue

            record_counts.append(len(sub_df))
            new_path = os.path.join(f"C={name}", "new_file.parquet")
            new_paths.append(new_path)

            sub_df.to_parquet(
                os.path.join(warehouse_loc, db_schema, table_name, "data", new_path)
            )

        # Commit the MERGE INTO COW operation
        success = bodo_iceberg_connector.commit_merge_cow(
            conn,
            db_schema,
            table_name,
            warehouse_loc,
            old_fnames,
            new_paths,
            [0] * len(new_paths),
            [{"rowCount": x} for x in record_counts],
            snapshot_id,
        )
        assert success, "MERGE INTO Commit Operation Failed"

        # See if the reported files to read is only the new file
        out_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        assert set(out_fnames) == {
            os.path.join(db_schema, table_name, "data", path) for path in new_paths
        }

    except Exception as e:
        passed = False
        raise e
    finally:
        passed = comm.bcast(passed)


@pytest.mark.slow
def test_merge_into_cow_write_api_snapshot_check(
    iceberg_database,
    iceberg_table_conn,
):
    comm = MPI.COMM_WORLD
    bodo.barrier()

    # Note that for the connector, conn_str and warehouse_loc are the same
    db_schema, warehouse_loc = iceberg_database()

    # Should always only run this test on rank O
    if bodo.get_rank() != 0:
        passed = comm.bcast(False)
        if not passed:
            raise Exception("Exception on Rank 0")
        return

    passed = True
    try:
        # Create a table to work off of
        table_name = "MERGE_INTO_COW_WRITE_API_SNAPSHOT_CHECK"
        df = pd.DataFrame({"A": [1, 2, 3, 4]})
        create_iceberg_table(df, [("A", "long", True)], table_name)

        conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
        # Format the connection string since we don't go through pd.read_sql_table
        conn = bodo.io.iceberg.format_iceberg_conn(conn)

        # Get relevant read info from connector
        snapshot_id = bodo_iceberg_connector.bodo_connector_get_current_snapshot_id(
            conn, db_schema, table_name
        )
        old_fnames = [
            x.orig_path
            for x in bodo_iceberg_connector.get_bodo_parquet_info(
                conn,
                db_schema,
                table_name,
                None,
            )[0]
        ]

        # Write new file into warehouse location
        new_fname = "new_file.parquet"
        df_new = pd.DataFrame({"A": [5, 6, 7, 8]})
        df_new.to_parquet(
            os.path.join(warehouse_loc, db_schema, table_name, "data", new_fname)
        )

        # Update the current snapshot ID by appending data to the table
        spark = get_spark()
        spark.sql(
            f"INSERT INTO hadoop_prod.{db_schema}.{table_name} VALUES (10), (11), (12), (13)"
        )

        # Attempt to commit a MERGE INTO operation with the old snapshot id
        # Expect it return False (and prints error)
        success = bodo_iceberg_connector.commit_merge_cow(
            conn,
            db_schema,
            table_name,
            warehouse_loc,
            old_fnames,
            [new_fname],
            [0],
            [{"rowCount": 4}],
            snapshot_id,
        )
        assert not success, "MERGE INTO Commit Operation should not have succeeded"

    except Exception as e:
        passed = False
        raise e
    finally:
        passed = comm.bcast(passed)


@pytest.mark.slow
def test_merge_into_cow_simple_e2e(iceberg_database, iceberg_table_conn):
    """
    Tests a simple end to end example of reading with _bodo_merge_into, performing some modifications,
    and that writing the changes back with iceberg_merge_cow_py
    """

    comm = MPI.COMM_WORLD

    # Create a table to work off of
    table_name = "MERGE_INTO_COW_WRITE_SIMPLE_E2E"
    if bodo.get_rank() == 0:
        df = pd.DataFrame({"A": np.arange(10)})
        create_iceberg_table(df, [("A", "long", True)], table_name)
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl1(table_name, conn, db_schema):
        df, old_fnames, old_snapshot_id = pd.read_sql_table(
            table_name, conn, db_schema, _bodo_merge_into=True
        )  # type: ignore
        df_update = df[df.A > 5]
        df_update["A"] = df_update["A"] * -10

        df = df.merge(df_update, on="_BODO_ROW_ID", how="left")
        df["A"] = df["A_y"].fillna(df["A_x"])

        df = df.drop(columns=["_BODO_ROW_ID", "A_y", "A_x"])
        bodo.io.iceberg.iceberg_merge_cow_py(
            table_name, conn, db_schema, df, old_snapshot_id, old_fnames
        )
        return old_snapshot_id

    old_snapshot_id = bodo.jit(impl1)(table_name, conn, db_schema)
    bodo.barrier()

    passed = True
    if bodo.get_rank() == 0:
        # We had issues with spark caching previously, this alleviates those issues
        spark = get_spark()
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

        snapshot_id_table = spark.sql(
            f"""
            select snapshot_id from hadoop_prod.{db_schema}.{table_name}.history order by made_current_at DESC
            """
        ).toPandas()

        # We expect to see two snapshot id's, the first from the creation/insertion of the initial values into
        # the table, and the second from when we do the merge into.
        passed = len(snapshot_id_table) == 2 and (
            old_snapshot_id == snapshot_id_table.iloc[1, 0]
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Snapshot ID's do not match expected output"

    passed = True
    if bodo.get_rank() == 0:
        bodo_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
        # See comment above, expected may change if we spark version changes,
        # but we will always see the negative values (-60, -70, -80, -90),
        # and never see values > 5
        expected = pd.DataFrame({"A": [0, 1, 2, 3, 4, 5, -60, -70, -80, -90]})
        passed = _test_equal_guard(
            bodo_out,
            expected,
            check_dtype=False,
            sort_output=True,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo function output doesn't match expected output"


@pytest.mark.slow
def test_merge_into_cow_simple_e2e_partitions(iceberg_database, iceberg_table_conn):
    """
    Tests a simple end to end example of reading with _bodo_merge_into, performing some modifications,
    and then writing the changes back with iceberg_merge_cow_py, this time, on a partitioned table,
    where we should only read/write back certain files.
    """

    comm = MPI.COMM_WORLD

    # Create a table to work off of
    table_name = "MERGE_INTO_COW_WRITE_SIMPLE_E2E_PARTITIONS"

    orig_df = pd.DataFrame(
        {
            "A": list(np.arange(8)) * 8,
            "B": gen_nonascii_list(64),
            "C": np.arange(64),
            "D": pd.date_range("2017-01-03", periods=8).repeat(8).tz_localize("UTC"),
        }
    )

    if bodo.get_rank() == 0:
        create_iceberg_table(
            orig_df,
            [
                ("A", "long", True),
                ("B", "string", True),
                ("C", "long", True),
                ("D", "timestamp", True),
            ],
            table_name,
            par_spec=[
                PartitionField("A", "identity", -1),
                PartitionField("D", "days", -1),
            ],
        )
    bodo.barrier()

    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl1(table_name, conn, db_schema):
        df, old_fnames, old_snapshot_id = pd.read_sql_table(
            table_name, conn, db_schema, _bodo_merge_into=True
        )  # type: ignore
        # df is partitioned on int column A and the date column D
        df_update = df[df.A > 4][["_BODO_ROW_ID", "A"]]
        df_update["A"] = df_update["A"] * -10

        df = df.merge(df_update, on="_BODO_ROW_ID", how="left")
        df["A"] = df["A_y"].fillna(df["A_x"])

        df = df[["A", "B", "C", "D"]]
        bodo.io.iceberg.iceberg_merge_cow_py(
            table_name, conn, db_schema, df, old_snapshot_id, old_fnames
        )
        return old_snapshot_id

    old_snapshot_id = bodo.jit(impl1)(table_name, conn, db_schema)
    bodo.barrier()

    passed = False
    if bodo.get_rank() == 0:
        spark = get_spark()
        # We had issues with spark caching previously, this alleviates those issues
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

        snapshot_id_table = spark.sql(
            f"""
            select snapshot_id from hadoop_prod.{db_schema}.{table_name}.history order by made_current_at DESC
            """
        ).toPandas()

        # We expect to see two snapshot id's, the first from the creation/insertion of the initial values into
        # the table, and the second from when we do the merge into.
        passed = len(snapshot_id_table) == 2 and (
            old_snapshot_id == snapshot_id_table.iloc[1, 0]
        )

    passed = comm.bcast(passed)
    assert passed, "Snapshot ID's do not match expected output"

    # Construct Expected Output
    expected_out = orig_df.copy()
    expected_out.loc[expected_out.A > 4, ["A"]] = (
        expected_out[expected_out.A > 4]["A"] * -10
    )
    expected_out = expected_out.sort_values(by="C", ascending=True)

    passed = False
    err = None
    if bodo.get_rank() == 0:
        try:
            spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
            spark_out = spark_out.sort_values(by="C", ascending=True)
            spark_out["D"] = (
                spark_out["D"].astype("datetime64[ns]").dt.tz_localize("UTC")
            )

            passed = bool(
                _test_equal_guard(
                    spark_out,
                    expected_out,
                    sort_output=True,
                    check_dtype=False,
                    reset_index=True,
                )
            )
        except Exception as e:
            err = e

    passed, err = comm.bcast((passed, err))
    if isinstance(err, Exception):
        raise err
    assert passed, "Spark output doesn't match expected output"

    # Validate Bodo read output
    bodo_out = bodo.jit(all_returns_distributed=True)(
        lambda: pd.read_sql_table(table_name, conn, db_schema)
    )()  # type: ignore
    bodo_out = _gather_output(bodo_out)

    passed = False
    if bodo.get_rank() == 0:
        passed = bool(
            _test_equal_guard(
                expected_out,
                bodo_out,
                sort_output=True,
                check_dtype=False,
                reset_index=True,
            )
        )

    passed = comm.bcast(passed)
    assert passed, "Bodo read output doesn't match expected output"


@pytest.mark.slow
def test_iceberg_write_nulls_in_dict(iceberg_database, iceberg_table_conn):
    """
    Test that Iceberg write works even when there are nulls in
    the dictionary of a dictionary encoded array. Arrow doesn't
    support this, so we drop the NAs from all dictionaries ourselves
    before the write step.
    See [BE-4331](https://bodo.atlassian.net/browse/BE-4331)
    for more context.
    We also explicitly test the table-format case since the codegen
    for it is slightly different.
    """

    S = pa.DictionaryArray.from_arrays(
        np.array([0, 2, 1, 0, 1, 3, 0, 1, 3, 3, 2, 0], dtype=np.int32),
        pd.Series(["B", None, "A", None]),
    )
    A = np.arange(12)
    empty_df = pd.DataFrame({"S": [], "A": []})
    sql_schema = [("S", "string", True), ("A", "long", False)]
    spark = None
    if bodo.get_rank() == 0:
        spark = get_spark()

    db_schema, warehouse_loc = iceberg_database()

    @bodo.jit(distributed=["S", "A"])
    def impl(S, A, table_name, conn, db_schema, if_exists="replace"):
        S = pd.Series(S)
        df = pd.DataFrame({"S": S, "A": A})
        df.to_sql(table_name, conn, db_schema, if_exists=if_exists)
        return df

    @bodo.jit(distributed=["S", "A"])
    def impl_table_format(S, A, table_name, conn, db_schema, if_exists="replace"):
        S = pd.Series(S)
        df = pd.DataFrame({"S": S, "A": A})
        # Force dataframe to use table format
        df = bodo.hiframes.pd_dataframe_ext._tuple_to_table_format_decoded(df)
        df.to_sql(table_name, conn, db_schema, if_exists=if_exists)
        return df

    def test_output(impl, S, A, table_name, if_exists="replace"):
        conn = iceberg_table_conn(
            table_name, db_schema, warehouse_loc, check_exists=False
        )

        # Write the data
        impl(_get_dist_arg(S), _get_dist_arg(A), table_name, conn, db_schema, if_exists)
        bodo.barrier()

        # Read back
        exp_df = pd.DataFrame({"S": S, "A": A})
        bodo_read_out = bodo.jit()(
            lambda: pd.read_sql_table(table_name, conn, db_schema)
        )()
        bodo_read_out = _gather_output(bodo_read_out)
        bodo.barrier()

        # Check correctness
        passed = 1
        if bodo.get_rank() == 0:
            passed = _test_equal_guard(
                exp_df,
                bodo_read_out,
                sort_output=True,
                reset_index=True,
                check_dtype=False,
            )
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size()

    # Test regular write
    table_name = "TEST_ICEBERG_WRITE_NULLS_IN_DICT_TABLE"
    test_output(impl, S, A, table_name)
    test_output(impl_table_format, S, A, f"{table_name}_table_format")

    # Test append to a table with partition on S (dict-encoded column)
    table_name = "TEST_ICEBERG_WRITE_NULLS_IN_DICT_TABLE_PART_S"
    if bodo.get_rank() == 0:
        create_iceberg_table(
            empty_df,
            sql_schema,
            table_name,
            spark,
            [PartitionField("S", "truncate", 3)],
        )
    test_output(impl, S, A, table_name, "append")

    table_name = "TEST_ICEBERG_WRITE_NULLS_IN_DICT_TABLE_PART_S_TABLE_FORMAT"
    if bodo.get_rank() == 0:
        create_iceberg_table(
            empty_df,
            sql_schema,
            table_name,
            spark,
            [PartitionField("S", "truncate", 3)],
        )
    test_output(impl_table_format, S, A, table_name, "append")

    # Test append to a table with partition on A (non dict-encoded column)
    table_name = "TEST_ICEBERG_WRITE_NULLS_IN_DICT_TABLE_PART_A"
    if bodo.get_rank() == 0:
        create_iceberg_table(
            empty_df,
            sql_schema,
            table_name,
            spark,
            [PartitionField("A", "bucket", 4)],
        )
    test_output(impl, S, A, table_name, "append")

    table_name = "TEST_ICEBERG_WRITE_NULLS_IN_DICT_TABLE_PART_A_TABLE_FORMAT"
    if bodo.get_rank() == 0:
        create_iceberg_table(
            empty_df,
            sql_schema,
            table_name,
            spark,
            [PartitionField("A", "bucket", 4)],
        )
    test_output(impl_table_format, S, A, table_name, "append")

    # Use the same dict-encoded array for both columns and
    # partition on one of it (should catch any refcount bugs)
    sql_schema = [("S", "string", True), ("A", "string", True)]

    table_name = "TEST_ICEBERG_WRITE_NULLS_IN_DICT_TABLE_PART_SS"
    if bodo.get_rank() == 0:
        create_iceberg_table(
            empty_df,
            sql_schema,
            table_name,
            spark,
            [PartitionField("A", "bucket", 4)],
        )
    test_output(impl, S, S, table_name, "append")


def test_batched_read_agg(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Test a simple use of batched Iceberg reads by
    getting the max of a column
    """

    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def impl(table_name, conn, db_schema):
        total_max = pd.Timestamp(year=1970, month=1, day=1, tz="UTC")
        is_last_global = False
        reader = pd.read_sql_table(table_name, conn, db_schema, _bodo_chunksize=4096)  # type: ignore

        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(table), 1, None
            )
            df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (table,), index_var, col_meta
            )
            df = df[df["B"] > 10]
            # Perform more compute in between to see caching speedup
            local_max = df["A"].max()
            total_max = max(local_max, total_max)

            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )

        arrow_reader_del(reader)
        return total_max

    table_name = "SIMPLE_PRIMITIVES_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (table_name, conn, db_schema),
            py_output=pd.Timestamp(year=2024, month=5, day=14, tz="US/Eastern"),
        )
        check_logger_msg(stream, "Filter pushdown successfully performed")
        check_logger_msg(stream, "Columns loaded ['A']")


def test_batched_read_only_len(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Test shape pushdown with batched Snowflake reads
    """

    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "A",
            "B",
            "C",
            "D",
        )
    )

    def impl(table_name, conn, db_schema):
        total_len = 0
        is_last_global = False

        reader = pd.read_sql_table(table_name, conn, db_schema, _bodo_chunksize=4096)  # type: ignore
        while not is_last_global:
            table, is_last = read_arrow_next(reader, True)
            index_var = bodo.hiframes.pd_index_ext.init_range_index(
                0, len(table), 1, None
            )
            df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (table,), index_var, col_meta
            )
            total_len += len(df)

            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )

        arrow_reader_del(reader)
        return total_len

    table_name = "SIMPLE_PRIMITIVES_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (table_name, conn, db_schema), py_output=200)
        check_logger_msg(stream, "Columns loaded []")


def test_filter_pushdown_arg(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Test reading an Iceberg with the _bodo_filter flag
    """
    import bodo.ir.filter as bif

    table_name = "FILTER_PUSHDOWN_TEST_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(
            table_name,
            conn,
            db_schema,
            _bodo_filter=bif.make_op(
                ">", bif.make_ref("A"), bif.make_scalar(date(2015, 1, 1))
            ),
        )  # type: ignore
        return df

    py_out = _run_spark(
        f"""
        select * from hadoop_prod.{db_schema}.{table_name}
        where A > '2015-01-01';
        """
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (table_name, conn, db_schema),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nbic.FilterExpr('>', [bic.ColumnRef('A'), bic.Scalar(f0)])",
        )


def test_filter_pushdown_complex(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that complex filters are correctly pushed down
    and processed by the Iceberg connector. Filters tested are:
    - OR, AND, NOT
    - Starts with
    - In
    """
    table_name = "FILTER_PUSHDOWN_TEST_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[
            ((df["A"] > date(2015, 1, 1)) & ~(df["TY"].str.startswith("A")))
            | (df["B"].isin([0, 6]))
        ]
        return df

    py_out = _run_spark(
        f"""
    select * from hadoop_prod.{db_schema}.{table_name}
    where A > '2015-01-01' and not TY like 'A%' or B in (0, 6);
    """
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (table_name, conn, db_schema),
            py_output=py_out,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(
            stream,
            "Iceberg Filter Pushed Down:\nbic.FilterExpr('OR', [bic.FilterExpr('AND', [bic.FilterExpr('>', [bic.ColumnRef('A'), bic.Scalar(f0)]), bic.FilterExpr('NOT', [bic.FilterExpr('STARTS_WITH', [bic.ColumnRef('TY'), bic.Scalar(f1)])])]), bic.FilterExpr('IN', [bic.ColumnRef('B'), bic.Scalar(f2)])])",
        )
