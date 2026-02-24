from __future__ import annotations

import io
import time
from datetime import date

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyspark.sql.types as spark_types
import pytest
from mpi4py import MPI
from numba.core import types  # noqa TID253

import bodo
from bodo.tests.iceberg_database_helpers import pyiceberg_reader, spark_reader
from bodo.tests.iceberg_database_helpers.utils import (
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
    check_func,
    sync_dtypes,
)

pytestmark = pytest.mark.iceberg


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

    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
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

    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
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

    py_out = pyiceberg_reader.read_iceberg_table_single_rank(table_name, db_schema)
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
    from bodo.utils.typing import BodoError

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
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next
    from bodo.spawn.utils import run_rank0

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
    from bodo.tests.utils_jit import ColumnDelTestPipeline, reduce_sum

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
    assert py_out.shape[0] == 0, (
        f"Expected DataFrame to be empty, found {py_out.shape[0]} rows instead."
    )

    check_func(impl, (table_name, conn, db_schema), py_output=py_out)


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
        with numba.objmode(sort_list=file_list_type):
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
            "Iceberg Filter Pushed Down:\npie.EqualTo('B', literal(f0))",
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
    from bodo.tests.utils_jit import DistTestPipeline

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
def test_iceberg_invalid_table(iceberg_database, iceberg_table_conn):
    """Tests error raised when a nonexistent Iceberg table is provided."""
    from bodo.utils.typing import BodoError

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
    from bodo.utils.typing import BodoError

    table_name = "FILTER_PUSHDOWN_TEST_TABLE"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    db_schema += "not"

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df["A"].sum()

    with pytest.raises(BodoError, match="No such Iceberg table found"):
        bodo.jit(impl)(table_name, conn, db_schema)


def test_batched_read_agg(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Test a simple use of batched Iceberg reads by
    getting the max of a column
    """
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next

    col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B", "C", "D"))

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
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next

    col_meta = bodo.utils.typing.ColNamesMetaType(("A", "B", "C", "D"))

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
            "Iceberg Filter Pushed Down:\npie.GreaterThan('A', literal(f0))",
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
            "Iceberg Filter Pushed Down:\npie.Or(pie.And(pie.GreaterThan('A', literal(f0)), pie.Not(pie.StartsWith('TY', literal(f1)))), pie.In('B', literal(f2)))",
        )


def test_time_travel_snapshot_id(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    from bodo.io.iceberg.catalog import conn_str_to_catalog

    table_name = "TIME_TRAVEL_SNAPSHOT_ID"

    db_schema, warehouse_loc = iceberg_database(table_name)
    spark = get_spark()

    # Create a table and append to it
    snap_one = pd.DataFrame(
        {
            "A": np.arange(1000),
        }
    )
    sql_schema = [
        ("A", "int", True),
    ]
    if bodo.get_rank() == 0:
        create_iceberg_table(snap_one, sql_schema, table_name, spark)
    bodo.barrier()
    # Get the snapshot id to read
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    catalog = conn_str_to_catalog(conn)
    table = catalog.load_table(f"{db_schema}.{table_name}")
    snapshot_id = table.current_snapshot().snapshot_id

    # Add a new column to make sure read works if the schema changes
    # between the latest and the snapshot being read
    if bodo.get_rank() == 0:
        spark.sql(
            f"ALTER TABLE hadoop_prod.{db_schema}.{table_name} ADD COLUMN B string"
        )
        snap_two = pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": ["awe", "awv2"] * 500,
            }
        )
        spark_schema = spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.IntegerType(), True),
                spark_types.StructField("B", spark_types.StringType(), False),
            ]
        )
        sdf = spark.createDataFrame(snap_two, schema=spark_schema)
        sdf.writeTo(f"hadoop_prod.{db_schema}.{table_name}").append()

    def impl(table_name, conn, db_schema, snapshot_id):
        return pd.read_sql_table(table_name, conn, db_schema, _snapshot_id=snapshot_id)

    check_func(
        impl,
        (table_name, conn, db_schema, snapshot_id),
        py_output=snap_one,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_time_travel_timestamp(iceberg_database, iceberg_table_conn, memory_leak_check):
    table_name = "TIME_TRAVEL_TIMESTAMP"

    db_schema, warehouse_loc = iceberg_database(table_name)
    spark = get_spark()

    # Create a table and append to it
    snap_one = pd.DataFrame(
        {
            "A": np.arange(1000),
        }
    )
    sql_schema = [
        ("A", "int", True),
    ]
    if bodo.get_rank() == 0:
        create_iceberg_table(snap_one, sql_schema, table_name, spark)
    bodo.barrier()
    # Get the timestamp to read
    timestamp_ms = int(time.time() * 1000)
    bodo.barrier()

    # Add a new column to make sure read works if the schema changes
    # between the latest and the snapshot being read
    if bodo.get_rank() == 0:
        spark.sql(
            f"ALTER TABLE hadoop_prod.{db_schema}.{table_name} ADD COLUMN B string"
        )
        snap_two = pd.DataFrame(
            {
                "A": np.arange(1000),
                "B": ["awe", "awv2"] * 500,
            }
        )
        spark_schema = spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.IntegerType(), True),
                spark_types.StructField("B", spark_types.StringType(), False),
            ]
        )
        sdf = spark.createDataFrame(snap_two, schema=spark_schema)
        sdf.writeTo(f"hadoop_prod.{db_schema}.{table_name}").append()

    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema, timestamp_ms):
        return pd.read_sql_table(
            table_name, conn, db_schema, _snapshot_timestamp_ms=timestamp_ms
        )

    check_func(
        impl,
        (table_name, conn, db_schema, timestamp_ms),
        py_output=snap_one,
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
