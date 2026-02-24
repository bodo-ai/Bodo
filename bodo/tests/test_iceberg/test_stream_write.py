from __future__ import annotations

import glob
import math
import os
import traceback
import typing as pt

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.iceberg_database_helpers.part_sort_table import (
    BASE_NAME as PART_SORT_TABLE_BASE_NAME,
)
from bodo.tests.iceberg_database_helpers.part_sort_table import (
    PARTITION_SPEC as PART_SORT_TABLE_PARTITION_SPEC,
)
from bodo.tests.iceberg_database_helpers.part_sort_table import (
    SORT_ORDER as PART_SORT_TABLE_SORT_ORDER,
)
from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLES_MAP,
)
from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)
from bodo.tests.test_iceberg.test_iceberg_write import (
    ICEBERG_FIELD_IDS_IN_PQ_SCHEMA_TEST_PARAMS,
    _setup_test_iceberg_field_ids_in_pq_schema,
    _test_file_part,
    _test_file_sorted,
    _verify_pq_schema_in_files,
)
from bodo.tests.utils import (
    _gather_output,
    _get_dist_arg,
    _test_equal_guard,
    convert_non_pandas_columns,
    pytest_mark_one_rank,
)

pytestmark = pytest.mark.iceberg


def _write_iceberg_table(
    df,
    table_id: str,
    conn: str,
    create_table_meta,
    mode: pt.Literal["create", "replace", "append"],
    parallel: bool = True,
):
    """helper that writes Iceberg table using Bodo's streaming write"""
    from bodo.io.iceberg.stream_iceberg_write import (
        iceberg_writer_append_table,
        iceberg_writer_init,
    )

    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    batch_size = 11

    @bodo.jit(distributed=(["df"] if parallel else False))
    def impl(df, table_id, conn):
        writer = iceberg_writer_init(
            -1,
            conn,
            table_id,
            col_meta,
            mode,
            create_table_meta=create_table_meta,
        )
        all_is_last = False
        iter_val = 0
        T1 = bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df)
        while not all_is_last:
            table = bodo.hiframes.table.table_local_filter(
                T1,
                slice(
                    (iter_val * batch_size),
                    ((iter_val + 1) * batch_size),
                ),
            )
            is_last = (iter_val * batch_size) >= len(df)
            all_is_last = iceberg_writer_append_table(
                writer, table, col_meta, is_last, iter_val
            )
            iter_val += 1

    impl((_get_dist_arg(df) if parallel else df), table_id, conn)


def init_create_table_meta(
    table_comments: str | None = None, column_comments=False, table_property=False
):
    """Helper function to initialize a CreateTableMetaType type.

    Args:
        table_name (_type_, optional): Table name required for generating table comments. Need to NotNull when table_comments is True Defaults to None.
        table_comments (bool, optional): Whether the test case will test table comments. Defaults to False.
        column_comments (bool, optional): Whether the test case will test column comments. Defaults to False.

    Returns:
        _type_: _description_
    """
    column_comment = (
        None
        if column_comments is None
        else tuple(f"{column_comments}{_}" for _ in range(2))
    )
    table_properties = (
        None
        if not table_property
        else (("TBLPROPERTIES", "A"), ("TEST_TBL", "T"), ("TEST_TBL", "TRUE"))
    )
    return bodo.utils.typing.CreateTableMetaType(
        table_comment=table_comments,
        column_comments=column_comment,
        table_properties=table_properties,
    )


def parse_table_property(s):
    """Given property column from "DESCRIBE TABLE EXTENDED", parse it into dict
        There might be builtin functions to extract proerties, but currently
        we implement our own version.

        Input string is in format of "['Foo'='A','Bar'='B, C, D']".
        Since "," can appear in values or keys, we can't simply use s.split(",") and
        then s.split('=').
        But a pattern we observe is when serving as key-value separators there are no
        spaces around ",". So we customize this split(",") to only split on "," with
        no subsequent space.

    Args:
        s (string): Input table property string

    Returns:
        dict: A dictionary with key-value pairs of table properties in string
    """

    i = 1
    properties = {}
    while i < len(s) - 1:
        j = i + 1
        while j < len(s) - 1 and (s[j] != "," or s[j + 1] == " "):
            j += 1
        properties[s[i:j].split("=")[0]] = s[i:j].split("=")[1]
        i = j + 1
    return properties


def check_table_comment(
    db_schema,
    table_name,
    number_columns,
    table_comments: str | None = None,
    column_comments=False,
    table_properties: dict | None = None,
):
    """Helper function to test table comments are correctly added

    Args:
        db_schema (_type_): Databse schema
        table_name (_type_): Table name
        number_columns (int): Number of columns of the table
        table_comments (bool, optional): Whether the test case will test table comments. Defaults to False.
        column_comments (bool, optional): Whether the test case will test column comments. Defaults to False.
    """
    spark = get_spark()
    spark.sql(f"REFRESH TABLE hadoop_prod.{db_schema}.{table_name}")
    table_cmt = (
        spark.sql(f"DESCRIBE TABLE EXTENDED hadoop_prod.{db_schema}.{table_name}")
        .filter("col_name = 'Comment'")
        .select("data_type")
        .head()
    )
    if table_comments is not None:
        assert table_cmt and table_cmt[0] == table_comments, (
            f'Expected table comment to be "{table_comments}", got "{table_cmt}"'
        )

    df = spark.sql(f"DESCRIBE TABLE hadoop_prod.{db_schema}.{table_name}").toPandas()
    for i in range(number_columns):
        if column_comments is None or i >= 2:
            assert pd.isna(df.iloc[i]["comment"]), (
                f"Expected column {i} comment to be None, but actual comment is {df.iloc[i]['comment']}"
            )
        else:
            assert df.iloc[i]["comment"] == f"{column_comments}{i}", (
                f'Expected column {i} comment to be "{column_comments}{i}", got {df.iloc[i]["comment"]}'
            )

    if table_properties is not None:
        str = (
            spark.sql(f"DESCRIBE TABLE EXTENDED hadoop_prod.{db_schema}.{table_name}")
            .filter("col_name = 'Table Properties'")
            .select("data_type")
            .head()[0]
        )

        parsed_properties = parse_table_property(str)

        for keys in table_properties:
            assert keys in parsed_properties, (
                f"Expected table properties {keys}, find nothing"
            )
            assert parsed_properties[keys] == table_properties[keys], (
                f"Expected key {keys} with value {table_properties[keys]}, got {parsed_properties[keys]}"
            )


def test_iceberg_write_basic(
    iceberg_database,
    iceberg_table_conn,
    simple_dataframe,
    memory_leak_check,
):
    """Test basic streaming Iceberg write"""
    import bodo.io.iceberg.stream_iceberg_write

    base_name, table_name, df = simple_dataframe
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    orig_chunk_size = bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    if base_name == "DICT_ENCODED_STRING_TABLE":
        bodo.hiframes.boxing._use_dict_str_type = True
    # set chunk size to a small number to make sure multiple iterations write files
    bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300

    create_table_meta = init_create_table_meta()

    try:
        _write_iceberg_table(df, table_id, conn, create_table_meta, "replace")
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
        bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size

    py_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
    py_out = _gather_output(py_out)

    comm = MPI.COMM_WORLD
    passed = None

    if comm.Get_rank() == 0:
        if "LIST" in table_name or "STRUCT" in table_name:
            df = convert_non_pandas_columns(df)
            py_out = convert_non_pandas_columns(py_out)
        passed = _test_equal_guard(
            py_out,
            df,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1


@pytest.mark.parametrize("column_comments", ["test_col_comments", None])
@pytest.mark.parametrize("table_comments", ["test_tbl_comments", "", None])
@pytest.mark.parametrize("table_properties", [True, False])
def test_iceberg_write_with_comment_and_properties(
    table_comments,
    column_comments,
    table_properties,
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    """Test basic streaming Iceberg write"""
    import bodo.io.iceberg.stream_iceberg_write

    table_name = "SIMPLE_STRING_TABLE"
    df = SIMPLE_TABLES_MAP["SIMPLE_STRING_TABLE"][0]
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"

    orig_chunk_size = bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    # set chunk size to a small number to make sure multiple iterations write files
    bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300

    create_table_meta = init_create_table_meta(
        table_comments=table_comments,
        column_comments=column_comments,
        table_property=table_properties,
    )

    try:
        _write_iceberg_table(df, table_id, conn, create_table_meta, "replace")
    finally:
        bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size

    py_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
    py_out = _gather_output(py_out)

    comm = MPI.COMM_WORLD
    passed = None

    if comm.Get_rank() == 0:
        if "LIST" in table_name or "STRUCT" in table_name:
            df = convert_non_pandas_columns(df)
            py_out = convert_non_pandas_columns(py_out)
        passed = _test_equal_guard(
            py_out,
            df,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
        ref_table_properties = (
            None if not table_properties else {"TBLPROPERTIES": "A", "TEST_TBL": "TRUE"}
        )
        check_table_comment(
            db_schema,
            table_name,
            len(py_out.columns),
            table_comments=table_comments,
            column_comments=column_comments,
            table_properties=ref_table_properties,
        )
    passed = comm.bcast(passed)
    assert passed == 1


def test_iceberg_write_basic_rep(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    """Test basic streaming Iceberg write with replicated input"""
    import bodo.io.iceberg.stream_iceberg_write

    table_name = "SIMPLE_STRING_TABLE"
    df = SIMPLE_TABLES_MAP["SIMPLE_STRING_TABLE"][0]
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"

    # set chunk size to a small number to make sure multiple iterations write files
    bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300

    create_table_meta = init_create_table_meta()

    _write_iceberg_table(
        df, table_id, conn, create_table_meta, "replace", parallel=False
    )

    py_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
    py_out = _gather_output(py_out)

    comm = MPI.COMM_WORLD
    passed = None

    if comm.Get_rank() == 0:
        passed = _test_equal_guard(
            py_out,
            df,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1


@pytest.mark.parametrize("use_dict_encoding_boxing", [False, True])
def test_iceberg_write_part_sort(
    iceberg_database,
    iceberg_table_conn,
    use_dict_encoding_boxing,
    memory_leak_check,
):
    """
    Append to a table with both a partition spec and a sort order,
    and verify that the append was done correctly, i.e. validate
    that each file is correctly sorted and partitioned.
    """
    import bodo.io.iceberg.stream_iceberg_write
    from bodo.tests.utils_jit import reduce_sum

    table_name = (
        f"PARTSORT_{PART_SORT_TABLE_BASE_NAME}_streaming_{use_dict_encoding_boxing}"
    )
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
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=True)
    table_id = f"{db_schema}.{table_name}"

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    bodo.hiframes.boxing._use_dict_str_type = use_dict_encoding_boxing
    orig_chunk_size = bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300

    try:
        _write_iceberg_table(df, table_id, conn, None, "append")
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
        bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size
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

    # Read the table back and validate that the
    # contents are as expected
    expected_df = pd.concat([df, df]).reset_index(drop=True)

    bodo_out = bodo.jit(distributed=["df"])(
        lambda: pd.read_sql_table(table_name, conn, db_schema)
    )()  # type: ignore
    bodo_out = _gather_output(bodo_out)

    comm = MPI.COMM_WORLD
    passed = None

    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodo_out,
            expected_df,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo read output doesn't match expected output"


@pytest.mark.parametrize(
    "base_name,pa_schema", ICEBERG_FIELD_IDS_IN_PQ_SCHEMA_TEST_PARAMS
)
@pytest.mark.parametrize("mode", ["create", "append"])
# TODO: Replace is removed due to inconsistent field ID assignment between PyIceberg
# and Iceberg Java
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
    import bodo.io.iceberg.stream_iceberg_write

    table_name = f"SIMPLE_{base_name}_pq_schema_test_{mode}_streaming"
    db_schema, warehouse_loc = iceberg_database()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    df, sql_schema = SIMPLE_TABLES_MAP[f"SIMPLE_{base_name}"]
    table_id = f"{db_schema}.{table_name}"

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

    # Write using Bodo
    orig_chunk_size = bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    # set chunk size to a small number to make sure multiple iterations write files
    bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 1 * 1024  # (1KiB)

    try:
        _write_iceberg_table(df, table_id, conn, None, mode)
    finally:
        bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size

    bodo.barrier()

    _verify_pq_schema_in_files(
        warehouse_loc, db_schema, table_name, data_files_before_write, expected_schema
    )


@pytest_mark_one_rank
@pytest.mark.parametrize("max_pq_chunksize", [6400000, 3200000, 1600000, 800000])
def test_iceberg_max_pq_chunksize(
    max_pq_chunksize, iceberg_database, iceberg_table_conn
):
    """
    Test that number of raw bytes written to each parquet file are less than "write.target-file-size-bytes"
    and that number of files generated is also consistent with the threshold
    """
    from bodo.spawn.utils import run_rank0

    table_name = "SIMPLE_INT_TABLE_test_pq_chunksize"

    db_schema, warehouse_loc = iceberg_database([table_name])
    some_rows, sql_schema = (
        {
            "A": np.array([0], dtype=np.int64),
            "B": np.array([400000], dtype=np.int64),
        },
        [
            ("A", "long", True),
            ("B", "long", True),
        ],
    )

    # generate a large number of random data
    np.random.seed(42)
    large_number_of_rows = {
        "A": np.arange(1, 400001, dtype=np.int64),
        "B": np.random.randint(1, 400000, size=400000, dtype=np.int64),
    }

    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"
    spark = get_spark()

    @run_rank0
    def _get_pq_files(warehouse_loc, db_schema, table_name):
        data_files = glob.glob(
            os.path.join(warehouse_loc, db_schema, table_name, "data", "*.parquet")
        )
        return data_files

    @run_rank0
    def setup():
        df1 = pd.DataFrame(some_rows)
        create_iceberg_table(df1, sql_schema, table_name, spark)
        spark.sql(
            f"ALTER TABLE hadoop_prod.iceberg_db.{table_name} SET TBLPROPERTIES ('write.target-file-size-bytes'='{max_pq_chunksize}')"
        )

    setup()

    files_before_write = _get_pq_files(warehouse_loc, db_schema, table_name)
    df2 = pd.DataFrame(large_number_of_rows)
    total_write_size = sum(df2.memory_usage(index=False))  # = 800000 * 8 bytes
    bodo.barrier()

    _write_iceberg_table(df2, table_id, conn, None, mode="append")
    bodo.barrier()
    files_after_write = _get_pq_files(warehouse_loc, db_schema, table_name)

    @run_rank0
    def check_files():
        expected_num_files = math.ceil(total_write_size / (max_pq_chunksize))
        new_files = list(set(files_after_write) - set(files_before_write))
        assert len(new_files) == expected_num_files, (
            "Expected number of files does not match"
        )
        for file in new_files:
            size_in_bytes = os.stat(file).st_size
            # actual size of files may be much smaller after compression
            assert size_in_bytes < (max_pq_chunksize * 1.25), (
                "Found file larger than expected size"
            )

    check_files()
