import glob
import os
import traceback

import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.io.stream_iceberg_write import (
    iceberg_writer_append_table,
    iceberg_writer_init,
)
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
from bodo.tests.test_iceberg import (
    ICEBERG_FIELD_IDS_IN_PQ_SCHEMA_TEST_PARAMS,
    _setup_test_iceberg_field_ids_in_pq_schema,
    _test_file_part,
    _test_file_sorted,
    _verify_pq_schema_in_files,
    simple_dataframe,  # noqa: F401
)
from bodo.tests.utils import (
    _gather_output,
    _get_dist_arg,
    _test_equal_guard,
    convert_non_pandas_columns,
    pytest_mark_one_rank,
    reduce_sum,
)
from bodo.utils.utils import run_rank0

pytestmark = pytest.mark.iceberg


def _write_iceberg_table(df, table_name, conn, db_schema, if_exists):
    """helper that writes Iceberg table using Bodo's streaming write"""

    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    batch_size = 11

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        writer = iceberg_writer_init(
            -1, conn, table_name, db_schema, col_meta, if_exists
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

    impl(_get_dist_arg(df), table_name, conn, db_schema)


@pytest.mark.timeout(1000)
def test_iceberg_write_basic(
    iceberg_database, iceberg_table_conn, simple_dataframe, memory_leak_check
):
    """Test basic streaming Iceberg write"""
    base_name, table_name, df = simple_dataframe
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    orig_chunk_size = bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    if base_name == "DICT_ENCODED_STRING_TABLE":
        bodo.hiframes.boxing._use_dict_str_type = True
    # set chunk size to a small number to make sure multiple iterations write files
    bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300

    try:
        _write_iceberg_table(df, table_name, conn, db_schema, "replace")
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
        bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size

    py_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
    py_out = _gather_output(py_out)

    comm = MPI.COMM_WORLD
    passed = None
    if comm.Get_rank() == 0:
        if "LIST" in table_name or "STRUCT" in table_name:
            df = convert_non_pandas_columns(df)
            py_out = convert_non_pandas_columns(py_out)
        passed = _test_equal_guard(
            df,
            py_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    passed = comm.bcast(passed)
    assert passed == 1


@pytest.mark.parametrize("use_dict_encoding_boxing", [False, True])
def test_write_part_sort(
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
    table_name = f"PARTSORT_{PART_SORT_TABLE_BASE_NAME}_streaming"
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

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    bodo.hiframes.boxing._use_dict_str_type = use_dict_encoding_boxing
    orig_chunk_size = bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300

    try:
        _write_iceberg_table(df, table_name, conn, db_schema, "append")
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
        bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size
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
            expected_df,
            bodo_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo read output doesn't match expected output"


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

    table_name = f"SIMPLE_{base_name}_pq_schema_test_{mode}_streaming"
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

    # Write using Bodo
    orig_chunk_size = bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    # set chunk size to a small number to make sure multiple iterations write files
    bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 1 * 1024  # (1KiB)

    try:
        _write_iceberg_table(df, table_name, conn, db_schema, if_exists)
    finally:
        bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size

    bodo.barrier()

    _verify_pq_schema_in_files(
        warehouse_loc, db_schema, table_name, data_files_before_write, expected_schema
    )


@run_rank0
def _get_pq_files(warehouse_loc, db_schema, table_name):
    data_files = glob.glob(
        os.path.join(warehouse_loc, db_schema, table_name, "data", "*.parquet")
    )
    return data_files


@pytest_mark_one_rank
@pytest.mark.parametrize("max_pq_chunksize", [6400000, 3200000, 1600000, 800000])
def test_iceberg_max_pq_chunksize(
    max_pq_chunksize, iceberg_database, iceberg_table_conn
):
    """
    Test that number of raw bytes written to each parquet file are less than "write.target-file-size-bytes"
    and that number of files generated is also consistent with the threshold
    """
    import math

    import numpy as np

    table_name = "SIMPLE_INT_TABLE_test_pq_chunksize"

    db_schema, warehouse_loc = iceberg_database([table_name])
    some_rows, sql_schema = (
        {
            "A": np.array([0]),
            "B": np.array([400000]),
        },
        [
            ("A", "long", True),
            ("B", "long", True),
        ],
    )

    # generate a large number of random data
    np.random.seed(42)
    large_number_of_rows = {
        "A": np.arange(1, 400001),
        "B": np.random.randint(1, 400000, size=400000),
    }

    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    spark = get_spark()

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

    _write_iceberg_table(df2, table_name, conn, db_schema, if_exists="append")
    bodo.barrier()
    files_after_write = _get_pq_files(warehouse_loc, db_schema, table_name)

    @run_rank0
    def check_files():
        expected_num_files = math.ceil(total_write_size / (max_pq_chunksize))
        new_files = list(set(files_after_write) - set(files_before_write))
        assert (
            len(new_files) == expected_num_files
        ), "Expected number of files does not match"
        for file in new_files:
            size_in_bytes = os.stat(file).st_size
            # actual size of files may be much smaller after compression
            assert (
                size_in_bytes < max_pq_chunksize
            ), "Found file larger than expected size"

    check_files()
