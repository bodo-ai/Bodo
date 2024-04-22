import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.io.stream_iceberg_write import (
    iceberg_writer_append_table,
    iceberg_writer_fetch_theta,
    iceberg_writer_init,
)
from bodo.tests.utils import check_func, temp_env_override

pytestmark = pytest.mark.iceberg


def write_iceberg_table_check_theta(
    df, table_name, conn, db_schema, if_exists, expected_theta
):
    """helper that writes Iceberg table using Bodo's streaming write and checks for theta values"""

    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    batch_size = 11

    def impl(df, table_name, conn, db_schema, if_exists):
        # Create the writer, including the theta sketch
        writer = iceberg_writer_init(
            -1, conn, table_name, db_schema, col_meta, if_exists
        )
        # Write the values from the df using the writer in batches
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
        # Fetch the theta sketch values
        result = iceberg_writer_fetch_theta(writer)
        return result

    check_func(
        impl,
        (df, table_name, conn, db_schema, if_exists),
        py_output=expected_theta,
        check_names=False,
        check_dtype=False,
        reset_index=True,
        is_out_distributed=False,
        use_dict_encoded_strings=True,
        only_1DVar=True,
        rtol=0.05,
    )


def test_iceberg_write_theta_estimates(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """Test basic streaming Iceberg write with theta sketches enabled to ensure
    that they are generated for 4/5 columns."""
    table_name = "iceberg_ctas_theta_test_table_1"
    # A: 10000 unique values, valid type (integer)
    # B: 256 unique values, valid type (integer)
    # C: 101 unique values, invalid type (float)
    # D: 1000 unique values, valid type (string)
    # E: 4365 unique values, valid type (date)
    df = pd.DataFrame(
        {
            "A": pd.array(np.arange(10000), dtype=pd.Int64Dtype()),
            "B": pd.array(np.arange(10000) % 256, dtype=pd.Int32Dtype()),
            "C": pd.array(np.round(np.arange(10000) ** 0.5)) ** 0.5,
            "D": [str(i)[:3] for i in range(10000)],
            "E": [
                datetime.date.fromordinal(730000 + int(int(i**0.91) ** 1.1))
                for i in range(10000)
            ],
        }
    )
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    orig_chunk_size = bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    bodo.hiframes.boxing._use_dict_str_type = True
    bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300
    try:
        with temp_env_override({"BODO_ENABLE_THETA_SKETCHES": "1"}):
            expected_theta = pd.array(
                [10000, 256, None, 1000, 4365], dtype=pd.Float64Dtype()
            )
            write_iceberg_table_check_theta(
                df, table_name, conn, db_schema, "replace", expected_theta
            )
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
        bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size


def test_iceberg_write_disabled_theta(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """Same as test_iceberg_write_theta_estimates but where theta sketches are disabled for
    all columns"""
    table_name = "iceberg_ctas_theta_test_table_2"
    df = pd.DataFrame(
        {
            "A": pd.array(np.arange(10000), dtype=pd.Int64Dtype()),
            "B": pd.array(np.arange(10000) % 256, dtype=pd.Int32Dtype()),
            "C": pd.array(np.round(np.arange(10000) ** 0.5)) ** 0.5,
            "D": [str(i)[:3] for i in range(10000)],
            "E": [
                datetime.date.fromordinal(730000 + int(int(i**0.91) ** 1.1))
                for i in range(10000)
            ],
        }
    )
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    orig_chunk_size = bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    bodo.hiframes.boxing._use_dict_str_type = True
    bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300
    try:
        with temp_env_override({"BODO_ENABLE_THETA_SKETCHES": "0"}):
            expected_theta = pd.array([None] * 5, dtype=pd.Float64Dtype())
            write_iceberg_table_check_theta(
                df, table_name, conn, db_schema, "replace", expected_theta
            )
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
        bodo.io.stream_iceberg_write.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size
