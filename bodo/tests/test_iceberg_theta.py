import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.io.stream_iceberg_write import (
    iceberg_writer_append_table,
    iceberg_writer_fetch_theta,
    iceberg_writer_init,
    read_puffin_file_ndvs,
)
from bodo.ir.sql_ext import remove_iceberg_prefix
from bodo.tests.iceberg_database_helpers.metadata_utils import (
    get_metadata_field,
    get_metadata_path,
)
from bodo.tests.utils import _get_dist_arg, check_func, temp_env_override
from bodo.utils.utils import run_rank0

pytestmark = pytest.mark.iceberg


def create_iceberg_table_with_puffin_files(df, table_name, conn, db_schema):
    """Helper to create an Iceberg table with puffin files."""
    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    batch_size = 5

    def impl(df, table_name, conn, db_schema):
        # Create the writer, including the theta sketch
        writer = iceberg_writer_init(
            -1, conn, table_name, db_schema, col_meta, "replace"
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

    return impl


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


@run_rank0
def check_ndv_metadata(warehouse_loc, db_schema, table_name, expected_ndvs):
    """
    Check the NDV information found in the metadata file and return the path to the
    puffin file for further testing.
    """
    metadata_path = get_metadata_path(warehouse_loc, db_schema, table_name)
    statistics_lst = get_metadata_field(metadata_path, "statistics")
    assert len(statistics_lst) == 1, "Expected only one statistics file"
    statistics = statistics_lst[0]
    # Check the NDVs match expectations
    blob_metadata = statistics["blob-metadata"]
    seen_fields = set()
    for blob in blob_metadata:
        fields = blob["fields"]
        assert len(fields) == 1, "Expected only one field in the puffin file"
        field = fields[0]
        properties = blob["properties"]
        ndv = properties["ndv"]
        assert field in expected_ndvs, "Unexpected field ID blob"
        assert ndv == expected_ndvs[field], f"Incorrect NDV for blob {field}"
        seen_fields.add(field)
    assert len(seen_fields) == len(
        expected_ndvs
    ), "An expected column didn't have a theta sketch"
    # Check the puffin file exists, can be read, and the theta sketch is correct.
    return statistics["statistics-path"]


@bodo.jit
def get_statistics_ndvs(puffin_file_name, iceberg_schema):
    return read_puffin_file_ndvs(puffin_file_name, iceberg_schema)


@run_rank0
def get_iceberg_pyarrow_schema(conn, db_schema, table_name):
    import bodo_iceberg_connector

    conn = remove_iceberg_prefix(conn)
    _, _, pyarrow_schema = bodo_iceberg_connector.get_iceberg_typing_schema(
        conn, db_schema, table_name
    )
    return pyarrow_schema


def test_full_iceberg_theta_write(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test the result of doing a full end to end CTAS operation that
    generates theta sketches for several types. This code should check:

    1. The theta sketches are available in the metadata file.
    2. The columns in the statistics file match our expected columns.
    3. Each column has an ndv property.
    4. The puffin file exists at the provided location.
    5. We can read the puffin file and deserialize the theta sketch.
    6. The NDV and the expectation resulting from the theta sketch are consistent
       with some rounding (I believe the NDV is usually an integer).
    7. The NDV matches a hardcoded expected value to avoid regressions. This is feasible
        because the result should be deterministic.
    """
    df = pd.DataFrame(
        {
            "A": pd.array(list(range(10)) * 2, dtype=pd.Int64Dtype()),
            # B shouldn't generate theta sketches right now because we disable
            # float by default.
            "B": pd.array([1.4, 1.5, 2.451, 0] * 5, dtype=pd.Float64Dtype()),
            "C": pd.array(["a", "ab", "cde", "af", "eg"] * 4, dtype=object),
            # We don't generate theta sketches for boolean types right now
            # because the max NDV is 2.
            "D": pd.array([True, False, None, False] * 5, dtype="boolean"),
            "E": pd.array(
                [datetime.date(2021, 1, 1)] * 20, dtype=pd.ArrowDtype(pa.date32())
            ),
        }
    )
    # These are hardcoded field IDs based upon how IDs are assigned starting from 1.
    ndvs = {1: "10", 3: "5", 5: "1"}
    ndvs_array = pd.array([10, None, 5, None, 1], dtype=pd.Float64Dtype())
    table_name = "basic_puffin_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    with temp_env_override({"BODO_ENABLE_THETA_SKETCHES": "1"}):
        f = create_iceberg_table_with_puffin_files(df, table_name, conn, db_schema)
        df = _get_dist_arg(df, var_length=True)
        bodo.jit(distributed=["df"])(f)(df, table_name, conn, db_schema)
        puffin_file_name = check_ndv_metadata(
            warehouse_loc, db_schema, table_name, ndvs
        )
        pyarrow_schema = get_iceberg_pyarrow_schema(conn, db_schema, table_name)
        ndvs = get_statistics_ndvs(puffin_file_name, pyarrow_schema)
        pd.testing.assert_extension_array_equal(ndvs, ndvs_array)
