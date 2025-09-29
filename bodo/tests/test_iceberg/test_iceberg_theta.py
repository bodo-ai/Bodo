import datetime

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)
from bodo.tests.utils import _get_dist_arg

pytestmark = [pytest.mark.jit_dependency, pytest.mark.iceberg]


def write_iceberg_table_with_puffin_files(df, table_id, conn, write_type):
    """Helper to create an Iceberg table with puffin files."""
    from bodo.io.iceberg.stream_iceberg_write import (
        iceberg_writer_append_table,
        iceberg_writer_init,
    )

    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    batch_size = 5

    def impl(df, table_id, conn, write_type):
        # Create the writer, including the theta sketch
        writer = iceberg_writer_init(
            -1,
            conn,
            table_id,
            col_meta,
            write_type,
            allow_theta_sketches=True,
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


def test_iceberg_write_theta_estimates(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """Test basic streaming Iceberg write with theta sketches enabled to ensure
    that they are generated for 4/5 columns. This"""
    import bodo.io.iceberg.stream_iceberg_write
    from bodo.tests.test_iceberg.utils_jit import (
        check_ndv_metadata,
        get_statistics_ndvs,
    )

    table_name = "iceberg_ctas_theta_test_table_1"
    # A: 10000 unique values, valid type (integer)
    # B: 256 unique values, valid type (integer)
    # C: 101 unique values, invalid type (float)
    # D: 1000 unique values, valid type (string)
    # E: 4365 unique values, valid type (date)
    # The ndv estimates won't match these values exactly, but they should be close.
    # We hardcode the expected values below.
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
    ndvs = {1: "9897", 2: "256", 4: "1000", 5: "4366"}
    ndvs_array = pd.array(
        [9897.762845698651, 256.0, None, 1000.0, 4366.14457972729],
        dtype=pd.Float64Dtype(),
    )
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    orig_chunk_size = bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    orig_enable_theta = bodo.enable_theta_sketches
    bodo.hiframes.boxing._use_dict_str_type = True
    bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300
    bodo.enable_theta_sketches = True
    try:
        f = write_iceberg_table_with_puffin_files(df, table_id, conn, "replace")
        df = _get_dist_arg(df, var_length=True)
        bodo.jit(distributed=["df"])(f)(df, table_id, conn, "replace")
        puffin_file_name = check_ndv_metadata(
            warehouse_loc, db_schema, table_name, ndvs
        )
        pyarrow_schema = get_iceberg_pyarrow_schema(conn, table_id)
        ndvs = get_statistics_ndvs(puffin_file_name, pyarrow_schema)
        pd.testing.assert_extension_array_equal(ndvs, ndvs_array, check_dtype=False)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
        bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size
        bodo.enable_theta_sketches = orig_enable_theta


def test_iceberg_write_disabled_theta(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """Same as test_iceberg_write_theta_estimates but where theta sketches are disabled for
    all columns"""
    import bodo.io.iceberg.stream_iceberg_write
    from bodo.tests.test_iceberg.utils_jit import check_no_statistics_file

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
    table_id = f"{db_schema}.{table_name}"

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    orig_chunk_size = bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE
    orig_enable_theta = bodo.enable_theta_sketches
    bodo.hiframes.boxing._use_dict_str_type = True
    bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = 300
    bodo.enable_theta_sketches = False
    try:
        f = write_iceberg_table_with_puffin_files(df, table_id, conn, "replace")
        df = _get_dist_arg(df, var_length=True)
        bodo.jit(distributed=["df"])(f)(df, table_id, conn, "replace")
        check_no_statistics_file(warehouse_loc, db_schema, table_name)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
        bodo.io.iceberg.ICEBERG_WRITE_PARQUET_CHUNK_SIZE = orig_chunk_size
        bodo.enable_theta_sketches = orig_enable_theta


def get_iceberg_pyarrow_schema(conn, table_id):
    import bodo.io.iceberg.read_compilation

    _, _, pyarrow_schema = bodo.io.iceberg.read_compilation.get_iceberg_orig_schema(
        conn, table_id
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
    from bodo.tests.test_iceberg.utils_jit import (
        check_ndv_metadata,
        get_statistics_ndvs,
    )

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
    table_name = "basic_puffin_table_full"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"
    orig_enable_theta = bodo.enable_theta_sketches
    bodo.enable_theta_sketches = True
    try:
        f = write_iceberg_table_with_puffin_files(df, table_id, conn, "replace")
        df = _get_dist_arg(df, var_length=True)
        bodo.jit(distributed=["df"])(f)(df, table_id, conn, "replace")
        puffin_file_name = check_ndv_metadata(
            warehouse_loc, db_schema, table_name, ndvs
        )
        pyarrow_schema = get_iceberg_pyarrow_schema(conn, table_id)
        ndvs = get_statistics_ndvs(puffin_file_name, pyarrow_schema)
        pd.testing.assert_extension_array_equal(ndvs, ndvs_array, check_dtype=False)
    finally:
        bodo.enable_theta_sketches = orig_enable_theta


def test_theta_sketch_detection(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test if we can correctly detect which columns have a theta sketch in the
    connector.
    """
    from bodo.io.iceberg.catalog import conn_str_to_catalog
    from bodo.io.iceberg.theta import table_columns_have_theta_sketches

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
    table_name = "sketch_detection_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"
    orig_enable_theta = bodo.enable_theta_sketches
    bodo.enable_theta_sketches = True
    try:
        f = write_iceberg_table_with_puffin_files(df, table_id, conn, "replace")
        df = _get_dist_arg(df, var_length=True)
        bodo.jit(distributed=["df"])(f)(df, table_id, conn, "replace")

        metadata = conn_str_to_catalog(conn).load_table(table_id).metadata
        theta_sketch_columns = table_columns_have_theta_sketches(metadata)
        expected_theta_sketch_columns = pd.array(
            [True, False, True, False, True], dtype=pd.BooleanDtype()
        )
        pd.testing.assert_extension_array_equal(
            theta_sketch_columns, expected_theta_sketch_columns, check_dtype=False
        )
    finally:
        bodo.enable_theta_sketches = orig_enable_theta


def test_no_theta_enabled_columns(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that when we don't have any enabled columns we don't write a puffin
    file, even if they are enabled.
    """
    from bodo.tests.test_iceberg.utils_jit import check_no_statistics_file

    df = pd.DataFrame(
        {
            "A": pd.array([1.4, 1.5, 2.451, 0] * 5, dtype=pd.Float64Dtype()),
            "B": pd.array([True, False, None, False] * 5, dtype="boolean"),
        }
    )
    table_name = "no_sketch_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"
    orig_enable_theta = bodo.enable_theta_sketches
    bodo.enable_theta_sketches = True
    try:
        f = write_iceberg_table_with_puffin_files(df, table_id, conn, "replace")
        df = _get_dist_arg(df, var_length=True)
        bodo.jit(distributed=["df"])(f)(df, table_id, conn, "replace")
        check_no_statistics_file(warehouse_loc, db_schema, table_name)
    finally:
        bodo.enable_theta_sketches = orig_enable_theta


@pytest.mark.skip(
    reason="This test is not working due to unsupported stats parsing in PyIceberg 0.8."
)
def test_theta_insert_into(iceberg_database, iceberg_table_conn, memory_leak_check):
    """
    Test that insert into operations generate theta sketches by updating the
    existing sketches.
    """
    from bodo.tests.test_iceberg.utils_jit import (
        check_ndv_metadata,
        get_statistics_ndvs,
    )

    df1 = pd.DataFrame(
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
    df2 = pd.DataFrame(
        {
            "A": pd.array(list(range(10, 20)), dtype=pd.Int64Dtype()),
            "B": pd.array([1.4, 1.5] * 5, dtype=pd.Float64Dtype()),
            "C": pd.array(["a", "af", "cfe", "afg", "egd"] * 2, dtype=object),
            "D": pd.array([False] * 10, dtype="boolean"),
            "E": pd.array(
                [
                    datetime.date(2021, 1, 2),
                    datetime.date(2021, 1, 1),
                    datetime.date(2021, 1, 2),
                    datetime.date(2021, 1, 4),
                    datetime.date(2021, 1, 3),
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.date32()),
            ),
        }
    )

    # These are hardcoded field IDs based upon how IDs are assigned starting from 1.
    ndvs = {1: "20", 3: "8", 5: "4"}
    ndvs_array = pd.array([20, None, 8, None, 4], dtype=pd.Float64Dtype())
    table_name = "insert_into_puffin_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"
    orig_enable_theta = bodo.enable_theta_sketches
    bodo.enable_theta_sketches = True
    try:
        f1 = write_iceberg_table_with_puffin_files(df1, table_id, conn, "replace")
        df1 = _get_dist_arg(df1, var_length=True)
        bodo.jit(distributed=["df"])(f1)(df1, table_id, conn, "replace")
        f2 = write_iceberg_table_with_puffin_files(df1, table_id, conn, "append")
        df2 = _get_dist_arg(df2, var_length=True)
        bodo.jit(distributed=["df"])(f2)(df2, table_id, conn, "append")
        puffin_file_name = check_ndv_metadata(
            warehouse_loc, db_schema, table_name, ndvs, num_statistics=2
        )
        pyarrow_schema = get_iceberg_pyarrow_schema(conn, table_id)
        ndvs = get_statistics_ndvs(puffin_file_name, pyarrow_schema)
        pd.testing.assert_extension_array_equal(ndvs, ndvs_array, check_dtype=False)
    finally:
        bodo.enable_theta_sketches = orig_enable_theta


def test_enable_sketches_per_column(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that we can enable/disable theta sketches on a per-column basis,
    through setting the table property of `bodo.write.theta_sketch_enabled.<column_name>`.
    """
    from bodo.tests.test_iceberg.utils_jit import (
        check_ndv_metadata,
        get_statistics_ndvs,
    )

    df = pd.DataFrame(
        {
            "A": pd.array(list(range(10)) * 2, dtype=pd.Int64Dtype()),
            "B": pd.array(list(range(10)) * 2, dtype=pd.Int64Dtype()),
            "C": pd.array(list(range(10)) * 2, dtype=pd.Int64Dtype()),
            # D and E are float types and thus should not generate theta sketches.
            "D": pd.array([1.4, 1.5, 2.451, 0] * 5, dtype=pd.Float64Dtype()),
            "E": pd.array([1.4, 1.5, 2.451, 0] * 5, dtype=pd.Float64Dtype()),
        }
    )
    sql_schema = [
        ("A", "int", True),
        ("B", "int", True),
        ("C", "int", True),
        ("D", "float", True),
        ("E", "float", True),
    ]
    # We will explicitly disable theta sketches for A, and explicitly enable them for D.
    # This should result in theta sketches being generated for B and C, but not A, D, or E.
    # Note that enabling theta sketches for D has no effect as floats are an unsupported type.

    expected_ndvs = {
        7: "10",
        8: "10",
    }
    expected_ndvs_array = pd.array([None, 10, 10, None, None], dtype=pd.Float64Dtype())
    table_name = "basic_puffin_table_column"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    table_id = f"{db_schema}.{table_name}"
    orig_enable_theta = bodo.enable_theta_sketches
    bodo.enable_theta_sketches = True
    try:
        spark = get_spark()
        # Create empty table
        if bodo.get_rank() == 0:
            create_iceberg_table(df, sql_schema, table_name, spark)

            # Set such that column A does not have theta sketches.
            # BodoSQL would normally enable theta sketches for column A by default.
            spark.sql(
                f"ALTER TABLE hadoop_prod.iceberg_db.{table_name} SET TBLPROPERTIES ('bodo.write.theta_sketch_enabled.A'='false')"
            )
            # Set such that column D does have theta sketches, which should have no effect.
            spark.sql(
                f"ALTER TABLE hadoop_prod.iceberg_db.{table_name} SET TBLPROPERTIES ('bodo.write.theta_sketch_enabled.D'='true')"
            )

        bodo.barrier()
        # Now write the data.
        f = write_iceberg_table_with_puffin_files(df, table_id, conn, "replace")
        df = _get_dist_arg(df, var_length=True)
        bodo.jit(distributed=["df"])(f)(df, table_id, conn, "replace")
        # Now make sure that column A is disabled, and the rest are enabled.
        puffin_file_name = check_ndv_metadata(
            warehouse_loc, db_schema, table_name, expected_ndvs
        )
        pyarrow_schema = get_iceberg_pyarrow_schema(conn, table_id)
        actual_ndvs_array = get_statistics_ndvs(puffin_file_name, pyarrow_schema)
        pd.testing.assert_extension_array_equal(
            actual_ndvs_array, expected_ndvs_array, check_dtype=False
        )
    finally:
        bodo.enable_theta_sketches = orig_enable_theta
