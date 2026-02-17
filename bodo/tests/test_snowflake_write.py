"""
Tests for writing to Snowflake using Python APIs
"""

from __future__ import annotations

import datetime
import io
import os
import random
import string
import traceback
import uuid
from decimal import Decimal
from tempfile import TemporaryDirectory

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI

import bodo
from bodo import BodoWarning
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    _get_dist_arg,
    _test_equal,
    check_func,
    create_snowflake_table_from_select_query,
    drop_snowflake_table,
    get_snowflake_connection_string,
    pytest_mark_one_rank,
    pytest_one_rank,
    pytest_snowflake,
    snowflake_cred_env_vars_present,
    temp_env_override,
)
from bodo.utils.testing import ensure_clean_snowflake_table

pytestmark = pytest_snowflake

# ---------------Distributed Snowflake Write Unit Tests ------------------


@pytest.mark.parametrize("is_temporary", [True, False])
def test_snowflake_write_create_internal_stage(is_temporary, memory_leak_check):
    """
    Tests creating an internal stage within Snowflake
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.snowflake import create_internal_stage, snowflake_connect
    from bodo.tests.utils_jit import reduce_sum

    # initialize global node_ranks before compiling to avoid hangs
    bodo.get_nodes_first_ranks()

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_create_internal_stage(cursor):
        with numba.objmode(stage_name="unicode_type"):
            stage_name = create_internal_stage(cursor, is_temporary=is_temporary)
        return stage_name

    bodo_impl = bodo.jit(distributed=False)(test_impl_create_internal_stage)

    # Call create_internal_stage
    stage_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        stage_name = bodo_impl(cursor)
    stage_name = comm.bcast(stage_name)

    bodo.barrier()
    passed = 1

    try:
        if bodo.get_rank() == 0:
            show_stages_sql = (
                "SHOW STAGES "
                "/* tests.test_sql:test_snowflake_create_internal_stage() */"
            )
            all_stages = cursor.execute(show_stages_sql, _is_internal=True).fetchall()
            all_stage_names = [x[1] for x in all_stages]
            assert stage_name in all_stage_names

    except Exception as e:
        print("".join(traceback.format_exception(None, e, e.__traceback__)))
        passed = 0

    if bodo.get_rank() == 0:
        cleanup_stage_sql = (
            f'DROP STAGE IF EXISTS "{stage_name}" '
            f"/* tests.test_sql:test_snowflake_write_create_internal_stage() */"
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()
    cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_create_internal_stage failed"
    bodo.barrier()


@pytest.mark.parametrize("is_temporary", [True, False])
def test_snowflake_write_drop_internal_stage(is_temporary, memory_leak_check):
    """
    Tests dropping an internal stage within Snowflake
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.snowflake import drop_internal_stage, snowflake_connect
    from bodo.tests.utils_jit import reduce_sum

    # initialize global node_ranks before compiling to avoid hangs
    bodo.get_nodes_first_ranks()

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_drop_internal_stage(cursor, stage_name):
        with numba.objmode():
            drop_internal_stage(cursor, stage_name)

    bodo_impl = bodo.jit(distributed=False)(test_impl_drop_internal_stage)

    # Create stage
    stage_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        stage_name = f"bodo_test_sql_{uuid.uuid4()}"
        create_stage_sql = (
            f'CREATE {"TEMPORARY " if is_temporary else ""}STAGE "{stage_name}" '
            f"/* tests.test_sql:test_snowflake_write_drop_internal_stage() */"
        )
        cursor.execute(create_stage_sql, _is_internal=True).fetchall()
    stage_name = comm.bcast(stage_name)

    # Call drop_internal_stage
    if bodo.get_rank() == 0:
        bodo_impl(cursor, stage_name)
    bodo.barrier()
    passed = 1

    try:
        if bodo.get_rank() == 0:
            show_stages_sql = (
                "SHOW STAGES /* tests.test_sql:test_snowflake_drop_internal_stage() */"
            )
            all_stages = cursor.execute(show_stages_sql, _is_internal=True).fetchall()
            all_stage_names = [x[1] for x in all_stages]
            assert stage_name not in all_stage_names

    except Exception as e:
        print("".join(traceback.format_exception(None, e, e.__traceback__)))
        passed = 0

    if bodo.get_rank() == 0:
        cleanup_stage_sql = (
            f'DROP STAGE IF EXISTS "{stage_name}" '
            f"/* tests.test_sql:test_snowflake_drop_internal_stage() */"
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()
        cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_drop_internal_stage failed"
    bodo.barrier()


def test_snowflake_write_do_upload_and_cleanup(memory_leak_check):
    """
    Tests uploading files to Snowflake internal stage using PUT command
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.snowflake import do_upload_and_cleanup, snowflake_connect
    from bodo.tests.utils_jit import get_start_end, reduce_sum

    db = "TEST_DB"
    schema = "SNOWFLAKE_WRITE_TEST"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_do_upload_and_cleanup(cursor, chunk_path, stage_name):
        with numba.objmode():
            do_upload_and_cleanup(cursor, 0, chunk_path, stage_name)

    bodo_impl = bodo.jit()(test_impl_do_upload_and_cleanup)

    # Set up schema and internal stage
    if bodo.get_rank() == 0:
        create_schema_sql = (
            f'CREATE OR REPLACE SCHEMA "{schema}" '
            f"/* tests.test_sql:test_snowflake_do_upload_and_cleanup() */"
        )
        cursor.execute(create_schema_sql, _is_internal=True).fetchall()

    bodo.barrier()

    stage_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        stage_name = f"bodo_test_sql_{uuid.uuid4()}"
        create_stage_sql = (
            f'CREATE STAGE "{stage_name}" '
            f"/* tests.test_sql:test_snowflake_write_do_upload_and_cleanup() */"
        )
        cursor.execute(create_stage_sql, _is_internal=True).fetchall()
    stage_name = comm.bcast(stage_name)

    bodo.barrier()

    with TemporaryDirectory() as tmp_folder:
        chunk_name = f"rank{bodo.get_rank()}_{uuid.uuid4()}.parquet"
        chunk_path = os.path.join(tmp_folder, chunk_name)

        # Build dataframe and write to parquet
        np.random.seed(5)
        random.seed(5)
        len_list = 20
        list_int = list(np.random.choice(10, len_list))
        list_double = list(np.random.choice([4.0, np.nan], len_list))
        letters = string.ascii_letters
        list_string = [
            "".join(random.choice(letters) for i in range(random.randrange(10, 100)))
            for _ in range(len_list)
        ]
        list_datetime = pd.date_range("2001-01-01", periods=len_list)
        list_date = pd.date_range("2001-01-01", periods=len_list).date
        df_in = pd.DataFrame(
            {
                "A": list_int,
                "B": list_double,
                "C": list_string,
                "D": list_datetime,
                "E": list_date,
            }
        )

        start, end = get_start_end(len(df_in))
        df_input = df_in.iloc[start:end]
        df_input.to_parquet(chunk_path)

        # Call do_upload_and_cleanup
        bodo_impl(cursor, chunk_path, stage_name)
        bodo.barrier()
        passed = 1
        npes = bodo.get_size()

    # Verify that files in stage form full dataframe when assembled
    try:
        # List files uploaded to stage
        list_stage_sql = (
            f'LIST @"{stage_name}" '
            f"/* tests.test_sql:test_snowflake_do_upload_and_cleanup() */"
        )
        listing = cursor.execute(list_stage_sql, _is_internal=True).fetchall()
        assert len(listing) == npes

        # Use GET to fetch all uploaded files
        with TemporaryDirectory() as tmp_folder:
            tmp_folder_path = tmp_folder.replace("\\", "/")
            get_stage_sql = (
                f"GET @\"{stage_name}\" 'file://{tmp_folder_path}' "
                f"/* tests.test_sql:test_snowflake_do_upload_and_cleanup() */"
            )
            cursor.execute(get_stage_sql, _is_internal=True)
            df_load = pd.read_parquet(tmp_folder)

        # Row order isn't defined, so sort the data.
        df_in_cols = df_in.columns.to_list()
        df_in_sort = df_in.sort_values(by=df_in_cols).reset_index(drop=True)
        df_load_cols = df_load.columns.to_list()
        df_load_sort = df_load.sort_values(by=df_load_cols).reset_index(drop=True)

        pd.testing.assert_frame_equal(df_in_sort, df_load_sort, check_column_type=False)

    except Exception as e:
        print("".join(traceback.format_exception(None, e, e.__traceback__)))
        passed = 0

    bodo.barrier()

    if bodo.get_rank() == 0:
        cleanup_stage_sql = (
            f'DROP STAGE IF EXISTS "{stage_name}" '
            f"/* tests.test_sql:test_snowflake_do_upload_and_cleanup() */"
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()

    cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_do_upload_and_cleanup failed"
    bodo.barrier()


def test_snowflake_write_create_table_handle_exists():
    """
    Test Snowflake write table creation, both with and without a pre-existing table
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.snowflake import create_table_handle_exists, snowflake_connect
    from bodo.tests.utils_jit import get_start_end, reduce_sum

    db = "TEST_DB"
    schema = "SNOWFLAKE_WRITE_TEST"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_create_table_handle_exists(
        cursor, location, sf_schema, if_exists, table_type=""
    ):
        with numba.objmode():
            create_table_handle_exists(
                cursor, location, sf_schema, if_exists, table_type
            )

    bodo_impl = bodo.jit(distributed=False)(test_impl_create_table_handle_exists)

    # Set up schema, internal stage, and table name
    if bodo.get_rank() == 0:
        create_schema_sql = (
            f'CREATE OR REPLACE SCHEMA "{schema}" '
            f"/* tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
        )
        cursor.execute(create_schema_sql, _is_internal=True).fetchall()

    bodo.barrier()

    stage_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        stage_name = f"bodo_test_sql_{uuid.uuid4()}"
        create_stage_sql = (
            f'CREATE STAGE "{stage_name}" '
            f"/* tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
        )
        cursor.execute(create_stage_sql, _is_internal=True).fetchall()
    stage_name = comm.bcast(stage_name)

    bodo.barrier()

    table_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        table_name = f'"snowflake_write_test_{uuid.uuid4()}"'
    table_name = comm.bcast(table_name)

    bodo.barrier()

    with TemporaryDirectory() as tmp_folder:
        df_name = f"rank{bodo.get_rank()}_{uuid.uuid4()}.parquet"
        df_path = os.path.join(tmp_folder, df_name)

        # Build dataframe, write to parquet, and upload to stage
        np.random.seed(5)
        random.seed(5)
        len_list = 20
        list_int = list(np.random.choice(10, len_list))
        list_double = list(np.random.choice([4.0, np.nan], len_list))
        letters = string.ascii_letters
        list_string = [
            "".join(random.choice(letters) for _ in range(random.randrange(10, 100)))
            for _ in range(len_list)
        ]
        list_datetime = pd.date_range("2001-01-01", periods=len_list)
        list_date = pd.date_range("2001-01-01", periods=len_list).date
        df_in = pd.DataFrame(
            {
                "A": list_int,
                "B": list_double,
                "C": list_string,
                "D": list_datetime,
                "E": list_date,
            }
        )

        start, end = get_start_end(len(df_in))
        df_input = df_in.iloc[start:end]
        df_input.to_parquet(df_path)

        df_path_path = df_path.replace("\\", "/")
        upload_put_sql = (
            f"PUT 'file://{df_path_path}' @\"{stage_name}\" AUTO_COMPRESS=FALSE "
            f"/* tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
        )
        cursor.execute(upload_put_sql, _is_internal=True)

    # Step 1: Call create_table_handle_exists.
    # This should succeed as the table doesn't exist yet.
    if bodo.get_rank() == 0:
        sf_schema = bodo.io.snowflake.gen_snowflake_schema(
            df_in.columns, bodo.typeof(df_in).data
        )
        bodo_impl(cursor, table_name, sf_schema, "fail")
    bodo.barrier()
    passed = 1

    first_table_creation_time = None  # Forward declaration
    if bodo.get_rank() == 0:
        try:
            show_tables_sql = (
                f"""SHOW TABLES STARTS WITH '{table_name.strip('"')}' """
                f"/* tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
            )
            tables_desc = cursor.execute(show_tables_sql, _is_internal=True).fetchall()
            first_table_creation_time = tables_desc[0]

            # Check that the table was created with the correct type
            # We expect TRANSIENT because we didn't specify a table type, and SNOWFLAKE_WRITE_TEST
            # is a TRANSIENT database.
            assert first_table_creation_time[4] in (
                "TABLE",
                "TRANSIENT",
                "TEMPORARY",
            ), "Table type is not an expected value"
            assert first_table_creation_time[4] == "TRANSIENT", (
                "Table schema is not TRANSIENT"
            )

            describe_table_columns_sql = (
                f"DESCRIBE TABLE {table_name} TYPE=COLUMNS "
                f"/* tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
            )
            columns_desc = cursor.execute(
                describe_table_columns_sql, _is_internal=True
            ).fetchall()
            column_names = pd.Index([elt[0] for elt in columns_desc])
            pd.testing.assert_index_equal(df_input.columns, column_names)

        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    bodo.barrier()

    # Step 2: Call create_table_handle_exists again with if_exists="fail".
    # This should fail as the table already exists
    if bodo.get_rank() == 0:
        import snowflake.connector

        err_msg = f"Object '{table_name}' already exists."
        with pytest.raises(snowflake.connector.ProgrammingError, match=err_msg):
            bodo_impl(cursor, table_name, sf_schema, "fail")
    bodo.barrier()

    # Step 3: Call create_table_handle_exists again with if_exists="append".
    # This should succeed and keep the same table from Step 1.
    if bodo.get_rank() == 0:
        bodo_impl(cursor, table_name, sf_schema, "append")
    bodo.barrier()

    if bodo.get_rank() == 0:
        try:
            show_tables_sql = (
                f"""SHOW TABLES STARTS WITH '{table_name.strip('"')}' """
                f"/* tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
            )
            tables_desc = cursor.execute(show_tables_sql, _is_internal=True).fetchall()
            assert tables_desc[0] == first_table_creation_time

        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    bodo.barrier()

    # Step 4: Call create_table_handle_exists again with if_exists="replace".
    # And a different table type ("TEMPORARY").
    # This should succeed after dropping the table from Step 1.
    if bodo.get_rank() == 0:
        bodo_impl(cursor, table_name, sf_schema, "replace", "TEMPORARY")
    bodo.barrier()

    if bodo.get_rank() == 0:
        try:
            show_tables_sql = (
                f"""SHOW TABLES STARTS WITH '{table_name.strip('"')}' """
                f"/* tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
            )
            tables_desc = cursor.execute(show_tables_sql, _is_internal=True).fetchall()
            assert tables_desc[0] != first_table_creation_time

            # Check that the table was created with the correct type
            assert tables_desc[0][4] in (
                "TABLE",
                "TRANSIENT",
                "TEMPORARY",
            ), "Table type is not an expected value"
            assert tables_desc[0][4] == "TEMPORARY", "Table schema is not TEMPORARY"

        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    bodo.barrier()

    if bodo.get_rank() == 0:
        cleanup_stage_sql = (
            f'DROP STAGE IF EXISTS "{stage_name}" '
            f"/* tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()

        cleanup_table_sql = (
            f"DROP TABLE IF EXISTS {table_name} "
            f"/* tests.test_sql:test_snowflake_create_table_handle_exists() */"
        )
        cursor.execute(cleanup_table_sql, _is_internal=True).fetchall()

    cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_write_create_table_handle_exists failed"
    bodo.barrier()


def test_snowflake_write_execute_copy_into(memory_leak_check):
    """
    Tests executing COPY_INTO into a Snowflake table from internal stage
    """
    import bodo.decorators  # isort:skip # noqa
    import bodo.io.snowflake
    from bodo.io.snowflake import execute_copy_into, snowflake_connect
    from bodo.tests.utils_jit import get_start_end, reduce_sum

    db = "TEST_DB"
    schema = "SNOWFLAKE_WRITE_TEST"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_execute_copy_into(cursor, stage_name, location, sf_schema, df_in):
        with numba.objmode(
            nsuccess="int64",
            nchunks="int64",
            nrows="int64",
            output="unicode_type",
            flatten_sql="unicode_type",
        ):
            nsuccess, nchunks, nrows, output, flatten_sql = execute_copy_into(
                cursor,
                stage_name,
                location,
                sf_schema,
                dict(zip(df_in.columns, bodo.typeof(df_in).data)),
            )
            output = repr(output)
        return nsuccess, nchunks, nrows, output, flatten_sql

    bodo_impl = bodo.jit()(test_impl_execute_copy_into)

    # Set up schema and internal stage
    if bodo.get_rank() == 0:
        create_schema_sql = (
            f'CREATE OR REPLACE SCHEMA "{schema}" '
            f"/* tests.test_sql:test_snowflake_write_execute_copy_into() */ "
        )
        cursor.execute(create_schema_sql, _is_internal=True).fetchall()

    bodo.barrier()

    stage_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        stage_name = f"bodo_test_sql_{uuid.uuid4()}"
        create_stage_sql = (
            f'CREATE STAGE "{stage_name}" '
            f"/* tests.test_sql:test_snowflake_write_execute_copy_into() */ "
        )
        cursor.execute(create_stage_sql, _is_internal=True).fetchall()
    stage_name = comm.bcast(stage_name)

    bodo.barrier()

    table_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        table_name = f'"snowflake_write_test_{uuid.uuid4()}"'
    table_name = comm.bcast(table_name)

    bodo.barrier()

    with TemporaryDirectory() as tmp_folder:
        df_name = f"rank{bodo.get_rank()}_{uuid.uuid4()}.parquet"
        df_path = os.path.join(tmp_folder, df_name)

        # Build dataframe, write to parquet, and upload to stage
        np.random.seed(5)
        random.seed(5)
        len_list = 20
        list_int = list(np.random.choice(10, len_list))
        list_double = list(np.random.choice([4.0, np.nan], len_list))
        letters = string.ascii_letters
        list_string = [
            "".join(random.choice(letters) for i in range(random.randrange(10, 100)))
            for _ in range(len_list)
        ]
        list_datetime = pd.date_range("2001-01-01", periods=len_list)
        list_date = pd.date_range("2001-01-01", periods=len_list).date
        df_in = pd.DataFrame(
            {
                "A": list_int,
                "B": list_double,
                "C": list_string,
                "D": list_datetime,
                "E": list_date,
            }
        )
        df_schema_str = (
            '"A" NUMBER(38, 0), "B" REAL, "C" TEXT, "D" TIMESTAMP_NTZ(9), "E" DATE'
        )

        start, end = get_start_end(len(df_in))
        df_input = df_in.iloc[start:end]

        # Write parquet file with Bodo to be able to handle timestamp tz type.
        def test_write(df_input):
            df_input.to_parquet(df_path, _bodo_timestamp_tz="UTC")

        bodo.jit(distributed=False)(test_write)(df_input)

        df_path_tmp = df_path.replace("\\", "/")
        upload_put_sql = (
            f"PUT 'file://{df_path_tmp}' @\"{stage_name}\" AUTO_COMPRESS=FALSE "
            f"/* tests.test_sql.test_snowflake_write_execute_copy_into() */ "
        )
        cursor.execute(upload_put_sql, _is_internal=True)

        if bodo.get_rank() == 0:
            create_table_sql = (
                f"CREATE TABLE IF NOT EXISTS {table_name} ({df_schema_str}) "
            )
            cursor.execute(create_table_sql, _is_internal=True)

    bodo.barrier()
    passed = 1
    npes = bodo.get_size()

    # Call execute_copy_into
    num_success = None
    num_chunks = None
    num_rows = None
    if bodo.get_rank() == 0:
        try:
            sf_schema = bodo.io.snowflake.gen_snowflake_schema(
                df_in.columns, bodo.typeof(df_in).data
            )
            num_success, num_chunks, num_rows, _, _ = bodo_impl(
                cursor,
                stage_name,
                table_name,
                sf_schema,
                df_in,
            )
        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    if bodo.get_rank() == 0:
        try:
            # Verify that copy_into result is correct
            assert num_success == num_chunks
            assert num_chunks == npes
            assert num_rows == len_list

            # Verify that data was copied correctly
            select_sql = (
                f"SELECT * FROM {table_name} "
                f"/* tests.test_sql:test_snowflake_write_execute_copy_into() */ "
            )
            df = cursor.execute(select_sql, _is_internal=True).fetchall()
            df_load = pd.DataFrame(df, columns=df_in.columns)

            # Row order isn't defined, so sort the data.
            df_in_cols = df_in.columns.to_list()
            df_in_sort = df_in.sort_values(by=df_in_cols).reset_index(drop=True)
            df_load_cols = df_load.columns.to_list()
            df_load_sort = df_load.sort_values(by=df_load_cols).reset_index(drop=True)

            pd.testing.assert_frame_equal(
                df_in_sort, df_load_sort, check_column_type=False, check_dtype=False
            )

        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    bodo.barrier()

    if bodo.get_rank() == 0:
        cleanup_stage_sql = (
            f'DROP STAGE IF EXISTS "{stage_name}" '
            f"/* tests.test_sql:test_snowflake_write_execute_copy_into() */ "
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()

        cleanup_table_sql = (
            f"DROP TABLE IF EXISTS {table_name} "
            f"/* tests.test_sql:test_snowflake_write_execute_copy_into() */ "
        )
        cursor.execute(cleanup_table_sql, _is_internal=True).fetchall()

    cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_write_execute_copy_into failed"
    bodo.barrier()


def test_to_sql_wrong_password():
    """
    Tests that df.to_sql produces a reasonable exception if
    a user provides the wrong password but the connection string
    still has the correct format.
    """

    @bodo.jit
    def impl(conn_str):
        df = pd.DataFrame({"A": np.arange(100)})
        df.to_sql("table", conn_str, schema="Public", index=False, if_exists="append")

    with (
        pytest.raises(
            RuntimeError, match="Incorrect username or password was specified"
        ),
        temp_env_override({"SF_PASSWORD": "wrong_password"}),
    ):
        impl(get_snowflake_connection_string("TEST_DB", "PUBLIC"))


@pytest.mark.parametrize(
    "table_names",
    [
        "table_lower",
        "TABLE_UPPER",
        "Table_mixed",
        '"table_lower"',
        '"TABLE_UPPER"',
        '"Table_mixed"',
    ],
)
def test_to_sql_table_name(table_names):
    """Test case sensitivity of table names written with DataFrame.to_sql().
    Escaping the table name with double quotes makes it case sensitive, but
    non-escaped table names default to upper-case when written to Snowflake DB.
    This test ensures that Bodo/BodoSQL users can use any combination of quotes
    and case sensitivity when naming tables to write.
    """

    @bodo.jit
    def write_impl(df, conn_str, table_name):
        df.to_sql(
            table_name, conn_str, schema="PUBLIC", index=False, if_exists="replace"
        )

    @bodo.jit
    def read_impl(conn_str, table_name):
        output_df = pd.read_sql(f"select a from {table_name}", conn_str)
        return output_df

    # Note: Bodo makes column names all lowercase internally
    # so we use "a" as the column name rather than "A" here for convenience.
    # In the future this may change when we match Snowflake behavior
    # for column names when writing tables.
    df = pd.DataFrame({"a": np.arange(100, 200)})
    conn_str = get_snowflake_connection_string(db="TEST_DB", schema="PUBLIC")

    write_impl(df, conn_str, table_names)
    output_df = read_impl(conn_str, table_names)
    pd.testing.assert_frame_equal(
        output_df.sort_values("a").reset_index(drop=True),
        df.sort_values("a").reset_index(drop=True),
        check_names=False,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("df_size", [17000 * 3, 2, 0])
@pytest.mark.parametrize(
    "sf_write_use_put",
    [
        pytest.param(True, id="with-put"),
        pytest.param(
            False,
            id="no-put",
        ),
    ],
)
@pytest.mark.parametrize(
    "snowflake_user",
    [pytest.param(1, id="s3"), pytest.param(3, id="adls", marks=pytest.mark.slow)],
)
def test_to_sql_snowflake(df_size, sf_write_use_put, snowflake_user, memory_leak_check):
    """
    Tests that df.to_sql works as expected. Since Snowflake has a limit of ~16k
    per insert, we insert 17k rows per rank to emulate a "large" insert.
    We also test for the empty dataframe case.
    """

    # Skip test if env vars with snowflake creds required for this test are not set.
    if not snowflake_cred_env_vars_present(snowflake_user):
        pytest.skip(reason="Required env vars with Snowflake credentials not set")

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema, user=snowflake_user)

    import bodo
    import bodo.io.snowflake
    from bodo.tests.utils_jit import get_start_end, reduce_sum

    rng = np.random.default_rng(5)
    letters = np.array(list(string.ascii_letters))
    py_output = pd.DataFrame(
        {
            "a": np.arange(df_size),
            "b": np.arange(df_size).astype(np.float64),
            "c": [
                "".join(rng.choice(letters, size=rng.integers(10, 100)))
                for _ in range(df_size)
            ],
            "d": pd.date_range("2001-01-01", periods=df_size),
            "e": pd.date_range("2001-01-01", periods=df_size).date,
        }
    )
    start, end = get_start_end(len(py_output))
    df = py_output.iloc[start:end]

    @bodo.jit(distributed=["df"])
    def test_write(df, name, conn, schema):
        df.to_sql(name, conn, if_exists="replace", index=False, schema=schema)
        bodo.barrier()

    # Specify Snowflake write hyperparameters
    old_sf_write_use_put = bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT

    try:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = sf_write_use_put

        with ensure_clean_snowflake_table(conn) as name:
            test_write(df, name, conn, schema)

            bodo.barrier()
            passed = 1
            if bodo.get_rank() == 0:
                try:
                    output_df = pd.read_sql(f"select * from {name}", conn)
                    # Row order isn't defined, so sort the data.
                    output_cols = output_df.columns.to_list()
                    output_df = output_df.sort_values(output_cols).reset_index(
                        drop=True
                    )
                    py_cols = py_output.columns.to_list()
                    py_output = py_output.sort_values(py_cols).reset_index(drop=True)
                    pd.testing.assert_frame_equal(
                        output_df, py_output, check_dtype=False, check_column_type=False
                    )
                except Exception as e:
                    print("".join(traceback.format_exception(None, e, e.__traceback__)))
                    passed = 0

            n_passed = reduce_sum(passed)
            assert n_passed == bodo.get_size()
            bodo.barrier()
    finally:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = old_sf_write_use_put


def test_snowflake_to_sql_bodo_datatypes_part1(memory_leak_check):
    """
    Tests that df.to_sql works with all Bodo's supported dataframe datatypes
    This compares the dataframe Bodo writes and reads vs.
    the dataframe Pandas writes and reads
    """
    from bodo.tests.utils_jit import reduce_sum

    bodo_tablename = "BODO_DT_P1"
    py_tablename = "PY_DT_P1"
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    rng = np.random.default_rng(5)
    # NOTE: nullable bool, Decimal, time, binary,
    # struct, list, list-list, tuple are not hard-coded.
    df_len = 7
    letters = string.ascii_letters
    list_string = [
        "".join(
            rng.choice(list(letters))
            for i in range(rng.integers(low=10, high=100, size=1)[0])
        )
        for _ in range(df_len)
    ]
    df = pd.DataFrame(
        {
            # int8
            "INT8_COL": np.array(rng.integers(-10, 10, df_len), dtype="int8"),
            # int16
            "INT16_COL": np.array(rng.integers(-10, 10, df_len), dtype="int16"),
            # int32
            "INT32_COL": np.array(rng.integers(-10, 10, df_len), dtype="int32"),
            # uint8
            "UINT8_COL": np.array(rng.integers(0, 20, df_len), dtype="uint8"),
            # uint16
            "UINT16_COL": np.array(rng.integers(0, 20, df_len), dtype="uint16"),
            # uint32
            "UINT32_COL": np.array(rng.integers(0, 20, df_len), dtype="uint32"),
            # int64
            "INT64_COL": rng.integers(0, 500, df_len),
            # nullable int
            "NULLABLE_INT_COL": pd.Series(
                pd.Series(rng.choice([4, np.nan], df_len), dtype="Int64"),
            ),
            # nullable boolean
            "NULLABLE_BOOL_COL": pd.Series(
                [True, False, True, None, True, False, None], dtype="boolean"
            ),
            # boolean
            "BOOL_COL": np.array(rng.choice([True, False], df_len)),
            # float
            "FLOAT_COL": np.array(rng.choice([1.4, 3.4, 2.5], df_len)),
            # float32
            "FLOAT32_COL": np.array(
                rng.choice([1.4, 3.4, 2.5], df_len), dtype="float32"
            ),
            # nullable float
            "NULLABLE_FLOAT_COL": pd.array(
                rng.choice([1.4, None, 3.4, 2.5], df_len), dtype="Float64"
            ),
            # nan with float
            "NAN_FLOAT_COL": np.array(rng.choice([np.nan, 2.5], df_len)),
            # Date
            "DATE_COL": pd.date_range("2001-01-01", periods=df_len).date,
            # timedelta
            "TIMEDELTA_COL": pd.Series(
                pd.timedelta_range(start="1 day", periods=7, unit="ns")
            ),
            # string
            "STRING_COL": list_string,
        }
    )

    def sf_write(df, tablename, conn, schema):
        df.to_sql(tablename, conn, if_exists="replace", index=False, schema=schema)

    def sf_read(conn, tablename):
        ans = pd.read_sql(f"select * from {tablename}", conn)
        return ans

    with ensure_clean_snowflake_table(conn, bodo_tablename) as b_tablename:
        bodo.jit(distributed=["df"])(sf_write)(
            _get_dist_arg(df), b_tablename, conn, schema
        )
        bodo.barrier()

        bodo_result = bodo.jit(sf_read)(conn, b_tablename)
        bodo_result = bodo.libs.distributed_api.gatherv(bodo_result)

    passed = 1
    if bodo.get_rank() == 0:
        try:
            with ensure_clean_snowflake_table(conn, py_tablename, False) as tablename:
                from snowflake.connector.pandas_tools import pd_writer

                df.to_sql(
                    tablename, conn, index=False, method=pd_writer, if_exists="replace"
                )
                py_output = sf_read(conn, tablename)
            # disable dtype check. Int8 vs. int64
            # Sort output as in some rare cases data read from SF
            # are in different order from written df.
            _test_equal(
                bodo_result.sort_values("int8_col").reset_index(drop=True),
                py_output.astype({"nullable_bool_col": "boolean"})
                .sort_values("int8_col")
                .reset_index(drop=True),
                check_dtype=False,
            )
        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_to_sql_bodo_datatypes failed"


def test_snowflake_to_sql_bodo_datatypes_part2(memory_leak_check):
    """
    Tests that df.to_sql works with all Bodo's supported dataframe datatypes
    This compare written table in snowflake by Bodo vs. original Dataframe
        # NOTE: these types are written differently with Pandas to_sql write
        # 1. Time error:
        #       pyarrow.lib.ArrowInvalid: ('Could not convert Time(12, 0, 0, 0, 0, 0, precision=9)
        #       with type Time: did not recognize Python value type when inferring an Arrow data type'
        # 2. Binary data is stored as VARCHAR.
        # 3. Decimal. I don't know why but probably precision difference.
        # 4. Others have datatype differences.
    """
    import bodo.types
    from bodo.tests.utils_jit import reduce_sum

    bodo_tablename = "BODO_DT_P2"
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    df_len = 7

    df = pd.DataFrame(
        {
            # Tuple
            "tuple_col": (1, 2, 3, 4, 5, 6, 7),
            # list
            "list_col": [1, 2, 3, 4, 5, 6, 7],
            # binary.
            "binary_col": [b"", b"abc", b"c", np.nan, b"ccdefg", b"abcde", bytes(3)],
            # datetime
            "datetime_col": pd.date_range("2022-01-01", periods=df_len),
            # Decimal
            "decimal_col": pd.Series(
                [
                    Decimal("17.9"),
                    Decimal("1.6"),
                    Decimal("-0.2"),
                    Decimal("44.2"),
                    Decimal("15.43"),
                    Decimal("8.4"),
                    Decimal("-100.56"),
                ]
            ),
            # TIME.
            # Examples here intentionaly don't have milli/micro/nano units.
            # TODO: add them after fixing time code to correctly store these
            # in parquet
            # with type Time: did not recognize Python value type when inferring an Arrow data type'
            # pa.time32("s")
            "time_col_p0": [
                bodo.types.Time(12, 0, precision=0),
                bodo.types.Time(1, 1, 3, precision=0),
                bodo.types.Time(2, precision=0),
                bodo.types.Time(12, 0, precision=0),
                bodo.types.Time(5, 6, 3, precision=0),
                bodo.types.Time(9, precision=0),
                bodo.types.Time(11, 19, 34, precision=0),
            ],
            # pa.time32("ms")
            "time_col_p3": [
                bodo.types.Time(12, 0, precision=3),
                bodo.types.Time(1, 1, 3, precision=3),
                bodo.types.Time(2, precision=3),
                bodo.types.Time(12, 0, precision=3),
                bodo.types.Time(6, 7, 13, precision=3),
                bodo.types.Time(2, precision=3),
                bodo.types.Time(17, 1, 3, precision=3),
            ],
            # # pa.time64("us")
            "time_col_p6": [
                bodo.types.Time(12, 0, precision=6),
                bodo.types.Time(1, 1, 3, precision=6),
                bodo.types.Time(2, precision=6),
                bodo.types.Time(12, 0, precision=6),
                bodo.types.Time(5, 11, 53, precision=6),
                bodo.types.Time(2, precision=6),
                bodo.types.Time(1, 1, 3, precision=6),
            ],
            # pa.time64("ns"). Default precision=9
            "time_col_p9": [
                bodo.types.Time(12, 0),
                bodo.types.Time(1, 1, 3),
                bodo.types.Time(2),
                bodo.types.Time(12, 1),
                bodo.types.Time(5, 6, 3),
                bodo.types.Time(9),
                bodo.types.Time(11, 19, 34),
            ],
            "timestamp_ltz_col": [
                pd.Timestamp("1999-12-15 11:03:40", tz="Asia/Dubai"),
                pd.Timestamp("2002-11-06 17:00:00", tz="Asia/Dubai"),
                pd.Timestamp("1924-09-16 03:20:00", tz="Asia/Dubai"),
                pd.Timestamp("1994-01-01 02:03:04", tz="Asia/Dubai"),
                pd.Timestamp("2020-12-26 03:50:30", tz="Asia/Dubai"),
                pd.Timestamp("2022-11-06 22:00:50", tz="Asia/Dubai"),
                pd.Timestamp("2023-10-06 13:30:00", tz="Asia/Dubai"),
            ],
        }
    )

    @bodo.jit(distributed=["df"])
    def test_write(df, name, conn, schema):
        df.to_sql(name, conn, if_exists="replace", index=False, schema=schema)

    def sf_read(conn, tablename):
        ans = pd.read_sql(f"select * from {tablename}", conn)
        return ans

    with ensure_clean_snowflake_table(conn, bodo_tablename) as b_tablename:
        test_write(_get_dist_arg(df), b_tablename, conn, schema)
        bodo_result = bodo.jit(sf_read)(conn, b_tablename)

    bodo_result = bodo.libs.distributed_api.gatherv(bodo_result)

    passed = 1
    if bodo.get_rank() == 0:
        try:
            # Re-write this column with precision=6
            # SF ignores nanoseconds precisions with COPY INTO FROM PARQUET-FILES
            # See for more information
            df["time_col_p9"] = [
                bodo.types.Time(12, 0, precision=6),
                bodo.types.Time(1, 1, 3, precision=6),
                bodo.types.Time(2, precision=6),
                bodo.types.Time(12, 1, precision=6),
                bodo.types.Time(5, 6, 3, precision=6),
                bodo.types.Time(9, precision=6),
                bodo.types.Time(11, 19, 34, precision=6),
            ]
            # Sort output as in some rare cases data read from SF
            # are in different order from written df.
            _test_equal(
                bodo_result.sort_values("datetime_col").reset_index(drop=True),
                df.sort_values("datetime_col").reset_index(drop=True),
                check_dtype=False,
            )
        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_to_sql_bodo_datatypes_part2 failed"


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {
                "idx": list(range(7)),
                # list list
                "a": pd.Series(
                    [
                        [1, 2],
                        [3],
                        [4, 5, 6, 7],
                        [4, 5],
                        [32, 45],
                        [1, 4, 7, 8],
                        [
                            3,
                            4,
                            6,
                        ],
                    ],
                    dtype=pd.ArrowDtype(pa.list_(pa.int64())),
                ),
            }
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "idx": list(range(7)),
                    # struct
                    "a": pd.Series(
                        [
                            [{"A": 1, "B": 2}, {"A": 10, "B": 20}],
                            [{"A": 3, "B": 4}],
                            [
                                {"A": 5, "B": 6},
                                {"A": 50, "B": 60},
                                {"A": 500, "B": 600},
                            ],
                            [{"A": 10, "B": 20}, {"A": 100, "B": 200}],
                            [{"A": 30, "B": 40}],
                            [
                                {"A": 50, "B": 60},
                                {"A": 500, "B": 600},
                                {"A": 5000, "B": 6000},
                            ],
                            [{"A": 30, "B": 40}],
                        ],
                        dtype=pd.ArrowDtype(
                            pa.list_(
                                pa.struct(
                                    [
                                        pa.field("A", pa.int64()),
                                        pa.field("B", pa.int64()),
                                    ]
                                )
                            )
                        ),
                    ),
                }
            ),
        ),
    ],
    ids=["list_list", "struct"],
)
def test_snowflake_to_sql_bodo_datatypes_part3(df, memory_leak_check):
    """
    Tests that df.to_sql works with all Bodo's supported dataframe datatypes
    """

    bodo_tablename = "BODO_DT_P3"
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    @bodo.jit(distributed=["df"])
    def test_write(df, name, conn, schema):
        df.to_sql(name, conn, if_exists="replace", index=False, schema=schema)

    def sf_read(conn, tablename):
        ans = pd.read_sql(f"select * from {tablename} order by idx", conn)
        return ans

    with ensure_clean_snowflake_table(conn, bodo_tablename) as b_tablename:
        test_write(_get_dist_arg(df), b_tablename, conn, schema)
        check_func(
            sf_read,
            (conn, b_tablename),
            py_output=df,
            check_dtype=False,
            check_names=False,
            convert_columns_to_pandas=True,
        )


def test_snowflake_to_sql_nullarray(memory_leak_check):
    """Tests that df.to_sql works with NullArrayType."""
    import bodo.libs.null_arr_ext
    from bodo.tests.utils_jit import reduce_sum

    bodo_tablename = "BODO_DT_P4"
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    @bodo.jit(distributed=["df"])
    def test_write(name, conn, schema):
        df = pd.DataFrame({"null_arr": bodo.libs.null_arr_ext.init_null_array(10)})
        df.to_sql(name, conn, if_exists="replace", index=False, schema=schema)

    def sf_read(conn, tablename):
        ans = pd.read_sql(f"select * from {tablename}", conn)
        return ans

    with ensure_clean_snowflake_table(conn, bodo_tablename) as b_tablename:
        test_write(b_tablename, conn, schema)
        bodo_result = bodo.jit(sf_read)(conn, b_tablename)
        if bodo.get_rank() == 0:
            py_output = sf_read(conn, b_tablename)

    bodo_result = bodo.libs.distributed_api.gatherv(bodo_result)

    passed = 1
    if bodo.get_rank() == 0:
        try:
            # avoid checking dtype
            # Attribute "dtype" are different
            # [left]:  string[pyarrow]
            # [right]: object
            _test_equal(
                bodo_result,
                py_output.astype(pd.ArrowDtype(pa.string())),
                check_dtype=False,
            )
        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_to_sql_nullarray failed"


@pytest_mark_one_rank
def test_to_sql_snowflake_user2(memory_leak_check):
    """
    Tests that df.to_sql works when the Snowflake account password has special
    characters.
    """
    import platform

    # This test runs on both Mac and Linux, so give each table a different
    # name for the highly unlikely but possible case the tests run concurrently.
    # We use uppercase table names as Snowflake by default quotes identifiers.
    if platform.system() == "Darwin":
        name = "TOSQLSMALLMAC"
    else:
        name = "TOSQLSMALLLINUX"
    db = "TEST_DB"
    schema = "PUBLIC"
    # User 2 has @ character in the password
    conn = get_snowflake_connection_string(db, schema, user=2)

    rng = np.random.default_rng(5)
    len_list = 1000
    letters = np.array(list(string.ascii_letters))
    df = pd.DataFrame(
        {
            "a": rng.choice(500, size=len_list),
            "b": rng.choice(500, size=len_list),
            "c": [
                "".join(rng.choice(letters, size=rng.integers(10, 100)))
                for _ in range(len_list)
            ],
            "d": pd.date_range("2002-01-01", periods=1000),
            "e": pd.date_range("2002-01-01", periods=1000).date,
        }
    )

    @bodo.jit
    def test_write(df, name, conn, schema):
        df.to_sql(name, conn, if_exists="replace", index=False, schema=schema)

    test_write(df, name, conn, schema)

    def read(conn):
        return pd.read_sql(f"select * from {name}", conn)

    # Can't read with pandas because it will throw an error if the username has
    # special characters
    check_func(read, (conn,), py_output=df, dist_test=False, check_dtype=False)


def test_snowflake_to_sql_colname_case(memory_leak_check):
    """
    Tests that df.to_sql works with different upper/lower case of column name
    """
    from bodo.tests.utils_jit import reduce_sum

    tablename = "TEST_CASE"
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    # Setting all data to be int as we don't care about values in this test.
    # We're just ensuring that column names match with different cases.
    rng = np.random.default_rng(5)
    # NOTE: We're testing the to_sql is writing columns correctly.
    # For read: all uppercase column names will fail if compared
    # to actual dataframe. This is because we match Pandas Behavior where
    # all uppercase names are returned as lowercase. See _get_sql_df_type_from_db
    df = pd.DataFrame(
        {
            # all lower
            "aaa": rng.integers(0, 500, 10),
            # all upper
            "BBB": rng.integers(0, 500, 10),
            # mix
            "CdEd": rng.integers(0, 500, 10),
            # lower with special char.
            "d_e_$": rng.integers(0, 500, 10),
            # upper with special char.
            "F_G_$": rng.integers(0, 500, 10),
            # no letters
            "_123$": rng.integers(0, 500, 10),
            # special character only
            "_$": rng.integers(0, 500, 10),
            # lower with number
            "a12": rng.integers(0, 500, 10),
            # upper with number
            "B12C": rng.integers(0, 500, 10),
        }
    )

    @bodo.jit(distributed=["df"])
    def test_write(df, name, conn, schema):
        df.to_sql(name, conn, if_exists="replace", index=False, schema=schema)

    with ensure_clean_snowflake_table(conn, tablename) as name:
        test_write(_get_dist_arg(df), name, conn, schema)

        def sf_read(conn):
            return pd.read_sql(f"select * from {name}", conn)

        bodo_result = bodo.jit(sf_read)(conn)
        bodo_result = bodo.libs.distributed_api.gatherv(bodo_result)

        passed = 1
        if bodo.get_rank() == 0:
            try:
                py_output = sf_read(conn)
                # disable dtype check. Int16 vs. int64
                pd.testing.assert_frame_equal(
                    bodo_result, py_output, check_dtype=False, check_column_type=False
                )
            except Exception as e:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))
                passed = 0

        n_passed = reduce_sum(passed)
        n_pes = bodo.get_size()
        assert n_passed == n_pes, "test_snowflake_to_sql_colname_case failed"


@pytest.mark.slow
def test_to_sql_schema_warning(memory_leak_check):
    """[BE-2117] This test for not using schema with snowflake"""

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    def impl(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
    tablename = "schema_warning_table"
    if bodo.get_rank() == 0:  # warning is thrown only on rank 0
        with pytest.warns(BodoWarning, match="schema argument is recommended"):
            bodo.jit(impl)(df, tablename, conn)
    else:
        bodo.jit(impl)(df, tablename, conn)


def test_to_sql_snowflake_nulls_in_dict(memory_leak_check):
    """
    Test that Snowflake write works even when there are nulls in
    the dictionary of a dictionary encoded array. Arrow doesn't
    support this, so we drop the NAs from all dictionaries ourselves
    before the write step.
    See [BE-4331](https://bodo.atlassian.net/browse/BE-4331)
    for more context.
    We also explicitly test the table-format case since the codegen
    for it is slightly different.
    """
    from bodo.tests.utils_jit import reduce_sum

    S = pa.DictionaryArray.from_arrays(
        np.array([0, 2, 1, 0, 1, 3, 0, 1, 3, 3, 2, 0, 2, 3, 1] * 2, dtype=np.int32),
        pd.Series(["B", None, "A", None]),
    )
    A = np.arange(30)

    @bodo.jit(distributed=["S", "A"])
    def impl(S, A, table_name, conn):
        S = pd.Series(S)
        df = pd.DataFrame({"S": S, "A": A})
        # Set a small chunksize to force each rank to write multiple files and
        # hence test that there's no refcount issues due to the slicing.
        df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=4)
        return df

    @bodo.jit(distributed=["S", "A"])
    def impl_table_format(S, A, table_name, conn):
        S = pd.Series(S)
        df = pd.DataFrame({"S": S, "A": A})
        # Set a small chunksize to force each rank to write multiple files and
        # hence test that there's no refcount issues due to the slicing.
        # Force dataframe to use table format.
        df = bodo.hiframes.pd_dataframe_ext._tuple_to_table_format_decoded(df)
        df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=4)
        return df

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    def test_impl(impl, S, A):
        with ensure_clean_snowflake_table(conn) as name:
            py_output = pd.DataFrame({"s": S, "a": A})
            impl(_get_dist_arg(S), _get_dist_arg(A), name, conn)

            bodo.barrier()
            passed = 1
            if bodo.get_rank() == 0:
                try:
                    output_df = pd.read_sql(f"select * from {name}", conn)
                    # Row order isn't defined, so sort the data.
                    output_cols = output_df.columns.to_list()
                    output_df = output_df.sort_values(output_cols).reset_index(
                        drop=True
                    )
                    py_cols = py_output.columns.to_list()
                    py_output = py_output.sort_values(py_cols).reset_index(drop=True)
                    pd.testing.assert_frame_equal(
                        output_df, py_output, check_dtype=False, check_column_type=False
                    )
                except Exception as e:
                    print("".join(traceback.format_exception(None, e, e.__traceback__)))
                    passed = 0

            n_passed = reduce_sum(passed)
            assert n_passed == bodo.get_size()
            bodo.barrier()

    test_impl(impl, S, A)
    test_impl(impl_table_format, S, A)
    # Use the same dict-encoded array for both columns (should catch any refcount bugs)
    test_impl(impl, S, S)
    test_impl(impl_table_format, S, S)


def test_snowflake_write_column_name_special_chars(memory_leak_check):
    """
    Tests that Snowflake write works correctly when a column name contains
    special characters and that column are still written case insensitive.
    """

    @bodo.jit(distributed=["df"])
    def write_impl(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    @bodo.jit(distributed=False)
    def read_impl(conn_str, table_name):
        output_df = pd.read_sql(
            f'select "^$OTHER_temp-col" from {table_name}', conn_str
        )
        return output_df

    expected_output = pd.DataFrame(
        {"$TEMP_COLUMN0": [1.12, 1.1] * 5, "^$OTHER_temp-col": [213, -7] * 5}
    )
    if bodo.get_rank() == 0:
        df = expected_output
    else:
        df = None
    df = bodo.scatterv(df)
    # Generate a unique table name
    comm = MPI.COMM_WORLD
    db = "TEST_DB"
    schema = "PUBLIC"
    table_name = None
    if bodo.get_rank() == 0:
        unique_name = str(uuid.uuid4()).replace("-", "_")
        table_name = f"SPECIAL_CHARS_{unique_name}".lower()
    table_name = comm.bcast(table_name)
    conn_str = get_snowflake_connection_string(db, schema)
    try:
        write_impl(df, table_name, conn_str)
        output_df = read_impl(conn_str, table_name)
        pd.testing.assert_frame_equal(
            output_df.sort_values(by=["^$OTHER_temp-col"]).reset_index(drop=True),
            expected_output[["^$OTHER_temp-col"]]
            .sort_values(by=["^$OTHER_temp-col"])
            .reset_index(drop=True),
            check_names=False,
            check_dtype=False,
            check_index_type=False,
        )
    finally:
        # Make sure every rank has finished writing and reading
        # before dropping the table
        bodo.barrier()
        drop_snowflake_table(table_name, db, schema)


@pytest.mark.parametrize(
    "sf_write_use_put",
    [
        pytest.param(True, id="with-put"),
        pytest.param(
            False,
            id="no-put",
        ),
    ],
)
@pytest.mark.parametrize(
    "sf_write_streaming_num_files",
    [
        pytest.param(0, id="num-files-0"),
        pytest.param(1, id="num-files-1"),
        pytest.param(64, id="num-files-64"),
    ],
)
@pytest.mark.parametrize(
    "is_distributed",
    [
        pytest.param(True, id="distributed"),
        pytest.param(
            False,
            id="replicated",
            # Note: Replicated write is not supported yet.
            marks=pytest_one_rank,
        ),
    ],
)
@pytest.mark.parametrize(
    "snowflake_user",
    [
        pytest.param(1, id="s3"),
        pytest.param(3, id="adls", marks=pytest.mark.slow),
    ],
)
@pytest.mark.timeout(500)
def test_batched_write_agg(
    sf_write_use_put,
    sf_write_streaming_num_files,
    is_distributed,
    snowflake_user,
    memory_leak_check,
):
    """
    Test a simple use of batched Snowflake writes by reading a table, writing
    the results, then reading again
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.arrow_reader import (
        arrow_reader_del,
        read_arrow_next,
    )
    from bodo.io.snowflake_write import (
        snowflake_writer_append_table,
        snowflake_writer_init,
    )
    from bodo.spawn.utils import run_rank0
    from bodo.tests.utils_jit import reduce_sum

    comm = MPI.COMM_WORLD
    col_meta = bodo.utils.typing.ColNamesMetaType(
        (
            "l_orderkey",
            "l_partkey",
            "l_suppkey",
            "l_linenumber",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
            "l_shipinstruct",
            "l_shipmode",
            "l_comment",
        )
    )

    conn_r = bodo.tests.utils.get_snowflake_connection_string(
        "SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1", user=snowflake_user
    )
    conn_w = bodo.tests.utils.get_snowflake_connection_string(
        "TEST_DB", "PUBLIC", user=snowflake_user
    )

    cursor_w = run_rank0(
        lambda: bodo.io.snowflake.snowflake_connect(conn_w).cursor(), bcast_result=False
    )()

    table_r = "LINEITEM"
    table_w = None  # Forward declaration
    if bodo.get_rank() == 0:
        table_w = f'"LINEITEM_TEST_{uuid.uuid4()}"'
    table_w = comm.bcast(table_w)

    # To test multiple COPY INTO, temporarily reduce Parquet write chunk size
    # and the number of files included in each streaming COPY INTO
    old_use_put = bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT
    old_chunk_size = bodo.io.snowflake.SF_WRITE_PARQUET_CHUNK_SIZE
    old_streaming_num_files = bodo.io.snowflake.SF_WRITE_STREAMING_NUM_FILES
    try:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = sf_write_use_put
        bodo.io.snowflake.SF_WRITE_PARQUET_CHUNK_SIZE = int(256e3)
        bodo.io.snowflake.SF_WRITE_STREAMING_NUM_FILES = sf_write_streaming_num_files
        ctas_meta = bodo.utils.typing.CreateTableMetaType()

        def impl_write(conn_r, conn_w):
            reader0 = pd.read_sql(
                f"SELECT * FROM {table_r} LIMIT 10000", conn_r, _bodo_chunksize=1000
            )  # type: ignore
            writer = snowflake_writer_init(
                -1,
                conn_w,
                table_w,
                "PUBLIC",
                "replace",
                "",
            )
            total0 = 0
            all_is_last = False
            iter_val = 0
            while not all_is_last:
                table, is_last = read_arrow_next(reader0, True)
                total0 += bodo.hiframes.table.local_len(table)
                all_is_last = snowflake_writer_append_table(
                    writer, table, col_meta, is_last, iter_val, None, ctas_meta
                )
                iter_val += 1
            arrow_reader_del(reader0)
            return total0

        @bodo.jit(cache=False)
        def impl_check(conn_w):
            reader1 = pd.read_sql(
                f"SELECT * FROM {table_w}", conn_w, _bodo_chunksize=1000
            )  # type: ignore
            total1 = 0

            all_is_last = False
            and_op = np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value)
            while not all_is_last:
                table, is_last = read_arrow_next(reader1, True)
                all_is_last = bodo.libs.distributed_api.dist_reduce(is_last, and_op)
                total1 += bodo.hiframes.table.local_len(table)
            arrow_reader_del(reader1)
            return total1

        if is_distributed:
            # Not specifying anything will run it in a distributed fashion
            # by default.
            total0 = bodo.jit(cache=False)(impl_write)(conn_r, conn_w)
            total0 = reduce_sum(total0)
            total1 = impl_check(conn_w)
            total1 = reduce_sum(total1)
            assert total0 == total1, (
                f"Distributed streaming write failed: "
                f"wrote {total0} but read back {total1}"
            )
        else:
            impl_write_repl = bodo.jit(cache=False, distributed=False)(impl_write)
            err = None
            total0 = 0
            if bodo.get_rank() == 0:
                try:
                    total0 = impl_write_repl(conn_r, conn_w)
                except Exception as e:
                    err = e
            err = comm.bcast(err)
            if isinstance(err, Exception):
                raise err
            total0 = comm.bcast(total0)
            total1 = impl_check(conn_w)
            assert total0 == total1, (
                f"Replicated streaming write failed: "
                f"wrote {total0} but read back {total1}"
            )

    finally:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = old_use_put
        bodo.io.snowflake.SF_WRITE_PARQUET_CHUNK_SIZE = old_chunk_size
        bodo.io.snowflake.SF_WRITE_STREAMING_NUM_FILES = old_streaming_num_files

        if bodo.get_rank() == 0:
            cleanup_table_sql = (
                f"DROP TABLE IF EXISTS {table_w} "
                f"/* tests.test_sql:test_batched_write_agg() */"
            )
            cursor_w.execute(cleanup_table_sql, _is_internal=True).fetchall()
            cursor_w.close()


@pytest.mark.parametrize(
    "is_variant",
    [
        pytest.param(False, id="not_variant"),
        pytest.param(True, id="variant"),
    ],
)
@pytest.mark.parametrize(
    "df, expected_df, column_type",
    [
        (  # array item array
            pd.DataFrame(
                {
                    "a": np.arange(10),
                    "b": pd.Series(
                        [np.arange(5)] * 10,
                        dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": np.arange(10),
                    "b": pd.Series(
                        [np.arange(5)] * 10,
                        dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                    ),
                }
            ),
            "array",
        ),
        pytest.param(  # struct array
            pd.DataFrame(
                {
                    "a": np.arange(10),
                    "b": pd.Series(
                        [
                            {"W": 1, "X": "AB", "Y": 1.100000000000000e00, "Z": i}
                            for i in range(10)
                        ],
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [
                                    pa.field("W", pa.int64()),
                                    pa.field("X", pa.string()),
                                    pa.field("Y", pa.float64()),
                                    pa.field("Z", pa.int64()),
                                ]
                            )
                        ),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": np.arange(10),
                    "b": pd.Series(
                        [
                            {"W": 1, "X": "AB", "Y": 1.100000000000000e00, "Z": i}
                            for i in range(10)
                        ],
                        dtype=pd.ArrowDtype(
                            pa.struct(
                                [
                                    pa.field("W", pa.int64()),
                                    pa.field("X", pa.string()),
                                    pa.field("Z", pa.int64()),
                                    pa.field("Y", pa.float64()),
                                ]
                            )
                        ),
                    ),
                }
            ),
            "object",
        ),
        (  # map array
            pd.DataFrame(
                {
                    "a": np.arange(10),
                    "b": pd.Series(
                        [{"2": 4.1, "1": 5.1}, {"3": 100.38}] * 5,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.float64())),
                    ),
                    "c": pd.Series(
                        [{"a": "a", "b": "b"}, {"c": "c"}] * 5,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
                    ),
                    "d": pd.Series(
                        [
                            {
                                "a": pd.Timestamp("2000-01-01"),
                                "b": pd.Timestamp("2000-01-02"),
                            },
                            {"c": pd.Timestamp("2000-01-03")},
                        ]
                        * 5,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.timestamp("ns"))),
                    ),
                    "e": pd.Series(
                        [{"a": datetime.date(2010, 1, 10)}] * 10,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.date32())),
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": np.arange(10),
                    "b": pd.Series(
                        [{"1": 5.1, "2": 4.1}, {"3": 100.38}] * 5,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.float64())),
                    ),
                    "c": pd.Series(
                        [{"a": "a", "b": "b"}, {"c": "c"}] * 5,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
                    ),
                    "d": pd.Series(
                        [
                            {
                                "a": pd.Timestamp("2000-01-01"),
                                "b": pd.Timestamp("2000-01-02"),
                            },
                            {"c": pd.Timestamp("2000-01-03")},
                        ]
                        * 5,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.timestamp("ns"))),
                    ),
                    "e": pd.Series(
                        [{"a": datetime.date(2010, 1, 10)}] * 10,
                        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.date32())),
                    ),
                }
            ),
            "object",
        ),
    ],
    ids=[
        "array_item_array",
        "struct_object",
        "map_object",
    ],
)
@pytest.mark.parametrize("write_type", ["append", "replace"])
def test_batched_write_nested_array(
    df, expected_df, column_type, is_variant, write_type, memory_leak_check
):
    """
    Test writing a table with a column of nested arrays to Snowflake
    and then reading it back
    """
    from bodo.spawn.utils import run_rank0

    if is_variant and write_type == "replace":
        pytest.skip("When replacing a table columns are never written as variant")

    if is_variant:
        column_type = "variant"

    from bodo.io.snowflake import snowflake_connect
    from bodo.io.snowflake_write import (
        snowflake_writer_append_table,
        snowflake_writer_init,
    )

    conn = bodo.tests.utils.get_snowflake_connection_string("TEST_DB", "PUBLIC")
    kept_cols = bodo.utils.typing.MetaType(tuple(range(len(df.columns))))
    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns.str.upper()))
    batch_size = 3

    with ensure_clean_snowflake_table(conn, "NESTED_ARRAY_WRITE_TEST") as table_name:
        # Create table with correct schema
        cursor = snowflake_connect(conn).cursor()
        if write_type == "append":
            run_rank0(
                lambda cursor, table_name, column_type, df: cursor.execute(
                    f"create or replace transient table {table_name} (a NUMBER,  {','.join([f'{col} {column_type}' for col in df.columns[1:]])})"
                ),
                bcast_result=False,
            )(cursor, table_name, column_type, df)
        cursor.close()
        ctas_meta = bodo.utils.typing.CreateTableMetaType()

        def write_impl(conn, df):
            writer = snowflake_writer_init(
                -1,
                conn,
                table_name,
                "PUBLIC",
                write_type,
                "TRANSIENT",
            )

            # Streaming read might output a different number of chunks on each
            # rank, but streaming write assumes the number of chunks is equal,
            # so we need to manually synchronize `is_last`. We assume that
            # arrow_reader handles repeated empty calls.
            all_is_last = False
            iter_val = 0
            T1 = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
                (),
                kept_cols,
                2,
            )
            while not all_is_last:
                table = bodo.hiframes.table.table_local_filter(
                    T1,
                    slice(
                        (iter_val * batch_size),
                        ((iter_val + 1) * batch_size),
                    ),
                )
                is_last = (iter_val * batch_size) >= len(df)
                all_is_last = snowflake_writer_append_table(
                    writer, table, col_meta, is_last, iter_val, None, ctas_meta
                )
                iter_val += 1

        def read_impl(conn):
            df = pd.read_sql(f"select * from {table_name}", conn)
            return df

        bodo.jit(write_impl)(conn, df)
        if bodo.get_rank() == 0:
            # Check the output columns type
            table_info = pd.read_sql(f"DESCRIBE TABLE {table_name}", conn)
            remote_column_type = table_info[table_info["name"] == "B"]["type"].iloc[0]
            assert remote_column_type == column_type.upper()
            if "e" in df.columns and "d" in df.columns:
                table_info = pd.read_sql(
                    f"SELECT TYPEOF(e:a), TYPEOF(d:a) from {table_name}", conn
                )
                assert table_info["TYPEOF(E:A)"].str.contains("DATE").any()
                assert table_info["TYPEOF(D:A)"].str.contains("TIMESTAMP").any()
        check_func(
            read_impl,
            (conn,),
            py_output=expected_df,
            sort_output=True,
            reset_index=True,
            only_1DVar=True,
            convert_columns_to_pandas=True,
        )


def test_write_with_string_precision(memory_leak_check):
    """
    Tests streaming write using string column precisions to specify the maximum
    number of bytes for each column.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.snowflake_write import (
        snowflake_writer_append_table,
        snowflake_writer_init,
    )

    global_1 = bodo.utils.typing.ColNamesMetaType(("A", "B", "C"))
    global_2 = bodo.utils.typing.MetaType((0, 1, 2))
    global_3 = bodo.utils.typing.MetaType((8, -1, 3))
    # Column B is written without any precision info, so we default to the maximum
    expected_types = [
        "VARCHAR(8)",
        "VARCHAR(16777216)",
        "VARCHAR(3)",
    ]
    snowflake_user = 1
    conn = bodo.tests.utils.get_snowflake_connection_string(
        "TEST_DB", "PUBLIC", user=snowflake_user
    )
    bodo_tablename = "BODO_TEST_SNOWFLAKE_WRITE_VARCHAR_PRECISION"
    old_use_put = bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT
    try:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = False
        with ensure_clean_snowflake_table(conn, bodo_tablename) as table_name:
            ctas_meta = bodo.utils.typing.CreateTableMetaType()

            def impl(conn):
                A1 = bodo.utils.conversion.make_replicated_array("ALPHABET", 50)
                A2 = bodo.utils.conversion.make_replicated_array(
                    "We Attack At Dawn", 50
                )
                A3 = bodo.utils.conversion.make_replicated_array("xyz", 50)
                df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                    (A1, A2, A3),
                    bodo.hiframes.pd_index_ext.init_range_index(0, 50, 1, None),
                    global_1,
                )
                T = bodo.hiframes.table.logical_table_to_table(
                    bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
                    (),
                    global_2,
                    3,
                )
                writer = snowflake_writer_init(
                    -1,
                    conn,
                    table_name,
                    "PUBLIC",
                    "replace",
                    "",
                )
                all_is_last = False
                iter_val = 0
                while not all_is_last:
                    all_is_last = snowflake_writer_append_table(
                        writer, T, global_1, True, iter_val, global_3, ctas_meta
                    )
                    iter_val += 1

            bodo.jit(cache=False)(impl)(conn)
            table_info = pd.read_sql(f"DESCRIBE TABLE {table_name}", conn)
            # Column B was written without any precision info, so we default to the maximum
            assert table_info["type"].to_list() == expected_types
    finally:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = old_use_put


def test_write_with_timestamp_time_precision(memory_leak_check):
    """
    Tests streaming write using timestamp/time column precisions to specify the
    precision for each column.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.snowflake_write import (
        snowflake_writer_append_table,
        snowflake_writer_init,
    )

    global_1 = bodo.utils.typing.ColNamesMetaType(("A", "B", "C", "D", "E", "F"))
    global_2 = bodo.utils.typing.MetaType((0, 1, 2, 3, 4, 5))
    # Note: BodoSQL shouldn't generate -1, but we check just in case.
    global_3 = bodo.utils.typing.MetaType((-1, 0, -1, 6, -1, 6))
    # Column B is written without any precision info, so we default to the maximum
    expected_types = [
        "TIMESTAMP_NTZ(9)",
        "TIMESTAMP_NTZ(0)",
        "TIMESTAMP_LTZ(9)",
        "TIMESTAMP_LTZ(6)",
        "TIME(9)",
        "TIME(6)",
    ]
    snowflake_user = 1
    conn = bodo.tests.utils.get_snowflake_connection_string(
        "TEST_DB", "PUBLIC", user=snowflake_user
    )
    bodo_tablename = "BODO_TEST_SNOWFLAKE_WRITE_VARCHAR_PRECISION"
    old_use_put = bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT
    try:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = False
        with ensure_clean_snowflake_table(conn, bodo_tablename) as table_name:
            ctas_meta = bodo.utils.typing.CreateTableMetaType()

            def impl(conn):
                A1 = bodo.utils.conversion.make_replicated_array(
                    pd.Timestamp("01-11-2023 04:23:15"), 50
                )
                A2 = bodo.utils.conversion.make_replicated_array(
                    pd.Timestamp("01-11-2023"), 50
                )
                A3 = bodo.utils.conversion.make_replicated_array(
                    pd.Timestamp("01-11-2023 04:23:15", tz="US/Pacific"), 50
                )
                A4 = bodo.utils.conversion.make_replicated_array(
                    pd.Timestamp("01-11-2023", tz="US/Pacific"), 50
                )
                A5 = bodo.utils.conversion.make_replicated_array(
                    bodo.types.Time(2, 3, 4), 50
                )
                A6 = bodo.utils.conversion.make_replicated_array(
                    bodo.types.Time(2, 3, 4, 5, precision=6), 50
                )
                df = bodo.hiframes.pd_dataframe_ext.init_dataframe(
                    (A1, A2, A3, A4, A5, A6),
                    bodo.hiframes.pd_index_ext.init_range_index(0, 50, 1, None),
                    global_1,
                )
                T = bodo.hiframes.table.logical_table_to_table(
                    bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
                    (),
                    global_2,
                    6,
                )
                writer = snowflake_writer_init(
                    -1,
                    conn,
                    table_name,
                    "PUBLIC",
                    "replace",
                    "",
                )
                all_is_last = False
                iter_val = 0
                while not all_is_last:
                    all_is_last = snowflake_writer_append_table(
                        writer, T, global_1, True, iter_val, global_3, ctas_meta
                    )
                    iter_val += 1
                return df

            expected_df = bodo.jit(cache=False)(impl)(conn)
            expected_df.columns = map(str.lower, expected_df.columns)
            table_info = pd.read_sql(f"DESCRIBE TABLE {table_name}", conn)
            # Column B was written without any precision info, so we default to the maximum
            assert table_info["type"].to_list() == expected_types

            def read_impl(conn, table_name):
                df = pd.read_sql(f"select * from {table_name}", conn)
                return df

            check_func(
                read_impl,
                (conn, table_name),
                py_output=expected_df,
                sort_output=True,
                reset_index=True,
                only_1DVar=True,
            )

    finally:
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = old_use_put


@pytest_mark_one_rank
def test_decimal_sub_38_precision_write(memory_leak_check):
    """
    Tests that reading and writing a number column that requires > int64 but
    is smaller than Number(38, 0) is read and written correctly by Bodo.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.snowflake_write import snowflake_writer_init

    snowflake_user = 1
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = bodo.tests.utils.get_snowflake_connection_string(
        db, schema, user=snowflake_user
    )
    bodo_write_tablename = "BODO_TEST_SNOWFLAKE_WRITE_DECIMAL_25_PRECISION"
    with ensure_clean_snowflake_table(conn, bodo_write_tablename) as write_table_name:
        bodo_read_tablename = "BODO_TEST_SNOWFLAKE_READ_DECIMAL_25_PRECISION"
        select_query = "Select 9999999999999999999999999::NUMBER(25, 0) as A from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION"
        with create_snowflake_table_from_select_query(
            select_query, bodo_read_tablename, db, schema
        ) as read_table_name:
            global_1 = bodo.utils.typing.ColNamesMetaType(("A",))
            ctas_meta = bodo.utils.typing.CreateTableMetaType()

            def impl(conn):
                reader = pd.read_sql(
                    read_table_name,
                    conn,
                    _bodo_is_table_input=True,
                    _bodo_chunksize=4096,
                    _bodo_read_as_table=True,
                )

                writer = snowflake_writer_init(
                    -1,
                    conn,
                    write_table_name,
                    "PUBLIC",
                    "replace",
                    "",
                )
                _iter_1 = 0
                _temp22 = False
                while not (_temp22):
                    (
                        T1,
                        __bodo_is_last_streaming_output_1,
                    ) = bodo.io.arrow_reader.read_arrow_next(reader, True)
                    _temp22 = bodo.io.snowflake_write.snowflake_writer_append_table(
                        writer,
                        T1,
                        global_1,
                        __bodo_is_last_streaming_output_1,
                        _iter_1,
                        None,
                        ctas_meta,
                    )
                    _iter_1 = _iter_1 + 1
                bodo.io.arrow_reader.arrow_reader_del(reader)

            bodo.jit(cache=False)(impl)(conn)
            table_info = pd.read_sql(f"DESCRIBE TABLE {read_table_name}", conn)
            expected_types = ["NUMBER(25,0)"]
            assert table_info["type"].to_list() == expected_types, (
                "Incorrect type found"
            )
            # Verify the result of an except query with these tables is empty
            result = pd.read_sql(
                f"SELECT * FROM {write_table_name} EXCEPT SELECT * FROM {read_table_name}",
                conn,
            )
            assert len(result) == 0, "Unexpected difference in tables"


def test_logged_queryid_write(memory_leak_check):
    """Test query id is printed in write step when verbose logging is set to 2"""
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    def impl(df, table_name, conn):
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        return "DONE"

    df = pd.DataFrame({"A": [1.12, 1.1] * 5, "B": [213, -7] * 5})
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        with ensure_clean_snowflake_table(conn) as table_name:
            bodo.jit(impl)(df, table_name, conn)

        check_logger_msg(stream, "Snowflake Query Submission (Write)")


@pytest_mark_one_rank
def test_aborted_detached_query(memory_leak_check):
    """
    Test that our snowflake connection always has ABORT_DETACHED_QUERY = False
    in a user's account. This is not a write test, but it is grouped here because
    if this is True it can cause write to fail.
    """
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    df = pd.read_sql("show parameters like 'ABORT_DETACHED_QUERY'", conn)
    old_value = df["value"].iloc[0]
    try:
        pd.read_sql("alter user set ABORT_DETACHED_QUERY=True", conn)
        # Test the values of the session from snowflake_connect
        connection = bodo.io.snowflake.snowflake_connect(conn)
        cur = connection.cursor()
        cur.execute("show parameters like 'ABORT_DETACHED_QUERY'")
        rows = cur.fetchall()
        result = rows[0][1]
        assert result.lower() == "false", (
            "ABORT_DETACHED_QUERY not set to False by snowflake_connect()"
        )
    finally:
        pd.read_sql(f"alter user set ABORT_DETACHED_QUERY={old_value}", conn)


def test_create_table_with_comments(memory_leak_check):
    """
    Tests using streaming Snowflake write functions to create a new table
    with table comments and column comments.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.snowflake_write import snowflake_writer_init

    if bodo.get_size() != 1:
        pytest.skip("This test is only designed for 1 rank")

    snowflake_user = 1
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = bodo.tests.utils.get_snowflake_connection_string(
        db, schema, user=snowflake_user
    )
    bodo_write_tablename = "BODO_TEST_SNOWFLAKE_CTAS_COMMENT"
    # Write the table to Snowflake using a table name that is created by the context
    # manager and will be deleted at the end
    with ensure_clean_snowflake_table(conn, bodo_write_tablename) as write_table_name:
        name_global = bodo.utils.typing.ColNamesMetaType(("A", "B", "C"))
        keep_global = bodo.utils.typing.MetaType((0, 1, 2))
        precisions_global = bodo.utils.typing.MetaType((38, 1, -1))
        ctas_global = bodo.utils.typing.CreateTableMetaType(
            table_comment="Records of notable events",
            column_comments=("Index of events", None, "Date of notable event"),
        )

        def impl_write(df, conn):
            writer = snowflake_writer_init(
                -1,
                conn,
                write_table_name,
                "PUBLIC",
                "replace",
                "",
            )
            T = bodo.hiframes.table.logical_table_to_table(
                bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df),
                (),
                keep_global,
                3,
            )
            bodo.io.snowflake_write.snowflake_writer_append_table(
                writer,
                T,
                name_global,
                True,
                0,
                precisions_global,
                ctas_global,
            )

        df = pd.DataFrame(
            {
                "A": [0, 1, 2, 3, 4],
                "B": ["A", "B", "C", "D", "E"],
                "C": [datetime.date.fromordinal(738800 + i**3) for i in range(2, 7)],
            }
        )
        bodo.jit(cache=False)(impl_write)(df, conn)

        # Request information about all of the columns of the table, then confirm
        # that the desired ones have the specified comments.
        column_info = pd.read_sql(f"DESCRIBE TABLE {write_table_name}", conn)
        assert len(column_info == 3), "Wrong number of column attributes"
        assert column_info["type"].to_list() == [
            "NUMBER(38,0)",
            "VARCHAR(1)",
            "DATE",
        ], "Wrong column types"
        assert column_info["null?"].to_list() == [
            "Y",
            "Y",
            "Y",
        ], "Wrong column nullability"
        assert column_info["comment"].to_list() == [
            "Index of events",
            np.nan,
            "Date of notable event",
        ], "Wrong column comments"

        # Request information about the overall table, then confirm that it has
        # the desired comment.
        table_info = pd.read_sql(
            f"SHOW TABLES LIKE '{write_table_name}' IN SCHEMA PUBLIC", conn
        )
        assert len(table_info == 1), "Wrong number of tables found"
        assert table_info["rows"][0] == 5, "Wrong number of rows"
        assert table_info["comment"][0] == "Records of notable events", (
            "Wrong table comment"
        )
