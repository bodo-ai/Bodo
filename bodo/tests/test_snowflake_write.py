# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests for writing to Snowflake using Python APIs
"""
import os
import random
import string
import traceback
import uuid
from decimal import Decimal
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.utils import (
    _get_dist_arg,
    check_func,
    get_snowflake_connection_string,
    get_start_end,
    reduce_sum,
    snowflake_cred_env_vars_present,
)
from bodo.utils.testing import ensure_clean_snowflake_table
from bodo.utils.typing import BodoWarning

# ---------------Distributed Snowflake Write Unit Tests ------------------


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
@pytest.mark.parametrize("is_temporary", [True, False])
def test_snowflake_write_create_internal_stage(is_temporary, memory_leak_check):
    """
    Tests creating an internal stage within Snowflake
    """
    from bodo.io.snowflake import create_internal_stage, snowflake_connect

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_create_internal_stage(cursor):
        with bodo.objmode(stage_name="unicode_type"):
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
                f"SHOW STAGES "
                f"/* Python:bodo.tests.test_sql:test_snowflake_create_internal_stage() */"
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
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_create_internal_stage() */"
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()
    cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_create_internal_stage failed"
    bodo.barrier()


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
@pytest.mark.parametrize("is_temporary", [True, False])
def test_snowflake_write_drop_internal_stage(is_temporary, memory_leak_check):
    """
    Tests dropping an internal stage within Snowflake
    """
    from bodo.io.snowflake import drop_internal_stage, snowflake_connect

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_drop_internal_stage(cursor, stage_name):
        with bodo.objmode():
            drop_internal_stage(cursor, stage_name)

    bodo_impl = bodo.jit(distributed=False)(test_impl_drop_internal_stage)

    # Create stage
    stage_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        stage_name = f"bodo_test_sql_{uuid.uuid4()}"
        create_stage_sql = (
            f'CREATE {"TEMPORARY " if is_temporary else ""}STAGE "{stage_name}" '
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_drop_internal_stage() */"
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
                f"SHOW STAGES "
                f"/* Python:bodo.tests.test_sql:test_snowflake_drop_internal_stage() */"
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
            f"/* Python:bodo.tests.test_sql:test_snowflake_drop_internal_stage() */"
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()
        cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_drop_internal_stage failed"
    bodo.barrier()


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_write_do_upload_and_cleanup(memory_leak_check):
    """
    Tests uploading files to Snowflake internal stage using PUT command
    """
    from bodo.io.snowflake import do_upload_and_cleanup, snowflake_connect

    db = "TEST_DB"
    schema = "SNOWFLAKE_WRITE_TEST"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_do_upload_and_cleanup(cursor, chunk_path, stage_name):
        with bodo.objmode():
            th = do_upload_and_cleanup(cursor, 0, chunk_path, stage_name)
            th.join()

    bodo_impl = bodo.jit()(test_impl_do_upload_and_cleanup)

    # Set up schema and internal stage
    if bodo.get_rank() == 0:
        create_schema_sql = (
            f'CREATE OR REPLACE SCHEMA "{schema}" '
            f"/* Python:bodo.tests.test_sql:test_snowflake_do_upload_and_cleanup() */"
        )
        cursor.execute(create_schema_sql, _is_internal=True).fetchall()

    bodo.barrier()

    stage_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        stage_name = f"bodo_test_sql_{uuid.uuid4()}"
        create_stage_sql = (
            f'CREATE STAGE "{stage_name}" '
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_do_upload_and_cleanup() */"
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
            f"/* Python:bodo.tests.test_sql:test_snowflake_do_upload_and_cleanup() */"
        )
        listing = cursor.execute(list_stage_sql, _is_internal=True).fetchall()
        assert len(listing) == npes

        # Use GET to fetch all uploaded files
        with TemporaryDirectory() as tmp_folder:
            get_stage_sql = (
                f"GET @\"{stage_name}\" 'file://{tmp_folder}' "
                f"/* Python:bodo.tests.test_sql:test_snowflake_do_upload_and_cleanup() */"
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
            f"/* Python:bodo.tests.test_sql:test_snowflake_do_upload_and_cleanup() */"
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()

    cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_do_upload_and_cleanup failed"
    bodo.barrier()


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_write_create_table_handle_exists(memory_leak_check):
    """
    Test Snowflake write table creation, both with and without a pre-existing table
    """
    from bodo.io.snowflake import create_table_handle_exists, snowflake_connect

    db = "TEST_DB"
    schema = "SNOWFLAKE_WRITE_TEST"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_create_table_handle_exists(
        cursor, stage_name, location, sf_schema, if_exists
    ):
        with bodo.objmode():
            create_table_handle_exists(
                cursor, stage_name, location, sf_schema, if_exists
            )

    bodo_impl = bodo.jit(distributed=False)(test_impl_create_table_handle_exists)

    # Set up schema, internal stage, and table name
    if bodo.get_rank() == 0:
        create_schema_sql = (
            f'CREATE OR REPLACE SCHEMA "{schema}" '
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
        )
        cursor.execute(create_schema_sql, _is_internal=True).fetchall()

    bodo.barrier()

    stage_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        stage_name = f"bodo_test_sql_{uuid.uuid4()}"
        create_stage_sql = (
            f'CREATE STAGE "{stage_name}" '
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
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

        upload_put_sql = (
            f"PUT 'file://{df_path}' @\"{stage_name}\" AUTO_COMPRESS=FALSE "
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
        )
        cursor.execute(upload_put_sql, _is_internal=True)

    # Step 1: Call create_table_handle_exists.
    # This should succeed as the table doesn't exist yet.
    if bodo.get_rank() == 0:
        sf_schema = bodo.io.snowflake.gen_snowflake_schema(
            df_in.columns, bodo.typeof(df_in).data
        )
        bodo_impl(cursor, stage_name, table_name, sf_schema, "fail")
    bodo.barrier()
    passed = 1

    first_table_creation_time = None  # Forward declaration
    if bodo.get_rank() == 0:
        try:
            show_tables_sql = (
                f"""SHOW TABLES STARTS WITH '{table_name.strip('"')}' """
                f"/* Python:bodo.tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
            )
            tables_desc = cursor.execute(show_tables_sql, _is_internal=True).fetchall()
            first_table_creation_time = tables_desc[0]

            describe_table_columns_sql = (
                f"DESCRIBE TABLE {table_name} TYPE=COLUMNS "
                f"/* Python:bodo.tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
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
            bodo_impl(cursor, stage_name, table_name, sf_schema, "fail")
    bodo.barrier()

    # Step 3: Call create_table_handle_exists again with if_exists="append".
    # This should succeed and keep the same table from Step 1.
    if bodo.get_rank() == 0:
        bodo_impl(cursor, stage_name, table_name, sf_schema, "append")
    bodo.barrier()

    if bodo.get_rank() == 0:
        try:
            show_tables_sql = (
                f"""SHOW TABLES STARTS WITH '{table_name.strip('"')}' """
                f"/* Python:bodo.tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
            )
            tables_desc = cursor.execute(show_tables_sql, _is_internal=True).fetchall()
            assert tables_desc[0] == first_table_creation_time

        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    bodo.barrier()

    # Step 4: Call create_table_handle_exists again with if_exists="replace".
    # This should succeed after dropping the table from Step 1.
    if bodo.get_rank() == 0:
        bodo_impl(cursor, stage_name, table_name, sf_schema, "replace")
    bodo.barrier()

    if bodo.get_rank() == 0:
        try:
            show_tables_sql = (
                f"""SHOW TABLES STARTS WITH '{table_name.strip('"')}' """
                f"/* Python:bodo.tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
            )
            tables_desc = cursor.execute(show_tables_sql, _is_internal=True).fetchall()
            assert tables_desc[0] != first_table_creation_time

        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    bodo.barrier()

    if bodo.get_rank() == 0:
        cleanup_stage_sql = (
            f'DROP STAGE IF EXISTS "{stage_name}" '
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()

        cleanup_table_sql = (
            f"DROP TABLE IF EXISTS {table_name} "
            f"/* Python:bodo.tests.test_sql:test_snowflake_create_table_handle_exists() */"
        )
        cursor.execute(cleanup_table_sql, _is_internal=True).fetchall()

    cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_write_create_table_handle_exists failed"
    bodo.barrier()


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_write_execute_copy_into(memory_leak_check):
    """
    Tests executing COPY_INTO into a Snowflake table from internal stage
    """
    from bodo.io.snowflake import execute_copy_into, snowflake_connect

    db = "TEST_DB"
    schema = "SNOWFLAKE_WRITE_TEST"
    conn = get_snowflake_connection_string(db, schema)
    cursor = snowflake_connect(conn).cursor()

    comm = MPI.COMM_WORLD

    def test_impl_execute_copy_into(cursor, stage_name, location, sf_schema):
        with bodo.objmode(
            nsuccess="int64", nchunks="int64", nrows="int64", output="unicode_type"
        ):
            nsuccess, nchunks, nrows, output = execute_copy_into(
                cursor, stage_name, location, sf_schema
            )
            output = repr(output)
        return nsuccess, nchunks, nrows, output

    bodo_impl = bodo.jit()(test_impl_execute_copy_into)

    # Set up schema and internal stage
    if bodo.get_rank() == 0:
        create_schema_sql = (
            f'CREATE OR REPLACE SCHEMA "{schema}" '
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_execute_copy_into() */ "
        )
        cursor.execute(create_schema_sql, _is_internal=True).fetchall()

    bodo.barrier()

    stage_name = None  # Forward declaration
    if bodo.get_rank() == 0:
        stage_name = f"bodo_test_sql_{uuid.uuid4()}"
        create_stage_sql = (
            f'CREATE STAGE "{stage_name}" '
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_execute_copy_into() */ "
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

        upload_put_sql = (
            f"PUT 'file://{df_path}' @\"{stage_name}\" AUTO_COMPRESS=FALSE "
            f"/* Python:bodo.tests.test_sql.test_snowflake_write_execute_copy_into() */ "
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
            num_success, num_chunks, num_rows, _ = bodo_impl(
                cursor, stage_name, table_name, sf_schema
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
                f"/* Python:bodo.tests.test_sql:test_snowflake_write_execute_copy_into() */ "
            )
            df = cursor.execute(select_sql, _is_internal=True).fetchall()
            df_load = pd.DataFrame(df, columns=df_in.columns)

            # Row order isn't defined, so sort the data.
            df_in_cols = df_in.columns.to_list()
            df_in_sort = df_in.sort_values(by=df_in_cols).reset_index(drop=True)
            df_load_cols = df_load.columns.to_list()
            df_load_sort = df_load.sort_values(by=df_load_cols).reset_index(drop=True)

            pd.testing.assert_frame_equal(
                df_in_sort, df_load_sort, check_column_type=False
            )

        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    bodo.barrier()

    if bodo.get_rank() == 0:
        cleanup_stage_sql = (
            f'DROP STAGE IF EXISTS "{stage_name}" '
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_execute_copy_into() */ "
        )
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()

        cleanup_table_sql = (
            f"DROP TABLE IF EXISTS {table_name} "
            f"/* Python:bodo.tests.test_sql:test_snowflake_write_execute_copy_into() */ "
        )
        cursor.execute(cleanup_table_sql, _is_internal=True).fetchall()

    cursor.close()

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_write_execute_copy_into failed"
    bodo.barrier()


def test_snowflake_write_join_all_threads(memory_leak_check):
    """
    Test that joining all threads will broadcast exceptions raised on any individual rank
    """
    from bodo.io.helpers import ExceptionPropagatingThread, join_all_threads

    def thread_target_success(s):
        return s

    def thread_target_failure(s):
        raise ValueError(s)

    # All threads succeed
    @bodo.jit
    def test_join_all_threads_impl_1():
        thread_list = []
        for i in range(4):
            with bodo.objmode(th="exception_propagating_thread_type"):
                th = ExceptionPropagatingThread(
                    target=thread_target_success,
                    args=(f"rank{bodo.get_rank()}_thread{i}",),
                )
                th.start()
            thread_list.append(th)

        with bodo.objmode():
            join_all_threads(thread_list)

    test_join_all_threads_impl_1()

    # Threads 1 and 3 fail on every rank.
    # Each rank should raise its own exception
    def test_join_all_threads_impl_2():
        thread_list = []
        for i in range(4):
            with bodo.objmode(th="exception_propagating_thread_type"):
                if i == 1 or i == 3:
                    thread_target = thread_target_failure
                else:
                    thread_target = thread_target_success

                th = ExceptionPropagatingThread(
                    target=thread_target, args=(f"rank{bodo.get_rank()}_thread{i}",)
                )
                th.start()
            thread_list.append(th)

        with bodo.objmode():
            join_all_threads(thread_list)

    err_msg = f"rank{bodo.get_rank()}_thread1"
    with pytest.raises(ValueError, match=err_msg):
        test_join_all_threads_impl_2()

    # Threads 0 and 3 fail only on rank 0
    # Rank 0 should raise its own exception, and all other ranks should raise Rank 0's
    def test_join_all_threads_impl_3():
        thread_list = []
        for i in range(4):
            with bodo.objmode(th="exception_propagating_thread_type"):
                if bodo.get_rank() == 0 and (i == 0 or i == 3):
                    thread_target = thread_target_failure
                else:
                    thread_target = thread_target_success

                th = ExceptionPropagatingThread(
                    target=thread_target, args=(f"rank{bodo.get_rank()}_thread{i}",)
                )
                th.start()
            thread_list.append(th)

        with bodo.objmode():
            join_all_threads(thread_list)

    err_msg = f"rank0_thread0"
    with pytest.raises(ValueError, match=err_msg):
        test_join_all_threads_impl_3()


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
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

    with pytest.raises(RuntimeError, match="Failed to connect to DB"):
        impl(
            "snowflake://SF_USERNAME:SF_PASSWORD@sf_account/database/PUBLIC?warehouse=warehouse"
        )


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
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
        output_df = pd.read_sql(f"select * from {table_name}", conn_str)
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
        output_df.reset_index(drop=True),
        df.reset_index(drop=True),
        check_names=False,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
@pytest.mark.parametrize("df_size", [17000 * 3, 2])
@pytest.mark.parametrize("sf_write_overlap", [True, False])
@pytest.mark.parametrize("sf_write_use_put", [True, False])
@pytest.mark.parametrize("snowflake_user", [1, pytest.param(3, marks=pytest.mark.slow)])
def test_to_sql_snowflake(
    df_size, sf_write_overlap, sf_write_use_put, snowflake_user, memory_leak_check
):
    """
    Tests that df.to_sql works as expected. Since Snowflake has a limit of ~16k
    per insert, we insert 17k rows per rank to emulate a "large" insert.
    """

    # Skip test if env vars with snowflake creds required for this test are not set.
    if not snowflake_cred_env_vars_present(snowflake_user):
        pytest.skip(reason="Required env vars with Snowflake credentials not set")

    if (
        ("AGENT_NAME" in os.environ)
        and sf_write_overlap
        and sf_write_use_put
        and (snowflake_user == 3)
    ):
        pytest.skip(
            reason="[BE-3758] This test fails on Azure pipelines for a yet to be determined reason."
        )

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema, user=snowflake_user)

    import platform

    import bodo
    import bodo.io.snowflake

    # Specify Snowflake write hyperparameters
    old_sf_write_overlap = bodo.io.snowflake.SF_WRITE_OVERLAP_UPLOAD
    bodo.io.snowflake.SF_WRITE_OVERLAP_UPLOAD = sf_write_overlap
    old_sf_write_use_put = bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT
    bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = sf_write_use_put

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

    try:
        with ensure_clean_snowflake_table(conn) as name:
            # If using a Azure Snowflake account, the stage will be ADLS backed, so
            # when writing to it directly, we need a proper ADLS/Haddop setup
            # and the bodo_azurefs_sas_token_provider library, both of which are only
            # done for Linux. So we verify that it shows user the appropriate warning
            # about falling back to the PUT method.
            if (
                snowflake_user == 3
                and (not sf_write_use_put)
                and (platform.system() != "Linux")
            ):
                if bodo.get_rank() == 0:
                    # Warning is only raised on rank 0
                    with pytest.warns(
                        BodoWarning,
                        match="Falling back to PUT command for upload for now.",
                    ):
                        test_write(df, name, conn, schema)
                else:
                    test_write(df, name, conn, schema)
            else:
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
        bodo.io.snowflake.SF_WRITE_OVERLAP_UPLOAD = old_sf_write_overlap
        bodo.io.snowflake.SF_WRITE_UPLOAD_USING_PUT = old_sf_write_use_put


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_to_sql_bodo_datatypes_part1(memory_leak_check):
    """
    Tests that df.to_sql works with all Bodo's supported dataframe datatypes
    This compares the dataframe Bodo writes and reads vs.
    the dataframe Pandas writes and reads
    """

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
            rng.choice([x for x in letters])
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
                rng.choice([1.4, None, 3.4, 2.5], df_len),
                "Float64" if bodo.libs.float_arr_ext._use_nullable_float else "float64",
            ),
            # nan with float
            "NAN_FLOAT_COL": np.array(rng.choice([np.nan, 2.5], df_len)),
            # Date
            "DATE_COL": pd.date_range("2001-01-01", periods=df_len).date,
            # timedelta
            "TIMEDELTA_COL": pd.Series(pd.timedelta_range(start="1 day", periods=7)),
            # TODO: timezone-aware (need support on pq_write to be per column)
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
        bodo_result = bodo.gatherv(bodo_result)

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
            pd.testing.assert_frame_equal(
                bodo_result.sort_values("int8_col").reset_index(drop=True),
                py_output.sort_values("int8_col").reset_index(drop=True),
                check_dtype=False,
            )
        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_to_sql_bodo_datatypes failed"


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
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
                bodo.Time(12, 0, precision=0),
                bodo.Time(1, 1, 3, precision=0),
                bodo.Time(2, precision=0),
                bodo.Time(12, 0, precision=0),
                bodo.Time(5, 6, 3, precision=0),
                bodo.Time(9, precision=0),
                bodo.Time(11, 19, 34, precision=0),
            ],
            # pa.time32("ms")
            "time_col_p3": [
                bodo.Time(12, 0, precision=3),
                bodo.Time(1, 1, 3, precision=3),
                bodo.Time(2, precision=3),
                bodo.Time(12, 0, precision=3),
                bodo.Time(6, 7, 13, precision=3),
                bodo.Time(2, precision=3),
                bodo.Time(17, 1, 3, precision=3),
            ],
            # # pa.time64("us")
            "time_col_p6": [
                bodo.Time(12, 0, precision=6),
                bodo.Time(1, 1, 3, precision=6),
                bodo.Time(2, precision=6),
                bodo.Time(12, 0, precision=6),
                bodo.Time(5, 11, 53, precision=6),
                bodo.Time(2, precision=6),
                bodo.Time(1, 1, 3, precision=6),
            ],
            # pa.time64("ns"). Default precision=9
            "time_col_p9": [
                bodo.Time(12, 0),
                bodo.Time(1, 1, 3),
                bodo.Time(2),
                bodo.Time(12, 1),
                bodo.Time(5, 6, 3),
                bodo.Time(9),
                bodo.Time(11, 19, 34),
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

    bodo_result = bodo.gatherv(bodo_result)

    passed = 1
    if bodo.get_rank() == 0:
        try:
            # Re-write this column with precision=6
            # SF ignores nanoseconds precisions with COPY INTO FROM PARQUET-FILES
            # See for more information
            df["time_col_p9"] = [
                bodo.Time(12, 0, precision=6),
                bodo.Time(1, 1, 3, precision=6),
                bodo.Time(2, precision=6),
                bodo.Time(12, 1, precision=6),
                bodo.Time(5, 6, 3, precision=6),
                bodo.Time(9, precision=6),
                bodo.Time(11, 19, 34, precision=6),
            ]
            # Sort output as in some rare cases data read from SF
            # are in different order from written df.
            pd.testing.assert_frame_equal(
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


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_to_sql_bodo_datatypes_part3(memory_leak_check):
    """
    Tests that df.to_sql works with all Bodo's supported dataframe datatypes
    Compare what Bodo reads vs. Pandas read (Note: Bodo writes these data and
    confirmed it's correct data but with different format)
    """

    bodo_tablename = "BODO_DT_P3"
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    df = pd.DataFrame(
        {
            # list list
            "list_list_col": [
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
            # struct
            "struct_col": np.array(
                [
                    [{"A": 1, "B": 2}, {"A": 10, "B": 20}],
                    [{"A": 3, "B": 4}],
                    [{"A": 5, "B": 6}, {"A": 50, "B": 60}, {"A": 500, "B": 600}],
                    [{"A": 10, "B": 20}, {"A": 100, "B": 200}],
                    [{"A": 30, "B": 40}],
                    [{"A": 50, "B": 60}, {"A": 500, "B": 600}, {"A": 5000, "B": 6000}],
                    [{"A": 30, "B": 40}],
                ]
            ),
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
        if bodo.get_rank() == 0:
            py_output = sf_read(conn, b_tablename)

    bodo_result = bodo.gatherv(bodo_result)

    passed = 1
    if bodo.get_rank() == 0:
        try:
            pd.testing.assert_frame_equal(
                bodo_result,
                py_output,
                check_dtype=False,
            )
        except Exception as e:
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            passed = 0

    n_passed = reduce_sum(passed)
    n_pes = bodo.get_size()
    assert n_passed == n_pes, "test_snowflake_to_sql_bodo_datatypes_part3 failed"


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_to_sql_snowflake_user2(memory_leak_check):
    """
    Tests that df.to_sql works when the Snowflake account password has special
    characters.
    """
    # Only test with one rank because we are just testing access
    if bodo.get_size() != 1:
        return
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


@pytest.mark.skipif("AGENT_NAME" not in os.environ, reason="requires Azure Pipelines")
def test_snowflake_to_sql_colname_case(memory_leak_check):
    """
    Tests that df.to_sql works with different upper/lower case of column name
    """

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
        bodo_result = bodo.gatherv(bodo_result)

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
