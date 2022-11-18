# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests for writing to Snowflake using Python APIs
"""
import os
import random
import string
import traceback
import uuid
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.utils import (
    get_snowflake_connection_string,
    get_start_end,
    reduce_sum,
)

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
        cursor, stage_name, location, df_columns, if_exists
    ):
        with bodo.objmode():
            create_table_handle_exists(
                cursor, stage_name, location, df_columns, if_exists
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
        bodo_impl(cursor, stage_name, table_name, df_in.columns, "fail")
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
            bodo_impl(cursor, stage_name, table_name, df_in.columns, "fail")
    bodo.barrier()

    # Step 3: Call create_table_handle_exists again with if_exists="append".
    # This should succeed and keep the same table from Step 1.
    if bodo.get_rank() == 0:
        bodo_impl(cursor, stage_name, table_name, df_in.columns, "append")
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
        bodo_impl(cursor, stage_name, table_name, df_in.columns, "replace")
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

    def test_impl_execute_copy_into(cursor, stage_name, location, df_columns):
        with bodo.objmode(
            nsuccess="int64", nchunks="int64", nrows="int64", output="unicode_type"
        ):
            nsuccess, nchunks, nrows, output = execute_copy_into(
                cursor, stage_name, location, df_columns
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
            num_success, num_chunks, num_rows, _ = bodo_impl(
                cursor, stage_name, table_name, df_in.columns
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
