import os
import shutil
from contextlib import contextmanager

import pandas as pd

import bodo


@contextmanager
def ensure_clean(filename):
    """deletes filename if exists after test is done."""
    try:
        yield
    finally:
        try:
            # wait for all ranks to complete
            bodo.barrier()
            # delete on rank 0
            if (
                bodo.get_rank() == 0
                and os.path.exists(filename)
                and os.path.isfile(filename)
            ):
                os.remove(filename)
        except Exception as e:
            print(f"Exception on removing file: {e}")


@contextmanager
def ensure_clean_dir(dirname):
    """deletes filename if exists after test is done."""
    try:
        yield
    finally:
        try:
            # wait for all ranks to complete
            bodo.barrier()
            # delete on rank 0
            if (
                bodo.get_rank() == 0
                and os.path.exists(dirname)
                and os.path.isdir(dirname)
            ):
                shutil.rmtree(dirname)
        except Exception as e:
            print(f"Exception on removing directory: {e}")


@contextmanager
def ensure_clean2(pathname):  # pragma: no cover
    """deletes pathname if exists after test is done."""
    try:
        yield
    finally:
        bodo.barrier()
        if bodo.get_rank() == 0:
            try:
                if os.path.exists(pathname) and os.path.isfile(pathname):
                    os.remove(pathname)
            except Exception as e:
                print(f"Exception on removing file: {e}")
            try:
                if os.path.exists(pathname) and os.path.isdir(pathname):
                    shutil.rmtree(pathname)
            except Exception as e:
                print(f"Exception on removing directory: {e}")


@contextmanager
def ensure_clean_mysql_psql_table(conn, table_name_prefix="test_small_table"):
    """
    Context Manager that creates a unique table name,
    and then drops the table with that name (if one exists)
    after the test is done.

    Args:
        conn (str): connection string
        table_name_prefix (str; optional): Prefix for the
            table name to generate. Default: "test_small_table"
    """
    import uuid

    from mpi4py import MPI
    from sqlalchemy import create_engine, text

    comm = MPI.COMM_WORLD

    try:
        table_name = None
        if bodo.get_rank() == 0:
            # Add a uuid to avoid potential conflict as this may be running in
            # several different CI sessions at once. This may be the source of
            # sporadic failures (although this is uncertain).
            # We do `.hex` since we don't want `-`s in the name.
            table_name = f"{table_name_prefix}_{uuid.uuid4().hex}"
        table_name = comm.bcast(table_name)
        yield table_name
    finally:
        # Drop the temporary table (if one was created) to avoid accumulating
        # too many tables in the database
        bodo.barrier()
        drop_err = None
        if bodo.get_rank() == 0:
            try:
                engine = create_engine(conn)
                connection = engine.connect()
                connection.execute(text(f"drop table if exists {table_name}"))
            except Exception as e:
                drop_err = e
        drop_err = comm.bcast(drop_err)
        if isinstance(drop_err, Exception):
            raise drop_err


@contextmanager
def ensure_clean_snowflake_table(conn, table_name_prefix="test_table", parallel=True):
    """
    Context Manager that creates a unique table name,
    and then drops the table with that name (if one exists)
    after the test is done.

    Args:
        conn (str): connection string
        table_name_prefix (str; optional): Prefix for the
            table name to generate. Default: "test_small_table"
        parallel (bool; optional): method called by all ranks. Default: True
    """
    import uuid

    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    try:
        table_name = None
        if bodo.get_rank() == 0 or (not parallel):
            # Add a uuid to avoid potential conflict as this may be running in
            # several different CI sessions at once. This may be the source of
            # sporadic failures (although this is uncertain).
            # We do `.hex` since we don't want `-`s in the name.
            # `.upper()` to avoid case sensitivity issues.
            table_name = f"{table_name_prefix}_{uuid.uuid4().hex}".upper()
        if parallel:
            table_name = comm.bcast(table_name)
        yield table_name
    finally:
        # Drop the temporary table (if one was created) to avoid accumulating
        # too many tables in Snowflake
        if parallel:
            bodo.barrier()
        drop_err = None
        if bodo.get_rank() == 0 or (not parallel):
            try:
                pd.read_sql(f"drop table if exists {table_name}", conn)
            except Exception as e:
                drop_err = e
        if parallel:
            drop_err = comm.bcast(drop_err)
        if isinstance(drop_err, Exception):
            raise drop_err
