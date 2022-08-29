# Copyright (C) 2022 Bodo Inc. All rights reserved.
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import bodo

cwd = Path(__file__).resolve().parent
datadir = cwd.parent / "tests" / "data"


@bodo.jit
def get_rank():  # pragma: no cover
    return bodo.libs.distributed_api.get_rank()


@bodo.jit
def barrier():  # pragma: no cover
    return bodo.libs.distributed_api.barrier()


@contextmanager
def ensure_clean(filename):
    """deletes filename if exists after test is done."""
    try:
        yield
    finally:
        try:
            # wait for all ranks to complete
            barrier()
            # delete on rank 0
            if (
                get_rank() == 0
                and os.path.exists(filename)
                and os.path.isfile(filename)
            ):
                os.remove(filename)
        except Exception as e:
            print("Exception on removing file: {error}".format(error=e))


@contextmanager
def ensure_clean_dir(dirname):
    """deletes filename if exists after test is done."""
    try:
        yield
    finally:
        try:
            # wait for all ranks to complete
            barrier()
            # delete on rank 0
            if get_rank() == 0 and os.path.exists(dirname) and os.path.isdir(dirname):
                shutil.rmtree(dirname)
        except Exception as e:
            print("Exception on removing directory: {error}".format(error=e))


@contextmanager
def ensure_clean2(pathname):  # pragma: no cover
    """deletes pathname if exists after test is done."""
    try:
        yield
    finally:
        barrier()
        if get_rank() == 0:
            try:
                if os.path.exists(pathname) and os.path.isfile(pathname):
                    os.remove(pathname)
            except Exception as e:
                print("Exception on removing file: {error}".format(error=e))
            try:
                if os.path.exists(pathname) and os.path.isdir(pathname):
                    shutil.rmtree(pathname)
            except Exception as e:
                print("Exception on removing directory: {error}".format(error=e))


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
    from sqlalchemy import create_engine

    comm = MPI.COMM_WORLD

    try:
        table_name = None
        if bodo.get_rank() == 0:
            # Add a uuid to avoid potential conflict as this may be running in
            # several different CI sessions at once. This may be the source of
            # sporadic failures (although this is uncertain).
            table_name = f"{table_name_prefix}_{uuid.uuid4()}"
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
                connection.execute(f"drop table if exists `{table_name}`")
            except Exception as e:
                drop_err = e
        drop_err = comm.bcast(drop_err)
        if isinstance(drop_err, Exception):
            raise drop_err
