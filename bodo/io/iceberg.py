# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
File that contains the main functionality for the Iceberg
integration within the Bodo repo. This does not contain the
main IR transformation.
"""
import os
import re

import bodo
from bodo.utils import tracing
from bodo.utils.typing import BodoError


def get_iceberg_type_info(table_name: str, con: str, database_schema: str):
    """
    Helper function to fetch Bodo types for an
    Iceberg table with the given table name, conn,
    and database_schema.

    Returns:
        - List of column names
        - List of column Bodo types
        - PyArrow Schema Object
    """
    import bodo_iceberg_connector
    import numba.core

    from bodo.io.parquet_pio import _get_numba_typ_from_pa_typ

    try:
        (
            col_names,
            col_types,
            pyarrow_schema,
        ) = bodo_iceberg_connector.get_iceberg_typing_schema(
            con, database_schema, table_name
        )

    except bodo_iceberg_connector.IcebergError as e:
        # Only include Java error info in dev mode because it contains at lot of
        # unnecessary info about internal packages and dependencies.
        if (
            isinstance(e, bodo_iceberg_connector.IcebergJavaError)
            and numba.core.config.DEVELOPER_MODE
        ):  # pragma: no cover
            raise BodoError(f"{e.message}:\n {e.java_error.stacktrace()}")
        else:
            raise BodoError(e.message)

    bodo_types = [
        _get_numba_typ_from_pa_typ(typ, False, True, None)[0] for typ in col_types
    ]

    return (col_names, bodo_types, pyarrow_schema)


def get_iceberg_file_list(table_name: str, conn: str, database_schema: str, filters):
    """
    Gets the list of parquet data files that need to be read
    from an Iceberg table by calling objmode. We also pass
    filters, which is in DNF format and the output of filter
    pushdown to Iceberg. Iceberg will use this information to
    prune any files that it can from just metadata, so this
    is an "inclusive" projection.
    """
    import bodo_iceberg_connector
    import numba.core

    try:
        lst = bodo_iceberg_connector.bodo_connector_get_parquet_file_list(
            conn, database_schema, table_name, filters
        )
    except bodo_iceberg_connector.IcebergError as e:
        # Only include Java error info in dev mode because it contains at lot of
        # unnecessary info about internal packages and dependencies.
        if (
            isinstance(e, bodo_iceberg_connector.IcebergJavaError)
            and numba.core.config.DEVELOPER_MODE
        ):  # pragma: no cover
            raise BodoError(f"{e.message}:\n {e.java_error.stacktrace()}")
        else:
            raise BodoError(e.message)

    return lst


class IcebergParquetDataset(object):
    """
    Store dataset info in the way expected by Arrow reader in C++.
    This is essentially a wrapper around the ParquetDataset
    object that we generate in our parquet infrastructure
    (see get_parquet_dataset in parquet_pio.py).
    conn, database schema, table_name and pa_table_schema
    are required.
    'pa_table_schema' is the pyarrow.lib.Schema object
    obtained from Iceberg at compile time, i.e. the expected final
    schema.
    'pq_dataset' is the parquet dataset object. It's required
    in cases there are files to read.
    We derive 'pieces', '_bodo_total_rows' and '_prefix' attributes
    from it.
    Eventually, the structure will change to include information
    regarding schema evolution, etc.
    """

    def __init__(
        self,
        conn,
        database_schema,
        table_name,
        pa_table_schema,
        pq_dataset=None,
    ):
        self.pq_dataset = pq_dataset
        self.conn = conn
        self.database_schema = database_schema
        self.table_name = table_name
        self.schema = pa_table_schema

        # For the 0-file case, i.e. pq_dataset is None
        self.pieces = []
        self._bodo_total_rows = 0
        self._prefix = ""
        self.filesystem = None

        # If pq_dataset is provided, get these properties
        # from it
        if pq_dataset is not None:
            self.pieces = pq_dataset.pieces
            self._bodo_total_rows = pq_dataset._bodo_total_rows
            self._prefix = pq_dataset._prefix
            # this is the filesystem that will be used to read data
            self.filesystem = pq_dataset.filesystem


def get_iceberg_pq_dataset(
    conn,
    database_schema,
    table_name,
    typing_pa_table_schema,
    dnf_filters=None,
    expr_filters=None,
    is_parallel=False,
):
    """
    Get IcebergParquetDataset object for the specified table.
    'conn', 'database_schema' and 'table_name' are the table identifiers
    provided by the user.
    'typing_pa_table_schema' is the pyarrow.lib.Schema object
    obtained from Iceberg at compile time, i.e. the expected final
    schema.
    Currently since we don't support schema evolution, we use this to
    detect if there is schema evolution and throwing appropriate error
    if that's the case.
    'dnf_filters' are the filters that are passed to Iceberg.
    'expr_filters' are applied by our Parquet infrastructure.
    'is_parallel' : True if reading in parallel
    """

    ev = tracing.Event("get_iceberg_pq_dataset")
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    pq_file_list_or_e = None
    # Get the list on just one rank to reduce JVM overheads
    # and general traffic to table for when there are
    # catalogs in the future.
    if bodo.get_rank() == 0 or not is_parallel:
        ev_iceberg_fl = tracing.Event("get_iceberg_file_list", is_parallel=False)
        try:
            pq_file_list_or_e = get_iceberg_file_list(
                table_name, conn, database_schema, dnf_filters
            )
            if tracing.is_tracing():  # pragma: no cover
                ICEBERG_TRACING_NUM_FILES_TO_LOG = int(
                    os.environ.get("BODO_ICEBERG_TRACING_NUM_FILES_TO_LOG", "50")
                )
                ev_iceberg_fl.add_attribute("num_files", len(pq_file_list_or_e))
                ev_iceberg_fl.add_attribute(
                    f"first_{ICEBERG_TRACING_NUM_FILES_TO_LOG}_files",
                    ", ".join(pq_file_list_or_e[:ICEBERG_TRACING_NUM_FILES_TO_LOG]),
                )
        except Exception as e:  # pragma: no cover
            pq_file_list_or_e = e
        ev_iceberg_fl.finalize()

    # Send list to all ranks
    if is_parallel:
        pq_file_list_or_e = comm.bcast(pq_file_list_or_e)

    # raise error on all processors if found (not just rank 0 which would cause hangs)
    if isinstance(pq_file_list_or_e, Exception):
        error = pq_file_list_or_e
        raise BodoError(
            f"Error reading Iceberg Table: {type(error).__name__}: {str(error)}\n"
        )

    pq_file_list = pq_file_list_or_e

    if len(pq_file_list) == 0:
        # In case all files were filered out by Iceberg, we need to
        # build an empty dataframe, but with the right schema.
        # get_parquet_dataset doesn't handle the case where
        # no files are available, so we just create a dummy object
        # with the schema known from compile time.
        pq_dataset = None
    else:
        # Currently this function also does schema validation and detecting
        # schema evolution but eventually we might want to refactor that
        # functionality.
        try:
            # We assume that `get_parquet_dataset` will raise the error
            # uniformly and reliably on all ranks.
            pq_dataset = bodo.io.parquet_pio.get_parquet_dataset(
                pq_file_list,
                get_row_counts=True,
                # dnf_filters were already handled by Iceberg
                expr_filters=expr_filters,
                is_parallel=is_parallel,
                # Provide baseline schema to detect schema evolution
                typing_pa_schema=typing_pa_table_schema,
                # Iceberg has its own partitioning info and stores
                # partition columns in the parquet files, so we
                # tell Arrow not to detect partitioning
                partitioning=None,
            )
        except BodoError as e:

            if re.search(r"Schema .* was different", str(e), re.IGNORECASE):
                raise BodoError(
                    f"Bodo currently doesn't support reading Iceberg tables with schema evolution.\n{e}"
                )
            else:
                raise
    iceberg_pq_dataset = IcebergParquetDataset(
        conn, database_schema, table_name, typing_pa_table_schema, pq_dataset
    )
    ev.finalize()

    return iceberg_pq_dataset
