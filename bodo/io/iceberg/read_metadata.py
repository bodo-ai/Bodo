"""
Operations to get file metadata from Iceberg tables during
read runtime. This includes getting a list of parquet files
and processing their metadata for later steps.
"""

from __future__ import annotations

import itertools
import os
import time
import typing as pt

import pyarrow as pa

import bodo
import bodo.utils.tracing as tracing
from bodo.io.iceberg.common import (
    FieldIDs,
    FieldNames,
    SchemaGroupIdentifier,
    flatten_tuple,
)
from bodo.io.iceberg.read_parquet import (
    IcebergPqDatasetMetrics,
    get_schema_group_identifier_from_pa_schema,
)
from bodo.mpi4py import MPI
from bodo.utils.utils import BodoError

if pt.TYPE_CHECKING:  # pragma: no cover
    from bodo_iceberg_connector import IcebergParquetInfo


def get_iceberg_file_list(
    table_name: str, conn: str, database_schema: str, filters: str | None
) -> tuple[list[IcebergParquetInfo], dict[int, pa.Schema], int]:
    """
    Gets the list of parquet data files that need to be read from an Iceberg table.

    We also pass filters, which is in DNF format and the output of filter
    pushdown to Iceberg. Iceberg will use this information to
    prune any files that it can from just metadata, so this
    is an "inclusive" projection.
    NOTE: This must only be called on rank 0.

    Returns:
        - List of file paths from Iceberg sanitized to be used by Bodo
            - Convert S3A paths to S3 paths
            - Convert relative paths to absolute paths
        - List of original file paths directly from Iceberg
    """
    import bodo_iceberg_connector as bic

    assert (
        bodo.get_rank() == 0
    ), "get_iceberg_file_list should only ever be called on rank 0, as the operation requires access to the py4j server, which is only available on rank 0"

    try:
        return bic.get_bodo_parquet_info(conn, database_schema, table_name, filters)
    except bic.IcebergError as e:
        raise BodoError(
            f"Failed to Get List of Parquet Data Files from Iceberg Table: {e.message}"
        )


def get_iceberg_snapshot_id(table_name: str, conn: str, database_schema: str) -> int:
    """
    Fetch the current snapshot id for an Iceberg table.

    Args:
        table_name (str): Iceberg Table Name
        conn (str): Iceberg connection string
        database_schema (str): Iceberg schema.

    Returns:
        int: Snapshot Id for the current version of the Iceberg table.
    """
    import bodo_iceberg_connector

    assert (
        bodo.get_rank() == 0
    ), "get_iceberg_snapshot_id should only ever be called on rank 0, as the operation requires access to the py4j server, which is only available on rank 0"

    try:
        return bodo_iceberg_connector.bodo_connector_get_current_snapshot_id(
            conn,
            database_schema,
            table_name,
        )
    except bodo_iceberg_connector.IcebergError as e:
        raise BodoError(
            f"Failed to Get the Snapshot ID from an Iceberg Table: {e.message}"
        )


def get_iceberg_file_list_parallel(
    conn: str,
    database_schema: str,
    table_name: str,
    filters: str | None = None,
) -> tuple[list[IcebergParquetInfo], int, dict[int, pa.Schema], int]:
    """
    Wrapper around 'get_iceberg_file_list' which calls it
    on rank 0 and handles all the required error
    synchronization and broadcasts the outputs
    to all ranks.
    NOTE: This function must be called in parallel
    on all ranks.

    Args:
        conn (str): Iceberg connection string
        database_schema (str): Iceberg database.
        table_name (str): Iceberg table's name
        filters (optional): Filters for file pruning. Defaults to None.

    Returns:
        tuple[IcebergParquetInfo, int, dict[int, pa.Schema]]:
        - List of Parquet file info from Iceberg including
            - Original and sanitized file paths
            - Additional metadata like schema and row count info
        - Snapshot ID that these files were taken from.
        - Schema group identifier to schema mapping
    """
    comm = MPI.COMM_WORLD
    exc = None
    pq_infos = None
    snapshot_id_or_e = None
    all_schemas = None
    get_file_to_schema_us = None
    # Get the list on just one rank to reduce JVM overheads
    # and general traffic to table for when there are
    # catalogs in the future.

    # Always get the list on rank 0 to avoid the need
    # to initialize a full JVM + gateway server on every rank.
    # Only runs on rank 0, so we add no cover to avoid coverage warning
    if bodo.get_rank() == 0:  # pragma: no cover
        ev_iceberg_fl = tracing.Event("get_iceberg_file_list", is_parallel=False)
        if tracing.is_tracing():  # pragma: no cover
            ev_iceberg_fl.add_attribute("g_filters", filters)
        try:
            (
                pq_infos,
                all_schemas,
                get_file_to_schema_us,
            ) = get_iceberg_file_list(table_name, conn, database_schema, filters)
            if tracing.is_tracing():  # pragma: no cover
                ICEBERG_TRACING_NUM_FILES_TO_LOG = int(
                    os.environ.get("BODO_ICEBERG_TRACING_NUM_FILES_TO_LOG", "50")
                )
                ev_iceberg_fl.add_attribute("num_files", len(pq_infos))
                ev_iceberg_fl.add_attribute(
                    f"first_{ICEBERG_TRACING_NUM_FILES_TO_LOG}_files",
                    ", ".join(
                        x.orig_path for x in pq_infos[:ICEBERG_TRACING_NUM_FILES_TO_LOG]
                    ),
                )
        except Exception as e:  # pragma: no cover
            exc = e

        ev_iceberg_fl.finalize()
        ev_iceberg_snapshot = tracing.Event("get_snapshot_id", is_parallel=False)
        try:
            snapshot_id_or_e = get_iceberg_snapshot_id(
                table_name, conn, database_schema
            )
        except Exception as e:  # pragma: no cover
            snapshot_id_or_e = e
        ev_iceberg_snapshot.finalize()

        if bodo.user_logging.get_verbose_level() >= 1 and isinstance(pq_infos, list):
            import bodo_iceberg_connector as bic

            # This should never fail given that pq_infos is not None, but just to be safe.
            try:
                total_num_files = bic.bodo_connector_get_total_num_pq_files_in_table(
                    conn, database_schema, table_name
                )
            except bic.errors.IcebergJavaError as e:
                total_num_files = (
                    "unknown (error getting total number of files: " + str(e) + ")"
                )

            num_files_read = len(pq_infos)

            if bodo.user_logging.get_verbose_level() >= 2:
                # Constant to limit the number of files to list in the log message
                # May want to increase this for higher verbosity levels
                num_files_to_list = 10

                file_list = ", ".join(x.orig_path for x in pq_infos[:num_files_to_list])
                log_msg = f"Total number of files is {total_num_files}. Reading {num_files_read} files: {file_list}"

                if num_files_read > num_files_to_list:
                    log_msg += f", ... and {num_files_read-num_files_to_list} more."
            else:
                log_msg = f"Total number of files is {total_num_files}. Reading {num_files_read} files."

            bodo.user_logging.log_message(
                "Iceberg File Pruning:",
                log_msg,
            )

    # Send list to all ranks
    (
        exc,
        pq_infos,
        snapshot_id_or_e,
        all_schemas,
        get_file_to_schema_us,
    ) = comm.bcast(
        (
            exc,
            pq_infos,
            snapshot_id_or_e,
            all_schemas,
            get_file_to_schema_us,
        )
    )

    # Raise error on all processors if found (not just rank 0 which would cause hangs)
    if isinstance(exc, Exception):
        raise BodoError(
            f"Error reading Iceberg Table: {type(exc).__name__}: {str(exc)}\n"
        )
    if isinstance(snapshot_id_or_e, Exception):
        error = snapshot_id_or_e
        raise BodoError(
            f"Error reading Iceberg Table: {type(error).__name__}: {str(error)}\n"
        )

    snapshot_id: int = snapshot_id_or_e
    return (
        pq_infos,
        snapshot_id,
        all_schemas,
        get_file_to_schema_us,
    )


def group_file_frags_by_schema_group_identifier(
    pq_infos: list[IcebergParquetInfo],
    file_schemas: list[pa.Schema],
    metrics: IcebergPqDatasetMetrics,
) -> dict[SchemaGroupIdentifier, list[IcebergParquetInfo]]:
    """
    Group a list of Parquet file fragments by their Schema Group identifier,
    i.e. based on the Iceberg Field IDs and corresponding
    field names.
    The fragments are assumed to have their metadata already populated.
    NOTE: This function is completely local and doesn't
    do any synchronization. It may raise Exceptions.
    The caller is expected to handle the error-synchronization.

    Args:
        pq_infos (list[ds.ParquetFileFragment]): List of Parquet Infos from Iceberg connector.
        metrics (IcebergPqDatasetMetrics): Metrics to update in place.

    Returns:
        dict[
            SchemaGroupIdentifier,
            list[IcebergParquetInfo]
        ]: Dictionary mapping the schema group identifier
            to the list of IcebergParquetInfo for that schema group identifier.
            The schema group identifier is a tuple of
            two ordered tuples. The first is an ordered tuple
            of the Iceberg field IDs in the file and the second
            is an ordered tuple of the corresponding field
            names. Note that only the top-level fields are considered.
    """
    ## Get the Field IDs and Column Names of the files:
    iceberg_field_ids: list[FieldIDs] = []
    pq_field_names: list[FieldNames] = []

    # Get the schema group identifier for each file using the pre-fetched metadata:
    start = time.monotonic()
    for pq_info, file_schema in zip(pq_infos, file_schemas):
        try:
            schema_group_identifier = get_schema_group_identifier_from_pa_schema(
                file_schema
            )
        except Exception as e:
            msg = (
                f"Encountered an error while generating the schema group identifier for file {pq_info.orig_path}. "
                "This is most likely either a corrupted/invalid Parquet file or represents a bug/gap in Bodo.\n"
                f"{str(e)}"
            )
            raise BodoError(msg)
        iceberg_field_ids.append(schema_group_identifier[0])
        pq_field_names.append(schema_group_identifier[1])
    metrics.get_sg_id_time += int((time.monotonic() - start) * 1_000_000)

    # Sort the files based on their schema group identifier
    start = time.monotonic()
    file_frags_schema_group_ids: list[
        tuple[IcebergParquetInfo, FieldIDs, FieldNames]
    ] = list(zip(pq_infos, iceberg_field_ids, pq_field_names))
    # Sort/Groupby the field-ids and field-names tuples.
    # We must flatten the tuples for sorting because you
    # cannot compare ints to tuples in Python and nested types
    # will generate tuples. This is safe because a nested field
    # can never become a primitive column (and vice-versa).
    sort_key_func = lambda item: (flatten_tuple(item[1]), flatten_tuple(item[2]))
    keyfunc = lambda item: (item[1], item[2])
    schema_group_id_to_frags: dict[SchemaGroupIdentifier, list[IcebergParquetInfo]] = {
        k: [x[0] for x in v]
        for k, v in itertools.groupby(
            sorted(file_frags_schema_group_ids, key=sort_key_func), keyfunc
        )
    }
    metrics.sort_by_sg_id_time += int((time.monotonic() - start) * 1_000_000)
    metrics.nunique_sgs_seen += len(schema_group_id_to_frags)

    return schema_group_id_to_frags
