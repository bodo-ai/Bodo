"""
Operations to get file metadata from Iceberg tables during
read runtime. This includes getting a list of parquet files
and processing their metadata for later steps.
"""

from __future__ import annotations

import itertools
import json
import os
import time
import typing as pt

import pyarrow as pa

import bodo
import bodo.utils.tracing as tracing
from bodo.io.iceberg.common import (
    FieldIDs,
    FieldNames,
    IcebergParquetInfo,
    SchemaGroupIdentifier,
    flatten_tuple,
)
from bodo.io.iceberg.read_parquet import (
    IcebergPqDatasetMetrics,
    get_schema_group_identifier_from_pa_schema,
)
from bodo.io.parquet_pio import fpath_without_protocol_prefix
from bodo.spawn.utils import run_rank0

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.catalog import Catalog
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.io import FileIO
    from pyiceberg.table import DataScan, FileScanTask, ManifestFile, Table


def _construct_parquet_infos(
    table: Table, table_scan: DataScan
) -> tuple[list[IcebergParquetInfo], int]:
    """
    Construct IcebergParquetInfo objects for each file
    from the Iceberg table scanner. This includes performing additional
    operations to get the schema ID and sanitized path for each file.
    """
    from pyiceberg.manifest import (
        DEFAULT_READ_VERSION,
        MANIFEST_ENTRY_SCHEMAS,
        AvroFile,
        DataFile,
        DataFileContent,
        FileFormat,
        ManifestEntry,
        ManifestEntryStatus,
    )
    from pyiceberg.typedef import KeyDefaultDict

    file_path_to_schema_id = {}
    tasks: pt.Iterable[FileScanTask] = table_scan.plan_files()

    s = time.monotonic_ns()
    # Construct a mapping from file path to schema ID
    snap = table.current_snapshot()
    assert snap is not None

    # Filter manifest files based on partition summaries, similar to:
    # https://github.com/apache/iceberg-python/blob/59dc8d13ad4e1500fff12946f1bfaddb5484f90e/pyiceberg/table/__init__.py#L1942
    manifest_evaluators: dict[int, pt.Callable[[ManifestFile], bool]] = KeyDefaultDict(
        table_scan._build_manifest_evaluator
    )
    manifests = [
        manifest_file
        for manifest_file in snap.manifests(table.io)
        if manifest_evaluators[manifest_file.partition_spec_id](manifest_file)
    ]

    for manifest_file in manifests:
        # Similar to PyIceberg's fetch_manifest_entry here:
        # https://github.com/apache/iceberg-python/blob/38ebb19a39407f52fe439289af8be81268932b0b/pyiceberg/manifest.py#L696
        input_file = table.io.new_input(manifest_file.manifest_path)
        with AvroFile[ManifestEntry](
            input_file,
            MANIFEST_ENTRY_SCHEMAS[DEFAULT_READ_VERSION],
            read_types={-1: ManifestEntry, 2: DataFile},
            read_enums={0: ManifestEntryStatus, 101: FileFormat, 134: DataFileContent},
        ) as reader:
            schema_id = int(json.loads(reader.header.meta["schema"])["schema-id"])
            for entry in reader:
                file_path = entry.data_file.file_path
                file_path_to_schema_id[file_path] = schema_id

    get_file_to_schema_us = time.monotonic_ns() - s

    # Construct the list of Parquet file info
    return [
        IcebergParquetInfo(
            file_task=task,
            schema_id=file_path_to_schema_id[task.file.file_path],
            sanitized_path=fpath_without_protocol_prefix(task.file.file_path),
        )
        for task in tasks
    ], get_file_to_schema_us // 1000


def _get_total_num_pq_files_in_table(table: Table) -> int:
    """
    Returns the total number of Parquet files in the given Iceberg table
    at the current snapshot. Used for logging the # of filtered files.
    Expected to only run on 1 rank
    """
    from pyiceberg.manifest import ManifestContent

    snap = table.current_snapshot()
    assert snap is not None

    # First, check if we can get the information from the summary
    if (summ := snap.summary) and (count := summ["total-data-files"]):
        return int(count)

    # If it doesn't exist in the summary, check the manifestList
    # TODO: is this doable? I can get the manifestList location, but I don't see a way
    # to get any metadata from this list.
    # A manifest list includes summary metadata that can be used to avoid scanning all of the
    # manifests
    # in a snapshot when planning a table scan. This includes the number of added, existing, and
    # deleted files,
    # and a summary of values for each field of the partition spec used to write the manifest.

    # If it doesn't exist in the manifestList, calculate it by iterating over each manifest file
    total_files = 0
    for manifest_file in snap.manifests(table.io):
        if manifest_file.content != ManifestContent.DATA:
            continue
        existing_files = manifest_file.existing_files_count
        added_files = manifest_file.added_files_count
        deleted_files = manifest_file.deleted_files_count

        if existing_files is None or added_files is None or deleted_files is None:
            # If any of the option fields are None, we have to manually read the file
            manifest_contents = manifest_file.fetch_manifest_entry(
                table.io, discard_deleted=True
            )
            total_files += len(manifest_contents)
        else:
            total_files += existing_files + added_files - deleted_files

    return total_files


@run_rank0
def get_iceberg_file_list_parallel(
    catalog: Catalog,
    table_id: str,
    filters: BooleanExpression,
    snapshot_id: int = -1,
    limit: int = -1,
) -> tuple[list[IcebergParquetInfo], dict, int, FileIO, int]:
    """
    Wrapper around 'get_iceberg_file_list' which calls it
    on rank 0 and handles all the required error
    synchronization and broadcasts the outputs to all ranks.
    NOTE: This function must be called in parallel
    on all ranks.

    Args:
        conn (str): Iceberg connection string
        table_id (str): Iceberg table identifier
        filters (optional): Filters for file pruning. Defaults to None.
        snapshot_id (int, optional): Snapshot ID to read from. Defaults to -1.
        limit (int, optional): Limit on the number of rows to read. Defaults to -1.

    Returns:
        tuple[IcebergParquetInfo, int, dict[int, pa.Schema]]:
        - List of Parquet file info from Iceberg including
            - Original and sanitized file paths
            - Additional metadata like schema and row count info
        - Snapshot ID that these files were taken from.
        - Schema group identifier to schema mapping
    """

    ev_iceberg_fl = tracing.Event("get_iceberg_file_list", is_parallel=False)
    if tracing.is_tracing():  # pragma: no cover
        ev_iceberg_fl.add_attribute("g_filters", filters)
    try:
        table = catalog.load_table(table_id)
        table_scan = table.scan(
            filters,
            snapshot_id=snapshot_id if snapshot_id > -1 else None,
            limit=limit if limit > -1 else None,
        )
        pq_infos, get_file_to_schema_us = _construct_parquet_infos(table, table_scan)

        if tracing.is_tracing():  # pragma: no cover
            ICEBERG_TRACING_NUM_FILES_TO_LOG = int(
                os.environ.get("BODO_ICEBERG_TRACING_NUM_FILES_TO_LOG", "50")
            )
            ev_iceberg_fl.add_attribute("num_files", len(pq_infos))
            ev_iceberg_fl.add_attribute(
                f"first_{ICEBERG_TRACING_NUM_FILES_TO_LOG}_files",
                ", ".join(x.path for x in pq_infos[:ICEBERG_TRACING_NUM_FILES_TO_LOG]),
            )
    except Exception as exc:  # pragma: no cover
        # TODO: raise BodoError in case of compiler (not dataframe library)
        raise ValueError(
            f"Error reading Iceberg Table: {type(exc).__name__}: {str(exc)}\n"
        ) from exc
    finally:
        ev_iceberg_fl.finalize()

    if bodo.user_logging.get_verbose_level() >= 1:
        try:
            total_num_files = str(_get_total_num_pq_files_in_table(table))
        except Exception as e:
            total_num_files = (
                "unknown (error getting total number of files: " + str(e) + ")"
            )

        num_files_read = len(pq_infos)
        if bodo.user_logging.get_verbose_level() >= 2:
            # Constant to limit the number of files to list in the log message
            # May want to increase this for higher verbosity levels
            num_files_to_list = 10

            file_list = ", ".join(x.path for x in pq_infos[:num_files_to_list])
            log_msg = f"Total number of files is {total_num_files}. Reading {num_files_read} files: {file_list}"

            if num_files_read > num_files_to_list:
                log_msg += f", ... and {num_files_read - num_files_to_list} more."
        else:
            log_msg = f"Total number of files is {total_num_files}. Reading {num_files_read} files."

        bodo.user_logging.log_message("Iceberg File Pruning:", log_msg)

    from pyiceberg.io.pyarrow import schema_to_pyarrow

    return (
        pq_infos,
        {i: schema_to_pyarrow(s) for i, s in table.schemas().items()},
        snapshot_id if snapshot_id > -1 else table.current_snapshot().snapshot_id,
        table.io,
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
                f"Encountered an error while generating the schema group identifier for file {pq_info.path}. "
                "This is most likely either a corrupted/invalid Parquet file or represents a bug/gap in Bodo.\n"
                f"{str(e)}"
            )
            # TODO: raise BodoError in case of compiler (not dataframe library)
            raise ValueError(msg)
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


def get_table_length(table, snapshot_id: int = -1) -> int:
    """
    Get the total number of rows in the Iceberg table.
    If a snapshot ID is provided, it will return the number of rows
    in that snapshot. Otherwise, it will return the number of rows
    in the current snapshot.

    Args:
        table (Table): The Iceberg table to get the length of.
        snapshot_id (int, optional): The snapshot ID to get the length of. Defaults to -1.

    Returns:
        int: The total number of rows in the Iceberg table.
    """
    from pyiceberg.manifest import ManifestContent, ManifestEntryStatus

    table_len = 0
    snapshot = (
        table.current_snapshot()
        if snapshot_id == -1
        else table.snapshot_by_id(snapshot_id)
    )
    assert snapshot is not None
    # If the snapshot has a summary with total-records, use that.
    if (
        hasattr(snapshot, "summary")
        and snapshot.summary is not None
        and "total-records" in snapshot.summary
    ):
        table_len = int(snapshot.summary["total-records"])
    # If the snapshot has manifests with existing_rows_count and added_rows_count,
    # use those to get the table length.
    elif all(
        hasattr(manifest, "existing_rows_count")
        and manifest.existing_rows_count is not None
        and hasattr(manifest, "added_rows_count")
        and manifest.added_rows_count is not None
        for manifest in snapshot.manifests(table.io)
    ):
        table_len = sum(
            [
                manifest.existing_rows_count + manifest.added_rows_count
                if manifest.content == ManifestContent.DATA
                else 0
                for manifest in snapshot.manifests(table.io)
            ]
        )
    # Otherwise we need to go through the manifest entries to estimate the table length.
    else:
        for manifest in snapshot.manifests(table.io):
            manifest_entries = manifest.fetch_manifest_entry(table.io)
            for entry in manifest_entries:
                if (
                    entry.status == ManifestEntryStatus.DELETED
                    or manifest.content != ManifestContent.DATA
                ):
                    continue
                datafile = entry.data_file
                if datafile is not None and datafile.record_count is not None:
                    table_len += datafile.record_count

    return table_len
