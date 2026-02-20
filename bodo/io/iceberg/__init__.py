"""
General support for reading and writing Iceberg tables with
underlying Parquet files. Most of the functionality is implemented
in other files and exported out of this module.
"""

from __future__ import annotations

import importlib
import os
import time
import typing as pt

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from mpi4py import MPI

import bodo
import bodo.utils.tracing as tracing
from bodo.io import arrow_cpp

from .common import (
    ICEBERG_FIELD_ID_MD_KEY,
    _fs_from_file_path,
    flatten_concatenation,
    flatten_tuple,
    verify_pyiceberg_installed,
)
from .read_metadata import (
    get_iceberg_file_list_parallel,
    group_file_frags_by_schema_group_identifier,
)
from .read_parquet import (
    IcebergParquetDataset,
    IcebergPiece,
    IcebergPqDatasetMetrics,
    IcebergSchemaGroup,
    SchemaGroupIdentifier,
    get_row_counts_for_schema_group,
    warn_if_non_ideal_io_parallelism,
)

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.catalog import Catalog
    from pyiceberg.expressions import BooleanExpression


ICEBERG_WRITE_PARQUET_CHUNK_SIZE = int(256e6)

pyiceberg_imported = False
try:
    importlib.import_module("pyiceberg")
    pyiceberg_imported = True
except ImportError:
    pass

if pyiceberg_imported:
    from . import monkey_patch as _  # noqa: F401


def get_iceberg_pq_dataset(
    catalog: Catalog,
    table_id: str,
    typing_pa_table_schema: pa.Schema,
    str_as_dict_cols: list[str],
    iceberg_filter: BooleanExpression,
    expr_filter_f_str: str | None = None,
    filter_scalars: list[tuple[str, pt.Any]] | None = None,
    force_row_level_read: bool = True,
    snapshot_id: int = -1,
    limit: int = -1,
) -> IcebergParquetDataset:
    """
    Top-Level Function for Planning Iceberg Parquet Files at Runtime

    Get IcebergParquetDataset object for the specified table.
    NOTE: This must be called on all ranks in parallel since
    all processing is parallelized for best performance.

    Args:
        catalog (Catalog): PyIceberg catalog to read table metadata.
        table_id (str): Table Identifier of the table to use.
        typing_pa_table_schema (pa.Schema): Final/Target PyArrow schema
            for the Iceberg table generated at compile time. This must
            have Iceberg Field IDs in the metadata of the fields.
        str_as_dict_cols (list[str]): List of column names
            that will be read with dictionary encoding.
        iceberg_filter (optional): Filters passed to the Iceberg Java library
            for file-pruning. Defaults to None.
        expr_filter_f_str (str, optional): f-string to use to generate
            the Arrow filters. See description of 'generate_expr_filter'
            for more details. Defaults to None.
        filter_scalars (list[tuple[str, Any]], optional): List of the
            scalars used in the expression filter. See description of
            'generate_expr_filter' for more details. Defaults to None.
        force_row_level_read (bool, default: true): TODO
        snapshot_id (int, default: -1): The snapshot ID to use for the Iceberg
            table. If -1, the latest snapshot will be used.
        limit (int, default: -1): Limit on the number of rows to read.

    Returns:
        IcebergParquetDataset: Contains all the pieces to read, along
        with the number of rows to read from them (after applying
        provided filters in the row_level read case) and the schema
        groups they belong to.
        The files/pieces are ordered by the Schema Group they belong
        to. This will be identical on all ranks, i.e. all ranks will
        have all pieces in their dataset. The caller is expected to
        split the work for the actual read.
    """
    _ = verify_pyiceberg_installed()

    ev = tracing.Event("get_iceberg_pq_dataset")
    metrics = IcebergPqDatasetMetrics()
    comm = MPI.COMM_WORLD

    # Get list of files. This is the list after
    # applying the iceberg_filter (metadata-level).
    start_time = time.monotonic()
    (
        pq_infos,
        all_schemas,
        snapshot_id,
        io,
        get_file_to_schema_us,
    ) = get_iceberg_file_list_parallel(
        catalog,
        table_id,
        iceberg_filter,
        snapshot_id,
        limit,
    )
    metrics.file_to_schema_time_us = get_file_to_schema_us
    metrics.file_list_time += int((time.monotonic() - start_time) * 1_000_000)

    # If no files exist/match, return an empty dataset.
    if len(pq_infos) == 0:
        return IcebergParquetDataset(
            True,
            typing_pa_table_schema,
            [],
            snapshot_id,
            filesystem=None,
            pieces=[],
            schema_groups=[],
            _bodo_total_rows=0,
            metrics=metrics,
        )

    start_time = time.monotonic()
    fs = _fs_from_file_path(pq_infos[0].path, io)
    metrics.get_fs_time += int((time.monotonic() - start_time) * 1_000_000)

    if expr_filter_f_str is not None and len(expr_filter_f_str) == 0:
        expr_filter_f_str = None
    if filter_scalars is None:
        filter_scalars = []
    if tracing.is_tracing():
        ev.add_attribute("g_expr_filter_f_str", str(expr_filter_f_str))

    # 1. Select a slice of the list of files based on the rank.
    n_pes, rank = bodo.get_size(), bodo.get_rank()
    total_num_files = len(pq_infos)
    start = bodo.get_start(total_num_files, n_pes, rank)
    end = bodo.get_end(total_num_files, n_pes, rank)

    local_pq_infos = pq_infos[start:end]
    metrics.n_files_analyzed += len(local_pq_infos)

    # 2. For this list of files:
    #    a. Determine the file schema.
    #    b. Group files by their schema-group.
    #    c. For each schema-group:
    #       i. Create a read-schema and then a expr_filter using it.
    #       ii. Get the row counts for each file.
    #           This is also where we will perform schema validation for all the
    #           files, i.e. the schema should be compatible with the read-schema.

    # 2a. We have the assumed file schema from the Iceberg connector.
    # However, the file schema may be different due to writer quirks.
    # Most quirks are handled by Arrow except for struct fields. If the table
    # schema has struct fields, we need to extract the file schemas.
    # TODO: Add null field casting support in Arrow to remove this.
    file_schemas: list[pa.Schema]
    if any(pa.types.is_struct(ty) for ty in typing_pa_table_schema.types):
        pq_format = ds.ParquetFileFormat()
        pq_frags = [
            pq_format.make_fragment(pq_info.sanitized_path, fs)
            for pq_info in local_pq_infos
        ]
        arrow_cpp.fetch_parquet_frags_metadata(pq_frags)
        file_schemas = [
            pq_frag.metadata.schema.to_arrow_schema() for pq_frag in pq_frags
        ]
    else:
        file_schemas = [all_schemas[pq_info.schema_id] for pq_info in local_pq_infos]

    err = None
    local_pieces: list[IcebergPiece] = []
    row_level: bool = True
    try:
        # Group the files based on their schema group.
        schema_group_identifier_to_pq_file_fragments = (
            group_file_frags_by_schema_group_identifier(
                local_pq_infos,
                file_schemas,
                metrics,
            )
        )

        # If we're not forced to do a row-level read, decide whether to do
        # a row-level or piece-level read based on how many files exist.
        # Allows for partial file reads (2 1/2 for example)
        if not force_row_level_read:
            min_files_per_rank = float(os.environ.get("BODO_MIN_PQ_FILES_PER_RANK", 1))
            row_level = total_num_files < int(min_files_per_rank * comm.Get_size())

        for (
            schema_group_identifier,
            schema_group_pq_infos,
        ) in schema_group_identifier_to_pq_file_fragments.items():
            file_pieces: list[IcebergPiece] = get_row_counts_for_schema_group(
                schema_group_identifier,
                schema_group_pq_infos,
                fs,
                typing_pa_table_schema,
                str_as_dict_cols,
                metrics,
                row_level,
                expr_filter_f_str,
                filter_scalars,
            )
            local_pieces.extend(file_pieces)
    except Exception as e:
        err = e
    bodo.spawn.utils.sync_and_reraise_error(err, _is_parallel=True)

    # Analyze number of row groups, their sizes, etc. and print warnings
    # similar to what we do for Parquet.
    g_total_rows = comm.allreduce(metrics.get_row_counts_nrows, op=MPI.SUM)
    g_total_rgs = comm.allreduce(metrics.get_row_counts_nrgs, op=MPI.SUM)
    g_total_size_bytes = comm.allreduce(metrics.get_row_counts_total_bytes, op=MPI.SUM)
    warn_if_non_ideal_io_parallelism(g_total_rgs, g_total_size_bytes, fs.type_name)

    if tracing.is_tracing():  # pragma: no cover
        ev.add_attribute("num_rows", metrics.get_row_counts_nrows)
        ev.add_attribute("num_row_groups", metrics.get_row_counts_nrgs)
        ev.add_attribute("row_group_size_bytes", metrics.get_row_counts_total_bytes)
        ev.add_attribute("row_filtering_time", metrics.exact_row_counts_time)
        ev.add_attribute("g_num_rows", g_total_rows)
        ev.add_attribute("g_num_row_groups", g_total_rgs)
        ev.add_attribute("g_row_group_size_bytes", g_total_size_bytes)
        ev.add_attribute("g_row_level_read", row_level)

    # 3. Allgather the pieces on all ranks.
    t0 = time.monotonic()
    all_pieces = comm.allgather(local_pieces)
    metrics.pieces_allgather_time += int((time.monotonic() - t0) * 1_000_000)
    pieces: list[IcebergPiece] = flatten_concatenation(all_pieces)

    # 4. Sort the list based on the schema-group identifier (filename to break ties).
    # We must flatten the tuples for sorting because you
    # cannot compare ints to tuples in Python and nested types
    # will generate tuples. This is safe because a nested field
    # can never become a primitive column (and vice-versa).
    t0 = time.monotonic()
    pieces = sorted(
        pieces,
        key=lambda piece: (
            (
                flatten_tuple(piece.schema_group_identifier[0]),
                flatten_tuple(piece.schema_group_identifier[1]),
            ),
            piece.path,
        ),
    )
    metrics.sort_all_pieces_time += int((time.monotonic() - t0) * 1_000_000)

    # 5. Create a list of SchemaGroups (same ordering scheme).
    #    Also create an IcebergPiece for each file. This is similar to ParquetPiece
    #    except it has a schema_group_idx. We don't need fields like frag, etc.
    #    Assign the piece.schema_group_idx for all the pieces.
    #    This is a deterministic process and therefore we can be sure that all
    #    ranks will end up with the same result.
    t0 = time.monotonic()
    schema_groups: list[IcebergSchemaGroup] = []
    curr_schema_group_id: SchemaGroupIdentifier | None = None
    iceberg_pieces: list[IcebergPiece] = []
    for piece in pieces:
        if (curr_schema_group_id is None) or (
            curr_schema_group_id != piece.schema_group_identifier
        ):
            schema_groups.append(
                IcebergSchemaGroup(
                    piece.schema_group_identifier[0],
                    piece.schema_group_identifier[1],
                    final_schema=typing_pa_table_schema,
                    expr_filter_f_str=expr_filter_f_str,
                    filter_scalars=filter_scalars,
                )
            )
            curr_schema_group_id = piece.schema_group_identifier
        schema_group_idx = len(schema_groups) - 1
        # Update the schema group index in the piece
        piece.schema_group_idx = schema_group_idx
        iceberg_pieces.append(piece)
    metrics.assemble_ds_time += int((time.monotonic() - t0) * 1_000_000)

    # 6. Create an IcebergParquetDataset object with this list of schema-groups,
    #    pieces and other relevant details.
    assert snapshot_id != -1
    iceberg_pq_dataset = IcebergParquetDataset(
        row_level,
        typing_pa_table_schema,
        [x.path for x in pq_infos],
        snapshot_id,
        fs,
        iceberg_pieces,
        schema_groups,
        _bodo_total_rows=g_total_rows,
        metrics=metrics,
    )

    if tracing.is_tracing():
        # get 5-number summary for rowcounts:
        # (min, max, 25, 50 -median-, 75 percentiles)
        data = np.array([p._bodo_num_rows for p in iceberg_pq_dataset.pieces])
        quartiles = np.percentile(data, [25, 50, 75])
        ev.add_attribute("g_row_counts_min", data.min())
        ev.add_attribute("g_row_counts_Q1", quartiles[0])
        ev.add_attribute("g_row_counts_median", quartiles[1])
        ev.add_attribute("g_row_counts_Q3", quartiles[2])
        ev.add_attribute("g_row_counts_max", data.max())
        ev.add_attribute("g_row_counts_mean", data.mean())
        ev.add_attribute("g_row_counts_std", data.std())
        ev.add_attribute("g_row_counts_sum", data.sum())
    ev.finalize()

    return iceberg_pq_dataset


__all__ = [
    "ICEBERG_FIELD_ID_MD_KEY",
    "get_iceberg_pq_dataset",
]
