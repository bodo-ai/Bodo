"""
Provides utilities for creating Bodo Lazy Plans from Iceberg tables.
"""

from __future__ import annotations

import itertools
import random
import typing as pt
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa

from bodo.pandas.plan import (
    LogicalGetIcebergRead,
)
from bodo.pandas.utils import (
    arrow_to_empty_df,
    wrap_str_fields_as_dict,
)

# Number of files to sample to determine the columns that should be
# dict-encoded. Sampling is done without MPI since planning runs in a single
# Python process (rank 0 / spawner), so we use a small fixed sample.
DICT_ENCODE_NUM_SAMPLE_FILES = 3


@dataclass
class JoinFilterInfo:
    filter_ids: list[int]
    equality_filter_columns: list[list[int]]
    orig_build_key_cols: list[list[int]]
    equality_is_first_locations: list[list[bool]]


def determine_str_as_dict_columns(
    table: pt.Any,
    pa_schema: pa.Schema,
    str_col_names: list[str],
    snapshot_id: int = -1,
) -> set[str]:
    """
    Determine the set of string columns in an Iceberg table that should be
    read as dict-encoded string columns. This mirrors the heuristic used by
    the JIT path (bodo.io.iceberg.read_compilation._determine_str_as_dict_columns):
    a column is dict-encoded if its average uncompressed size per row (over a
    sample of files) is below READ_STR_AS_DICT_THRESHOLD.

    NOTE: This runs in a single process (no MPI collectives), so we sample a
    small fixed number of files instead of one file per rank.

    Args:
        table: PyIceberg table object to scan.
        pa_schema (pa.Schema): Arrow schema of the table (must have Iceberg
            field IDs in field metadata).
        str_col_names (list[str]): Names of string columns to check.
        snapshot_id (int): Snapshot ID to use for the scan. -1 means latest.

    Returns:
        set[str]: Set of column names that should be dict-encoded
            (subset of str_col_names).
    """
    import pyarrow.parquet as pq

    from bodo.io.dict_encode import str_as_dict_from_col_sizes
    from bodo.io.iceberg.common import _fs_from_file_path, b_ICEBERG_FIELD_ID_MD_KEY
    from bodo.io.parquet_pio import fpath_without_protocol_prefix

    if len(str_col_names) == 0:
        return set()

    # Get a small list of files to probe. No filters are known at this time,
    # so no file pruning can be done. Limit the number of files we plan to
    # avoid reading too many manifest entries for large tables.
    file_tasks = list(
        itertools.islice(
            table.scan(
                selected_fields=tuple(str_col_names),
                snapshot_id=snapshot_id if snapshot_id != -1 else None,
            ).plan_files(),
            1000,
        )
    )
    if len(file_tasks) == 0:
        return set()

    sample_files = (
        random.Random(37).sample(file_tasks, DICT_ENCODE_NUM_SAMPLE_FILES)
        if len(file_tasks) > DICT_ENCODE_NUM_SAMPLE_FILES
        else file_tasks
    )

    # Map Iceberg field IDs to the index of the column in str_col_names so we
    # can find the right columns in Parquet files even with schema evolution.
    str_col_name_to_field_id: dict[str, int] = {}
    for field in pa_schema:
        if field.name not in str_col_names:
            continue
        if field.metadata is None or b_ICEBERG_FIELD_ID_MD_KEY not in field.metadata:
            raise ValueError(
                "iceberg_utils.determine_str_as_dict_columns: Schema does not "
                "have Iceberg field IDs."
            )
        str_col_name_to_field_id[field.name] = int(
            field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
        )
    field_id_to_col_idx: dict[int, int] = {
        str_col_name_to_field_id[name]: i for i, name in enumerate(str_col_names)
    }

    total_uncompressed_sizes = [0] * len(str_col_names)
    total_rows = 0
    fs = None
    for fpath in sample_files:
        try:
            if fs is None:
                fs = _fs_from_file_path(fpath.file.file_path, table.io)
            sanitized_path = fpath_without_protocol_prefix(fpath.file.file_path)
            pq_file = pq.ParquetFile(sanitized_path, filesystem=fs)
            metadata = pq_file.metadata
            # Map the file's columns to the columns to check. Prefer Iceberg
            # field IDs (robust to renames), falling back to column names for
            # files that don't have field ID metadata (e.g. externally
            # registered files).
            file_col_to_check_idx: dict[int, int] = {}
            for idx, field in enumerate(pq_file.schema_arrow):
                col_idx = None
                if field.metadata is not None and (
                    b_ICEBERG_FIELD_ID_MD_KEY in field.metadata
                ):
                    col_idx = field_id_to_col_idx.get(
                        int(field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])
                    )
                elif field.name in str_col_name_to_field_id:
                    col_idx = str_col_names.index(field.name)
                if col_idx is not None:
                    file_col_to_check_idx[idx] = col_idx
            for idx, col_idx in file_col_to_check_idx.items():
                for i in range(pq_file.num_row_groups):
                    total_uncompressed_sizes[col_idx] += (
                        metadata.row_group(i).column(idx).total_uncompressed_size
                    )
            total_rows += metadata.num_rows
        except (OSError, FileNotFoundError):
            # Skip the file that produced the error (error will be reported at
            # runtime if the file is actually read).
            continue

    return str_as_dict_from_col_sizes(
        str_col_names, total_uncompressed_sizes, total_rows
    )


def build_iceberg_read_plan(
    table_identifier: str,
    catalog_name: str | None = None,
    catalog_properties: dict[str, pt.Any] | None = None,
    row_filter: str | None = None,
    snapshot_id: int | None = None,
    location: str | None = None,
    join_filter_info: JoinFilterInfo | None = None,
    selected_fields: list[str] | None = None,
    limit: int | None = None,
) -> tuple[LogicalGetIcebergRead, pd.DataFrame, pa.Schema]:
    """Create an Iceberg read plan for the given table and return the plan, an empty
    dataframe with the correct schema and the arrow schema
    """
    import pyiceberg.catalog
    import pyiceberg.expressions
    import pyiceberg.table

    from bodo.io.dict_encode import select_str_as_dict_cols
    from bodo.io.iceberg.read_metadata import get_table_length
    from bodo.pandas.utils import BodoLibNotImplementedException

    # Support simple directory only calls like:
    # pd.read_iceberg("table", location="/path/to/table")
    if catalog_name is None and catalog_properties is None and location is not None:
        if location.startswith("arn:aws:s3tables:"):
            from bodo.io.iceberg.catalog.s3_tables import (
                construct_catalog_properties as construct_s3_tables_catalog_properties,
            )

            catalog_properties = construct_s3_tables_catalog_properties(location)
        else:
            catalog_properties = {
                pyiceberg.catalog.PY_CATALOG_IMPL: "bodo.io.iceberg.catalog.dir.DirCatalog",
                pyiceberg.catalog.WAREHOUSE_LOCATION: location,
            }
    elif location is not None:
        raise BodoLibNotImplementedException(
            "'location' is only supported for filesystem catalog and cannot be used "
            "with catalog_name or catalog_properties."
        )
    elif catalog_properties is None:
        catalog_properties = {}

    catalog = pyiceberg.catalog.load_catalog(catalog_name, **catalog_properties)

    # Get the output schema
    table = catalog.load_table(table_identifier)
    pyiceberg_schema = table.schema()
    arrow_read_schema = pyiceberg_schema.as_arrow()
    empty_df = arrow_to_empty_df(arrow_read_schema)

    # Get the table length estimate, if there's not a filter it will be exact
    table_len_estimate = get_table_length(table, snapshot_id or -1)

    # If there's a row filter, we need to estimate the selectivity
    # and adjust the table length estimate accordingly.
    if row_filter is not None and table_len_estimate > 0:
        # TODO: do something smarter here like sampling or turn the filter into a
        # separate node so the planner can handle it
        #
        # This matches duckdb's default selectivity estimate for filters
        filter_selectivity_estimate = 0.2
        table_len_estimate = int(table_len_estimate * filter_selectivity_estimate)

    # None here implies all fields should be selected
    selected_idxs = None
    if selected_fields is not None:
        selected_idxs = [
            arrow_read_schema.get_field_index(field_name)
            for field_name in selected_fields
        ]
        empty_df = empty_df[list(selected_fields)]
        arrow_out_schema = pa.schema(
            [arrow_read_schema.field(i) for i in selected_idxs]
        )
    else:
        arrow_out_schema = arrow_read_schema

    # Determine which string columns should be read with dictionary encoding.
    # The selection is done at the logical plan level so df.dtypes and the
    # physical reader output agree. str_as_dict_cols are indices into the full
    # read schema (arrow_read_schema), matching the convention used by the
    # physical reader and the JIT path.
    selected_names = (
        selected_fields
        if selected_fields is not None
        else list(arrow_read_schema.names)
    )
    str_col_names = [
        field.name
        for field in arrow_read_schema
        if field.name in selected_names
        and (pa.types.is_string(field.type) or pa.types.is_large_string(field.type))
    ]
    str_as_dict_names = select_str_as_dict_cols(
        str_col_names,
        lambda names: determine_str_as_dict_columns(
            table, arrow_read_schema, names, snapshot_id or -1
        ),
    )
    str_as_dict_cols = [
        arrow_read_schema.get_field_index(name) for name in str_as_dict_names
    ]
    if len(str_as_dict_cols) > 0:
        # str_as_dict_cols are indices into the full read schema, but
        # arrow_out_schema may be a projection of it when selected_fields is
        # provided, so map the indices to positions in the output schema
        # before wrapping.
        out_str_as_dict_cols = (
            [selected_idxs.index(i) for i in str_as_dict_cols]
            if selected_idxs is not None
            else str_as_dict_cols
        )
        # Wrap the selected fields as dictionary types in the output schema
        # so planning and the physical reader agree on the output dtypes.
        arrow_out_schema = wrap_str_fields_as_dict(
            arrow_out_schema, out_str_as_dict_cols
        )
        empty_df = arrow_to_empty_df(arrow_out_schema)

    plan = LogicalGetIcebergRead(
        empty_df,
        table_identifier,
        catalog_name,
        catalog_properties,
        pyiceberg.table._parse_row_filter(row_filter)
        if row_filter
        else pyiceberg.expressions.AlwaysTrue(),
        # We need to pass the pyiceberg schema so we can bind the iceberg filter to it
        # during filter conversion. See bodo/io/iceberg/common.py::pyiceberg_filter_to_pyarrow_format_str_and_scalars
        pyiceberg_schema,
        arrow_read_schema,
        snapshot_id if snapshot_id is not None else -1,
        table_len_estimate,
        arrow_schema=arrow_out_schema,
        join_filter_info=join_filter_info,
        selected_fields=selected_idxs,
        limit=limit,
        str_as_dict_cols=str_as_dict_cols,
    )

    return plan, empty_df, arrow_out_schema
