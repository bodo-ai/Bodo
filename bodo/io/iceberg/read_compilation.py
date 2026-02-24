"""
Operations to get basic metadata about an Iceberg table
needed for compilation. For example, the table schema & dictionary encoding
"""

from __future__ import annotations

import random
import sys
import typing as pt

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mpi4py import MPI
from numba.core import types

import bodo
from bodo.io.helpers import _get_numba_typ_from_pa_typ
from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.io.iceberg.common import (
    _fs_from_file_path,
    b_ICEBERG_FIELD_ID_MD_KEY,
    verify_pyiceberg_installed,
)
from bodo.io.parquet_pio import fpath_without_protocol_prefix
from bodo.spawn.utils import run_rank0
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.utils import BodoError

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.table import Table


EMPTY_LIST: pt.Final = []


# Class for a PyObject that is a list.
this_module = sys.modules[__name__]
install_py_obj_class(
    types_name="pyobject_of_list_type",
    python_type=None,
    module=this_module,
    class_name="PyObjectOfListType",
    model_name="PyObjectOfListModel",
)


def is_snowflake_managed_iceberg_wh(con: str) -> bool:
    """
    Does the connection string correspond to a Snowflake-managed
    Iceberg catalog.

    Args:
        con (str): Iceberg connection string

    Returns:
        bool: Whether it's a Snowflake-managed Iceberg catalog.
    """
    return con.startswith("iceberg+snowflake://")


@run_rank0
def _get_table_schema(
    table: Table,
    is_merge_into_cow: bool = False,
    snapshot_id: int = -1,
) -> tuple[list[str], list[types.ArrayCompatible], pa.Schema]:
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    table_schema = table.schema()

    # Get the schema for the snapshot we're reading if it's not the latest
    snapshot = table.snapshot_by_id(snapshot_id) if snapshot_id != -1 else None
    if snapshot is not None:
        assert snapshot.schema_id is not None, (
            "_get_table_schema: Snapshot schema ID is None"
        )
        table_schema = table.schemas()[snapshot.schema_id]

    # Base PyArrow Schema for the snapshot we're reading
    pa_schema: pa.Schema = schema_to_pyarrow(table_schema)

    # Column Names to Scan
    col_names = pa_schema.names.copy()

    # Convert to Bodo Schema
    bodo_types = [
        _get_numba_typ_from_pa_typ(pa_schema.field(col_name), False, True, None)[0]
        for col_name in col_names
    ]

    # Special MERGE INTO COW Handling for Row ID Column
    if is_merge_into_cow:
        col_names.append("_BODO_ROW_ID")
        bodo_types.append(types.Array(types.int64, 1, "C"))

    return (col_names, bodo_types, pa_schema)


def _determine_str_as_dict_columns(
    table: Table,
    str_col_names_to_check: list[str],
    final_schema: pa.Schema,
    snapshot_id: int = -1,
) -> set[str]:
    """
    Determine the set of string columns in an Iceberg table
    that must be read as dict-encoded string columns.
    This is done by probing some files (one file per rank)
    and checking if the compression would be beneficial.
    This handles schema evolution as well. In case the file
    chosen for probing doesn't have the column, we set
    the size as 0, i.e. encourage dictionary encoding.
    NOTE: Must be called in parallel on all ranks at compile
    time.

    Args:
        table: PyIceberg table object to scan
        str_col_names_to_check (list[str]): List of column names to check.
        final_schema (pa.Schema): The target/final Arrow
            schema of the Iceberg table.
        snapshot_id (int): Snapshot ID to use for the scan.

    Returns:
        set[str]: Set of column names that should be dict-encoded
            (subset of str_col_names_to_check).
    """

    comm = MPI.COMM_WORLD
    if len(str_col_names_to_check) == 0:
        return set()  # No string as dict columns

    # Get a small list of files. No filters are know at this time, so
    # no file pruning can be done.
    # XXX We should push down some type of file limit to the scanner
    #     to avoid retrieving millions of files for no reason
    all_file_tasks = list(
        table.scan(
            selected_fields=tuple(str_col_names_to_check),
            snapshot_id=snapshot_id if snapshot_id != -1 else None,
        ).plan_files()
    )

    # Take a sample of N files where N is the number of ranks. Each rank looks at
    # the metadata of a different random file
    if len(all_file_tasks) > bodo.get_size():
        # Create a new instance of Random so that the global state is not
        # affected.
        my_random = random.Random(37)
        sample_files = my_random.sample(all_file_tasks, bodo.get_size())
    else:
        sample_files = all_file_tasks

    # Get a PyArrow FS to use to read the files
    fs = _fs_from_file_path(sample_files[0].file.file_path, table.io)

    # Get the list of field IDs corresponding to the string columns
    str_col_names_to_check_set: set[str] = set(str_col_names_to_check)
    str_col_name_to_iceberg_field_id: dict[str, int] = {}
    for field in final_schema:
        if field.metadata is None or b_ICEBERG_FIELD_ID_MD_KEY not in field.metadata:
            raise BodoError(
                "iceberg.py::determine_str_as_dict_columns: Schema does not have Field IDs!"
            )
        if field.name in str_col_names_to_check_set:
            str_col_name_to_iceberg_field_id[field.name] = int(
                field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
            )
    assert len(str_col_name_to_iceberg_field_id.keys()) == len(str_col_names_to_check)

    # Map the field ID to the index of the column in str_col_names_to_check.
    str_col_field_id_to_idx: dict[int, int] = {
        str_col_name_to_iceberg_field_id[str_col_names_to_check[i]]: i
        for i in range(len(str_col_names_to_check))
    }

    # Use pq.ParquetFile to open the file assigned to this rank.
    # Then, find the columns corresponding to the field IDs and get their
    # statistics. If the column doesn't exist, the uncompressed size will
    # implicitly be 0, i.e. encourage dict encoding.
    total_uncompressed_sizes = np.zeros(len(str_col_names_to_check), dtype=np.int64)
    total_uncompressed_sizes_recv = np.zeros(
        len(str_col_names_to_check), dtype=np.int64
    )
    if bodo.get_rank() < len(sample_files):
        fpath = sample_files[bodo.get_rank()]
        try:
            sanitized_path = fpath_without_protocol_prefix(fpath.file.file_path)
            pq_file = pq.ParquetFile(sanitized_path, filesystem=fs)
            metadata = pq_file.metadata
            for idx, field in enumerate(pq_file.schema_arrow):
                if (
                    field.metadata is None
                    or b_ICEBERG_FIELD_ID_MD_KEY not in field.metadata
                ):
                    raise BodoError(
                        f"Iceberg Parquet File ({fpath}) does not have Field IDs!"
                    )
                field_id: int = int(field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])
                if field_id in str_col_field_id_to_idx:
                    for i in range(pq_file.num_row_groups):
                        total_uncompressed_sizes[str_col_field_id_to_idx[field_id]] += (
                            metadata.row_group(i).column(idx).total_uncompressed_size
                        )
            num_rows = metadata.num_rows
        except Exception as e:
            if isinstance(e, (OSError, FileNotFoundError)):
                # Skip the path that produced the error (error will be reported at runtime)
                num_rows = 0
            else:
                raise
    else:
        num_rows = 0
    total_rows = comm.allreduce(num_rows, op=MPI.SUM)
    if total_rows == 0:
        return set()  # no string as dict columns
    comm.Allreduce(total_uncompressed_sizes, total_uncompressed_sizes_recv, op=MPI.SUM)
    str_column_metrics = total_uncompressed_sizes_recv / total_rows
    str_as_dict: set[str] = set()
    for i, metric in enumerate(str_column_metrics):
        # Don't import as `from ... import READ_STR_AS_DICT_THRESHOLD`
        # because this will break a test
        if metric < bodo.io.parquet_pio.READ_STR_AS_DICT_THRESHOLD:
            str_as_dict.add(str_col_names_to_check[i])
    return str_as_dict


def get_iceberg_orig_schema(
    conn_str: str,
    table_id: str,
    is_merge_into_cow: bool = False,
) -> tuple[list[str], list[types.ArrayCompatible], pa.Schema]:
    """
    Helper function to fetch Bodo types for an Iceberg table with the given
    connection info. Will include an additional Row ID column for MERGE INTO
    COW operations.

    Returns:
        - List of column names
        - List of column Bodo types
        - PyArrow Schema Object
    """
    _ = verify_pyiceberg_installed()

    # Get the pyarrow schema from Iceberg
    catalog = conn_str_to_catalog(conn_str)
    table = catalog.load_table(table_id)
    return _get_table_schema(table, is_merge_into_cow)


def get_orig_and_runtime_schema(
    conn_str: str,
    table_id: str,
    selected_cols: list[str] | None = None,
    read_as_dict_cols: list[str] = EMPTY_LIST,
    detect_dict_cols: bool = False,
    is_merge_into_cow: bool = False,
    snapshot_id: int = -1,
) -> tuple[
    # Original Schemas
    list[str],
    list[types.ArrayCompatible],
    pa.Schema,
    # Runtime Schema
    list[str],
    list[types.ArrayCompatible],
]:
    _ = verify_pyiceberg_installed()

    import pyiceberg.exceptions

    try:
        catalog = conn_str_to_catalog(conn_str)
        table = catalog.load_table(table_id)
    except pyiceberg.exceptions.NoSuchTableError as e:
        raise BodoError("No such Iceberg table found") from e

    orig_col_names, orig_col_types, pa_schema = _get_table_schema(
        table, is_merge_into_cow, snapshot_id=snapshot_id
    )

    orig_col_name_to_type = dict(zip(orig_col_names, orig_col_types))
    col_names = orig_col_names if selected_cols is None else selected_cols
    col_types = (
        orig_col_types
        if selected_cols is None
        else [orig_col_name_to_type[c] for c in col_names]
    )

    # ----------- Get dictionary-encoded string columns ----------- #

    # 1) Check user-provided dict-encoded columns for errors
    for c in read_as_dict_cols:
        if c not in orig_col_name_to_type:
            raise BodoError(
                f"pandas.read_sql_table(): column name '{c}' in _bodo_read_as_dict is not in table columns {col_names}"
            )
        if orig_col_name_to_type[c] != bodo.types.string_array_type:
            raise BodoError(
                f"pandas.read_sql_table(): column name '{c}' in _bodo_read_as_dict is not a string column"
            )

    all_dict_str_cols = set(read_as_dict_cols)
    if detect_dict_cols:
        # Estimate which string columns should be dict-encoded using existing Parquet
        # infrastructure.
        str_columns: list[str] = bodo.io.parquet_pio.get_str_columns_from_pa_schema(
            pa_schema
        )
        # remove user provided dict-encoded columns
        str_columns = list(
            set(str_columns).intersection(col_names) - set(read_as_dict_cols)
        )
        # Sort the columns to ensure same order on all ranks
        str_columns = sorted(str_columns)

        dict_str_cols = _determine_str_as_dict_columns(
            table,
            str_columns,
            pa_schema,
            snapshot_id=snapshot_id,
        )
        all_dict_str_cols.update(dict_str_cols)

    # Change string array types to dict-encoded
    col_name_to_idx = {c: i for i, c in enumerate(col_names)}
    for c in all_dict_str_cols:
        col_types[col_name_to_idx[c]] = bodo.types.dict_str_arr_type

    return orig_col_names, orig_col_types, pa_schema, col_names, col_types


def resolve_snapshot_id(
    con_str: str, table_id: str, snapshot_id: int = -1, snapshot_timestamp_ms: int = -1
) -> int:
    """
    Resolve the snapshot ID of an Iceberg table.
    If `snapshot_id` is -1, and snapshot_timestamp_ms is -1, -1 is returned.
    If `snapshot_id` is -1 and snapshot_timestamp_ms is not -1, the snapshot ID corresponding to the
    given timestamp is returned.
    If `snapshot_id` is not -1, it is validated and returned.

    Args:
        con_str (str): Iceberg connection string
        table_id (str): Iceberg table ID
        snapshot_id (int): Snapshot ID to resolve. Default is -1.
        snapshot_timestamp_ms (int): Snapshot timestamp in milliseconds to resolve. Default is -1.

    Returns:
        int: Snapshot ID to use
    """
    _ = verify_pyiceberg_installed()

    import pyiceberg.exceptions

    try:
        catalog = conn_str_to_catalog(con_str)
        table = catalog.load_table(table_id)
    except pyiceberg.exceptions.NoSuchTableError as e:
        raise BodoError("No such Iceberg table found") from e
    if snapshot_id != -1 and snapshot_timestamp_ms != -1:
        raise BodoError(
            "resolve_snapshot_id: Cannot specify both snapshot_id and snapshot_timestamp_ms"
        )
    snapshot = None
    if snapshot_timestamp_ms != -1:
        snapshot = table.snapshot_as_of_timestamp(snapshot_timestamp_ms)
        if snapshot is None:
            raise BodoError(
                f"Snapshot with timestamp {snapshot_timestamp_ms} not found in table {table_id}"
            )
    elif snapshot_id != -1:
        snapshot = table.snapshot_by_id(snapshot_id)
        if snapshot is None:
            raise BodoError(
                f"Snapshot with ID {snapshot_id} not found in table {table_id}"
            )
    snapshot_id = snapshot.snapshot_id if snapshot is not None else -1
    return snapshot_id
