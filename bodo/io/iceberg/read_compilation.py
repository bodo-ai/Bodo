"""
Operations to get basic metadata about an Iceberg table
needed for compilation. For example, the table schema
"""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba.core import types

import bodo
from bodo.io.helpers import _get_numba_typ_from_pa_typ
from bodo.io.iceberg.common import b_ICEBERG_FIELD_ID_MD_KEY, get_iceberg_fs
from bodo.io.iceberg.read_metadata import get_iceberg_file_list_parallel
from bodo.io.parquet_pio import (
    get_fpath_without_protocol_prefix,
    parse_fpath,
)
from bodo.mpi4py import MPI
from bodo.utils.utils import BodoError, run_rank0


def get_iceberg_type_info(
    table_name: str,
    con: str,
    database_schema: str,
    is_merge_into_cow: bool = False,
):
    """
    Helper function to fetch Bodo types for an Iceberg table with the given
    connection info. Will include an additional Row ID column for MERGE INTO
    COW operations.

    Returns:
        - List of column names
        - List of column Bodo types
        - PyArrow Schema Object
    """
    import bodo_iceberg_connector

    # In the case that we encounter an error, we store the exception in col_names_or_err
    col_names_or_err = None
    col_types = None
    pyarrow_schema = None

    # Only runs on rank 0, so we add no cover to avoid coverage warning
    if bodo.get_rank() == 0:  # pragma: no cover
        try:
            (
                col_names_or_err,
                col_types,
                pyarrow_schema,
            ) = bodo_iceberg_connector.get_iceberg_typing_schema(
                con, database_schema, table_name
            )

            if pyarrow_schema is None:
                raise BodoError("No such Iceberg table found")

        except bodo_iceberg_connector.IcebergError as e:
            col_names_or_err = BodoError(
                f"Failed to Get Typing Info from Iceberg Table: {e.message}"
            )

    comm = MPI.COMM_WORLD
    col_names_or_err = comm.bcast(col_names_or_err)
    if isinstance(col_names_or_err, Exception):
        raise col_names_or_err

    col_names = col_names_or_err
    col_types = comm.bcast(col_types)
    pyarrow_schema = comm.bcast(pyarrow_schema)

    bodo_types = [
        _get_numba_typ_from_pa_typ(typ, False, True, None)[0] for typ in col_types
    ]

    # Special MERGE INTO COW Handling for Row ID Column
    if is_merge_into_cow:
        col_names.append("_BODO_ROW_ID")
        bodo_types.append(types.Array(types.int64, 1, "C"))

    return (col_names, bodo_types, pyarrow_schema)


def is_snowflake_managed_iceberg_wh(con: str) -> bool:
    """
    Does the connection string correspond to a Snowflake-managed
    Iceberg catalog.

    Args:
        con (str): Iceberg connection string

    Returns:
        bool: Whether it's a Snowflake-managed Iceberg catalog.
    """
    import bodo_iceberg_connector

    catalog_type, _ = run_rank0(bodo_iceberg_connector.parse_iceberg_conn_str)(con)
    return catalog_type == "snowflake"


def determine_str_as_dict_columns(
    conn: str,
    database_schema: str,
    table_name: str,
    str_col_names_to_check: list[str],
    final_schema: pa.Schema,
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
        conn (str): Iceberg connection string.
        database_schema (str): Iceberg database that the table
            is in.
        table_name (str): Table name to read.
        str_col_names_to_check (list[str]): List of column
            names to check.
        final_schema (pa.Schema): The target/final Arrow
            schema of the Iceberg table.

    Returns:
        set[str]: Set of column names that should be dict-encoded
            (subset of str_col_names_to_check).
    """
    comm = MPI.COMM_WORLD
    if len(str_col_names_to_check) == 0:
        return set()  # No string as dict columns

    # Get list of files. No filters are know at this time, so
    # no file pruning can be done.
    # XXX We should push down some type of file limit to the
    # Iceberg Java Library to avoid retrieving millions of files
    # for no reason.
    all_pq_infos = get_iceberg_file_list_parallel(
        conn, database_schema, table_name, filters=None
    )[0]
    # Take a sample of N files where N is the number of ranks. Each rank looks at
    # the metadata of a different random file
    if len(all_pq_infos) > bodo.get_size():
        # Create a new instance of Random so that the global state is not
        # affected.
        my_random = random.Random(37)
        pq_infos = my_random.sample(all_pq_infos, bodo.get_size())
    else:
        pq_infos = all_pq_infos
    pq_abs_path_file_list = [pq_info.standard_path for pq_info in pq_infos]

    pq_abs_path_file_list, parse_result, protocol = parse_fpath(pq_abs_path_file_list)

    fs = get_iceberg_fs(
        protocol, conn, database_schema, table_name, pq_abs_path_file_list
    )
    pq_abs_path_file_list, _ = get_fpath_without_protocol_prefix(
        pq_abs_path_file_list, protocol, parse_result
    )

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
    if bodo.get_rank() < len(pq_abs_path_file_list):
        fpath = pq_abs_path_file_list[bodo.get_rank()]
        try:
            pq_file = pq.ParquetFile(fpath, filesystem=fs)
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
