# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
File that contains the main functionality for the Iceberg
integration within the Bodo repo. This does not contain the
main IR transformation.
"""
from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from mpi4py import MPI
from numba.core import types
from numba.extending import intrinsic

import bodo
from bodo.io.fs_io import get_s3_bucket_region_njit
from bodo.io.helpers import _get_numba_typ_from_pa_typ, pyarrow_schema_type
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    py_table_to_cpp_table,
)
from bodo.libs.str_ext import unicode_to_utf8
from bodo.utils import tracing
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import BodoError

# ------------------------------ Constants ------------------------------ #


# ===================================================================
# Must match the values in bodo_iceberg_connector/schema_helper.py
# ===================================================================
# This is the key used for storing the Iceberg Field ID in the
# metadata of the Arrow fields.
# Taken from: https://github.com/apache/arrow/blob/c23a097965b5c626cbc91b229c76a6c13d36b4e8/cpp/src/parquet/arrow/schema.cc#L245.
ICEBERG_FIELD_ID_MD_KEY = "PARQUET:field_id"

# PyArrow stores the metadata keys and values as bytes, so we need
# to use this encoded version when trying to access existing
# metadata in fields.
b_ICEBERG_FIELD_ID_MD_KEY = str.encode(ICEBERG_FIELD_ID_MD_KEY)
# ===================================================================

# ----------------------------- Helper Funcs ---------------------------- #


def format_iceberg_conn(conn_str: str) -> str:
    """
    Determine if connection string points to an Iceberg database and reconstruct
    the correct connection string needed to connect to the Iceberg metastore
    """

    parse_res = urlparse(conn_str)
    if not conn_str.startswith("iceberg+glue") and parse_res.scheme not in (
        "iceberg",
        "iceberg+file",
        "iceberg+s3",
        "iceberg+thrift",
        "iceberg+http",
        "iceberg+https",
        "iceberg+snowflake",
    ):
        raise BodoError(
            "'con' must start with one of the following: 'iceberg://', 'iceberg+file://', "
            "'iceberg+s3://', 'iceberg+thrift://', 'iceberg+http://', 'iceberg+https://', 'iceberg+glue', 'iceberg+snowflake'"
        )

    # Remove Iceberg Prefix when using Internally
    # For support before Python 3.9
    # TODO: Remove after deprecating Python 3.8
    if sys.version_info.minor < 9:  # pragma: no cover
        if conn_str.startswith("iceberg+"):
            conn_str = conn_str[len("iceberg+") :]
        if conn_str.startswith("iceberg://"):
            conn_str = conn_str[len("iceberg://") :]
    else:
        conn_str = conn_str.removeprefix("iceberg+").removeprefix("iceberg://")

    return conn_str


@numba.njit
def format_iceberg_conn_njit(conn_str):  # pragma: no cover
    """
    njit wrapper around format_iceberg_conn

    Args:
        conn_str (str): connection string passed in read_sql/read_sql_table/to_sql

    Returns:
        str: connection string without the iceberg(+*?) prefix
    """
    with numba.objmode(conn_str="unicode_type"):
        conn_str = format_iceberg_conn(conn_str)
    return conn_str


# ----------------------------- Iceberg Read -----------------------------#
def get_iceberg_type_info(
    table_name: str, con: str, database_schema: str, is_merge_into_cow: bool = False
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

    # Sonar cube only runs on rank 0, so we add no cover to avoid the warning
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


def get_iceberg_file_list(
    table_name: str, conn: str, database_schema: str, filters
) -> Tuple[List[str], List[str]]:
    """
    Gets the list of parquet data files that need to be read from an Iceberg table.

    We also pass filters, which is in DNF format and the output of filter
    pushdown to Iceberg. Iceberg will use this information to
    prune any files that it can from just metadata, so this
    is an "inclusive" projection.

    Returns:
        - List of file paths from Iceberg sanitized to be used by Bodo
            - Convert S3A paths to S3 paths
            - Convert relative paths to absolute paths
        - List of original file paths directly from Iceberg
    """
    import bodo_iceberg_connector

    assert (
        bodo.get_rank() == 0
    ), "get_iceberg_file_list should only ever be called on rank 0, as the operation requires access to the py4j server, which is only available on rank 0"

    try:
        res = bodo_iceberg_connector.bodo_connector_get_parquet_file_list(
            conn, database_schema, table_name, filters
        )
    except bodo_iceberg_connector.IcebergError as e:
        raise BodoError(
            f"Failed to Get List of Parquet Data Files from Iceberg Table: {e.message}"
        )

    return res


def get_iceberg_snapshot_id(table_name: str, conn: str, database_schema: str):
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
        snapshot_id = bodo_iceberg_connector.bodo_connector_get_current_snapshot_id(
            conn,
            database_schema,
            table_name,
        )
    except bodo_iceberg_connector.IcebergError as e:
        raise BodoError(
            f"Failed to Get the Snapshot ID from an Iceberg Table: {e.message}"
        )

    return snapshot_id


class IcebergParquetDataset:
    """
    Store dataset info in the way expected by Arrow reader in C++.
    This is essentially a wrapper around the ParquetDataset
    object that we generate in our parquet infrastructure
    (see get_parquet_dataset in parquet_pio.py).
    conn, database schema, table_name and pa_table_schema
    are required.
    'pa_table_schema' is the PyArrow schema object
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
        pq_file_list,
        snapshot_id,
        pq_dataset=None,
    ):
        self.pq_dataset = pq_dataset
        self.conn = conn
        self.database_schema = database_schema
        self.table_name = table_name
        self.schema = pa_table_schema
        # List of files exactly as given by Iceberg. This is used for operations like delete/merge.
        # There files are likely the relative paths to the Iceberg table for local files.
        #
        # For example if the absolute path was /Users/bodo/iceberg_db/my_table/part01.pq
        # and the iceberg directory is iceberg_db, then the path in the list would be
        # iceberg_db/my_table/part01.pq.
        self.file_list = pq_file_list
        # Snapshot id. This is used for operations like delete/merge.
        self.snapshot_id = snapshot_id

        # For the 0-file case, i.e. pq_dataset is None
        self.pieces = []
        self._bodo_total_rows = 0
        self._prefix = ""
        self.filesystem = None

        # If pq_dataset is provided, get these properties from it
        if pq_dataset is not None:
            self.pieces = pq_dataset.pieces
            self._bodo_total_rows = pq_dataset._bodo_total_rows
            self._prefix = pq_dataset._prefix
            # this is the filesystem that will be used to read data
            self.filesystem = pq_dataset.filesystem


def get_iceberg_pq_dataset(
    conn: str,
    database_schema: str,
    table_name: str,
    typing_pa_table_schema: pa.Schema,
    dnf_filters=None,
    expr_filters=None,
    tot_rows_to_read=None,
    is_parallel=False,
    get_row_counts=True,
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
    'get_row_counts' : flag for getting row counts of Parquet files and validate
        schemas, passed directly to get_parquet_dataset(). only needed during runtime.
    """
    # Only gather tracing if we are at runtime and tracing is turned on
    do_tracing = get_row_counts and tracing.is_tracing()
    if do_tracing:  # pragma: no cover
        ev = tracing.Event("get_iceberg_pq_dataset")

    comm = MPI.COMM_WORLD

    pq_abs_path_file_list_or_e = None
    snapshot_id_or_e = None
    iceberg_relative_path_file_list = None
    # Get the list on just one rank to reduce JVM overheads
    # and general traffic to table for when there are
    # catalogs in the future.

    # Always get the list on rank 0 to avoid the need
    # to initialize a full JVM + gateway server on every rank.
    # Sonar cube only runs on rank 0, so we add no cover to avoid the warning
    if bodo.get_rank() == 0:  # pragma: no cover
        if do_tracing:  # pragma: no cover
            ev_iceberg_fl = tracing.Event("get_iceberg_file_list", is_parallel=False)
            ev_iceberg_fl.add_attribute("g_dnf_filter", str(dnf_filters))
        try:
            # We return two list of Iceberg files. pq_abs_path_file_list_or_e contains the full
            # paths that can be used to read individual files. iceberg_relative_path_file_list contains
            # the path as given exactly by Iceberg, which may be a relative path for local files.
            #
            # For example if the absolute path was /Users/bodo/iceberg_db/my_table/part01.pq
            # and the iceberg directory is iceberg_db, then the path in pq_abs_path_file_list_or_e would be
            # /Users/bodo/iceberg_db/my_table/part01.pq and the path in iceberg_relative_path_file_list
            # would be iceberg_db/my_table/part01.pq.
            (
                pq_abs_path_file_list_or_e,
                iceberg_relative_path_file_list,
            ) = get_iceberg_file_list(table_name, conn, database_schema, dnf_filters)
            if do_tracing:  # pragma: no cover
                ICEBERG_TRACING_NUM_FILES_TO_LOG = int(
                    os.environ.get("BODO_ICEBERG_TRACING_NUM_FILES_TO_LOG", "50")
                )
                ev_iceberg_fl.add_attribute(
                    "num_files", len(pq_abs_path_file_list_or_e)
                )
                ev_iceberg_fl.add_attribute(
                    f"first_{ICEBERG_TRACING_NUM_FILES_TO_LOG}_files",
                    ", ".join(
                        pq_abs_path_file_list_or_e[:ICEBERG_TRACING_NUM_FILES_TO_LOG]
                    ),
                )
        except Exception as e:  # pragma: no cover
            pq_abs_path_file_list_or_e = e
        if do_tracing:  # pragma: no cover
            ev_iceberg_fl.finalize()
            ev_iceberg_snapshot = tracing.Event("get_snapshot_id", is_parallel=False)
        try:
            snapshot_id_or_e = get_iceberg_snapshot_id(
                table_name, conn, database_schema
            )
        except Exception as e:  # pragma: no cover
            snapshot_id_or_e = e
        if do_tracing:  # pragma: no cover
            ev_iceberg_snapshot.finalize()

    # Send list to all ranks
    (
        pq_abs_path_file_list_or_e,
        snapshot_id_or_e,
        iceberg_relative_path_file_list,
    ) = comm.bcast(
        (pq_abs_path_file_list_or_e, snapshot_id_or_e, iceberg_relative_path_file_list)
    )

    # raise error on all processors if found (not just rank 0 which would cause hangs)
    if isinstance(pq_abs_path_file_list_or_e, Exception):
        error = pq_abs_path_file_list_or_e
        raise BodoError(
            f"Error reading Iceberg Table: {type(error).__name__}: {str(error)}\n"
        )
    if isinstance(snapshot_id_or_e, Exception):
        error = snapshot_id_or_e
        raise BodoError(
            f"Error reading Iceberg Table: {type(error).__name__}: {str(error)}\n"
        )

    pq_abs_path_file_list: List[str] = pq_abs_path_file_list_or_e
    snapshot_id: int = snapshot_id_or_e

    if len(pq_abs_path_file_list) == 0:
        # In case all files were filtered out by Iceberg, we need to
        # build an empty DataFrame, but with the right schema.
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
                pq_abs_path_file_list,
                get_row_counts=get_row_counts,
                # dnf_filters were already handled by Iceberg
                # We only pass expr_filters if we don't need to load the whole file.
                expr_filters=expr_filters,
                is_parallel=is_parallel,
                # Provide baseline schema to detect schema evolution
                typing_pa_schema=typing_pa_table_schema,
                # Iceberg has its own partitioning info and stores
                # partition columns in the parquet files, so we
                # tell Arrow not to detect partitioning
                partitioning=None,
                # Iceberg Limit Pushdown
                tot_rows_to_read=tot_rows_to_read,
            )

        except BodoError as e:
            if re.search(r"Schema .* was different", str(e), re.IGNORECASE):
                raise BodoError(
                    f"Bodo currently doesn't support reading Iceberg tables with schema evolution.\n{e}"
                )
            else:
                raise

    iceberg_pq_dataset = IcebergParquetDataset(
        conn,
        database_schema,
        table_name,
        typing_pa_table_schema,
        iceberg_relative_path_file_list,
        snapshot_id,
        pq_dataset,
    )
    if do_tracing:  # pragma: no cover
        ev.finalize()

    return iceberg_pq_dataset


# ----------------------------- Iceberg Write ----------------------------- #
def are_schemas_compatible(
    pa_schema: pa.Schema, df_schema: pa.Schema, allow_downcasting: bool = False
) -> bool:
    """
    Check if the input Dataframe schema is compatible with the Iceberg table's
    schema for append-like operations (including MERGE INTO). Compatibility
    consists of the following:
    - The df_schema either has the same columns as pa_schema or is only missing
      optional columns
    - Every column C from df_schema with a matching column C' from pa_schema is
      compatible, where compatibility is:
        - C and C' have the same datatype
        - C and C' are both nullable or both non-nullable
        - C is not-nullable and C' is nullable
        - C is an int64 while C' is an int32 (if allow_downcasting is True)
        - C is an float64 while C' is an float32 (if allow_downcasting is True)
        - C is nullable while C' is non-nullable (if allow_downcasting is True)

    Note that allow_downcasting should be used if the output Dataframe df will be
    casted to fit pa_schema (making sure there are no nulls, downcasting arrays).
    """
    if pa_schema.equals(df_schema):
        return True

    # If the schemas are not the same size, it is still possible that the dataframe
    # can be appended iff the dataframe schema is a subset of the iceberg schema and
    # each missing field is optional
    if len(df_schema) < len(pa_schema):
        # Create a new "subset" pa schema that only contains the fields that are
        # in the dataframe schema
        subset_fields = []
        for pa_field in pa_schema:
            df_field = df_schema.field_by_name(pa_field.name)

            # Don't include the field only if it isn't in the dataframe schema
            # and it is nullable
            if not (df_field is None and pa_field.nullable):
                subset_fields.append(pa_field)
        pa_schema = pa.schema(subset_fields)

    if len(pa_schema) != len(df_schema):
        return False

    # Compare each field individually for "compatibility"
    # Only the dataframe schema is potentially modified during this step
    for idx in range(len(df_schema)):
        df_field = df_schema.field(idx)
        pa_field = pa_schema.field(idx)

        if df_field.equals(pa_field):
            continue

        df_type = df_field.type
        pa_type = pa_field.type

        # df_field can only be downcasted as of now
        # TODO: Should support upcasting in the future if necessary
        if (
            not df_type.equals(pa_type)
            and allow_downcasting
            and (
                (
                    pa.types.is_signed_integer(df_type)
                    and pa.types.is_signed_integer(pa_type)
                )
                or (pa.types.is_floating(df_type) and pa.types.is_floating(pa_type))
            )
            and df_type.bit_width > pa_type.bit_width
        ):
            df_field = df_field.with_type(pa_type)

        if not df_field.nullable and pa_field.nullable:
            df_field = df_field.with_nullable(True)
        elif allow_downcasting and df_field.nullable and not pa_field.nullable:
            df_field = df_field.with_nullable(False)

        df_schema = df_schema.set(idx, df_field)

    return df_schema.equals(pa_schema)


def with_iceberg_field_id_md(
    field: pa.Field, next_field_id: int
) -> tuple[pa.Field, int]:
    """
    Adds/Updates Iceberg Field IDs in the PyArrow field's metadata.
    This field will be assigned the field ID 'next_field_id'.
    'next_field_id' will then be updated and returned so that the next field
    ID assignment uses a unique ID.
    In the case of nested types, we recurse and assign unique field IDs to
    the child fields as well.

    Args:
        field (pa.Field): Original field
        next_field_id (list[int]): Next available field ID.

    Returns:
        tuple[pa.Field, int]:
            - New field with the field ID added to the field
            metadata (including all the child fields).
            - Next available field ID after assigning field
            ID to this field and all its child fields.
    """
    # Construct new metadata for this field:
    new_md = {} if field.metadata is None else deepcopy(field.metadata)
    new_md.update({b_ICEBERG_FIELD_ID_MD_KEY: str(next_field_id)})
    next_field_id += 1

    new_field: pa.Field = None
    # Recurse in the nested data type case:
    if pa.types.is_list(field.type):
        new_element_field, next_field_id = with_iceberg_field_id_md(
            field.type.field(0), next_field_id
        )
        new_field = field.with_type(pa.list_(new_element_field)).with_metadata(new_md)
    elif pa.types.is_fixed_size_list(field.type):
        new_element_field, next_field_id = with_iceberg_field_id_md(
            field.type.field(0), next_field_id
        )
        new_field = field.with_type(
            pa.list_(new_element_field, list_size=field.type.list_size)
        ).with_metadata(new_md)
    elif pa.types.is_large_list(field.type):
        new_element_field, next_field_id = with_iceberg_field_id_md(
            field.type.field(0), next_field_id
        )
        new_field = field.with_type(pa.large_list(new_element_field)).with_metadata(
            new_md
        )
    elif pa.types.is_struct(field.type):
        new_children_fields = []
        for _, child_field in enumerate(field.type):
            new_child_field, next_field_id = with_iceberg_field_id_md(
                child_field, next_field_id
            )
            new_children_fields.append(new_child_field)
        new_field = field.with_type(pa.struct(new_children_fields)).with_metadata(
            new_md
        )
    elif pa.types.is_map(field.type):
        new_key_field, next_field_id = with_iceberg_field_id_md(
            field.type.key_field, next_field_id
        )
        new_item_field, next_field_id = with_iceberg_field_id_md(
            field.type.item_field, next_field_id
        )
        new_field = field.with_type(
            pa.map_(new_key_field, new_item_field)
        ).with_metadata(new_md)
    else:
        new_field = field.with_metadata(new_md)
    return new_field, next_field_id


def with_iceberg_field_id_md_from_ref_field(
    field: pa.Field, ref_field: pa.Field
) -> pa.Field:
    """
    Adds/Updates Iceberg Field IDs in the PyArrow field's metadata
    based on the reference field. The reference field must have
    the field ID in its metadata.
    In the case of nested types, we recurse and assign unique field IDs to
    the child fields as well.

    Args:
        field (pa.Field): Original field
        ref_field (pa.Field): Reference field to get the Iceberg field ID from.

    Returns:
        pa.Field:  New field with the field ID added to the field
            metadata (including all the child fields).
    """
    assert ref_field.metadata is not None, ref_field
    assert b_ICEBERG_FIELD_ID_MD_KEY in ref_field.metadata, (
        ref_field,
        ref_field.metadata,
    )
    # Construct new metadata for this field:
    new_md = {} if field.metadata is None else deepcopy(field.metadata)
    new_md.update(
        {b_ICEBERG_FIELD_ID_MD_KEY: ref_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]}
    )

    new_field: pa.Field = None
    # Recurse in the nested data type case:
    if pa.types.is_list(field.type):
        assert pa.types.is_list(ref_field.type), ref_field
        new_value_field = with_iceberg_field_id_md_from_ref_field(
            field.type.value_field, ref_field.type.value_field
        )
        new_field = field.with_type(pa.list_(new_value_field)).with_metadata(new_md)
    elif pa.types.is_fixed_size_list(field.type):
        assert pa.types.is_fixed_size_list(ref_field.type), ref_field
        new_value_field = with_iceberg_field_id_md_from_ref_field(
            field.type.value_field, ref_field.type.value_field
        )
        new_field = field.with_type(
            pa.list_(new_value_field, list_size=field.type.list_size)
        ).with_metadata(new_md)
    elif pa.types.is_large_list(field.type):
        assert pa.types.is_large_list(ref_field.type), ref_field
        new_value_field = with_iceberg_field_id_md_from_ref_field(
            field.type.value_field, ref_field.type.value_field
        )
        new_field = field.with_type(pa.large_list(new_value_field)).with_metadata(
            new_md
        )
    elif pa.types.is_struct(field.type):
        assert pa.types.is_struct(ref_field.type), ref_field
        new_children_fields = []
        for i in range(field.type.num_fields):
            new_children_fields.append(
                with_iceberg_field_id_md_from_ref_field(
                    field.type.field(i), ref_field.type.field(i)
                )
            )
        new_field = field.with_type(pa.struct(new_children_fields)).with_metadata(
            new_md
        )
    elif pa.types.is_map(field.type):
        assert pa.types.is_map(ref_field.type), ref_field
        new_key_field = with_iceberg_field_id_md_from_ref_field(
            field.type.key_field, ref_field.type.key_field
        )
        new_item_field = with_iceberg_field_id_md_from_ref_field(
            field.type.item_field, ref_field.type.item_field
        )
        new_field = field.with_type(
            pa.map_(new_key_field, new_item_field)
        ).with_metadata(new_md)
    else:
        new_field = field.with_metadata(new_md)
    return new_field


def add_iceberg_field_id_md_to_pa_schema(
    schema: pa.Schema, ref_schema: Optional[pa.Schema] = None
) -> pa.Schema:
    """
    Create a new Schema where all the fields (including nested fields)
    have their Iceberg Field ID in the field metadata.
    If a reference schema is provided (append case), copy over
    the field IDs from that schema, else (create/replace case) assign new IDs.
    In the latter (create/replace; no ref schema) case, we call the Iceberg
    Java library to assign the field IDs to ensure consistency
    with the field IDs that will be assigned when creating the
    table metadata. See the docstring of BodoIcebergHandler.getInitSchema
    (in BodoIcebergHandler.java) for a more detailed explanation of why
    this is required.

    Args:
        schema (pa.Schema): Original schema (possibly without the Iceberg
            field IDs in the fields' metadata).
        ref_schema (Optional[pa.Schema], optional): Reference schema
            to use in the append case. If provided, all fields
            must have their Iceberg Field ID in the field metadata,
            including all the nested fields. Defaults to None.

    Returns:
        pa.Schema: Schema with Iceberg Field IDs correctly assigned
            in the metadata of all its fields.
    """
    import bodo_iceberg_connector as connector

    if ref_schema is None:
        new_fields = []
        next_field_id = 1
        # Add dummy IDs. Note that we need the IDs to be semantically
        # correct, i.e. we can't set all field IDs to the same number
        # since there's a validation step during conversion to a
        # Iceberg Schema object in get_schema_with_init_field_ids.
        for idx in range(len(schema)):
            new_field, next_field_id = with_iceberg_field_id_md(
                schema.field(idx), next_field_id
            )
            new_fields.append(new_field)
        intermediate_schema = pa.schema(new_fields)
        return connector.get_schema_with_init_field_ids(intermediate_schema)
    else:
        new_fields = []
        for field in schema:
            ref_field = ref_schema.field(field.name)
            assert ref_field is not None, field
            new_field = with_iceberg_field_id_md_from_ref_field(field, ref_field)
            new_fields.append(new_field)
        return pa.schema(new_fields)


def get_table_details_before_write(
    table_name: str,
    conn: str,
    database_schema: str,
    df_schema: pa.Schema,
    if_exists: str,
    allow_downcasting: bool = False,
):
    """
    Wrapper around bodo_iceberg_connector.get_typing_info to perform
    dataframe typechecking, collect typing-related information for
    Iceberg writes, fill in nulls, and project across all ranks.
    """

    ev = tracing.Event("iceberg_get_table_details_before_write")

    import bodo_iceberg_connector as connector

    comm = MPI.COMM_WORLD

    comm_exc = None
    iceberg_schema_id = None
    table_loc = ""
    partition_spec = []
    sort_order = []
    iceberg_schema_str = ""
    pa_schema = None

    # Map column name to index for efficient lookup
    col_name_to_idx_map = {col: i for (i, col) in enumerate(df_schema.names)}

    # Communicate with the connector to check if the table exists.
    # It will return the warehouse location, iceberg-schema-id,
    # pyarrow-schema, iceberg-schema (as a string, so it can be written
    # to the schema metadata in the parquet files), partition-spec
    # and sort-order.
    if comm.Get_rank() == 0:
        try:
            (
                table_loc,
                iceberg_schema_id,
                pa_schema,
                iceberg_schema_str,
                partition_spec,
                sort_order,
            ) = connector.get_typing_info(conn, database_schema, table_name)

            if if_exists != "append":
                # In the create/replace case, disregard some of the properties
                pa_schema = None
                iceberg_schema_str = ""
                partition_spec = []
                sort_order = []
            else:
                # Ensure that all column names in the partition spec and sort order are
                # in the dataframe being written
                for col_name, *_ in partition_spec:
                    assert (
                        col_name in col_name_to_idx_map
                    ), f"Iceberg Partition column {col_name} not found in dataframe"
                for col_name, *_ in sort_order:
                    assert (
                        col_name in col_name_to_idx_map
                    ), f"Iceberg Sort column {col_name} not found in dataframe"

                # Transform the partition spec and sort order tuples to convert
                # column name to index in Bodo table
                partition_spec = [
                    (col_name_to_idx_map[col_name], *rest)
                    for col_name, *rest in partition_spec
                ]

                sort_order = [
                    (col_name_to_idx_map[col_name], *rest)
                    for col_name, *rest in sort_order
                ]

                if (pa_schema is not None) and (
                    not are_schemas_compatible(pa_schema, df_schema, allow_downcasting)
                ):
                    # TODO: https://bodo.atlassian.net/browse/BE-4019
                    # for improving docs on Iceberg write support
                    if numba.core.config.DEVELOPER_MODE:
                        raise BodoError(
                            f"DataFrame schema needs to be an ordered subset of Iceberg table for append\n\n"
                            f"Iceberg:\n{pa_schema}\n\n"
                            f"DataFrame:\n{df_schema}\n"
                        )
                    else:
                        raise BodoError(
                            "DataFrame schema needs to be an ordered subset of Iceberg table for append"
                        )

            # Add Iceberg Field ID to the fields' metadata.
            # If we received an existing schema (pa_schema) in the append case,
            # then port over the existing field IDs, else generate new ones.
            df_schema = add_iceberg_field_id_md_to_pa_schema(
                df_schema, ref_schema=pa_schema
            )

            if (if_exists != "append") or (iceberg_schema_id is None):
                # When the table doesn't exist, i.e. we're creating a new one,
                # we need to create iceberg_schema_str from the PyArrow schema
                # of the dataframe.
                iceberg_schema_str = connector.pyarrow_to_iceberg_schema_str(df_schema)

        except connector.IcebergError as e:
            comm_exc = BodoError(e.message)
        except Exception as e:
            comm_exc = e

    comm_exc = comm.bcast(comm_exc)
    if isinstance(comm_exc, Exception):
        raise comm_exc

    table_loc = comm.bcast(table_loc)
    iceberg_schema_id = comm.bcast(iceberg_schema_id)
    partition_spec = comm.bcast(partition_spec)
    sort_order = comm.bcast(sort_order)
    iceberg_schema_str = comm.bcast(iceberg_schema_str)
    pa_schema = comm.bcast(pa_schema)
    df_schema = comm.bcast(df_schema)

    if iceberg_schema_id is None:
        already_exists = False
        iceberg_schema_id = -1
    else:
        already_exists = True

    ev.finalize()

    return (
        already_exists,
        table_loc,
        iceberg_schema_id,
        partition_spec,
        sort_order,
        iceberg_schema_str,
        # We use the expected schema at the parquet write step (in C++)
        # so we can reuse the df_schema for create and replace cases
        pa_schema if if_exists == "append" and pa_schema is not None else df_schema,
        # This has the Iceberg field IDs in the fields' metadata.
        df_schema,
    )


def collect_file_info(iceberg_files_info) -> Tuple[List[str], List[int], List[int]]:
    """
    Collect C++ Iceberg File Info to a single rank
    and process before handing off to the connector / committing functions
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    # Metrics to provide to Iceberg:
    # Required:
    # 1. record_count -- Number of records/rows in this file
    # 2. file_size_in_bytes -- Total file size in bytes

    # TODO [BE-3099] Metrics currently not provided:
    # Optional:
    # 3. column_sizes
    # 4. value_counts
    # 5. null_value_counts
    # 6. nan_value_counts
    # 7. distinct_counts
    # 8. lower_bounds
    # 9. upper_bounds

    # Collect the file names
    fnames_local = [x[0] for x in iceberg_files_info]
    fnames_lists = comm.gather(fnames_local)

    # Flatten the list of lists
    fnames = (
        [item for sub in fnames_lists for item in sub] if comm.Get_rank() == 0 else None
    )

    # Collect the metrics
    record_counts_local = np.array([x[1] for x in iceberg_files_info], dtype=np.int64)
    file_sizes_local = np.array([x[2] for x in iceberg_files_info], dtype=np.int64)
    record_counts = bodo.gatherv(record_counts_local).tolist()
    file_sizes = bodo.gatherv(file_sizes_local).tolist()

    return fnames, record_counts, file_sizes  # type: ignore


def register_table_write(
    conn_str: str,
    db_name: str,
    table_name: str,
    table_loc: str,
    fnames: List[str],
    all_metrics: Dict[str, List[Any]],  # TODO: Explain?
    iceberg_schema_id: int,
    pa_schema,
    partition_spec,
    sort_order,
    mode: str,
):
    """
    Wrapper around bodo_iceberg_connector.commit_write to run on
    a single rank and broadcast the result
    """
    ev = tracing.Event("iceberg_register_table_write")

    import bodo_iceberg_connector

    comm = MPI.COMM_WORLD

    success = False
    if comm.Get_rank() == 0:
        schema_id = None if iceberg_schema_id < 0 else iceberg_schema_id

        success = bodo_iceberg_connector.commit_write(
            conn_str,
            db_name,
            table_name,
            table_loc,
            fnames,
            all_metrics,
            schema_id,
            pa_schema,
            partition_spec,
            sort_order,
            mode,
        )

    success = comm.bcast(success)
    ev.finalize()
    return success


def register_table_merge_cow(
    conn_str: str,
    db_name: str,
    table_name: str,
    table_loc: str,
    old_fnames: List[str],
    new_fnames: List[str],
    all_metrics: Dict[str, List[Any]],  # TODO: Explain?
    snapshot_id: int,
):  # pragma: no cover
    """
    Wrapper around bodo_iceberg_connector.commit_merge_cow to run on
    a single rank and broadcast the result
    """
    ev = tracing.Event("iceberg_register_table_merge_cow")

    import bodo_iceberg_connector

    comm = MPI.COMM_WORLD

    success = False
    if comm.Get_rank() == 0:
        success = bodo_iceberg_connector.commit_merge_cow(
            conn_str,
            db_name,
            table_name,
            table_loc,
            old_fnames,
            new_fnames,
            all_metrics,
            snapshot_id,
        )

    success: bool = comm.bcast(success)
    ev.finalize()
    return success


from numba.extending import NativeValue, box, models, register_model, unbox


# TODO Use install_py_obj_class
class PythonListOfHeterogeneousTuples(types.Opaque):
    """
    It is just a Python object (list of tuples) to be passed to C++.
    Used for iceberg partition-spec, sort-order and iceberg-file-info
    descriptions.
    """

    def __init__(self):
        super(PythonListOfHeterogeneousTuples, self).__init__(
            name="PythonListOfHeterogeneousTuples"
        )


python_list_of_heterogeneous_tuples_type = PythonListOfHeterogeneousTuples()
types.python_list_of_heterogeneous_tuples_type = (  # type: ignore
    python_list_of_heterogeneous_tuples_type
)
register_model(PythonListOfHeterogeneousTuples)(models.OpaqueModel)


@unbox(PythonListOfHeterogeneousTuples)
def unbox_python_list_of_heterogeneous_tuples_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


@box(PythonListOfHeterogeneousTuples)
def box_python_list_of_heterogeneous_tuples_type(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return val


# Class for a PyObject that is a list.
this_module = sys.modules[__name__]
PyObjectOfList = install_py_obj_class(
    types_name="pyobject_of_list_type",
    python_type=None,
    module=this_module,
    class_name="PyObjectOfListType",
    model_name="PyObjectOfListModel",
)


@numba.njit
def iceberg_pq_write(
    table_loc,
    bodo_table,
    col_names,
    partition_spec,
    sort_order,
    iceberg_schema_str,
    is_parallel,
    expected_schema,
):  # pragma: no cover
    """
    Writes a table to Parquet files in an Iceberg table's data warehouse
    following Iceberg rules and semantics.
    Args:
        table_loc (str): Location of the data/ folder in the warehouse
        bodo_table: Table object to pass to C++
        col_names: Array object containing column names (passed to C++)
        partition_spec: Array of Tuples containing Partition Spec for Iceberg Table (passed to C++)
        sort_order: Array of Tuples containing Sort Order for Iceberg Table (passed to C++)
        iceberg_schema_str (str): JSON Encoding of Iceberg Schema to include in Parquet metadata
        is_parallel (bool): Whether the write is occurring on a distributed dataframe
        expected_schema (pyarrow.Schema): Expected schema of output PyArrow table written
            to Parquet files in the Iceberg table. None if not necessary

    Returns:
        Distributed list of written file info needed by Iceberg for committing
        1) file_path (after the table_loc prefix)
        2) record_count / Number of rows
        3) File size in bytes
        4) *partition-values
    """

    bucket_region = get_s3_bucket_region_njit(table_loc, is_parallel)
    # TODO [BE-3248] compression and row-group-size (and other properties)
    # should be taken from table properties
    # https://iceberg.apache.org/docs/latest/configuration/#write-properties
    # Using snappy and our row group size default for now
    compression = "snappy"
    rg_size = -1

    # Call the C++ function to write the parquet files.
    # Information about them will be returned as a list of tuples
    # See docstring for format
    iceberg_files_info = iceberg_pq_write_table_cpp(
        unicode_to_utf8(table_loc),
        bodo_table,
        col_names,
        partition_spec,
        sort_order,
        unicode_to_utf8(compression),
        is_parallel,
        unicode_to_utf8(bucket_region),
        rg_size,
        unicode_to_utf8(iceberg_schema_str),
        expected_schema,
    )

    return iceberg_files_info


@numba.njit
def iceberg_write(
    table_name,
    conn,
    database_schema,
    bodo_table,
    col_names,
    # Same semantics as pandas to_sql for now
    if_exists,
    is_parallel,
    df_pyarrow_schema,  # Additional Param to Compare Compile-Time and Iceberg Schema
    allow_downcasting=False,
):  # pragma: no cover
    """
    Iceberg Basic Write Implementation for parquet based tables.
    Args:
        table_name (str): name of iceberg table
        conn (str): connection string
        database_schema (str): schema in iceberg database
        bodo_table : table object to pass to c++
        col_names : array object containing column names (passed to c++)
        if_exists (str): behavior when table exists. must be one of ['fail', 'append', 'replace']
        is_parallel (bool): whether the write is occurring on a distributed dataframe
        df_pyarrow_schema (pyarrow.Schema): PyArrow schema of the dataframe being written
        allow_downcasting (bool): Perform write downcasting on table columns to fit Iceberg schema
            This includes both type and nullability downcasting

    Raises:
        ValueError, Exception, BodoError
    """

    ev = tracing.Event("iceberg_write_py", is_parallel)
    # Supporting REPL requires some refactor in the parquet write infrastructure,
    # so we're not implementing it for now. It will be added in a following PR.
    assert is_parallel, "Iceberg Write only supported for distributed dataframes"
    with numba.objmode(
        already_exists="bool_",
        table_loc="unicode_type",
        iceberg_schema_id="i8",
        partition_spec="python_list_of_heterogeneous_tuples_type",
        sort_order="python_list_of_heterogeneous_tuples_type",
        iceberg_schema_str="unicode_type",
        expected_schema="pyarrow_schema_type",
        df_pyarrow_schema="pyarrow_schema_type",
    ):
        (
            already_exists,
            table_loc,
            iceberg_schema_id,
            partition_spec,
            sort_order,
            iceberg_schema_str,
            expected_schema,
            # This has the Iceberg field IDs in the metadata of every field
            # which is required for correctness.
            df_pyarrow_schema,
        ) = get_table_details_before_write(
            table_name,
            conn,
            database_schema,
            df_pyarrow_schema,
            if_exists,
            allow_downcasting,
        )

    if already_exists and if_exists == "fail":
        # Ideally we'd like to throw the same error as pandas
        # (https://github.com/pandas-dev/pandas/blob/4bfe3d07b4858144c219b9346329027024102ab6/pandas/io/sql.py#L833)
        # but using values not known at compile time, in Exceptions
        # doesn't seem to work with Numba
        raise ValueError(f"Table already exists.")

    if already_exists:
        mode = if_exists
    else:
        mode = "create"

    iceberg_files_info = iceberg_pq_write(
        table_loc,
        bodo_table,
        col_names,
        partition_spec,
        sort_order,
        iceberg_schema_str,
        is_parallel,
        expected_schema,
    )

    with numba.objmode(success="bool_"):
        fnames, record_counts, file_sizes = collect_file_info(iceberg_files_info)

        # Send file names, metrics and schema to Iceberg connector
        success = register_table_write(
            conn,
            database_schema,
            table_name,
            table_loc,
            fnames,
            {"size": file_sizes, "record_count": record_counts},
            iceberg_schema_id,
            df_pyarrow_schema,
            partition_spec,
            sort_order,
            mode,
        )

    if not success:
        # TODO [BE-3249] If it fails due to schema changing, then delete the files.
        # Note that this might not always be possible since
        # we might not have DeleteObject permissions, for instance.
        raise BodoError("Iceberg write failed.")

    ev.finalize()


@numba.generated_jit(nopython=True)
def iceberg_merge_cow_py(
    table_name,
    conn,
    database_schema,
    bodo_df,
    snapshot_id,
    old_fnames,
    is_parallel=False,
):
    if not is_parallel:
        raise BodoError(
            "Merge Into with Iceberg Tables are only supported on distributed DataFrames"
        )

    df_pyarrow_schema = bodo.io.helpers.numba_to_pyarrow_schema(
        bodo_df, is_iceberg=True
    )
    col_names_py = pd.array(bodo_df.columns)

    if bodo_df.is_table_format:
        bodo_table_type = bodo_df.table_type

        def impl(
            table_name,
            conn,
            database_schema,
            bodo_df,
            snapshot_id,
            old_fnames,
            is_parallel=False,
        ):  # pragma: no cover
            iceberg_merge_cow(
                table_name,
                format_iceberg_conn_njit(conn),
                database_schema,
                py_table_to_cpp_table(
                    bodo.hiframes.pd_dataframe_ext.get_dataframe_table(bodo_df),
                    bodo_table_type,
                ),
                snapshot_id,
                old_fnames,
                array_to_info(col_names_py),
                df_pyarrow_schema,
                is_parallel,
            )

    else:
        data_args = ", ".join(
            "array_to_info(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(bodo_df, {}))".format(
                i
            )
            for i in range(len(bodo_df.columns))
        )

        func_text = "def impl(\n"
        func_text += "    table_name,\n"
        func_text += "    conn,\n"
        func_text += "    database_schema,\n"
        func_text += "    bodo_df,\n"
        func_text += "    snapshot_id,\n"
        func_text += "    old_fnames,\n"
        func_text += "    is_parallel=False,\n"
        func_text += "):\n"
        func_text += "    info_list = [{}]\n".format(data_args)
        func_text += "    table = arr_info_list_to_table(info_list)\n"
        func_text += "    iceberg_merge_cow(\n"
        func_text += "        table_name,\n"
        func_text += "        format_iceberg_conn_njit(conn),\n"
        func_text += "        database_schema,\n"
        func_text += "        table,\n"
        func_text += "        snapshot_id,\n"
        func_text += "        old_fnames,\n"
        func_text += "        array_to_info(col_names_py),\n"
        func_text += "        df_pyarrow_schema,\n"
        func_text += "        is_parallel,\n"
        func_text += "    )\n"

        locals = dict()
        globals = {
            "bodo": bodo,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "iceberg_merge_cow": iceberg_merge_cow,
            "format_iceberg_conn_njit": format_iceberg_conn_njit,
            "col_names_py": col_names_py,
            "df_pyarrow_schema": df_pyarrow_schema,
        }
        exec(func_text, globals, locals)
        impl = locals["impl"]

    return impl


@numba.njit()
def iceberg_merge_cow(
    table_name,
    conn,
    database_schema,
    bodo_table,
    snapshot_id,
    old_fnames,
    col_names,
    df_pyarrow_schema,
    is_parallel,
):  # pragma: no cover
    ev = tracing.Event("iceberg_merge_cow_py", is_parallel)
    # Supporting REPL requires some refactor in the parquet write infrastructure,
    # so we're not implementing it for now. It will be added in a following PR.
    assert is_parallel, "Iceberg Write only supported for distributed dataframes"

    with numba.objmode(
        already_exists="bool_",
        table_loc="unicode_type",
        partition_spec="python_list_of_heterogeneous_tuples_type",
        sort_order="python_list_of_heterogeneous_tuples_type",
        iceberg_schema_str="unicode_type",
        expected_schema="pyarrow_schema_type",
    ):
        (
            already_exists,
            table_loc,
            _,
            partition_spec,
            sort_order,
            iceberg_schema_str,
            expected_schema,
            _,
        ) = get_table_details_before_write(
            table_name,
            conn,
            database_schema,
            df_pyarrow_schema,
            "append",
            allow_downcasting=True,
        )

    if not already_exists:
        raise ValueError(f"Iceberg MERGE INTO: Table does not exist at write")

    iceberg_files_info = iceberg_pq_write(
        table_loc,
        bodo_table,
        col_names,
        partition_spec,
        sort_order,
        iceberg_schema_str,
        is_parallel,
        expected_schema,
    )

    with numba.objmode(success="bool_"):
        fnames, record_counts, file_sizes = collect_file_info(iceberg_files_info)

        # Send file names, metrics and schema to Iceberg connector
        success = register_table_merge_cow(
            conn,
            database_schema,
            table_name,
            table_loc,
            old_fnames,
            fnames,
            {"size": file_sizes, "record_count": record_counts},
            snapshot_id,
        )

    if not success:
        # TODO [BE-3249] If it fails due to snapshot changing, then delete the files.
        # Note that this might not always be possible since
        # we might not have DeleteObject permissions, for instance.
        raise BodoError("Iceberg MERGE INTO: write failed")

    ev.finalize()


import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types

from bodo.io import arrow_cpp

ll.add_symbol("iceberg_pq_write_py_entry", arrow_cpp.iceberg_pq_write_py_entry)


@intrinsic(prefer_literal=True)
def iceberg_pq_write_table_cpp(
    typingctx,
    table_data_loc_t,
    table_t,
    col_names_t,
    partition_spec_t,
    sort_order_t,
    compression_t,
    is_parallel_t,
    bucket_region,
    row_group_size,
    iceberg_metadata_t,
    iceberg_schema_t,
):
    """
    Call C++ iceberg parquet write function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            # Iceberg Files Info (list of tuples)
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                # Partition Spec
                lir.IntType(8).as_pointer(),
                # Sort Order
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="iceberg_pq_write_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        types.python_list_of_heterogeneous_tuples_type(  # type: ignore
            types.voidptr,
            table_t,
            col_names_t,
            python_list_of_heterogeneous_tuples_type,
            python_list_of_heterogeneous_tuples_type,
            types.voidptr,
            types.boolean,
            types.voidptr,
            types.int64,
            types.voidptr,
            pyarrow_schema_type,
        ),
        codegen,
    )
