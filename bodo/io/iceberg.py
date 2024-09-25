# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
File that contains the main functionality for the Iceberg
integration within the Bodo repo. This does not contain the
main IR transformation.
"""
from __future__ import annotations

import itertools
import os
import random
import re
import sys
import time
import typing as pt
import warnings
from copy import deepcopy
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlparse

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests
from numba.core import types
from numba.extending import box, intrinsic, models, overload, register_model, unbox

import bodo
import bodo.utils.tracing as tracing
from bodo.ext import s3_reader
from bodo.io.fs_io import (
    ArrowFs,
    arrow_filesystem_del,
    validate_gcsfs_installed,
)
from bodo.io.helpers import (
    _get_numba_typ_from_pa_typ,
    is_pyarrow_list_type,
    pyarrow_schema_type,
    sync_and_reraise_error,
)
from bodo.io.parquet_pio import (
    REMOTE_FILESYSTEMS,
    filter_row_groups_from_start_of_dataset,
    filter_row_groups_from_start_of_dataset_heuristic,
    get_fpath_without_protocol_prefix,
    getfs,
    parse_fpath,
    schema_with_dict_cols,
)
from bodo.io.s3_fs import (
    create_iceberg_aws_credentials_provider,
    create_s3_fs_instance,
    get_region_from_creds_provider,
)
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    py_table_to_cpp_table,
)
from bodo.libs.bool_arr_ext import alloc_false_bool_array
from bodo.libs.str_ext import unicode_to_utf8
from bodo.mpi4py import MPI
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import BodoError, BodoWarning
from bodo.utils.utils import run_rank0

if pt.TYPE_CHECKING:  # pragma: no cover
    from bodo_iceberg_connector import IcebergParquetInfo
    from pyarrow._dataset import Dataset
    from pyarrow._fs import PyFileSystem

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


# ------------------------------ Types ---------------------------------- #
class IcebergConnectionType(types.Type):
    """
    Abstract base class for IcebergConnections
    """

    def __init__(self, name):  # pragma: no cover
        super(IcebergConnectionType, self).__init__(
            name=name,
        )

    def get_conn_str(self) -> str:
        raise NotImplementedError("IcebergConnectionType should not be instanstiated")


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
        "iceberg+abfs",
        "iceberg+abfss",
        "iceberg+rest",
    ):
        raise BodoError(
            "'con' must start with one of the following: 'iceberg://', 'iceberg+file://', "
            "'iceberg+s3://', 'iceberg+thrift://', 'iceberg+http://', 'iceberg+https://', 'iceberg+glue', 'iceberg+snowflake://', "
            "'iceberg+abfs://', 'iceberg+abfss://', 'iceberg+rest://'"
        )

    # Remove Iceberg Prefix when using Internally
    conn_str = conn_str.removeprefix("iceberg+").removeprefix("iceberg://")

    # Reformat Snowflake connection string to be iceberg-connector compatible
    if conn_str.startswith("snowflake://"):
        from bodo.io.snowflake import parse_conn_str

        conn_contents = parse_conn_str(conn_str)
        account: str = conn_contents.pop("account")
        # Flatten Session Parameters
        session_params = conn_contents.pop("session_parameters", {})
        conn_contents.update(session_params)
        # Remove Snowflake Specific Parameters
        conn_contents.pop("warehouse", None)
        conn_contents.pop("database", None)
        conn_contents.pop("schema", None)
        conn_str = (
            f"snowflake://{account}.snowflakecomputing.com/?{urlencode(conn_contents)}"
        )

    return conn_str


def format_iceberg_conn_njit(conn: str) -> str:  # type: ignore
    pass


@overload(format_iceberg_conn_njit)
def overload_format_iceberg_conn_njit(conn):  # pragma: no cover
    """
    Wrapper around format_iceberg_conn for strings
    Gets the connection string from conn_str attr for IcebergConnectionType

    Args:
        conn_str (str | IcebergConnectionType): connection passed in read_sql/read_sql_table/to_sql

    Returns:
        str: connection string without the iceberg(+*?) prefix
    """
    if isinstance(conn, (types.UnicodeType, types.StringLiteral)):

        def impl(conn):
            with bodo.no_warning_objmode(conn_str="unicode_type"):
                conn_str = format_iceberg_conn(conn)
            return conn_str

        return impl
    else:
        assert isinstance(
            conn, IcebergConnectionType
        ), f"format_iceberg_conn_njit: Invalid type for conn, got {conn}"

        def impl(conn):
            return conn.conn_str

        return impl


# ----------------------------- Iceberg Read -----------------------------#
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


def get_iceberg_file_list(
    table_name: str, conn: str, database_schema: str, filters: str | None
) -> tuple[list["IcebergParquetInfo"], dict[int, pa.Schema], int]:
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


def generate_expr_filter(
    expr_filter_f_str: str,
    filter_scalars: list[tuple[str, pt.Any]],
    col_rename_map: dict[str, str],
) -> pc.Expression:
    """
    Helper function to dynamically generate the Arrow expressions
    for filtering at runtime.
    The 'expr_filter_f_str' is generated by
    'bodo.ir.connector.generate_arrow_filters' by setting the
    'output_expr_filters_as_f_string' parameter.
    'filter_scalars' is generated using 'get_filter_scalars_pyobject'.
    See '_gen_sql_reader_py' and '_gen_iceberg_reader_chunked_py'
    in 'iceberg_ext.py' for more details on how these
    are generated.
    Note that we don't cache this computation, so it's best
    to call it once per IcebergSchemaGroup.


    Args:
        expr_filter_f_str (str): An f-string version of the
            final expression. We will populate the templated
            variables in this f-string using the col_rename_map.
        filter_scalars (list[tuple[str, Any]]): List of tuples
            of the form ('fXX', Any). The first element is the
            name of the variable in the expr_filter_f_str
            that will assume this value and the second element
            is the actual value itself. This value can be
            any Python object (e.g. string, int, list[Any], etc.)
        col_rename_map (dict[str, str]): Column rename map
            used to populate the templated variables in
            expr_filter_f_str. This is the mapping of the
            column names from the 'final_schema' to the
            'read_schema' of an IcebergSchemaGroup.

    Returns:
        pc.Expression: Generated expression object that can
            be used to filter the table during read.
    """

    # Fill in the templated column names.
    expr_filter_str = expr_filter_f_str.format(**col_rename_map)
    # Produce the parameters for the function.
    # e.g. 'f0, f1, f2'
    input_vars_str = ",".join([x[0] for x in filter_scalars])
    glbs = globals()
    glbs["ds"] = ds
    loc_vars = {}
    # By passing in the scalars as arguments, they will
    # get mapped correctly in the expr_filter_str.
    func_text = f"def impl({input_vars_str}):\n  return {expr_filter_str}"
    input_vars = [x[1] for x in filter_scalars]
    exec(func_text, glbs, loc_vars)
    expr_filter = loc_vars["impl"](*input_vars)
    return expr_filter


def sanitize_col_name(col_name: str) -> str:  # pragma: no cover
    """
    Sanitize a column name to remove
    any spaces, quotes, etc.
    Ref: https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python.
    Essentially turns a string to a valid
    Python variable name, which is sufficient for our purposes.
    Note that these are not guaranteed to be 1 to 1, i.e.
    two inputs could produce the same output.

    Args:
        col_name (str): String to sanitize.

    Returns:
        str: Sanitized string
    """
    return re.sub(r"\W|^(?=\d)", "_", col_name)


class ThetaSketchCollectionType(types.Type):
    """Type for C++ pointer to a collection of theta sketches"""

    def __init__(self):  # pragma: no cover
        super(ThetaSketchCollectionType, self).__init__(
            name=f"ThetaSketchCollectionType(r)"
        )


register_model(ThetaSketchCollectionType)(models.OpaqueModel)

theta_sketch_collection_type = ThetaSketchCollectionType()


FieldID: pt.TypeAlias = int | tuple["FieldIDs", ...]
FieldIDs: pt.TypeAlias = tuple[FieldID, ...]
FieldName: pt.TypeAlias = str | tuple["FieldNames", ...]
FieldNames: pt.TypeAlias = tuple[FieldName, ...]
SchemaGroupIdentifier: pt.TypeAlias = tuple[FieldIDs, FieldNames]


class IcebergSchemaGroup:
    """
    Class to store the details about one "Schema Group"
    during Iceberg read. A schema group is a group of files
    where their schemas are "similar" in terms of the
    Iceberg fields they contain and the names of those
    fields. Therefore, a schema group is identified
    by two ordered tuples:
    1. The Iceberg Field IDs
    2. The corresponding fields' names.
    The idea is that we can read these files as one
    Arrow dataset. This is useful since Arrow can do
    async read-ahead on the files in a dataset, which
    improves performance.
    The shared Arrow expression filter applied to these
    files is generated based on the mapping of column names
    in the "final_schema" (the schema of the table we want
    to read the dataset *as*) to the names in the "read_schema"
    (i.e. the intermediate schema that we will give to Arrow
    during read so that it can perform the filters correctly
    and fill in nulls for columns where they don't exist).
    """

    def __init__(
        self,
        iceberg_field_ids: FieldIDs,
        parquet_field_names: FieldNames,
        final_schema: pa.Schema,
        expr_filter_f_str: pt.Optional[str] = None,
        filter_scalars: pt.Optional[list[tuple[str, pt.Any]]] = None,
    ):
        """
        Construct a new Schema Group.

        Args:
            iceberg_field_ids (tuple[int]): Ordered tuple of Iceberg field
                IDs of the top-level columns (i.e. field IDs of the nested
                fields is not included in these).
            parquet_field_names (tuple[str]): Ordered tuple of the field
                names for the fields corresponding to those in
                'iceberg_field_ids'.
            final_schema (pa.Schema): The 'final_schema' that will be used
                for generating the read-schema.
            expr_filter_f_str (Optional[str], optional): An f-string
                representation of the Arrow expression filter to use to generate
                the filter expression. See description of 'generate_expr_filter'
                for more details. Defaults to None.
            filter_scalars (Optional[list[tuple[str, Any]]], optional): List of
                tuples with the variable names and values of the scalars
                present in the expr_filter_f_str. See description of
                'generate_expr_filter'for more details. Defaults to None.
        """
        assert len(iceberg_field_ids) == len(parquet_field_names)
        self.iceberg_field_ids = iceberg_field_ids
        self.parquet_field_names = parquet_field_names
        self.final_schema: pa.Schema = final_schema
        self.read_schema: pa.Schema = self.gen_read_schema(
            self.iceberg_field_ids, self.parquet_field_names, self.final_schema
        )
        self.expr_filter: pt.Optional[pc.Expression] = None
        if (expr_filter_f_str is not None) and (len(expr_filter_f_str) > 0):
            filter_scalars = [] if filter_scalars is None else filter_scalars
            col_rename_map: dict[str, str] = {
                self.final_schema.field(i).name: self.read_schema.field(i).name
                for i in range(len(self.final_schema.names))
            }
            self.expr_filter = generate_expr_filter(
                expr_filter_f_str, filter_scalars, col_rename_map
            )

    @property
    def group_identifier(self) -> SchemaGroupIdentifier:
        """
        The tuple that uniquely identifies a Schema Group.

        Returns:
            SchemaGroupIdentifier
        """
        return (self.iceberg_field_ids, self.parquet_field_names)

    @staticmethod
    def gen_read_field(
        iceberg_field_ids: FieldID,
        parquet_field_names: FieldName,
        final_field: pa.Field,
        field_name_for_err_msg: str,
    ) -> pa.Field:
        """
        Recursive helper for gen_read_schema to generate
        the Iceberg Schema Group's read field.

        Args:
            iceberg_field_ids (int | tuple): Iceberg Field ID
                of this field in the parquet file. In the
                semi-structured type case, this will be a tuple
                where the first element is the Field ID of the
                semi-structured type itself and the rest will
                be Field IDs of it sub-fields (each of these may
                also be tuples since they might be semi-structured
                themselves).
            parquet_field_names (str | tuple): Corresponding
                fields' names.
            final_field (pa.Field): The target field.
            field_name_for_err_msg (str): Since this function is
                called recursively, we use this to build up
                a more meaningful name for any error messages that
                we raise.

        Returns:
            pa.Field: Field to use when reading the files in this
                schema group.
        """

        assert (
            final_field.metadata is not None
        ), f"Field {field_name_for_err_msg} does not have metadata! This is most likely a bug in Bodo."
        assert b_ICEBERG_FIELD_ID_MD_KEY in final_field.metadata, (
            f"Field {field_name_for_err_msg} does not have the Iceberg Field ID in its metadata. "
            f"Metadata:\n{final_field.metadata}\nThis is most likely a bug in Bodo."
        )
        iceberg_field_id = int(final_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])

        if isinstance(iceberg_field_ids, int):
            assert isinstance(parquet_field_names, str)
            if iceberg_field_id != iceberg_field_ids:
                raise RuntimeError(
                    f"Field {field_name_for_err_msg} does not have the expected Iceberg Field ID! "
                    f"Expected {iceberg_field_id} but got {iceberg_field_ids} instead."
                )
            if (
                pa.types.is_map(final_field.type)
                or pa.types.is_list(final_field.type)
                or pa.types.is_large_list(final_field.type)
                or pa.types.is_fixed_size_list(final_field.type)
                or pa.types.is_struct(final_field.type)
            ):
                raise RuntimeError(
                    f"Expected field type for Iceberg Field ID {iceberg_field_id} ({field_name_for_err_msg}) "
                    f"to be a nested type ({final_field.type}), but it was a primitive type instead!"
                )
            # Note that "final_field" is assumed to be "compatible",
            # i.e. the same or less strict (in terms of type and
            # nullability) than whatever is in the files. The actual
            # validation is performed at read-time at a
            # per-file basis.
            # During the read, we will use the "old" name for this
            # file.
            return final_field.with_name(parquet_field_names)
        else:
            assert isinstance(iceberg_field_ids, tuple)
            assert isinstance(parquet_field_names, tuple)
            assert len(iceberg_field_ids) == len(parquet_field_names)
            assert isinstance(iceberg_field_ids[0], int)
            assert isinstance(parquet_field_names[0], str)
            if iceberg_field_id != iceberg_field_ids[0]:
                raise RuntimeError(
                    f"Field {field_name_for_err_msg} does not have the expected Iceberg Field ID! "
                    f"Expected {iceberg_field_id} but got {iceberg_field_ids[0]} instead."
                )
            if not (
                pa.types.is_map(final_field.type)
                or pa.types.is_list(final_field.type)
                or pa.types.is_large_list(final_field.type)
                or pa.types.is_fixed_size_list(final_field.type)
                or pa.types.is_struct(final_field.type)
            ):
                raise RuntimeError(
                    f"Expected field type for Iceberg Field ID {iceberg_field_id} ({field_name_for_err_msg}) "
                    f"to be a primitive type ({final_field.type}), but it was a nested type instead!"
                )

            if pa.types.is_struct(final_field.type):
                # Struct is the tricky case where we must handle
                # evolution ourselves at read time. Unlike
                # top-level fields, Arrow doesn't like it if
                # we add extra sub-fields to the struct (to fill
                # them with nulls). It also doesn't like it if we
                # don't provide the fields in the same order as they
                # will appear in the parquet files. However, it
                # does allow "skipping" sub-fields as long as the order
                # of the rest of the fields is consistent with that in the
                # parquet file. It can also still
                # perform nullability/type promotion (including multiple
                # levels down in its sub-fields). Therefore, we will
                # only include the sub-fields that exist in the parquet
                # file and will maintain the order.
                # If a sub-field from the final_field doesn't exist in the parquet file,
                # we won't add it to the read_schema at this point. We
                # will add it later (see 'EvolveRecordBatch' in 'iceberg_parquet_reader.cpp').
                # We will keep the fields in the same original order and with
                # the same names. We will do the re-ordering and renaming later
                # as part of 'EvolveRecordBatch'.
                # We will however skip the fields that we no longer need.
                # We will also perform the type/nullability promotion
                # as per the final_field.

                final_sub_fields_iceberg_field_id_to_idx: dict[int, int] = {
                    int(
                        final_field.type.field(i).metadata[b_ICEBERG_FIELD_ID_MD_KEY]
                    ): i
                    for i in range(final_field.type.num_fields)
                }

                read_fields: list[pa.Field] = []
                iceberg_field_ids_in_schema_group_sub_fields: set[int] = set([])
                # Sub-fields start at index 1.
                for i in range(1, len(iceberg_field_ids)):
                    sub_field_iceberg_field_id: int = (
                        iceberg_field_ids[i]
                        if isinstance(iceberg_field_ids[i], int)
                        else iceberg_field_ids[i][0]
                    )
                    iceberg_field_ids_in_schema_group_sub_fields.add(
                        sub_field_iceberg_field_id
                    )
                    if (
                        sub_field_iceberg_field_id
                        in final_sub_fields_iceberg_field_id_to_idx
                    ):
                        final_sub_field_: pa.Field = final_field.type.field(
                            final_sub_fields_iceberg_field_id_to_idx[
                                sub_field_iceberg_field_id
                            ]
                        )
                        read_schema_sub_field = IcebergSchemaGroup.gen_read_field(
                            iceberg_field_ids[i],
                            parquet_field_names[i],
                            final_sub_field_,
                            field_name_for_err_msg=f"{field_name_for_err_msg}.{final_sub_field_.name}",
                        )
                        read_fields.append(read_schema_sub_field)

                # Verify that all the required sub fields in the final field exist
                # in the schema group field.
                for i in range(final_field.type.num_fields):
                    final_sub_field: pa.Field = final_field.type.field(i)
                    assert final_sub_field.metadata is not None
                    assert b_ICEBERG_FIELD_ID_MD_KEY in final_sub_field.metadata
                    final_sub_field_iceberg_field_id = int(
                        final_sub_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
                    )
                    if (not final_sub_field.nullable) and (
                        final_sub_field_iceberg_field_id
                        not in iceberg_field_ids_in_schema_group_sub_fields
                    ):
                        raise RuntimeError(
                            f"Non-nullable field '{field_name_for_err_msg}.{final_sub_field.name}' "
                            f"(Iceberg Field ID: {final_sub_field_iceberg_field_id}) not found in "
                            "the schema group!"
                        )

                return final_field.with_type(pa.struct(read_fields)).with_name(
                    parquet_field_names[0]
                )
            elif pa.types.is_large_list(final_field.type):
                assert len(iceberg_field_ids) == len(parquet_field_names) == 2
                read_value_field = IcebergSchemaGroup.gen_read_field(
                    iceberg_field_ids[1],
                    parquet_field_names[1],
                    final_field.type.value_field,
                    field_name_for_err_msg=f"{field_name_for_err_msg}.element",
                )
                return final_field.with_type(pa.large_list(read_value_field)).with_name(
                    parquet_field_names[0]
                )
            else:
                assert pa.types.is_map(final_field.type)
                assert len(iceberg_field_ids) == len(parquet_field_names) == 3
                read_key_field = IcebergSchemaGroup.gen_read_field(
                    iceberg_field_ids[1],
                    parquet_field_names[1],
                    final_field.type.key_field,
                    field_name_for_err_msg=f"{field_name_for_err_msg}.key",
                )
                read_item_field = IcebergSchemaGroup.gen_read_field(
                    iceberg_field_ids[2],
                    parquet_field_names[2],
                    final_field.type.item_field,
                    field_name_for_err_msg=f"{field_name_for_err_msg}.value",
                )
                return final_field.with_type(
                    pa.map_(read_key_field, read_item_field)
                ).with_name(parquet_field_names[0])

    @staticmethod
    def gen_read_schema(
        iceberg_field_ids: FieldIDs,
        parquet_field_names: FieldNames,
        final_schema: pa.Schema,
    ) -> pa.Schema:
        """
        Generate the "read_schema", i.e. the schema given
        to Arrow Dataset Scanners when reading the files
        belonging to this schema group, from the final/target
        schema.
        The "read_schema" will have the same number of
        fields as the "final_schema" and the fields
        corresponding to the "final_schema" will be
        in the same order as the "final_schema".
        The "final_schema" must have Iceberg Field IDs
        in the metadata of the fields.
        Nested fields are handled by calling 'gen_read_field'
        on them recursively.

        Args:
            iceberg_field_ids (tuple[int | tuple]): Iceberg field IDs
                of the fields in the schema of the files in
                the schema-group.
            parquet_field_names (tuple[str | tuple]): The corresponding
                field names.
            final_schema (pa.Schema): The target schema.

        Returns:
            pa.Schema: 'read_schema' for the schema group.
        """

        # Create a map from Iceberg Field Id to the column index for the
        # top-level fields.
        schema_group_field_id_to_schema_group_col_idx: dict[int, int] = {}
        for i in range(len(iceberg_field_ids)):
            if isinstance(iceberg_field_ids[i], int):
                assert isinstance(parquet_field_names[i], str)
                schema_group_field_id_to_schema_group_col_idx[iceberg_field_ids[i]] = i
            else:
                assert isinstance(iceberg_field_ids[i], tuple)
                assert isinstance(parquet_field_names[i], tuple)
                assert isinstance(iceberg_field_ids[i][0], int)
                assert isinstance(parquet_field_names[i][0], str)
                schema_group_field_id_to_schema_group_col_idx[
                    iceberg_field_ids[i][0]
                ] = i

        read_schema_fields: list[pa.Field] = []
        for i, field in enumerate(final_schema):
            assert (
                field.metadata is not None
            ), f"Target schema field doesn't have metadata! This is most likely a bug in Bodo. Field:\n{field}."
            assert b_ICEBERG_FIELD_ID_MD_KEY in field.metadata, (
                f"Target schema field metadata doesn't have the required Iceberg Field ID. "
                f"This is most likely a bug in Bodo.\nField: {field}\nField metadata: {field.metadata}."
            )
            iceberg_field_id = int(field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])
            if iceberg_field_id in schema_group_field_id_to_schema_group_col_idx:
                # If this field exists in the file:
                schema_group_field_idx = schema_group_field_id_to_schema_group_col_idx[
                    iceberg_field_id
                ]
                read_schema_field = IcebergSchemaGroup.gen_read_field(
                    iceberg_field_ids[schema_group_field_idx],
                    parquet_field_names[schema_group_field_idx],
                    field,
                    field_name_for_err_msg=field.name,
                )
                read_schema_fields.append(read_schema_field)
            else:
                # Field is not in the file, i.e. it was added to the table
                # after these files were written or the column in optional
                # and the original writer chose not to write them.
                # To avoid name conflicts, we will use a unique name.
                # The column will automatically be filled with nulls at read
                # time.
                if not field.nullable:
                    raise RuntimeError(
                        f"Non-nullable field '{field.name}' (Field ID: "
                        f"{iceberg_field_id}) not found in the schema group!"
                    )
                sanitized_field_name = sanitize_col_name(field.name)
                _uniq_name = f"_BODO_TEMP_{iceberg_field_id}_{sanitized_field_name}"
                assert _uniq_name not in parquet_field_names, (
                    f"Generated unique name for Iceberg field already exists in the file! "
                    f"This is most likely a bug in Bodo.\n{_uniq_name=}\n{parquet_field_names=}"
                )
                read_schema_fields.append(field.with_name(_uniq_name))

        return pa.schema(read_schema_fields)


@dataclass
class IcebergPiece:
    """
    A simple dataclass representing a parquet
    file to read during an Iceberg table read.
    These are used in 'IcebergParquetDataset'
    to store information about the files to read.
    """

    # Path to the file. This may or may not
    # include the protocol prefix.
    path: str
    # Index of the schema group that this
    # file belongs to. This corresponds to
    # the schema groups in
    # IcebergParquetDataset.schema_groups.
    # -1 if we're setting the schema_group_identifier instead.
    schema_group_idx: int
    # Schema group identifier. This is used when
    # the schema groups haven't been created yet.
    schema_group_identifier: SchemaGroupIdentifier | None
    # Number of rows to read from this file
    # In case of row-level filtering, this is the count
    # after applying the filters. In case of piece-level filtering,
    # this is the total number of rows in the piece.
    _bodo_num_rows: int


@dataclass
class IcebergPqDatasetMetrics:
    """
    Metrics for the get_iceberg_pq_dataset step.
    All timers are in microseconds.
    """

    file_list_time: int = 0
    file_to_schema_time_us: int = 0
    get_fs_time: int = 0
    n_files_analyzed: int = 0
    file_frags_creation_time: int = 0
    get_sg_id_time: int = 0
    sort_by_sg_id_time: int = 0
    nunique_sgs_seen: int = 0
    exact_row_counts_time: int = 0
    get_row_counts_nrgs: int = 0
    get_row_counts_nrows: int = 0
    get_row_counts_total_bytes: int = 0
    pieces_allgather_time: int = 0
    sort_all_pieces_time: int = 0
    assemble_ds_time: int = 0


@dataclass
class IcebergParquetDataset:
    """
    Store dataset info in the way expected by Arrow reader in C++.
    """

    # Whether this is a row-level or piece-level read.
    # In case of a row-level read, we get the exact row counts
    # for each piece after applying filters. In the piece-level read
    # case, we only prune out pieces based on metadata and report
    # the row count of the entire piece.
    row_level: bool
    # 'conn', 'database schema' & 'table_name' describe the Iceberg
    # table we're reading.
    conn: str
    database_schema: str
    table_name: str
    # This is the PyArrow schema object for the final/target schema to read
    # the table as. This is obtained from Iceberg at compile time, i.e. the
    # expected final schema. It must have Iceberg Field IDs in the metadata
    # of its fields.
    pa_table_schema: pa.Schema
    # List of files exactly as given by Iceberg. This is used for operations like delete/merge.
    # There files are likely the relative paths to the Iceberg table for local files.
    # For example if the absolute path was /Users/bodo/iceberg_db/my_table/part01.pq
    # and the iceberg directory is iceberg_db, then the path in the list would be
    # iceberg_db/my_table/part01.pq.
    file_list: list[str]
    # Snapshot id. This is used for operations like delete/merge.
    snapshot_id: int
    # Filesystem can be None when there are no files to read.
    filesystem: "PyFileSystem" | pa.fs.FileSystem | None
    # Parquet files to read ordered by the schema group
    # they belong to. We order them this way so that when
    # we split this list between ranks for the actual read,
    # each rank will (ideally) only need to handle a subset
    # of the schema groups (and hence minimize the number of
    # Arrow scanners/record-batch-readers that it needs
    # to create).
    pieces: list[IcebergPiece]
    # Ordered list of schema groups. These are all the
    # different schemas we will need to handle during
    # the actual read for handling schema evolution.
    schema_groups: list[IcebergSchemaGroup]
    # Total number of rows that we will read (globally).
    _bodo_total_rows: int
    # Metrics
    metrics: IcebergPqDatasetMetrics


def validate_file_schema_field_compatible_with_read_schema_field(
    file_schema_field: pa.Field,
    read_schema_field: pa.Field,
    field_name_for_err_msg: str,
):
    """
    Helper function for 'validate_file_schema_compatible_with_read_schema'
    to validate specific fields recursively.

    Args:
        file_schema_field (pa.Field): Field in the file.
        read_schema_field (pa.Field): "Expected" field from the
            schema group's read_schema.
        field_name_for_err_msg (str): Since this function is
            called recursively, we pass in a string to display
            a more readable name for the nested fields.
            e.g. Instead of saying that the field 'a' is
            incompatible, this allows us to say that the field
            'A.a' is incompatible.

    Raises:
        RuntimeError: If the file_schema_field is incompatible with
            the read_schema_field or if an unsupported field type is
            found in the file_schema_field
    """
    # Check that the field id is what we expect it to be.
    if (file_schema_field.metadata is None) or (
        b_ICEBERG_FIELD_ID_MD_KEY not in file_schema_field.metadata
    ):
        raise RuntimeError(
            f"Field '{field_name_for_err_msg}' doesn't have an Iceberg field ID specified in the file!"
        )

    read_schema_field_iceberg_id = int(
        read_schema_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
    )
    file_schema_field_iceberg_id = int(
        file_schema_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
    )
    if read_schema_field_iceberg_id != file_schema_field_iceberg_id:
        raise RuntimeError(
            f"Iceberg Field ID mismatch in file for '{field_name_for_err_msg}' field! "
            f"Expected: {read_schema_field_iceberg_id}, got {file_schema_field_iceberg_id} instead."
        )

    field_repr: str = f"field '{field_name_for_err_msg}' (Iceberg Field ID: {read_schema_field_iceberg_id})"

    # Then check the following:
    # - It shouldn't be nullable if the read schema field isn't nullable.
    if (not read_schema_field.nullable) and file_schema_field.nullable:
        raise RuntimeError(f"Required {field_repr} is optional in the file!")

    # - Check that that the types are in the same 'class', i.e. can be upcast safely.
    read_schema_field_type: pa.DataType = read_schema_field.type
    if pa.types.is_signed_integer(read_schema_field_type):
        if not pa.types.is_signed_integer(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a signed integer, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_unsigned_integer(read_schema_field_type):
        if not pa.types.is_unsigned_integer(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be an unsigned integer, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_floating(read_schema_field_type):
        if not pa.types.is_floating(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a floating point number, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_decimal(read_schema_field_type):
        if not pa.types.is_decimal(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a decimal, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
        if read_schema_field_type.scale != file_schema_field.type.scale:
            raise RuntimeError(
                f"Scale of decimal {field_repr} doesn't match exactly. Expected {read_schema_field_type.scale}, "
                f"got {file_schema_field.type.scale} instead."
            )
        if read_schema_field_type.precision < file_schema_field.type.precision:
            raise RuntimeError(
                f"Precision of decimal {field_repr} in file is larger ({file_schema_field.type.precision}) "
                f"than what's allowed ({read_schema_field_type.precision})!"
            )
    elif pa.types.is_boolean(read_schema_field_type):
        if not pa.types.is_boolean(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a boolean, got {file_schema_field.type} instead!"
            )
    elif pa.types.is_string(read_schema_field_type) or pa.types.is_large_string(
        read_schema_field_type
    ):
        if not (
            pa.types.is_string(file_schema_field.type)
            or pa.types.is_large_string(file_schema_field.type)
        ):
            raise RuntimeError(
                f"Expected {field_repr} to be a string, got {file_schema_field.type} instead!"
            )
    elif (
        pa.types.is_binary(read_schema_field_type)
        or pa.types.is_large_binary(read_schema_field_type)
        or pa.types.is_fixed_size_binary(read_schema_field_type)
    ):
        if not (
            pa.types.is_binary(file_schema_field.type)
            or pa.types.is_large_binary(file_schema_field.type)
            or pa.types.is_fixed_size_binary(file_schema_field.type)
        ):
            raise RuntimeError(
                f"Expected {field_repr} to be a binary, got {file_schema_field.type} instead!"
            )
    elif pa.types.is_date(read_schema_field_type):
        if not pa.types.is_date(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a date, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_time(read_schema_field_type):
        if not pa.types.is_time(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a time, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_timestamp(read_schema_field_type):
        if not pa.types.is_timestamp(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a timestamp, got {file_schema_field.type} instead!"
            )
        # Timestamps always have a bit-width of 64.
        # XXX TODO Could add checks based on tz/unit here in the future if needed.
    elif (
        pa.types.is_list(read_schema_field_type)
        or pa.types.is_large_list(read_schema_field_type)
        or pa.types.is_fixed_size_list(read_schema_field_type)
    ):
        if not (
            pa.types.is_list(file_schema_field.type)
            or pa.types.is_large_list(file_schema_field.type)
            or pa.types.is_fixed_size_list(file_schema_field.type)
        ):
            raise RuntimeError(
                f"Expected {field_repr} to be a list, got {file_schema_field.type} instead!"
            )
        # Check the value field recursively.
        validate_file_schema_field_compatible_with_read_schema_field(
            file_schema_field.type.value_field,
            read_schema_field_type.value_field,
            field_name_for_err_msg=f"{field_name_for_err_msg}.value",
        )
    elif pa.types.is_map(read_schema_field_type):
        if not pa.types.is_map(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a map, got {file_schema_field.type} instead!"
            )
        # Check the key and item fields recursively.
        validate_file_schema_field_compatible_with_read_schema_field(
            file_schema_field.type.key_field,
            read_schema_field_type.key_field,
            field_name_for_err_msg=f"{field_name_for_err_msg}.key",
        )
        validate_file_schema_field_compatible_with_read_schema_field(
            file_schema_field.type.item_field,
            read_schema_field_type.item_field,
            field_name_for_err_msg=f"{field_name_for_err_msg}.value",
        )
    elif pa.types.is_struct(read_schema_field_type):
        if not pa.types.is_struct(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a struct, got {file_schema_field.type} instead!"
            )

        # We need all fields in the read schema to exist in the file schema, but not vice-versa.
        # However, all the the fields in the file must be in the same order as they are in the
        # read_schema.
        file_schema_field_type = file_schema_field.type
        file_schema_last_idx: int = -1
        for sub_idx in range(read_schema_field_type.num_fields):
            read_schema_sub_field: pa.Field = read_schema_field_type.field(sub_idx)
            file_schema_sub_field_idx: int = file_schema_field_type.get_field_index(
                read_schema_sub_field.name
            )
            if file_schema_sub_field_idx == -1:
                raise RuntimeError(
                    f"Expected struct {field_repr} to have subfield {read_schema_sub_field.name} "
                    "but it was not found!"
                )
            # file_schema_last_idx should be strictly lower than file_schema_sub_field_idx.
            # If it isn't, then that means the fields are not in the required order.
            # If it always is, then we are guaranteed that the fields are in the
            # required order.
            if file_schema_last_idx >= file_schema_sub_field_idx:
                expected_field_order: list[str] = [
                    read_schema_field_type.field(i).name
                    for i in range(read_schema_field_type.num_fields)
                ]
                actual_field_order: list[str] = [
                    file_schema_field_type.field(i).name
                    for i in range(file_schema_field_type.num_fields)
                ]
                raise RuntimeError(
                    f"Struct {field_repr} does not have subfield {read_schema_sub_field.name} in the right order! "
                    f"Expected ordered subset: {expected_field_order}, but got {actual_field_order} instead."
                )
            file_schema_last_idx = file_schema_sub_field_idx
            file_schema_sub_field: pa.Field = file_schema_field_type.field(
                file_schema_sub_field_idx
            )
            validate_file_schema_field_compatible_with_read_schema_field(
                file_schema_sub_field,
                read_schema_sub_field,
                field_name_for_err_msg=f"{field_name_for_err_msg}.{read_schema_sub_field.name}",
            )
    else:
        raise RuntimeError(
            "bodo.io.iceberg.validate_file_schema_field_compatible_with_read_schema_field: "
            f"Unsupported dtype '{read_schema_field_type}' for {field_repr}."
        )


def validate_file_schema_compatible_with_read_schema(
    file_schema: pa.Schema, read_schema: pa.Schema
):
    """
    Validate that the schema of the Iceberg Parquet file
    is compatible with the read schema of the schema
    group it belongs to.
    At this point, nested fields are expected to match
    exactly, but the top-level fields support all
    types of schema evolution that Iceberg supports.

    Args:
        file_schema (pa.Schema): Schema of the file.
        read_schema (pa.Schema): Schema of the schema group.

        The Iceberg field IDs must be in the metadata
        of the fields in both these schemas.
    """
    for read_schema_field in read_schema:
        # Check if the field exists in the file.
        field_name: str = read_schema_field.name
        if (file_schema_field_idx := file_schema.get_field_index(field_name)) != -1:
            file_schema_field: pa.Field = file_schema.field(file_schema_field_idx)
            validate_file_schema_field_compatible_with_read_schema_field(
                file_schema_field,
                read_schema_field,
                field_name_for_err_msg=field_name,
            )
        else:
            # If a field by that name doesn't exist in the file,
            # then verify that the field is nullable in the read schema.
            iceberg_field_id: int = int(
                read_schema_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
            )
            if not read_schema_field.nullable:
                raise RuntimeError(
                    f"Field '{field_name}' (Iceberg Field ID: {iceberg_field_id}) not "
                    "found in the file even though the field is not nullable/optional!"
                )


def get_iceberg_file_list_parallel(
    conn: str,
    database_schema: str,
    table_name: str,
    filters: str | None = None,
) -> tuple[list["IcebergParquetInfo"], int, dict[int, pa.Schema], int]:
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
    # Sonar cube only runs on rank 0, so we add no cover to avoid the warning
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


def get_schema_group_identifier_from_pa_field(
    field: pa.Field,
    field_name_for_err_msg: str,
) -> SchemaGroupIdentifier:
    """
    Recursive helper for 'get_schema_group_identifier_from_pa_schema'
    to get the schema group identifier for a specific
    field (or sub-field of a nested field). These will
    then be stitched back together to form the
    full schema group identifier.

    Args:
        field (pa.Field): The field to generate the group
            identifier based off of. This could be
            a nested field.
        field_name_for_err_msg (str): Since this function
            is called recursively, we use this field to
            have a more meaningful field name that can be
            used in the error messages.

    Returns:
        SchemaGroupIdentifier: Schema group identifier
            for this field.
    """
    field_type = field.type

    if (field.metadata is None) or (b_ICEBERG_FIELD_ID_MD_KEY not in field.metadata):
        raise RuntimeError(
            f"Field {field_name_for_err_msg} does not have an Iceberg Field ID!"
        )

    iceberg_field_id: int = int(field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])

    if pa.types.is_struct(field_type):
        sub_field_schema_group_identifiers: list[SchemaGroupIdentifier] = [
            get_schema_group_identifier_from_pa_field(
                field_type.field(i),
                f"{field_name_for_err_msg}.{field_type.field(i).name}",
            )
            for i in range(field_type.num_fields)
        ]
        field_ids = [iceberg_field_id] + [
            x[0] for x in sub_field_schema_group_identifiers
        ]
        field_names = [field.name] + [x[1] for x in sub_field_schema_group_identifiers]
        return tuple(field_ids), tuple(field_names)

    elif pa.types.is_map(field_type):
        key_field_schema_group_identifier = get_schema_group_identifier_from_pa_field(
            field_type.key_field, f"{field_name_for_err_msg}.key"
        )
        item_field_schema_group_identifier = get_schema_group_identifier_from_pa_field(
            field_type.item_field, f"{field_name_for_err_msg}.value"
        )
        return (
            iceberg_field_id,
            key_field_schema_group_identifier[0],
            item_field_schema_group_identifier[0],
        ), (
            field.name,
            key_field_schema_group_identifier[1],
            item_field_schema_group_identifier[1],
        )
    elif (
        pa.types.is_list(field_type)
        or pa.types.is_large_list(field_type)
        or pa.types.is_fixed_size_list(field_type)
    ):
        value_field_schema_group_identifier = get_schema_group_identifier_from_pa_field(
            field_type.value_field, f"{field_name_for_err_msg}.element"
        )
        return (iceberg_field_id, value_field_schema_group_identifier[0]), (
            field.name,
            value_field_schema_group_identifier[1],
        )
    else:
        return (iceberg_field_id, field.name)


def get_schema_group_identifier_from_pa_schema(
    schema: pa.Schema,
) -> SchemaGroupIdentifier:
    """
    Generate the schema group identifier from
    the schema of a parquet file. The schema group
    identifier is a tuple of tuples. The first
    is a tuple of Iceberg Field IDs and the second
    is a tuple of the corresponding field names
    in the Parquet file. Nested fields are represented
    by nested tuples within the top-level tuples.

    Args:
        schema (pa.Schema): Schema to generate the
            schema group identifier based on.

    Returns:
        SchemaGroupIdentifier: The schema group identifier.
    """
    field_identifiers = [
        get_schema_group_identifier_from_pa_field(f, f.name) for f in schema
    ]
    iceberg_field_ids = tuple(x[0] for x in field_identifiers)
    pq_field_names = tuple(x[1] for x in field_identifiers)
    return iceberg_field_ids, pq_field_names


def group_file_frags_by_schema_group_identifier(
    pq_infos: list["IcebergParquetInfo"],
    file_schemas: list[pa.Schema],
    metrics: IcebergPqDatasetMetrics,
) -> dict[SchemaGroupIdentifier, list["IcebergParquetInfo"]]:
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
        tuple["IcebergParquetInfo", FieldIDs, FieldNames]
    ] = list(zip(pq_infos, iceberg_field_ids, pq_field_names))
    # Sort/Groupby the field-ids and field-names tuples.
    # We must flatten the tuples for sorting because you
    # cannot compare ints to tuples in Python and nested types
    # will generate tuples. This is safe because a nested field
    # can never become a primitive column (and vice-versa).
    sort_key_func = lambda item: (flatten_tuple(item[1]), flatten_tuple(item[2]))
    keyfunc = lambda item: (item[1], item[2])
    schema_group_id_to_frags: dict[
        SchemaGroupIdentifier, list["IcebergParquetInfo"]
    ] = {
        k: [x[0] for x in v]
        for k, v in itertools.groupby(
            sorted(file_frags_schema_group_ids, key=sort_key_func), keyfunc
        )
    }
    metrics.sort_by_sg_id_time += int((time.monotonic() - start) * 1_000_000)
    metrics.nunique_sgs_seen += len(schema_group_id_to_frags)

    return schema_group_id_to_frags


T = pt.TypeVar("T")
TVals = T | tuple["TVals", ...]


def flatten_tuple(x: tuple[TVals, ...]) -> tuple[T]:
    """
    Flatten a tuple of tuples into a single tuple. This is needed
    to handle nested tuples in the schema group identifier due to
    nested data types.
    """
    values = []
    for val in x:
        if isinstance(val, tuple):
            values.extend(flatten_tuple(val))
        else:
            values.append(val)
    return tuple(values)


def get_pieces_with_exact_row_counts(
    schema_group: IcebergSchemaGroup,
    schema_group_identifier: SchemaGroupIdentifier,
    pq_infos: list["IcebergParquetInfo"],
    fs: "PyFileSystem" | pa.fs.FileSystem,
    final_schema: pa.Schema,
    str_as_dict_cols: list[str],
    metrics: IcebergPqDatasetMetrics,
) -> list[IcebergPiece]:
    """
    Helper function for 'get_row_counts_for_schema_group' to get pieces with
    the exact row counts for a list of files (after applying filters)
    which all belong to the same schema group.
    NOTE: The file fragments are expected to have their metadata already
    populated.

    Args:
        schema_group (IcebergSchemaGroup): SchemaGroup that the files
            belong to.
        schema_group_identifier (SchemaGroupIdentifier):
            Group identifier. This is a tuple of two ordered tuples.
            The first is an ordered tuple of Iceberg Field IDs and
            the second is an ordered tuple of the corresponding
            field names.
        pq_file_fragments (list[IcebergParquetInfo]): List of files
            to get the row counts for.
        fs (PyFileSystem&quot; | pa.fs.FileSystem): Filesystem to
            use for accessing the files and getting the row count
            and metadata information.
            NOTE: This is only used when there are dict-encoded
            columns and we need to re-create the fragments from a
            new ParquetFileFormat which sets the dict-encoded
            columns correctly.
        final_schema (pa.Schema): Target schema for the Iceberg table.
        str_as_dict_cols (list[str]): List of column names (in final schema)
            that will be read with dictionary encoding.
        metrics (IcebergPqDatasetMetrics): Metrics to update in place.

    Returns:
        list[IcebergPiece]: Pieces with exact row count information.
    """

    # For frag.scanner().count_rows(), we use the expected schema instead
    # of the file schema. This schema should be a less-restrictive
    # superset of the file schema, so it should be valid.
    read_schema: pa.Schema = schema_group.read_schema

    # When using frag.scanner().count_rows(),
    # we need to use the schema with pa.dictionary fields for the
    # dictionary encoded fields. This is important for correctness
    # since this is what we will do during the actual read
    # (see 'get_dataset_for_schema_group'). Without this,
    # the row count may be inaccurate (potentially due to bugs in Arrow).
    # See [BSE-2790] for more context.

    # Create a ParquetFileFormat where we specify the columns to
    # dict encode.
    col_rename_map: dict[str, str] = {
        final_schema.field(i).name: read_schema.field(i).name
        for i in range(len(final_schema.names))
    }
    schema_group_str_as_dict_cols: list[str] = [
        col_rename_map[f] for f in str_as_dict_cols
    ]
    pq_file_format = ds.ParquetFileFormat(
        dictionary_columns=schema_group_str_as_dict_cols
    )
    # Set columns to be read as dictionary encoded in the read schema
    read_schema = schema_with_dict_cols(read_schema, schema_group_str_as_dict_cols)

    # Create ParquetFileFragments for parallel row-count calculation
    start = time.monotonic()
    pq_file_fragments: list[ds.ParquetFileFragment] = []
    for pq_info in pq_infos:
        pq_file_fragments.append(
            pq_file_format.make_fragment(pq_info.standard_path, fs)
        )
    metrics.file_frags_creation_time += int((time.monotonic() - start) * 1_000_000)

    pieces: list[IcebergPiece] = []

    # Determine the row counts for each file fragment in parallel.
    # Presumably the work is partitioned more or less equally among ranks,
    # and we are mostly (or just) reading metadata, so we assign four IO
    # threads to every rank.
    # XXX Use a separate env var for this?
    nthreads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), 4)
    pa_orig_io_thread_count = pa.io_thread_count()
    pa.set_io_thread_count(nthreads)
    pa_orig_cpu_thread_count = pa.cpu_count()
    pa.set_cpu_count(nthreads)
    try:
        t0: float = time.monotonic()
        file_row_counts = arrow_cpp.fetch_parquet_frag_row_counts(
            pq_file_fragments, schema_group.expr_filter, read_schema
        )
        for frag, file_row_count in zip(pq_file_fragments, file_row_counts):
            pieces.append(
                IcebergPiece(frag.path, -1, schema_group_identifier, file_row_count)
            )
            metrics.get_row_counts_nrows += file_row_count
            metrics.get_row_counts_nrgs += frag.num_row_groups
            metrics.get_row_counts_total_bytes += sum(
                rg.total_byte_size for rg in frag.row_groups
            )
        metrics.exact_row_counts_time += int((time.monotonic() - t0) * 1_000_000)
    finally:
        # Restore pyarrow default IO thread count
        pa.set_io_thread_count(pa_orig_io_thread_count)
        pa.set_cpu_count(pa_orig_cpu_thread_count)

    return pieces


def get_row_counts_for_schema_group(
    schema_group_identifier: SchemaGroupIdentifier,
    pq_infos: list["IcebergParquetInfo"],
    fs: "PyFileSystem" | pa.fs.FileSystem,
    final_schema: pa.Schema,
    str_as_dict_cols: list[str],
    metrics: IcebergPqDatasetMetrics,
    row_level: bool = False,
    expr_filter_f_str: str | None = None,
    filter_scalars: list[tuple[str, pt.Any]] | None = None,
) -> list[IcebergPiece]:
    """
    Get the row counts for files belonging to the same
    Schema Group. Note that this is the row count
    after applying the provided filters in the row_level=True
    case. In the row_level=False case, we only apply the filters
    at the row group metadata level and hence the row counts
    are simply the number of rows in the row groups that weren't
    entirely pruned out.
    Note that this also validates the schemas of the files
    to ensure that they are compatible with this schema
    group.
    NOTE: This function is completely local and doesn't
    do any synchronization. It may raise Exceptions.
    The caller is expected to handle the error-synchronization.

    NOTE: The file fragments are expected to have their metadata already
    populated.

    Args:
        schema_group_identifier (SchemaGroupIdentifier):
            Group identifier. This is a tuple of two ordered tuples.
            The first is an ordered tuple of Iceberg Field IDs and
            the second is an ordered tuple of the corresponding
            field names.
        pq_infos (list[IcebergFileInfo]): List of files
            to get the row counts for.
        fs (PyFileSystem | pa.fs.FileSystem): Filesystem to
             use for accessing the files and getting the row count
             and metadata information.
             NOTE: This is only used in the row_level=True case when
             there are dict-encoded columns and we need to re-create the
             fragments from a new ParquetFileFormat which sets the
             dict-encoded columns correctly.
        final_schema (pa.Schema): Target schema for the Iceberg table.
            This will be used to generate a "read_schema" for this
            schema group.
        str_as_dict_cols (list[str]): List of column names
            that will be read with dictionary encoding.
        metrics: (IcebergPqDatasetMetrics): Metrics to update in place.
        row_level (bool): Whether the row counts need to be done with
            row-level filtering or if row-group level filtering
            is sufficient.
        expr_filter_f_str (str, optional): f-string to use for
            generating the filter. See description
            of 'generate_expr_filter' for more details. Defaults to None.
        filter_scalars (list[tuple[str, Any]], optional): The scalars
            to use for generating the filter. See description
            of 'generate_expr_filter' for more details. Defaults to None.

    Returns:
        list[IcebergPiece]: List of 'IcebergPiece's.
            In the row_level=False case, this includes details about
            the selected row groups within the files.
    """

    # Create a temporary IcebergSchemaGroup.
    schema_group: IcebergSchemaGroup = IcebergSchemaGroup(
        iceberg_field_ids=schema_group_identifier[0],
        parquet_field_names=schema_group_identifier[1],
        final_schema=final_schema,
        expr_filter_f_str=expr_filter_f_str,
        filter_scalars=filter_scalars,
    )

    ## 1. Validate that the file schemas are all compatible.
    # This will incur an expensive metadata read, so its behind a debug flag
    if bodo.check_parquet_schema:
        pq_file_format = ds.ParquetFileFormat()
        for pq_info in pq_infos:
            frag = pq_file_format.make_fragment(pq_info.standard_path, fs)
            file_schema = frag.metadata.schema.to_arrow_schema()
            try:
                # We use the original read-schema from the schema group
                # here (i.e. without the dictionary types) since that's
                # what the file is supposed to contain.
                validate_file_schema_compatible_with_read_schema(
                    file_schema, schema_group.read_schema
                )
            except Exception as e:
                msg = f"Schema of file {pq_info.orig_path} is not compatible.\n" + str(
                    e
                )
                raise BodoError(msg)

    ## 2. Perform filtering to get row counts and construct the IcebergPieces.
    pieces: list[IcebergPiece] = []
    if row_level:
        ## 2.1 If we need to get exact row counts, we will use the dataset
        # scanner API and apply the filter. Arrow will try to calculate this by
        # by reading only the file's metadata, and if it needs to
        # access data it will read as little as possible (only the
        # required columns and only subset of row groups if possible).
        pieces = get_pieces_with_exact_row_counts(
            schema_group,
            schema_group_identifier,
            pq_infos,
            fs,
            final_schema,
            str_as_dict_cols,
            metrics,
        )
    else:
        ## 2.2 If we are only doing piece level filtering, we can reuse Iceberg-level
        # row counts for the estimates. This skips row-group filtering
        pieces: list[IcebergPiece] = []
        for pq_info in pq_infos:
            pieces.append(
                IcebergPiece(
                    pq_info.standard_path,
                    -1,
                    schema_group_identifier,
                    pq_info.row_count,
                )
            )
            metrics.get_row_counts_nrows += pq_info.row_count

    return pieces


def flatten_concatenation(list_of_lists: list[list[pt.Any]]) -> list[pt.Any]:
    """
    Helper function to flatten a list of lists into a
    single list.

    Ref: https://realpython.com/python-flatten-list/

    Args:
        list_of_lists (list[list[Any]]): List to flatten.

    Returns:
        list[Any]: Flattened list.
    """
    flat_list: list[pt.Any] = []
    for row in list_of_lists:
        flat_list += row
    return flat_list


def warn_if_non_ideal_io_parallelism(
    g_total_rgs: int, g_total_size_bytes: int, protocol: str
):
    """
    Helper function for raising warnings on rank-0 when
    the file properties are not ideal for effective
    parallelism.

    Args:
        g_total_rgs (int): Total number of row groups (global).
        g_total_size_bytes (int): Total size of all row groups
            to read from all files (global).
        protocol (str): Filesystem protocol. This is used
            to determine if we're reading from a remote
            filesystem.
    """
    if bodo.get_rank() == 0 and g_total_rgs < bodo.get_size() and g_total_rgs != 0:
        warnings.warn(
            BodoWarning(
                f"Total number of row groups in Iceberg dataset ({g_total_rgs}) is too small for effective IO parallelization."
                f"For best performance the number of row groups should be greater than the number of workers ({bodo.get_size()}). "
                "For more details, refer to https://docs.bodo.ai/latest/file_io/#parquet-section."
            )
        )
    # Print a warning if average row group size < 1 MB and reading from remote filesystem
    if g_total_rgs == 0:
        avg_row_group_size_bytes = 0
    else:
        avg_row_group_size_bytes = g_total_size_bytes // g_total_rgs
    if (
        bodo.get_rank() == 0
        and g_total_size_bytes >= (20 * 1024 * 1024)
        and avg_row_group_size_bytes < (1024 * 1024)
        and protocol in REMOTE_FILESYSTEMS
    ):
        warnings.warn(
            BodoWarning(
                f"Parquet (Iceberg) average row group size is small ({avg_row_group_size_bytes} bytes) "
                "and can have negative impact on performance when reading from remote sources."
            )
        )


@run_rank0
def get_rest_catalog_config(conn: str) -> tuple[str, str, str] | None:
    """
    Get the configuration for a rest catalog connection string.
    @param conn: Iceberg connection string.
    @return: Tuple of uri, user_token, warehouse if successful, None otherwise (e.g. invalid connection string or not a rest catalog).
    """
    parsed_conn = urlparse(conn)
    if parsed_conn.scheme.lower() != "rest":
        return None
    parsed_conn = parsed_conn._replace(scheme="https")
    parsed_params = parse_qs(parsed_conn.query)
    # Clear the params
    parsed_conn = parsed_conn._replace(query="")
    uri = parsed_conn.geturl()

    user_token, credential, warehouse = (
        parsed_params.get("token"),
        parsed_params.get("credential"),
        parsed_params.get("warehouse"),
    )
    if user_token is not None:
        user_token = user_token[0]
    if warehouse is not None:
        warehouse = warehouse[0]
    # If we have a credential, we need to use it to get a user_token
    if credential is not None:
        credential = credential[0]
        client_id, client_secret = credential.split(":")
        user_token_request = requests.post(
            f"{uri}/v1/oauth/tokens",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if user_token_request.status_code != 200:
            raise BodoError(
                f"Unable to authenticate with {uri}. Please check your connection string."
            )
        user_token = user_token_request.json().get("access_token")

    if user_token is None:
        raise BodoError(
            f"Unable to authenticate with {uri}. Please check your connection string."
        )
    return uri, str(user_token), str(warehouse)


@numba.njit
def get_rest_catalog_fs(
    catalog_uri: str,
    bearer_token: str,
    warehouse: str,
    database_schema: str,
    table_name: str,
) -> pa.fs.FileSystem:
    """
    Get a filesystem object for the rest catalog.
    args:
        catalog_uri: URI of the rest catalog.
        bearer_token: Bearer token for authentication.
        warehouse: Warehouse name.
        database_schema: Schema the relevant table is in
        table_name: Name of the table
    """
    creds_provider = create_iceberg_aws_credentials_provider(
        catalog_uri, bearer_token, warehouse, database_schema, table_name
    )
    region = get_region_from_creds_provider(creds_provider)
    return create_s3_fs_instance(credentials_provider=creds_provider, region=region)


def get_iceberg_pq_dataset(
    conn: str,
    database_schema: str,
    table_name: str,
    typing_pa_table_schema: pa.Schema,
    str_as_dict_cols: list[str],
    iceberg_filter: str | None = None,
    expr_filter_f_str: str | None = None,
    filter_scalars: list[tuple[str, pt.Any]] | None = None,
    force_row_level_read: bool = True,
) -> IcebergParquetDataset:
    """
    Get IcebergParquetDataset object for the specified table.
    NOTE: This must be called on all ranks in parallel since
    all processing is parallelized for best performance.

    Args:
        conn (str): Iceberg connection string provided by the user.
        database_schema (str): Iceberg database that the table
            exists in.
        table_name (str): Name of the table to use.
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
    ev = tracing.Event("get_iceberg_pq_dataset")
    metrics = IcebergPqDatasetMetrics()
    comm = MPI.COMM_WORLD

    # Get list of files. This is the list after
    # applying the iceberg_filter (metadata-level).
    start_time = time.monotonic()
    (
        pq_infos,
        snapshot_id,
        all_schemas,
        get_file_to_schema_us,
    ) = get_iceberg_file_list_parallel(
        conn,
        database_schema,
        table_name,
        iceberg_filter,
    )
    metrics.file_to_schema_time_us = get_file_to_schema_us
    metrics.file_list_time += int((time.monotonic() - start_time) * 1_000_000)

    # If no files exist/match, return an empty dataset.
    if len(pq_infos) == 0:
        return IcebergParquetDataset(
            True,
            conn,
            database_schema,
            table_name,
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
    # Clean up file paths further and remove filesystem info
    pq_abs_path_file_list, parse_result, protocol = parse_fpath(
        [x.standard_path for x in pq_infos]
    )
    pq_abs_path_file_list, _ = get_fpath_without_protocol_prefix(
        pq_abs_path_file_list, protocol, parse_result
    )
    for x, abs_path in zip(pq_infos, pq_abs_path_file_list):
        x.standard_path = abs_path

    # Construct a filesystem.
    fs: "PyFileSystem" | pa.fs.FileSystem
    if protocol in {"gcs", "gs"}:
        validate_gcsfs_installed()
    rest_catalog_conf = get_rest_catalog_config(conn)
    if rest_catalog_conf is not None:
        uri, bearer_token, warehouse = rest_catalog_conf
        fs = get_rest_catalog_fs(
            uri, bearer_token, warehouse, database_schema, table_name
        )
    else:
        fs = getfs(
            pq_abs_path_file_list,
            protocol,
            storage_options=None,
            parallel=True,
        )
    metrics.get_fs_time += int((time.monotonic() - start_time) * 1_000_000)

    if expr_filter_f_str is not None and len(expr_filter_f_str) == 0:
        expr_filter_f_str = None
    if filter_scalars is None:
        filter_scalars = []
    if tracing.is_tracing():
        ev.add_attribute("g_expr_filter_f_str", str(expr_filter_f_str))

    # 1. Select a slice of the list of files based on the rank.
    n_pes, rank = bodo.get_size(), bodo.get_rank()
    total_num_files = len(pq_abs_path_file_list)
    start = bodo.libs.distributed_api.get_start(total_num_files, n_pes, rank)
    end = bodo.libs.distributed_api.get_end(total_num_files, n_pes, rank)

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
            pq_format.make_fragment(pq_info.standard_path, fs)
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
    sync_and_reraise_error(err, _is_parallel=True)

    # Analyze number of row groups, their sizes, etc. and print warnings
    # similar to what we do for Parquet.
    g_total_rows = comm.allreduce(metrics.get_row_counts_nrows, op=MPI.SUM)
    g_total_rgs = comm.allreduce(metrics.get_row_counts_nrgs, op=MPI.SUM)
    g_total_size_bytes = comm.allreduce(metrics.get_row_counts_total_bytes, op=MPI.SUM)
    warn_if_non_ideal_io_parallelism(g_total_rgs, g_total_size_bytes, protocol)

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
    iceberg_pq_dataset: IcebergParquetDataset = IcebergParquetDataset(
        row_level,
        conn,
        database_schema,
        table_name,
        typing_pa_table_schema,
        [x.orig_path for x in pq_infos],
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


def distribute_pieces(pieces: list[IcebergPiece]) -> list[IcebergPiece]:
    """
    Distribute Iceberg File pieces between all ranks so that all ranks
    have to read roughly the same number of rows.
    To do this, we use a greedy algorithm described here:
    https://www.cs.cmu.edu/~15451-f23/lectures/lecture19-approx.pdf.
    The algorithm is deterministic, i.e. should yield the same result
    on all ranks. Therefore, no synchronization is performed.

    Args:
        pieces (list[IcebergPiece]): List of file pieces to
            distribute between all ranks. This must be the global
            list of pieces and must be ordered the same on all ranks.

    Returns:
        list[IcebergPiece]: List of pieces assigned that this
            rank should read. This will be ordered by the
            schema_group_idx so that all files in the same SchemaGroup
            are next to each other.
    """

    # Use a simple greedy algorithm to assign pieces to respective ranks.
    # Sort the pieces from the largest to smallest.
    # Iterate through the pieces and assign it to the rank with the
    # fewest rows.
    # XXX There's a concern that if the piece is at a row-group level,
    # row groups from the same file may get assigned to different
    # ranks, which can lead to wasted IO. To alleviate this, we could
    # first do this algorithm on the file-level pieces. If there's
    # significant skew, we can break up the files-level pieces into
    # row-group-level pieces and repeat the algorithm.

    import heapq

    comm = MPI.COMM_WORLD
    myrank: int = comm.Get_rank()
    n_pes: int = comm.Get_size()

    # Sort the pieces
    sorted_pieces: list[IcebergPiece] = list(
        sorted(pieces, key=lambda p: (p._bodo_num_rows, p.path))
    )

    pieces_myrank: list[IcebergPiece] = []

    # To assign the pieces, we iterate through the pieces and assign the piece
    # to the rank with the least rows already assigned to it. To keep track
    # of the ranks and how many rows they're assigned, we use a heap
    # where each element is of the form (num_rows, rank). This allows us
    # to get the min rank in logarithmic time in each iteration.
    rank_heap: list[tuple[int, int]] = [(0, i) for i in range(n_pes)]
    heapq.heapify(rank_heap)

    for piece in sorted_pieces:
        piece_nrows = piece._bodo_num_rows
        least_rows, rank = heapq.heappop(rank_heap)
        if rank == myrank:
            pieces_myrank.append(piece)
        heapq.heappush(rank_heap, (least_rows + piece_nrows, rank))

    # Sort by schema_group_idx before returning
    pieces_myrank = list(
        sorted(pieces_myrank, key=lambda p: (p.schema_group_idx, p.path))
    )

    return pieces_myrank


def get_dataset_for_schema_group(
    schema_group: IcebergSchemaGroup,
    files: list[str],
    files_rows_to_read: list[int],
    final_schema: pa.Schema,
    str_as_dict_cols: list[str],
    filesystem: "PyFileSystem" | pa.fs.FileSystem,
    start_offset: int,
    len_all_fpaths: int,
) -> tuple["Dataset", pa.Schema, int]:
    """
    Create an Arrow Dataset for files belonging
    to the same Iceberg Schema Group.
    Args:
        schema_group (IcebergSchemaGroup): Schema Group for
            the files.
        files (list[str]): List of files.
        files_rows_to_read (list[int]): Number of rows to
            read from each of the files.
        final_schema (pa.Schema): Target schema for the final
            Iceberg table. This is used for certain column
            renaming during the read.
        str_as_dict_cols (list[str]): List of column names
            that must be read with dictionary encoding.
        filesystem (PyFileSystem | pa.fs.FileSystem): Filesystem
            to use for reading the files.
        start_offset (int): The starting row offset to read from
            in the first file.
        len_all_fpaths (int): Total number of files across all schema
            groups that this rank will read. This is used in some
            heuristics to decide whether or not to split the
            file-level dataset into row-group-level dataset.
    Returns:
        tuple[Dataset, pa.Schema, int]:
            - Arrow Dataset for the files in the
            schema group.
            - The schema that the Dataset will use
            while reading the file(s). This may be slightly
            different than the read_schema of the schema-group
            since some columns may be dict-encoded.
            - Updated start_offset.
    """
    read_schema: pa.Schema = schema_group.read_schema

    # Create a ParquetFileFormat where we specify the columns to
    # dict encode.
    col_rename_map: dict[str, str] = {
        final_schema.field(i).name: read_schema.field(i).name
        for i in range(len(final_schema.names))
    }
    schema_group_str_as_dict_cols: list[str] = [
        col_rename_map[f] for f in str_as_dict_cols
    ]
    pq_format = ds.ParquetFileFormat(dictionary_columns=schema_group_str_as_dict_cols)

    # Set columns to be read as dictionary encoded in the read schema
    read_schema = schema_with_dict_cols(read_schema, schema_group_str_as_dict_cols)

    dataset = ds.dataset(
        files,
        filesystem=filesystem,
        schema=read_schema,
        format=pq_format,
    )

    # For the first schema group, prune out row groups if it could be beneficial:
    if (start_offset > 0) and filter_row_groups_from_start_of_dataset_heuristic(
        # We will consider number of files across all schema groups and not just
        # this one to determine whether or not to prune row groups.
        len_all_fpaths,
        start_offset,
        schema_group.expr_filter,
    ):
        # The starting offset the Parquet reader knows about is from the first
        # file, not the first row group, so we need to communicate this back to C++
        dataset, start_offset = filter_row_groups_from_start_of_dataset(
            dataset,
            start_offset,
            sum(files_rows_to_read),
            pq_format,
        )

    return dataset, read_schema, start_offset


def get_pyarrow_datasets(
    fpaths: list[str],
    file_nrows_to_read: list[int],
    file_schema_group_idxs: list[int],
    schema_groups: list[IcebergSchemaGroup],
    avg_num_pieces: float,
    is_parallel: bool,
    filesystem: "PyFileSystem" | pa.fs.FileSystem,
    str_as_dict_cols: list[str],
    start_offset: int,
    final_schema: pa.Schema,
) -> tuple[list["Dataset"], list[pa.Schema], list[pc.Expression], int]:
    """
    Get the PyArrow Datasets for the given files.
    This will return one Dataset for every unique schema
    group that these filepaths will use.
    This will also return an updated offset for the file/piece.
    Args:
        fpaths (list[str]): List of files to read from. The files
            must be ordered by their corresponding schema group.
        file_nrows_to_read (list[int]): Total number of rows this
            process is going to read from each of these files.
            From the first file, it will read starting from 'start_offset'.
        file_schema_group_idxs (list[int]): Index of the schema group
            that each of these files belongs to.
        schema_groups (list[IcebergSchemaGroup]): List of all the schema
            groups.
        avg_num_pieces (float): Average number of pieces that every
            rank will read. If a rank is going to read many more
            files than average, we assign it more IO threads.
        is_parallel (bool): Whether this is being called in parallel
            across all ranks.
        filesystem (PyFileSystem | pa.fs.FileSystem): Filesystem to
            use for reading the files.
        str_as_dict_cols (list[str]): List of column names
            that must be read with dictionary encoding.
        start_offset (int): The starting row offset to read from
            in the first piece. This is only applicable when reading at
            a row-level. If reading at a piece-level, this should be
            set to 0.
        final_schema (pa.Schema): Target schema for the final
            Iceberg table. This is used for certain column
            renaming during the read.
    Returns:
        tuple[list["Dataset"], list[pa.Schema], list[pc.Expression], int]:
            - List of Arrow Datasets. There will be one
            per Schema Group that this rank will end up reading from.
            - List of the corresponding read_schema for each of
            the datasets (based on the schema group the dataset belongs to).
            - List of the corresponding filter to apply for each of
            the datasets (based on the schema group the dataset belongs to).
            - Update row offset into the first file/piece. Only applicable
            in the row-level read case.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count == 0:
        cpu_count = 2
    default_io_threads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), cpu_count)
    max_io_threads = min(int(os.environ.get("BODO_MAX_IO_THREADS", 16)), cpu_count)
    # Assign more threads to ranks that have to read many more files
    # than the others.
    # TODO Unset this after the read??
    if (
        is_parallel
        and len(fpaths) > max_io_threads
        and len(fpaths) / avg_num_pieces >= 2.0
    ):
        pa.set_io_thread_count(max_io_threads)
    else:
        pa.set_io_thread_count(default_io_threads)

    if len(fpaths) == 0:
        return [], [], start_offset

    datasets: list["Dataset"] = []
    dataset_read_schemas: list[pa.Schema] = []
    dataset_expr_filters: list[pc.Expression] = []

    # Assuming the files are ordered by their corresponding
    # schema group index, we can iterate and group files that way.
    curr_file_idx = 0
    curr_schema_group_idx = file_schema_group_idxs[0]
    while curr_file_idx < len(fpaths):
        # Accumulate files for the group:
        curr_schema_group_files: list[str] = []
        curr_schema_group_files_rows_to_read: list[int] = []
        while (curr_file_idx < len(fpaths)) and (
            file_schema_group_idxs[curr_file_idx] == curr_schema_group_idx
        ):
            curr_schema_group_files.append(fpaths[curr_file_idx])
            curr_schema_group_files_rows_to_read.append(
                file_nrows_to_read[curr_file_idx]
            )
            curr_file_idx += 1

        # Get schema group for this set of files.
        schema_group: IcebergSchemaGroup = schema_groups[curr_schema_group_idx]
        # Get the row counts for these files.
        dataset, dataset_read_schema, new_start_offset = get_dataset_for_schema_group(
            schema_group,
            curr_schema_group_files,
            curr_schema_group_files_rows_to_read,
            final_schema,
            str_as_dict_cols,
            filesystem,
            # 'start_offset' is only applicable to the
            # first schema-group. It's 0 for the rest.
            start_offset if len(datasets) == 0 else 0,
            len(fpaths),
        )
        # Update the overall start_offset if this is the first
        # schema-group.
        start_offset = new_start_offset if len(datasets) == 0 else start_offset
        datasets.append(dataset)
        dataset_read_schemas.append(dataset_read_schema)
        dataset_expr_filters.append(schema_group.expr_filter)

        # Update the schema group index.
        if curr_file_idx < len(fpaths):
            curr_schema_group_idx = file_schema_group_idxs[curr_file_idx]

    return datasets, dataset_read_schemas, dataset_expr_filters, start_offset


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
    if protocol in {"gcs", "gs"}:
        validate_gcsfs_installed()

    fs: "PyFileSystem" | pa.fs.FileSystem
    if protocol in {"gcs", "gs"}:
        validate_gcsfs_installed()
    rest_catalog_conf = get_rest_catalog_config(conn)
    if rest_catalog_conf is not None:
        uri, bearer_token, warehouse = rest_catalog_conf
        fs = get_rest_catalog_fs(
            uri, bearer_token, warehouse, database_schema, table_name
        )
    else:
        fs = getfs(
            pq_abs_path_file_list,
            protocol,
            storage_options=None,
            parallel=True,
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


def prefetch_sf_tables(conn_str: str, table_paths: list[str]) -> None:
    "Helper function for the Python contents of prefetch_sf_tables_njit."

    import bodo_iceberg_connector as bic

    comm = MPI.COMM_WORLD
    exc = None

    conn_str = format_iceberg_conn(conn_str)
    if bodo.get_rank() == 0:
        try:
            bic.prefetch_sf_tables(conn_str, table_paths)
        except bic.IcebergError as e:
            exc = BodoError(
                f"Failed to prefetch Snowflake-managed Iceberg table paths: {e.message}"
            )

    exc = comm.bcast(exc)
    if exc is not None:
        raise exc


def prefetch_sf_tables_njit(conn_str: str, table_paths: list[str]) -> None:
    """
    Prefetch the metadata path for a list of Snowflake-managed Iceberg tables.
    This function is called in parallel across all ranks. It is mainly used
    for SQL code generation.

    Args:
        conn_str (str): Snowflake connection string to connect to.
        table_paths (list[str]): List of table paths to prefetch paths for.
    """
    pass


@overload(prefetch_sf_tables_njit)
def overload_prefetch_sf_tables_njit(conn_str, table_paths):
    def impl(conn_str, table_paths):
        with bodo.no_warning_objmode():
            prefetch_sf_tables(conn_str, table_paths)

    return impl


# ----------------------------- Iceberg Write ----------------------------- #
def are_schemas_compatible(
    pa_schema: pa.Schema, df_schema: pa.Schema, allow_downcasting: bool = False
) -> bool:
    """
    Check if the input DataFrame schema is compatible with the Iceberg table's
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

    Note that allow_downcasting should be used if the output DataFrame df will be
    casted to fit pa_schema (making sure there are no nulls, downcasting arrays).
    """
    if pa_schema.equals(df_schema):
        return True

    # If the schemas are not the same size, it is still possible that the DataFrame
    # can be appended iff the DataFrame schema is a subset of the iceberg schema and
    # each missing field is optional
    if len(df_schema) < len(pa_schema):
        # Replace df_schema with a fully expanded schema tha contains the default
        # values for missing fields.
        kept_fields = []
        for pa_field in pa_schema:
            df_field_index = df_schema.get_field_index(pa_field.name)
            if df_field_index != -1:
                kept_fields.append(df_schema.field(df_field_index))
            elif pa_field.nullable:
                # Append optional missing fields.
                kept_fields.append(pa_field)

        df_schema = pa.schema(kept_fields)

    if len(df_schema) != len(pa_schema):
        return False

    # Compare each field individually for "compatibility"
    # Only the DataFrame schema is potentially modified during this step
    for idx in range(len(df_schema)):
        df_field = df_schema.field(idx)
        pa_field = pa_schema.field(idx)
        new_field = _update_field(df_field, pa_field, allow_downcasting)
        df_schema = df_schema.set(idx, new_field)

    return df_schema.equals(pa_schema)


def _update_field(
    df_field: pa.Field, pa_field: pa.Field, allow_downcasting: bool
) -> pa.Field:
    """
    Update the field 'df_field' to match the type and nullability of 'pa_field',
    including ignoring any optional fields.
    """
    if df_field.equals(pa_field):
        return df_field

    df_type = df_field.type
    pa_type = pa_field.type

    if pa.types.is_struct(df_type) and pa.types.is_struct(pa_type):
        kept_child_fields = []
        for pa_child_field in pa_type:
            df_child_field_index = df_type.get_field_index(pa_child_field.name)
            if df_child_field_index != -1:
                kept_child_fields.append(
                    _update_field(
                        df_type.field(df_child_field_index),
                        pa_child_field,
                        allow_downcasting,
                    )
                )
            elif pa_child_field.nullable:
                # Append optional missing fields.
                kept_child_fields.append(pa_child_field)
        struct_type = pa.struct(kept_child_fields)
        df_field = df_field.with_type(struct_type)
    elif pa.types.is_map(df_type) and pa.types.is_map(pa_type):
        new_key_field = _update_field(
            df_type.key_field, pa_type.key_field, allow_downcasting
        )
        new_item_field = _update_field(
            df_type.item_field, pa_type.item_field, allow_downcasting
        )
        map_type = pa.map_(new_key_field, new_item_field)
        df_field = df_field.with_type(map_type)
    # We always convert the expected type to large list
    elif (
        pa.types.is_list(df_type)
        or pa.types.is_large_list(df_type)
        or pa.types.is_fixed_size_list(df_type)
    ) and pa.types.is_large_list(pa_type):
        new_element_field = _update_field(
            df_type.field(0), pa_type.field(0), allow_downcasting
        )
        list_type = pa.large_list(new_element_field)
        df_field = df_field.with_type(list_type)
    # We always convert the expected type to large string
    elif (
        pa.types.is_string(df_type) or pa.types.is_large_string(df_type)
    ) and pa.types.is_large_string(pa_type):
        df_field = df_field.with_type(pa.large_string())
    # We always convert the expected type to large binary
    elif (
        pa.types.is_binary(df_type)
        or pa.types.is_large_binary(df_type)
        or pa.types.is_fixed_size_binary(df_type)
    ) and pa.types.is_large_binary(pa_type):
        df_field = df_field.with_type(pa.large_binary())
    # df_field can only be downcasted as of now
    # TODO: Should support upcasting in the future if necessary
    elif (
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

    return df_field


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

    new_field: pa.Field | None = None
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
    Replaces the Iceberg field with the reference field containing the correct
    type and field ID. In the case of nested types, we recurse to ensure we
    accurately select the proper subset of structs.

    Note: The ref_field must also contain metadata with the Iceberg field ID.
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
        # Reference type may have already been converted to large list.
        assert is_pyarrow_list_type(ref_field.type), ref_field
        new_value_field = with_iceberg_field_id_md_from_ref_field(
            field.type.value_field, ref_field.type.value_field
        )
        new_field = field.with_type(pa.list_(new_value_field)).with_metadata(new_md)
    elif pa.types.is_fixed_size_list(field.type):
        # Reference type may have already been converted to large list.
        assert is_pyarrow_list_type(ref_field.type), ref_field
        new_value_field = with_iceberg_field_id_md_from_ref_field(
            field.type.value_field, ref_field.type.value_field
        )
        new_field = field.with_type(
            pa.list_(new_value_field, list_size=field.type.list_size)
        ).with_metadata(new_md)
    elif pa.types.is_large_list(field.type):
        # Reference type may have already been converted to large list.
        assert is_pyarrow_list_type(ref_field.type), ref_field
        new_value_field = with_iceberg_field_id_md_from_ref_field(
            field.type.value_field, ref_field.type.value_field
        )
        new_field = field.with_type(pa.large_list(new_value_field)).with_metadata(
            new_md
        )
    elif pa.types.is_struct(field.type):
        assert pa.types.is_struct(ref_field.type), ref_field
        new_children_fields = []
        field_type = field.type
        ref_type = ref_field.type
        for child_field in field_type:
            ref_field_index = ref_type.get_field_index(child_field.name)
            ref_field = ref_type.field(ref_field_index)
            new_children_fields.append(
                with_iceberg_field_id_md_from_ref_field(child_field, ref_field)
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
        new_field = ref_field.with_metadata(new_md)
    return new_field


def add_iceberg_field_id_md_to_pa_schema(
    schema: pa.Schema, ref_schema: pt.Optional[pa.Schema] = None
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
            # This ensures we select the correct subset of any structs.
            new_field = with_iceberg_field_id_md_from_ref_field(field, ref_field)
            new_fields.append(new_field)
        pyarrow_schema = pa.schema(new_fields)
        return connector.schema_helper.convert_arrow_schema_to_large_types(
            pyarrow_schema
        )


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
    DataFrame typechecking, collect typing-related information for
    Iceberg writes, fill in nulls, and project across all ranks.
    """
    ev = tracing.Event("iceberg_get_table_details_before_write")

    import bodo_iceberg_connector as connector

    comm = MPI.COMM_WORLD

    already_exists = None
    comm_exc = None
    iceberg_schema_id = None
    partition_spec = []
    sort_order = []
    iceberg_schema_str = ""
    output_pyarrow_schema = None
    mode = ""
    table_loc = ""

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
            already_exists = iceberg_schema_id is not None
            iceberg_schema_id = iceberg_schema_id if already_exists else -1

            if already_exists and if_exists == "fail":
                # Ideally we'd like to throw the same error as pandas
                # (https://github.com/pandas-dev/pandas/blob/4bfe3d07b4858144c219b9346329027024102ab6/pandas/io/sql.py#L833)
                # but using values not known at compile time, in Exceptions
                # doesn't seem to work with Numba
                raise ValueError(f"Table already exists.")

            if already_exists:
                mode = if_exists
            else:
                if if_exists == "replace":
                    mode = "replace"
                else:
                    mode = "create"

            if if_exists != "append":
                # In the create/replace case, disregard some of the properties
                pa_schema = None
                iceberg_schema_str = ""
                partition_spec = []
                sort_order = []
            else:
                # Ensure that all column names in the partition spec and sort order are
                # in the DataFrame being written
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
            output_pyarrow_schema = add_iceberg_field_id_md_to_pa_schema(
                df_schema, ref_schema=pa_schema
            )

            if (if_exists != "append") or (not already_exists):
                # When the table doesn't exist, i.e. we're creating a new one,
                # we need to create iceberg_schema_str from the PyArrow schema
                # of the dataframe.
                iceberg_schema_str = connector.pyarrow_to_iceberg_schema_str(
                    output_pyarrow_schema
                )

        except connector.IcebergError as e:
            comm_exc = BodoError(e.message)
        except Exception as e:
            comm_exc = e

    comm_exc = comm.bcast(comm_exc)
    if isinstance(comm_exc, Exception):
        raise comm_exc

    table_loc = comm.bcast(table_loc)
    already_exists = comm.bcast(already_exists)
    mode = comm.bcast(mode)
    iceberg_schema_id = comm.bcast(iceberg_schema_id)
    partition_spec = comm.bcast(partition_spec)
    sort_order = comm.bcast(sort_order)
    iceberg_schema_str = comm.bcast(iceberg_schema_str)
    output_pyarrow_schema = comm.bcast(output_pyarrow_schema)

    ev.finalize()

    return (
        table_loc,
        already_exists,
        iceberg_schema_id,
        partition_spec,
        sort_order,
        iceberg_schema_str,
        output_pyarrow_schema,
        mode,
    )


def generate_data_file_info(
    iceberg_files_info: pt.List[pt.Tuple[pt.Any]],
) -> pt.Tuple[pt.List[str], pt.List[int], pt.List[pt.Dict[str, pt.Any]]]:
    """
    Collect C++ Iceberg File Info to a single rank
    and process before handing off to the connector / committing functions
    """
    from bodo.mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Information we need:
    # 1. File names
    # 2. file_size_in_bytes

    # Metrics we provide to Iceberg:
    # 1. rowCount -- Number of rows in this file
    # 2. valueCounts -- Number of records per field id. This is most useful for
    #    nested data types where each row may have multiple records.
    # 3. nullValueCounts - Null count per field id.
    # 4. lowerBounds - Lower bounds per field id.
    # 5. upperBounds - Upper bounds per field id.

    def extract_and_gather(i: int) -> pt.List[pt.Any]:
        """Extract field i from iceberg_files_info
        and gather the results on rank 0.

        Args:
            i (int): The field index

        Returns:
            pt.List[pt.Any]: The gathered result
        """
        values_local = [x[i] for x in iceberg_files_info]
        values_local_list = comm.gather(values_local)
        # Flatten the list of lists
        return (
            [item for sub in values_local_list for item in sub]
            if comm.Get_rank() == 0
            else None
        )

    fnames = extract_and_gather(0)

    # Collect the metrics
    record_counts_local = np.array([x[1] for x in iceberg_files_info], dtype=np.int64)
    file_sizes_local = np.array([x[2] for x in iceberg_files_info], dtype=np.int64)
    record_counts = bodo.gatherv(record_counts_local).tolist()
    file_sizes = bodo.gatherv(file_sizes_local).tolist()
    # Collect the file based metrics
    value_counts = extract_and_gather(3)
    null_counts = extract_and_gather(4)
    lower_bounds = extract_and_gather(5)
    upper_bounds = extract_and_gather(6)
    metrics = [
        # Note: These names must match Metrics.java fields in Iceberg.
        {
            "rowCount": record_counts[i],
            "valueCounts": value_counts[i],
            "nullValueCounts": null_counts[i],
            "lowerBounds": lower_bounds[i],
            "upperBounds": upper_bounds[i],
        }
        for i in range(len(record_counts))
    ]
    return fnames, file_sizes, metrics


def register_table_write(
    transaction_id: int,
    conn_str: str,
    db_name: str,
    table_name: str,
    table_loc: str,
    fnames: list[str],
    file_size_bytes: list[int],
    all_metrics: dict[str, pt.Any],  # TODO: Explain?
    iceberg_schema_id: int,
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
            transaction_id,
            conn_str,
            db_name,
            table_name,
            table_loc,
            fnames,
            file_size_bytes,
            all_metrics,
            schema_id,
            mode,
        )

    success = comm.bcast(success)
    ev.finalize()
    return success


@run_rank0
def remove_transaction(
    transaction_id: int,
    conn_str: str,
    db_name: str,
    table_name: str,
):
    """Indicate that a transaction is no longer
    needed and can be remove from any internal state.
    This DOES NOT finalize or commit a transaction.

    Args:
        transaction_id (int): Transaction ID to remove.
        conn_str (str): Connection string for indexing into our object list.
        db_name (str): Name of the database for indexing into our object list.
        table_name (str): Name of the table for indexing into our object list.
    """
    import bodo_iceberg_connector

    bodo_iceberg_connector.remove_transaction(
        transaction_id, conn_str, db_name, table_name
    )


@run_rank0
def fetch_puffin_metadata(
    transaction_id: int,
    conn_str: str,
    db_name: str,
    table_name: str,
) -> tuple[int, int, str]:
    """Fetch the puffin file metadata that we need from the committed
    transaction to write the puffin file. These are the:
        1. Snapshot ID for the committed data
        2. Sequence Number for the committed data
        3. The Location at which to write the puffin file.

    Args:
        transaction_id (int): Transaction ID to remove.
        conn_str (str): Connection string for indexing into our object list.
        db_name (str): Name of the database for indexing into our object list.
        table_name (str): Name of the table for indexing into our object list.

    Returns:
        tuple[int, int, str]: Tuple of the snapshot ID, sequence number, and
        location at which to write the puffin file.
    """
    import bodo_iceberg_connector

    ev = tracing.Event("fetch_puffin_file_metadata")
    metadata = bodo_iceberg_connector.fetch_puffin_metadata(
        transaction_id, conn_str, db_name, table_name
    )
    ev.finalize()
    return metadata


@run_rank0
def commit_statistics_file(
    conn_str: str,
    db_name: str,
    table_name: str,
    snapshot_id: int,
    statistic_file_info,
):
    """
    Commit the statistics file to the iceberg table. This occurs after
    the puffin file has already been written and records the statistic_file_info
    in the metadata.

    Args:
        conn_str (str): The Iceberg connector string.
        db_name (str): The iceberg database name.
        table_name (str): The iceberg table.
        statistic_file_info (bodo_iceberg_connector.StatisticsFile):
            The Python object containing the statistics file information.
    """
    import bodo_iceberg_connector

    ev = tracing.Event("commit_statistics_file")
    bodo_iceberg_connector.commit_statistics_file(
        conn_str, db_name, table_name, snapshot_id, statistic_file_info
    )
    ev.finalize()


@run_rank0
def table_columns_have_theta_sketches(conn_str: str, db_name: str, table_name: str):
    import bodo_iceberg_connector

    return bodo_iceberg_connector.table_columns_have_theta_sketches(
        conn_str, db_name, table_name
    )


@run_rank0
def table_columns_enabled_theta_sketches(conn_str: str, db_name: str, table_name: str):
    """
    Get an array of booleans indicating whether each column in the table
    has theta sketches enabled, as per the table property of
    'bodo.write.theta_sketch_enabled.<column_name>'.

    Args:
        conn_str (str): The Iceberg connector string.
        db_name (str): The iceberg database name.
        table_name (str): The iceberg table.
    """
    import bodo_iceberg_connector

    return bodo_iceberg_connector.table_columns_enabled_theta_sketches(
        conn_str, db_name, table_name
    )


@run_rank0
def get_old_statistics_file_path(
    txn_id: int, conn_str: str, db_name: str, table_name: str
):
    """
    Get the old puffin file path from the connector. We know that the puffin file
    must exist because of previous checks.
    """
    import bodo_iceberg_connector

    return bodo_iceberg_connector.get_old_statistics_file_path(
        txn_id, conn_str, db_name, table_name
    )


def register_table_merge_cow(
    conn_str: str,
    db_name: str,
    table_name: str,
    table_loc: str,
    old_fnames: list[str],
    new_fnames: list[str],
    file_size_bytes: list[int],
    all_metrics: dict[str, list[pt.Any]],  # TODO: Explain?
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
            file_size_bytes,
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
install_py_obj_class(
    types_name="pyobject_of_list_type",
    python_type=None,
    module=this_module,
    class_name="PyObjectOfListType",
    model_name="PyObjectOfListModel",
)

# Create a type for the Iceberg StatisticsFile object
# if we have the connector.
statistics_file_type = None
try:
    import bodo_iceberg_connector

    statistics_file_type = bodo_iceberg_connector.StatisticsFile
except ImportError:
    pass

install_py_obj_class(
    types_name="statistics_file_type",
    python_type=statistics_file_type,
    module=this_module,
    class_name="StatisticsFileType",
    model_name="StatisticsFileModel",
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
    arrow_fs,
    sketch_collection,
    bucket_region,
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
        is_parallel (bool): Whether the write is occurring on a distributed DataFrame
        expected_schema (pyarrow.Schema): Expected schema of output PyArrow table written
            to Parquet files in the Iceberg table. None if not necessary
        arrow_fs (Arrow.fs.FileSystem): Optional Arrow FileSystem object to use for writing, will fallback to parsing
            the table_loc if not provided
        sketch_collection: collection of theta sketches being used to build NDV values during write

    Returns:
        Distributed list of written file info needed by Iceberg for committing
        1) file_path (after the table_loc prefix)
        2) record_count / Number of rows
        3) File size in bytes
        4) *partition-values
    """
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
        arrow_fs,
        sketch_collection,
    )

    return iceberg_files_info


@numba.njit
def iceberg_write(
    conn,
    database_schema,
    table_name,
    bodo_table,
    col_names,
    create_table_info,
    # Same semantics as pandas to_sql for now
    if_exists,
    is_parallel,
    df_pyarrow_schema,  # Additional Param to Compare Compile-Time and Iceberg Schema
    n_cols,
    allow_downcasting=False,
):  # pragma: no cover
    """
    Iceberg Basic Write Implementation for parquet based tables.
    Args:
        conn (str): connection string
        database_schema (str): schema in iceberg database
        table_name (str): name of iceberg table
        bodo_table : table object to pass to c++
        col_names : array object containing column names (passed to c++)
        if_exists (str): behavior when table exists. must be one of ['fail', 'append', 'replace']
        is_parallel (bool): whether the write is occurring on a distributed DataFrame
        df_pyarrow_schema (pyarrow.Schema): PyArrow schema of the DataFrame being written
        allow_downcasting (bool): Perform write downcasting on table columns to fit Iceberg schema
            This includes both type and nullability downcasting

    Raises:
        ValueError, Exception, BodoError
    """

    ev = tracing.Event("iceberg_write_py", is_parallel)
    # Supporting REPL requires some refactor in the parquet write infrastructure,
    # so we're not implementing it for now. It will be added in a following PR.
    assert is_parallel, "Iceberg Write only supported for distributed DataFrames"
    with bodo.no_warning_objmode(
        txn_id="i8",
        table_loc="unicode_type",
        iceberg_schema_id="i8",
        partition_spec="python_list_of_heterogeneous_tuples_type",
        sort_order="python_list_of_heterogeneous_tuples_type",
        iceberg_schema_str="unicode_type",
        output_pyarrow_schema="pyarrow_schema_type",
        mode="unicode_type",
        catalog_uri="unicode_type",
        bearer_token="unicode_type",
        warehouse="unicode_type",
    ):
        (
            table_loc,
            _,
            iceberg_schema_id,
            partition_spec,
            sort_order,
            iceberg_schema_str,
            # This has the Iceberg field IDs in the metadata of every field
            # which is required for correctness.
            output_pyarrow_schema,
            mode,
        ) = get_table_details_before_write(
            table_name,
            conn,
            database_schema,
            df_pyarrow_schema,
            if_exists,
            allow_downcasting,
        )
        (
            txn_id,
            table_loc,
        ) = wrap_start_write(
            conn,
            database_schema,
            table_name,
            table_loc,
            iceberg_schema_id,
            create_table_info,
            output_pyarrow_schema,
            partition_spec,
            sort_order,
            mode,
        )
        conf = get_rest_catalog_config(conn)
        catalog_uri, bearer_token, warehouse = "", "", ""

        if conf is not None:
            catalog_uri, bearer_token, warehouse = conf
    fs = None
    if catalog_uri and bearer_token and warehouse:
        fs = create_s3_fs_instance(
            credentials_provider=create_iceberg_aws_credentials_provider(
                catalog_uri, bearer_token, warehouse, database_schema, table_name
            )
        )

    dummy_theta_sketch = bodo.io.stream_iceberg_write.init_theta_sketches_wrapper(
        alloc_false_bool_array(n_cols)
    )
    bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(table_loc, is_parallel)
    iceberg_files_info = iceberg_pq_write(
        table_loc,
        bodo_table,
        col_names,
        partition_spec,
        sort_order,
        iceberg_schema_str,
        is_parallel,
        output_pyarrow_schema,
        fs,
        dummy_theta_sketch,
        bucket_region,
    )
    arrow_filesystem_del(fs)

    with bodo.no_warning_objmode(success="bool_"):
        fnames, file_size_bytes, metrics = generate_data_file_info(iceberg_files_info)
        # Send file names, metrics and schema to Iceberg connector
        success = register_table_write(
            txn_id,
            conn,
            database_schema,
            table_name,
            table_loc,
            fnames,
            file_size_bytes,
            metrics,
            iceberg_schema_id,
            mode,
        )
        remove_transaction(txn_id, conn, database_schema, table_name)

    if not success:
        # TODO [BE-3249] If it fails due to schema changing, then delete the files.
        # Note that this might not always be possible since
        # we might not have DeleteObject permissions, for instance.
        raise BodoError("Iceberg write failed.")

    bodo.io.stream_iceberg_write.delete_theta_sketches(dummy_theta_sketch)

    ev.finalize()


@numba.generated_jit(nopython=True)
def iceberg_merge_cow_py(
    table_name,
    conn,
    database_schema,
    bodo_df,
    snapshot_id,
    old_fnames,
    is_parallel=True,
):
    df_pyarrow_schema = bodo.io.helpers.numba_to_pyarrow_schema(
        bodo_df, is_iceberg=True
    )
    col_names_py = pd.array(bodo_df.columns)
    num_cols = len(col_names_py)

    if bodo_df.is_table_format:
        bodo_table_type = bodo_df.table_type

        def impl(
            table_name,
            conn,
            database_schema,
            bodo_df,
            snapshot_id,
            old_fnames,
            is_parallel=True,
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
                num_cols,
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
        func_text += "    is_parallel=True,\n"
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
        func_text += f"        {num_cols},\n"
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
    num_cols,
    is_parallel,
):  # pragma: no cover
    ev = tracing.Event("iceberg_merge_cow_py", is_parallel)
    # Supporting REPL requires some refactor in the parquet write infrastructure,
    # so we're not implementing it for now. It will be added in a following PR.
    assert is_parallel, "Iceberg Write only supported for distributed DataFrames"

    with bodo.no_warning_objmode(
        already_exists="bool_",
        table_loc="unicode_type",
        partition_spec="python_list_of_heterogeneous_tuples_type",
        sort_order="python_list_of_heterogeneous_tuples_type",
        iceberg_schema_str="unicode_type",
        output_pyarrow_schema="pyarrow_schema_type",
        catalog_uri="unicode_type",
        bearer_token="unicode_type",
        warehouse="unicode_type",
    ):
        (
            table_loc,
            already_exists,
            _,
            partition_spec,
            sort_order,
            iceberg_schema_str,
            output_pyarrow_schema,
            _,
        ) = get_table_details_before_write(
            table_name,
            conn,
            database_schema,
            df_pyarrow_schema,
            "append",
            allow_downcasting=True,
        )
        catalog_uri, bearer_token, warehouse = "", "", ""
        conf = get_rest_catalog_config(conn)
        if conf is not None:
            catalog_uri, bearer_token, warehouse = conf

    if not already_exists:
        raise ValueError(f"Iceberg MERGE INTO: Table does not exist at write")

    arrow_fs = None
    if catalog_uri and bearer_token and warehouse:
        arrow_fs = create_s3_fs_instance(
            credentials_provider=create_iceberg_aws_credentials_provider(
                catalog_uri, bearer_token, warehouse, database_schema, table_name
            ),
        )

    dummy_theta_sketch = bodo.io.stream_iceberg_write.init_theta_sketches_wrapper(
        alloc_false_bool_array(num_cols)
    )
    bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(table_loc, is_parallel)
    iceberg_files_info = iceberg_pq_write(
        table_loc,
        bodo_table,
        col_names,
        partition_spec,
        sort_order,
        iceberg_schema_str,
        is_parallel,
        output_pyarrow_schema,
        arrow_fs,
        dummy_theta_sketch,
        bucket_region,
    )

    with bodo.no_warning_objmode(success="bool_"):
        fnames, file_size_bytes, metrics = generate_data_file_info(iceberg_files_info)

        # Send file names, metrics and schema to Iceberg connector
        success = register_table_merge_cow(
            conn,
            database_schema,
            table_name,
            table_loc,
            old_fnames,
            fnames,
            file_size_bytes,
            metrics,
            snapshot_id,
        )

    if not success:
        # TODO [BE-3249] If it fails due to snapshot changing, then delete the files.
        # Note that this might not always be possible since
        # we might not have DeleteObject permissions, for instance.
        raise BodoError("Iceberg MERGE INTO: write failed")

    bodo.io.stream_iceberg_write.delete_theta_sketches(dummy_theta_sketch)

    ev.finalize()


import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types

from bodo.io import arrow_cpp

ll.add_symbol("iceberg_pq_write_py_entry", arrow_cpp.iceberg_pq_write_py_entry)

ll.add_symbol(
    "create_iceberg_aws_credentials_provider_py_entry",
    s3_reader.create_iceberg_aws_credentials_provider_py_entry,
)
ll.add_symbol(
    "destroy_iceberg_aws_credentials_provider_py_entry",
    s3_reader.destroy_iceberg_aws_credentials_provider_py_entry,
)


@intrinsic
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
    arrow_fs,
    sketch_collection_t,
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
                lir.LiteralStructType([lir.IntType(8).as_pointer(), lir.IntType(1)]),
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
            types.optional(ArrowFs()),
            theta_sketch_collection_type,
        ),
        codegen,
    )


@run_rank0
def wrap_start_write(
    conn: str,
    database_schema: str,
    table_name: str,
    table_loc: str,
    iceberg_schema_id: int,
    create_table_info,
    output_pyarrow_schema: pa.Schema,
    partition_spec: list,
    sort_order: list,
    mode: str,
):
    """
    Wrapper around bodo_iceberg_connector.start_write to run on
    a single rank and broadcast the result.
    Necessary to not import bodo_iceberg_connector into the global context
    args:
    conn (str): connection string
    database_schema (str): schema in iceberg database
    table_name (str): name of iceberg table
    table_loc (str): location of the data/ folder in the warehouse
    iceberg_schema_id (int): iceberg schema id
    create_table_info: meta information about table and column comments
    output_pyarrow_schema (pyarrow.Schema): PyArrow schema of the dataframe being written
    partition_spec (list): partition spec
    sort_order (list): sort order
    mode (str): What write operation we are doing. This must be one of
        ['create', 'append', 'replace']
    """
    import bodo_iceberg_connector as connector

    return connector.start_write(
        conn,
        database_schema,
        table_name,
        table_loc,
        iceberg_schema_id,
        create_table_info,
        output_pyarrow_schema,
        partition_spec,
        sort_order,
        mode,
    )
