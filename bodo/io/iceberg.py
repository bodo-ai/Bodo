# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
File that contains the main functionality for the Iceberg
integration within the Bodo repo. This does not contain the
main IR transformation.
"""
import os
import re
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse
from uuid import uuid4

import numba
import numpy as np
import pyarrow as pa
from mpi4py import MPI
from numba.core import types
from numba.extending import intrinsic

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    PDCategoricalDtype,
)
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.io.fs_io import get_s3_bucket_region_njit
from bodo.io.helpers import is_nullable
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.binary_arr_ext import binary_array_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import unicode_to_utf8
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.utils import tracing
from bodo.utils.typing import BodoError, raise_bodo_error


# ----------------------------- Helper Funcs -----------------------------#
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
    ):
        raise BodoError(
            "'con' must start with one of the following: 'iceberg://', 'iceberg+file://', "
            "'iceberg+s3://', 'iceberg+thrift://', 'iceberg+http://', 'iceberg+https://', 'iceberg+glue'"
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


def _clean_schema(schema: pa.Schema) -> pa.Schema:
    """
    Constructs a new PyArrow schema that Bodo can support while conforming
    to Iceberg's typing specification
    - Converts all floating-point fields to non-nullable (since Bodo does
      not have nullable floating-point arrays)
    - Enforces that list fields are constructed using a field with name
      'element' (to aid Bodo's read_parquet infrastructure)
    """
    working_schema = schema

    for i in range(len(schema)):
        field = schema.field(i)

        # Set all floating point fields to not null by default
        if pa.types.is_floating(field.type):
            working_schema = working_schema.set(i, field.with_nullable(False))
        # Set all field names to 'element' so we can compare without worrying about
        # different names due to pyarrow ('item', 'element', 'field0', etc.) in case of lists.
        elif pa.types.is_list(field.type):
            working_schema = working_schema.set(
                i, field.with_type(pa.list_(pa.field("element", field.type.value_type)))
            )

    return working_schema


def _schemas_equal(schema: pa.Schema, other: pa.Schema) -> bool:
    """
    Determines if two PyArrow schemas are equal,
    accounting for Bodo and Iceberg-specific details
    """

    if schema.equals(other):
        return True

    schema_filtered = _clean_schema(schema)
    other_filtered = _clean_schema(other)
    return schema_filtered.equals(other_filtered)


# ----------------------------- Iceberg Read -----------------------------#
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
            # Only include Java error info in dev mode because it contains at lot of
            # unnecessary info about internal packages and dependencies.
            if (
                isinstance(e, bodo_iceberg_connector.IcebergJavaError)
                and numba.core.config.DEVELOPER_MODE
            ):  # pragma: no cover
                col_names_or_err = BodoError(f"{e.message}: {e.java_error}")
            else:
                col_names_or_err = BodoError(e.message)

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

    return (col_names, bodo_types, pyarrow_schema)


def get_iceberg_file_list(
    table_name: str, conn: str, database_schema: str, filters
) -> List[str]:
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

    assert (
        bodo.get_rank() == 0
    ), "get_iceberg_file_list should only ever be called on rank 0, as the operation requires access to the py4j server, which is only available on rank 0"

    try:
        lst = bodo_iceberg_connector.bodo_connector_get_parquet_file_list(
            conn, database_schema, table_name, filters
        )
    except bodo_iceberg_connector.IcebergError as e:
        # Only include Java error info in dev mode because it contains at lot of
        # unnecessary info about internal packages and dependencies.
        if (
            isinstance(e, bodo_iceberg_connector.IcebergJavaError)
            and numba.core.config.DEVELOPER_MODE  # type: ignore
        ):  # pragma: no cover
            raise BodoError(f"{e.message}:\n{e.java_error}")
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
    tot_rows_to_read=None,
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

    comm = MPI.COMM_WORLD

    pq_file_list_or_e = None
    # Get the list on just one rank to reduce JVM overheads
    # and general traffic to table for when there are
    # catalogs in the future.

    # Always get the list on rank 0 to avoid the need
    # to initialize a full JVM + gateway server on every rank.
    # Sonar cube only runs on rank 0, so we add no cover to avoid the warning
    if bodo.get_rank() == 0:  # pragma: no cover
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
    pq_file_list_or_e = comm.bcast(pq_file_list_or_e)

    # raise error on all processors if found (not just rank 0 which would cause hangs)
    if isinstance(pq_file_list_or_e, Exception):
        error = pq_file_list_or_e
        raise BodoError(
            f"Error reading Iceberg Table: {type(error).__name__}: {str(error)}\n"
        )

    pq_file_list: List[str] = pq_file_list_or_e

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
        conn, database_schema, table_name, typing_pa_table_schema, pq_dataset
    )
    ev.finalize()

    return iceberg_pq_dataset


# ---------------------- Iceberg Compile-Time Write ---------------------- #
# This is only used by iceberg write and has some iceberg specific behavior
_numba_pyarrow_type_map = {
    # Signed Int Types
    types.int8: pa.int8(),
    types.int16: pa.int16(),
    types.int32: pa.int32(),
    types.int64: pa.int64(),
    # Unsigned Int Types
    types.uint8: pa.uint8(),
    types.uint16: pa.uint16(),
    types.uint32: pa.uint32(),
    types.uint64: pa.uint64(),
    # Float Types (TODO: float16?)
    types.float32: pa.float32(),
    types.float64: pa.float64(),
    # Date and Time
    types.NPDatetime("ns"): pa.date64(),
    # For Iceberg, all timestamp data needs to be written
    # as microseconds, so that's the type we
    # specify. We convert our nanoseconds to
    # microseconds during write.
    # See https://iceberg.apache.org/spec/#primitive-types,
    # https://iceberg.apache.org/spec/#parquet
    # We've also made the decision to always
    # write the `timestamptz` type when writing
    # Iceberg data, similar to Spark.
    # The underlying already is in UTC already
    # for timezone aware types, and for timezone
    # naive, it won't matter.
    bodo.datetime64ns: pa.timestamp("us", "UTC"),
    # (TODO: time32, time64, ...)
}


def _numba_to_pyarrow_type(numba_type: types.ArrayCompatible):
    """
    Convert Numba / Bodo Array Types to Equivalent PyArrow Type
    This is currently only used in Iceberg and thus may conform to Iceberg type requirements
    """
    if isinstance(numba_type, ArrayItemArrayType):
        dtype = pa.list_(_numba_to_pyarrow_type(numba_type.dtype)[0])  # type: ignore

    elif isinstance(numba_type, StructArrayType):
        fields = []
        for name, inner_type in zip(numba_type.names, numba_type.data):
            pa_type, _ = _numba_to_pyarrow_type(inner_type)
            # We set nullable as true here to match the schema
            # written to parquet files, which doesn't contain
            # nullability info (and hence defaults to nullable).
            # This should be changed when we implement [BE-3247].
            fields.append(pa.field(name, pa_type, True))
        dtype = pa.struct(fields)

    elif isinstance(numba_type, DecimalArrayType):
        dtype = pa.decimal128(numba_type.precision, numba_type.scale)

    elif isinstance(numba_type, CategoricalArrayType):
        cat_dtype: PDCategoricalDtype = numba_type.dtype  # type: ignore
        dtype = pa.dictionary(
            _numba_to_pyarrow_type(cat_dtype.int_type)[0],
            _numba_to_pyarrow_type(cat_dtype.elem_type)[0],
            ordered=False if cat_dtype.ordered is None else cat_dtype.ordered,
        )

    elif numba_type == boolean_array:
        dtype = pa.bool_()
    elif numba_type in (string_array_type, bodo.dict_str_arr_type):
        dtype = pa.string()
    elif numba_type == binary_array_type:
        dtype = pa.binary()
    elif numba_type == datetime_date_array_type:
        dtype = pa.date32()
    elif isinstance(numba_type, bodo.DatetimeArrayType):
        # See note in _numba_pyarrow_type_map for
        # bodo.datetime64ns
        dtype = pa.timestamp("us", "UTC")

    elif (
        isinstance(numba_type, (types.Array, IntegerArrayType))
        and numba_type.dtype in _numba_pyarrow_type_map
    ):
        dtype = _numba_pyarrow_type_map[numba_type.dtype]  # type: ignore
    else:
        raise BodoError(
            "Conversion from Bodo array type {} to PyArrow type not supported yet".format(
                numba_type
            )
        )

    return dtype, is_nullable(numba_type)


def pyarrow_schema(df: DataFrameType) -> pa.Schema:
    """Construct a PyArrow Schema from Bodo's DataFrame Type"""
    fields = []
    for name, col_type in zip(df.columns, df.data):
        try:
            pyarrow_type, nullable = _numba_to_pyarrow_type(col_type)
        except BodoError as e:
            raise_bodo_error(e.msg, e.loc)

        fields.append(pa.field(name, pyarrow_type, nullable))
    return pa.schema(fields)


# ----------------------------- Iceberg Write -----------------------------#
@numba.njit
def gen_iceberg_pq_fname():  # pragma: no cover
    """
    Generate a random file name for Iceberg Table write
    Returns:
        str: random filename of form {rank:05}-rank-{uuid}.parquet
    """
    with numba.objmode(file_name="unicode_type"):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # Random data file name using UUID that also contains the MPI rank as a part number.
        # This is similar to how Spark does it.
        file_name = f"{rank:05}-{rank}-{uuid4()}.parquet"
    return file_name


def get_table_details_before_write(
    table_name: str,
    conn: str,
    database_schema: str,
    df_pyarrow_schema,
    if_exists: str,
):
    """
    Wrapper around bodo_iceberg_connector.get_typing_info to perform
    dataframe typechecking, collect typing-related information for
    Iceberg writes, and project across all ranks.
    """

    import bodo_iceberg_connector as connector

    comm = MPI.COMM_WORLD

    comm_exc = None
    iceberg_schema_id = None
    table_loc = ""
    partition_spec = ""
    sort_order = ""
    iceberg_schema_str = ""

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

            if (
                if_exists == "append"
                and (pa_schema is not None)
                and not _schemas_equal(pa_schema, df_pyarrow_schema)
            ):
                if numba.core.config.DEVELOPER_MODE:  # type: ignore
                    raise BodoError(
                        f"Iceberg Table and DataFrame Schemas Need to be Equal for Append\n\n"
                        f"Iceberg:\n{pa_schema}\n\n"
                        f"DataFrame:\n{df_pyarrow_schema}\n"
                    )
                else:
                    raise BodoError(
                        "Iceberg Table and DataFrame Schemas Need to be Equal for Append"
                    )

            if iceberg_schema_id is None:
                # When the table doesn't exist, i.e. we're creating a new one,
                # `iceberg_schema_str` will be empty, so we need to create it
                # from the pyarrow schema of the dataframe.
                iceberg_schema_str = connector.pyarrow_to_iceberg_schema_str(
                    df_pyarrow_schema
                )
        except connector.IcebergError as e:
            # Only include Java error info in dev mode because it contains at lot of
            # unnecessary info about internal packages and dependencies.
            if (
                isinstance(e, connector.IcebergJavaError)
                and numba.core.config.DEVELOPER_MODE
            ):  # pragma: no cover
                comm_exc = BodoError(f"{e.message}: {e.java_error}")
            else:
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

    if iceberg_schema_id is None:
        already_exists = False
        iceberg_schema_id = -1
    else:
        already_exists = True

    return (
        already_exists,
        table_loc,
        iceberg_schema_id,
        partition_spec,
        sort_order,
        iceberg_schema_str,
    )


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
    return success


@numba.njit()
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
):  # pragma: no cover
    """
    Iceberg Basic Write Implementation for parquet based tables.
    Args:
        table_name (str): name of iceberg table
        conn (str): connection string
        database_schema (str): schema in iceberg database
        bodo_table : table object to pass to c++
        col_names : array object containing column names (passed to c++)
        index_col : array object containing table index (passed to c++)
        write_index (bool): whether or not to write the index
        index_name_ptr (str): name of index column
        if_exists (str): behavior when table exists. must be one of ['fail', 'append', 'replace']
        is_parallel (bool): whether the write is occuring on a distributed dataframe
        df_pyarrow_schema (pyarrow.Schema): pyarrow schema of the dataframe being written

    Raises:
        ValueError, Exception, BodoError
    """
    # Supporting REPL requires some refactor in the parquet write infrastructure,
    # so we're not implementing it for now. It will be added in a following PR.
    assert is_parallel, "Iceberg Write only supported for distributed dataframes"
    with numba.objmode(
        already_exists="bool_",
        table_loc="unicode_type",
        iceberg_schema_id="i8",
        partition_spec="unicode_type",
        sort_order="unicode_type",
        iceberg_schema_str="unicode_type",
    ):
        (
            already_exists,
            table_loc,
            iceberg_schema_id,
            partition_spec,
            sort_order,
            iceberg_schema_str,
        ) = get_table_details_before_write(
            table_name, conn, database_schema, df_pyarrow_schema, if_exists
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

    fname = gen_iceberg_pq_fname()
    bucket_region = get_s3_bucket_region_njit(table_loc, is_parallel)
    # TODO [BE-3248] compression and row-group-size (and other properties)
    # should be taken from table properties
    # https://iceberg.apache.org/docs/latest/configuration/#write-properties
    # Using snappy and our row group size default for now
    compression = "snappy"
    rg_size = -1

    record_count = np.zeros(1, dtype=np.int64)
    file_size_in_bytes = np.zeros(1, dtype=np.int64)

    # Write the file using the parquet infra and get the metrics
    if not partition_spec and not sort_order:
        iceberg_pq_write_table_cpp(
            unicode_to_utf8(fname),
            unicode_to_utf8(table_loc),
            bodo_table,
            col_names,
            unicode_to_utf8(compression),
            is_parallel,
            unicode_to_utf8(bucket_region),
            rg_size,
            unicode_to_utf8(iceberg_schema_str),
            record_count.ctypes,
            file_size_in_bytes.ctypes,
        )  # type: ignore  Due to additional first argument typingctx
    else:
        raise Exception("Partition Spec and Sort Order not supported yet.")

    # Metrics to provide to Iceberg:
    # Required:
    # 1. record_count -- Number of records/rows in this file
    # 2. file_size_in_bytes -- Total file size in bytes

    # TODO [BE-3099]
    # Optional:
    # 3. column_sizes
    # 4. value_counts
    # 5. null_value_counts
    # 6. nan_value_counts
    # 7. distinct_counts
    # 8. lower_bounds
    # 9. upper_bounds

    # Collect the file names and meterics
    with numba.objmode(fnames="types.List(types.unicode_type)"):
        comm = MPI.COMM_WORLD
        fnames = comm.gather(fname)
        # We need to return a list of strings on all ranks,
        # so, we return dummy values on non rank 0
        if comm.Get_rank() != 0:
            fnames = ["a", "b"]

    record_counts = bodo.gatherv(record_count)
    file_sizes = bodo.gatherv(file_size_in_bytes)

    # Send file names, metrics and schema to Iceberg connector
    with numba.objmode(success="bool_"):
        success = register_table_write(
            conn,
            database_schema,
            table_name,
            table_loc,
            fnames,
            {"size": file_sizes.tolist(), "record_count": record_counts.tolist()},
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


import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types

if bodo.utils.utils.has_pyarrow():
    from bodo.io import arrow_cpp

    ll.add_symbol("iceberg_pq_write", arrow_cpp.iceberg_pq_write)


@intrinsic
def iceberg_pq_write_table_cpp(
    typingctx,
    fname_t,
    path_name_t,
    table_t,
    col_names_t,
    compression_t,
    is_parallel_t,
    bucket_region,
    row_group_size,
    iceberg_metadata_t,
    record_count_t,
    file_size_in_bytes_t,
):
    """
    Call C++ iceberg parquet write function
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="iceberg_pq_write"
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    return (
        types.void(
            types.voidptr,
            types.voidptr,
            table_t,
            col_names_t,
            types.voidptr,
            types.boolean,
            types.voidptr,
            types.int64,
            types.voidptr,
            types.voidptr,
            types.voidptr,
        ),
        codegen,
    )
