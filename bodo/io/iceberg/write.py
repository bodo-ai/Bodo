"""
Operations for performing writes to Iceberg tables.
This file contains both code for
- The transaction handling (setup and teardown)
- Writing the Parquet files in the expected format
"""

from __future__ import annotations

import sys
import typing as pt
from itertools import zip_longest

import llvmlite.binding as ll
import numba
import pyarrow as pa
import pyarrow.fs
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic

import bodo
import bodo.utils.tracing as tracing
from bodo.io import arrow_cpp
from bodo.io.fs_io import pyarrow_fs_type
from bodo.io.helpers import pyarrow_schema_type
from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.io.iceberg.common import _format_data_loc, _fs_from_file_path
from bodo.io.iceberg.theta import theta_sketch_collection_type
from bodo.libs.bool_arr_ext import alloc_false_bool_array
from bodo.libs.str_ext import unicode_to_utf8
from bodo.mpi4py import MPI
from bodo.utils.py_objs import install_opaque_class, install_py_obj_class
from bodo.utils.typing import EMPTY_CREATE_TABLE_META, CreateTableMetaType
from bodo.utils.utils import BodoError, run_rank0

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyarrow.fs import FileSystem
    from pyiceberg.catalog import Catalog
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.table import Table, Transaction
    from pyiceberg.table.sorting import SortOrder


# ----------------------- Compiler Utils ----------------------- #
ll.add_symbol("iceberg_pq_write_py_entry", arrow_cpp.iceberg_pq_write_py_entry)

try:
    from pyiceberg.catalog import Catalog
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table import Transaction
except ImportError:
    # PyIceberg is not installed
    Catalog = None
    PartitionSpec = None
    Transaction = None

this_module = sys.modules[__name__]
_, iceberg_catalog_type = install_py_obj_class(
    types_name="iceberg_catalog_type",
    module=this_module,
    python_type=Catalog,
    class_name="IcebergCatalogType",
    model_name="IcebergCatalogModel",
)
_, transaction_type = install_py_obj_class(
    types_name="transaction_type",
    module=this_module,
    python_type=Transaction,
    class_name="TransactionType",
    model_name="TransactionModel",
)
_, partition_spec_type = install_py_obj_class(
    types_name="partition_spec_type",
    module=this_module,
    python_type=PartitionSpec,
    class_name="PartitionSpecType",
    model_name="PartitionSpecModel",
)
_, python_list_of_heterogeneous_tuples_type = install_opaque_class(
    types_name="python_list_of_heterogeneous_tuples_type",
    module=this_module,
    class_name="PythonListOfHeterogeneousTuples",
)
_, dict_type = install_py_obj_class(
    types_name="dict_type",
    module=this_module,
    class_name="DictType",
    model_name="DictModel",
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
            pyarrow_fs_type,
            theta_sketch_collection_type,
        ),
        codegen,
    )


# ----------------------- Helper Functions ----------------------- #


def _get_write_data_path(properties: dict[str, str], location: str) -> str:
    """
    Get the path to write Parquet files to for an Iceberg table
    given the tables properties and the tables base location.
    """

    data_path = properties.get("write.data.path")
    return data_path if data_path else f"{location}/data"


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


def are_schemas_compatible(
    table_schema: pa.Schema, df_schema: pa.Schema, allow_downcasting: bool = False
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
    # Note that PyIceberg has a similar function
    # pyiceberg.schema._check_schema_compatible
    # but it does not support downcasting checks

    if table_schema.equals(df_schema):
        return True

    # If the schemas are not the same size, it is still possible that the DataFrame
    # can be appended iff the DataFrame schema is a subset of the iceberg schema and
    # each missing field is optional
    if len(df_schema) < len(table_schema):
        # Replace df_schema with a fully expanded schema tha contains the default
        # values for missing fields.
        kept_fields = []
        for pa_field in table_schema:
            df_field_index = df_schema.get_field_index(pa_field.name)
            if df_field_index != -1:
                kept_fields.append(df_schema.field(df_field_index))
            elif pa_field.nullable:
                # Append optional missing fields.
                kept_fields.append(pa_field)

        df_schema = pa.schema(kept_fields)

    if len(df_schema) != len(table_schema):
        return False

    # Compare each field individually for "compatibility"
    # Only the DataFrame schema is potentially modified during this step
    for idx in range(len(df_schema)):
        df_field = df_schema.field(idx)
        pa_field = table_schema.field(idx)
        new_field = _update_field(df_field, pa_field, allow_downcasting)
        df_schema = df_schema.set(idx, new_field)

    return df_schema.equals(table_schema)


def generate_data_file_info(
    iceberg_files_info: list[tuple[pt.Any, pt.Any, pt.Any]],
) -> tuple[
    list[str] | None,
    list[dict[str, pt.Any]] | None,
    list[tuple] | None,
]:
    """
    Collect C++ Iceberg File Info to a single rank
    and process before handing off to the connector / committing functions
    """
    comm = MPI.COMM_WORLD
    # Information we need:
    # 1. File names
    # 2. file_size_in_bytes
    # Metrics we provide to Iceberg:
    # 1. recordCount -- Number of rows in this file
    # 2. valueCounts -- Number of records per field id. This is most useful for
    #    nested data types where each row may have multiple records.
    # 3. nullValueCounts - Null count per field id.
    # 4. lowerBounds - Lower bounds per field id.
    # 5. upperBounds - Upper bounds per field id.

    combined_data: list[list[tuple]] | None = comm.gather(iceberg_files_info)
    # Flatten the list of lists
    file_infos = (
        [item for sub in combined_data for item in sub] if combined_data else None
    )

    fnames, file_records, partition_infos = None, None, None
    if file_infos:
        fnames, file_records, partition_infos = [], [], []
        for i in range(len(file_infos)):
            finfo = file_infos[i]
            fnames.append(finfo[0])
            file_records.append(
                {
                    "file_size_in_bytes": finfo[2],
                    "record_count": finfo[1],
                    "value_counts": finfo[3],
                    "null_value_counts": finfo[4],
                    "lower_bounds": finfo[5],
                    "upper_bounds": finfo[6],
                }
            )
            partition_infos.append(finfo[7:])

    return fnames, file_records, partition_infos


@run_rank0
def register_table_write(
    transaction: Transaction,
    fnames: list[str] | None,
    file_records: list[dict[str, pt.Any]] | None,
    partition_infos: list[tuple] | None,
    partition_spec: PartitionSpec,
    sort_order_id: int | None,
):
    """
    Commit the transaction with the given file information
    """
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.typedef import Record

    ev = tracing.Event("iceberg_register_table_write")
    if fnames is None:
        fnames = []
    if file_records is None:
        file_records = []
    if partition_infos is None:
        partition_infos = []

    with transaction.update_snapshot().fast_append() as add:
        for file_name, file_record, partition_info in zip(
            fnames, file_records, partition_infos
        ):
            data_file = DataFile(
                content=DataFileContent.DATA,
                file_format=FileFormat.PARQUET,
                file_path=transaction.table_metadata.location + "/data/" + file_name,
                # Partition and Sort Order
                partition=Record(
                    **{
                        field.name: info
                        for field, info in zip(partition_spec.fields, partition_info)
                    }
                ),
                sort_order_id=sort_order_id,
                spec_id=partition_spec.spec_id,
                equality_ids=None,
                key_metadata=None,
                column_sizes=None,
                nan_value_counts=None,
                split_offsets=None,
                # Additional Metrics
                **file_record,
            )
            add.append_data_file(data_file)

    # Commit the transaction
    try:
        transaction.commit_transaction()
        return True
    except Exception:
        return False
    finally:
        ev.finalize()


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
    properties,
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
    rg_size = -1
    with bodo.no_warning_objmode(compression="unicode_type"):
        compression = properties.get("write.parquet.compression-codec", "snappy")

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


def validate_append_target(
    df_schema: pa.Schema,
    table: Table,
    allow_downcasting: bool,
) -> None:
    """
    Validate that the DataFrame we are appending is compatible with the target table.
    In particular, ensure that the schemas are compatible and that columns
    for partitioning and sorting are present in the DataFrame.
    """

    df_col_names = set(df_schema.names)

    table_schema = table.schema()
    partition_spec = table.spec()
    sort_order = table.sort_order()

    # Ensure that all column names in the partition spec and sort order are
    # in the DataFrame being written
    for partition_field in partition_spec.fields:
        col_name = table_schema.find_field(partition_field.source_id).name
        assert col_name in df_col_names, (
            f"Iceberg Partition column {col_name} not found in dataframe"
        )
    for sort_field in sort_order.fields:
        col_name = table_schema.find_field(sort_field.source_id).name
        assert col_name in df_col_names, (
            f"Iceberg Sort column {col_name} not found in dataframe"
        )

    if not are_schemas_compatible(
        table_schema.as_arrow(), df_schema, allow_downcasting
    ):
        # TODO: https://bodo.atlassian.net/browse/BE-4019
        # for improving docs on Iceberg write support
        if numba.core.config.DEVELOPER_MODE:
            raise BodoError(
                f"DataFrame schema needs to be an ordered subset of Iceberg table for append\n\n"
                f"Iceberg:\n{table_schema}\n\n"
                f"DataFrame:\n{df_schema}\n"
            )
        else:
            raise BodoError(
                "DataFrame schema needs to be an ordered subset of Iceberg table for append"
            )


def build_partition_sort_tuples(
    schema: Schema,
    partition_spec: PartitionSpec,
    sort_order: SortOrder,
) -> tuple[list[tuple[int, str, int, str]], list[tuple[int, str, int, bool, bool]]]:
    """
    Convert PyIceberg PartitionSpec and SortOrder objects into
    primitive Python containers (list of tuples of primitive types)
    for easier passing and using in C++.
    """
    from pyiceberg.table.sorting import NullOrder, SortDirection

    iceberg_source_id_to_col_idx = {
        field.field_id: idx for idx, field in enumerate(schema.fields)
    }

    def parse_transform_str(transform: str) -> tuple[str, int]:
        if transform.startswith("truncate["):
            return "truncate", int(transform[(len("truncate") + 1) : -1])
        elif transform.startswith("bucket["):
            return "bucket", int(transform[(len("bucket") + 1) : -1])
        else:
            return transform, -1

    partition_tuples = [
        (
            iceberg_source_id_to_col_idx[field.source_id],
            *parse_transform_str(str(field.transform)),
            field.name,
        )
        for field in partition_spec.fields
    ]
    sort_tuples = [
        (
            iceberg_source_id_to_col_idx[field.source_id],
            *parse_transform_str(str(field.transform)),
            field.direction == SortDirection.ASC,
            field.null_order == NullOrder.NULLS_LAST,
        )
        for field in sort_order.fields
    ]
    return partition_tuples, sort_tuples


def list_field_names(
    df_schema: pa.StructType | pa.Schema, prefix: str = ""
) -> pt.Generator[str]:
    """
    Iterate over all field names in a PyArrow schema, including nested fields
    inside of structs and the elements of lists. Note that we don't
    need to output the outer struct or list field name cause PyIceberg
    will auto-include it if any nested fields are used.
    """
    for field in df_schema:
        if pa.types.is_struct(field.type):
            yield from list_field_names(field.type, prefix + field.name + ".")
        elif pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            yield prefix + field.name + ".element"
        elif pa.types.is_map(field.type):
            yield prefix + field.name + ".key"
            yield prefix + field.name + ".value"
        else:
            yield prefix + field.name


def _replace_schema(txn: Transaction, schema: Schema):
    """
    PyIceberg doesn't have a way to replace a schema for REPLACE TABLE
    So this function duplicates that. Given an Iceberg schema and transaction,
    it will remove all fields and add the new schema.
    """
    with txn.update_schema(True) as update:
        for field in txn.table_metadata.schema().fields:
            update._deletes.add(field.field_id)
        for field in schema.fields:
            update.add_column(field.name, field.field_type, field.doc, field.required)


def start_write_rank_0(
    conn: str,
    table_id: str,
    df_schema: pa.Schema,
    if_exists: pt.Literal["fail", "append", "replace"],
    allow_downcasting: bool,
    create_table_info_arg: CreateTableMetaType | None = None,
) -> tuple[
    Transaction,
    FileSystem,
    str,
    pa.Schema,
    str,
    PartitionSpec,
    list,
    int,
    list,
    dict[str, str],
]:
    """
    Args:
        catalog: Iceberg catalog to construct table in
        table_id (str): Period-delimited table identifier for PyIceberg
        df_schema (pyarrow.Schema): PyArrow schema of the DataFrame being written
        if_exists (str): What write operation we are doing. This must be one of
            ['fail', 'append', 'replace']
    """
    from pyiceberg.io import load_file_io
    from pyiceberg.io.pyarrow import _pyarrow_to_schema_without_ids
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
    from pyiceberg.schema import assign_fresh_schema_ids, prune_columns
    from pyiceberg.table import ALWAYS_TRUE
    from pyiceberg.table.sorting import UNSORTED_SORT_ORDER
    from pyiceberg.typedef import EMPTY_DICT

    assert bodo.get_rank() == 0, (
        "bodo.io.iceberg.write.start_write_rank_0:: This function must only run on rank 0"
    )

    catalog = conn_str_to_catalog(conn)
    # Determine what action to perform based on if_exists and table status
    table_exists = catalog.table_exists(table_id)
    if table_exists and if_exists == "fail":
        raise ValueError("Iceberg table already exists.")
    elif not table_exists:
        mode = "create"
    elif if_exists == "replace":
        mode = "replace"
    else:
        mode = "append"

    create_table_info = create_table_info_arg or EMPTY_CREATE_TABLE_META
    properties = dict(create_table_info.table_properties or EMPTY_DICT)
    if create_table_info.table_comment is not None:
        properties["comment"] = create_table_info.table_comment

    if mode == "create":
        output_schema = assign_fresh_schema_ids(
            _pyarrow_to_schema_without_ids(df_schema)
        )
        output_schema = output_schema.model_copy(
            update={
                "fields": tuple(
                    field.model_copy(update={"doc": comment})
                    for field, comment in zip_longest(
                        output_schema.fields, create_table_info.column_comments or []
                    )
                )
            }
        )

        try:
            txn = catalog.create_table_transaction(
                table_id,
                output_schema,
                properties=properties,
            )
        except NotImplementedError:
            # Catalog doesn't support Create Table Transaction
            # Create the table without a transaction and write under transaction
            txn = catalog.create_table(
                table_id,
                output_schema,
                properties=properties,
            ).transaction()

        data_loc = _get_write_data_path(
            txn.table_metadata.properties, txn.table_metadata.location
        )
        io = load_file_io(
            {**catalog.properties, **txn._table.io.properties},
            txn.table_metadata.location,
        )
        # The default created transaction io doesn't do correct file path resolution
        # for table create transactions because metadata_location is None
        txn._table.io = io
        # Empty Partition Spec and Sort Order
        partition_spec = UNPARTITIONED_PARTITION_SPEC
        sort_order = UNSORTED_SORT_ORDER
        properties = txn.table_metadata.properties

    elif mode == "replace":
        output_schema = _pyarrow_to_schema_without_ids(df_schema)
        output_schema = output_schema.model_copy(
            update={
                "fields": tuple(
                    field.model_copy(update={"doc": comment})
                    for field, comment in zip_longest(
                        output_schema.fields, create_table_info.column_comments or []
                    )
                )
            }
        )

        table = catalog.load_table(table_id)
        txn = table.transaction()
        txn.delete(ALWAYS_TRUE)
        txn.set_properties(properties)
        if assign_fresh_schema_ids(output_schema) != assign_fresh_schema_ids(
            table.schema()
        ):
            _replace_schema(txn, output_schema)
        output_schema = txn.table_metadata.schema()

        io = table.io
        data_loc = _get_write_data_path(properties, table.location())
        # Empty Partition Spec and Sort Order
        partition_spec = UNPARTITIONED_PARTITION_SPEC
        sort_order = UNSORTED_SORT_ORDER

    else:
        assert mode == "append"
        table = catalog.load_table(table_id)
        txn = table.transaction()
        io = table.io
        data_loc = _get_write_data_path(table.properties, table.location())

        # Check the input and table metadata are compatible
        validate_append_target(df_schema, table, allow_downcasting)
        # Construct the output schema
        table_schema = table.schema()
        field_ids = {
            table_schema._name_to_id[name] for name in list_field_names(df_schema)
        }
        output_schema = prune_columns(table_schema, field_ids, False)
        # Extract Partition Spec and Sort Order from Metadata
        partition_spec = table.spec()
        sort_order = table.sort_order()
        properties = table.properties

    # Convert Iceberg schema to primitive types
    iceberg_schema_str = output_schema.model_dump_json()
    output_pa_schema = output_schema.as_arrow()
    partition_tuple, sort_tuple = build_partition_sort_tuples(
        output_schema, partition_spec, sort_order
    )
    fs = _fs_from_file_path(data_loc, io)

    data_loc = _format_data_loc(data_loc, fs)

    return (
        txn,
        fs,
        data_loc,
        output_pa_schema,
        iceberg_schema_str,
        partition_spec,
        partition_tuple,
        sort_order.order_id,
        sort_tuple,
        properties,
    )


def wrap_start_write(
    conn: str,
    table_id: str,
    df_schema: pa.Schema,
    if_exists: pt.Literal["fail", "append", "replace"],
    allow_downcasting: bool,
    create_table_info_arg: CreateTableMetaType | None = None,
):
    comm = MPI.COMM_WORLD

    txn = None
    result = ()
    err: Exception | None = None
    if comm.Get_rank() == 0:
        try:
            txn, *result = start_write_rank_0(
                conn,
                table_id,
                df_schema,
                if_exists,
                allow_downcasting,
                create_table_info_arg,
            )
        except Exception as e:
            err = e

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err

    result = comm.bcast(result)
    return txn, *result


@numba.njit
def iceberg_write(
    conn,
    table_id,
    bodo_table,
    col_names,
    # Same semantics as pandas to_sql for now
    if_exists,
    is_parallel,
    df_pyarrow_schema,  # Additional Param to Compare Compile-Time and Iceberg Schema
    allow_downcasting,
):  # pragma: no cover
    """
    Iceberg Basic Write Implementation for parquet based tables.
    Args:
        conn (str): connection string
        table_id (str): Table Identifier of the iceberg database
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
        txn="transaction_type",
        fs="pyarrow_fs_type",
        data_loc="unicode_type",
        output_schema="pyarrow_schema_type",
        iceberg_schema_str="unicode_type",
        partition_spec="partition_spec_type",
        partition_tuples="python_list_of_heterogeneous_tuples_type",
        sort_order_id="i8",
        sort_tuples="python_list_of_heterogeneous_tuples_type",
        num_cols="i8",
        properties=dict_type,
    ):
        (
            txn,
            fs,
            data_loc,
            output_schema,
            iceberg_schema_str,
            partition_spec,
            partition_tuples,
            sort_order_id,
            sort_tuples,
            properties,
        ) = wrap_start_write(
            conn,
            table_id,
            df_pyarrow_schema,
            if_exists,
            allow_downcasting,
        )
        num_cols = len(df_pyarrow_schema)

    dummy_theta_sketch = bodo.io.iceberg.theta.init_theta_sketches_wrapper(
        alloc_false_bool_array(num_cols)
    )
    bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(data_loc, is_parallel)
    iceberg_files_info = iceberg_pq_write(
        data_loc,
        bodo_table,
        col_names,
        partition_tuples,
        sort_tuples,
        iceberg_schema_str,
        is_parallel,
        output_schema,
        fs,
        dummy_theta_sketch,
        bucket_region,
        properties,
    )

    with bodo.no_warning_objmode(success="bool_"):
        fnames, file_records, partition_infos = generate_data_file_info(
            iceberg_files_info
        )
        success = True
        # Send file names, metrics and schema to Iceberg connector
        success = register_table_write(
            txn,
            fnames,
            file_records,
            partition_infos,
            partition_spec,
            None if sort_order_id == 0 else sort_order_id,
        )

    if not success:
        # TODO [BE-3249] If it fails due to schema changing, then delete the files.
        # Note that this might not always be possible since
        # we might not have DeleteObject permissions, for instance.
        raise BodoError("Iceberg write failed.")

    bodo.io.iceberg.theta.delete_theta_sketches(dummy_theta_sketch)
    ev.finalize()
