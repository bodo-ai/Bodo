"""
Operations for performing writes to Iceberg tables.
This file contains both code for
- The transaction handling (setup and teardown)
- Writing the Parquet files in the expected format
"""

import sys
import typing as pt
from copy import deepcopy

import llvmlite.binding as ll
import numba
import numpy as np
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import NativeValue, box, intrinsic, models, register_model, unbox

import bodo
import bodo.utils.tracing as tracing
from bodo.ext import s3_reader
from bodo.io import arrow_cpp
from bodo.io.fs_io import (
    ArrowFs,
    arrow_filesystem_del,
)
from bodo.io.helpers import (
    is_pyarrow_list_type,
    pyarrow_schema_type,
)
from bodo.io.iceberg.common import b_ICEBERG_FIELD_ID_MD_KEY, get_rest_catalog_config
from bodo.io.s3_fs import create_iceberg_aws_credentials_provider, create_s3_fs_instance
from bodo.libs.bool_arr_ext import alloc_false_bool_array
from bodo.libs.str_ext import unicode_to_utf8
from bodo.mpi4py import MPI
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.utils import BodoError, run_rank0

# ----------------------- Compiler Utils ----------------------- #
ll.add_symbol("iceberg_pq_write_py_entry", arrow_cpp.iceberg_pq_write_py_entry)

ll.add_symbol(
    "create_iceberg_aws_credentials_provider_py_entry",
    s3_reader.create_iceberg_aws_credentials_provider_py_entry,
)
ll.add_symbol(
    "destroy_iceberg_aws_credentials_provider_py_entry",
    s3_reader.destroy_iceberg_aws_credentials_provider_py_entry,
)


# TODO Use install_py_obj_class
class PythonListOfHeterogeneousTuples(types.Opaque):
    """
    It is just a Python object (list of tuples) to be passed to C++.
    Used for iceberg partition-spec, sort-order and iceberg-file-info
    descriptions.
    """

    def __init__(self):
        super().__init__(name="PythonListOfHeterogeneousTuples")


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


class ThetaSketchCollectionType(types.Type):
    """Type for C++ pointer to a collection of theta sketches"""

    def __init__(self):  # pragma: no cover
        super().__init__(name="ThetaSketchCollectionType(r)")


register_model(ThetaSketchCollectionType)(models.OpaqueModel)

theta_sketch_collection_type = ThetaSketchCollectionType()


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


# ----------------------- Helper Functions ----------------------- #


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

    new_field: pa.Field
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
    schema: pa.Schema, ref_schema: pa.Schema | None = None
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
    import bodo_iceberg_connector as bic

    assert (
        bodo.get_rank() == 0
    ), "bodo/io/iceberg.py::add_iceberg_field_id_md_to_pa_schema must be called from rank 0 only"

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
        return bic.get_schema_with_init_field_ids(intermediate_schema)
    else:
        new_fields = []
        for field in schema:
            ref_field = ref_schema.field(field.name)
            assert ref_field is not None, field
            # This ensures we select the correct subset of any structs.
            new_field = with_iceberg_field_id_md_from_ref_field(field, ref_field)
            new_fields.append(new_field)
        pyarrow_schema = pa.schema(new_fields)
        return bic.schema_helper.convert_arrow_schema_to_large_types(pyarrow_schema)


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

    import bodo_iceberg_connector as bic

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
            ) = bic.get_typing_info(conn, database_schema, table_name)
            already_exists = iceberg_schema_id is not None
            iceberg_schema_id = iceberg_schema_id if already_exists else -1

            if already_exists and if_exists == "fail":
                # Ideally we'd like to throw the same error as pandas
                # (https://github.com/pandas-dev/pandas/blob/4bfe3d07b4858144c219b9346329027024102ab6/pandas/io/sql.py#L833)
                # but using values not known at compile time, in Exceptions
                # doesn't seem to work with Numba
                raise ValueError("Table already exists.")

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
                iceberg_schema_str = bic.pyarrow_to_iceberg_schema_str(
                    output_pyarrow_schema
                )

        except bic.IcebergError as e:
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


def generate_data_file_info(
    iceberg_files_info: list[tuple[pt.Any, pt.Any, pt.Any]],
) -> tuple[list[str], list[int], list[dict[str, pt.Any]]]:
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

    def extract_and_gather(i: int) -> list[pt.Any] | None:
        """Extract field i from iceberg_files_info
        and gather the results on rank 0.

        Args:
            i (int): The field index

        Returns:
            pt.List[pt.Any]: The gathered result
        """
        values_local = [x[i] for x in iceberg_files_info]
        values_local_list: list = comm.gather(values_local)  # type: ignore
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

    assert (
        bodo.get_rank() == 0
    ), "bodo/io/iceberg.py::remove_transaction must be called from rank 0 only"

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

    assert (
        bodo.get_rank() == 0
    ), "bodo/io/iceberg.py::fetch_puffin_metadata must be called from rank 0 only"

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

    assert (
        bodo.get_rank() == 0
    ), "bodo/io/iceberg.py::commit_statistics_file must be called from rank 0 only"

    ev = tracing.Event("commit_statistics_file")
    bodo_iceberg_connector.commit_statistics_file(
        conn_str, db_name, table_name, snapshot_id, statistic_file_info
    )
    ev.finalize()


@run_rank0
def table_columns_have_theta_sketches(conn_str: str, db_name: str, table_name: str):
    import bodo_iceberg_connector

    assert (
        bodo.get_rank() == 0
    ), "bodo/io/iceberg.py::table_columns_have_theta_sketches must be called from rank 0 only"
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

    assert (
        bodo.get_rank() == 0
    ), "bodo/io/iceberg.py::table_columns_enabled_theta_sketches must be called from rank 0 only"
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

    assert (
        bodo.get_rank() == 0
    ), "bodo/io/iceberg.py::get_old_statistics_file_path must be called from rank 0 only"
    return bodo_iceberg_connector.get_old_statistics_file_path(
        txn_id, conn_str, db_name, table_name
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
    import bodo_iceberg_connector as bic

    assert (
        bodo.get_rank() == 0
    ), "bodo/io/iceberg.py::wrap_start_write must be called from rank 0 only"

    return bic.start_write(
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

    dummy_theta_sketch = (
        bodo.io.iceberg.stream_iceberg_write.init_theta_sketches_wrapper(
            alloc_false_bool_array(n_cols)
        )
    )
    bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(table_loc, is_parallel)
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

    bodo.io.iceberg.stream_iceberg_write.delete_theta_sketches(dummy_theta_sketch)

    ev.finalize()
