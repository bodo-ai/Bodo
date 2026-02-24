"""
Operator for streaming Iceberg writer. Currently used by BodoSQL only.
Not intended for external use
"""

from __future__ import annotations

import operator
import os
import traceback
import typing as pt

import numba
import pandas as pd
import pyarrow as pa
from mpi4py import MPI
from numba.core import types
from numba.core.imputils import impl_ret_borrowed
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.extending import (
    box,
    intrinsic,
    lower_builtin,
    models,
    overload,
    register_model,
    unbox,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import TableType
from bodo.io.helpers import (
    _get_stream_writer_payload,
    pyarrow_fs_type,
    pyarrow_schema_type,
    stream_writer_alloc_codegen,
)
from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.io.iceberg.theta import (
    _write_puffin_file,
    commit_statistics_file,
    delete_theta_sketches,
    fetch_puffin_metadata,
    get_default_theta_sketch_columns,
    get_old_statistics_file_path,
    get_supported_theta_sketch_columns,
    init_theta_sketches_wrapper,
    table_columns_enabled_theta_sketches,
    table_columns_have_theta_sketches,
    theta_sketch_collection_type,
)
from bodo.io.iceberg.write import (
    dict_type,
    generate_data_file_info,
    iceberg_pq_write,
    partition_spec_type,
    python_list_of_heterogeneous_tuples_type,
    register_table_write,
    transaction_type,
    wrap_start_write,
)
from bodo.libs.array import (
    array_to_info,
    py_table_to_cpp_table,
)
from bodo.libs.bool_arr_ext import alloc_false_bool_array
from bodo.libs.str_ext import unicode_to_utf8
from bodo.libs.streaming.base import StreamingStateType
from bodo.libs.table_builder import TableBuilderStateType
from bodo.utils import tracing
from bodo.utils.typing import (
    BodoError,
    ColNamesMetaType,
    get_overload_const_str,
    is_overload_bool,
    is_overload_constant_str,
    is_overload_none,
    unwrap_typeref,
)

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyarrow.fs import FileSystem
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table import Transaction


from bodo.io.iceberg import ICEBERG_WRITE_PARQUET_CHUNK_SIZE


def get_enable_theta() -> bool:  # type: ignore
    pass


@overload(get_enable_theta)
def overload_get_enable_theta():
    """
    Returns whether theta sketches are currently enabled.
    """

    def impl():  # pragma: no cover
        with bodo.ir.object_mode.no_warning_objmode(ret_var="bool_"):
            ret_var = bodo.enable_theta_sketches
        return ret_var

    return impl


class IcebergWriterType(StreamingStateType):
    """Data type for streaming Iceberg writer's internal state"""

    def __init__(self, input_table_type=types.unknown):
        self.input_table_type = input_table_type
        super().__init__(name=f"IcebergWriterType({input_table_type})")

    def is_precise(self):
        return self.input_table_type != types.unknown

    def unify(self, typingctx, other):
        """Unify two IcebergWriterType instances when one doesn't have a resolved
        input_table_type.
        """
        if isinstance(other, IcebergWriterType):
            if not other.is_precise() and self.is_precise():
                return self

            # Prefer the new type in case write append changed its table type
            return other


class IcebergWriterPayloadType(types.Type):
    """Data type for streaming Iceberg writer's payload"""

    def __init__(self):
        super().__init__(name="IcebergWriterPayloadType")


iceberg_writer_payload_type = IcebergWriterPayloadType()


iceberg_writer_payload_members = (
    # Connection String to Iceberg Table
    ("conn", types.unicode_type),
    # Table identifier to write to
    ("table_id", types.unicode_type),
    # Action Iceberg takes if table exists. Currently one of "fail", "append", "replace"
    ("if_exists", types.unicode_type),
    # Location of the data/ folder in the warehouse
    ("table_loc", types.unicode_type),
    # JSON Encoding of Iceberg Schema to include in Parquet metadata
    ("iceberg_schema_str", types.unicode_type),
    # Output pyarrow schema that should be written to the Iceberg table.
    # This also contains the Iceberg field IDs in the fields' metadata
    # which is important during the commit step.
    ("output_pyarrow_schema", pyarrow_schema_type),
    # Partition Spec Object for Iceberg Table
    ("partition_spec", partition_spec_type),
    # Array of Tuples containing Partition Spec for Iceberg Table (passed to C++)
    ("partition_tuples", python_list_of_heterogeneous_tuples_type),
    # Sort Order ID for Iceberg Table
    ("sort_order_id", types.int64),
    # Array of Tuples containing Sort Order for Iceberg Table (passed to C++)
    ("sort_tuples", python_list_of_heterogeneous_tuples_type),
    # Properties for Iceberg Table
    ("properties", dict_type),
    # List of written file infos needed by Iceberg for committing
    ("iceberg_files_info", python_list_of_heterogeneous_tuples_type),
    # Whether write is occurring in parallel
    ("parallel", types.boolean),
    # Non-blocking is_last sync state (communicator, request, flags, ...)
    ("is_last_state", bodo.libs.distributed_api.is_last_state_type),
    # Whether this rank has finished appending data to the table
    ("finished", types.boolean),
    # Batches collected to write
    ("batches", TableBuilderStateType()),
    # Property that encapsulates both if theta sketches are enabled
    # for this table and if we have any theta sketches to write.
    ("use_theta_sketches", types.boolean),
    # Collection of theta sketch data for the columns that have it
    ("theta_sketches", theta_sketch_collection_type),
    # Transaction ID for the write
    ("txn", transaction_type),
    # Arrow filesystem for Iceberg FS
    ("arrow_fs", pyarrow_fs_type),
    # location of the s3 bucket
    ("bucket_region", types.unicode_type),
    # Max Parquet file chunk size
    ("max_pq_chunksize", types.int64),
)
iceberg_writer_payload_members_dict = dict(iceberg_writer_payload_members)


@register_model(IcebergWriterPayloadType)
class IcebergWriterPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):  # pragma: no cover
        members = iceberg_writer_payload_members
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(IcebergWriterType)
class IcebergWriterModel(models.StructModel):
    def __init__(self, dmm, fe_type):  # pragma: no cover
        payload_type = iceberg_writer_payload_type
        members = [
            ("meminfo", types.MemInfoPointer(payload_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@intrinsic
def iceberg_writer_alloc(typingctx, expected_state_type_t):  # pragma: no cover
    expected_state_type = unwrap_typeref(expected_state_type_t)
    if is_overload_none(expected_state_type):
        iceberg_writer_type = IcebergWriterType()
    else:
        iceberg_writer_type = expected_state_type

    def codegen(context, builder, sig, args):  # pragma: no cover
        """Creates meminfo and sets dtor for Iceberg writer"""
        return stream_writer_alloc_codegen(
            context,
            builder,
            iceberg_writer_payload_type,
            iceberg_writer_type,
            iceberg_writer_payload_members,
        )

    return iceberg_writer_type(expected_state_type_t), codegen


@intrinsic(prefer_literal=True)
def iceberg_writer_getattr(typingctx, writer_typ, attr_typ):  # pragma: no cover
    """Get attribute of a Iceberg writer"""
    assert isinstance(writer_typ, IcebergWriterType), (
        f"iceberg_writer_getattr: expected `writer` to be a IcebergWriterType, "
        f"but found {writer_typ}"
    )
    assert is_overload_constant_str(attr_typ), (
        f"iceberg_writer_getattr: expected `attr` to be a literal string type, "
        f"but found {attr_typ}"
    )
    attr = get_overload_const_str(attr_typ)
    val_typ = iceberg_writer_payload_members_dict[attr]
    if attr == "batches":
        val_typ = TableBuilderStateType(writer_typ.input_table_type)

    def codegen(context, builder, sig, args):  # pragma: no cover
        writer, _ = args
        payload, _ = _get_stream_writer_payload(
            context, builder, writer_typ, iceberg_writer_payload_type, writer
        )
        return impl_ret_borrowed(
            context, builder, sig.return_type, getattr(payload, attr)
        )

    return val_typ(writer_typ, attr_typ), codegen


@intrinsic(prefer_literal=True)
def iceberg_writer_setattr(
    typingctx, writer_typ, attr_typ, val_typ
):  # pragma: no cover
    """Set attribute of a Iceberg writer"""
    assert isinstance(writer_typ, IcebergWriterType), (
        f"iceberg_writer_setattr: expected `writer` to be a IcebergWriterType, "
        f"but found {writer_typ}"
    )
    assert is_overload_constant_str(attr_typ), (
        f"iceberg_writer_setattr: expected `attr` to be a literal string type, "
        f"but found {attr_typ}"
    )
    attr = get_overload_const_str(attr_typ)

    # Storing a literal type into the payload causes a type mismatch
    val_typ = numba.types.unliteral(val_typ)

    def codegen(context, builder, sig, args):  # pragma: no cover
        writer, _, val = args
        payload, meminfo_data_ptr = _get_stream_writer_payload(
            context, builder, writer_typ, iceberg_writer_payload_type, writer
        )
        context.nrt.decref(builder, val_typ, getattr(payload, attr))
        context.nrt.incref(builder, val_typ, val)
        setattr(payload, attr, val)
        builder.store(payload._getvalue(), meminfo_data_ptr)
        return context.get_dummy_value()

    return types.none(writer_typ, attr_typ, val_typ), codegen


@overload(operator.getitem, no_unliteral=True)
def iceberg_writer_getitem(writer, attr):
    if not isinstance(writer, IcebergWriterType):
        return

    return lambda writer, attr: iceberg_writer_getattr(writer, attr)  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def iceberg_writer_setitem(writer, attr, val):
    if not isinstance(writer, IcebergWriterType):
        return

    return lambda writer, attr, val: iceberg_writer_setattr(
        writer, attr, val
    )  # pragma: no cover


@box(IcebergWriterType)
def box_iceberg_writer(typ, val, c):
    # Boxing is disabled, to avoid boxing overheads anytime a writer attribute
    # is accessed from objmode. As a workaround, store the necessary attributes
    # into local variables in numba native code before entering objmode
    raise NotImplementedError(
        "Boxing is disabled for IcebergWriter mutable struct."
    )  # pragma: no cover


@unbox(IcebergWriterType)
def unbox_iceberg_writer(typ, val, c):
    raise NotImplementedError(
        "Unboxing is disabled for IcebergWriter mutable struct."
    )  # pragma: no cover


def conn_wrapper_to_str(conn_wrapper) -> str:  # pragma: no cover
    pass


@overload(conn_wrapper_to_str)
def overload_conn_wrapper_to_str(conn_wrapper):
    """Convert a connection wrapper to a string"""
    from bodo.ir.iceberg_ext import IcebergConnectionType

    if isinstance(conn_wrapper, IcebergConnectionType):

        def impl(conn_wrapper):
            return conn_wrapper.conn_str

        return impl

    assert conn_wrapper == types.unicode_type

    def impl(conn_wrapper):
        return conn_wrapper

    return impl


def start_write_wrapper(
    conn,
    table_id,
    if_exists,
    df_schema,
    create_table_info,
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
]:  # type: ignore
    pass


@overload(start_write_wrapper)
def overload_start_write_wrapper(
    conn,
    table_id,
    if_exists,
    df_schema,
    create_table_info,
):
    """Wrapper around objmode call to wrap_start_write to avoid Numba compiler errors"""

    def impl(
        conn,
        table_id,
        if_exists,
        df_schema,
        create_table_info,
    ):  # pragma: no cover
        with bodo.ir.object_mode.no_warning_objmode(
            txn=transaction_type,
            fs="pyarrow_fs_type",
            data_loc="unicode_type",
            output_schema="pyarrow_schema_type",
            iceberg_schema_str="unicode_type",
            partition_spec="partition_spec_type",
            partition_tuples="python_list_of_heterogeneous_tuples_type",
            sort_order_id="i8",
            sort_tuples="python_list_of_heterogeneous_tuples_type",
            properties="dict_type",
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
                df_schema,
                if_exists,
                False,  # allow_downcasting
                create_table_info,
            )
        return (
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
        )

    return impl


def get_empty_pylist():  # pragma: no cover
    pass


@overload(get_empty_pylist)
def overload_get_empty_pylist():
    """Return an empty Python list object"""

    def impl():  # pragma: no cover
        with bodo.ir.object_mode.no_warning_objmode(
            a="python_list_of_heterogeneous_tuples_type"
        ):
            a = []
        return a

    return impl


@numba.njit
def get_table_target_file_size_bytes(properties):
    """Get's the 'write.target_file_size_bytes' property from a table. If the property is
    not found or there is an issue with reading the property, a default value is used.

    Args:
        properties (dict[str, str]): The properties of the table
    Returns:
        int: The value of 'write.target_file_size_bytes'
    """
    with bodo.ir.object_mode.no_warning_objmode(output="i8"):
        output = properties.get(
            "write.target-file-size-bytes", ICEBERG_WRITE_PARQUET_CHUNK_SIZE
        )
    return output


def iceberg_writer_init(
    operator_id,
    conn,
    table_id,
    col_names_meta,
    if_exists,
    create_table_meta=None,
    allow_theta_sketches=False,
    input_dicts_unified=False,
    _is_parallel=False,
):  # pragma: no cover
    pass


def gen_iceberg_writer_init_impl(
    iceberg_writer_type,
    operator_id,
    conn,
    table_id,
    col_names_meta,
    if_exists,
    create_table_meta=None,
    allow_theta_sketches=False,
    input_dicts_unified=False,
    _is_parallel=False,
):  # pragma: no cover
    """Initialize Iceberg stream writer"""
    from bodo.hiframes.pd_dataframe_ext import DataFrameType

    col_names_meta = unwrap_typeref(col_names_meta)
    col_names = col_names_meta.meta

    create_table_info = None
    if not is_overload_none(create_table_meta):
        create_table_info = unwrap_typeref(create_table_meta)
        if not isinstance(
            create_table_info, bodo.utils.typing.CreateTableMetaType
        ):  # pragma: no cover
            raise BodoError(
                f"iceberg_writer_init: Expected type CreateTableMetaType "
                f"for `create_table_meta`, found {create_table_info}"
            )

    table_builder_state_type = TableBuilderStateType(
        iceberg_writer_type.input_table_type
    )

    input_df_type = DataFrameType(
        iceberg_writer_type.input_table_type.arr_types, None, col_names
    )
    df_pyarrow_schema = bodo.io.helpers.numba_to_pyarrow_schema(
        input_df_type, is_iceberg=True
    )
    _n_cols = len(df_pyarrow_schema)

    def impl_iceberg_writer_init(
        operator_id,
        conn,
        table_id,
        col_names_meta,
        if_exists,
        create_table_meta=None,
        allow_theta_sketches=False,
        input_dicts_unified=False,
        _is_parallel=False,
    ):
        ev = tracing.Event("iceberg_writer_init", is_parallel=_is_parallel)
        conn_str = conn_wrapper_to_str(conn)
        (
            txn,
            fs,
            table_loc,
            output_pa_schema,
            iceberg_schema_str,
            partition_spec,
            partition_tuples,
            sort_order_id,
            sort_tuples,
            properties,
        ) = start_write_wrapper(
            conn_str,
            table_id,
            if_exists,
            df_pyarrow_schema,
            create_table_meta,
        )

        # Initialize writer
        writer = iceberg_writer_alloc(iceberg_writer_type)
        writer["conn"] = conn_str
        writer["table_id"] = table_id
        writer["if_exists"] = if_exists
        writer["table_loc"] = table_loc
        writer["iceberg_schema_str"] = iceberg_schema_str
        writer["output_pyarrow_schema"] = output_pa_schema
        writer["partition_spec"] = partition_spec
        writer["partition_tuples"] = partition_tuples
        writer["sort_order_id"] = sort_order_id
        writer["sort_tuples"] = sort_tuples
        writer["properties"] = properties
        writer["iceberg_files_info"] = get_empty_pylist()
        writer["parallel"] = _is_parallel
        writer["finished"] = False
        writer["txn"] = txn
        writer["arrow_fs"] = fs
        writer["is_last_state"] = bodo.libs.distributed_api.init_is_last_state()
        writer["batches"] = bodo.libs.table_builder.init_table_builder_state(
            operator_id,
            table_builder_state_type,
            input_dicts_unified=input_dicts_unified,
        )
        writer["bucket_region"] = bodo.io.fs_io.get_s3_bucket_region_wrapper(
            table_loc, _is_parallel
        )

        # Since streaming write is used only for SQL, replicated in this
        # context means actually replicated data (instead of independent sequential
        # functions with different data).
        writer["max_pq_chunksize"] = get_table_target_file_size_bytes(properties)
        allow_theta = allow_theta_sketches and get_enable_theta()
        if allow_theta:
            if if_exists == "append":
                # For insert into the columns are an intersection of the existing sketches
                # and the types we can support.
                existing_columns = table_columns_have_theta_sketches_wrapper(
                    writer["txn"]
                )
                possible_columns = get_supported_theta_sketch_columns(
                    writer["output_pyarrow_schema"]
                )
                theta_columns = existing_columns & possible_columns
            else:
                theta_columns = get_default_theta_sketch_columns(
                    writer["output_pyarrow_schema"]
                )

            # Find the columns that are enabled for theta sketches
            enabled_columns = table_columns_enabled_theta_sketches_wrapper(
                writer["txn"]
            )
            # The final enabled theta sketches are an intersection of the columns
            # the have the property set and the supported columns by Bodo.
            theta_columns = theta_columns & enabled_columns
        else:
            theta_columns = alloc_false_bool_array(_n_cols)
        use_theta_sketches = allow_theta and theta_columns.any()
        writer["use_theta_sketches"] = use_theta_sketches
        writer["theta_sketches"] = init_theta_sketches_wrapper(theta_columns)

        # Barrier ensures that internal stage exists before we upload files to it
        bodo.barrier()
        ev.finalize()
        return writer

    return impl_iceberg_writer_init


def table_columns_have_theta_sketches_wrapper(txn):  # pragma: no cover
    pass


@overload(table_columns_have_theta_sketches_wrapper)
def overload_table_columns_have_theta_sketches_wrapper(txn):
    """Check if the columns in the table have theta sketches enabled. This extra
    wrapper is added to avoid calling into objmode inside control flow."""
    _output_type = bodo.types.boolean_array_type

    def impl(txn):  # pragma: no cover
        with bodo.ir.object_mode.no_warning_objmode(existing_columns=_output_type):
            existing_columns = table_columns_have_theta_sketches(txn.table_metadata)
        return existing_columns

    return impl


def table_columns_enabled_theta_sketches_wrapper(txn):  # pragma: no cover
    pass


@overload(table_columns_enabled_theta_sketches_wrapper)
def overload_table_columns_enabled_theta_sketches_wrapper(txn):
    """Check if the columns in the table have theta sketches enabled. This extra
    wrapper is added to avoid calling into objmode inside control flow."""
    _output_type = bodo.types.boolean_array_type

    def impl(txn):  # pragma: no cover
        with bodo.ir.object_mode.no_warning_objmode(enabled_columns=_output_type):
            enabled_columns = table_columns_enabled_theta_sketches(txn)
        return enabled_columns

    return impl


@infer_global(iceberg_writer_init)
class IcebergWriterInitInfer(AbstractTemplate):
    """Typer for iceberg_writer_init that returns writer type"""

    def generic(self, args, kws):
        iceberg_writer_type = IcebergWriterType()
        pysig = numba.core.utils.pysignature(iceberg_writer_init)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(iceberg_writer_type, *folded_args).replace(pysig=pysig)


IcebergWriterInitInfer._no_unliteral = True


@lower_builtin(iceberg_writer_init, types.VarArg(types.Any))
def lower_iceberg_writer_init(context, builder, sig, args):
    """lower iceberg_writer_init() using gen_iceberg_writer_init_impl above"""
    impl = gen_iceberg_writer_init_impl(sig.return_type, *sig.args)
    return context.compile_internal(builder, impl, sig, args)


def append_py_list(pylist, to_append):  # pragma: no cover
    pass


@overload(append_py_list)
def overload_append_py_list(pylist, to_append):
    """Append a Python list object to existing Python list object"""

    def impl(pylist, to_append):  # pragma: no cover
        with bodo.ir.object_mode.no_warning_objmode:
            pylist.extend(to_append)

    return impl


def iceberg_writer_append_table_inner(
    writer, table, col_names_meta, local_is_last, iter
):  # pragma: no cover
    pass


@overload(iceberg_writer_append_table_inner)
def gen_iceberg_writer_append_table_impl_inner(
    writer,
    table,
    col_names_meta,
    local_is_last,
    iter,
):  # pragma: no cover
    if not isinstance(writer, IcebergWriterType):  # pragma: no cover
        raise BodoError(
            f"iceberg_writer_append_table: Expected type IcebergWriterType "
            f"for `writer`, found {writer}"
        )
    if not isinstance(table, TableType):  # pragma: no cover
        raise BodoError(
            f"iceberg_writer_append_table: Expected type TableType "
            f"for `table`, found {table}"
        )
    if not is_overload_bool(local_is_last):  # pragma: no cover
        raise BodoError(
            f"iceberg_writer_append_table: Expected type boolean "
            f"for `local_is_last`, found {local_is_last}"
        )

    col_names_meta = unwrap_typeref(col_names_meta)
    if not isinstance(col_names_meta, ColNamesMetaType):  # pragma: no cover
        raise BodoError(
            f"iceberg_writer_append_table: Expected type ColNamesMetaType "
            f"for `col_names_meta`, found {col_names_meta}"
        )
    if not isinstance(col_names_meta.meta, tuple):  # pragma: no cover
        raise BodoError(
            "iceberg_writer_append_table: Expected col_names_meta "
            "to contain a tuple of column names"
        )

    py_table_typ = table
    col_names_arr = pd.array(col_names_meta.meta)

    def impl_iceberg_writer_append_table(
        writer, table, col_names_meta, local_is_last, iter
    ):  # pragma: no cover
        if writer["finished"]:
            return True
        ev = tracing.Event(
            "iceberg_writer_append_table", is_parallel=writer["parallel"]
        )

        # ===== Part 1: Accumulate batch in writer and compute total size
        ev_append_batch = tracing.Event("append_batch", is_parallel=True)
        table_builder_state = writer["batches"]
        bodo.libs.table_builder.table_builder_append(table_builder_state, table)
        table_bytes = bodo.libs.table_builder.table_builder_nbytes(table_builder_state)
        ev_append_batch.add_attribute("table_bytes", table_bytes)
        ev_append_batch.finalize()
        bucket_region = writer["bucket_region"]

        is_last = bodo.libs.distributed_api.sync_is_last_non_blocking(
            writer["is_last_state"], local_is_last
        )

        # ===== Part 2: Write Parquet file if file size threshold is exceeded
        if is_last or table_bytes >= writer["max_pq_chunksize"]:
            # Note: Our write batches are at least as large as our read batches. It may
            # be advantageous in the future to split up large incoming batches into
            # multiple Parquet files to write.

            # NOTE: table_builder_reset() below affects the table builder state so
            # out_table should be used immediately and not be stored.
            out_table = bodo.libs.table_builder.table_builder_get_data(
                table_builder_state
            )
            out_table_len = len(out_table)

            # Write only on rank 0 for replicated input. Since streaming write is used
            # only for SQL, replicated in this context means actually replicated data
            # (instead of independent sequential functions with different data).
            if out_table_len > 0 and (writer["parallel"] or bodo.get_rank() == 0):
                ev_upload_table = tracing.Event("upload_table", is_parallel=False)

                table_info = py_table_to_cpp_table(out_table, py_table_typ)
                col_names_info = array_to_info(col_names_arr)
                iceberg_files_info = iceberg_pq_write(
                    writer["table_loc"],
                    table_info,
                    col_names_info,
                    writer["partition_tuples"],
                    writer["sort_tuples"],
                    writer["iceberg_schema_str"],
                    # Don't pass parallel=True because streaming is not synchronized.
                    False,
                    writer["output_pyarrow_schema"],
                    writer["arrow_fs"],
                    writer["theta_sketches"],
                    bucket_region,
                    writer["properties"],
                )
                append_py_list(writer["iceberg_files_info"], iceberg_files_info)

                ev_upload_table.finalize()
            bodo.libs.table_builder.table_builder_reset(table_builder_state)

        # ===== Part 3: Commit Iceberg write
        if is_last:
            if_exists = writer["if_exists"]
            all_iceberg_files_infos = writer["iceberg_files_info"]
            txn = writer["txn"]
            partition_spec = writer["partition_spec"]
            sort_order_id = writer["sort_order_id"]

            # Fetch any existing puffin files:
            use_theta_sketches = writer["use_theta_sketches"]
            with bodo.ir.object_mode.no_warning_objmode(
                old_puffin_file_path="unicode_type"
            ):
                if use_theta_sketches and if_exists == "append":
                    old_puffin_file_path = get_old_statistics_file_path(txn)
                else:
                    old_puffin_file_path = ""

            with bodo.ir.object_mode.no_warning_objmode(success="bool_"):
                (
                    fnames,
                    file_records,
                    partition_infos,
                ) = generate_data_file_info(all_iceberg_files_infos)

                # Register file names, metrics and schema in transaction
                success = register_table_write(
                    txn,
                    fnames,
                    file_records,
                    partition_infos,
                    partition_spec,
                    sort_order_id,
                )

            if use_theta_sketches:
                with bodo.ir.object_mode.no_warning_objmode(
                    snapshot_id="int64",
                    sequence_number="int64",
                    puffin_loc="unicode_type",
                ):
                    snapshot_id, sequence_number, puffin_loc = fetch_puffin_metadata(
                        txn
                    )

                statistic_file_info = _write_puffin_file(
                    unicode_to_utf8(puffin_loc),
                    unicode_to_utf8(bucket_region),
                    snapshot_id,
                    sequence_number,
                    writer["theta_sketches"],
                    writer["output_pyarrow_schema"],
                    writer["arrow_fs"],
                    unicode_to_utf8(old_puffin_file_path),
                )
                conn_str = writer["conn"]
                table_id = writer["table_id"]
                with bodo.ir.object_mode.no_warning_objmode():
                    commit_statistics_file(conn_str, table_id, statistic_file_info)

            # Delete the theta sketches. An object exists even if there are no sketches.
            delete_theta_sketches(writer["theta_sketches"])

            if not success:
                # TODO [BE-3249] If it fails due to schema changing, then delete the files.
                # Note that this might not always be possible since
                # we might not have DeleteObject permissions, for instance.
                raise BodoError("Iceberg write failed.")

            if writer["parallel"]:
                bodo.barrier()
            writer["finished"] = True

        ev.finalize()
        return is_last

    return impl_iceberg_writer_append_table


def iceberg_writer_append_table(
    writer, table, col_names_meta, local_is_last, iter
):  # pragma: no cover
    pass


@infer_global(iceberg_writer_append_table)
class IcebergWriterAppendInfer(AbstractTemplate):
    """Typer for iceberg_writer_append_table that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(iceberg_writer_append_table)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        # Update state type in signature to include build table type from input
        input_table_type = folded_args[1]
        new_state_type = IcebergWriterType(input_table_type)
        folded_args = (new_state_type, *folded_args[1:])
        return signature(types.bool_, *folded_args).replace(pysig=pysig)


IcebergWriterAppendInfer._no_unliteral = True


# Using a wrapper to keep iceberg_writer_append_table_inner as overload and avoid
# Numba objmode bugs (e.g. leftover ir.Del in IR leading to errors)
def impl_wrapper(
    writer, table, col_names_meta, local_is_last, iter
):  # pragma: no cover
    return iceberg_writer_append_table_inner(
        writer, table, col_names_meta, local_is_last, iter
    )


@lower_builtin(iceberg_writer_append_table, types.VarArg(types.Any))
def lower_iceberg_writer_append_table(context, builder, sig, args):
    """lower iceberg_writer_append_table() using gen_iceberg_writer_append_table_impl above"""
    return context.compile_internal(builder, impl_wrapper, sig, args)


def convert_to_snowflake_iceberg_table_py(
    snowflake_conn, iceberg_conn, iceberg_base, iceberg_volume, table_name, replace
):  # pragma: no cover
    """Convert Iceberg table written by Bodo to object storage to a Snowflake-managed
    Iceberg table.

    Args:
        snowflake_conn (str): Snowflake connection string
        iceberg_conn (str): Iceberg connection string used to contact the Iceberg catalog for
            information.
        iceberg_base (str): base storage path for Iceberg table (excluding volume bucket path)
        iceberg_volume (str): Snowflake Iceberg volume name
        table_name (str): table name
    """

    comm = MPI.COMM_WORLD
    err = None  # Forward declaration
    if bodo.get_rank() == 0:
        try:
            catalog = conn_str_to_catalog(iceberg_conn)
            table = catalog.load_table(f"{iceberg_base.replace('/', '.')}.{table_name}")
            full_metadata_path = table.metadata_location

            # Extract the metadata path that starts with our base location
            idx = full_metadata_path.find(iceberg_base)
            if idx == -1:
                raise RuntimeError(
                    f"Metadata path {full_metadata_path} does not contain base location {iceberg_base}"
                )
            else:
                metadata_path = full_metadata_path[idx:]

            # Connect to snowflake
            conn = bodo.io.snowflake.snowflake_connect(snowflake_conn)
            cursor = conn.cursor()

            # TODO[BSE-2666]: Add robust error checking

            # Make sure catalog integration exists
            catalog_integration_name = "BodoTmpObjectStoreCatalogInt"
            catalog_integration_query = f"""
            CREATE CATALOG INTEGRATION IF NOT EXISTS {catalog_integration_name}
                CATALOG_SOURCE=OBJECT_STORE
                TABLE_FORMAT=ICEBERG
                ENABLED=TRUE;
            """
            cursor.execute(catalog_integration_query)

            # Create Iceberg table
            base = f"{iceberg_base}/{table_name}"
            if replace:
                or_replace = "OR REPLACE"
            else:
                or_replace = ""
            create_query = f"""
                CREATE {or_replace} ICEBERG TABLE {table_name}
                EXTERNAL_VOLUME='{iceberg_volume}'
                CATALOG='{catalog_integration_name}'
                METADATA_FILE_PATH='{metadata_path}';
            """
            cursor.execute(create_query)

            # Convert Iceberg table to Snowflake managed
            convert_query = f"""
                ALTER ICEBERG TABLE {table_name} CONVERT TO MANAGED
                    BASE_LOCATION = '{base}';
            """
            cursor.execute(convert_query)

        except Exception as e:
            err = RuntimeError(str(e))
            if int(os.environ.get("BODO_SF_DEBUG_LEVEL", "0")) >= 1:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err


def convert_to_snowflake_iceberg_table(
    snowflake_conn,
    iceberg_conn,
    iceberg_base,
    iceberg_volume,
    schema,
    table_name,
    replace,
):  # pragma: no cover
    pass


@overload(convert_to_snowflake_iceberg_table)
def overload_convert_to_snowflake_iceberg_table(
    snowflake_conn, iceberg_conn, iceberg_base, iceberg_volume, table_name, replace
):  # pragma: no cover
    """JIT wrapper around convert_to_snowflake_iceberg_table_py above"""

    def impl(
        snowflake_conn, iceberg_conn, iceberg_base, iceberg_volume, table_name, replace
    ):  # pragma: no cover
        with bodo.ir.object_mode.no_warning_objmode:
            convert_to_snowflake_iceberg_table_py(
                snowflake_conn,
                iceberg_conn,
                iceberg_base,
                iceberg_volume,
                table_name,
                replace,
            )

    return impl
