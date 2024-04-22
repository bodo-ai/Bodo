import operator
import os
import traceback

import llvmlite.binding as ll
import numba
import pandas as pd
from llvmlite import ir as lir
from mpi4py import MPI
from numba.core import cgutils, types
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
    pyarrow_schema_type,
    stream_writer_alloc_codegen,
)
from bodo.io.iceberg import (
    generate_data_file_info,
    get_table_details_before_write,
    iceberg_pq_write,
    python_list_of_heterogeneous_tuples_type,
    register_table_write,
    theta_sketch_collection_type,
    wrap_start_write,
)
from bodo.libs import puffin_file, theta_sketches
from bodo.libs.array import (
    ArrayInfoType,
    array_to_info,
    delete_info,
    info_to_array,
    py_table_to_cpp_table,
)
from bodo.libs.table_builder import TableBuilderStateType
from bodo.utils import tracing
from bodo.utils.transform import get_call_expr_arg
from bodo.utils.typing import (
    BodoError,
    ColNamesMetaType,
    get_overload_const_str,
    is_overload_bool,
    is_overload_constant_str,
    is_overload_none,
    unwrap_typeref,
)

ll.add_symbol("init_theta_sketches", theta_sketches.init_theta_sketches_py_entrypt)
ll.add_symbol(
    "fetch_ndv_approximations", theta_sketches.fetch_ndv_approximations_py_entrypt
)
ll.add_symbol("write_puffin_file", puffin_file.write_puffin_file_py_entrypt)

# Maximum Parquet file size for streaming Iceberg write
# TODO[BSE-2609] get max file size from Iceberg metadata
ICEBERG_WRITE_PARQUET_CHUNK_SIZE = int(256e6)


def get_env_value(env_var):  # pragma: no cover
    pass


@overload(get_env_value)
def overload_get_env_value(env_var):
    """
    Returns the current runtime value of an environment variable
    """

    def impl(env_var):  # pragma: no cover
        with bodo.objmode(env_value="string"):
            env_value = os.environ.get(env_var, "0")
        return env_value

    return impl


@intrinsic(prefer_literal=True)
def _init_theta_sketches(
    typingctx,
    output_pyarrow_schema_t,
    already_exists_t,
    enable_theta_sketches_t,
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # table_info*
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="init_theta_sketches"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        theta_sketch_collection_type(
            pyarrow_schema_type,
            types.bool_,
            types.bool_,
        ),
        codegen,
    )


def init_theta_sketches_wrapper(
    con_str, schema, table_name, output_pyarrow_schema, already_exists
):  # pragma: no cover
    pass


@overload(init_theta_sketches_wrapper)
def overload_init_theta_sketches_wrapper(
    con_str, schema, table_name, output_pyarrow_schema, already_exists
):
    """
    Creates a new theta sketch collection when starting to write an Iceberg table.
    For now, most of the arguments are unused and we just worry about the schema,
    whether the table already exists, and whether theta sketches are allowed.

    con_str: Iceberg connection string
    schema: database schema where the table is being written to
    table_name: table name being written
    output_pyarrow_schema: Iceberg pyarrow schema of the final table
    already_exists: true if the table is being inserted into, false if it is being created from scratch
    """

    def impl(
        con_str, schema, table_name, output_pyarrow_schema, already_exists
    ):  # pragma: no cover
        enable_theta = get_env_value("BODO_ENABLE_THETA_SKETCHES") != "0"
        return _init_theta_sketches(output_pyarrow_schema, already_exists, enable_theta)

    return impl


@intrinsic(prefer_literal=True)
def _iceberg_writer_fetch_theta(typingctx, array_info_t, output_pyarrow_schema_t):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # table_info*
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="fetch_ndv_approximations"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        ArrayInfoType()(theta_sketch_collection_type, pyarrow_schema_type),
        codegen,
    )


def iceberg_writer_fetch_theta(writer):
    pass


@overload(iceberg_writer_fetch_theta)
def overload_iceberg_writer_fetch_theta(writer):
    """
    Fetches the current values of the theta sketch approximations
    of NDV for each column in an iceberg writer. For each column
    that does not have a theta sketch, returns null instead. Largely
    used for testing purposes.
    """
    arr_type = bodo.FloatingArrayType(types.float64)

    def impl(writer):  # pragma: no cover
        res_info = _iceberg_writer_fetch_theta(
            writer["theta_sketches"], writer["output_pyarrow_schema"]
        )
        res = info_to_array(res_info, arr_type)
        delete_info(res_info)
        return res

    return impl


@intrinsic
def _write_puffin_file(
    typingctx,
    table_loc_t,
    snapshot_id_t,
    sequence_number_t,
    theta_sketches_t,
    output_pyarrow_schema_t,
    already_exists_t,
):
    def codegen(context, builder, sig, args):
        (
            table_loc_str,
            snapshot_id,
            sequence_number,
            theta_sketches,
            output_pyarrow_schema,
            already_exists,
        ) = args
        table_loc_struct = cgutils.create_struct_proxy(types.unicode_type)(
            context, builder, value=table_loc_str
        )
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),  # table_loc buffer
                lir.IntType(64),  # table_loc length
                lir.IntType(64),  # snapshot_id
                lir.IntType(64),  # sequence_number
                lir.IntType(8).as_pointer(),  # theta_sketches
                lir.IntType(8).as_pointer(),  # output_pyarrow_schema
                lir.IntType(1),  # already_exists
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="write_puffin_file"
        )
        ret = builder.call(
            fn_tp,
            [
                table_loc_struct.data,
                table_loc_struct.length,
                snapshot_id,
                sequence_number,
                theta_sketches,
                output_pyarrow_schema,
                already_exists,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        types.void(
            table_loc_t,
            snapshot_id_t,
            sequence_number_t,
            theta_sketches_t,
            output_pyarrow_schema_t,
            already_exists_t,
        ),
        codegen,
    )


def fetch_snapshot_id(rank, conn, db_schema, table_name):
    """
    Fetches the snapshot_id from the current Iceberg transaction.
    """
    from bodo_iceberg_connector.catalog_conn import parse_conn_str
    from bodo_iceberg_connector.py4j_support import get_java_table_handler

    snapshot_id = -1
    if rank == 0:
        catalog_type, _ = parse_conn_str(conn)
        bodo_iceberg_table_reader = get_java_table_handler(
            conn,
            catalog_type,
            db_schema,
            table_name,
        )
        # TODO: get this from the write commit function instead so we can
        # avoid getting the wrong snapshotID
        snapshot_id = bodo_iceberg_table_reader.getSnapshotId()
    snapshot_id = MPI.COMM_WORLD.bcast(snapshot_id)
    return snapshot_id


class IcebergWriterType(types.Type):
    """Data type for streaming Iceberg writer's internal state"""

    def __init__(self, input_table_type=types.unknown):
        self.input_table_type = input_table_type
        super().__init__(name=f"IcebergWriterType({input_table_type})")


class IcebergWriterPayloadType(types.Type):
    """Data type for streaming Iceberg writer's payload"""

    def __init__(self):
        super().__init__(name="IcebergWriterPayloadType")


iceberg_writer_payload_type = IcebergWriterPayloadType()


iceberg_writer_payload_members = (
    # Iceberg connection string
    ("conn", types.unicode_type),
    # Table name to write
    ("table_name", types.unicode_type),
    # Database schema to create a table
    ("db_schema", types.unicode_type),
    # Action to take if table already exists: fail, replace, append
    ("if_exists", types.unicode_type),
    # Location of the data/ folder in the warehouse
    ("table_loc", types.unicode_type),
    # Known Schema ID when files were written
    ("iceberg_schema_id", types.int64),
    # JSON Encoding of Iceberg Schema to include in Parquet metadata
    ("iceberg_schema_str", types.unicode_type),
    # Output pyarrow schema that should be written to the Iceberg table.
    # This also contains the Iceberg field IDs in the fields' metadata
    # which is important during the commit step.
    ("output_pyarrow_schema", pyarrow_schema_type),
    # Array of Tuples containing Partition Spec for Iceberg Table (passed to C++)
    ("partition_spec", python_list_of_heterogeneous_tuples_type),
    # Array of Tuples containing Sort Order for Iceberg Table (passed to C++)
    ("sort_order", python_list_of_heterogeneous_tuples_type),
    # List of written file infos needed by Iceberg for committing
    ("iceberg_files_info", python_list_of_heterogeneous_tuples_type),
    # Whether write is occurring in parallel
    ("parallel", types.boolean),
    # Whether this rank has finished appending data to the table
    ("finished", types.boolean),
    # Batches collected to write
    ("batches", TableBuilderStateType()),
    # Whether the table is being inserted into, as opposed to created from scratch
    ("already_exists", types.boolean),
    # Collection of theta sketch data for the columns that have it
    ("theta_sketches", theta_sketch_collection_type),
    # Transaction ID for the write
    ("txn_id", types.int64),
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


@intrinsic(prefer_literal=True)
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
        f"Boxing is disabled for IcebergWriter mutable struct."
    )  # pragma: no cover


@unbox(IcebergWriterType)
def unbox_iceberg_writer(typ, val, c):
    raise NotImplementedError(
        f"Unboxing is disabled for IcebergWriter mutable struct."
    )  # pragma: no cover


def start_write_wrapper(
    conn,
    schema,
    table_name,
    table_loc,
    iceberg_schema_id,
    already_exists,
    output_pyarrow_schema,
    partition_spec,
    sort_order,
    mode,
):
    pass


@overload(start_write_wrapper)
def overload_start_write_wrapper(
    conn,
    schema,
    table_name,
    table_loc,
    iceberg_schema_id,
    already_exists,
    output_pyarrow_schema,
    partition_spec,
    sort_order,
    mode,
):
    """Wrapper around objmode call to connector.start_write() to avoid Numba
    compiler errors"""

    def impl(
        conn,
        schema,
        table_name,
        table_loc,
        iceberg_schema_id,
        already_exists,
        output_pyarrow_schema,
        partition_spec,
        sort_order,
        mode,
    ):  # pragma: no cover
        with bodo.no_warning_objmode(txn_id="i8", table_loc="unicode_type"):
            (txn_id, table_loc) = wrap_start_write(
                conn,
                schema,
                table_name,
                table_loc,
                iceberg_schema_id,
                already_exists,
                output_pyarrow_schema,
                partition_spec,
                sort_order,
                mode,
            )
        return txn_id, table_loc

    return impl


def get_table_details_before_write_wrapper(
    table_name, conn, schema, df_pyarrow_schema, if_exists, allow_downcasting
):  # pragma: no cover
    pass


@overload(get_table_details_before_write_wrapper)
def overload_get_table_details_before_write_wrapper(
    table_name, conn, schema, df_pyarrow_schema, if_exists, allow_downcasting
):
    """Wrapper around objmode call to get_table_details_before_write() to avoid Numba
    compiler errors
    """

    def impl(
        table_name, conn, schema, df_pyarrow_schema, if_exists, allow_downcasting
    ):  # pragma: no cover
        with bodo.no_warning_objmode(
            table_loc="unicode_type",
            iceberg_schema_id="i8",
            partition_spec="python_list_of_heterogeneous_tuples_type",
            sort_order="python_list_of_heterogeneous_tuples_type",
            iceberg_schema_str="unicode_type",
            output_pyarrow_schema="pyarrow_schema_type",
            mode="unicode_type",
            already_exists="boolean",
        ):
            (
                table_loc,
                already_exists,
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
                schema,
                df_pyarrow_schema,
                if_exists,
                allow_downcasting,
            )
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

    return impl


def get_empty_pylist():  # pragma: no cover
    pass


@overload(get_empty_pylist)
def overload_get_empty_pylist():
    """Return an empty Python list object"""

    def impl():  # pragma: no cover
        with bodo.no_warning_objmode(a="python_list_of_heterogeneous_tuples_type"):
            a = []
        return a

    return impl


def iceberg_writer_init(
    operator_id,
    conn,
    table_name,
    schema,
    col_names_meta,
    if_exists,
    expected_state_type=None,
    input_dicts_unified=False,
    _is_parallel=False,
):  # pragma: no cover
    pass


def gen_iceberg_writer_init_impl(
    operator_id,
    conn,
    table_name,
    schema,
    col_names_meta,
    if_exists,
    expected_state_type=None,
    input_dicts_unified=False,
    _is_parallel=False,
):  # pragma: no cover
    """Initialize Iceberg stream writer"""
    from bodo.hiframes.pd_dataframe_ext import DataFrameType

    col_names_meta = unwrap_typeref(col_names_meta)
    col_names = col_names_meta.meta

    expected_state_type = unwrap_typeref(expected_state_type)
    if is_overload_none(expected_state_type):
        iceberg_writer_type = IcebergWriterType()
    else:
        iceberg_writer_type = expected_state_type

    table_builder_state_type = TableBuilderStateType(
        iceberg_writer_type.input_table_type
    )

    input_df_type = DataFrameType(
        iceberg_writer_type.input_table_type.arr_types, None, col_names
    )
    df_pyarrow_schema = bodo.io.helpers.numba_to_pyarrow_schema(
        input_df_type, is_iceberg=True
    )

    def impl_iceberg_writer_init(
        operator_id,
        conn,
        table_name,
        schema,
        col_names_meta,
        if_exists,
        expected_state_type=None,
        input_dicts_unified=False,
        _is_parallel=False,
    ):
        ev = tracing.Event("iceberg_writer_init", is_parallel=_is_parallel)
        assert _is_parallel, "Iceberg Write only supported for distributed dataframes"
        con_str = bodo.io.iceberg.format_iceberg_conn_njit(conn)

        (
            table_loc,
            already_exists,
            iceberg_schema_id,
            partition_spec,
            sort_order,
            iceberg_schema_str,
            # This has the Iceberg Field IDs
            # in the fields' metadata.
            output_pyarrow_schema,
            mode,
        ) = get_table_details_before_write_wrapper(
            table_name,
            con_str,
            schema,
            df_pyarrow_schema,
            if_exists,
            # allow_downcasting
            False,
        )
        (
            txn_id,
            table_loc,
        ) = start_write_wrapper(
            con_str,
            schema,
            table_name,
            table_loc,
            iceberg_schema_id,
            already_exists,
            output_pyarrow_schema,
            partition_spec,
            sort_order,
            mode,
        )

        # Initialize writer
        writer = iceberg_writer_alloc(expected_state_type)
        writer["conn"] = con_str
        writer["table_name"] = table_name
        writer["db_schema"] = schema
        writer["if_exists"] = mode
        writer["table_loc"] = table_loc
        writer["iceberg_schema_id"] = iceberg_schema_id
        writer["iceberg_schema_str"] = iceberg_schema_str
        writer["output_pyarrow_schema"] = output_pyarrow_schema
        writer["partition_spec"] = partition_spec
        writer["sort_order"] = sort_order
        writer["iceberg_files_info"] = get_empty_pylist()
        writer["parallel"] = _is_parallel
        writer["finished"] = False
        writer["batches"] = bodo.libs.table_builder.init_table_builder_state(
            operator_id,
            table_builder_state_type,
            input_dicts_unified=input_dicts_unified,
        )
        writer["already_exists"] = already_exists
        writer["theta_sketches"] = init_theta_sketches_wrapper(
            con_str, schema, table_name, output_pyarrow_schema, already_exists
        )
        writer["txn_id"] = txn_id

        # Barrier ensures that internal stage exists before we upload files to it
        bodo.barrier()
        ev.finalize()
        return writer

    return impl_iceberg_writer_init


@infer_global(iceberg_writer_init)
class IcebergWriterInitInfer(AbstractTemplate):
    """Typer for iceberg_writer_init that returns writer type"""

    def generic(self, args, kws):
        kws = dict(kws)
        expected_state_type = get_call_expr_arg(
            "iceberg_writer_init",
            args,
            kws,
            6,
            "expected_state_type",
            default=types.none,
        )
        expected_state_type = unwrap_typeref(expected_state_type)
        if is_overload_none(expected_state_type):
            iceberg_writer_type = IcebergWriterType()
        else:
            iceberg_writer_type = expected_state_type

        pysig = numba.core.utils.pysignature(iceberg_writer_init)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(iceberg_writer_type, *folded_args).replace(pysig=pysig)


IcebergWriterInitInfer._no_unliteral = True


@lower_builtin(iceberg_writer_init, types.VarArg(types.Any))
def lower_iceberg_writer_init(context, builder, sig, args):
    """lower iceberg_writer_init() using gen_iceberg_writer_init_impl above"""
    impl = gen_iceberg_writer_init_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def append_py_list(pylist, to_append):  # pragma: no cover
    pass


@overload(append_py_list)
def overload_append_py_list(pylist, to_append):
    """Append a Python list object to existing Python list object"""

    def impl(pylist, to_append):  # pragma: no cover
        with bodo.no_warning_objmode:
            pylist.extend(to_append)

    return impl


def iceberg_writer_append_table_inner(
    writer, table, col_names_meta, is_last, iter
):  # pragma: no cover
    pass


@overload(iceberg_writer_append_table_inner)
def gen_iceberg_writer_append_table_impl_inner(
    writer,
    table,
    col_names_meta,
    is_last,
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
    if not is_overload_bool(is_last):  # pragma: no cover
        raise BodoError(
            f"iceberg_writer_append_table: Expected type boolean "
            f"for `is_last`, found {is_last}"
        )

    col_names_meta = unwrap_typeref(col_names_meta)
    if not isinstance(col_names_meta, ColNamesMetaType):  # pragma: no cover
        raise BodoError(
            f"iceberg_writer_append_table: Expected type ColNamesMetaType "
            f"for `col_names_meta`, found {col_names_meta}"
        )
    if not isinstance(col_names_meta.meta, tuple):  # pragma: no cover
        raise BodoError(
            f"iceberg_writer_append_table: Expected col_names_meta "
            f"to contain a tuple of column names"
        )

    py_table_typ = table
    col_names_arr = pd.array(col_names_meta.meta)

    def impl_iceberg_writer_append_table(
        writer, table, col_names_meta, is_last, iter
    ):  # pragma: no cover
        if writer["finished"]:
            return True
        ev = tracing.Event(
            "iceberg_writer_append_table", is_parallel=writer["parallel"]
        )
        is_last = bodo.libs.distributed_api.sync_is_last(is_last, iter)

        # ===== Part 1: Accumulate batch in writer and compute total size
        ev_append_batch = tracing.Event(f"append_batch", is_parallel=True)
        table_builder_state = writer["batches"]
        bodo.libs.table_builder.table_builder_append(table_builder_state, table)
        table_bytes = bodo.libs.table_builder.table_builder_nbytes(table_builder_state)
        ev_append_batch.add_attribute("table_bytes", table_bytes)
        ev_append_batch.finalize()

        # ===== Part 2: Write Parquet file if file size threshold is exceeded
        if is_last or table_bytes >= ICEBERG_WRITE_PARQUET_CHUNK_SIZE:
            # Note: Our write batches are at least as large as our read batches. It may
            # be advantageous in the future to split up large incoming batches into
            # multiple Parquet files to write.

            # NOTE: table_builder_reset() below affects the table builder state so
            # out_table should be used immediately and not be stored.
            out_table = bodo.libs.table_builder.table_builder_get_data(
                table_builder_state
            )
            out_table_len = len(out_table)
            if out_table_len > 0:
                ev_upload_table = tracing.Event("upload_table", is_parallel=False)

                table_info = py_table_to_cpp_table(out_table, py_table_typ)
                col_names_info = array_to_info(col_names_arr)
                iceberg_files_info = iceberg_pq_write(
                    writer["table_loc"],
                    table_info,
                    col_names_info,
                    writer["partition_spec"],
                    writer["sort_order"],
                    writer["iceberg_schema_str"],
                    writer["parallel"],
                    writer["output_pyarrow_schema"],
                    writer["theta_sketches"],
                )
                append_py_list(writer["iceberg_files_info"], iceberg_files_info)

                ev_upload_table.finalize()
            bodo.libs.table_builder.table_builder_reset(table_builder_state)

        # ===== Part 3: Commit Iceberg write
        if is_last:
            conn = writer["conn"]
            db_schema = writer["db_schema"]
            table_name = writer["table_name"]
            table_loc = writer["table_loc"]
            iceberg_schema_id = writer["iceberg_schema_id"]
            if_exists = writer["if_exists"]
            all_iceberg_files_infos = writer["iceberg_files_info"]
            txn_id = writer["txn_id"]
            with bodo.no_warning_objmode(success="bool_"):
                (
                    fnames,
                    file_size_bytes,
                    metrics,
                ) = generate_data_file_info(all_iceberg_files_infos)

                # Send file names, metrics and schema to Iceberg connector
                success = register_table_write(
                    txn_id,
                    conn,
                    db_schema,
                    table_name,
                    table_loc,
                    fnames,
                    file_size_bytes,
                    metrics,
                    iceberg_schema_id,
                    if_exists,
                )

            # If theta sketches are turned on, fetch the snapshot id of the
            # current transaction and use it to write the Puffin file
            enable_theta = get_env_value("BODO_ENABLE_THETA_SKETCHES") != "0"
            if enable_theta:
                snapshot_id = -1
                rank = bodo.get_rank()
                with bodo.no_warning_objmode(snapshot_id="int64"):
                    snapshot_id = fetch_snapshot_id(rank, conn, db_schema, table_name)

                # TODO: find real way to determine sequence_number
                sequence_number = 1
                _write_puffin_file(
                    table_loc,
                    snapshot_id,
                    sequence_number,
                    writer["theta_sketches"],
                    writer["output_pyarrow_schema"],
                    writer["already_exists"],
                )

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
    writer, table, col_names_meta, is_last, iter
):  # pragma: no cover
    pass


@infer_global(iceberg_writer_append_table)
class IcebergWriterAppendInfer(AbstractTemplate):
    """Typer for iceberg_writer_append_table that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(iceberg_writer_append_table)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(types.bool_, *folded_args).replace(pysig=pysig)


IcebergWriterAppendInfer._no_unliteral = True


# Using a wrapper to keep iceberg_writer_append_table_inner as overload and avoid
# Numba objmode bugs (e.g. leftover ir.Del in IR leading to errors)
def impl_wrapper(writer, table, col_names_meta, is_last, iter):  # pragma: no cover
    return iceberg_writer_append_table_inner(
        writer, table, col_names_meta, is_last, iter
    )


@lower_builtin(iceberg_writer_append_table, types.VarArg(types.Any))
def lower_iceberg_writer_append_table(context, builder, sig, args):
    """lower iceberg_writer_append_table() using gen_iceberg_writer_append_table_impl above"""
    return context.compile_internal(builder, impl_wrapper, sig, args)


def convert_to_snowflake_iceberg_table_py(
    snowflake_conn, iceberg_base, iceberg_volume, table_name
):  # pragma: no cover
    """Convert Iceberg table written by Bodo to object storage to a Snowflake-managed
    Iceberg table.

    Args:
        snowflake_conn (str): Snowflake connection string
        iceberg_base (str): base storage path for Iceberg table (excluding volume bucket path)
        iceberg_volume (str): Snowflake Iceberg volume name
        table_name (str): table name
    """

    comm = MPI.COMM_WORLD
    err = None  # Forward declaration
    if bodo.get_rank() == 0:
        try:
            # Connect to snowflake
            conn = bodo.io.snowflake.snowflake_connect(snowflake_conn)
            cursor = conn.cursor()

            # TODO[BSE-2666]: Add robust error checking

            # Make sure catalog integration exists
            catalog_integration_name = "BodoTmpCatalogInt"
            catalog_integration_query = f"""
            CREATE OR REPLACE CATALOG INTEGRATION {catalog_integration_name}
                CATALOG_SOURCE=OBJECT_STORE
                TABLE_FORMAT=ICEBERG
                ENABLED=TRUE;
            """
            cursor.execute(catalog_integration_query)

            # Create Iceberg table
            base = f"{iceberg_base}/{table_name}"
            create_query = f"""
                CREATE ICEBERG TABLE {table_name}
                EXTERNAL_VOLUME='{iceberg_volume}'
                CATALOG='{catalog_integration_name}'
                METADATA_FILE_PATH='{base}/metadata/v1.metadata.json';
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
    snowflake_conn, iceberg_base, iceberg_volume, schema, table_name
):  # pragma: no cover
    pass


@overload(convert_to_snowflake_iceberg_table)
def overload_convert_to_snowflake_iceberg_table(
    snowflake_conn, iceberg_base, iceberg_volume, table_name
):  # pragma: no cover
    """JIT wrapper around convert_to_snowflake_iceberg_table_py above"""

    def impl(
        snowflake_conn, iceberg_base, iceberg_volume, table_name
    ):  # pragma: no cover
        with bodo.no_warning_objmode:
            convert_to_snowflake_iceberg_table_py(
                snowflake_conn, iceberg_base, iceberg_volume, table_name
            )

    return impl
