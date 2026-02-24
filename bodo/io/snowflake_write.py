import operator
import os
import traceback

import numba
import numpy as np
import pandas as pd
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
    stream_writer_alloc_codegen,
)
from bodo.io.parquet_write import parquet_write_table_cpp
from bodo.io.snowflake import (
    snowflake_connector_cursor_type,
    temporary_directory_type,
)
from bodo.libs.array import array_to_info, cpp_table_map_to_list, py_table_to_cpp_table
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


class SnowflakeWriterType(StreamingStateType):
    """Data type for streaming Snowflake writer's internal state"""

    def __init__(self, input_table_type=types.unknown):
        self.input_table_type = input_table_type
        super().__init__(name=f"SnowflakeWriterType({input_table_type})")

    def is_precise(self):
        return self.input_table_type != types.unknown

    def unify(self, typingctx, other):
        """Unify two SnowflakeWriterType instances when one doesn't have a resolved
        input_table_type.
        """
        if isinstance(other, SnowflakeWriterType):
            if not other.is_precise() and self.is_precise():
                return self

            # Prefer the new type in case write append changed its table type
            return other


class SnowflakeWriterPayloadType(types.Type):
    """Data type for streaming Snowflake writer's payload"""

    def __init__(self):
        super().__init__(name="SnowflakeWriterPayloadType")


snowflake_writer_payload_type = SnowflakeWriterPayloadType()


snowflake_writer_payload_members = (
    # Snowflake connection string
    ("conn", types.unicode_type),
    # Location on Snowflake to create a table
    ("location", types.unicode_type),
    # Action to take if table already exists: fail, replace, append
    ("if_exists", types.unicode_type),
    # Type of table to create: permanent, temporary, transient
    ("table_type", types.unicode_type),
    # Whether write is occurring in parallel
    ("parallel", types.boolean),
    # Whether this rank has finished appending data to the table
    ("finished", types.boolean),
    # Region of internal stage bucket
    ("bucket_region", types.unicode_type),
    # Total number of Parquet files written on this rank that have not yet
    # been COPY INTO'd
    ("file_count_local", types.int64),
    # Total number of Parquet files written across all ranks that have not yet
    # been COPY INTO'd. In the parallel case, this may be out of date as we
    # only sync every `bodo.stream_loop_sync_iters` iterations
    ("file_count_global", types.int64),
    # Copy into directory
    ("copy_into_dir", types.unicode_type),
    #  Snowflake query ID for previous COPY INTO command
    ("copy_into_prev_sfqid", types.unicode_type),
    # SQL that should be applied at the end to flatten map ararys
    ("flatten_sql", types.unicode_type),
    # Tempoary table for flattening map arrays
    ("flatten_table", types.unicode_type),
    # Flag indicating if the Snowflake transaction has started
    ("is_initialized", types.boolean),
    # File count for previous COPY INTO command
    ("file_count_global_prev", types.int64),
    # Snowflake connection cursor. Only on rank 0, unless PUT method is used
    ("cursor", snowflake_connector_cursor_type),
    # Python TemporaryDirectory object, which stores Parquet files during PUT upload
    ("tmp_folder", temporary_directory_type),
    # Name of created internal stage
    ("stage_name", types.unicode_type),
    # Parquet path of internal stage, could be an S3/ADLS URI or a local path
    # in case of upload using PUT. Includes a trailing slash
    ("stage_path", types.unicode_type),
    # Whether we are using the Snowflake PUT command to upload files. This is
    # set to True if we don't support the stage type returned by Snowflake
    ("upload_using_snowflake_put", types.boolean),
    # Old environment variables that were overwritten to update credentials
    # for uploading to stage
    ("old_creds", types.DictType(types.unicode_type, types.unicode_type)),
    # Batches collected to write
    ("batches", TableBuilderStateType()),
    # Whether the `copy_intosfqids` exists.
    ("copy_into_sfqids_exists", types.boolean),
    # Keep track of list of copy into async Snowflake query ids
    ("copy_into_sfqids", types.unicode_type),
)
snowflake_writer_payload_members_dict = dict(snowflake_writer_payload_members)


@register_model(SnowflakeWriterPayloadType)
class SnowflakeWriterPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):  # pragma: no cover
        members = snowflake_writer_payload_members
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(SnowflakeWriterType)
class SnowflakeWriterModel(models.StructModel):
    def __init__(self, dmm, fe_type):  # pragma: no cover
        payload_type = snowflake_writer_payload_type
        members = [
            ("meminfo", types.MemInfoPointer(payload_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@intrinsic
def sf_writer_alloc(typingctx, expected_state_type_t):  # pragma: no cover
    expected_state_type = unwrap_typeref(expected_state_type_t)
    if is_overload_none(expected_state_type):
        snowflake_writer_type = SnowflakeWriterType()
    else:
        snowflake_writer_type = expected_state_type

    def codegen(context, builder, sig, args):  # pragma: no cover
        """Creates meminfo and sets dtor for Snowflake writer"""
        return stream_writer_alloc_codegen(
            context,
            builder,
            snowflake_writer_payload_type,
            snowflake_writer_type,
            snowflake_writer_payload_members,
        )

    return snowflake_writer_type(expected_state_type_t), codegen


@intrinsic(prefer_literal=True)
def sf_writer_getattr(typingctx, writer_typ, attr_typ):  # pragma: no cover
    """Get attribute of a Snowflake writer"""
    assert isinstance(writer_typ, SnowflakeWriterType), (
        f"sf_writer_getattr: expected `writer` to be a SnowflakeWriterType, "
        f"but found {writer_typ}"
    )
    assert is_overload_constant_str(attr_typ), (
        f"sf_writer_getattr: expected `attr` to be a literal string type, "
        f"but found {attr_typ}"
    )
    attr = get_overload_const_str(attr_typ)
    val_typ = snowflake_writer_payload_members_dict[attr]
    if attr == "batches":
        val_typ = TableBuilderStateType(writer_typ.input_table_type)

    def codegen(context, builder, sig, args):  # pragma: no cover
        writer, _ = args
        payload, _ = _get_stream_writer_payload(
            context, builder, writer_typ, snowflake_writer_payload_type, writer
        )
        return impl_ret_borrowed(
            context, builder, sig.return_type, getattr(payload, attr)
        )

    return val_typ(writer_typ, attr_typ), codegen


@intrinsic(prefer_literal=True)
def sf_writer_setattr(typingctx, writer_typ, attr_typ, val_typ):  # pragma: no cover
    """Set attribute of a Snowflake writer"""
    assert isinstance(writer_typ, SnowflakeWriterType), (
        f"sf_writer_setattr: expected `writer` to be a SnowflakeWriterType, "
        f"but found {writer_typ}"
    )
    assert is_overload_constant_str(attr_typ), (
        f"sf_writer_setattr: expected `attr` to be a literal string type, "
        f"but found {attr_typ}"
    )
    attr = get_overload_const_str(attr_typ)

    # Storing a literal type into the payload causes a type mismatch
    val_typ = numba.types.unliteral(val_typ)

    def codegen(context, builder, sig, args):  # pragma: no cover
        writer, _, val = args
        payload, meminfo_data_ptr = _get_stream_writer_payload(
            context, builder, writer_typ, snowflake_writer_payload_type, writer
        )
        context.nrt.decref(builder, val_typ, getattr(payload, attr))
        context.nrt.incref(builder, val_typ, val)
        setattr(payload, attr, val)
        builder.store(payload._getvalue(), meminfo_data_ptr)
        return context.get_dummy_value()

    return types.none(writer_typ, attr_typ, val_typ), codegen


@overload(operator.getitem, no_unliteral=True)
def snowflake_writer_getitem(writer, attr):
    if not isinstance(writer, SnowflakeWriterType):
        return

    return lambda writer, attr: sf_writer_getattr(writer, attr)  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def snowflake_writer_setitem(writer, attr, val):
    if not isinstance(writer, SnowflakeWriterType):
        return

    return lambda writer, attr, val: sf_writer_setattr(
        writer, attr, val
    )  # pragma: no cover


@box(SnowflakeWriterType)
def box_snowflake_writer(typ, val, c):
    # Boxing is disabled, to avoid boxing overheads anytime a writer attribute
    # is accessed from objmode. As a workaround, store the necessary attributes
    # into local variables in numba native code before entering objmode
    raise NotImplementedError(
        "Boxing is disabled for SnowflakeWriter mutable struct."
    )  # pragma: no cover


@unbox(SnowflakeWriterType)
def unbox_snowflake_writer(typ, val, c):
    raise NotImplementedError(
        "Unboxing is disabled for SnowflakeWriter mutable struct."
    )  # pragma: no cover


def begin_write_transaction(
    cursor, location, sf_schema, if_exists, table_type, create_table_info
):
    """
    Begin the write transactions within the connector
    This include the BEGIN transaction as well as CREATE TABLE
    """
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    err = None
    if my_rank == 0:
        try:
            cursor.execute("BEGIN /* io.snowflake_write.begin_write_transaction() */")
            bodo.io.snowflake.create_table_handle_exists(
                cursor,
                location,
                sf_schema,
                if_exists,
                table_type,
                always_escape_col_names=True,
                create_table_info=create_table_info,
            )
        except Exception as e:
            err = RuntimeError(str(e))
            if int(os.environ.get("BODO_SF_DEBUG_LEVEL", "0")) >= 1:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err


def snowflake_writer_init(
    operator_id,
    conn,
    table_name,
    schema,
    if_exists,
    table_type,
    input_dicts_unified=False,
    _is_parallel=False,
):  # pragma: no cover
    pass


def _get_schema_str(schema):  # pragma: no cover
    pass


@overload(_get_schema_str)
def overload_get_schema_str(schema):
    """return schema as string if not None"""

    if not is_overload_none(schema):
        return lambda schema: '"' + schema + '".'  # pragma: no cover

    return lambda schema: ""  # pragma: no cover


def connect_and_get_upload_info_jit(conn):  # pragma: no cover
    pass


@overload(connect_and_get_upload_info_jit)
def overload_connect_and_get_upload_info_jit(conn):
    """JIT version of connect_and_get_upload_info which wraps objmode (isolated to avoid Numba objmode bugs)"""

    def impl(conn):
        with bodo.ir.object_mode.no_warning_objmode(
            cursor="snowflake_connector_cursor_type",
            tmp_folder="temporary_directory_type",
            stage_name="unicode_type",
            stage_path="unicode_type",
            upload_using_snowflake_put="boolean",
            old_creds="DictType(unicode_type, unicode_type)",
        ):
            (
                cursor,
                tmp_folder,
                stage_name,
                stage_path,
                upload_using_snowflake_put,
                old_creds,
            ) = bodo.io.snowflake.connect_and_get_upload_info(conn)

        return (
            cursor,
            tmp_folder,
            stage_name,
            stage_path,
            upload_using_snowflake_put,
            old_creds,
        )

    return impl


def gen_snowflake_writer_init_impl(
    snowflake_writer_type,
    operator_id,
    conn,
    table_name,
    schema,
    if_exists,
    table_type,
    input_dicts_unified=False,
    _is_parallel=False,
):  # pragma: no cover
    table_builder_state_type = TableBuilderStateType(
        snowflake_writer_type.input_table_type
    )

    def impl_snowflake_writer_init(
        operator_id,
        conn,
        table_name,
        schema,
        if_exists,
        table_type,
        input_dicts_unified=False,
        _is_parallel=False,
    ):
        ev = tracing.Event("snowflake_writer_init", is_parallel=_is_parallel)
        location = _get_schema_str(schema)
        location += table_name
        # Initialize writer
        writer = sf_writer_alloc(snowflake_writer_type)
        writer["conn"] = conn
        writer["location"] = location
        writer["if_exists"] = if_exists
        writer["table_type"] = table_type
        writer["parallel"] = _is_parallel
        writer["finished"] = False
        writer["file_count_local"] = 0
        writer["file_count_global"] = 0
        writer["copy_into_prev_sfqid"] = ""
        writer["flatten_sql"] = ""
        writer["flatten_table"] = ""
        writer["copy_into_sfqids_exists"] = False
        writer["copy_into_sfqids"] = ""
        writer["file_count_global_prev"] = 0
        writer["batches"] = bodo.libs.table_builder.init_table_builder_state(
            operator_id,
            table_builder_state_type,
            input_dicts_unified=input_dicts_unified,
        )
        # Connect to Snowflake on rank 0 and get internal stage credentials
        # Note: Identical to the initialization code in df.to_sql()
        (
            cursor,
            tmp_folder,
            stage_name,
            stage_path,
            upload_using_snowflake_put,
            old_creds,
        ) = connect_and_get_upload_info_jit(conn)
        writer["cursor"] = cursor
        writer["tmp_folder"] = tmp_folder
        writer["stage_name"] = stage_name
        writer["stage_path"] = stage_path
        writer["upload_using_snowflake_put"] = upload_using_snowflake_put
        writer["old_creds"] = old_creds
        # Barrier ensures that internal stage exists before we upload files to it
        bodo.barrier()
        # Compute bucket region
        writer["bucket_region"] = bodo.io.fs_io.get_s3_bucket_region_wrapper(
            stage_path, _is_parallel
        )
        # Set up internal stage directory for COPY INTO
        writer["copy_into_dir"] = make_new_copy_into_dir(
            upload_using_snowflake_put, stage_path, _is_parallel
        )
        ev.finalize()
        return writer

    return impl_snowflake_writer_init


@infer_global(snowflake_writer_init)
class SnowflakeWriterInitInfer(AbstractTemplate):
    """Typer for snowflake_writer_init that returns writer type"""

    def generic(self, args, kws):
        snowflake_writer_type = SnowflakeWriterType()
        pysig = numba.core.utils.pysignature(snowflake_writer_init)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(snowflake_writer_type, *folded_args).replace(pysig=pysig)


SnowflakeWriterInitInfer._no_unliteral = True


@lower_builtin(snowflake_writer_init, types.VarArg(types.Any))
def lower_snowflake_writer_init(context, builder, sig, args):
    """lower snowflake_writer_init() using gen_snowflake_writer_init_impl above"""
    impl = gen_snowflake_writer_init_impl(sig.return_type, *sig.args)
    return context.compile_internal(builder, impl, sig, args)


def snowflake_writer_append_table_inner(
    writer, table, col_names_meta, is_last, iter, col_precisions_meta, create_table_meta
):  # pragma: no cover
    pass


@overload(snowflake_writer_append_table_inner)
def gen_snowflake_writer_append_table_impl_inner(
    writer, table, col_names_meta, is_last, iter, col_precisions_meta, create_table_meta
):  # pragma: no cover
    if not isinstance(writer, SnowflakeWriterType):  # pragma: no cover
        raise BodoError(
            f"snowflake_writer_append_table: Expected type SnowflakeWriterType "
            f"for `writer`, found {writer}"
        )
    if not isinstance(table, TableType):  # pragma: no cover
        raise BodoError(
            f"snowflake_writer_append_table: Expected type TableType "
            f"for `table`, found {table}"
        )
    if not is_overload_bool(is_last):  # pragma: no cover
        raise BodoError(
            f"snowflake_writer_append_table: Expected type boolean "
            f"for `is_last`, found {is_last}"
        )

    col_names_meta = (
        col_names_meta.instance_type
        if isinstance(col_names_meta, types.TypeRef)
        else col_names_meta
    )
    if not isinstance(col_names_meta, ColNamesMetaType):  # pragma: no cover
        raise BodoError(
            f"snowflake_writer_append_table: Expected type ColNamesMetaType "
            f"for `col_names_meta`, found {col_names_meta}"
        )
    if not isinstance(col_names_meta.meta, tuple):  # pragma: no cover
        raise BodoError(
            "snowflake_writer_append_table: Expected col_names_meta "
            "to contain a tuple of column names"
        )

    col_names_arr = pd.array(col_names_meta.meta)
    n_cols = len(col_names_meta)
    py_table_typ = table

    if col_precisions_meta == bodo.types.none:
        col_precisions_tup = None
    else:
        col_precisions_tup = unwrap_typeref(col_precisions_meta).meta
        if (
            not isinstance(col_precisions_tup, tuple)
            or len(col_precisions_tup) != n_cols
            or any(not isinstance(elem, int) for elem in col_precisions_tup)
        ):  # pragma: no cover
            raise BodoError(
                f"snowflake_writer_append_table: Expected col_precisions_meta "
                f"to contain a tuple of {n_cols} precision values as integers"
            )

    create_table_info = unwrap_typeref(create_table_meta)
    if not isinstance(
        create_table_info, bodo.utils.typing.CreateTableMetaType
    ):  # pragma: no cover
        raise BodoError(
            f"snowflake_writer_append_table: Expected type CreateTableMetaType "
            f"for `create_table_meta`, found {create_table_info}"
        )

    sf_schema = bodo.io.snowflake.gen_snowflake_schema(
        col_names_meta.meta, table.arr_types, col_precisions_tup
    )

    column_datatypes = dict(zip(col_names_meta.meta, table.arr_types))
    # Use default number of iterations for sync if not specified by user
    sync_iters = (
        bodo.default_stream_loop_sync_iters
        if bodo.stream_loop_sync_iters == -1
        else bodo.stream_loop_sync_iters
    )

    # This function must be called the same number of times on all ranks.
    # This is because we only execute COPY INTO commands from rank 0, so
    # all ranks must finish writing their respective files to Snowflake
    # internal stage and sync with rank 0 before it issues COPY INTO.
    def impl_snowflake_writer_append_table(
        writer,
        table,
        col_names_meta,
        is_last,
        iter,
        col_precisions_meta,
        create_table_meta,
    ):  # pragma: no cover
        if writer["finished"]:
            return True
        ev = tracing.Event(
            "snowflake_writer_append_table", is_parallel=writer["parallel"]
        )
        is_last = bodo.libs.distributed_api.sync_is_last(is_last, iter)
        # ===== Part 1: Accumulate batch in writer and compute total size
        ev_append_batch = tracing.Event("append_batch", is_parallel=True)
        table_builder_state = writer["batches"]
        bodo.libs.table_builder.table_builder_append(table_builder_state, table)
        table_bytes = bodo.libs.table_builder.table_builder_nbytes(table_builder_state)
        ev_append_batch.add_attribute("table_bytes", table_bytes)
        ev_append_batch.finalize()
        # ===== Part 2: Write Parquet file if file size threshold is exceeded
        if is_last or table_bytes >= bodo.io.snowflake.SF_WRITE_PARQUET_CHUNK_SIZE:
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
                chunk_file_path = f"{writer['copy_into_dir']}/file{writer['file_count_local']}_rank{bodo.get_rank()}_{bodo.io.helpers.uuid4_helper()}.parquet"
                # Note: writer['stage_path'] already has trailing slash
                if (
                    writer["stage_path"].startswith("abfs://")
                    or writer["stage_path"].startswith("abfss://")
                ) and "?" in writer["stage_path"]:
                    # We need to move the query parameters to the end of the path
                    container_path, query = writer["stage_path"].split("?")
                    chunk_path = f"{container_path}{chunk_file_path}?{query}"
                else:
                    chunk_path = f"{writer['stage_path']}{chunk_file_path}"
                # To escape backslashes, we want to replace ( \ ) with ( \\ ), which can
                # be written as the string literals ( \\ ) and ( \\\\ ).
                # To escape quotes, we want to replace ( ' ) with ( \' ), which can
                # be written as the string literals ( ' ) and ( \\' ).
                chunk_path = chunk_path.replace("\\", "\\\\").replace("'", "\\'")
                # Copied from bodo.hiframes.pd_dataframe_ext.to_sql_overload
                # TODO: Refactor both sections to generate this code in a helper function
                ev_pq_write_cpp = tracing.Event("pq_write_cpp", is_parallel=False)
                ev_pq_write_cpp.add_attribute("out_table_len", out_table_len)
                ev_pq_write_cpp.add_attribute("chunk_idx", writer["file_count_local"])
                ev_pq_write_cpp.add_attribute("chunk_path", chunk_path)
                parquet_write_table_cpp(
                    unicode_to_utf8(chunk_path),
                    # Convert map columns to list(struct) before writing since Snowflake
                    # reads map as "key_value" rows that our flattening code cannot
                    # handle currently
                    cpp_table_map_to_list(
                        py_table_to_cpp_table(out_table, py_table_typ)
                    ),
                    array_to_info(col_names_arr),
                    unicode_to_utf8("null"),  # metadata
                    unicode_to_utf8(bodo.io.snowflake.SF_WRITE_PARQUET_COMPRESSION),
                    False,  # is_parallel
                    unicode_to_utf8(writer["bucket_region"]),
                    out_table_len,  # row_group_size
                    unicode_to_utf8("null"),  # prefix
                    True,  # Explicitly cast timedelta to int64
                    unicode_to_utf8("UTC"),  # Explicitly set tz='UTC'
                    True,  # Explicitly downcast nanoseconds to microseconds
                    False,  # Create write directory if not exists
                )
                ev_pq_write_cpp.finalize()
                # In case of Snowflake PUT, upload local parquet to internal stage
                # in a separate Python thread
                if writer["upload_using_snowflake_put"]:
                    cursor = writer["cursor"]
                    file_count_local = writer["file_count_local"]
                    stage_name = writer["stage_name"]
                    copy_into_dir = writer["copy_into_dir"]
                    with bodo.ir.object_mode.no_warning_objmode():
                        bodo.io.snowflake.do_upload_and_cleanup(
                            cursor,
                            file_count_local,
                            chunk_path,
                            stage_name,
                            copy_into_dir,
                        )
                writer["file_count_local"] += 1
                ev_upload_table.finalize()
            bodo.libs.table_builder.table_builder_reset(table_builder_state)
        # Count number of newly written files. This is also an implicit barrier
        # To reduce synchronization, we do this infrequently
        # Note: This requires append() to be called the same number of times on all ranks
        if writer["parallel"]:
            if is_last or (iter % sync_iters == 0):
                sum_op = np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)
                writer["file_count_global"] = bodo.libs.distributed_api.dist_reduce(
                    writer["file_count_local"], sum_op
                )
        else:
            writer["file_count_global"] = writer["file_count_local"]
        # ===== Part 3: Execute COPY INTO from Rank 0 if file count threshold is exceeded.
        # In case of Snowflake PUT, first wait for all upload threads to finish
        if (
            is_last
            or writer["file_count_global"]
            > bodo.io.snowflake.SF_WRITE_STREAMING_NUM_FILES
        ):
            # For the first COPY INTO, begin the transaction and create table if it doesn't exist
            if not writer["is_initialized"]:
                cursor = writer["cursor"]
                location = writer["location"]
                if_exists = writer["if_exists"]
                table_type = writer["table_type"]
                with bodo.ir.object_mode.no_warning_objmode():
                    begin_write_transaction(
                        cursor,
                        location,
                        sf_schema,
                        if_exists,
                        table_type,
                        create_table_info,
                    )
                writer["is_initialized"] = True
            # If an async COPY INTO command is in progress, retrieve and validate it.
            # Broadcast errors across ranks as needed.
            parallel = writer["parallel"]
            if (not parallel or bodo.get_rank() == 0) and writer[
                "copy_into_prev_sfqid"
            ] != "":
                cursor = writer["cursor"]
                copy_into_prev_sfqid = writer["copy_into_prev_sfqid"]
                file_count_global_prev = writer["file_count_global_prev"]
                with bodo.ir.object_mode.no_warning_objmode():
                    err = bodo.io.snowflake.retrieve_async_copy_into(
                        cursor, copy_into_prev_sfqid, file_count_global_prev
                    )
                    bodo.spawn.utils.sync_and_reraise_error(err, _is_parallel=parallel)
            else:
                with bodo.ir.object_mode.no_warning_objmode():
                    bodo.spawn.utils.sync_and_reraise_error(None, _is_parallel=parallel)
            # Execute async COPY INTO form rank 0
            if bodo.get_rank() == 0:
                cursor = writer["cursor"]
                stage_name = writer["stage_name"]
                location = writer["location"]
                copy_into_dir = writer["copy_into_dir"]
                flatten_table = writer["flatten_table"]
                with bodo.ir.object_mode.no_warning_objmode(
                    copy_into_new_sfqid="unicode_type",
                    flatten_sql="unicode_type",
                    flatten_table="unicode_type",
                ):
                    (
                        copy_into_new_sfqid,
                        flatten_sql,
                        flatten_table,
                    ) = bodo.io.snowflake.execute_copy_into(
                        cursor,
                        stage_name,
                        location,
                        sf_schema,
                        column_datatypes,
                        synchronous=False,
                        stage_dir=copy_into_dir,
                        flatten_table=flatten_table,
                        always_escape_col_names=True,
                    )
                writer["copy_into_prev_sfqid"] = copy_into_new_sfqid
                writer["flatten_sql"] = flatten_sql
                writer["flatten_table"] = flatten_table
                writer["file_count_global_prev"] = writer["file_count_global"]
                if bodo.user_logging.get_verbose_level() >= 2:
                    if writer["copy_into_sfqids_exists"]:
                        writer["copy_into_sfqids"] += f", {copy_into_new_sfqid}"
                    else:
                        writer["copy_into_sfqids_exists"] = True
                        writer["copy_into_sfqids"] = f"{copy_into_new_sfqid}"
            # Create a new COPY INTO internal stage directory
            writer["file_count_local"] = 0
            writer["file_count_global"] = 0
            writer["copy_into_dir"] = make_new_copy_into_dir(
                writer["upload_using_snowflake_put"],
                writer["stage_path"],
                writer["parallel"],
            )
        # ===== Part 4: Snowflake Post Handling
        # Retrieve and validate the last COPY INTO command
        if is_last:
            parallel = writer["parallel"]
            if (not parallel or bodo.get_rank() == 0) and writer[
                "copy_into_prev_sfqid"
            ] != "":
                cursor = writer["cursor"]
                copy_into_prev_sfqid = writer["copy_into_prev_sfqid"]
                file_count_global_prev = writer["file_count_global_prev"]
                with bodo.ir.object_mode.no_warning_objmode():
                    err = bodo.io.snowflake.retrieve_async_copy_into(
                        cursor, copy_into_prev_sfqid, file_count_global_prev
                    )
                    bodo.spawn.utils.sync_and_reraise_error(err, _is_parallel=parallel)
                    if flatten_sql == "":
                        cursor.execute(
                            "COMMIT /* io.snowflake_write.snowflake_writer_append_table() */"
                        )
            else:
                with bodo.ir.object_mode.no_warning_objmode():
                    bodo.spawn.utils.sync_and_reraise_error(None, _is_parallel=parallel)
            if (not parallel or bodo.get_rank() == 0) and writer["flatten_sql"] != "":
                cursor = writer["cursor"]
                flatten_sql = writer["flatten_sql"]
                with bodo.ir.object_mode.no_warning_objmode(
                    flatten_sfqid="unicode_type"
                ):
                    err = None
                    try:
                        # TODO: BSE-1929 call flatten_sql once for each copy into
                        #
                        # This assumes flatten_sql is the same for all batches, otherwise it would have to be run for every copy into.
                        cursor.execute(flatten_sql)
                        flatten_sfqid = cursor.sfqid
                    except Exception as e:
                        err = e
                    bodo.spawn.utils.sync_and_reraise_error(err, _is_parallel=parallel)
                    cursor.execute(
                        "COMMIT /* io.snowflake_write.snowflake_writer_append_table() */"
                    )
                if bodo.user_logging.get_verbose_level() >= 2:
                    if writer["copy_into_sfqids_exists"]:
                        writer["copy_into_sfqids"] = ", ".join(
                            [writer["copy_into_sfqids"], flatten_sfqid]
                        )
                    else:
                        writer["copy_into_sfqids_exists"] = True
                        writer["copy_into_sfqids"] = flatten_sfqid
            else:
                with bodo.ir.object_mode.no_warning_objmode():
                    bodo.spawn.utils.sync_and_reraise_error(None, _is_parallel=parallel)
            if bodo.get_rank() == 0:
                writer["copy_into_prev_sfqid"] = ""
                writer["flatten_sql"] = ""
                writer["flatten_table"] = ""
                writer["file_count_global_prev"] = 0
            # Drop internal stage, close Snowflake connection cursor, put back
            # environment variables, restore contents in case of ADLS stage
            cursor = writer["cursor"]
            stage_name = writer["stage_name"]
            old_creds = writer["old_creds"]
            tmp_folder = writer["tmp_folder"]
            with bodo.ir.object_mode.no_warning_objmode():
                if cursor is not None:
                    bodo.io.snowflake.drop_internal_stage(cursor, stage_name)
                    cursor.close()
                bodo.io.snowflake.update_env_vars(old_creds)
                tmp_folder.cleanup()
            if writer["parallel"]:
                bodo.barrier()
            writer["finished"] = True
            if bodo.user_logging.get_verbose_level() >= 2:
                bodo.user_logging.log_message(
                    "Snowflake Query Submission (Write)",
                    "/* async_execute_copy_into */ Snowflake Query IDs: "
                    + writer["copy_into_sfqids"],
                )
        ev.finalize()
        return is_last

    return impl_snowflake_writer_append_table


def snowflake_writer_append_table(
    writer, table, col_names_meta, is_last, iter, col_precisions_meta, create_table_meta
):  # pragma: no cover
    pass


@infer_global(snowflake_writer_append_table)
class SnowflakeWriterAppendInfer(AbstractTemplate):
    """Typer for snowflake_writer_append_table that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(snowflake_writer_append_table)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        # Update state type in signature to include build table type from input
        input_table_type = folded_args[1]
        new_state_type = SnowflakeWriterType(input_table_type)
        folded_args = (new_state_type, *folded_args[1:])
        return signature(types.bool_, *folded_args).replace(pysig=pysig)


SnowflakeWriterAppendInfer._no_unliteral = True


# Using a wrapper to keep snowflake_writer_append_table_inner as overload and avoid
# Numba objmode bugs (e.g. leftover ir.Del in IR leading to errors)
def impl_wrapper(
    writer, table, col_names_meta, is_last, iter, col_precisions_meta, create_table_meta
):  # pragma: no cover
    return snowflake_writer_append_table_inner(
        writer,
        table,
        col_names_meta,
        is_last,
        iter,
        col_precisions_meta,
        create_table_meta,
    )


@lower_builtin(snowflake_writer_append_table, types.VarArg(types.Any))
def lower_snowflake_writer_append_table(context, builder, sig, args):
    """lower snowflake_writer_append_table() using gen_snowflake_writer_append_table_impl above"""
    return context.compile_internal(builder, impl_wrapper, sig, args)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def make_new_copy_into_dir(
    upload_using_snowflake_put, stage_path, _is_parallel
):  # pragma: no cover
    """Generate a new COPY INTO directory using uuid4 and synchronize the
    result across ranks. This is intended to be called from every rank, as
    each rank's copy_into_dir will be created in a different TemporaryDirectory.
    All ranks share the same `copy_into_dir` suffix."""
    if not is_overload_bool(_is_parallel):  # pragma: no cover
        raise BodoError(
            f"make_new_copy_into_dir: Expected type boolean "
            f"for _is_parallel, found {_is_parallel}"
        )

    func_text = (
        "def impl(upload_using_snowflake_put, stage_path, _is_parallel):\n"
        "    copy_into_dir = ''\n"
        "    if not _is_parallel or bodo.get_rank() == 0:\n"
        "        copy_into_dir = bodo.io.helpers.uuid4_helper()\n"
        "    if _is_parallel:\n"
        "        copy_into_dir = bodo.libs.distributed_api.bcast_scalar(copy_into_dir)\n"
        # In case of upload using PUT, chunk_path is a local directory,
        # so it must be created. `makedirs_helper` is intended to be called
        # from all ranks at once, as each rank has a different TemporaryDirectory
        # and thus a different input `stage_path`.
        "    if upload_using_snowflake_put:\n"
        "        copy_into_path = stage_path + copy_into_dir\n"
        "        bodo.io.helpers.makedirs_helper(copy_into_path, exist_ok=True)\n"
        "    return copy_into_dir\n"
    )

    glbls = {
        "bodo": bodo,
    }

    l = {}
    exec(func_text, glbls, l)
    return l["impl"]
