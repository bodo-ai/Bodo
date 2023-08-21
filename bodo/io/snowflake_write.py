import operator
import os
import traceback

import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from mpi4py import MPI
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.extending import (
    box,
    intrinsic,
    models,
    overload,
    register_model,
    unbox,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import TableType
from bodo.io.helpers import exception_propagating_thread_type
from bodo.io.parquet_pio import parquet_write_table_cpp
from bodo.io.snowflake import (
    snowflake_connector_cursor_type,
    temporary_directory_type,
)
from bodo.libs.array import array_to_info, py_table_to_cpp_table
from bodo.libs.str_ext import unicode_to_utf8
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


class SnowflakeWriterType(types.Type):
    """Data type for streaming Snowflake writer's internal state"""

    def __init__(self, input_table_type=types.unknown):
        self.input_table_type = input_table_type
        super().__init__(name=f"SnowflakeWriterType({input_table_type})")


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
    # Snowflake query ID for previous COPY INTO command
    ("copy_into_prev_sfqid", types.unicode_type),
    # Flag indicating if the Snowflake transaction has started
    ("is_initialized", types.boolean),
    # File count for previous COPY INTO command
    ("file_count_global_prev", types.int64),
    # If we are using the PUT command to upload files, a list of upload
    # threads currently in progress
    ("upload_threads", types.List(exception_propagating_thread_type)),
    # Whether the `upload_threads` list exists. Needed for typing purposes,
    # as initializing an empty list in `init()` causes an error
    ("upload_threads_exists", types.boolean),
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
    # Whether the stage is ADLS backed and we'll be writing parquet files to it
    # directly using our existing HDFS and Parquet infrastructure
    ("azure_stage_direct_upload", types.boolean),
    # If azure_stage_direct_upload=True, we replace bodo.HDFS_CORE_SITE_LOC
    # with a new core-site.xml. `old_core_site` contains the original contents
    # of the file or "__none__" if file didn't originally exist, so that it
    # can be restored later after copy into
    ("old_core_site", types.unicode_type),
    # If azure_stage_direct_upload=True, we replace contents in
    # SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION if any with the SAS token for
    # this upload. `old_sas_token` contains the original contents of the file
    # or "__none__" if file didn't originally exist, so that it can be
    # restored later after copy into
    ("old_sas_token", types.unicode_type),
    # Batches collected to write
    ("batches", TableBuilderStateType()),
    # Uncompressed memory usage of batches
    ("curr_mem_size", types.int64),
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


def define_snowflake_writer_dtor(
    context, builder, snowflake_writer_type, payload_type
):  # pragma: no cover
    """
    Define destructor for Snowflake writer type if not already defined
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    fn = cgutils.get_or_insert_function(mod, fnty, name=".dtor.snowflake_writer")

    # End early if the dtor is already defined
    if not fn.is_declaration:
        return fn

    fn.linkage = "linkonce_odr"
    # Populate the dtor
    builder = lir.IRBuilder(fn.append_basic_block())
    base_ptr = fn.args[0]  # void*

    # Get payload struct
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_helper(builder, payload_type, ref=payload_ptr)

    # Decref each payload field
    for attr, fe_type in snowflake_writer_payload_members:
        context.nrt.decref(builder, fe_type, getattr(payload, attr))

    # Delete table builder state
    c_fnty = lir.FunctionType(
        lir.VoidType(),
        [lir.IntType(8).as_pointer()],
    )
    fn_tp = cgutils.get_or_insert_function(
        builder.module, c_fnty, name="delete_table_builder_state"
    )
    builder.call(fn_tp, [payload.batches])

    builder.ret_void()
    return fn


@intrinsic
def sf_writer_alloc(typingctx, expected_state_type_t):  # pragma: no cover
    expected_state_type = unwrap_typeref(expected_state_type_t)
    if is_overload_none(expected_state_type):
        snowflake_writer_type = SnowflakeWriterType()
    else:
        snowflake_writer_type = expected_state_type

    def codegen(context, builder, sig, args):  # pragma: no cover
        """Creates meminfo and sets dtor for Snowflake writer"""
        # Create payload type
        payload_type = snowflake_writer_payload_type
        alloc_type = context.get_value_type(payload_type)
        alloc_size = context.get_abi_sizeof(alloc_type)

        # Define dtor
        dtor_fn = define_snowflake_writer_dtor(
            context, builder, snowflake_writer_type, payload_type
        )

        # Create meminfo
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder, context.get_constant(types.uintp, alloc_size), dtor_fn
        )
        meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
        meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

        # Alloc values in payload. Note: garbage values will be stored in all
        # fields until sf_writer_setattr is called for the first time
        payload = cgutils.create_struct_proxy(payload_type)(context, builder)
        builder.store(payload._getvalue(), meminfo_data_ptr)

        # Construct Snowflake writer from payload
        snowflake_writer = context.make_helper(builder, snowflake_writer_type)
        snowflake_writer.meminfo = meminfo
        return snowflake_writer._getvalue()

    return snowflake_writer_type(expected_state_type_t), codegen


def _get_snowflake_writer_payload(
    context, builder, writer_typ, writer
):  # pragma: no cover
    """Get payload struct proxy for a Snowflake writer value"""
    snowflake_writer = context.make_helper(builder, writer_typ, writer)
    payload_type = snowflake_writer_payload_type
    meminfo_void_ptr = context.nrt.meminfo_data(builder, snowflake_writer.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_void_ptr, context.get_value_type(payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload, meminfo_data_ptr


@intrinsic
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
        payload, _ = _get_snowflake_writer_payload(context, builder, writer_typ, writer)
        return impl_ret_borrowed(
            context, builder, sig.return_type, getattr(payload, attr)
        )

    return val_typ(writer_typ, attr_typ), codegen


@intrinsic
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
        payload, meminfo_data_ptr = _get_snowflake_writer_payload(
            context, builder, writer_typ, writer
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
        f"Boxing is disabled for SnowflakeWriter mutable struct."
    )  # pragma: no cover


@unbox(SnowflakeWriterType)
def unbox_snowflake_writer(typ, val, c):
    raise NotImplementedError(
        f"Unboxing is disabled for SnowflakeWriter mutable struct."
    )  # pragma: no cover


def begin_write_transaction(cursor, location, sf_schema, if_exists, table_type):
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
                cursor, location, sf_schema, if_exists, table_type
            )
        except Exception as e:
            err = RuntimeError(str(e))
            if os.environ.get("BODO_SF_WRITE_DEBUG") is not None:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))

    err = comm.bcast(err)
    if isinstance(err, Exception):
        raise err


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def snowflake_writer_init(
    conn,
    table_name,
    schema,
    if_exists,
    table_type,
    expected_state_type=None,
    _is_parallel=False,
):  # pragma: no cover
    expected_state_type = unwrap_typeref(expected_state_type)
    if is_overload_none(expected_state_type):
        snowflake_writer_type = SnowflakeWriterType()
    else:
        snowflake_writer_type = expected_state_type

    table_builder_state_type = TableBuilderStateType(
        snowflake_writer_type.input_table_type
    )

    func_text = (
        "def impl(conn, table_name, schema, if_exists, table_type, expected_state_type=None, _is_parallel=False):\n"
        "    ev = tracing.Event('snowflake_writer_init', is_parallel=_is_parallel)\n"
        "    location = ''\n"
    )

    if not is_overload_none(schema):
        func_text += "    location += '\"' + schema + '\".'\n"

    func_text += (
        "    location += table_name\n"
        # Initialize writer
        "    writer = sf_writer_alloc(expected_state_type)\n"
        "    writer['conn'] = conn\n"
        "    writer['location'] = location\n"
        "    writer['if_exists'] = if_exists\n"
        "    writer['table_type'] = table_type\n"
        "    writer['parallel'] = _is_parallel\n"
        "    writer['finished'] = False\n"
        "    writer['file_count_local'] = 0\n"
        "    writer['file_count_global'] = 0\n"
        "    writer['copy_into_prev_sfqid'] = ''\n"
        "    writer['file_count_global_prev'] = 0\n"
        "    writer['upload_threads_exists'] = False\n"
        "    writer['batches'] = bodo.libs.table_builder.init_table_builder_state(table_builder_state_type)\n"
        "    writer['curr_mem_size'] = 0\n"
        # Connect to Snowflake on rank 0 and get internal stage credentials
        # Note: Identical to the initialization code in df.to_sql()
        "    with bodo.objmode(\n"
        "        cursor='snowflake_connector_cursor_type',\n"
        "        tmp_folder='temporary_directory_type',\n"
        "        stage_name='unicode_type',\n"
        "        stage_path='unicode_type',\n"
        "        upload_using_snowflake_put='boolean',\n"
        "        old_creds='DictType(unicode_type, unicode_type)',\n"
        "        azure_stage_direct_upload='boolean',\n"
        "        old_core_site='unicode_type',\n"
        "        old_sas_token='unicode_type',\n"
        "    ):\n"
        "        cursor, tmp_folder, stage_name, stage_path, upload_using_snowflake_put, old_creds, azure_stage_direct_upload, old_core_site, old_sas_token = bodo.io.snowflake.connect_and_get_upload_info(conn)\n"
        "    writer['cursor'] = cursor\n"
        "    writer['tmp_folder'] = tmp_folder\n"
        "    writer['stage_name'] = stage_name\n"
        "    writer['stage_path'] = stage_path\n"
        "    writer['upload_using_snowflake_put'] = upload_using_snowflake_put\n"
        "    writer['old_creds'] = old_creds\n"
        "    writer['azure_stage_direct_upload'] = azure_stage_direct_upload\n"
        "    writer['old_core_site'] = old_core_site\n"
        "    writer['old_sas_token'] = old_sas_token\n"
        # Barrier ensures that internal stage exists before we upload files to it
        "    bodo.barrier()\n"
        # Force reset the existing hadoop filesystem instance, to use new SAS token.
        # See to_sql() for more detailed comments
        "    if azure_stage_direct_upload:\n"
        "        bodo.libs.distributed_api.disconnect_hdfs_njit()\n"
        # Compute bucket region
        "    writer['bucket_region'] = bodo.io.fs_io.get_s3_bucket_region_njit(stage_path, _is_parallel)\n"
        # Set up internal stage directory for COPY INTO
        "    writer['copy_into_dir'] = make_new_copy_into_dir(\n"
        "        upload_using_snowflake_put, stage_path, _is_parallel\n"
        "    )\n"
        "    ev.finalize()\n"
        "    return writer\n"
    )

    # Passing in all globals is for some reason required for caching.
    glbls = globals()  # TODO: fix globals after Numba's #3355 is resolved
    glbls["table_builder_state_type"] = table_builder_state_type
    l = {}
    exec(func_text, glbls, l)
    return l["impl"]


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def snowflake_writer_append_table(
    writer, table, col_names_meta, is_last, iter
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
            f"snowflake_writer_append_table: Expected col_names_meta "
            f"to contain a tuple of column names"
        )

    col_names_arr = pd.array(col_names_meta.meta)
    sf_schema = bodo.io.snowflake.gen_snowflake_schema(
        col_names_meta.meta, table.arr_types
    )
    n_cols = len(col_names_meta)
    py_table_typ = table

    # This function must be called the same number of times on all ranks.
    # This is because we only execute COPY INTO commands from rank 0, so
    # all ranks must finish writing their respective files to Snowflake
    # internal stage and sync with rank 0 before it issues COPY INTO.
    def impl(writer, table, col_names_meta, is_last, iter):  # pragma: no cover
        if writer["finished"]:
            return True
        ev = tracing.Event(
            "snowflake_writer_append_table", is_parallel=writer["parallel"]
        )
        is_last = bodo.libs.distributed_api.sync_is_last(is_last, iter)
        # ===== Part 1: Accumulate batch in writer and compute total size
        ev_append_batch = tracing.Event(f"append_batch", is_parallel=True)
        table_builder_state = writer["batches"]
        bodo.libs.table_builder.table_builder_append(table_builder_state, table)
        nbytes_arr = np.empty(n_cols, np.int64)
        bodo.utils.table_utils.generate_table_nbytes(table, nbytes_arr, 0)
        nbytes = np.sum(nbytes_arr)
        writer["curr_mem_size"] += nbytes
        ev_append_batch.add_attribute("nbytes", nbytes)
        ev_append_batch.finalize()
        # ===== Part 2: Write Parquet file if file size threshold is exceeded
        if (
            is_last
            or writer["curr_mem_size"] >= bodo.io.snowflake.SF_WRITE_PARQUET_CHUNK_SIZE
        ):
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
                # Note: writer['stage_path'] already has trailing slash
                chunk_path = f'{writer["stage_path"]}{writer["copy_into_dir"]}/file{writer["file_count_local"]}_rank{bodo.get_rank()}_{bodo.io.helpers.uuid4_helper()}.parquet'
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
                    py_table_to_cpp_table(out_table, py_table_typ),
                    array_to_info(col_names_arr),
                    0,
                    False,  # write_index
                    unicode_to_utf8("null"),  # metadata
                    unicode_to_utf8(bodo.io.snowflake.SF_WRITE_PARQUET_COMPRESSION),
                    False,  # is_parallel
                    0,  # write_rangeindex_to_metadata
                    0,
                    0,
                    0,  # range index start, stop, step
                    unicode_to_utf8("null"),  # idx_name
                    unicode_to_utf8(writer["bucket_region"]),
                    out_table_len,  # row_group_size
                    unicode_to_utf8("null"),  # prefix
                    True,  # Explicitly cast timedelta to int64
                    unicode_to_utf8("UTC"),  # Explicitly set tz='UTC'
                    True,  # Explicitly downcast nanoseconds to microseconds
                )
                ev_pq_write_cpp.finalize()
                # In case of Snowflake PUT, upload local parquet to internal stage
                # in a separate Python thread
                if writer["upload_using_snowflake_put"]:
                    cursor = writer["cursor"]
                    file_count_local = writer["file_count_local"]
                    stage_name = writer["stage_name"]
                    copy_into_dir = writer["copy_into_dir"]
                    if bodo.io.snowflake.SF_WRITE_OVERLAP_UPLOAD:
                        with numba.objmode(
                            upload_thread="exception_propagating_thread_type"
                        ):
                            upload_thread = bodo.io.snowflake.do_upload_and_cleanup(
                                cursor,
                                file_count_local,
                                chunk_path,
                                stage_name,
                                copy_into_dir,
                            )
                        if writer["upload_threads_exists"]:
                            writer["upload_threads"].append(upload_thread)
                        else:
                            writer["upload_threads_exists"] = True
                            writer["upload_threads"] = [upload_thread]
                    else:
                        with bodo.objmode():
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
            writer["curr_mem_size"] = 0
        # Count number of newly written files. This is also an implicit barrier
        # To reduce synchronization, we do this infrequently
        # Note: This requires append() to be called the same number of times on all ranks
        if writer["parallel"]:
            if is_last or (iter % bodo.stream_loop_sync_iters == 0):
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
            if (
                writer["upload_using_snowflake_put"]
                and bodo.io.snowflake.SF_WRITE_OVERLAP_UPLOAD
            ):
                parallel = writer["parallel"]
                if writer["upload_threads_exists"]:
                    upload_threads = writer["upload_threads"]
                    with bodo.objmode():
                        bodo.io.helpers.join_all_threads(upload_threads, parallel)
                    writer["upload_threads"].clear()
                else:
                    with bodo.objmode():
                        bodo.io.helpers.join_all_threads([], parallel)
            # For the first COPY INTO, begin the transaction and create table if it doesn't exist
            if not writer["is_initialized"]:
                cursor = writer["cursor"]
                location = writer["location"]
                if_exists = writer["if_exists"]
                table_type = writer["table_type"]
                with bodo.objmode():
                    begin_write_transaction(
                        cursor, location, sf_schema, if_exists, table_type
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
                with bodo.objmode():
                    err = bodo.io.snowflake.retrieve_async_copy_into(
                        cursor, copy_into_prev_sfqid, file_count_global_prev
                    )
                    bodo.io.helpers.sync_and_reraise_error(err, _is_parallel=parallel)
            else:
                with bodo.objmode():
                    bodo.io.helpers.sync_and_reraise_error(None, _is_parallel=parallel)
            # Execute async COPY INTO form rank 0
            if bodo.get_rank() == 0:
                cursor = writer["cursor"]
                stage_name = writer["stage_name"]
                location = writer["location"]
                copy_into_dir = writer["copy_into_dir"]
                with numba.objmode(copy_into_new_sfqid="unicode_type"):
                    copy_into_new_sfqid = bodo.io.snowflake.execute_copy_into(
                        cursor,
                        stage_name,
                        location,
                        sf_schema,
                        synchronous=False,
                        stage_dir=copy_into_dir,
                    )
                writer["copy_into_prev_sfqid"] = copy_into_new_sfqid
                writer["file_count_global_prev"] = writer["file_count_global"]
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
                with bodo.objmode():
                    err = bodo.io.snowflake.retrieve_async_copy_into(
                        cursor, copy_into_prev_sfqid, file_count_global_prev
                    )
                    bodo.io.helpers.sync_and_reraise_error(err, _is_parallel=parallel)
                    cursor.execute(
                        "COMMIT /* io.snowflake_write.snowflake_writer_append_table() */"
                    )
            else:
                with bodo.objmode():
                    bodo.io.helpers.sync_and_reraise_error(None, _is_parallel=parallel)
            if bodo.get_rank() == 0:
                writer["copy_into_prev_sfqid"] = ""
                writer["file_count_global_prev"] = 0
            # Force reset the existing Hadoop filesystem instance to avoid
            # conflicts with any future ADLS operations in the same process
            if writer["azure_stage_direct_upload"]:
                bodo.libs.distributed_api.disconnect_hdfs_njit()
            # Drop internal stage, close Snowflake connection cursor, put back
            # environment variables, restore contents in case of ADLS stage
            cursor = writer["cursor"]
            stage_name = writer["stage_name"]
            old_creds = writer["old_creds"]
            tmp_folder = writer["tmp_folder"]
            azure_stage_direct_upload = writer["azure_stage_direct_upload"]
            old_core_site = writer["old_core_site"]
            old_sas_token = writer["old_sas_token"]
            with bodo.objmode():
                if cursor is not None:
                    bodo.io.snowflake.drop_internal_stage(cursor, stage_name)
                    cursor.close()
                bodo.io.snowflake.update_env_vars(old_creds)
                tmp_folder.cleanup()
                if azure_stage_direct_upload:
                    bodo.io.snowflake.update_file_contents(
                        bodo.HDFS_CORE_SITE_LOC, old_core_site
                    )
                    bodo.io.snowflake.update_file_contents(
                        bodo.io.snowflake.SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION,
                        old_sas_token,
                    )
            if writer["parallel"]:
                bodo.barrier()
            writer["finished"] = True
        ev.finalize()
        return is_last

    return impl


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
