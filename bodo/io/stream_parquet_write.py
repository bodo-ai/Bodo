import math
import operator

import numba
import pandas as pd
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
from bodo.io.parquet_pio import parquet_write_table_cpp, pq_write_create_dir
from bodo.libs.array import array_to_info, py_table_to_cpp_table
from bodo.libs.str_ext import unicode_to_utf8
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

# Maximum Parquet file size for streaming Parquet write
PARQUET_WRITE_CHUNK_SIZE = int(256e6)


class ParquetWriterType(types.Type):
    """Data type for streaming Parquet writer's internal state"""

    def __init__(self, input_table_type=types.unknown):
        self.input_table_type = input_table_type
        super().__init__(name=f"ParquetWriterType({input_table_type})")


class ParquetWriterPayloadType(types.Type):
    """Data type for streaming Parquet writer's payload"""

    def __init__(self):
        super().__init__(name="ParquetWriterPayloadType")


parquet_writer_payload_type = ParquetWriterPayloadType()


parquet_writer_payload_members = (
    # Path to write data
    ("path", types.unicode_type),
    # Parquet file compression scheme to use
    ("compression", types.unicode_type),
    # Row group size in Parquet files
    ("row_group_size", types.int64),
    # File name prefix for Parquet data files
    ("bodo_file_prefix", types.unicode_type),
    # Time zone to set in written files
    ("bodo_timestamp_tz", types.unicode_type),
    # S3 bucket region if path is S3
    ("bucket_region", types.unicode_type),
    # Whether write is occurring in parallel
    ("parallel", types.boolean),
    # Non-blocking is_last sync state (communicator, request, flags, ...)
    ("is_last_state", bodo.libs.distributed_api.is_last_state_type),
    # Whether this rank has finished appending data to the table
    ("finished", types.boolean),
    # Batches collected to write
    ("batches", TableBuilderStateType()),
)
parquet_writer_payload_members_dict = dict(parquet_writer_payload_members)


@register_model(ParquetWriterPayloadType)
class ParquetWriterPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):  # pragma: no cover
        members = parquet_writer_payload_members
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(ParquetWriterType)
class ParquetWriterModel(models.StructModel):
    def __init__(self, dmm, fe_type):  # pragma: no cover
        payload_type = parquet_writer_payload_type
        members = [
            ("meminfo", types.MemInfoPointer(payload_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@intrinsic
def parquet_writer_alloc(typingctx, expected_state_type_t):  # pragma: no cover
    expected_state_type = unwrap_typeref(expected_state_type_t)
    if is_overload_none(expected_state_type):
        parquet_writer_type = ParquetWriterType()
    else:
        parquet_writer_type = expected_state_type

    def codegen(context, builder, sig, args):  # pragma: no cover
        """Creates meminfo and sets dtor for Parquet writer"""
        return stream_writer_alloc_codegen(
            context,
            builder,
            parquet_writer_payload_type,
            parquet_writer_type,
            parquet_writer_payload_members,
        )

    return parquet_writer_type(expected_state_type_t), codegen


@intrinsic(prefer_literal=True)
def parquet_writer_getattr(typingctx, writer_typ, attr_typ):  # pragma: no cover
    """Get attribute of a Parquet writer"""
    assert isinstance(writer_typ, ParquetWriterType), (
        f"parquet_writer_getattr: expected `writer` to be a ParquetWriterType, "
        f"but found {writer_typ}"
    )
    assert is_overload_constant_str(attr_typ), (
        f"parquet_writer_getattr: expected `attr` to be a literal string type, "
        f"but found {attr_typ}"
    )
    attr = get_overload_const_str(attr_typ)
    val_typ = parquet_writer_payload_members_dict[attr]
    if attr == "batches":
        val_typ = TableBuilderStateType(writer_typ.input_table_type)

    def codegen(context, builder, sig, args):  # pragma: no cover
        writer, _ = args
        payload, _ = _get_stream_writer_payload(
            context, builder, writer_typ, parquet_writer_payload_type, writer
        )
        return impl_ret_borrowed(
            context, builder, sig.return_type, getattr(payload, attr)
        )

    return val_typ(writer_typ, attr_typ), codegen


@intrinsic(prefer_literal=True)
def parquet_writer_setattr(
    typingctx, writer_typ, attr_typ, val_typ
):  # pragma: no cover
    """Set attribute of a Parquet writer"""
    assert isinstance(writer_typ, ParquetWriterType), (
        f"parquet_writer_setattr: expected `writer` to be a ParquetWriterType, "
        f"but found {writer_typ}"
    )
    assert is_overload_constant_str(attr_typ), (
        f"parquet_writer_setattr: expected `attr` to be a literal string type, "
        f"but found {attr_typ}"
    )
    attr = get_overload_const_str(attr_typ)

    # Storing a literal type into the payload causes a type mismatch
    val_typ = numba.types.unliteral(val_typ)

    def codegen(context, builder, sig, args):  # pragma: no cover
        writer, _, val = args
        payload, meminfo_data_ptr = _get_stream_writer_payload(
            context, builder, writer_typ, parquet_writer_payload_type, writer
        )
        context.nrt.decref(builder, val_typ, getattr(payload, attr))
        context.nrt.incref(builder, val_typ, val)
        setattr(payload, attr, val)
        builder.store(payload._getvalue(), meminfo_data_ptr)
        return context.get_dummy_value()

    return types.none(writer_typ, attr_typ, val_typ), codegen


@overload(operator.getitem, no_unliteral=True)
def parquet_writer_getitem(writer, attr):
    if not isinstance(writer, ParquetWriterType):
        return

    return lambda writer, attr: parquet_writer_getattr(writer, attr)  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def parquet_writer_setitem(writer, attr, val):
    if not isinstance(writer, ParquetWriterType):
        return

    return lambda writer, attr, val: parquet_writer_setattr(
        writer, attr, val
    )  # pragma: no cover


@box(ParquetWriterType)
def box_parquet_writer(typ, val, c):
    # Boxing is disabled, to avoid boxing overheads anytime a writer attribute
    # is accessed from objmode. As a workaround, store the necessary attributes
    # into local variables in numba native code before entering objmode
    raise NotImplementedError(
        f"Boxing is disabled for ParquetWriter mutable struct."
    )  # pragma: no cover


@unbox(ParquetWriterType)
def unbox_parquet_writer(typ, val, c):
    raise NotImplementedError(
        f"Unboxing is disabled for ParquetWriter mutable struct."
    )  # pragma: no cover


def parquet_writer_init(
    operator_id,
    path,
    compression,
    row_group_size,
    bodo_file_prefix,
    bodo_timestamp_tz,
    expected_state_type=None,
    input_dicts_unified=False,
    _is_parallel=False,
):  # pragma: no cover
    pass


def gen_parquet_writer_init_impl(
    operator_id,
    path,
    compression,
    row_group_size,
    bodo_file_prefix,
    bodo_timestamp_tz,
    expected_state_type=None,
    input_dicts_unified=False,
    _is_parallel=False,
):  # pragma: no cover
    """Initialize Parquet stream writer"""

    expected_state_type = unwrap_typeref(expected_state_type)
    if is_overload_none(expected_state_type):
        parquet_writer_type = ParquetWriterType()
    else:
        parquet_writer_type = expected_state_type

    table_builder_state_type = TableBuilderStateType(
        parquet_writer_type.input_table_type
    )

    def impl_parquet_writer_init(
        operator_id,
        path,
        compression,
        row_group_size,
        bodo_file_prefix,
        bodo_timestamp_tz,
        expected_state_type=None,
        input_dicts_unified=False,
        _is_parallel=False,
    ):
        ev = tracing.Event("parquet_writer_init", is_parallel=_is_parallel)

        bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(
            path, parallel=_is_parallel
        )
        pq_write_create_dir(unicode_to_utf8(path))

        # Initialize writer
        writer = parquet_writer_alloc(expected_state_type)

        writer["path"] = path
        writer["compression"] = compression
        writer["row_group_size"] = row_group_size
        writer["bodo_file_prefix"] = bodo_file_prefix
        writer["bodo_timestamp_tz"] = bodo_timestamp_tz
        writer["bucket_region"] = bucket_region
        writer["parallel"] = _is_parallel
        writer["finished"] = False
        writer["is_last_state"] = bodo.libs.distributed_api.init_is_last_state()
        writer["batches"] = bodo.libs.table_builder.init_table_builder_state(
            operator_id,
            table_builder_state_type,
            input_dicts_unified=input_dicts_unified,
        )

        bodo.barrier()
        ev.finalize()
        return writer

    return impl_parquet_writer_init


@infer_global(parquet_writer_init)
class ParquetWriterInitInfer(AbstractTemplate):
    """Typer for parquet_writer_init that returns writer type"""

    def generic(self, args, kws):
        kws = dict(kws)
        expected_state_type = get_call_expr_arg(
            "parquet_writer_init",
            args,
            kws,
            6,
            "expected_state_type",
            default=types.none,
        )
        expected_state_type = unwrap_typeref(expected_state_type)
        if is_overload_none(expected_state_type):
            parquet_writer_type = ParquetWriterType()
        else:
            parquet_writer_type = expected_state_type

        pysig = numba.core.utils.pysignature(parquet_writer_init)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(parquet_writer_type, *folded_args).replace(pysig=pysig)


ParquetWriterInitInfer._no_unliteral = True


@lower_builtin(parquet_writer_init, types.VarArg(types.Any))
def lower_parquet_writer_init(context, builder, sig, args):
    """lower parquet_writer_init() using gen_parquet_writer_init_impl above"""
    impl = gen_parquet_writer_init_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def get_fname_prefix(base_prefix, iter):  # pragma: no cover
    pass


@overload(get_fname_prefix)
def overload_get_fname_prefix(base_prefix, iter):
    """Generate a Parquet file name prefix for each iteration in such a way that
    iteration file names are lexicographically sorted. This allows the data to be read
    in the written order later.
    SQL rows don't have order but this is still desirable for printing and debugging
    purposes.

    We add zeros before iteration number similar to rank number in Parquet write:
    https://github.com/Bodo-inc/Bodo/blob/d28883f3cffe37dfc2e6e68e5779f327ea4a16b7/bodo/io/_fs_io.cpp#L66

    However, iteration numbers are unbounded and we don't know how many zeros to add.
    Therefore, we assume a max number initially but increase it if it was reached.
    This is effectively creating a new "batch". To make sure batches are
    lexicographically sorted, we add an extra character that is greater than all numbers
    to each batch.

    The generated prefixes are of the form (assuming MAX_ITER = 100):
    00, 01, .. 99, b0100 .. b9999, bb010000 ..

    Args:
        base_prefix (str): initial prefix for writing Parquet parts ("part-" by default)
        iter (int): streaming iteration number
    """

    def impl(base_prefix, iter):  # pragma: no cover
        with bodo.no_warning_objmode(out="unicode_type"):
            MAX_ITER = 1000
            n_max_digits = math.ceil(math.log10(MAX_ITER))

            # Number of prefix characters to add ("batch" number)
            n_prefix = 0 if iter == 0 else math.floor(math.log(iter, MAX_ITER))

            iter_str = str(iter)
            n_zeros = ((n_prefix + 1) * n_max_digits) - len(iter_str)
            iter_str = ("0" * n_zeros) + iter_str

            out = base_prefix + ("b" * n_prefix) + iter_str + "-"
        return out

    return impl


def parquet_writer_append_table_inner(
    writer, table, col_names_meta, local_is_last, iter
):  # pragma: no cover
    pass


@overload(parquet_writer_append_table_inner)
def gen_parquet_writer_append_table_impl_inner(
    writer,
    table,
    col_names_meta,
    local_is_last,
    iter,
):  # pragma: no cover
    if not isinstance(writer, ParquetWriterType):  # pragma: no cover
        raise BodoError(
            f"parquet_writer_append_table: Expected type ParquetWriterType "
            f"for `writer`, found {writer}"
        )
    if not isinstance(table, TableType):  # pragma: no cover
        raise BodoError(
            f"parquet_writer_append_table: Expected type TableType "
            f"for `table`, found {table}"
        )
    if not is_overload_bool(local_is_last):  # pragma: no cover
        raise BodoError(
            f"parquet_writer_append_table: Expected type boolean "
            f"for `local_is_last`, found {local_is_last}"
        )

    col_names_meta = unwrap_typeref(col_names_meta)
    if not isinstance(col_names_meta, ColNamesMetaType):  # pragma: no cover
        raise BodoError(
            f"parquet_writer_append_table: Expected type ColNamesMetaType "
            f"for `col_names_meta`, found {col_names_meta}"
        )
    if not isinstance(col_names_meta.meta, tuple):  # pragma: no cover
        raise BodoError(
            f"parquet_writer_append_table: Expected col_names_meta "
            f"to contain a tuple of column names"
        )

    py_table_typ = table
    col_names_arr = pd.array(col_names_meta.meta)

    def impl_parquet_writer_append_table(
        writer, table, col_names_meta, local_is_last, iter
    ):  # pragma: no cover
        if writer["finished"]:
            return True
        ev = tracing.Event(
            "parquet_writer_append_table", is_parallel=writer["parallel"]
        )

        # ===== Part 1: Accumulate batch in writer and compute total size
        ev_append_batch = tracing.Event(f"append_batch", is_parallel=True)
        table_builder_state = writer["batches"]
        bodo.libs.table_builder.table_builder_append(table_builder_state, table)
        table_bytes = bodo.libs.table_builder.table_builder_nbytes(table_builder_state)
        ev_append_batch.add_attribute("table_bytes", table_bytes)
        ev_append_batch.finalize()

        is_last = bodo.libs.distributed_api.sync_is_last_non_blocking(
            writer["is_last_state"], local_is_last
        )

        # ===== Part 2: Write Parquet file if file size threshold is exceeded
        if is_last or table_bytes >= PARQUET_WRITE_CHUNK_SIZE:
            # Note: Our write batches are at least as large as our read batches. It may
            # be advantageous in the future to split up large incoming batches into
            # multiple Parquet files to write.

            # NOTE: table_builder_reset() below affects the table builder state so
            # out_table should be used immediately and not be stored.
            out_table = bodo.libs.table_builder.table_builder_get_data(
                table_builder_state
            )
            out_table_len = len(out_table)
            fname_prefix = get_fname_prefix(writer["bodo_file_prefix"], iter)

            # Write only on rank 0 for replicated input. Since streaming write is used
            # only for SQL, replicated in this context means actually replicated data
            # (instead of independent sequential functions with different data).
            if out_table_len > 0 and (writer["parallel"] or bodo.get_rank() == 0):
                ev_pq_write_cpp = tracing.Event("pq_write_cpp", is_parallel=False)
                ev_pq_write_cpp.add_attribute("out_table_len", out_table_len)
                parquet_write_table_cpp(
                    unicode_to_utf8(writer["path"]),
                    py_table_to_cpp_table(out_table, py_table_typ),
                    array_to_info(col_names_arr),
                    0,
                    # write_index
                    False,
                    # metadata
                    unicode_to_utf8("null"),
                    unicode_to_utf8(writer["compression"]),
                    # Set parallel=True even in replicated case since streaming write
                    # requires a directory to write multiple pieces.
                    True,
                    # write_rangeindex_to_metadata
                    0,
                    # range index start, stop, step
                    0,
                    0,
                    0,
                    # idx_name
                    unicode_to_utf8("null"),
                    unicode_to_utf8(writer["bucket_region"]),
                    writer["row_group_size"],
                    # prefix
                    unicode_to_utf8(fname_prefix),
                    # Explicitly cast timedelta to int64
                    False,
                    # Set timezone
                    unicode_to_utf8(writer["bodo_timestamp_tz"]),
                    # Explicitly downcast nanoseconds to microseconds
                    False,
                    # Create write directory if not exists (directory already created in
                    # writer init)
                    False,
                )
                ev_pq_write_cpp.finalize()
            bodo.libs.table_builder.table_builder_reset(table_builder_state)

        if is_last:
            writer["finished"] = True

        ev.finalize()
        return is_last

    return impl_parquet_writer_append_table


def parquet_writer_append_table(
    writer, table, col_names_meta, local_is_last, iter
):  # pragma: no cover
    """Append data batch to Parquet write buffer and write to output if necessary.

    Args:
        writer (ParquetWriterType): streaming Parquet writer object
        table (TableType): input data batch
        col_names_meta (ColNamesMetaType): table column names
        local_is_last (bool): is last batch flag
        iter (int): iteration number
    """
    pass


@infer_global(parquet_writer_append_table)
class ParquetWriterAppendInfer(AbstractTemplate):
    """Typer for parquet_writer_append_table that returns bool as output type"""

    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(parquet_writer_append_table)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        return signature(types.bool_, *folded_args).replace(pysig=pysig)


ParquetWriterAppendInfer._no_unliteral = True


# Using a wrapper to keep parquet_writer_append_table_inner as overload and avoid
# Numba objmode bugs (e.g. leftover ir.Del in IR leading to errors)
def impl_wrapper(
    writer, table, col_names_meta, local_is_last, iter
):  # pragma: no cover
    return parquet_writer_append_table_inner(
        writer, table, col_names_meta, local_is_last, iter
    )


@lower_builtin(parquet_writer_append_table, types.VarArg(types.Any))
def lower_parquet_writer_append_table(context, builder, sig, args):
    """lower parquet_writer_append_table() using gen_parquet_writer_append_table_impl above"""
    return context.compile_internal(builder, impl_wrapper, sig, args)
