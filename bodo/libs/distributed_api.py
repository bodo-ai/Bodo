import datetime
import time
import warnings
from collections import defaultdict
from enum import Enum

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, ir_utils, types
from numba.core.typing import signature
from numba.core.typing.builtins import IndexValueType
from numba.core.typing.templates import AbstractTemplate, ConcreteTemplate, infer_global
from numba.extending import (
    intrinsic,
    lower_builtin,
    models,
    overload,
    register_jitable,
    register_model,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo

# Import compiler
import bodo.decorators  # isort:skip # noqa

from mpi4py import MPI

from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.datetime_timedelta_ext import timedelta_array_type
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.hiframes.time_ext import TimeArrayType
from bodo.libs import hdist
from bodo.libs.array import (
    array_info_type,
    array_to_info,
    cpp_table_to_py_table,
    delete_info,
    delete_table,
    info_to_array,
    py_table_to_cpp_table,
    table_type,
)
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayType,
    offset_type,
)
from bodo.libs.binary_arr_ext import binary_array_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.interval_arr_ext import IntervalArrayType
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType
from bodo.libs.str_arr_ext import (
    get_data_ptr,
    get_null_bitmap_ptr,
    get_offset_ptr,
    num_total_chars,
    pre_alloc_string_array,
    set_bit_to,
    string_array_type,
)
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    ColNamesMetaType,
    ExternalFunctionErrorChecked,
    decode_if_dict_array,
    is_overload_false,
    is_overload_none,
    is_str_arr_type,
)
from bodo.utils.utils import (
    CTypeEnum,
    bodo_exec,
    cached_call_internal,
    check_and_propagate_cpp_exception,
    is_array_typ,
    numba_to_c_type,
)

ll.add_symbol("get_time", hdist.get_time)
ll.add_symbol("dist_reduce", hdist.dist_reduce)
ll.add_symbol("dist_arr_reduce", hdist.dist_arr_reduce)
ll.add_symbol("dist_exscan", hdist.dist_exscan)
ll.add_symbol("dist_irecv", hdist.dist_irecv)
ll.add_symbol("dist_isend", hdist.dist_isend)
ll.add_symbol("dist_wait", hdist.dist_wait)
ll.add_symbol("dist_get_item_pointer", hdist.dist_get_item_pointer)
ll.add_symbol("get_dummy_ptr", hdist.get_dummy_ptr)
ll.add_symbol("allgather", hdist.allgather)
ll.add_symbol("oneD_reshape_shuffle", hdist.oneD_reshape_shuffle)
ll.add_symbol("permutation_int", hdist.permutation_int)
ll.add_symbol("permutation_array_index", hdist.permutation_array_index)
ll.add_symbol("c_get_rank", hdist.dist_get_rank)
ll.add_symbol("c_get_size", hdist.dist_get_size)
ll.add_symbol("c_get_remote_size", hdist.dist_get_remote_size)
ll.add_symbol("c_barrier", hdist.barrier)
ll.add_symbol("c_gather_scalar", hdist.c_gather_scalar)
ll.add_symbol("c_gatherv", hdist.c_gatherv)
ll.add_symbol("c_scatterv", hdist.c_scatterv)
ll.add_symbol("c_allgatherv", hdist.c_allgatherv)
ll.add_symbol("c_bcast", hdist.c_bcast)
ll.add_symbol("c_recv", hdist.dist_recv)
ll.add_symbol("c_send", hdist.dist_send)
ll.add_symbol("timestamptz_reduce", hdist.timestamptz_reduce)
ll.add_symbol("_dist_transpose_comm", hdist._dist_transpose_comm)
ll.add_symbol("init_is_last_state", hdist.init_is_last_state)
ll.add_symbol("delete_is_last_state", hdist.delete_is_last_state)
ll.add_symbol("sync_is_last_non_blocking", hdist.sync_is_last_non_blocking)
ll.add_symbol("decimal_reduce", hdist.decimal_reduce)
ll.add_symbol("gather_table_py_entry", hdist.gather_table_py_entry)
ll.add_symbol("gather_array_py_entry", hdist.gather_array_py_entry)
ll.add_symbol("get_cpu_id", hdist.get_cpu_id)
ll.add_symbol("broadcast_array_py_entry", hdist.broadcast_array_py_entry)
ll.add_symbol("broadcast_table_py_entry", hdist.broadcast_table_py_entry)


DEFAULT_ROOT = 0


# Wrapper for getting process rank from C (MPI rank currently)
def get_rank():
    return hdist.get_rank_py_wrapper()


def get_size():
    return hdist.get_size_py_wrapper()


# XXX same as _distributed.h::BODO_ReduceOps::ReduceOpsEnum
class Reduce_Type(Enum):
    Sum = 0
    Prod = 1
    Min = 2
    Max = 3
    Argmin = 4
    Argmax = 5
    Bit_Or = 6
    Bit_And = 7
    Bit_Xor = 8
    Logical_Or = 9
    Logical_And = 10
    Logical_Xor = 11
    Concat = 12
    No_Op = 13


_barrier = types.ExternalFunction("c_barrier", types.int32())
_get_cpu_id = types.ExternalFunction("get_cpu_id", types.int32())
get_remote_size = types.ExternalFunction("c_get_remote_size", types.int32(types.int64))


@infer_global(get_rank)
class GetRankInfer(ConcreteTemplate):
    cases = [signature(types.int32)]


@lower_builtin(
    get_rank,
)
def lower_get_rank(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(32),
        [],
    )
    fn_typ = cgutils.get_or_insert_function(builder.module, fnty, name="c_get_rank")
    out = builder.call(fn_typ, args)
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return out


@infer_global(get_size)
class GetSizeInfer(ConcreteTemplate):
    cases = [signature(types.int32)]


@lower_builtin(
    get_size,
)
def lower_get_size(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.IntType(32),
        [],
    )
    fn_typ = cgutils.get_or_insert_function(builder.module, fnty, name="c_get_size")
    out = builder.call(fn_typ, args)
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return out


@numba.njit(cache=True)
def barrier():  # pragma: no cover
    """wrapper for barrier (MPI barrier currently)"""
    _barrier()


@overload(bodo.barrier)
def barrier_overload():  # pragma: no cover
    return lambda: _barrier()


@overload(bodo.get_rank)
def get_rank_overload():  # pragma: no cover
    return lambda: get_rank()


@overload(bodo.get_size)
def get_size_overload():  # pragma: no cover
    return lambda: get_size()


@numba.njit(cache=True)
def get_cpu_id():  # pragma: no cover
    """
    Wrapper for get_cpu_id -- get id of the cpu that the process
    is currently running on. This may change depending on if the
    process is pinned or not, the OS, etc.)
    This is not explicitly used anywhere, but is useful for
    checking if the processes are pinned as expected.
    """
    return _get_cpu_id()


@infer_global(time.time)
class TimeInfer(ConcreteTemplate):
    cases = [signature(types.float64)]


@lower_builtin(time.time)
def lower_time_time(context, builder, sig, args):
    _get_time = types.ExternalFunction("get_time", types.float64())

    return cached_call_internal(context, builder, lambda: _get_time(), sig, args)


@numba.generated_jit(nopython=True)
def get_type_enum(arr):
    arr = arr.instance_type if isinstance(arr, types.TypeRef) else arr
    dtype = arr.dtype
    if isinstance(dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
        dtype = bodo.hiframes.pd_categorical_ext.get_categories_int_type(dtype)

    typ_val = numba_to_c_type(dtype)
    return lambda arr: np.int32(typ_val)


_send = types.ExternalFunction(
    "c_send",
    types.void(types.voidptr, types.int32, types.int32, types.int32, types.int32),
)


@numba.njit(cache=True)
def send(val, rank, tag):  # pragma: no cover
    # dummy array for val
    send_arr = np.full(1, val)
    type_enum = get_type_enum(send_arr)
    _send(send_arr.ctypes, 1, type_enum, rank, tag)


_recv = types.ExternalFunction(
    "c_recv",
    types.void(types.voidptr, types.int32, types.int32, types.int32, types.int32),
)


@numba.njit(cache=True)
def recv(dtype, rank, tag):  # pragma: no cover
    # dummy array for val
    recv_arr = np.empty(1, dtype)
    type_enum = get_type_enum(recv_arr)
    _recv(recv_arr.ctypes, 1, type_enum, rank, tag)
    return recv_arr[0]


@numba.generated_jit(nopython=True)
def isend(arr, size, pe, tag, cond=True):
    """call MPI isend with input data"""
    # get size dynamically from C code (mpich 3.2 is 4 bytes but openmpi 1.6 is 8)
    mpi_req_numba_type = getattr(types, "int" + str(8 * hdist.mpi_req_num_bytes))
    _isend = types.ExternalFunction(
        "dist_isend",
        mpi_req_numba_type(
            types.voidptr,
            types.int32,
            types.int32,
            types.int32,
            types.int32,
            types.bool_,
        ),
    )

    # Numpy array
    if isinstance(arr, types.Array):

        def impl(arr, size, pe, tag, cond=True):  # pragma: no cover
            type_enum = get_type_enum(arr)
            return _isend(arr.ctypes, size, type_enum, pe, tag, cond)

        return impl

    # Primitive array
    if isinstance(arr, bodo.libs.primitive_arr_ext.PrimitiveArrayType):

        def impl(arr, size, pe, tag, cond=True):  # pragma: no cover
            np_arr = bodo.libs.primitive_arr_ext.primitive_to_np(arr)
            type_enum = get_type_enum(np_arr)
            return _isend(np_arr.ctypes, size, type_enum, pe, tag, cond)

        return impl

    if arr == boolean_array_type:
        # Nullable booleans need their own implementation because the
        # data array stores 1 bit per boolean. As a result, the data array
        # requires separate handling.
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_bool(arr, size, pe, tag, cond=True):  # pragma: no cover
            n_bytes = (size + 7) >> 3
            data_req = _isend(arr._data.ctypes, n_bytes, char_typ_enum, pe, tag, cond)
            null_req = _isend(
                arr._null_bitmap.ctypes, n_bytes, char_typ_enum, pe, tag, cond
            )
            return (data_req, null_req)

        return impl_bool

    # nullable arrays
    if (
        isinstance(
            arr,
            (
                IntegerArrayType,
                FloatingArrayType,
                DecimalArrayType,
                TimeArrayType,
                DatetimeArrayType,
            ),
        )
        or arr == datetime_date_array_type
        or arr == bodo.types.timedelta_array_type
    ):
        # return a tuple of requests for data and null arrays
        type_enum = np.int32(numba_to_c_type(arr.dtype))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_nullable(arr, size, pe, tag, cond=True):  # pragma: no cover
            n_bytes = (size + 7) >> 3
            data_req = _isend(arr._data.ctypes, size, type_enum, pe, tag, cond)
            null_req = _isend(
                arr._null_bitmap.ctypes, n_bytes, char_typ_enum, pe, tag, cond
            )
            return (data_req, null_req)

        return impl_nullable

    # TZ-Aware Timestamp arrays
    if isinstance(arr, DatetimeArrayType):

        def impl_tz_arr(arr, size, pe, tag, cond=True):  # pragma: no cover
            # Just send the underlying data. TZ info is all in the type.
            data_arr = arr._data
            type_enum = get_type_enum(data_arr)
            return _isend(data_arr.ctypes, size, type_enum, pe, tag, cond)

        return impl_tz_arr

    # string arrays
    if is_str_arr_type(arr) or arr == binary_array_type:
        offset_typ_enum = np.int32(numba_to_c_type(offset_type))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        # using blocking communication for string arrays instead since the array
        # slice passed in shift() may not stay alive (not a view of the original array)
        def impl_str_arr(arr, size, pe, tag, cond=True):  # pragma: no cover
            arr = decode_if_dict_array(arr)
            # send number of characters first
            n_chars = np.int64(bodo.libs.str_arr_ext.num_total_chars(arr))
            send(n_chars, pe, tag - 1)

            n_bytes = (size + 7) >> 3
            _send(
                bodo.libs.str_arr_ext.get_offset_ptr(arr),
                size + 1,
                offset_typ_enum,
                pe,
                tag,
            )
            _send(
                bodo.libs.str_arr_ext.get_data_ptr(arr), n_chars, char_typ_enum, pe, tag
            )
            _send(
                bodo.libs.str_arr_ext.get_null_bitmap_ptr(arr),
                n_bytes,
                char_typ_enum,
                pe,
                tag,
            )
            return None

        return impl_str_arr

    # voidptr input, pointer to bytes
    typ_enum = numba_to_c_type(types.uint8)

    def impl_voidptr(arr, size, pe, tag, cond=True):  # pragma: no cover
        return _isend(arr, size, typ_enum, pe, tag, cond)

    return impl_voidptr


@numba.generated_jit(nopython=True)
def irecv(arr, size, pe, tag, cond=True):  # pragma: no cover
    """post MPI irecv for array and return the request"""
    import bodo.libs.distributed_impl

    return bodo.libs.distributed_impl.irecv_impl(arr, size, pe, tag, cond)


@numba.njit(cache=True)
def gather_scalar(data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0):
    return gather_scalar_impl_jit(data, allgather, warn_if_rep, root, comm)


@numba.generated_jit(nopython=True)
def gather_scalar_impl_jit(
    data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
):
    data = types.unliteral(data)
    typ_val = numba_to_c_type(data)
    dtype = data

    c_gather_scalar = types.ExternalFunction(
        "c_gather_scalar",
        types.void(
            types.voidptr,
            types.voidptr,
            types.int32,
            types.bool_,
            types.int32,
            types.int64,
        ),
    )

    def gather_scalar_impl(
        data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
    ):  # pragma: no cover
        n_pes = bodo.libs.distributed_api.get_size()
        rank = bodo.libs.distributed_api.get_rank()
        is_receiver = rank == root
        if comm != 0:
            is_receiver = root == MPI.ROOT
            if is_receiver:
                n_pes = bodo.libs.distributed_api.get_remote_size(comm)

        send = np.full(1, data, dtype)
        res_size = n_pes if (is_receiver or allgather) else 0
        res = np.empty(res_size, dtype)
        c_gather_scalar(
            send.ctypes, res.ctypes, np.int32(typ_val), allgather, np.int32(root), comm
        )
        return res

    return gather_scalar_impl


@intrinsic
def value_to_ptr(typingctx, val_tp=None):
    """convert value to a pointer on stack
    WARNING: avoid using since pointers on stack cannot be passed around safely
    TODO[BSE-1399]: refactor uses and remove
    """

    def codegen(context, builder, sig, args):
        ptr = cgutils.alloca_once(builder, args[0].type)
        builder.store(args[0], ptr)
        return builder.bitcast(ptr, lir.IntType(8).as_pointer())

    return types.voidptr(val_tp), codegen


@intrinsic
def value_to_ptr_as_int64(typingctx, val_tp=None):
    def codegen(context, builder, sig, args):
        ptr = cgutils.alloca_once(builder, args[0].type)
        builder.store(args[0], ptr)
        void_star = builder.bitcast(ptr, lir.IntType(8).as_pointer())
        return builder.ptrtoint(void_star, lir.IntType(64))

    return types.int64(val_tp), codegen


@intrinsic
def load_val_ptr(typingctx, ptr_tp, val_tp=None):
    def codegen(context, builder, sig, args):
        ptr = builder.bitcast(args[0], args[1].type.as_pointer())
        return builder.load(ptr)

    return val_tp(ptr_tp, val_tp), codegen


@numba.njit(cache=True)
def dist_reduce(value, reduce_op, comm=0):
    return dist_reduce_impl(value, reduce_op, comm)


@numba.generated_jit(nopython=True)
def dist_reduce_impl(value, reduce_op, comm):
    if isinstance(value, types.Array):
        typ_enum = np.int32(numba_to_c_type(value.dtype))

        _dist_arr_reduce = types.ExternalFunction(
            "dist_arr_reduce",
            types.void(types.voidptr, types.int64, types.int32, types.int32),
        )

        def impl_arr(value, reduce_op, comm):  # pragma: no cover
            assert comm == 0, "dist_reduce_impl: intercomm not supported for arrays"
            A = np.ascontiguousarray(value)
            _dist_arr_reduce(A.ctypes, A.size, reduce_op, typ_enum)
            return A

        return impl_arr

    target_typ = types.unliteral(value)
    if isinstance(target_typ, IndexValueType):
        target_typ = target_typ.val_typ
        supported_typs = [
            types.bool_,
            types.uint8,
            types.int8,
            types.uint16,
            types.int16,
            types.uint32,
            types.int32,
            types.float32,
            types.float64,
            types.int64,
            bodo.types.datetime64ns,
            bodo.types.timedelta64ns,
            bodo.types.datetime_date_type,
            bodo.types.TimeType,
            bodo.types.pd_timedelta_type,
        ]

        if target_typ not in supported_typs and not isinstance(
            target_typ, (bodo.types.Decimal128Type, bodo.types.PandasTimestampType)
        ):  # pragma: no cover
            raise BodoError(f"argmin/argmax not supported for type {target_typ}")

    typ_enum = np.int32(numba_to_c_type(target_typ))

    if isinstance(target_typ, bodo.types.Decimal128Type):
        # For index-value types, the data pointed to has different amounts of padding depending on machine type.
        # as a workaround, we can pass the index separately.

        _decimal_reduce = types.ExternalFunction(
            "decimal_reduce",
            types.void(
                types.int64, types.voidptr, types.voidptr, types.int32, types.int32
            ),
        )

        if isinstance(types.unliteral(value), IndexValueType):

            def impl(value, reduce_op, comm):  # pragma: no cover
                assert comm == 0, (
                    "dist_reduce_impl: intercomm not supported for decimal"
                )
                if reduce_op in {Reduce_Type.Argmin.value, Reduce_Type.Argmax.value}:
                    in_ptr = value_to_ptr(value.value)
                    out_ptr = value_to_ptr(value)
                    _decimal_reduce(value.index, in_ptr, out_ptr, reduce_op, typ_enum)
                    return load_val_ptr(out_ptr, value)
                else:
                    raise BodoError(
                        "Only argmin/argmax/max/min scalar reduction is supported for Decimal"
                    )

        else:

            def impl(value, reduce_op, comm):  # pragma: no cover
                assert comm == 0, "dist_reduce_impl: intercomm not supported for arrays"
                if reduce_op in {Reduce_Type.Min.value, Reduce_Type.Max.value}:
                    in_ptr = value_to_ptr(value)
                    out_ptr = value_to_ptr(value)
                    _decimal_reduce(-1, in_ptr, out_ptr, reduce_op, typ_enum)
                    return load_val_ptr(out_ptr, value)
                else:
                    raise BodoError(
                        "Only argmin/argmax/max/min scalar reduction is supported for Decimal"
                    )

        return impl

    if isinstance(value, bodo.types.TimestampTZType):
        # This requires special handling because TimestampTZ's scalar
        # representation isn't the same as it's array representation - as such,
        # we need to extract the timestamp and offset separately, otherwise the
        # pointer passed into reduce will be a pointer to the following struct:
        #  struct {
        #      pd.Timestamp timestamp;
        #      int64_t offset;
        #  }
        # This is problematic since `timestamp` itself is a struct, and
        # extracting the right values is error-prone (and possibly not
        # portable).
        # TODO(aneesh): unify array and scalar representations of TimestampTZ to
        # avoid this.

        _timestamptz_reduce = types.ExternalFunction(
            "timestamptz_reduce",
            types.void(
                types.int64, types.int64, types.voidptr, types.voidptr, types.boolean
            ),
        )

        def impl(value, reduce_op, comm):  # pragma: no cover
            assert comm == 0, "dist_reduce_impl: intercomm not supported for arrays"
            if reduce_op not in {Reduce_Type.Min.value, Reduce_Type.Max.value}:
                raise BodoError(
                    "Only max/min scalar reduction is supported for TimestampTZ"
                )

            value_ts = value.utc_timestamp.value
            # using i64 for all numeric values
            out_ts_ptr = value_to_ptr(value_ts)
            out_offset_ptr = value_to_ptr(value_ts)
            _timestamptz_reduce(
                value.utc_timestamp.value,
                value.offset_minutes,
                out_ts_ptr,
                out_offset_ptr,
                reduce_op == Reduce_Type.Max.value,
            )
            out_ts = load_val_ptr(out_ts_ptr, value_ts)
            out_offset = load_val_ptr(out_offset_ptr, value_ts)
            return bodo.types.TimestampTZ(pd.Timestamp(out_ts), out_offset)

        return impl

    _dist_reduce = types.ExternalFunction(
        "dist_reduce",
        types.void(types.voidptr, types.voidptr, types.int32, types.int32, types.int64),
    )

    def impl(value, reduce_op, comm):  # pragma: no cover
        in_ptr = value_to_ptr(value)
        out_ptr = value_to_ptr(value)
        _dist_reduce(in_ptr, out_ptr, reduce_op, typ_enum, comm)
        return load_val_ptr(out_ptr, value)

    return impl


@numba.njit(cache=True)
def dist_exscan(value, reduce_op):
    return dist_exscan_impl(value, reduce_op)


@numba.generated_jit(nopython=True)
def dist_exscan_impl(value, reduce_op):
    target_typ = types.unliteral(value)
    typ_enum = np.int32(numba_to_c_type(target_typ))
    zero = target_typ(0)

    _dist_exscan = types.ExternalFunction(
        "dist_exscan",
        types.void(types.voidptr, types.voidptr, types.int32, types.int32),
    )

    def impl(value, reduce_op):  # pragma: no cover
        in_ptr = value_to_ptr(value)
        out_ptr = value_to_ptr(zero)
        _dist_exscan(in_ptr, out_ptr, reduce_op, typ_enum)
        return load_val_ptr(out_ptr, value)

    return impl


# from GetBit() in Arrow
@numba.njit(cache=True)
def get_bit(bits, i):  # pragma: no cover
    return (bits[i >> 3] >> (i & 0x07)) & 1


@numba.njit(cache=True)
def copy_gathered_null_bytes(
    null_bitmap_ptr, tmp_null_bytes, recv_counts_nulls, recv_counts
):  # pragma: no cover
    curr_tmp_byte = 0  # current location in buffer with all data
    curr_str = 0  # current string in output bitmap
    # for each chunk
    for i in range(len(recv_counts)):
        n_strs = recv_counts[i]
        n_bytes = recv_counts_nulls[i]
        chunk_bytes = tmp_null_bytes[curr_tmp_byte : curr_tmp_byte + n_bytes]
        # for each string in chunk
        for j in range(n_strs):
            set_bit_to(null_bitmap_ptr, curr_str, get_bit(chunk_bytes, j))
            curr_str += 1

        curr_tmp_byte += n_bytes


def gatherv(data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=None):
    """Gathers data from all ranks to root."""
    from mpi4py import MPI

    import bodo.libs.distributed_impl
    from bodo.libs.distributed_impl import gatherv_impl_wrapper

    if allgather and comm is not None:
        raise BodoError("gatherv(): allgather flag not supported in intercomm case")

    # Get data type on receiver in case of intercomm (since doesn't have any local data)
    rank = bodo.libs.distributed_api.get_rank()
    if comm is not None:
        # Receiver has to set root to MPI.ROOT in case of intercomm
        is_receiver = root == MPI.ROOT
        # Get data type in receiver
        if is_receiver:
            dtype = comm.recv(source=0, tag=11)
            data = get_value_for_type(dtype)
        elif rank == 0:
            dtype = bodo.typeof(data)
            comm.send(dtype, dest=0, tag=11)

    # Pass Comm pointer to native code (0 means not provided).
    if comm is None:
        comm_ptr = 0
    else:
        comm_ptr = MPI._addressof(comm)

    return gatherv_impl_wrapper(data, allgather, warn_if_rep, root, comm_ptr)


@overload(bodo.gatherv)
@overload(gatherv)
def gatherv_overload(
    data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
):
    """support gatherv inside jit functions"""
    from bodo.libs.distributed_impl import gatherv_impl_jit

    return (
        lambda data,
        allgather=False,
        warn_if_rep=True,
        root=DEFAULT_ROOT,
        comm=0: gatherv_impl_jit(data, allgather, warn_if_rep, root, comm)
    )  # pragma: no cover


def distributed_transpose(arr):  # pragma: no cover
    pass


@overload(distributed_transpose)
def overload_distributed_transpose(arr):
    """Implements distributed array transpose. First lays out data in contiguous chunks
    and calls alltoallv, and then transposes the output of alltoallv.
    See here for example code with similar algorithm:
    https://docs.oracle.com/cd/E19061-01/hpc.cluster5/817-0090-10/1-sided.html
    """
    assert isinstance(arr, types.Array) and arr.ndim == 2, (
        "distributed_transpose: 2D array expected"
    )
    c_type = numba_to_c_type(arr.dtype)
    _dist_transpose_comm = types.ExternalFunction(
        "_dist_transpose_comm",
        types.void(types.voidptr, types.voidptr, types.int32, types.int64, types.int64),
    )

    def impl(arr):  # pragma: no cover
        n_loc_rows, n_cols = arr.shape
        n_rows = bodo.libs.distributed_api.dist_reduce(
            n_loc_rows, np.int32(Reduce_Type.Sum.value)
        )
        n_out_cols = n_rows

        rank = bodo.libs.distributed_api.get_rank()
        n_pes = bodo.libs.distributed_api.get_size()
        n_out_loc_rows = bodo.libs.distributed_api.get_node_portion(n_cols, n_pes, rank)

        # Output of alltoallv is transpose of final output
        out_arr = np.empty((n_out_cols, n_out_loc_rows), arr.dtype)

        # Fill send buffer with contiguous data chunks for target ranks
        send_buff = np.empty(arr.size, arr.dtype)
        curr_ind = 0
        for p in range(n_pes):
            start = bodo.libs.distributed_api.get_start(n_cols, n_pes, p)
            count = bodo.libs.distributed_api.get_node_portion(n_cols, n_pes, p)
            for i in range(n_loc_rows):
                for j in range(start, start + count):
                    send_buff[curr_ind] = arr[i, j]
                    curr_ind += 1

        _dist_transpose_comm(
            out_arr.ctypes, send_buff.ctypes, np.int32(c_type), n_loc_rows, n_cols
        )

        # Keep the output in Fortran layout to match output Numba type of original
        # transpose IR statement being replaced in distributed pass.
        return out_arr.T

    return impl


@numba.njit(cache=True)
def rebalance(data, dests=None, random=False, random_seed=None, parallel=False):
    return rebalance_impl(data, dests, random, random_seed, parallel)


@overload(bodo.rebalance)
def rebalance_overload(
    data, dests=None, random=False, random_seed=None, parallel=False
):
    return (
        lambda data,
        dests=None,
        random=False,
        random_seed=None,
        parallel=False: rebalance_impl(data, dests, random, random_seed, parallel)
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def rebalance_impl(data, dests=None, random=False, random_seed=None, parallel=False):
    bodo.hiframes.pd_dataframe_ext.check_runtime_cols_unsupported(
        data, "bodo.rebalance()"
    )
    func_text = (
        "def impl(data, dests=None, random=False, random_seed=None, parallel=False):\n"
    )
    func_text += "    if random:\n"
    func_text += "        if random_seed is None:\n"
    func_text += "            random = 1\n"
    func_text += "        else:\n"
    func_text += "            random = 2\n"
    func_text += "    if random_seed is None:\n"
    func_text += "        random_seed = -1\n"
    # dataframe case, create a table and pass to C++
    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        df = data
        n_cols = len(df.columns)
        for i in range(n_cols):
            func_text += f"    data_{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {i})\n"
        func_text += "    ind_arr = bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data))\n"
        data_args = ", ".join(f"data_{i}" for i in range(n_cols))
        func_text += "    info_list_total = [{}, array_to_info(ind_arr)]\n".format(
            ", ".join(f"array_to_info(data_{x})" for x in range(n_cols))
        )
        func_text += "    table_total = arr_info_list_to_table(info_list_total)\n"
        # NOTE: C++ will delete table pointer
        func_text += "    if dests is None:\n"
        func_text += "        out_table = shuffle_renormalization(table_total, random, random_seed, parallel)\n"
        func_text += "    else:\n"
        func_text += "        out_table = shuffle_renormalization_group(table_total, random, random_seed, parallel, len(dests), np.array(dests, dtype=np.int32).ctypes)\n"
        for i_col in range(n_cols):
            func_text += f"    out_arr_{i_col} = array_from_cpp_table(out_table, {i_col}, data_{i_col})\n"
        func_text += (
            f"    out_arr_index = array_from_cpp_table(out_table, {n_cols}, ind_arr)\n"
        )
        func_text += "    delete_table(out_table)\n"
        data_args = ", ".join(f"out_arr_{i}" for i in range(n_cols))
        index = "bodo.utils.conversion.index_from_array(out_arr_index)"
        func_text += f"    return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args},), {index}, __col_name_meta_value_rebalance)\n"
    # Series case, create a table and pass to C++
    elif isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):
        func_text += "    data_0 = bodo.hiframes.pd_series_ext.get_series_data(data)\n"
        func_text += "    ind_arr = bodo.utils.conversion.index_to_array(bodo.hiframes.pd_series_ext.get_series_index(data))\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(data)\n"
        func_text += "    table_total = arr_info_list_to_table([array_to_info(data_0), array_to_info(ind_arr)])\n"
        # NOTE: C++ will delete table pointer
        func_text += "    if dests is None:\n"
        func_text += "        out_table = shuffle_renormalization(table_total, random, random_seed, parallel)\n"
        func_text += "    else:\n"
        func_text += "        out_table = shuffle_renormalization_group(table_total, random, random_seed, parallel, len(dests), np.array(dests, dtype=np.int32).ctypes)\n"
        func_text += "    out_arr_0 = array_from_cpp_table(out_table, 0, data_0)\n"
        func_text += "    out_arr_index = array_from_cpp_table(out_table, 1, ind_arr)\n"
        func_text += "    delete_table(out_table)\n"
        index = "bodo.utils.conversion.index_from_array(out_arr_index)"
        func_text += f"    return bodo.hiframes.pd_series_ext.init_series(out_arr_0, {index}, name)\n"
    # Numpy arrays, using dist_oneD_reshape_shuffle since numpy arrays can be multi-dim
    elif isinstance(data, types.Array):
        assert is_overload_false(random), "Call random_shuffle instead of rebalance"
        func_text += "    if not parallel:\n"
        func_text += "        return data\n"
        func_text += "    dim0_global_size = bodo.libs.distributed_api.dist_reduce(data.shape[0], np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value))\n"
        func_text += "    if dests is None:\n"
        func_text += "        dim0_local_size = bodo.libs.distributed_api.get_node_portion(dim0_global_size, bodo.get_size(), bodo.get_rank())\n"
        func_text += "    elif bodo.get_rank() not in dests:\n"
        func_text += "        dim0_local_size = 0\n"
        func_text += "    else:\n"
        func_text += "        dim0_local_size = bodo.libs.distributed_api.get_node_portion(dim0_global_size, len(dests), dests.index(bodo.get_rank()))\n"
        func_text += "    out = np.empty((dim0_local_size,) + tuple(data.shape[1:]), dtype=data.dtype)\n"
        func_text += "    bodo.libs.distributed_api.dist_oneD_reshape_shuffle(out, data, dim0_global_size, dests)\n"
        func_text += "    return out\n"
    # other array types, create a table and pass to C++
    elif bodo.utils.utils.is_array_typ(data, False):
        func_text += "    table_total = arr_info_list_to_table([array_to_info(data)])\n"
        # NOTE: C++ will delete table pointer
        func_text += "    if dests is None:\n"
        func_text += "        out_table = shuffle_renormalization(table_total, random, random_seed, parallel)\n"
        func_text += "    else:\n"
        func_text += "        out_table = shuffle_renormalization_group(table_total, random, random_seed, parallel, len(dests), np.array(dests, dtype=np.int32).ctypes)\n"
        func_text += "    out_arr = array_from_cpp_table(out_table, 0, data)\n"
        func_text += "    delete_table(out_table)\n"
        func_text += "    return out_arr\n"
    else:
        raise BodoError(f"Type {data} not supported for bodo.rebalance")
    loc_vars = {}
    glbls = {
        "np": np,
        "bodo": bodo,
        "array_to_info": bodo.libs.array.array_to_info,
        "shuffle_renormalization": bodo.libs.array.shuffle_renormalization,
        "shuffle_renormalization_group": bodo.libs.array.shuffle_renormalization_group,
        "arr_info_list_to_table": bodo.libs.array.arr_info_list_to_table,
        "array_from_cpp_table": bodo.libs.array.array_from_cpp_table,
        "delete_table": bodo.libs.array.delete_table,
    }
    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        glbls.update({"__col_name_meta_value_rebalance": ColNamesMetaType(df.columns)})
    exec(
        func_text,
        glbls,
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


@numba.njit(cache=True)
def random_shuffle(data, seed=None, dests=None, n_samples=None, parallel=False):
    return random_shuffle_impl(data, seed, dests, n_samples, parallel)


@overload(bodo.random_shuffle)
def random_shuffle_overload(
    data, seed=None, dests=None, n_samples=None, parallel=False
):
    return (
        lambda data,
        seed=None,
        dests=None,
        n_samples=None,
        parallel=False: random_shuffle_impl(data, seed, dests, n_samples, parallel)
    )


@numba.generated_jit(nopython=True)
def random_shuffle_impl(data, seed=None, dests=None, n_samples=None, parallel=False):
    func_text = (
        "def impl(data, seed=None, dests=None, n_samples=None, parallel=False):\n"
    )
    if isinstance(data, types.Array):
        if not is_overload_none(dests):
            raise BodoError("not supported")
        func_text += "    if seed is None:\n"
        func_text += "        seed = bodo.libs.distributed_api.bcast_scalar(np.random.randint(0, 2**31))\n"
        func_text += "    np.random.seed(seed)\n"
        func_text += "    if not parallel:\n"
        func_text += "        data = data.copy()\n"
        func_text += "        np.random.shuffle(data)\n"
        if not is_overload_none(n_samples):
            func_text += "        data = data[:n_samples]\n"
        func_text += "        return data\n"
        func_text += "    else:\n"
        func_text += "        dim0_global_size = bodo.libs.distributed_api.dist_reduce(data.shape[0], np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value))\n"
        func_text += "        permutation = np.arange(dim0_global_size)\n"
        func_text += "        np.random.shuffle(permutation)\n"
        if not is_overload_none(n_samples):
            func_text += (
                "        n_samples = max(0, min(dim0_global_size, n_samples))\n"
            )
        else:
            func_text += "        n_samples = dim0_global_size\n"
        func_text += "        dim0_local_size = bodo.libs.distributed_api.get_node_portion(dim0_global_size, bodo.get_size(), bodo.get_rank())\n"
        func_text += "        dim0_output_size = bodo.libs.distributed_api.get_node_portion(n_samples, bodo.get_size(), bodo.get_rank())\n"
        func_text += "        output = np.empty((dim0_output_size,) + tuple(data.shape[1:]), dtype=data.dtype)\n"
        func_text += "        dtype_size = bodo.io.np_io.get_dtype_size(data.dtype)\n"
        func_text += "        bodo.libs.distributed_api.dist_permutation_array_index(output, dim0_global_size, dtype_size, data, permutation, len(permutation), n_samples)\n"
        func_text += "        return output\n"
    else:
        func_text += "    output = bodo.libs.distributed_api.rebalance(data, dests=dests, random=True, random_seed=seed, parallel=parallel)\n"
        # Add support for `n_samples` argument used in sklearn.utils.shuffle:
        # Since the output is already distributed, to avoid the need to
        # communicate across ranks, we take the first `n_samples // num_procs`
        # items from each rank. This differs from sklearn's implementation
        # of n_samples, which just takes the first n_samples items of the
        # output as in `output = output[:n_samples]`.
        if not is_overload_none(n_samples):
            # Compute local number of samples. E.g. for n_samples = 11 and
            # mpi_size = 3, ranks (0,1,2) would sample (4,4,3) items, respectively
            func_text += "    local_n_samples = bodo.libs.distributed_api.get_node_portion(n_samples, bodo.get_size(), bodo.get_rank())\n"
            func_text += "    output = output[:local_n_samples]\n"
        func_text += "    return output\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "bodo": bodo,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


@numba.njit(cache=True)
def allgatherv(data, warn_if_rep=True, root=DEFAULT_ROOT):
    return allgatherv_impl(data, warn_if_rep, root)


@numba.generated_jit(nopython=True)
def allgatherv_impl(data, warn_if_rep=True, root=DEFAULT_ROOT):
    return lambda data, warn_if_rep=True, root=DEFAULT_ROOT: gatherv(
        data, True, warn_if_rep, root
    )  # pragma: no cover


@overload(bodo.allgatherv)
def allgatherv_overload(data, warn_if_rep=True, root=DEFAULT_ROOT):
    """support bodo.allgatherv() inside jit functions"""
    return lambda data, warn_if_rep=True, root=DEFAULT_ROOT: gatherv(
        data, True, warn_if_rep, root
    )  # pragma: no cover


def _bcast_dtype(data, root=DEFAULT_ROOT, comm=None):
    """broadcast data type from rank 0 using mpi4py"""
    try:
        from mpi4py import MPI
    except ImportError:  # pragma: no cover
        raise BodoError("mpi4py is required for scatterv")

    if comm is None:
        comm = MPI.COMM_WORLD

    data = comm.bcast(data, root)
    return data


def _get_array_first_val_fix_decimal_dict(arr):
    """Get first value of array but make sure decimal array returns PyArrow scalar
    which preserves precision/scale (Pandas by default returns decimal.Decimal).
    Also makes sure dictionary-encoded string array returns DictStringSentinel to allow
    proper unboxing type inference.
    """

    from bodo.hiframes.boxing import DictStringSentinel

    assert len(arr) > 0, "_get_array_first_val_fix_decimal_dict: empty array"

    if isinstance(arr, pd.arrays.ArrowExtensionArray) and pa.types.is_decimal128(
        arr.dtype.pyarrow_dtype
    ):
        return arr._pa_array[0]

    if isinstance(arr, pd.arrays.ArrowExtensionArray) and pa.types.is_dictionary(
        arr.dtype.pyarrow_dtype
    ):
        return DictStringSentinel()

    return arr[0]


# skipping coverage since only called on multiple core case
def get_value_for_type(dtype, use_arrow_time=False):  # pragma: no cover
    """returns a value of type 'dtype' to enable calling an njit function with the
    proper input type.

    Args:
        dtype (types.Type): input data type
        use_arrow_time (bool, optional): Use Arrow time64 array for TimeArray input (limited to precision=9 cases, used in nested arrays). Defaults to False.
    """
    # object arrays like decimal array can't be empty since they are not typed so we
    # create all arrays with size of 1 to be consistent

    # numpy arrays
    if isinstance(dtype, types.Array):
        return np.zeros((1,) * dtype.ndim, numba.np.numpy_support.as_dtype(dtype.dtype))

    # string array
    if dtype == string_array_type:
        return pd.array(["A"], "string")

    if dtype == bodo.types.dict_str_arr_type:
        return pd.array(["a"], pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string())))

    if dtype == binary_array_type:
        return np.array([b"A"], dtype=object)

    # Int array
    if isinstance(dtype, IntegerArrayType):
        pd_dtype = "{}Int{}".format(
            "" if dtype.dtype.signed else "U", dtype.dtype.bitwidth
        )
        return pd.array([3], pd_dtype)

    # Float array
    if isinstance(dtype, FloatingArrayType):
        pd_dtype = f"Float{dtype.dtype.bitwidth}"
        return pd.array([3.0], pd_dtype)

    # bool array
    if dtype == boolean_array_type:
        return pd.array([True], "boolean")

    # Decimal array
    if isinstance(dtype, DecimalArrayType):
        return pd.array(
            [0], dtype=pd.ArrowDtype(pa.decimal128(dtype.precision, dtype.scale))
        )

    # date array
    if dtype == datetime_date_array_type:
        return np.array([datetime.date(2011, 8, 9)])

    # timedelta array
    if dtype == timedelta_array_type:
        # Use Arrow duration array to ensure pd.Index() below doesn't convert it to
        # a non-nullable numpy timedelta64 array (leading to parallel errors).
        return pd.array(
            [datetime.timedelta(33)], dtype=pd.ArrowDtype(pa.duration("ns"))
        )

    # Index types
    if bodo.hiframes.pd_index_ext.is_pd_index_type(dtype):
        name = get_value_for_type(dtype.name_typ)
        if isinstance(dtype, bodo.hiframes.pd_index_ext.RangeIndexType):
            return pd.RangeIndex(1, name=name)
        arr_type = bodo.utils.typing.get_index_data_arr_types(dtype)[0]
        arr = get_value_for_type(arr_type)
        if isinstance(dtype, bodo.types.PeriodIndexType):
            return pd.period_range(
                start="2023-01-01", periods=1, freq=dtype.freq, name=name
            )
        return pd.Index(arr, name=name)

    # MultiIndex index
    if isinstance(dtype, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        name = get_value_for_type(dtype.name_typ)
        names = tuple(get_value_for_type(t) for t in dtype.names_typ)
        arrs = tuple(get_value_for_type(t) for t in dtype.array_types)
        # convert pyarrow arrays to numpy to avoid errors in pd.MultiIndex.from_arrays
        arrs = tuple(a.to_numpy(False) if isinstance(a, pa.Array) else a for a in arrs)
        val = pd.MultiIndex.from_arrays(arrs, names=names)
        val.name = name
        return val

    # Series
    if isinstance(dtype, bodo.hiframes.pd_series_ext.SeriesType):
        name = get_value_for_type(dtype.name_typ)
        arr = get_value_for_type(dtype.data)
        index = get_value_for_type(dtype.index)
        return pd.Series(arr, index, name=name)

    # DataFrame
    if isinstance(dtype, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        arrs = tuple(get_value_for_type(t) for t in dtype.data)
        index = get_value_for_type(dtype.index)
        # Set column names separately since there could be duplicate names
        df = pd.DataFrame({f"{i}": A for i, A in enumerate(arrs)}, index)
        df.columns = dtype.columns
        return df

    # Table
    if isinstance(dtype, bodo.types.TableType):
        arrs = tuple(get_value_for_type(t) for t in dtype.arr_types)
        return bodo.hiframes.table.Table(arrs)

    # CategoricalArray
    if isinstance(dtype, CategoricalArrayType):
        # Using -1 for code since categories can be empty
        return pd.Categorical.from_codes(
            [-1], dtype.dtype.categories, dtype.dtype.ordered
        )

    # Tuple
    if isinstance(dtype, types.BaseTuple):
        return tuple(get_value_for_type(t) for t in dtype.types)

    # ArrayItemArray
    if isinstance(dtype, ArrayItemArrayType):
        pa_arr = pa.LargeListArray.from_arrays(
            [0, 1], get_value_for_type(dtype.dtype, True)
        )
        return pd.arrays.ArrowExtensionArray(pa_arr)

    # IntervalArray
    if isinstance(dtype, IntervalArrayType):
        arr_type = get_value_for_type(dtype.arr_type)
        return pd.arrays.IntervalArray([pd.Interval(arr_type[0], arr_type[0])])

    # DatetimeArray
    if isinstance(dtype, DatetimeArrayType):
        return pd.array(
            [pd.Timestamp("2024/1/1", tz=dtype.tz)],
            pd.ArrowDtype(pa.timestamp("ns", tz=dtype.tz)),
        )

    # TimestampTZ array
    if dtype == bodo.types.timestamptz_array_type:
        return np.array([bodo.types.TimestampTZ(pd.Timestamp(0), 0)])

    # TimeArray
    if isinstance(dtype, TimeArrayType):
        precision = dtype.precision
        if use_arrow_time:
            assert precision == 9, (
                "get_value_for_type: only nanosecond precision is supported for nested data"
            )
            return pd.array(
                [bodo.types.Time(3, precision=precision)],
                pd.ArrowDtype(pa.time64("ns")),
            )
        return np.array([bodo.types.Time(3, precision=precision)], object)

    # NullArray
    if dtype == bodo.types.null_array_type:
        return pd.arrays.ArrowExtensionArray(pa.nulls(1))

    # StructArray
    if isinstance(dtype, bodo.types.StructArrayType):
        # Handle empty struct corner case which can have typing issues
        if dtype == bodo.types.StructArrayType((), ()):
            return pd.array([{}], pd.ArrowDtype(pa.struct([])))

        pa_arr = pa.StructArray.from_arrays(
            tuple(get_value_for_type(t, True) for t in dtype.data), dtype.names
        )
        return pd.arrays.ArrowExtensionArray(pa_arr)

    # TupleArray
    if isinstance(dtype, bodo.types.TupleArrayType):
        # TODO[BSE-4213]: Use Arrow arrays
        return pd.array(
            [
                tuple(
                    _get_array_first_val_fix_decimal_dict(get_value_for_type(t))
                    for t in dtype.data
                )
            ],
            object,
        )._ndarray

    # MapArrayType
    if isinstance(dtype, bodo.types.MapArrayType):
        pa_arr = pa.MapArray.from_arrays(
            [0, 1],
            get_value_for_type(dtype.key_arr_type, True),
            get_value_for_type(dtype.value_arr_type, True),
        )
        return pd.arrays.ArrowExtensionArray(pa_arr)

    # Numpy Matrix
    if isinstance(dtype, bodo.types.MatrixType):
        return np.asmatrix(
            get_value_for_type(types.Array(dtype.dtype, 2, dtype.layout))
        )

    if isinstance(dtype, types.List):
        return [get_value_for_type(dtype.dtype)]

    if isinstance(dtype, types.DictType):
        return {
            get_value_for_type(dtype.key_type): get_value_for_type(dtype.value_type)
        }

    if dtype == bodo.types.string_type:
        # make names unique with next_label to avoid MultiIndex unboxing issue #811
        return "_" + str(ir_utils.next_label())

    if isinstance(dtype, types.StringLiteral):
        return dtype.literal_value

    if dtype == types.int64:
        return ir_utils.next_label()

    if dtype == types.none:
        return None

    # TODO: Add missing data types
    raise BodoError(f"get_value_for_type(dtype): Missing data type {dtype}")


def scatterv(data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=None):
    """scatterv() distributes data from rank 0 to all ranks.
    Rank 0 passes the data but the other ranks should just pass None.
    """
    from mpi4py import MPI

    import bodo.libs.distributed_impl
    from bodo.libs.distributed_impl import scatterv_impl

    rank = bodo.libs.distributed_api.get_rank()
    if rank != DEFAULT_ROOT and data is not None:  # pragma: no cover
        warnings.warn(
            BodoWarning(
                "bodo.scatterv(): A non-None value for 'data' was found on a rank other than the root. "
                "This data won't be sent to any other ranks and will be overwritten with data from rank 0."
            )
        )

    # make sure all ranks receive the proper data type as input (instead of None)
    dtype = bodo.typeof(data)
    dtype = _bcast_dtype(dtype, root, comm)

    is_sender = rank == root
    if comm is not None:
        # Sender has to set root to MPI.ROOT in case of intercomm
        is_sender = root == MPI.ROOT

    if not is_sender:
        data = get_value_for_type(dtype)

    # Pass Comm pointer to native code (0 means not provided).
    if comm is None:
        comm_ptr = 0
    else:
        comm_ptr = MPI._addressof(comm)

    return scatterv_impl(data, send_counts, warn_if_dist, root, comm_ptr)


@overload(bodo.scatterv)
@overload(scatterv)
def scatterv_overload(
    data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
):
    """support scatterv inside jit functions"""
    import bodo.libs.distributed_impl
    from bodo.libs.distributed_impl import scatterv_impl_jit

    bodo.hiframes.pd_dataframe_ext.check_runtime_cols_unsupported(
        data, "bodo.scatterv()"
    )
    return (
        lambda data,
        send_counts=None,
        warn_if_dist=True,
        root=DEFAULT_ROOT,
        comm=0: scatterv_impl_jit(data, send_counts, warn_if_dist, root, comm)
    )  # pragma: no cover


@intrinsic
def cptr_to_voidptr(typingctx, cptr_tp=None):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], lir.IntType(8).as_pointer())

    return types.voidptr(cptr_tp), codegen


def bcast_preallocated(data, root=DEFAULT_ROOT):  # pragma: no cover
    return


@overload(bcast_preallocated, no_unliteral=True)
def bcast_preallocated_overload(data, root=DEFAULT_ROOT):
    """broadcast array from root rank. 'data' array is assumed to be pre-allocated in
    non-root ranks.
    This is for limited internal use in kernels like rolling windows and also parallel
    index handling where output data type and length are known ahead of time in non-root
    ranks.
    Only supports basic numeric and string data types (e.g. no nested arrays).
    """
    INT_MAX = np.iinfo(np.int32).max

    # Numpy arrays
    if isinstance(data, types.Array):

        def bcast_impl(data, root=DEFAULT_ROOT):  # pragma: no cover
            typ_enum = get_type_enum(data)
            count = data.size
            assert count < INT_MAX
            c_bcast(data.ctypes, np.int32(count), typ_enum, np.int32(root), 0)

        return bcast_impl

    # Decimal arrays
    if isinstance(data, DecimalArrayType):

        def bcast_decimal_arr(data, root=DEFAULT_ROOT):  # pragma: no cover
            count = data._data.size
            assert count < INT_MAX
            c_bcast(
                data._data.ctypes,
                np.int32(count),
                CTypeEnum.Int128.value,
                np.int32(root),
                0,
            )
            bodo.libs.distributed_api.bcast_preallocated(data._null_bitmap, root)

        return bcast_decimal_arr

    # nullable int/float/bool/date/time arrays
    if isinstance(
        data, (IntegerArrayType, FloatingArrayType, TimeArrayType, DatetimeArrayType)
    ) or data in (
        boolean_array_type,
        datetime_date_array_type,
        bodo.types.timedelta_array_type,
    ):

        def bcast_impl_int_arr(data, root=DEFAULT_ROOT):  # pragma: no cover
            bodo.libs.distributed_api.bcast_preallocated(data._data, root)
            bodo.libs.distributed_api.bcast_preallocated(data._null_bitmap, root)

        return bcast_impl_int_arr

    # string arrays
    if is_str_arr_type(data) or data == binary_array_type:
        offset_typ_enum = np.int32(numba_to_c_type(offset_type))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def bcast_str_impl(data, root=DEFAULT_ROOT):  # pragma: no cover
            data = decode_if_dict_array(data)
            n_loc = len(data)
            n_all_chars = num_total_chars(data)
            assert n_loc < INT_MAX
            assert n_all_chars < INT_MAX

            offset_ptr = get_offset_ptr(data)
            data_ptr = get_data_ptr(data)
            null_bitmap_ptr = get_null_bitmap_ptr(data)
            n_bytes = (n_loc + 7) >> 3

            c_bcast(offset_ptr, np.int32(n_loc + 1), offset_typ_enum, np.int32(root), 0)
            c_bcast(data_ptr, np.int32(n_all_chars), char_typ_enum, np.int32(root), 0)
            c_bcast(
                null_bitmap_ptr, np.int32(n_bytes), char_typ_enum, np.int32(root), 0
            )

        return bcast_str_impl


# sendbuf, sendcount, dtype, root
c_bcast = types.ExternalFunction(
    "c_bcast",
    types.void(types.voidptr, types.int32, types.int32, types.int32, types.int64),
)


@numba.njit(cache=True)
def bcast_scalar(val, root=DEFAULT_ROOT, comm=0):
    """broadcast for a scalar value.
    Assumes all ranks `val` has same type.
    """
    return bcast_scalar_impl(val, root, comm)


def bcast_scalar_impl(val, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
    return


@infer_global(bcast_scalar_impl)
class BcastScalarInfer(AbstractTemplate):
    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(bcast_scalar_impl)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        assert len(folded_args) == 3
        val = args[0]

        if not (
            isinstance(
                val,
                (
                    types.Integer,
                    types.Float,
                    bodo.types.PandasTimestampType,
                ),
            )
            or val
            in [
                bodo.types.datetime64ns,
                bodo.types.timedelta64ns,
                bodo.types.string_type,
                types.none,
                types.bool_,
                bodo.types.datetime_date_type,
                bodo.types.timestamptz_type,
            ]
        ):
            raise BodoError(
                f"bcast_scalar requires an argument of type Integer, Float, datetime64ns, timestamptz, timedelta64ns, string, None, or Bool. Found type {val}"
            )

        return signature(val, *folded_args)


def gen_bcast_scalar_impl(val, root=DEFAULT_ROOT, comm=0):
    if val == types.none:
        return lambda val, root=DEFAULT_ROOT, comm=0: None

    if val == bodo.types.timestamptz_type:

        def impl(val, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
            updated_timestamp = bodo.libs.distributed_api.bcast_scalar(
                val.utc_timestamp, root, comm
            )
            updated_offset = bodo.libs.distributed_api.bcast_scalar(
                val.offset_minutes, root, comm
            )
            return bodo.types.TimestampTZ(updated_timestamp, updated_offset)

        return impl

    if val == bodo.types.datetime_date_type:
        c_type = numba_to_c_type(types.int32)

        # Note: There are issues calling this function with recursion.
        # As a result we just implement it directly.
        def impl(val, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
            send = np.empty(1, np.int32)
            send[0] = bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int(val)
            c_bcast(send.ctypes, np.int32(1), np.int32(c_type), np.int32(root), comm)
            return bodo.hiframes.datetime_date_ext.cast_int_to_datetime_date(send[0])

        return impl

    if isinstance(val, bodo.types.PandasTimestampType):
        c_type = numba_to_c_type(types.int64)
        tz = val.tz

        # Note: There are issues calling this function with recursion.
        # As a result we just implement it directly.
        def impl(val, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
            send = np.empty(1, np.int64)
            send[0] = val.value
            c_bcast(send.ctypes, np.int32(1), np.int32(c_type), np.int32(root), comm)
            # Use convert_val_to_timestamp to other modifying the value
            return pd.Timestamp(send[0], tz=tz)

        return impl

    if val == bodo.types.string_type:
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_str(val, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            is_sender = rank == root
            if comm != 0:
                is_sender = root == MPI.ROOT

            if not is_sender:
                n_char = 0
                utf8_str = np.empty(0, np.uint8).ctypes
            else:
                utf8_str, n_char = bodo.libs.str_ext.unicode_to_utf8_and_len(val)
            n_char = bodo.libs.distributed_api.bcast_scalar(n_char, root, comm)

            if not is_sender:
                # add null termination character
                utf8_str_arr = np.empty(n_char + 1, np.uint8)
                utf8_str_arr[n_char] = 0
                utf8_str = utf8_str_arr.ctypes
            c_bcast(utf8_str, np.int32(n_char), char_typ_enum, np.int32(root), comm)
            return bodo.libs.str_arr_ext.decode_utf8(utf8_str, n_char)

        return impl_str

    # TODO: other types like boolean
    typ_val = numba_to_c_type(val)
    dtype = numba.np.numpy_support.as_dtype(val)

    # TODO: fix np.full and refactor
    def bcast_scalar_impl(val, root=DEFAULT_ROOT, comm=0):
        send = np.empty(1, dtype)
        send[0] = val
        c_bcast(send.ctypes, np.int32(1), np.int32(typ_val), np.int32(root), comm)
        return send[0]

    return bcast_scalar_impl


@lower_builtin(bcast_scalar_impl, types.Any, types.VarArg(types.Any))
def bcast_scalar_impl_any(context, builder, sig, args):
    impl = gen_bcast_scalar_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@numba.njit(cache=True)
def bcast_tuple(val, root=DEFAULT_ROOT, comm=0):
    return bcast_tuple_impl_jit(val, root, comm)


@numba.generated_jit(nopython=True)
def bcast_tuple_impl_jit(val, root=DEFAULT_ROOT, comm=0):
    """broadcast a tuple value
    calls bcast_scalar() on individual elements
    """
    assert isinstance(val, types.BaseTuple), (
        "Internal Error: Argument to bcast tuple must be of type tuple"
    )
    n_elem = len(val)
    func_text = f"def bcast_tuple_impl(val, root={DEFAULT_ROOT}, comm=0):\n"
    func_text += "  return ({}{})".format(
        ",".join(f"bcast_scalar(val[{i}], root, comm)" for i in range(n_elem)),
        "," if n_elem else "",
    )

    loc_vars = {}
    exec(
        func_text,
        {"bcast_scalar": bcast_scalar},
        loc_vars,
    )
    bcast_tuple_impl = loc_vars["bcast_tuple_impl"]
    return bcast_tuple_impl


# if arr is string array, pre-allocate on non-root the same size as root
def prealloc_str_for_bcast(arr, root=DEFAULT_ROOT):  # pragma: no cover
    return arr


@overload(prealloc_str_for_bcast, no_unliteral=True)
def prealloc_str_for_bcast_overload(arr, root=DEFAULT_ROOT):
    if arr == string_array_type:

        def prealloc_impl(arr, root=DEFAULT_ROOT):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = bcast_scalar(len(arr), root)
            n_all_char = bcast_scalar(np.int64(num_total_chars(arr)), root)
            if rank != root:
                arr = pre_alloc_string_array(n_loc, n_all_char)
            return arr

        return prealloc_impl

    return lambda arr, root=DEFAULT_ROOT: arr


def get_local_slice(idx, arr_start, total_len):  # pragma: no cover
    return idx


@overload(
    get_local_slice,
    no_unliteral=True,
    jit_options={"cache": True, "no_cpython_wrapper": True},
)
def get_local_slice_overload(idx, arr_start, total_len):
    """get local slice of a global slice, using start of array chunk and total array
    length.
    """

    if not idx.has_step:  # pragma: no cover
        # Generate a separate implement if there
        # is no step so types match.
        def impl(idx, arr_start, total_len):  # pragma: no cover
            # normalize slice
            slice_index = numba.cpython.unicode._normalize_slice(idx, total_len)
            new_start = max(arr_start, slice_index.start) - arr_start
            new_stop = max(slice_index.stop - arr_start, 0)
            return slice(new_start, new_stop)

    else:

        def impl(idx, arr_start, total_len):  # pragma: no cover
            # normalize slice
            slice_index = numba.cpython.unicode._normalize_slice(idx, total_len)
            start = slice_index.start
            step = slice_index.step

            offset = (
                0
                if step == 1 or start > arr_start
                else (abs(step - (arr_start % step)) % step)
            )
            new_start = max(arr_start, slice_index.start) - arr_start + offset
            new_stop = max(slice_index.stop - arr_start, 0)
            return slice(new_start, new_stop, step)

    return impl


def slice_getitem(arr, slice_index, arr_start, total_len):  # pragma: no cover
    return arr[slice_index]


@overload(slice_getitem, no_unliteral=True, jit_options={"cache": True})
def slice_getitem_overload(arr, slice_index, arr_start, total_len):
    def getitem_impl(arr, slice_index, arr_start, total_len):  # pragma: no cover
        new_slice = get_local_slice(slice_index, arr_start, total_len)
        return bodo.utils.conversion.ensure_contig_if_np(arr[new_slice])

    return getitem_impl


def int_getitem(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
    return arr[ind]


def int_optional_getitem(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
    pass


def int_isna(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
    pass


def transform_str_getitem_output(data, length):
    """
    Transform the final output of string/bytes data.
    Strings need to decode utf8 values from the data array.
    Bytes need to transform the final data from uint8 array to bytes array.
    """


@overload(transform_str_getitem_output)
def overload_transform_str_getitem_output(data, length):
    if data == bodo.types.string_type:
        return lambda data, length: bodo.libs.str_arr_ext.decode_utf8(
            data._data, length
        )  # pragma: no cover
    if data == types.Array(types.uint8, 1, "C"):
        return lambda data, length: bodo.libs.binary_arr_ext.init_bytes_type(
            data, length
        )  # pragma: no cover
    raise BodoError(f"Internal Error: Expected String or Uint8 Array, found {data}")


@overload(int_getitem, no_unliteral=True)
def int_getitem_overload(arr, ind, arr_start, total_len, is_1D):
    ANY_SOURCE = np.int32(hdist.ANY_SOURCE)
    dummy_use = numba.njit(cache=True, no_cpython_wrapper=True)(lambda a: None)

    if is_str_arr_type(arr) or arr == bodo.types.binary_array_type:
        # TODO: other kinds, unicode
        kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))
        # Dtype used for allocating the empty data. Either string or bytes
        _alloc_dtype = arr.dtype

        def str_getitem_impl(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
            if ind >= total_len:
                raise IndexError("index out of bounds")

            arr = decode_if_dict_array(arr)
            # Share the array contents by sending the raw bytes.
            # Match unicode support by only performing the decode at
            # the end after the data has been broadcast.

            # normalize negative slice
            ind = ind % total_len
            # TODO: avoid sending to root in case of 1D since position can be
            # calculated

            # send data to rank 0 and broadcast
            root = np.int32(0)
            size_tag = np.int32(10)
            tag = np.int32(11)
            send_size = np.zeros(1, np.int64)
            # We send the value to the root first and then have the root broadcast
            # the value because we don't know which rank holds the data in the 1DVar
            # case.
            if arr_start <= ind < (arr_start + len(arr)):
                ind = ind - arr_start
                data_arr = arr._data
                start_offset = bodo.libs.array_item_arr_ext.get_offsets_ind(
                    data_arr, ind
                )
                end_offset = bodo.libs.array_item_arr_ext.get_offsets_ind(
                    data_arr, ind + 1
                )
                length = end_offset - start_offset
                ptr = data_arr[ind]
                send_size[0] = length
                isend(send_size, np.int32(1), root, size_tag, True)
                isend(ptr, np.int32(length), root, tag, True)

            rank = bodo.libs.distributed_api.get_rank()
            # Allocate a dummy value for type inference. Note we allocate a value
            # instead of doing constant lowering because Bytes need a uint8 array, and
            # lowering an Array constant converts the type to read only.
            val = bodo.libs.str_ext.alloc_empty_bytes_or_string_data(
                _alloc_dtype, kind, 0, 1
            )
            l = 0
            if rank == root:
                l = recv(np.int64, ANY_SOURCE, size_tag)
                val = bodo.libs.str_ext.alloc_empty_bytes_or_string_data(
                    _alloc_dtype, kind, l, 1
                )
                data_ptr = bodo.libs.str_ext.get_unicode_or_numpy_data(val)
                _recv(data_ptr, np.int32(l), char_typ_enum, ANY_SOURCE, tag)

            dummy_use(send_size)
            l = bcast_scalar(l)
            dummy_use(arr)
            if rank != root:
                val = bodo.libs.str_ext.alloc_empty_bytes_or_string_data(
                    _alloc_dtype, kind, l, 1
                )
            data_ptr = bodo.libs.str_ext.get_unicode_or_numpy_data(val)
            c_bcast(data_ptr, np.int32(l), char_typ_enum, np.int32(root), 0)
            val = transform_str_getitem_output(val, l)
            return val

        return str_getitem_impl

    if isinstance(arr, bodo.types.CategoricalArrayType):
        elem_width = bodo.hiframes.pd_categorical_ext.get_categories_int_type(arr.dtype)

        def cat_getitem_impl(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
            # Support Categorical getitem by sending the code and then doing the
            # getitem from the categories.

            if ind >= total_len:
                raise IndexError("index out of bounds")

            # normalize negative slice
            ind = ind % total_len
            # TODO: avoid sending to root in case of 1D since position can be
            # calculated

            # send code data to rank 0 and broadcast
            root = np.int32(0)
            tag = np.int32(11)
            send_arr = np.zeros(1, elem_width)
            # We send the value to the root first and then have the root broadcast
            # the value because we don't know which rank holds the data in the 1DVar
            # case.
            if arr_start <= ind < (arr_start + len(arr)):
                codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(arr)
                data = codes[ind - arr_start]
                send_arr = np.full(1, data, elem_width)
                isend(send_arr, np.int32(1), root, tag, True)

            rank = bodo.libs.distributed_api.get_rank()
            # Set initial value to null.
            val = elem_width(-1)
            if rank == root:
                val = recv(elem_width, ANY_SOURCE, tag)

            dummy_use(send_arr)
            val = bcast_scalar(val)
            # Convert the code to the actual value to match getiem semantics
            output_val = arr.dtype.categories[max(val, 0)]
            return output_val

        return cat_getitem_impl

    if isinstance(arr, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType):
        tz_val = arr.tz

        def tz_aware_getitem_impl(
            arr, ind, arr_start, total_len, is_1D
        ):  # pragma: no cover
            if ind >= total_len:
                raise IndexError("index out of bounds")

            # normalize negative slice
            ind = ind % total_len
            # TODO: avoid sending to root in case of 1D since position can be
            # calculated

            # send data to rank 0 and broadcast
            root = np.int32(0)
            tag = np.int32(11)
            send_arr = np.zeros(1, np.int64)
            if arr_start <= ind < (arr_start + len(arr)):
                data = arr[ind - arr_start].value
                send_arr = np.full(1, data)
                isend(send_arr, np.int32(1), root, tag, True)

            rank = bodo.libs.distributed_api.get_rank()
            val = 0  # TODO: better way to get zero of type
            if rank == root:
                val = recv(np.int64, ANY_SOURCE, tag)

            dummy_use(send_arr)
            val = bcast_scalar(val)
            return bodo.hiframes.pd_timestamp_ext.convert_val_to_timestamp(val, tz_val)

        return tz_aware_getitem_impl

    if arr == bodo.types.null_array_type:

        def null_getitem_impl(
            arr, ind, arr_start, total_len, is_1D
        ):  # pragma: no cover
            if ind >= total_len:
                raise IndexError("index out of bounds")
            return None

        return null_getitem_impl

    if arr == bodo.types.datetime_date_array_type:

        def date_getitem_impl(
            arr, ind, arr_start, total_len, is_1D
        ):  # pragma: no cover
            if ind >= total_len:
                raise IndexError("index out of bounds")

            # normalize negative slice
            ind = ind % total_len
            # TODO: avoid sending to root in case of 1D since position can be
            # calculated

            # send data to rank 0 and broadcast
            root = np.int32(0)
            tag = np.int32(11)
            send_arr = np.zeros(1, np.int32)
            if arr_start <= ind < (arr_start + len(arr)):
                data = bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int(
                    arr[ind - arr_start]
                )
                send_arr = np.full(1, data)
                isend(send_arr, np.int32(1), root, tag, True)

            rank = bodo.libs.distributed_api.get_rank()
            val = np.int32(0)  # TODO: better way to get zero of type
            if rank == root:
                val = recv(np.int32, ANY_SOURCE, tag)

            dummy_use(send_arr)
            val = bcast_scalar(val)
            return bodo.hiframes.datetime_date_ext.cast_int_to_datetime_date(val)

        return date_getitem_impl

    if arr == bodo.types.timestamptz_array_type:

        def timestamp_tz_getitem_impl(
            arr, ind, arr_start, total_len, is_1D
        ):  # pragma: no cover
            if ind >= total_len:
                raise IndexError("index out of bounds")

            # normalize negative slice
            ind = ind % total_len
            # TODO: avoid sending to root in case of 1D since position can be
            # calculated

            # send data to rank 0 and broadcast
            root = np.int32(0)
            tag1 = np.int32(11)
            tag2 = np.int32(12)
            send_arr1 = np.zeros(1, np.int64)
            send_arr2 = np.zeros(1, np.int16)
            if arr_start <= ind < (arr_start + len(arr)):
                idx = ind - arr_start
                ts = arr.data_ts[idx]
                offset = arr.data_offset[idx]
                send_arr1 = np.full(1, ts)
                send_arr2 = np.full(1, offset)
                isend(send_arr1, np.int32(1), root, tag1, True)
                isend(send_arr2, np.int32(1), root, tag2, True)

            rank = bodo.libs.distributed_api.get_rank()
            new_ts = np.int64(0)  # TODO: better way to get zero of type
            new_offset = np.int16(0)  # TODO: better way to get zero of type
            if rank == root:
                new_ts = recv(np.int64, ANY_SOURCE, tag1)
                new_offset = recv(np.int16, ANY_SOURCE, tag2)

            dummy_use(send_arr1)
            dummy_use(send_arr2)
            return bcast_scalar(
                bodo.hiframes.timestamptz_ext.TimestampTZ(
                    pd.Timestamp(new_ts), new_offset
                )
            )

        return timestamp_tz_getitem_impl

    np_dtype = arr.dtype

    if isinstance(ind, types.BaseTuple):
        assert isinstance(arr, types.Array), (
            "int_getitem_overload: Numpy array expected"
        )
        assert all(isinstance(a, types.Integer) for a in ind.types), (
            "int_getitem_overload: only integer indices supported"
        )
        # TODO[BSE-2374]: support non-integer indices

        def getitem_impl(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
            ind_0 = ind[0]

            if ind_0 >= total_len:
                raise IndexError("index out of bounds")

            # normalize negative slice
            ind_0 = ind_0 % total_len
            # TODO: avoid sending to root in case of 1D since position can be
            # calculated

            # send data to rank 0 and broadcast
            root = np.int32(0)
            tag = np.int32(11)
            send_arr = np.zeros(1, np_dtype)
            if arr_start <= ind_0 < (arr_start + len(arr)):
                data = arr[(ind_0 - arr_start,) + ind[1:]]
                send_arr = np.full(1, data)
                isend(send_arr, np.int32(1), root, tag, True)

            rank = bodo.libs.distributed_api.get_rank()
            val = np.zeros(1, np_dtype)[0]  # TODO: better way to get zero of type
            if rank == root:
                val = recv(np_dtype, ANY_SOURCE, tag)

            dummy_use(send_arr)
            val = bcast_scalar(val)
            return val

        return getitem_impl

    assert isinstance(ind, types.Integer), "int_getitem_overload: int index expected"

    def getitem_impl(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
        if ind >= total_len:
            raise IndexError("index out of bounds")

        # normalize negative slice
        ind = ind % total_len
        # TODO: avoid sending to root in case of 1D since position can be
        # calculated

        # send data to rank 0 and broadcast
        root = np.int32(0)
        tag = np.int32(11)
        send_arr = np.zeros(1, np_dtype)
        if arr_start <= ind < (arr_start + len(arr)):
            data = arr[ind - arr_start]
            send_arr = np.full(1, data)
            isend(send_arr, np.int32(1), root, tag, True)

        rank = bodo.libs.distributed_api.get_rank()
        val = np.zeros(1, np_dtype)[0]  # TODO: better way to get zero of type
        if rank == root:
            val = recv(np_dtype, ANY_SOURCE, tag)

        dummy_use(send_arr)
        val = bcast_scalar(val)
        return val

    return getitem_impl


@overload(int_optional_getitem, no_unliteral=True)
def int_optional_getitem_overload(arr, ind, arr_start, total_len, is_1D):
    if bodo.utils.typing.is_nullable(arr):
        # If the array type is nullable then have an optional return type.
        def impl(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
            if int_isna(arr, ind, arr_start, total_len, is_1D):
                return None
            else:
                return int_getitem(arr, ind, arr_start, total_len, is_1D)

    else:

        def impl(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
            return int_getitem(arr, ind, arr_start, total_len, is_1D)

    return impl


@overload(int_isna, no_unliteral=True)
def int_isna_overload(arr, ind, arr_start, total_len, is_1D):
    ANY_SOURCE = np.int32(hdist.ANY_SOURCE)
    dummy_use = numba.njit(cache=True, no_cpython_wrapper=True)(lambda a: None)

    def impl(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
        if ind >= total_len:
            raise IndexError("index out of bounds")

        # TODO: avoid sending to root in case of 1D since position can be
        # calculated

        # send data to rank 0 and broadcast
        root = np.int32(0)
        tag = np.int32(11)
        send_arr = np.zeros(1, np.bool_)
        if arr_start <= ind < (arr_start + len(arr)):
            data = bodo.libs.array_kernels.isna(arr, ind - arr_start)
            send_arr = np.full(1, data)
            isend(send_arr, np.int32(1), root, tag, True)

        rank = bodo.libs.distributed_api.get_rank()
        val = False
        if rank == root:
            val = recv(np.bool_, ANY_SOURCE, tag)

        dummy_use(send_arr)
        val = bcast_scalar(val)
        return val

    return impl


def get_chunk_bounds(A):  # pragma: no cover
    pass


@overload(get_chunk_bounds, jit_options={"cache": True})
def get_chunk_bounds_overload(A, parallel=False):
    """get chunk boundary value (last element) of array A for each rank and make it
    available on all ranks.
    For example, given A data on rank 0 [1, 4, 6], and on rank 1 [7, 8, 11],
    output will be [6, 11] on all ranks.

    Designed for MERGE INTO support currently. Only supports Numpy int arrays, and
    handles empty chunk corner cases to support boundaries of sort in ascending order.
    See https://bodo.atlassian.net/wiki/spaces/B/pages/1157529601/MERGE+INTO+Design.

    Also used in implementation of window functions without partitions (e.g. ROW_NUMBER)
    for shuffling the rows back to the right rank after computation.

    Args:
        A (Bodo Numpy int array): input array chunk on this rank

    Returns:
        Bodo Numpy int array: chunk boundaries of all ranks
    """
    if not (isinstance(A, types.Array) and isinstance(A.dtype, types.Integer)):
        raise BodoError("get_chunk_bounds() only supports Numpy int input currently.")

    def impl(A, parallel=False):  # pragma: no cover
        if not parallel:
            # In the replicated case this is expected to be a NO-OP. This path exists
            # to avoid MPI calls in case we cannot optimize out this function for some reason.
            return np.empty(0, np.int64)

        n_pes = get_size()
        all_bounds = np.empty(n_pes, np.int64)
        all_empty = np.empty(n_pes, np.int8)

        # using int64 min value in case the first chunk is empty. This will ensure
        # the first rank will be assigned an empty output chunk in sort.
        val = numba.cpython.builtins.get_type_min_value(numba.core.types.int64)
        empty = 1
        if len(A) != 0:
            val = A[-1]
            empty = 0

        allgather(all_bounds, np.int64(val))
        allgather(all_empty, empty)

        # for empty chunks, use the boundary from previous rank to ensure empty output
        # chunk in sort (ascending order)
        for i, empty in enumerate(all_empty):
            if empty and i != 0:
                all_bounds[i] = all_bounds[i - 1]

        return all_bounds

    return impl


@numba.njit(cache=True)
def get_start_count(n):  # pragma: no cover
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    start = bodo.libs.distributed_api.get_start(n, n_pes, rank)
    count = bodo.libs.distributed_api.get_node_portion(n, n_pes, rank)
    return start, count


@numba.njit(cache=True)
def get_start(total_size, pes, rank):  # pragma: no cover
    """get start index in 1D distribution"""
    res = total_size % pes
    blk_size = (total_size - res) // pes
    return rank * blk_size + min(rank, res)


@numba.njit(cache=True)
def get_end(total_size, pes, rank):  # pragma: no cover
    """get end point of range for parfor division"""
    res = total_size % pes
    blk_size = (total_size - res) // pes
    return (rank + 1) * blk_size + min(rank + 1, res)


@numba.njit(cache=True)
def get_node_portion(total_size, pes, rank):  # pragma: no cover
    """get portion of size for alloc division"""
    res = total_size % pes
    blk_size = (total_size - res) // pes
    if rank < res:
        return blk_size + 1
    else:
        return blk_size


@numba.njit(cache=True)
def dist_cumsum(in_arr, out_arr):
    return dist_cumsum_impl(in_arr, out_arr)


@numba.generated_jit(nopython=True)
def dist_cumsum_impl(in_arr, out_arr):
    zero = in_arr.dtype(0)
    op = np.int32(Reduce_Type.Sum.value)

    def cumsum_impl(in_arr, out_arr):  # pragma: no cover
        c = zero
        for v in np.nditer(in_arr):
            c += v.item()
        prefix_var = dist_exscan(c, op)
        for i in range(in_arr.size):
            prefix_var += in_arr[i]
            out_arr[i] = prefix_var
        return 0

    return cumsum_impl


@numba.njit(cache=True)
def dist_cumprod(in_arr, out_arr):
    return dist_cumprod_impl(in_arr, out_arr)


@numba.generated_jit(nopython=True)
def dist_cumprod_impl(in_arr, out_arr):
    neutral_val = in_arr.dtype(1)
    op = np.int32(Reduce_Type.Prod.value)

    def cumprod_impl(in_arr, out_arr):  # pragma: no cover
        c = neutral_val
        for v in np.nditer(in_arr):
            c *= v.item()
        prefix_var = dist_exscan(c, op)
        # The MPI_Exscan has the default that on the first node, the value
        # are not set to their neutral value (0 for sum, 1 for prod, etc.)
        # bad design.
        # For dist_cumsum that is ok since variable are set to 0 by python.
        # But for product/min/max, we need to do it manually.
        if get_rank() == 0:
            prefix_var = neutral_val
        for i in range(in_arr.size):
            prefix_var *= in_arr[i]
            out_arr[i] = prefix_var
        return 0

    return cumprod_impl


@numba.njit(cache=True)
def dist_cummin(in_arr, out_arr):
    return dist_cummin_impl(in_arr, out_arr)


@numba.generated_jit(nopython=True)
def dist_cummin_impl(in_arr, out_arr):
    if isinstance(in_arr.dtype, types.Float):
        neutral_val = np.finfo(in_arr.dtype(1).dtype).max
    else:
        neutral_val = np.iinfo(in_arr.dtype(1).dtype).max
    op = np.int32(Reduce_Type.Min.value)

    def cummin_impl(in_arr, out_arr):  # pragma: no cover
        c = neutral_val
        for v in np.nditer(in_arr):
            c = min(c, v.item())
        prefix_var = dist_exscan(c, op)
        # Remarks for dist_cumprod applies here
        if get_rank() == 0:
            prefix_var = neutral_val
        for i in range(in_arr.size):
            prefix_var = min(prefix_var, in_arr[i])
            out_arr[i] = prefix_var
        return 0

    return cummin_impl


@numba.njit(cache=True)
def dist_cummax(in_arr, out_arr):
    return dist_cummax_impl(in_arr, out_arr)


@numba.generated_jit(nopython=True)
def dist_cummax_impl(in_arr, out_arr):
    if isinstance(in_arr.dtype, types.Float):
        neutral_val = np.finfo(in_arr.dtype(1).dtype).min
    else:
        neutral_val = np.iinfo(in_arr.dtype(1).dtype).min
    neutral_val = in_arr.dtype(1)
    op = np.int32(Reduce_Type.Max.value)

    def cummax_impl(in_arr, out_arr):  # pragma: no cover
        c = neutral_val
        for v in np.nditer(in_arr):
            c = max(c, v.item())
        prefix_var = dist_exscan(c, op)
        # Remarks for dist_cumprod applies here
        if get_rank() == 0:
            prefix_var = neutral_val
        for i in range(in_arr.size):
            prefix_var = max(prefix_var, in_arr[i])
            out_arr[i] = prefix_var
        return 0

    return cummax_impl


_allgather = types.ExternalFunction(
    "allgather", types.void(types.voidptr, types.int32, types.voidptr, types.int32)
)


@numba.njit(cache=True)
def allgather(arr, val):  # pragma: no cover
    type_enum = get_type_enum(arr)
    _allgather(arr.ctypes, 1, value_to_ptr(val), type_enum)


def dist_return(A):  # pragma: no cover
    return A


def rep_return(A):  # pragma: no cover
    return A


# array analysis extension for dist_return
def dist_return_equiv(self, scope, equiv_set, loc, args, kws):
    """dist_return output has the same shape as input"""
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_distributed_api_dist_return = dist_return_equiv
ArrayAnalysis._analyze_op_call_bodo_libs_distributed_api_rep_return = dist_return_equiv


def threaded_return(A):  # pragma: no cover
    return A


# dummy function to set a distributed array without changing the index in distributed
# pass
@numba.njit(cache=True)
def set_arr_local(arr, ind, val):  # pragma: no cover
    arr[ind] = val


# dummy function to specify local allocation size, to enable bypassing distributed
# transformations
@numba.njit(cache=True)
def local_alloc_size(n, in_arr):  # pragma: no cover
    return n


# TODO: move other funcs to old API?
@infer_global(threaded_return)
@infer_global(dist_return)
@infer_global(rep_return)
class ThreadedRetTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1  # array
        return signature(args[0], *args)


@numba.njit(cache=True)
def parallel_print(*args):  # pragma: no cover
    print(*args)


@overload(bodo.parallel_print)
def overload_parallel_print(*args):
    """print input arguments on all ranks in parallel"""

    def impl(*args):  # pragma: no cover
        parallel_print(*args)

    return impl


@numba.njit(cache=True)
def single_print(*args):  # pragma: no cover
    if bodo.libs.distributed_api.get_rank() == 0:
        print(*args)


def print_if_not_empty(args):  # pragma: no cover
    pass


@overload(print_if_not_empty)
def overload_print_if_not_empty(*args):
    """print input arguments only if rank == 0 or any data on current rank is not empty"""

    any_not_empty = (
        "("
        + " or ".join(
            ["False"]
            + [
                f"len(args[{i}]) != 0"
                for i, arg_type in enumerate(args)
                if is_array_typ(arg_type)
                or isinstance(arg_type, bodo.hiframes.pd_dataframe_ext.DataFrameType)
            ]
        )
        + ")"
    )
    func_text = (
        f"def impl(*args):\n"
        f"    if {any_not_empty} or bodo.get_rank() == 0:\n"
        f"        print(*args)"
    )
    loc_vars = {}
    # TODO: Provide specific globals after Numba's #3355 is resolved
    exec(func_text, globals(), loc_vars)
    impl = loc_vars["impl"]
    return impl


@numba.generated_jit(nopython=True)
def wait(req, cond=True):
    """wait on MPI request"""

    # get size dynamically from C code (mpich 3.2 is 4 bytes but openmpi 1.6 is 8)
    mpi_req_numba_type = getattr(types, "int" + str(8 * hdist.mpi_req_num_bytes))
    _wait = types.ExternalFunction(
        "dist_wait", types.void(mpi_req_numba_type, types.bool_)
    )

    # Tuple of requests (e.g. nullable arrays)
    if isinstance(req, types.BaseTuple):
        count = len(req.types)
        tup_call = ",".join(f"_wait(req[{i}], cond)" for i in range(count))
        func_text = "def bodo_wait(req, cond=True):\n"
        func_text += f"  return {tup_call}\n"
        return bodo_exec(func_text, {"_wait": _wait}, {}, __name__)

    # None passed means no request to wait on (no-op), happens for shift() for string
    # arrays since we use blocking communication instead
    if is_overload_none(req):
        return lambda req, cond=True: None  # pragma: no cover

    return lambda req, cond=True: _wait(req, cond)  # pragma: no cover


@register_jitable
def _set_if_in_range(A, val, index, chunk_start):  # pragma: no cover
    if index >= chunk_start and index < chunk_start + len(A):
        A[index - chunk_start] = val


@register_jitable
def _root_rank_select(old_val, new_val):  # pragma: no cover
    if get_rank() == 0:
        return old_val
    return new_val


def get_tuple_prod(t):  # pragma: no cover
    return np.prod(t)


@overload(get_tuple_prod, no_unliteral=True)
def get_tuple_prod_overload(t):
    # handle empty tuple seperately since empty getiter doesn't work
    if t == numba.core.types.containers.Tuple(()):
        return lambda t: 1

    def get_tuple_prod_impl(t):  # pragma: no cover
        res = 1
        for a in t:
            res *= a
        return res

    return get_tuple_prod_impl


sig = types.void(
    types.voidptr,  # output array
    types.voidptr,  # input array
    types.intp,  # old_len
    types.intp,  # new_len
    types.intp,  # input lower_dim size in bytes
    types.intp,  # output lower_dim size in bytes
    types.int32,
    types.voidptr,
)

oneD_reshape_shuffle = types.ExternalFunction("oneD_reshape_shuffle", sig)


@numba.njit(cache=True, no_cpython_wrapper=True)
def dist_oneD_reshape_shuffle(
    lhs, in_arr, new_dim0_global_len, dest_ranks=None
):  # pragma: no cover
    """shuffles the data for ndarray reshape to fill the output array properly.
    if dest_ranks != None the data will be sent only to the specified ranks"""
    c_in_arr = np.ascontiguousarray(in_arr)
    in_lower_dims_size = get_tuple_prod(c_in_arr.shape[1:])
    out_lower_dims_size = get_tuple_prod(lhs.shape[1:])

    if dest_ranks is not None:
        dest_ranks_arr = np.array(dest_ranks, dtype=np.int32)
    else:
        dest_ranks_arr = np.empty(0, dtype=np.int32)

    dtype_size = bodo.io.np_io.get_dtype_size(in_arr.dtype)
    oneD_reshape_shuffle(
        lhs.ctypes,
        c_in_arr.ctypes,
        new_dim0_global_len,
        len(in_arr),
        dtype_size * out_lower_dims_size,
        dtype_size * in_lower_dims_size,
        len(dest_ranks_arr),
        dest_ranks_arr.ctypes,
    )
    check_and_propagate_cpp_exception()


permutation_int = types.ExternalFunction(
    "permutation_int", types.void(types.voidptr, types.intp)
)


@numba.njit(cache=True)
def dist_permutation_int(lhs, n):  # pragma: no cover
    permutation_int(lhs.ctypes, n)


permutation_array_index = types.ExternalFunction(
    "permutation_array_index",
    types.void(
        types.voidptr,
        types.intp,
        types.intp,
        types.voidptr,
        types.int64,
        types.voidptr,
        types.intp,
        types.int64,
    ),
)


@numba.njit(cache=True)
def dist_permutation_array_index(
    lhs, lhs_len, dtype_size, rhs, p, p_len, n_samples
):  # pragma: no cover
    c_rhs = np.ascontiguousarray(rhs)
    lower_dims_size = get_tuple_prod(c_rhs.shape[1:])
    elem_size = dtype_size * lower_dims_size
    permutation_array_index(
        lhs.ctypes,
        lhs_len,
        elem_size,
        c_rhs.ctypes,
        c_rhs.shape[0],
        p.ctypes,
        p_len,
        n_samples,
    )
    check_and_propagate_cpp_exception()


def bcast(data, comm_ranks=None, root=DEFAULT_ROOT, comm=None):  # pragma: no cover
    """bcast() sends data from rank 0 to comm_ranks."""
    from mpi4py import MPI

    rank = bodo.libs.distributed_api.get_rank()
    # make sure all ranks receive proper data type as input
    dtype = bodo.typeof(data)
    dtype = _bcast_dtype(dtype, root, comm)

    is_sender = rank == root
    if comm is not None:
        # Sender has to set root to MPI.ROOT in case of intercomm
        is_sender = root == MPI.ROOT

    if not is_sender:
        data = get_value_for_type(dtype)

    # Pass empty array for comm_ranks to downstream code meaning all ranks are targets
    if comm_ranks is None:
        comm_ranks = np.array([], np.int32)

    # Pass Comm pointer to native code (0 means not provided).
    if comm is None:
        comm_ptr = 0
    else:
        comm_ptr = MPI._addressof(comm)

    return bcast_impl_wrapper(data, comm_ranks, root, comm_ptr)


@numba.njit(cache=True)
def bcast_impl_wrapper(data, comm_ranks, root, comm):
    return bcast_impl(data, comm_ranks, root, comm)


@overload(bcast)
def bcast_overload(data, comm_ranks, root=DEFAULT_ROOT, comm=0):
    """support bcast inside jit functions"""
    return lambda data, comm_ranks, root=DEFAULT_ROOT, comm=0: bcast_impl(
        data, comm_ranks, root, comm
    )  # pragma: no cover


@numba.generated_jit(nopython=True)
def bcast_impl(data, comm_ranks, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
    """nopython implementation of bcast()"""
    bodo.hiframes.pd_dataframe_ext.check_runtime_cols_unsupported(data, "bodo.bcast()")
    c_broadcast_array = ExternalFunctionErrorChecked(
        "broadcast_array_py_entry",
        array_info_type(array_info_type, array_info_type, types.int32, types.int64),
    )

    if isinstance(data, types.Array) and data.ndim > 1:
        ndim = data.ndim
        zero_shape = (0,) * ndim

        def impl_array_multidim(
            data, comm_ranks, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            data_in = np.ascontiguousarray(data.reshape(-1))

            # broadcast shape to all processors
            shape = zero_shape
            is_sender = rank == root
            if comm != 0:
                is_sender = root == MPI.ROOT
            if is_sender:
                shape = data.shape
            shape = bcast_tuple(shape, root, comm)

            data_cpp = array_to_info(data_in)
            comm_ranks_cpp = array_to_info(comm_ranks)
            our_arr_cpp = c_broadcast_array(data_cpp, comm_ranks_cpp, root, comm)
            out_arr = info_to_array(our_arr_cpp, data_in)
            delete_info(our_arr_cpp)

            # Ranks not in comm_ranks return empty arrays that need reshaped to zero
            # length dimensions
            if len(out_arr) == 0:
                shape = zero_shape

            return out_arr.reshape(shape)

        return impl_array_multidim

    if bodo.utils.utils.is_array_typ(data, False):

        def impl_array(data, comm_ranks, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
            data_cpp = array_to_info(data)
            comm_ranks_cpp = array_to_info(comm_ranks)
            our_arr_cpp = c_broadcast_array(data_cpp, comm_ranks_cpp, root, comm)
            out_arr = info_to_array(our_arr_cpp, data)
            delete_info(our_arr_cpp)
            return out_arr

        return impl_array

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        col_name_meta_value_bcast = ColNamesMetaType(data.columns)

        if data.is_table_format:

            def impl_df_table(
                data, comm_ranks, root=DEFAULT_ROOT, comm=0
            ):  # pragma: no cover
                T = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(data)
                T2 = bodo.libs.distributed_api.bcast_impl(T, comm_ranks, root, comm)
                index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)
                g_index = bodo.libs.distributed_api.bcast_impl(
                    index, comm_ranks, root, comm
                )
                return bodo.hiframes.pd_dataframe_ext.init_dataframe(
                    (T2,), g_index, col_name_meta_value_bcast
                )

            return impl_df_table

        n_cols = len(data.columns)
        data_args = ", ".join(f"g_data_{i}" for i in range(n_cols))

        func_text = f"def impl_df(data, comm_ranks, root={DEFAULT_ROOT}, comm=0):\n"
        for i in range(n_cols):
            func_text += f"  data_{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {i})\n"
            func_text += f"  g_data_{i} = bodo.libs.distributed_api.bcast_impl(data_{i}, comm_ranks, root, comm)\n"
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
        )
        func_text += "  g_index = bodo.libs.distributed_api.bcast_impl(index, comm_ranks, root, comm)\n"
        func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args},), g_index, __col_name_meta_value_bcast)\n"

        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "__col_name_meta_value_bcast": col_name_meta_value_bcast,
            },
            loc_vars,
        )
        impl_df = loc_vars["impl_df"]
        return impl_df

    if isinstance(data, bodo.hiframes.table.TableType):
        data_type = data
        out_cols_arr = np.arange(len(data.arr_types), dtype=np.int64)
        c_broadcast_table = ExternalFunctionErrorChecked(
            "broadcast_table_py_entry",
            table_type(table_type, array_info_type, types.int32, types.int64),
        )

        def impl_table(data, comm_ranks, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
            data_cpp = py_table_to_cpp_table(data, data_type)
            comm_ranks_cpp = array_to_info(comm_ranks)
            out_cpp_table = c_broadcast_table(data_cpp, comm_ranks_cpp, root, comm)
            out_table = cpp_table_to_py_table(out_cpp_table, out_cols_arr, data_type, 0)
            delete_table(out_cpp_table)
            return out_table

        return impl_table

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):

        def impl_range_index(
            data, comm_ranks, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            start = data._start
            stop = data._stop
            step = data._step
            name = data._name

            name = bcast_scalar(name, root, comm)
            start = bcast_scalar(start, root, comm)
            stop = bcast_scalar(stop, root, comm)
            step = bcast_scalar(step, root, comm)

            # Return empty RangeIndex in case of ranks out of target ranks to match
            # empty arrays in the output DataFrame.
            rank = bodo.libs.distributed_api.get_rank()
            if len(comm_ranks) > 0 and rank not in comm_ranks:
                start, stop = 0, 0

            return bodo.hiframes.pd_index_ext.init_range_index(start, stop, step, name)

        return impl_range_index

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):

        def impl_pd_index(
            data, comm_ranks, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            data_in = data._data
            name = data._name
            arr = bodo.libs.distributed_api.bcast_impl(data_in, comm_ranks, root, comm)
            return bodo.utils.conversion.index_from_array(arr, name)

        return impl_pd_index

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl_series(
            data, comm_ranks, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # bcast data
            out_name = bodo.libs.distributed_api.bcast_impl(
                name, comm_ranks, root, comm
            )
            out_arr = bodo.libs.distributed_api.bcast_impl(arr, comm_ranks, root, comm)
            out_index = bodo.libs.distributed_api.bcast_impl(
                index, comm_ranks, root, comm
            )
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, out_name)

        return impl_series

    # Tuple of data containers
    if isinstance(data, types.BaseTuple):
        func_text = f"def impl_tuple(data, comm_ranks, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(
                f"bcast_impl(data[{i}], comm_ranks, root, comm)"
                for i in range(len(data))
            ),
            "," if len(data) > 0 else "",
        )
        loc_vars = {}
        exec(func_text, {"bcast_impl": bcast_impl}, loc_vars)
        impl_tuple = loc_vars["impl_tuple"]
        return impl_tuple

    if data is types.none:  # pragma: no cover
        return (
            lambda data, comm_ranks, root=DEFAULT_ROOT, comm=0: None
        )  # pragma: no cover

    raise BodoError(f"bcast(): unsupported input type {data}")


node_ranks = None


def get_host_ranks(comm: MPI.Comm = MPI.COMM_WORLD):  # pragma: no cover
    """Get dict holding hostname and its associated ranks"""
    global node_ranks
    if node_ranks is None:
        hostname = MPI.Get_processor_name()
        rank_host = comm.allgather(hostname)
        node_ranks = defaultdict(list)
        for i, host in enumerate(rank_host):
            node_ranks[host].append(i)
    return node_ranks


def create_subcomm_mpi4py(comm_ranks):  # pragma: no cover
    """Create sub-communicator from MPI.COMM_WORLD with specific ranks only"""
    comm = MPI.COMM_WORLD
    world_group = comm.Get_group()
    new_group = world_group.Incl(comm_ranks)
    new_comm = comm.Create_group(new_group)
    return new_comm


def get_nodes_first_ranks(comm: MPI.Comm = MPI.COMM_WORLD):  # pragma: no cover
    """Get first rank in each node"""
    host_ranks = get_host_ranks(comm)
    return np.array([ranks[0] for ranks in host_ranks.values()], dtype="int32")


def get_num_nodes():  # pragma: no cover
    """Get number of nodes"""
    return len(get_host_ranks())


def get_num_gpus(framework="torch"):  # pragma: no cover
    """Get number of GPU devices on this host"""
    if framework == "torch":
        try:
            import torch

            if hasattr(torch, "accelerator"):
                return torch.accelerator.device_count()
            else:
                return torch.cuda.device_count()
        except ImportError:
            raise RuntimeError(
                "PyTorch is not installed. Please install PyTorch to use GPU features."
            )
    elif framework == "tensorflow":
        try:
            import tensorflow as tf

            return len(tf.config.list_physical_devices("GPU"))
        except ImportError:
            raise RuntimeError(
                "TensorFlow is not installed. Please install TensorFlow to use GPU features."
            )
    else:
        raise RuntimeError(f"Framework {framework} not recognized")


def get_gpu_ranks():  # pragma: no cover
    """Calculate and return the global list of ranks to pin to GPUs
    Return list of ranks to pin to the GPUs.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    host_ranks = get_host_ranks()
    nodes_first_ranks = get_nodes_first_ranks()
    if rank in nodes_first_ranks:
        # the first rank on each host collects the number of GPUs on the host
        # and sends them to rank 0. rank 0 will calculate global gpu rank list
        try:
            num_gpus_in_node = get_num_gpus()
        except Exception as e:  # pragma: no cover
            num_gpus_in_node = e
        subcomm = create_subcomm_mpi4py(nodes_first_ranks)
        num_gpus_per_node = subcomm.gather(num_gpus_in_node)
        if rank == 0:
            gpu_ranks = []
            error = None
            for i, ranks in enumerate(host_ranks.values()):  # pragma: no cover
                n_gpus = num_gpus_per_node[i]
                if isinstance(n_gpus, Exception):
                    error = n_gpus
                    break
                if n_gpus == 0:
                    continue
                cores_per_gpu = len(ranks) // n_gpus
                for local_rank, global_rank in enumerate(ranks):
                    if local_rank % cores_per_gpu == 0:
                        # pin this rank to GPU
                        my_gpu = local_rank // cores_per_gpu
                        if my_gpu < n_gpus:
                            gpu_ranks.append(global_rank)
            if error:  # pragma: no cover
                comm.bcast(error)
                raise error
            else:
                comm.bcast(gpu_ranks)
    if rank != 0:  # pragma: no cover
        # wait for global list of GPU ranks from rank 0.
        gpu_ranks = comm.bcast(None)
        if isinstance(gpu_ranks, Exception):
            e = gpu_ranks
            raise e
    return gpu_ranks


# Use default number of iterations for sync if not specified by user
sync_iters = (
    bodo.default_stream_loop_sync_iters
    if bodo.stream_loop_sync_iters == -1
    else bodo.stream_loop_sync_iters
)


@numba.njit(cache=True)
def sync_is_last(condition, iter):  # pragma: no cover
    """Check if condition is true for all ranks if iter % bodo.stream_loop_sync_iters == 0, return false otherwise"""
    if iter % sync_iters == 0:
        return dist_reduce(
            condition, np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value)
        )
    else:
        return False


class IsLastStateType(types.Type):
    """Type for C++ IsLastState pointer"""

    def __init__(self):
        super().__init__("IsLastStateType()")


register_model(IsLastStateType)(models.OpaqueModel)
is_last_state_type = IsLastStateType()

init_is_last_state = types.ExternalFunction("init_is_last_state", is_last_state_type())

# NOTE: using int32 types to avoid i1 vs i8 boolean errors in lowering
sync_is_last_non_blocking = types.ExternalFunction(
    "sync_is_last_non_blocking", types.int32(is_last_state_type, types.int32)
)
