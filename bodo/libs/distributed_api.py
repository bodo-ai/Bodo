# Copyright (C) 2022 Bodo Inc. All rights reserved.
import atexit
import datetime
import sys
import time
import warnings
from collections import defaultdict
from decimal import Decimal
from enum import Enum

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
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
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.datetime_timedelta_ext import datetime_timedelta_array_type
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.hiframes.time_ext import TimeArrayType
from bodo.libs import hdist
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayType,
    np_offset_type,
    offset_type,
)
from bodo.libs.binary_arr_ext import binary_array_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType, set_bit_to_arr
from bodo.libs.interval_arr_ext import IntervalArrayType
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType
from bodo.libs.str_arr_ext import (
    convert_len_arr_to_offset,
    get_bit_bitmap,
    get_data_ptr,
    get_null_bitmap_ptr,
    get_offset_ptr,
    num_total_chars,
    pre_alloc_string_array,
    set_bit_to,
    string_array_type,
)
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.mpi4py import MPI
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    ColNamesMetaType,
    decode_if_dict_array,
    is_overload_false,
    is_overload_none,
    is_str_arr_type,
)
from bodo.utils.utils import (
    CTypeEnum,
    check_and_propagate_cpp_exception,
    empty_like_type,
    is_array_typ,
    numba_to_c_type,
)

ll.add_symbol("dist_get_time", hdist.dist_get_time)
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
ll.add_symbol("c_barrier", hdist.barrier)
ll.add_symbol("c_alltoall", hdist.c_alltoall)
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


# get size dynamically from C code (mpich 3.2 is 4 bytes but openmpi 1.6 is 8)
mpi_req_numba_type = getattr(types, "int" + str(8 * hdist.mpi_req_num_bytes))

MPI_ROOT = 0
ANY_SOURCE = np.int32(hdist.ANY_SOURCE)


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


_get_rank = types.ExternalFunction("c_get_rank", types.int32())
_get_size = types.ExternalFunction("c_get_size", types.int32())
_barrier = types.ExternalFunction("c_barrier", types.int32())
_dist_transpose_comm = types.ExternalFunction(
    "_dist_transpose_comm",
    types.void(types.voidptr, types.voidptr, types.int32, types.int64, types.int64),
)


@numba.njit
def get_rank():  # pragma: no cover
    """wrapper for getting process rank (MPI rank currently)"""
    return _get_rank()


@numba.njit
def get_size():  # pragma: no cover
    """wrapper for getting number of processes (MPI COMM size currently)"""
    return _get_size()


@numba.njit
def barrier():  # pragma: no cover
    """wrapper for barrier (MPI barrier currently)"""
    _barrier()


_get_time = types.ExternalFunction("get_time", types.float64())
dist_time = types.ExternalFunction("dist_get_time", types.float64())


@infer_global(time.time)
class TimeInfer(ConcreteTemplate):
    cases = [signature(types.float64)]


@lower_builtin(time.time)
def lower_time_time(context, builder, sig, args):
    return context.compile_internal(builder, lambda: _get_time(), sig, args)


@numba.generated_jit(nopython=True)
def get_type_enum(arr):
    arr = arr.instance_type if isinstance(arr, types.TypeRef) else arr
    dtype = arr.dtype
    if isinstance(dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
        dtype = bodo.hiframes.pd_categorical_ext.get_categories_int_type(dtype)

    typ_val = numba_to_c_type(dtype)
    return lambda arr: np.int32(typ_val)


INT_MAX = np.iinfo(np.int32).max

_send = types.ExternalFunction(
    "c_send",
    types.void(types.voidptr, types.int32, types.int32, types.int32, types.int32),
)


@numba.njit
def send(val, rank, tag):  # pragma: no cover
    # dummy array for val
    send_arr = np.full(1, val)
    type_enum = get_type_enum(send_arr)
    _send(send_arr.ctypes, 1, type_enum, rank, tag)


_recv = types.ExternalFunction(
    "c_recv",
    types.void(types.voidptr, types.int32, types.int32, types.int32, types.int32),
)


@numba.njit
def recv(dtype, rank, tag):  # pragma: no cover
    # dummy array for val
    recv_arr = np.empty(1, dtype)
    type_enum = get_type_enum(recv_arr)
    _recv(recv_arr.ctypes, 1, type_enum, rank, tag)
    return recv_arr[0]


_isend = types.ExternalFunction(
    "dist_isend",
    mpi_req_numba_type(
        types.voidptr, types.int32, types.int32, types.int32, types.int32, types.bool_
    ),
)


@numba.generated_jit(nopython=True)
def isend(arr, size, pe, tag, cond=True):
    """call MPI isend with input data"""
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


_irecv = types.ExternalFunction(
    "dist_irecv",
    mpi_req_numba_type(
        types.voidptr, types.int32, types.int32, types.int32, types.int32, types.bool_
    ),
)


@numba.generated_jit(nopython=True)
def irecv(arr, size, pe, tag, cond=True):  # pragma: no cover
    """post MPI irecv for array and return the request"""

    # Numpy array
    if isinstance(arr, types.Array):

        def impl(arr, size, pe, tag, cond=True):  # pragma: no cover
            type_enum = get_type_enum(arr)
            return _irecv(arr.ctypes, size, type_enum, pe, tag, cond)

        return impl

    # Primitive array
    if isinstance(arr, bodo.libs.primitive_arr_ext.PrimitiveArrayType):

        def impl(arr, size, pe, tag, cond=True):  # pragma: no cover
            np_arr = bodo.libs.primitive_arr_ext.primitive_to_np(arr)
            type_enum = get_type_enum(np_arr)
            return _irecv(np_arr.ctypes, size, type_enum, pe, tag, cond)

        return impl

    if arr == boolean_array_type:
        # Nullable booleans need their own implementation because the
        # data array stores 1 bit per boolean. As a result, the data array
        # requires separate handling.
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_bool(arr, size, pe, tag, cond=True):  # pragma: no cover
            n_bytes = (size + 7) >> 3
            data_req = _irecv(arr._data.ctypes, n_bytes, char_typ_enum, pe, tag, cond)
            null_req = _irecv(
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
    ):
        # return a tuple of requests for data and null arrays
        type_enum = np.int32(numba_to_c_type(arr.dtype))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_nullable(arr, size, pe, tag, cond=True):  # pragma: no cover
            n_bytes = (size + 7) >> 3
            data_req = _irecv(arr._data.ctypes, size, type_enum, pe, tag, cond)
            null_req = _irecv(
                arr._null_bitmap.ctypes, n_bytes, char_typ_enum, pe, tag, cond
            )
            return (data_req, null_req)

        return impl_nullable

    # string arrays
    if arr in [binary_array_type, string_array_type]:
        offset_typ_enum = np.int32(numba_to_c_type(offset_type))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        # using blocking communication for string arrays instead since the array
        # slice passed in shift() may not stay alive (not a view of the original array)
        if arr == binary_array_type:
            alloc_fn = "bodo.libs.binary_arr_ext.pre_alloc_binary_array"
        else:
            alloc_fn = "bodo.libs.str_arr_ext.pre_alloc_string_array"
        func_text = f"""def impl(arr, size, pe, tag, cond=True):
            # recv the number of string characters and resize buffer to proper size
            n_chars = bodo.libs.distributed_api.recv(np.int64, pe, tag - 1)
            new_arr = {alloc_fn}(size, n_chars)
            bodo.libs.str_arr_ext.move_str_binary_arr_payload(arr, new_arr)

            n_bytes = (size + 7) >> 3
            bodo.libs.distributed_api._recv(
                bodo.libs.str_arr_ext.get_offset_ptr(arr),
                size + 1,
                offset_typ_enum,
                pe,
                tag,
            )
            bodo.libs.distributed_api._recv(
                bodo.libs.str_arr_ext.get_data_ptr(arr), n_chars, char_typ_enum, pe, tag
            )
            bodo.libs.distributed_api._recv(
                bodo.libs.str_arr_ext.get_null_bitmap_ptr(arr),
                n_bytes,
                char_typ_enum,
                pe,
                tag,
            )
            return None"""

        loc_vars = dict()
        exec(
            func_text,
            {
                "bodo": bodo,
                "np": np,
                "offset_typ_enum": offset_typ_enum,
                "char_typ_enum": char_typ_enum,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    raise BodoError(f"irecv(): array type {arr} not supported yet")


_alltoall = types.ExternalFunction(
    "c_alltoall", types.void(types.voidptr, types.voidptr, types.int32, types.int32)
)


@numba.njit
def alltoall(send_arr, recv_arr, count):  # pragma: no cover
    # TODO: handle int64 counts
    assert count < INT_MAX
    type_enum = get_type_enum(send_arr)
    _alltoall(send_arr.ctypes, recv_arr.ctypes, np.int32(count), type_enum)


@numba.njit
def gather_scalar(data, allgather=False, warn_if_rep=True, root=MPI_ROOT):
    return gather_scalar_impl_jit(data, allgather, warn_if_rep, root)


@numba.generated_jit(nopython=True)
def gather_scalar_impl_jit(data, allgather=False, warn_if_rep=True, root=MPI_ROOT):
    data = types.unliteral(data)
    typ_val = numba_to_c_type(data)
    dtype = data

    def gather_scalar_impl(
        data, allgather=False, warn_if_rep=True, root=MPI_ROOT
    ):  # pragma: no cover
        n_pes = bodo.libs.distributed_api.get_size()
        rank = bodo.libs.distributed_api.get_rank()
        send = np.full(1, data, dtype)
        res_size = n_pes if (rank == root or allgather) else 0
        res = np.empty(res_size, dtype)
        c_gather_scalar(
            send.ctypes, res.ctypes, np.int32(typ_val), allgather, np.int32(root)
        )
        return res

    return gather_scalar_impl


c_gather_scalar = types.ExternalFunction(
    "c_gather_scalar",
    types.void(types.voidptr, types.voidptr, types.int32, types.bool_, types.int32),
)


# sendbuf, sendcount, recvbuf, recv_counts, displs, dtype
c_gatherv = types.ExternalFunction(
    "c_gatherv",
    types.void(
        types.voidptr,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int32,
        types.bool_,
        types.int32,
    ),
)

# sendbuff, sendcounts, displs, recvbuf, recv_count, dtype
c_scatterv = types.ExternalFunction(
    "c_scatterv",
    types.void(
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int32,
        types.int32,
    ),
)


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


_dist_reduce = types.ExternalFunction(
    "dist_reduce", types.void(types.voidptr, types.voidptr, types.int32, types.int32)
)

_dist_arr_reduce = types.ExternalFunction(
    "dist_arr_reduce", types.void(types.voidptr, types.int64, types.int32, types.int32)
)

_timestamptz_reduce = types.ExternalFunction(
    "timestamptz_reduce",
    types.void(types.int64, types.int64, types.voidptr, types.voidptr, types.boolean),
)

_decimal_reduce = types.ExternalFunction(
    "decimal_reduce",
    types.void(types.int64, types.voidptr, types.voidptr, types.int32, types.int32),
)


@numba.njit
def dist_reduce(value, reduce_op):
    return dist_reduce_impl(value, reduce_op)


@numba.generated_jit(nopython=True)
def dist_reduce_impl(value, reduce_op):
    if isinstance(value, types.Array):
        typ_enum = np.int32(numba_to_c_type(value.dtype))

        def impl_arr(value, reduce_op):  # pragma: no cover
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
        ]
        # TODO: Support uint64
        if not sys.platform.startswith("win"):
            # long is 4 byte on Windows
            supported_typs.append(types.int64)
            supported_typs.append(bodo.datetime64ns)
            supported_typs.append(bodo.timedelta64ns)
            supported_typs.append(bodo.datetime_date_type)
            supported_typs.append(bodo.TimeType)
        if target_typ not in supported_typs and not isinstance(
            target_typ, bodo.Decimal128Type
        ):  # pragma: no cover
            raise BodoError(
                "argmin/argmax not supported for type {}".format(target_typ)
            )

    typ_enum = np.int32(numba_to_c_type(target_typ))

    if isinstance(target_typ, bodo.Decimal128Type) and isinstance(
        types.unliteral(value), IndexValueType
    ):
        # For index-value types, the data pointed too has different amounts of padding depending on machine type.
        # as a workaround, we can pass the index separately.
        def impl(value, reduce_op):  # pragma: no cover
            in_ptr = value_to_ptr(value.value)
            out_ptr = value_to_ptr(value)
            _decimal_reduce(value.index, in_ptr, out_ptr, reduce_op, typ_enum)
            return load_val_ptr(out_ptr, value)

        return impl

    if isinstance(value, bodo.TimestampTZType):
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
        def impl(value, reduce_op):  # pragma: no cover
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
            return bodo.TimestampTZ(pd.Timestamp(out_ts), out_offset)

        return impl

    def impl(value, reduce_op):  # pragma: no cover
        in_ptr = value_to_ptr(value)
        out_ptr = value_to_ptr(value)
        _dist_reduce(in_ptr, out_ptr, reduce_op, typ_enum)
        return load_val_ptr(out_ptr, value)

    return impl


_dist_exscan = types.ExternalFunction(
    "dist_exscan", types.void(types.voidptr, types.voidptr, types.int32, types.int32)
)


@numba.njit
def dist_exscan(value, reduce_op):
    return dist_exscan_impl(value, reduce_op)


@numba.generated_jit(nopython=True)
def dist_exscan_impl(value, reduce_op):
    target_typ = types.unliteral(value)
    typ_enum = np.int32(numba_to_c_type(target_typ))
    zero = target_typ(0)

    def impl(value, reduce_op):  # pragma: no cover
        in_ptr = value_to_ptr(value)
        out_ptr = value_to_ptr(zero)
        _dist_exscan(in_ptr, out_ptr, reduce_op, typ_enum)
        return load_val_ptr(out_ptr, value)

    return impl


# from GetBit() in Arrow
@numba.njit
def get_bit(bits, i):  # pragma: no cover
    return (bits[i >> 3] >> (i & 0x07)) & 1


@numba.njit
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


@numba.njit
def gatherv(data, allgather=False, warn_if_rep=True, root=MPI_ROOT):
    return gatherv_impl_jit(data, allgather, warn_if_rep, root)


@numba.generated_jit(nopython=True)
def gatherv_impl_jit(data, allgather=False, warn_if_rep=True, root=MPI_ROOT):
    """gathers distributed data into rank 0 or all ranks if 'allgather' is set.
    'warn_if_rep' flag controls if a warning is raised if the input is replicated and
    gatherv has no effect (applicable only inside jit functions).
    """
    from bodo.libs.csr_matrix_ext import CSRMatrixType

    bodo.hiframes.pd_dataframe_ext.check_runtime_cols_unsupported(
        data, "bodo.gatherv()"
    )

    if isinstance(data, CategoricalArrayType):

        def impl_cat(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            codes = bodo.gatherv(data.codes, allgather, root=root)
            return bodo.hiframes.pd_categorical_ext.init_categorical_array(
                codes, data.dtype
            )

        return impl_cat

    if data == bodo.null_array_type:

        def impl_null_array(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            # Gather the lengths of the null array
            lengths = gather_scalar(len(data), allgather, root=root)
            # Sum the results to get the total length
            total_length = lengths.sum()
            return bodo.libs.null_arr_ext.init_null_array(total_length)

        return impl_null_array

    if isinstance(data, types.Array):
        typ_val = numba_to_c_type(data.dtype)

        def gatherv_impl(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            data = np.ascontiguousarray(data)
            rank = bodo.libs.distributed_api.get_rank()
            # size to handle multi-dim arrays
            n_loc = data.size
            recv_counts = gather_scalar(np.int32(n_loc), allgather, root=root)
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            displs = np.empty(1, np.int32)
            if rank == root or allgather:
                displs = bodo.ir.join.calc_disp(recv_counts)
            # print(rank, n_loc, n_total, recv_counts, displs)
            c_gatherv(
                data.ctypes,
                np.int32(n_loc),
                all_data.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                np.int32(typ_val),
                allgather,
                np.int32(root),
            )
            # handle multi-dim case
            return all_data.reshape((-1,) + data.shape[1:])

        return gatherv_impl

    if data == bodo.string_array_type:

        def gatherv_str_arr_impl(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            # call gatherv() on underlying array(item) array
            all_data = bodo.gatherv(data._data, allgather, warn_if_rep, root)
            return bodo.libs.str_arr_ext.init_str_arr(all_data)

        return gatherv_str_arr_impl

    if data == bodo.dict_str_arr_type:

        def gatherv_dict_arr_impl(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            # gather data as string for simplicity (but return dict to have proper type)
            # TODO[BSE-1567]: use the C++ implementation which keeps data as dict
            str_data = decode_if_dict_array(data)
            all_data = bodo.gatherv(str_data, allgather, warn_if_rep, root)
            return bodo.libs.str_arr_ext.str_arr_to_dict_str_arr(all_data)

        return gatherv_dict_arr_impl

    if data == binary_array_type:

        def gatherv_binary_arr_impl(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            # call gatherv() on underlying array(item) array
            all_data = bodo.gatherv(data._data, allgather, warn_if_rep, root)
            return bodo.libs.binary_arr_ext.init_binary_arr(all_data)

        return gatherv_binary_arr_impl

    # Handle datetime error as special because no _data field
    if data == datetime_timedelta_array_type:
        typ_val = numba_to_c_type(types.int64)
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def gatherv_impl_int_arr(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_bytes = (n_loc + 7) >> 3
            recv_counts = gather_scalar(np.int32(n_loc), allgather, root=root)
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            displs = np.empty(1, np.int32)
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)
            if rank == root or allgather:
                displs = bodo.ir.join.calc_disp(recv_counts)
                recv_counts_nulls = np.empty(len(recv_counts), np.int32)
                for i in range(len(recv_counts)):
                    recv_counts_nulls[i] = (recv_counts[i] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)
            c_gatherv(
                data._days_data.ctypes,
                np.int32(n_loc),
                all_data._days_data.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                np.int32(typ_val),
                allgather,
                np.int32(root),
            )
            c_gatherv(
                data._seconds_data.ctypes,
                np.int32(n_loc),
                all_data._seconds_data.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                np.int32(typ_val),
                allgather,
                np.int32(root),
            )
            c_gatherv(
                data._microseconds_data.ctypes,
                np.int32(n_loc),
                all_data._microseconds_data.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                np.int32(typ_val),
                allgather,
                np.int32(root),
            )
            c_gatherv(
                data._null_bitmap.ctypes,
                np.int32(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_nulls.ctypes,
                displs_nulls.ctypes,
                char_typ_enum,
                allgather,
                np.int32(root),
            )
            copy_gathered_null_bytes(
                all_data._null_bitmap.ctypes,
                tmp_null_bytes,
                recv_counts_nulls,
                recv_counts,
            )
            return all_data

        return gatherv_impl_int_arr

    if data == boolean_array_type:
        # Nullable booleans need their own implementation because the
        # data array stores 1 bit per boolean. As a result, the data array
        # requires separate handling.
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def gatherv_impl_bool_arr(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_bytes = (n_loc + 7) >> 3
            recv_counts = gather_scalar(np.int32(n_loc), allgather, root=root)
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            recv_counts_bytes = np.empty(1, np.int32)
            displs_bytes = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)
            if rank == root or allgather:
                recv_counts_bytes = np.empty(len(recv_counts), np.int32)
                for i in range(len(recv_counts)):
                    recv_counts_bytes[i] = (recv_counts[i] + 7) >> 3
                displs_bytes = bodo.ir.join.calc_disp(recv_counts_bytes)
                tmp_data_bytes = np.empty(recv_counts_bytes.sum(), np.uint8)
                tmp_null_bytes = np.empty(recv_counts_bytes.sum(), np.uint8)

            c_gatherv(
                data._data.ctypes,
                np.int32(n_bytes),
                tmp_data_bytes.ctypes,
                recv_counts_bytes.ctypes,
                displs_bytes.ctypes,
                char_typ_enum,
                allgather,
                np.int32(root),
            )
            c_gatherv(
                data._null_bitmap.ctypes,
                np.int32(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_bytes.ctypes,
                displs_bytes.ctypes,
                char_typ_enum,
                allgather,
                np.int32(root),
            )
            copy_gathered_null_bytes(
                all_data._data.ctypes,
                tmp_data_bytes,
                recv_counts_bytes,
                recv_counts,
            )
            copy_gathered_null_bytes(
                all_data._null_bitmap.ctypes,
                tmp_null_bytes,
                recv_counts_bytes,
                recv_counts,
            )
            return all_data

        return gatherv_impl_bool_arr

    if (
        isinstance(
            data,
            (
                IntegerArrayType,
                FloatingArrayType,
                DecimalArrayType,
                bodo.TimeArrayType,
                DatetimeArrayType,
            ),
        )
        or data == datetime_date_array_type
    ):
        typ_val = numba_to_c_type(data.dtype)
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def gatherv_impl_int_arr(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_bytes = (n_loc + 7) >> 3
            recv_counts = gather_scalar(np.int32(n_loc), allgather, root=root)
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            displs = np.empty(1, np.int32)
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)
            if rank == root or allgather:
                displs = bodo.ir.join.calc_disp(recv_counts)
                recv_counts_nulls = np.empty(len(recv_counts), np.int32)
                for i in range(len(recv_counts)):
                    recv_counts_nulls[i] = (recv_counts[i] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)
            c_gatherv(
                data._data.ctypes,
                np.int32(n_loc),
                all_data._data.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                np.int32(typ_val),
                allgather,
                np.int32(root),
            )
            c_gatherv(
                data._null_bitmap.ctypes,
                np.int32(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_nulls.ctypes,
                displs_nulls.ctypes,
                char_typ_enum,
                allgather,
                np.int32(root),
            )
            copy_gathered_null_bytes(
                all_data._null_bitmap.ctypes,
                tmp_null_bytes,
                recv_counts_nulls,
                recv_counts,
            )
            return all_data

        return gatherv_impl_int_arr

    # primitive array
    if isinstance(data, bodo.libs.primitive_arr_ext.PrimitiveArrayType):

        def gatherv_prim_arr_impl(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            np_arr = bodo.libs.primitive_arr_ext.primitive_to_np(data)
            all_data = bodo.gatherv(np_arr, allgather, warn_if_rep, root)
            return bodo.libs.primitive_arr_ext.np_to_primitive(all_data)

        return gatherv_prim_arr_impl

    # interval array
    if isinstance(data, IntervalArrayType):
        # gather the left/right arrays
        def impl_interval_arr(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            all_left = bodo.gatherv(data._left, allgather, warn_if_rep, root)
            all_right = bodo.gatherv(data._right, allgather, warn_if_rep, root)
            return bodo.libs.interval_arr_ext.init_interval_array(all_left, all_right)

        return impl_interval_arr

    # TimestampTZ Array
    if data == bodo.timestamptz_array_type:
        ts_typ_enum = np.int32(numba_to_c_type(data.ts_dtype()))
        offset_typ_enum = np.int32(numba_to_c_type(data.offset_dtype()))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def gatherv_impl_timestamp_tz_arr(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_bytes = (n_loc + 7) >> 3
            recv_counts = gather_scalar(np.int32(n_loc), allgather, root=root)
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            displs = np.empty(1, np.int32)
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)
            if rank == root or allgather:
                displs = bodo.ir.join.calc_disp(recv_counts)
                recv_counts_nulls = np.empty(len(recv_counts), np.int32)
                for i in range(len(recv_counts)):
                    recv_counts_nulls[i] = (recv_counts[i] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)
            c_gatherv(
                data.data_ts.ctypes,
                np.int32(n_loc),
                all_data.data_ts.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                ts_typ_enum,
                allgather,
                np.int32(root),
            )
            c_gatherv(
                data.data_offset.ctypes,
                np.int32(n_loc),
                all_data.data_offset.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                offset_typ_enum,
                allgather,
                np.int32(root),
            )
            c_gatherv(
                data._null_bitmap.ctypes,
                np.int32(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_nulls.ctypes,
                displs_nulls.ctypes,
                char_typ_enum,
                allgather,
                np.int32(root),
            )
            copy_gathered_null_bytes(
                all_data._null_bitmap.ctypes,
                tmp_null_bytes,
                recv_counts_nulls,
                recv_counts,
            )
            return all_data

        return gatherv_impl_timestamp_tz_arr

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # gather data
            out_arr = bodo.libs.distributed_api.gatherv(
                arr, allgather, warn_if_rep, root
            )
            out_index = bodo.gatherv(index, allgather, warn_if_rep, root)
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

        return impl

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):
        INT64_MAX = np.iinfo(np.int64).max
        INT64_MIN = np.iinfo(np.int64).min

        def impl_range_index(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            # NOTE: assuming processes have chunks of a global RangeIndex with equal
            # steps. using min/max reductions to get start/stop of global range
            start = data._start
            stop = data._stop
            # ignore empty ranges coming from slicing, see test_getitem_slice
            if len(data) == 0:
                start = INT64_MAX
                stop = INT64_MIN
            start = bodo.libs.distributed_api.dist_reduce(
                start, np.int32(Reduce_Type.Min.value)
            )
            stop = bodo.libs.distributed_api.dist_reduce(
                stop, np.int32(Reduce_Type.Max.value)
            )
            total_len = bodo.libs.distributed_api.dist_reduce(
                len(data), np.int32(Reduce_Type.Sum.value)
            )
            # output is empty if all range chunks are empty
            if start == INT64_MAX and stop == INT64_MIN:
                start = 0
                stop = 0

            # make sure global length is consistent in case the user passes in incorrect
            # RangeIndex chunks (e.g. trivial index in each chunk), see test_rebalance
            l = max(0, -(-(stop - start) // data._step))
            if l < total_len:
                stop = start + data._step * total_len

            # gatherv() of dataframe returns 0-length arrays so index should
            # be 0-length to match
            if bodo.get_rank() != root and not allgather:
                start = 0
                stop = 0
            return bodo.hiframes.pd_index_ext.init_range_index(
                start, stop, data._step, data._name
            )

        return impl_range_index

    # Index types
    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):
        from bodo.hiframes.pd_index_ext import PeriodIndexType

        if isinstance(data, PeriodIndexType):
            freq = data.freq

            def impl_pd_index(
                data, allgather=False, warn_if_rep=True, root=MPI_ROOT
            ):  # pragma: no cover
                arr = bodo.libs.distributed_api.gatherv(
                    data._data, allgather, root=root
                )
                return bodo.hiframes.pd_index_ext.init_period_index(
                    arr, data._name, freq
                )

        else:

            def impl_pd_index(
                data, allgather=False, warn_if_rep=True, root=MPI_ROOT
            ):  # pragma: no cover
                arr = bodo.libs.distributed_api.gatherv(
                    data._data, allgather, root=root
                )
                return bodo.utils.conversion.index_from_array(arr, data._name)

        return impl_pd_index

    # MultiIndex index
    if isinstance(data, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        # just gather the data arrays
        # TODO: handle `levels` and `codes` when available
        def impl_multi_index(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            all_data = bodo.gatherv(data._data, allgather, root=root)
            return bodo.hiframes.pd_multi_index_ext.init_multi_index(
                all_data, data._names, data._name
            )

        return impl_multi_index

    if isinstance(data, bodo.hiframes.table.TableType):
        glbls = {
            "bodo": bodo,
            "get_table_block": bodo.hiframes.table.get_table_block,
            "ensure_column_unboxed": bodo.hiframes.table.ensure_column_unboxed,
            "set_table_block": bodo.hiframes.table.set_table_block,
            "set_table_len": bodo.hiframes.table.set_table_len,
            "alloc_list_like": bodo.hiframes.table.alloc_list_like,
            "init_table": bodo.hiframes.table.init_table,
        }
        func_text = f"def impl_table(data, allgather=False, warn_if_rep=True, root={MPI_ROOT}):\n"
        func_text += "  T = data\n"
        func_text += "  T2 = init_table(T, False)\n"

        for typ, input_blk in data.type_to_blk.items():
            output_blk = data.type_to_blk[typ]
            glbls[f"arr_inds_{input_blk}"] = np.array(
                data.block_to_arr_ind[input_blk], dtype=np.int64
            )
            func_text += (
                f"  arr_list_{input_blk} = get_table_block(T, {input_blk})\n"
                f"  out_arr_list_{input_blk} = alloc_list_like(arr_list_{input_blk}, len(arr_list_{input_blk}), False)\n"
                f"  for i in range(len(arr_list_{input_blk})):\n"
                f"    arr_ind_{input_blk} = arr_inds_{input_blk}[i]\n"
                f"    ensure_column_unboxed(T, arr_list_{input_blk}, i, arr_ind_{input_blk})\n"
                f"    out_arr_{input_blk} = bodo.gatherv(arr_list_{input_blk}[i], allgather, warn_if_rep, root)\n"
                f"    out_arr_list_{input_blk}[i] = out_arr_{input_blk}\n"
                f"  T2 = set_table_block(T2, out_arr_list_{input_blk}, {output_blk})\n"
            )
        func_text += (
            f"  length = T._len if bodo.get_rank() == root or allgather else 0\n"
            f"  T2 = set_table_len(T2, length)\n"
            f"  return T2\n"
        )
        loc_vars = {}
        exec(func_text, glbls, loc_vars)
        impl_table = loc_vars["impl_table"]
        return impl_table

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        # empty dataframe case
        if n_cols == 0:
            __col_name_meta_value_gatherv_no_cols = ColNamesMetaType(())

            def impl(
                data, allgather=False, warn_if_rep=True, root=MPI_ROOT
            ):  # pragma: no cover
                index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)
                g_index = bodo.gatherv(index, allgather, warn_if_rep, root)
                return bodo.hiframes.pd_dataframe_ext.init_dataframe(
                    (), g_index, __col_name_meta_value_gatherv_no_cols
                )

            return impl

        data_args = ", ".join(f"g_data_{i}" for i in range(n_cols))

        func_text = (
            "def impl_df(data, allgather=False, warn_if_rep=True, root={}):\n".format(
                MPI_ROOT
            )
        )
        if data.is_table_format:
            data_args = "T2"
            func_text += (
                "  T = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(data)\n"
                "  T2 = bodo.gatherv(T, allgather, warn_if_rep, root)\n"
            )
        else:
            for i in range(n_cols):
                func_text += "  data_{} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {})\n".format(
                    i, i
                )
                func_text += "  g_data_{} = bodo.gatherv(data_{}, allgather, warn_if_rep, root)\n".format(
                    i, i
                )
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
            "  g_index = bodo.gatherv(index, allgather, warn_if_rep, root)\n"
            f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args},), g_index, __col_name_meta_value_gatherv_with_cols)\n"
        )

        loc_vars = {}
        glbls = {
            "bodo": bodo,
            "__col_name_meta_value_gatherv_with_cols": ColNamesMetaType(data.columns),
        }
        exec(func_text, glbls, loc_vars)
        impl_df = loc_vars["impl_df"]
        return impl_df

    # array(item) array
    if isinstance(data, ArrayItemArrayType):
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def gatherv_array_item_arr_impl(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            offsets_arr = bodo.libs.array_item_arr_ext.get_offsets(data)
            data_arr = bodo.libs.array_item_arr_ext.get_data(data)
            # remove excess data due to possible over-allocation
            data_arr = data_arr[: offsets_arr[-1]]
            null_bitmap_arr = bodo.libs.array_item_arr_ext.get_null_bitmap(data)
            n_loc = len(data)

            # allocate buffer for sending lengths of lists
            send_list_lens = np.empty(n_loc, np.uint32)  # XXX offset type is uint32
            n_bytes = (n_loc + 7) >> 3

            for i in range(n_loc):
                send_list_lens[i] = offsets_arr[i + 1] - offsets_arr[i]

            recv_counts = gather_scalar(np.int32(n_loc), allgather, root=root)
            n_total = recv_counts.sum()

            # displacements
            displs = np.empty(1, np.int32)
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)

            if rank == root or allgather:
                displs = bodo.ir.join.calc_disp(recv_counts)
                recv_counts_nulls = np.empty(len(recv_counts), np.int32)
                for k in range(len(recv_counts)):
                    recv_counts_nulls[k] = (recv_counts[k] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)

            out_lens_arr = np.empty(n_total + 1, np.uint32)
            out_data_arr = bodo.gatherv(data_arr, allgather, warn_if_rep, root)
            out_null_bitmap_arr = np.empty((n_total + 7) >> 3, np.uint8)

            # index offset
            c_gatherv(
                send_list_lens.ctypes,
                np.int32(n_loc),
                out_lens_arr.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                int32_typ_enum,
                allgather,
                np.int32(root),
            )
            # nulls
            c_gatherv(
                null_bitmap_arr.ctypes,
                np.int32(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_nulls.ctypes,
                displs_nulls.ctypes,
                char_typ_enum,
                allgather,
                np.int32(root),
            )
            dummy_use(data)  # needed?

            # TODO: don't hardcode offset type. Also, if offset is 32 bit we can
            # use the same buffer
            out_offsets_arr = np.empty(n_total + 1, np.uint64)
            convert_len_arr_to_offset(
                out_lens_arr.ctypes, out_offsets_arr.ctypes, n_total
            )
            copy_gathered_null_bytes(
                out_null_bitmap_arr.ctypes,
                tmp_null_bytes,
                recv_counts_nulls,
                recv_counts,
            )
            out_arr = bodo.libs.array_item_arr_ext.init_array_item_array(
                n_total, out_data_arr, out_offsets_arr, out_null_bitmap_arr
            )
            return out_arr

        return gatherv_array_item_arr_impl

    if isinstance(data, StructArrayType):
        names = data.names
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_struct_arr(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            data_arrs = bodo.libs.struct_arr_ext.get_data(data)
            null_bitmap = bodo.libs.struct_arr_ext.get_null_bitmap(data)
            out_data_arrs = bodo.gatherv(data_arrs, allgather=allgather, root=root)
            # gather the null bits similar to other arrays
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_bytes = (n_loc + 7) >> 3
            recv_counts = gather_scalar(np.int32(n_loc), allgather, root=root)
            n_total = recv_counts.sum()
            out_null_bitmap = np.empty((n_total + 7) >> 3, np.uint8)
            # displacements
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)
            if rank == root or allgather:
                recv_counts_nulls = np.empty(len(recv_counts), np.int32)
                for i in range(len(recv_counts)):
                    recv_counts_nulls[i] = (recv_counts[i] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)

            c_gatherv(
                null_bitmap.ctypes,
                np.int32(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_nulls.ctypes,
                displs_nulls.ctypes,
                char_typ_enum,
                allgather,
                np.int32(root),
            )
            copy_gathered_null_bytes(
                out_null_bitmap.ctypes,
                tmp_null_bytes,
                recv_counts_nulls,
                recv_counts,
            )
            return bodo.libs.struct_arr_ext.init_struct_arr(
                n_total, out_data_arrs, out_null_bitmap, names
            )

        return impl_struct_arr

    if data == binary_array_type:
        # gather the data array
        def impl_bin_arr(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            all_data = bodo.gatherv(data._data, allgather, warn_if_rep, root)
            return bodo.libs.binary_arr_ext.init_binary_arr(all_data)

        return impl_bin_arr

    if isinstance(data, TupleArrayType):
        # gather the data array
        def impl_tuple_arr(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            all_data = bodo.gatherv(data._data, allgather, warn_if_rep, root)
            return bodo.libs.tuple_arr_ext.init_tuple_arr(all_data)

        return impl_tuple_arr

    if isinstance(data, MapArrayType):
        # gather the data array
        def impl_map_arr(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            all_data = bodo.gatherv(data._data, allgather, warn_if_rep, root)
            return bodo.libs.map_arr_ext.init_map_arr(all_data)

        return impl_map_arr

    # CSR Matrix
    if isinstance(data, CSRMatrixType):

        def impl_csr_matrix(
            data, allgather=False, warn_if_rep=True, root=MPI_ROOT
        ):  # pragma: no cover
            # gather local data
            all_data = bodo.gatherv(data.data, allgather, warn_if_rep, root)
            all_col_inds = bodo.gatherv(data.indices, allgather, warn_if_rep, root)
            all_indptr = bodo.gatherv(data.indptr, allgather, warn_if_rep, root)
            all_local_rows = gather_scalar(data.shape[0], allgather, root=root)
            n_rows = all_local_rows.sum()
            n_cols = bodo.libs.distributed_api.dist_reduce(
                data.shape[1], np.int32(Reduce_Type.Max.value)
            )

            # using np.int64 in output since maximum index value is not known at
            # compilation time
            new_indptr = np.empty(n_rows + 1, np.int64)
            all_col_inds = all_col_inds.astype(np.int64)

            # construct indptr for output
            new_indptr[0] = 0
            out_ind = 1  # current position in output new_indptr
            indptr_ind = 0  # current position in input all_indptr
            for n_loc_rows in all_local_rows:
                for _ in range(n_loc_rows):
                    row_size = all_indptr[indptr_ind + 1] - all_indptr[indptr_ind]
                    new_indptr[out_ind] = new_indptr[out_ind - 1] + row_size
                    out_ind += 1
                    indptr_ind += 1
                indptr_ind += 1  # skip extra since each arr is n_rows + 1

            return bodo.libs.csr_matrix_ext.init_csr_matrix(
                all_data, all_col_inds, new_indptr, (n_rows, n_cols)
            )

        return impl_csr_matrix

    # Tuple of data containers
    if isinstance(data, types.BaseTuple):
        func_text = "def impl_tuple(data, allgather=False, warn_if_rep=True, root={}):\n".format(
            MPI_ROOT
        )
        func_text += "  return ({}{})\n".format(
            ", ".join(
                "bodo.gatherv(data[{}], allgather, warn_if_rep, root)".format(i)
                for i in range(len(data))
            ),
            "," if len(data) > 0 else "",
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl_tuple = loc_vars["impl_tuple"]
        return impl_tuple

    if data is types.none:
        return (
            lambda data, allgather=False, warn_if_rep=True, root=MPI_ROOT: None
        )  # pragma: no cover

    try:
        import bodosql
        from bodosql.context_ext import BodoSQLContextType
    except ImportError:  # pragma: no cover
        BodoSQLContextType = None
    if BodoSQLContextType is not None and isinstance(data, BodoSQLContextType):
        func_text = f"def impl_bodosql_context(data, allgather=False, warn_if_rep=True, root={MPI_ROOT}):\n"
        comma_sep_names = ", ".join([f"'{name}'" for name in data.names])
        comma_sep_dfs = ", ".join(
            [
                f"bodo.gatherv(data.dataframes[{i}], allgather, warn_if_rep, root)"
                for i in range(len(data.dataframes))
            ]
        )
        func_text += f"  return bodosql.context_ext.init_sql_context(({comma_sep_names}, ), ({comma_sep_dfs}, ), data.catalog, None)\n"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)
        impl_bodosql_context = loc_vars["impl_bodosql_context"]
        return impl_bodosql_context
    try:
        import bodosql
        from bodosql import TablePathType
    except ImportError:  # pragma: no cover
        TablePathType = None
    if TablePathType is not None and isinstance(data, TablePathType):
        # Table Path info is all compile time so we return the same data.
        func_text = f"def impl_table_path(data, allgather=False, warn_if_rep=True, root={MPI_ROOT}):\n"
        func_text += f"  return data\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        impl_table_path = loc_vars["impl_table_path"]
        return impl_table_path

    raise BodoError("gatherv() not available for {}".format(data))  # pragma: no cover


def distributed_transpose(arr):  # pragma: no cover
    pass


@overload(distributed_transpose)
def overload_distributed_transpose(arr):
    """Implements distributed array transpose. First lays out data in contiguous chunks
    and calls alltoallv, and then transposes the output of alltoallv.
    See here for example code with similar algorithm:
    https://docs.oracle.com/cd/E19061-01/hpc.cluster5/817-0090-10/1-sided.html
    """
    assert (
        isinstance(arr, types.Array) and arr.ndim == 2
    ), "distributed_transpose: 2D array expected"
    c_type = numba_to_c_type(arr.dtype)

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


@numba.njit
def rebalance(data, dests=None, random=False, random_seed=None, parallel=False):
    return rebalance_impl(data, dests, random, random_seed, parallel)


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
            ", ".join("array_to_info(data_{})".format(x) for x in range(n_cols))
        )
        func_text += "    table_total = arr_info_list_to_table(info_list_total)\n"
        # NOTE: C++ will delete table pointer
        func_text += "    if dests is None:\n"
        func_text += "        out_table = shuffle_renormalization(table_total, random, random_seed, parallel)\n"
        func_text += "    else:\n"
        func_text += "        out_table = shuffle_renormalization_group(table_total, random, random_seed, parallel, len(dests), np.array(dests, dtype=np.int32).ctypes)\n"
        for i_col in range(n_cols):
            func_text += "    out_arr_{0} = array_from_cpp_table(out_table, {0}, data_{0})\n".format(
                i_col
            )
        func_text += (
            "    out_arr_index = array_from_cpp_table(out_table, {}, ind_arr)\n".format(
                n_cols
            )
        )
        func_text += "    delete_table(out_table)\n"
        data_args = ", ".join("out_arr_{}".format(i) for i in range(n_cols))
        index = "bodo.utils.conversion.index_from_array(out_arr_index)"
        func_text += "    return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), {}, __col_name_meta_value_rebalance)\n".format(
            data_args, index
        )
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


@numba.njit
def random_shuffle(data, seed=None, dests=None, n_samples=None, parallel=False):
    return random_shuffle_impl(data, seed, dests, n_samples, parallel)


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


@numba.njit
def allgatherv(data, warn_if_rep=True, root=MPI_ROOT):
    return allgatherv_impl(data, warn_if_rep, root)


@numba.generated_jit(nopython=True)
def allgatherv_impl(data, warn_if_rep=True, root=MPI_ROOT):
    return lambda data, warn_if_rep=True, root=MPI_ROOT: gatherv(
        data, True, warn_if_rep, root
    )  # pragma: no cover


@numba.njit
def get_scatter_null_bytes_buff(
    null_bitmap_ptr, sendcounts, sendcounts_nulls
):  # pragma: no cover
    """copy null bytes into a padded buffer for scatter.
    Padding is needed since processors receive whole bytes and data inside border bytes
    has to be split.
    Only the root rank has the input data and needs to create a valid send buffer.
    """
    # non-root ranks don't have scatter input
    if bodo.get_rank() != MPI_ROOT:
        return np.empty(1, np.uint8)

    null_bytes_buff = np.empty(sendcounts_nulls.sum(), np.uint8)

    curr_tmp_byte = 0  # current location in scatter buffer
    curr_str = 0  # current string in input bitmap

    # for each rank
    for i_rank in range(len(sendcounts)):
        n_strs = sendcounts[i_rank]
        n_bytes = sendcounts_nulls[i_rank]
        chunk_bytes = null_bytes_buff[curr_tmp_byte : curr_tmp_byte + n_bytes]
        # for each string in chunk
        for j in range(n_strs):
            set_bit_to_arr(chunk_bytes, j, get_bit_bitmap(null_bitmap_ptr, curr_str))
            curr_str += 1

        curr_tmp_byte += n_bytes

    return null_bytes_buff


def _bcast_dtype(data, root=MPI_ROOT):
    """broadcast data type from rank 0 using mpi4py"""
    try:
        from bodo.mpi4py import MPI
    except:  # pragma: no cover
        raise BodoError("mpi4py is required for scatterv")

    comm = MPI.COMM_WORLD
    data = comm.bcast(data, root)
    return data


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _get_scatterv_send_counts(send_counts, n_pes, n):
    """compute send counts if 'send_counts' is None."""
    if not is_overload_none(send_counts):
        return lambda send_counts, n_pes, n: send_counts

    def impl(send_counts, n_pes, n):  # pragma: no cover
        # compute send counts if not available
        send_counts = np.empty(n_pes, np.int32)
        for i in range(n_pes):
            send_counts[i] = get_node_portion(n, n_pes, i)
        return send_counts

    return impl


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _scatterv_np(data, send_counts=None, warn_if_dist=True):
    """scatterv() implementation for numpy arrays, refactored here with
    no_cpython_wrapper=True to enable int128 data array of decimal arrays. Otherwise,
    Numba creates a wrapper and complains about unboxing int128.
    """
    typ_val = numba_to_c_type(data.dtype)
    ndim = data.ndim
    dtype = data.dtype
    # using np.dtype since empty() doesn't work with typeref[datetime/timedelta]
    if dtype == types.NPDatetime("ns"):
        dtype = np.dtype("datetime64[ns]")
    elif dtype == types.NPTimedelta("ns"):
        dtype = np.dtype("timedelta64[ns]")
    zero_shape = (0,) * ndim

    def scatterv_arr_impl(
        data, send_counts=None, warn_if_dist=True
    ):  # pragma: no cover
        rank = bodo.libs.distributed_api.get_rank()
        n_pes = bodo.libs.distributed_api.get_size()

        data_in = np.ascontiguousarray(data)
        data_ptr = data.ctypes

        # broadcast shape to all processors
        shape = zero_shape
        if rank == MPI_ROOT:
            shape = data_in.shape
        shape = bcast_tuple(shape)
        n_elem_per_row = get_tuple_prod(shape[1:])

        send_counts = _get_scatterv_send_counts(send_counts, n_pes, shape[0])
        send_counts *= n_elem_per_row

        # allocate output
        n_loc = send_counts[rank]  # total number of elements on this PE
        recv_data = np.empty(n_loc, dtype)

        # displacements
        displs = bodo.ir.join.calc_disp(send_counts)

        c_scatterv(
            data_ptr,
            send_counts.ctypes,
            displs.ctypes,
            recv_data.ctypes,
            np.int32(n_loc),
            np.int32(typ_val),
        )

        # handle multi-dim case
        return recv_data.reshape((-1,) + shape[1:])

    return scatterv_arr_impl


# skipping coverage since only called on multiple core case
def _get_name_value_for_type(name_typ):  # pragma: no cover
    """get a value for name of a Series/Index type"""
    # assuming name is either None or a string
    assert (
        isinstance(name_typ, (types.UnicodeType, types.StringLiteral))
        or name_typ == types.none
    )
    # make names unique with next_label to avoid MultiIndex unboxing issue #811
    return None if name_typ == types.none else "_" + str(ir_utils.next_label())


# skipping coverage since only called on multiple core case
def get_value_for_type(dtype):  # pragma: no cover
    """returns a value of type 'dtype' to enable calling an njit function with the
    proper input type.
    """
    # object arrays like decimal array can't be empty since they are not typed so we
    # create all arrays with size of 1 to be consistent

    # numpy arrays
    if isinstance(dtype, types.Array):
        return np.zeros((1,) * dtype.ndim, numba.np.numpy_support.as_dtype(dtype.dtype))

    # string array
    if dtype == string_array_type:
        return pd.array(["A"], "string")

    if dtype == bodo.dict_str_arr_type:
        import pyarrow as pa

        return pa.array(["a"], type=pa.dictionary(pa.int32(), pa.large_string()))

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
        pd_dtype = "Float{}".format(dtype.dtype.bitwidth)
        return pd.array([3.0], pd_dtype)

    # bool array
    if dtype == boolean_array_type:
        return pd.array([True], "boolean")

    # Decimal array
    if isinstance(dtype, DecimalArrayType):
        return np.array([Decimal("32.1")])

    # date array
    if dtype == datetime_date_array_type:
        return np.array([datetime.date(2011, 8, 9)])

    # timedelta array
    if dtype == datetime_timedelta_array_type:
        return np.array([datetime.timedelta(33)])

    # Index types
    if bodo.hiframes.pd_index_ext.is_pd_index_type(dtype):
        name = _get_name_value_for_type(dtype.name_typ)
        if isinstance(dtype, bodo.hiframes.pd_index_ext.RangeIndexType):
            return pd.RangeIndex(1, name=name)
        arr_type = bodo.utils.typing.get_index_data_arr_types(dtype)[0]
        arr = get_value_for_type(arr_type)
        return pd.Index(arr, name=name)

    # MultiIndex index
    if isinstance(dtype, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        import pyarrow as pa

        name = _get_name_value_for_type(dtype.name_typ)
        names = tuple(_get_name_value_for_type(t) for t in dtype.names_typ)
        arrs = tuple(get_value_for_type(t) for t in dtype.array_types)
        # convert pyarrow arrays to numpy to avoid errors in pd.MultiIndex.from_arrays
        arrs = tuple(a.to_numpy(False) if isinstance(a, pa.Array) else a for a in arrs)
        val = pd.MultiIndex.from_arrays(arrs, names=names)
        val.name = name
        return val

    if isinstance(dtype, bodo.hiframes.pd_series_ext.SeriesType):
        name = _get_name_value_for_type(dtype.name_typ)
        arr = get_value_for_type(dtype.data)
        index = get_value_for_type(dtype.index)
        return pd.Series(arr, index, name=name)

    if isinstance(dtype, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        arrs = tuple(get_value_for_type(t) for t in dtype.data)
        index = get_value_for_type(dtype.index)
        return pd.DataFrame(
            {name: arr for name, arr in zip(dtype.columns, arrs)}, index
        )

    if isinstance(dtype, CategoricalArrayType):
        return pd.Categorical.from_codes([0], dtype.dtype.categories)

    if isinstance(dtype, types.BaseTuple):
        return tuple(get_value_for_type(t) for t in dtype.types)

    if isinstance(dtype, ArrayItemArrayType):
        return pd.Series(
            [get_value_for_type(dtype.dtype), get_value_for_type(dtype.dtype)]
        ).values

    if isinstance(dtype, IntervalArrayType):
        arr_type = get_value_for_type(dtype.arr_type)
        return pd.arrays.IntervalArray([pd.Interval(arr_type[0], arr_type[0])])

    if dtype == bodo.timestamptz_array_type:
        return np.array([bodo.TimestampTZ(pd.Timestamp(0), 0)])

    # TODO: Add missing data types
    raise BodoError(f"get_value_for_type(dtype): Missing data type {dtype}")


def scatterv(data, send_counts=None, warn_if_dist=True):
    """scatterv() distributes data from rank 0 to all ranks.
    Rank 0 passes the data but the other ranks should just pass None.
    """
    rank = bodo.libs.distributed_api.get_rank()
    if rank != MPI_ROOT and data is not None:  # pragma: no cover
        warnings.warn(
            BodoWarning(
                "bodo.scatterv(): A non-None value for 'data' was found on a rank other than the root. "
                "This data won't be sent to any other ranks and will be overwritten with data from rank 0."
            )
        )

    # make sure all ranks receive the proper data type as input (instead of None)
    dtype = bodo.typeof(data)
    dtype = _bcast_dtype(dtype)
    if rank != MPI_ROOT:
        data = get_value_for_type(dtype)

    return scatterv_impl(data, send_counts)


@overload(scatterv)
def scatterv_overload(data, send_counts=None, warn_if_dist=True):
    """support scatterv inside jit functions"""
    bodo.hiframes.pd_dataframe_ext.check_runtime_cols_unsupported(
        data, "bodo.scatterv()"
    )
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(data, "bodo.scatterv()")
    return lambda data, send_counts=None, warn_if_dist=True: scatterv_impl_jit(
        data, send_counts
    )  # pragma: no cover


@numba.njit
def scatterv_impl(data, send_counts=None, warn_if_dist=True):
    return scatterv_impl_jit(data, send_counts, warn_if_dist)


@numba.generated_jit(nopython=True)
def scatterv_impl_jit(data, send_counts=None, warn_if_dist=True):
    """nopython implementation of scatterv()"""
    if isinstance(data, types.Array):
        return lambda data, send_counts=None, warn_if_dist=True: _scatterv_np(
            data, send_counts
        )  # pragma: no cover

    if data in (string_array_type, binary_array_type):
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        if data == binary_array_type:
            alloc_fn = "bodo.libs.binary_arr_ext.pre_alloc_binary_array"
        else:
            alloc_fn = "bodo.libs.str_arr_ext.pre_alloc_string_array"

        func_text = f"""def impl(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_pes = bodo.libs.distributed_api.get_size()
            n_all = bodo.libs.distributed_api.bcast_scalar(len(data))

            # convert offsets to lengths of strings
            send_arr_lens = np.empty(
                len(data), np.uint32
            )  # XXX offset type is offset_type, lengths for comm are uint32
            for i in range(len(data)):
                send_arr_lens[i] = bodo.libs.str_arr_ext.get_str_arr_item_length(
                    data, i
                )

            # ------- calculate buffer counts -------

            send_counts = bodo.libs.distributed_api._get_scatterv_send_counts(send_counts, n_pes, n_all)

            # displacements
            displs = bodo.ir.join.calc_disp(send_counts)

            # compute send counts for characters
            send_counts_char = np.empty(n_pes, np.int32)
            if rank == 0:
                curr_str = 0
                for i in range(n_pes):
                    c = 0
                    for _ in range(send_counts[i]):
                        c += send_arr_lens[curr_str]
                        curr_str += 1
                    send_counts_char[i] = c

            bodo.libs.distributed_api.bcast(send_counts_char)

            # displacements for characters
            displs_char = bodo.ir.join.calc_disp(send_counts_char)

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            # alloc output array
            n_loc = send_counts[rank]  # total number of elements on this PE
            n_loc_char = send_counts_char[rank]
            recv_arr = {alloc_fn}(n_loc, n_loc_char)

            # ----- string lengths -----------

            recv_lens = np.empty(n_loc, np.uint32)
            bodo.libs.distributed_api.c_scatterv(
                send_arr_lens.ctypes,
                send_counts.ctypes,
                displs.ctypes,
                recv_lens.ctypes,
                np.int32(n_loc),
                int32_typ_enum,
            )

            # TODO: don't hardcode offset type. Also, if offset is 32 bit we can
            # use the same buffer
            bodo.libs.str_arr_ext.convert_len_arr_to_offset(recv_lens.ctypes, bodo.libs.str_arr_ext.get_offset_ptr(recv_arr), n_loc)

            # ----- string characters -----------

            bodo.libs.distributed_api.c_scatterv(
                bodo.libs.str_arr_ext.get_data_ptr(data),
                send_counts_char.ctypes,
                displs_char.ctypes,
                bodo.libs.str_arr_ext.get_data_ptr(recv_arr),
                np.int32(n_loc_char),
                char_typ_enum,
            )

            # ----------- null bitmap -------------

            n_recv_bytes = (n_loc + 7) >> 3

            send_null_bitmap = bodo.libs.distributed_api.get_scatter_null_bytes_buff(
                bodo.libs.str_arr_ext.get_null_bitmap_ptr(data), send_counts, send_counts_nulls
            )

            bodo.libs.distributed_api.c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                bodo.libs.str_arr_ext.get_null_bitmap_ptr(recv_arr),
                np.int32(n_recv_bytes),
                char_typ_enum,
            )

            return recv_arr"""

        loc_vars = dict()
        exec(
            func_text,
            {
                "bodo": bodo,
                "np": np,
                "int32_typ_enum": int32_typ_enum,
                "char_typ_enum": char_typ_enum,
                "decode_if_dict_array": decode_if_dict_array,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    if isinstance(data, ArrayItemArrayType):
        # Code adapted from the string code. Both the string and array(item) codes should be
        # refactored.
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def scatterv_array_item_impl(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            in_offsets_arr = bodo.libs.array_item_arr_ext.get_offsets(data)
            in_data_arr = bodo.libs.array_item_arr_ext.get_data(data)
            in_data_arr = in_data_arr[: in_offsets_arr[-1]]
            in_null_bitmap_arr = bodo.libs.array_item_arr_ext.get_null_bitmap(data)

            rank = bodo.libs.distributed_api.get_rank()
            n_pes = bodo.libs.distributed_api.get_size()
            n_all = bcast_scalar(len(data))

            # convert offsets to lengths of lists
            send_arr_lens = np.empty(
                len(data), np.uint32
            )  # XXX offset type is offset_type
            for i in range(len(data)):
                send_arr_lens[i] = in_offsets_arr[i + 1] - in_offsets_arr[i]

            # ------- calculate buffer counts -------

            send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)

            # displacements
            displs = bodo.ir.join.calc_disp(send_counts)

            # compute send counts for items
            send_counts_item = np.empty(n_pes, np.int32)
            if rank == 0:
                curr_item = 0
                for i in range(n_pes):
                    c = 0
                    for _ in range(send_counts[i]):
                        c += send_arr_lens[curr_item]
                        curr_item += 1
                    send_counts_item[i] = c

            bcast(send_counts_item)

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            # alloc output array
            n_loc = send_counts[rank]  # total number of elements on this PE
            recv_offsets_arr = np.empty(n_loc + 1, np_offset_type)

            recv_data_arr = bodo.libs.distributed_api.scatterv_impl(
                in_data_arr, send_counts_item
            )
            n_recv_null_bytes = (n_loc + 7) >> 3
            recv_null_bitmap_arr = np.empty(n_recv_null_bytes, np.uint8)

            # ----- list of item lengths -----------

            recv_lens = np.empty(n_loc, np.uint32)
            c_scatterv(
                send_arr_lens.ctypes,
                send_counts.ctypes,
                displs.ctypes,
                recv_lens.ctypes,
                np.int32(n_loc),
                int32_typ_enum,
            )

            # TODO: don't hardcode offset type. Also, if offset is 32 bit we can
            # use the same buffer
            convert_len_arr_to_offset(recv_lens.ctypes, recv_offsets_arr.ctypes, n_loc)

            # ----------- null bitmap -------------

            send_null_bitmap = get_scatter_null_bytes_buff(
                in_null_bitmap_arr.ctypes, send_counts, send_counts_nulls
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                recv_null_bitmap_arr.ctypes,
                np.int32(n_recv_null_bytes),
                char_typ_enum,
            )

            return bodo.libs.array_item_arr_ext.init_array_item_array(
                n_loc, recv_data_arr, recv_offsets_arr, recv_null_bitmap_arr
            )

        return scatterv_array_item_impl

    if data == boolean_array_type:
        # Nullable booleans need their own implementation because the
        # data array stores 1 bit per boolean. As a result, the counts may split
        # may split the data array mid-byte, so we need to handle it the same
        # way we handle the null bitmap. The send count also doesn't reflect the
        # number of bytes to send, so we need to calculate that separately.
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def scatterv_impl_bool_arr(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_pes = bodo.libs.distributed_api.get_size()
            data_in = data._data
            null_bitmap = data._null_bitmap
            # Calculate the displacements for nulls and data, each of
            # which is a single bit.
            n_in = len(data)
            n_all = bcast_scalar(n_in)

            send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)
            # Calculate the local N for how many elements are in the array
            n_loc = np.int64(send_counts[rank])
            # compute send counts bytes
            send_counts_bytes = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                send_counts_bytes[i] = (send_counts[i] + 7) >> 3

            displs_bytes = bodo.ir.join.calc_disp(send_counts_bytes)

            send_data_bitmap = get_scatter_null_bytes_buff(
                data_in.ctypes, send_counts, send_counts_bytes
            )
            send_null_bitmap = get_scatter_null_bytes_buff(
                null_bitmap.ctypes, send_counts, send_counts_bytes
            )
            # Allocate the output arrays
            n_recv_bytes = send_counts_bytes[rank]
            data_recv = np.empty(n_recv_bytes, np.uint8)
            bitmap_recv = np.empty(n_recv_bytes, np.uint8)

            c_scatterv(
                send_data_bitmap.ctypes,
                send_counts_bytes.ctypes,
                displs_bytes.ctypes,
                data_recv.ctypes,
                np.int32(n_recv_bytes),
                char_typ_enum,
            )
            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_bytes.ctypes,
                displs_bytes.ctypes,
                bitmap_recv.ctypes,
                np.int32(n_recv_bytes),
                char_typ_enum,
            )
            return bodo.libs.bool_arr_ext.init_bool_array(data_recv, bitmap_recv, n_loc)

        return scatterv_impl_bool_arr

    if (
        isinstance(
            data,
            (IntegerArrayType, FloatingArrayType, DecimalArrayType, DatetimeArrayType),
        )
        or data == datetime_date_array_type
    ):
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        # these array need a data array and a null bitmap array to be initialized by
        # their init functions
        if isinstance(data, IntegerArrayType):
            init_func = bodo.libs.int_arr_ext.init_integer_array
        if isinstance(data, FloatingArrayType):
            init_func = bodo.libs.float_arr_ext.init_float_array
        if isinstance(data, DecimalArrayType):
            precision = data.precision
            scale = data.scale
            init_func = numba.njit(no_cpython_wrapper=True)(
                lambda d, b: bodo.libs.decimal_arr_ext.init_decimal_array(
                    d, b, precision, scale
                )  # pragma: no cover
            )
        if data == datetime_date_array_type:
            init_func = bodo.hiframes.datetime_date_ext.init_datetime_date_array

        def scatterv_impl_int_arr(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            n_pes = bodo.libs.distributed_api.get_size()

            data_in = data._data
            null_bitmap = data._null_bitmap
            n_in = len(data_in)

            data_recv = _scatterv_np(data_in, send_counts)

            n_all = bcast_scalar(n_in)
            n_recv_bytes = (len(data_recv) + 7) >> 3
            bitmap_recv = np.empty(n_recv_bytes, np.uint8)

            send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            send_null_bitmap = get_scatter_null_bytes_buff(
                null_bitmap.ctypes, send_counts, send_counts_nulls
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                bitmap_recv.ctypes,
                np.int32(n_recv_bytes),
                char_typ_enum,
            )
            return init_func(data_recv, bitmap_recv)

        return scatterv_impl_int_arr

    # interval array
    if isinstance(data, IntervalArrayType):
        # scatter the left/right arrays
        def impl_interval_arr(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            left_chunk = bodo.libs.distributed_api.scatterv_impl(
                data._left, send_counts
            )
            right_chunk = bodo.libs.distributed_api.scatterv_impl(
                data._right, send_counts
            )
            return bodo.libs.interval_arr_ext.init_interval_array(
                left_chunk, right_chunk
            )

        return impl_interval_arr

    # TimestampTZ array
    if data == bodo.timestamptz_array_type:
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_timestamp_tz_arr(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            n_pes = bodo.libs.distributed_api.get_size()

            data_ts_in = data.data_ts
            data_offset_in = data.data_offset
            null_bitmap = data._null_bitmap
            n_in = len(data_ts_in)

            data_ts_recv = _scatterv_np(data_ts_in, send_counts)
            data_offset_recv = _scatterv_np(data_offset_in, send_counts)

            n_all = bcast_scalar(n_in)
            n_recv_bytes = (len(data_ts_recv) + 7) >> 3
            bitmap_recv = np.empty(n_recv_bytes, np.uint8)

            send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            send_null_bitmap = get_scatter_null_bytes_buff(
                null_bitmap.ctypes, send_counts, send_counts_nulls
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                bitmap_recv.ctypes,
                np.int32(n_recv_bytes),
                char_typ_enum,
            )
            return bodo.hiframes.timestamptz_ext.init_timestamptz_array(
                data_ts_recv, data_offset_recv, bitmap_recv
            )

        return impl_timestamp_tz_arr

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):
        # TODO: support send_counts
        def impl_range_index(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_pes = bodo.libs.distributed_api.get_size()

            start = data._start
            stop = data._stop
            step = data._step
            name = data._name

            name = bcast_scalar(name)

            start = bcast_scalar(start)
            stop = bcast_scalar(stop)
            step = bcast_scalar(step)
            n_items = bodo.libs.array_kernels.calc_nitems(start, stop, step)
            chunk_start = bodo.libs.distributed_api.get_start(n_items, n_pes, rank)
            chunk_count = bodo.libs.distributed_api.get_node_portion(
                n_items, n_pes, rank
            )
            new_start = start + step * chunk_start
            new_stop = start + step * (chunk_start + chunk_count)
            new_stop = min(new_stop, stop)
            return bodo.hiframes.pd_index_ext.init_range_index(
                new_start, new_stop, step, name
            )

        return impl_range_index

    # Period index requires special handling because index_from_array
    # doesn't work properly (can't infer the index).
    # See [BE-2067]
    if isinstance(data, bodo.hiframes.pd_index_ext.PeriodIndexType):
        freq = data.freq

        def impl_period_index(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            data_in = data._data
            name = data._name
            name = bcast_scalar(name)
            arr = bodo.libs.distributed_api.scatterv_impl(data_in, send_counts)
            return bodo.hiframes.pd_index_ext.init_period_index(arr, name, freq)

        return impl_period_index

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):

        def impl_pd_index(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            data_in = data._data
            name = data._name
            name = bcast_scalar(name)
            arr = bodo.libs.distributed_api.scatterv_impl(data_in, send_counts)
            return bodo.utils.conversion.index_from_array(arr, name)

        return impl_pd_index

    # MultiIndex index
    if isinstance(data, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        # TODO: handle `levels` and `codes` when available
        def impl_multi_index(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            all_data = bodo.libs.distributed_api.scatterv_impl(data._data, send_counts)
            name = bcast_scalar(data._name)
            names = bcast_tuple(data._names)
            return bodo.hiframes.pd_multi_index_ext.init_multi_index(
                all_data, names, name
            )

        return impl_multi_index

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl_series(data, send_counts=None, warn_if_dist=True):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # scatter data
            out_name = bcast_scalar(name)
            out_arr = bodo.libs.distributed_api.scatterv_impl(arr, send_counts)
            out_index = bodo.libs.distributed_api.scatterv_impl(index, send_counts)
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, out_name)

        return impl_series

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        __col_name_meta_scaterv_impl = ColNamesMetaType(data.columns)

        func_text = "def impl_df(data, send_counts=None, warn_if_dist=True):\n"
        if data.is_table_format:
            func_text += (
                "  table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(data)\n"
            )
            func_text += "  g_table = bodo.libs.distributed_api.scatterv_impl(table, send_counts)\n"
            data_args = "g_table"
        else:
            for i in range(n_cols):
                func_text += f"  data_{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {i})\n"
                func_text += f"  g_data_{i} = bodo.libs.distributed_api.scatterv_impl(data_{i}, send_counts)\n"
            data_args = ", ".join(f"g_data_{i}" for i in range(n_cols))
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
        )
        func_text += (
            "  g_index = bodo.libs.distributed_api.scatterv_impl(index, send_counts)\n"
        )
        func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args},), g_index, __col_name_meta_scaterv_impl)\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "__col_name_meta_scaterv_impl": __col_name_meta_scaterv_impl,
            },
            loc_vars,
        )
        impl_df = loc_vars["impl_df"]
        return impl_df

    if isinstance(data, bodo.TableType):
        func_text = "def impl_table(data, send_counts=None, warn_if_dist=True):\n"
        func_text += "  T = data\n"
        func_text += "  T2 = init_table(T, False)\n"
        func_text += "  l = 0\n"

        glbls = {}
        for blk in data.type_to_blk.values():
            glbls[f"arr_inds_{blk}"] = np.array(
                data.block_to_arr_ind[blk], dtype=np.int64
            )
            func_text += f"  arr_list_{blk} = get_table_block(T, {blk})\n"
            func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_{blk}, len(arr_list_{blk}), False)\n"
            func_text += f"  for i in range(len(arr_list_{blk})):\n"
            func_text += f"    arr_ind_{blk} = arr_inds_{blk}[i]\n"
            func_text += (
                f"    ensure_column_unboxed(T, arr_list_{blk}, i, arr_ind_{blk})\n"
            )
            func_text += f"    out_arr_{blk} = bodo.libs.distributed_api.scatterv_impl(arr_list_{blk}[i], send_counts)\n"
            func_text += f"    out_arr_list_{blk}[i] = out_arr_{blk}\n"
            func_text += f"    l = len(out_arr_{blk})\n"
            func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"
        func_text += f"  T2 = set_table_len(T2, l)\n"
        func_text += f"  return T2\n"

        glbls.update(
            {
                "bodo": bodo,
                "init_table": bodo.hiframes.table.init_table,
                "get_table_block": bodo.hiframes.table.get_table_block,
                "ensure_column_unboxed": bodo.hiframes.table.ensure_column_unboxed,
                "set_table_block": bodo.hiframes.table.set_table_block,
                "set_table_len": bodo.hiframes.table.set_table_len,
                "alloc_list_like": bodo.hiframes.table.alloc_list_like,
            }
        )
        loc_vars = {}
        exec(func_text, glbls, loc_vars)
        return loc_vars["impl_table"]

    if data == bodo.dict_str_arr_type:

        def impl_dict_arr(
            data, send_counts=None, warn_if_dist=True
        ):  # pragma: no cover
            # broadcast the dictionary data (string array)
            # (needs length and number of chars to be broadcast first for pre-allocation
            # of output string array)
            if bodo.get_rank() == 0:
                str_arr = data._data
                bodo.libs.distributed_api.bcast_scalar(len(str_arr))
                bodo.libs.distributed_api.bcast_scalar(
                    np.int64(bodo.libs.str_arr_ext.num_total_chars(str_arr))
                )
            else:
                l = bodo.libs.distributed_api.bcast_scalar(0)
                n_chars = bodo.libs.distributed_api.bcast_scalar(0)
                str_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, n_chars)
            bodo.libs.distributed_api.bcast(str_arr)
            # scatter indices array
            new_indices = bodo.libs.distributed_api.scatterv_impl(
                data._indices, send_counts
            )
            # the dictionary is global by construction (broadcast)
            return bodo.libs.dict_arr_ext.init_dict_arr(
                str_arr, new_indices, True, data._has_unique_local_dictionary, None
            )

        return impl_dict_arr

    if isinstance(data, CategoricalArrayType):

        def impl_cat(data, send_counts=None, warn_if_dist=True):  # pragma: no cover
            codes = bodo.libs.distributed_api.scatterv_impl(data.codes, send_counts)
            return bodo.hiframes.pd_categorical_ext.init_categorical_array(
                codes, data.dtype
            )

        return impl_cat

    # Tuple of data containers
    if isinstance(data, types.BaseTuple):
        func_text = "def impl_tuple(data, send_counts=None, warn_if_dist=True):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(
                f"bodo.libs.distributed_api.scatterv_impl(data[{i}], send_counts)"
                for i in range(len(data))
            ),
            "," if len(data) > 0 else "",
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl_tuple = loc_vars["impl_tuple"]
        return impl_tuple

    if data is types.none:  # pragma: no cover
        return lambda data, send_counts=None, warn_if_dist=True: None

    raise BodoError("scatterv() not available for {}".format(data))  # pragma: no cover


@intrinsic
def cptr_to_voidptr(typingctx, cptr_tp=None):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], lir.IntType(8).as_pointer())

    return types.voidptr(cptr_tp), codegen


# TODO: test
# TODO: large BCast


def bcast(data, root=MPI_ROOT):  # pragma: no cover
    return


@overload(bcast, no_unliteral=True)
def bcast_overload(data, root=MPI_ROOT):
    """broadcast array from rank root. 'data' array is assumed to be pre-allocated in
    non-root ranks.
    """
    # numpy arrays
    if isinstance(data, types.Array):

        def bcast_impl(data, root=MPI_ROOT):  # pragma: no cover
            typ_enum = get_type_enum(data)
            count = data.size
            assert count < INT_MAX
            c_bcast(
                data.ctypes,
                np.int32(count),
                typ_enum,
                np.array([-1]).ctypes,
                0,
                np.int32(root),
            )
            return

        return bcast_impl

    # Decimal arrays
    if isinstance(data, DecimalArrayType):

        def bcast_decimal_arr(data, root=MPI_ROOT):  # pragma: no cover
            count = data._data.size
            assert count < INT_MAX
            c_bcast(
                data._data.ctypes,
                np.int32(count),
                CTypeEnum.Int128.value,
                np.array([-1]).ctypes,
                0,
                np.int32(root),
            )
            bcast(data._null_bitmap, root)
            return

        return bcast_decimal_arr

    # nullable int/float/bool/date/time arrays
    if isinstance(
        data, (IntegerArrayType, FloatingArrayType, TimeArrayType, DatetimeArrayType)
    ) or data in (
        boolean_array_type,
        datetime_date_array_type,
    ):

        def bcast_impl_int_arr(data, root=MPI_ROOT):  # pragma: no cover
            bcast(data._data, root)
            bcast(data._null_bitmap, root)
            return

        return bcast_impl_int_arr

    # string arrays
    if is_str_arr_type(data) or data == binary_array_type:
        offset_typ_enum = np.int32(numba_to_c_type(offset_type))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def bcast_str_impl(data, root=MPI_ROOT):  # pragma: no cover
            data = decode_if_dict_array(data)
            n_loc = len(data)
            n_all_chars = num_total_chars(data)
            assert n_loc < INT_MAX
            assert n_all_chars < INT_MAX

            offset_ptr = get_offset_ptr(data)
            data_ptr = get_data_ptr(data)
            null_bitmap_ptr = get_null_bitmap_ptr(data)
            n_bytes = (n_loc + 7) >> 3

            c_bcast(
                offset_ptr,
                np.int32(n_loc + 1),
                offset_typ_enum,
                np.array([-1]).ctypes,
                0,
                np.int32(root),
            )
            c_bcast(
                data_ptr,
                np.int32(n_all_chars),
                char_typ_enum,
                np.array([-1]).ctypes,
                0,
                np.int32(root),
            )
            c_bcast(
                null_bitmap_ptr,
                np.int32(n_bytes),
                char_typ_enum,
                np.array([-1]).ctypes,
                0,
                np.int32(root),
            )

        return bcast_str_impl


# sendbuf, sendcount, dtype, comm_ranks, nranks_in_comm
c_bcast = types.ExternalFunction(
    "c_bcast",
    types.void(
        types.voidptr, types.int32, types.int32, types.voidptr, types.int32, types.int32
    ),
)


@numba.njit
def bcast_scalar(val, root=MPI_ROOT):
    """broadcast for a scalar value.
    Assumes all ranks `val` has same type.
    """
    return bcast_scalar_impl(val, root)


def bcast_scalar_impl(val, root=MPI_ROOT):  # pragma: no cover
    return


@infer_global(bcast_scalar_impl)
class BcastScalarInfer(AbstractTemplate):
    def generic(self, args, kws):
        pysig = numba.core.utils.pysignature(bcast_scalar_impl)
        folded_args = bodo.utils.transform.fold_argument_types(pysig, args, kws)
        assert len(folded_args) == 2
        val = args[0]

        if not (
            isinstance(
                val,
                (
                    types.Integer,
                    types.Float,
                    bodo.PandasTimestampType,
                ),
            )
            or val
            in [
                bodo.datetime64ns,
                bodo.timedelta64ns,
                bodo.string_type,
                types.none,
                types.bool_,
                bodo.datetime_date_type,
                bodo.timestamptz_type,
            ]
        ):
            raise BodoError(
                f"bcast_scalar requires an argument of type Integer, Float, datetime64ns, timestamptz, timedelta64ns, string, None, or Bool. Found type {val}"
            )

        return signature(val, *folded_args)


def gen_bcast_scalar_impl(val, root=MPI_ROOT):
    if val == types.none:
        return lambda val, root=MPI_ROOT: None

    if val == bodo.timestamptz_type:

        def impl(val, root=MPI_ROOT):  # pragma: no cover
            updated_timestamp = bodo.libs.distributed_api.bcast_scalar(
                val.utc_timestamp, root
            )
            updated_offset = bodo.libs.distributed_api.bcast_scalar(
                val.offset_minutes, root
            )
            return bodo.TimestampTZ(updated_timestamp, updated_offset)

        return impl

    if val == bodo.datetime_date_type:
        c_type = numba_to_c_type(types.int32)

        # Note: There are issues calling this function with recursion.
        # As a result we just implement it directly.
        def impl(val, root=MPI_ROOT):  # pragma: no cover
            send = np.empty(1, np.int32)
            send[0] = bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int(val)
            c_bcast(
                send.ctypes,
                np.int32(1),
                np.int32(c_type),
                np.array([-1]).ctypes,
                0,
                np.int32(root),
            )
            return bodo.hiframes.datetime_date_ext.cast_int_to_datetime_date(send[0])

        return impl

    if isinstance(val, bodo.PandasTimestampType):
        c_type = numba_to_c_type(types.int64)
        tz = val.tz

        # Note: There are issues calling this function with recursion.
        # As a result we just implement it directly.
        def impl(val, root=MPI_ROOT):  # pragma: no cover
            send = np.empty(1, np.int64)
            send[0] = val.value
            c_bcast(
                send.ctypes,
                np.int32(1),
                np.int32(c_type),
                np.array([-1]).ctypes,
                0,
                np.int32(root),
            )
            # Use convert_val_to_timestamp to other modifying the value
            return pd.Timestamp(send[0], tz=tz)

        return impl

    if val == bodo.string_type:
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_str(val, root=MPI_ROOT):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            if rank != root:
                n_char = 0
                utf8_str = np.empty(0, np.uint8).ctypes
            else:
                utf8_str, n_char = bodo.libs.str_ext.unicode_to_utf8_and_len(val)
            n_char = bodo.libs.distributed_api.bcast_scalar(n_char, root)
            if rank != root:
                # add null termination character
                utf8_str_arr = np.empty(n_char + 1, np.uint8)
                utf8_str_arr[n_char] = 0
                utf8_str = utf8_str_arr.ctypes
            c_bcast(
                utf8_str,
                np.int32(n_char),
                char_typ_enum,
                np.array([-1]).ctypes,
                0,
                np.int32(root),
            )
            return bodo.libs.str_arr_ext.decode_utf8(utf8_str, n_char)

        return impl_str

    # TODO: other types like boolean
    typ_val = numba_to_c_type(val)
    # TODO: fix np.full and refactor
    func_text = (
        f"def bcast_scalar_impl(val, root={MPI_ROOT}):\n"
        "  send = np.empty(1, dtype)\n"
        "  send[0] = val\n"
        f"  c_bcast(send.ctypes, np.int32(1), np.int32({typ_val}), np.array([-1]).ctypes, 0, np.int32(root))\n"
        "  return send[0]\n"
    )

    dtype = numba.np.numpy_support.as_dtype(val)
    loc_vars = {}
    exec(
        func_text,
        {"bodo": bodo, "np": np, "c_bcast": c_bcast, "dtype": dtype},
        loc_vars,
    )
    bcast_scalar_impl = loc_vars["bcast_scalar_impl"]
    return bcast_scalar_impl


@lower_builtin(bcast_scalar_impl, types.Any, types.VarArg(types.Any))
def bcast_scalar_impl_any(context, builder, sig, args):
    impl = gen_bcast_scalar_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@numba.njit
def bcast_tuple(val, root=MPI_ROOT):
    return bcast_tuple_impl_jit(val, root)


@numba.generated_jit(nopython=True)
def bcast_tuple_impl_jit(val, root=MPI_ROOT):
    """broadcast a tuple value
    calls bcast_scalar() on individual elements
    """
    assert isinstance(
        val, types.BaseTuple
    ), "Internal Error: Argument to bcast tuple must be of type tuple"
    n_elem = len(val)
    func_text = f"def bcast_tuple_impl(val, root={MPI_ROOT}):\n"
    func_text += "  return ({}{})".format(
        ",".join("bcast_scalar(val[{}], root)".format(i) for i in range(n_elem)),
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
def prealloc_str_for_bcast(arr, root=MPI_ROOT):  # pragma: no cover
    return arr


@overload(prealloc_str_for_bcast, no_unliteral=True)
def prealloc_str_for_bcast_overload(arr, root=MPI_ROOT):
    if arr == string_array_type:

        def prealloc_impl(arr, root=MPI_ROOT):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = bcast_scalar(len(arr), root)
            n_all_char = bcast_scalar(np.int64(num_total_chars(arr)), root)
            if rank != root:
                arr = pre_alloc_string_array(n_loc, n_all_char)
            return arr

        return prealloc_impl

    return lambda arr, root=MPI_ROOT: arr


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


dummy_use = numba.njit(no_cpython_wrapper=True)(lambda a: None)


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
    if data == bodo.string_type:
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
    if is_str_arr_type(arr) or arr == bodo.binary_array_type:
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
            c_bcast(
                data_ptr,
                np.int32(l),
                char_typ_enum,
                np.array([-1]).ctypes,
                0,
                np.int32(root),
            )
            val = transform_str_getitem_output(val, l)
            return val

        return str_getitem_impl

    if isinstance(arr, bodo.CategoricalArrayType):
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

    if arr == bodo.null_array_type:

        def null_getitem_impl(
            arr, ind, arr_start, total_len, is_1D
        ):  # pragma: no cover
            if ind >= total_len:
                raise IndexError("index out of bounds")
            return None

        return null_getitem_impl

    if arr == bodo.datetime_date_array_type:

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

    if arr == bodo.timestamptz_array_type:

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
        assert isinstance(
            arr, types.Array
        ), "int_getitem_overload: Numpy array expected"
        assert all(
            isinstance(a, types.Integer) for a in ind.types
        ), "int_getitem_overload: only integer indices supported"
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
def int_isn_overload(arr, ind, arr_start, total_len, is_1D):
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


# send_data, recv_data, send_counts, recv_counts, send_disp, recv_disp, typ_enum
c_alltoallv = types.ExternalFunction(
    "c_alltoallv",
    types.void(
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int32,
    ),
)


# TODO: test
# TODO: big alltoallv
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def alltoallv(
    send_data, out_data, send_counts, recv_counts, send_disp, recv_disp
):  # pragma: no cover
    typ_enum = get_type_enum(send_data)
    typ_enum_o = get_type_enum(out_data)
    assert typ_enum == typ_enum_o

    if isinstance(
        send_data, (IntegerArrayType, FloatingArrayType, DecimalArrayType)
    ) or send_data in (
        boolean_array_type,
        datetime_date_array_type,
    ):
        # TODO: Move boolean_array_type to its own section because we use 1 bit per boolean
        # TODO: Send the null bitmap
        return lambda send_data, out_data, send_counts, recv_counts, send_disp, recv_disp: c_alltoallv(
            send_data._data.ctypes,
            out_data._data.ctypes,
            send_counts.ctypes,
            recv_counts.ctypes,
            send_disp.ctypes,
            recv_disp.ctypes,
            typ_enum,
        )  # pragma: no cover

    if isinstance(send_data, bodo.CategoricalArrayType):
        return lambda send_data, out_data, send_counts, recv_counts, send_disp, recv_disp: c_alltoallv(
            send_data.codes.ctypes,
            out_data.codes.ctypes,
            send_counts.ctypes,
            recv_counts.ctypes,
            send_disp.ctypes,
            recv_disp.ctypes,
            typ_enum,
        )  # pragma: no cover

    return lambda send_data, out_data, send_counts, recv_counts, send_disp, recv_disp: c_alltoallv(
        send_data.ctypes,
        out_data.ctypes,
        send_counts.ctypes,
        recv_counts.ctypes,
        send_disp.ctypes,
        recv_disp.ctypes,
        typ_enum,
    )  # pragma: no cover


def alltoallv_tup(
    send_data, out_data, send_counts, recv_counts, send_disp, recv_disp
):  # pragma: no cover
    return


@overload(alltoallv_tup, no_unliteral=True)
def alltoallv_tup_overload(
    send_data, out_data, send_counts, recv_counts, send_disp, recv_disp
):
    count = send_data.count
    assert out_data.count == count

    func_text = (
        "def f(send_data, out_data, send_counts, recv_counts, send_disp, recv_disp):\n"
    )
    for i in range(count):
        func_text += "  alltoallv(send_data[{}], out_data[{}], send_counts, recv_counts, send_disp, recv_disp)\n".format(
            i, i
        )
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {"alltoallv": alltoallv}, loc_vars)
    a2a_impl = loc_vars["f"]
    return a2a_impl


@numba.njit
def get_start_count(n):  # pragma: no cover
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    start = bodo.libs.distributed_api.get_start(n, n_pes, rank)
    count = bodo.libs.distributed_api.get_node_portion(n, n_pes, rank)
    return start, count


@numba.njit
def get_start(total_size, pes, rank):  # pragma: no cover
    """get start index in 1D distribution"""
    res = total_size % pes
    blk_size = (total_size - res) // pes
    return rank * blk_size + min(rank, res)


@numba.njit
def get_end(total_size, pes, rank):  # pragma: no cover
    """get end point of range for parfor division"""
    res = total_size % pes
    blk_size = (total_size - res) // pes
    return (rank + 1) * blk_size + min(rank + 1, res)


@numba.njit
def get_node_portion(total_size, pes, rank):  # pragma: no cover
    """get portion of size for alloc division"""
    res = total_size % pes
    blk_size = (total_size - res) // pes
    if rank < res:
        return blk_size + 1
    else:
        return blk_size


@numba.njit
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


@numba.njit
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


@numba.njit
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


@numba.njit
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


@numba.njit
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
@numba.njit
def set_arr_local(arr, ind, val):  # pragma: no cover
    arr[ind] = val


# dummy function to specify local allocation size, to enable bypassing distributed
# transformations
@numba.njit
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


@numba.njit
def parallel_print(*args):  # pragma: no cover
    print(*args)


@numba.njit
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


_wait = types.ExternalFunction("dist_wait", types.void(mpi_req_numba_type, types.bool_))


@numba.generated_jit(nopython=True)
def wait(req, cond=True):
    """wait on MPI request"""
    # Tuple of requests (e.g. nullable arrays)
    if isinstance(req, types.BaseTuple):
        count = len(req.types)
        tup_call = ",".join(f"_wait(req[{i}], cond)" for i in range(count))
        func_text = "def f(req, cond=True):\n"
        func_text += f"  return {tup_call}\n"
        loc_vars = {}
        exec(func_text, {"_wait": _wait}, loc_vars)
        impl = loc_vars["f"]
        return impl

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


@numba.njit(no_cpython_wrapper=True, cache=True)
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


@numba.njit
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


@numba.njit
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


########### finalize MPI & s3_reader, disconnect hdfs when exiting ############


from bodo.io import fsspec_reader, hdfs_reader

ll.add_symbol("finalize", hdist.finalize)
finalize = types.ExternalFunction("finalize", types.int32())

ll.add_symbol("finalize_fsspec", fsspec_reader.finalize_fsspec)
finalize_fsspec = types.ExternalFunction("finalize_fsspec", types.int32())

ll.add_symbol("disconnect_hdfs", hdfs_reader.disconnect_hdfs)
disconnect_hdfs = types.ExternalFunction("disconnect_hdfs", types.int32())


def _check_for_cpp_errors():
    pass


@overload(_check_for_cpp_errors)
def overload_check_for_cpp_errors():
    """wrapper to call check_and_propagate_cpp_exception()
    Avoids errors when JIT is disabled since intrinsics throw errors in non-JIT mode.
    """
    return lambda: check_and_propagate_cpp_exception()  # pragma: no cover


@numba.njit
def disconnect_hdfs_njit():  # pragma: no cover
    """
    Simple njit wrapper around disconnect_hdfs.
    This is useful for resetting the singleton
    hadoop filesystem instance. This is a NOP
    if the filesystem hasn't been initialized yet.
    """
    disconnect_hdfs()


@numba.njit
def call_finalize():  # pragma: no cover
    finalize()
    finalize_fsspec()
    _check_for_cpp_errors()
    disconnect_hdfs()


def flush_stdout():
    # using a function since pytest throws an error sometimes
    # if flush function is passed directly to atexit
    if not sys.stdout.closed:
        sys.stdout.flush()


atexit.register(call_finalize)
# Flush output before finalize
atexit.register(flush_stdout)


def bcast_comm(data, comm_ranks, nranks, root=MPI_ROOT):  # pragma: no cover
    """bcast() sends data from rank 0 to comm_ranks."""
    rank = bodo.libs.distributed_api.get_rank()
    # make sure all ranks receive proper data type as input
    dtype = bodo.typeof(data)
    dtype = _bcast_dtype(dtype, root)
    if rank != MPI_ROOT:
        data = get_value_for_type(dtype)
    return bcast_comm_impl(data, comm_ranks, nranks, root)


@overload(bcast_comm)
def bcast_comm_overload(data, comm_ranks, nranks, root=MPI_ROOT):
    """support bcast_comm inside jit functions"""
    return lambda data, comm_ranks, nranks, root=MPI_ROOT: bcast_comm_impl(
        data, comm_ranks, nranks, root
    )  # pragma: no cover


@numba.generated_jit(nopython=True)
def bcast_comm_impl(data, comm_ranks, nranks, root=MPI_ROOT):  # pragma: no cover
    """nopython implementation of bcast_comm()"""
    bodo.hiframes.pd_dataframe_ext.check_runtime_cols_unsupported(
        data, "bodo.bcast_comm()"
    )
    if isinstance(data, (types.Integer, types.Float)):
        typ_val = numba_to_c_type(data)
        func_text = (
            f"def bcast_scalar_impl(data, comm_ranks, nranks, root={MPI_ROOT}):\n"
            "  send = np.empty(1, dtype)\n"
            "  send[0] = data\n"
            "  c_bcast(send.ctypes, np.int32(1), np.int32({}), comm_ranks,ctypes, np.int32({}), np.int32(root))\n"
            "  return send[0]\n"
        ).format(typ_val, nranks)

        dtype = numba.np.numpy_support.as_dtype(data)
        loc_vars = {}
        exec(
            func_text,
            {"bodo": bodo, "np": np, "c_bcast": c_bcast, "dtype": dtype},
            loc_vars,
        )
        bcast_scalar_impl = loc_vars["bcast_scalar_impl"]
        return bcast_scalar_impl

    if isinstance(data, types.Array):
        return lambda data, comm_ranks, nranks, root=MPI_ROOT: _bcast_np(
            data, comm_ranks, nranks, root
        )  # pragma: no cover

    if (
        isinstance(
            data,
            (IntegerArrayType, FloatingArrayType, DecimalArrayType, DatetimeArrayType),
        )
        or data == datetime_date_array_type
    ):
        # these array need a data array and a null bitmap array to be initialized by
        # their init functions
        if isinstance(data, IntegerArrayType):
            init_func = bodo.libs.int_arr_ext.init_integer_array
        if isinstance(data, FloatingArrayType):
            init_func = bodo.libs.float_arr_ext.init_float_array
        if isinstance(data, DecimalArrayType):
            precision = data.precision
            scale = data.scale
            init_func = numba.njit(no_cpython_wrapper=True)(
                lambda d, b: bodo.libs.decimal_arr_ext.init_decimal_array(
                    d, b, precision, scale
                )  # pragma: no cover
            )
        if data == datetime_date_array_type:
            init_func = bodo.hiframes.datetime_date_ext.init_datetime_date_array

        def impl_range_nullable(
            data, comm_ranks, nranks, root=MPI_ROOT
        ):  # pragma: no cover
            data_in = data._data
            null_bitmap = data._null_bitmap
            data_recv = _bcast_np(data_in, comm_ranks, nranks, root)
            bitmap_recv = _bcast_np(null_bitmap, comm_ranks, nranks, root)
            return init_func(data_recv, bitmap_recv)

        return impl_range_nullable

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        data_args = ", ".join("g_data_{}".format(i) for i in range(n_cols))
        col_name_meta_value_bcast_comm = ColNamesMetaType(data.columns)

        func_text = f"def impl_df(data, comm_ranks, nranks, root={MPI_ROOT}):\n"
        for i in range(n_cols):
            func_text += "  data_{} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {})\n".format(
                i, i
            )
            func_text += "  g_data_{} = bodo.libs.distributed_api.bcast_comm_impl(data_{}, comm_ranks, nranks, root)\n".format(
                i, i
            )
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
        )
        func_text += "  g_index = bodo.libs.distributed_api.bcast_comm_impl(index, comm_ranks, nranks, root)\n"
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), g_index, __col_name_meta_value_bcast_comm)\n".format(
            data_args
        )

        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "__col_name_meta_value_bcast_comm": col_name_meta_value_bcast_comm,
            },
            loc_vars,
        )
        impl_df = loc_vars["impl_df"]
        return impl_df

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):

        def impl_range_index(
            data, comm_ranks, nranks, root=MPI_ROOT
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_pes = bodo.libs.distributed_api.get_size()

            start = data._start
            stop = data._stop
            step = data._step
            name = data._name

            name = bcast_scalar(name, root)

            start = bcast_scalar(start, root)
            stop = bcast_scalar(stop, root)
            step = bcast_scalar(step, root)
            n_items = bodo.libs.array_kernels.calc_nitems(start, stop, step)
            chunk_start = bodo.libs.distributed_api.get_start(n_items, n_pes, rank)
            chunk_count = bodo.libs.distributed_api.get_node_portion(
                n_items, n_pes, rank
            )
            new_start = start + step * chunk_start
            new_stop = start + step * (chunk_start + chunk_count)
            new_stop = min(new_stop, stop)
            return bodo.hiframes.pd_index_ext.init_range_index(
                new_start, new_stop, step, name
            )

        return impl_range_index

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):

        def impl_pd_index(data, comm_ranks, nranks, root=MPI_ROOT):  # pragma: no cover
            data_in = data._data
            name = data._name
            arr = bodo.libs.distributed_api.bcast_comm_impl(
                data_in, comm_ranks, nranks, root
            )
            return bodo.utils.conversion.index_from_array(arr, name)

        return impl_pd_index

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl_series(data, comm_ranks, nranks, root=MPI_ROOT):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # bcast data
            out_name = bodo.libs.distributed_api.bcast_comm_impl(
                name, comm_ranks, nranks, root
            )
            out_arr = bodo.libs.distributed_api.bcast_comm_impl(
                arr, comm_ranks, nranks, root
            )
            out_index = bodo.libs.distributed_api.bcast_comm_impl(
                index, comm_ranks, nranks, root
            )
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, out_name)

        return impl_series
    # Tuple of data containers
    if isinstance(data, types.BaseTuple):
        func_text = f"def impl_tuple(data, comm_ranks, nranks, root={MPI_ROOT}):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(
                "bcast_comm_impl(data[{}], comm_ranks, nranks, root)".format(i)
                for i in range(len(data))
            ),
            "," if len(data) > 0 else "",
        )
        loc_vars = {}
        exec(func_text, {"bcast_comm_impl": bcast_comm_impl}, loc_vars)
        impl_tuple = loc_vars["impl_tuple"]
        return impl_tuple
    if data is types.none:  # pragma: no cover
        return lambda data, comm_ranks, nranks, root=MPI_ROOT: None


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _bcast_np(data, comm_ranks, nranks, root=MPI_ROOT):
    """bcast() implementation for numpy arrays, refactored here with
    no_cpython_wrapper=True to enable int128 data array of decimal arrays. Otherwise,
    Numba creates a wrapper and complains about unboxing int128.
    """
    typ_val = numba_to_c_type(data.dtype)
    ndim = data.ndim
    dtype = data.dtype
    # using np.dtype since empty() doesn't work with typeref[datetime/timedelta]
    if dtype == types.NPDatetime("ns"):
        dtype = np.dtype("datetime64[ns]")
    elif dtype == types.NPTimedelta("ns"):
        dtype = np.dtype("timedelta64[ns]")
    zero_shape = (0,) * ndim

    def bcast_arr_impl(data, comm_ranks, nranks, root=MPI_ROOT):  # pragma: no cover
        rank = bodo.libs.distributed_api.get_rank()
        data_in = np.ascontiguousarray(data)
        data_ptr = data.ctypes

        # broadcast shape to all processors
        shape = zero_shape
        if rank == root:
            shape = data_in.shape
        # shape = bcast_comm_impl(shape, comm_ranks, nranks)
        shape = bcast_tuple(shape, root)
        n_elem_per_row = get_tuple_prod(shape[1:])
        send_counts = shape[0] * n_elem_per_row
        recv_data = np.empty(send_counts, dtype)
        if rank == MPI_ROOT:
            c_bcast(
                data_ptr,
                np.int32(send_counts),
                np.int32(typ_val),
                comm_ranks.ctypes,
                np.int32(nranks),
                np.int32(root),
            )
            return data
        else:
            c_bcast(
                recv_data.ctypes,
                np.int32(send_counts),
                np.int32(typ_val),
                comm_ranks.ctypes,
                np.int32(nranks),
                np.int32(root),
            )
            # handle multi-dim case
            return recv_data.reshape((-1,) + shape[1:])

    return bcast_arr_impl


node_ranks = None


def get_host_ranks():  # pragma: no cover
    """Get dict holding hostname and its associated ranks"""
    global node_ranks
    if node_ranks is None:
        comm = MPI.COMM_WORLD
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


def get_nodes_first_ranks():  # pragma: no cover
    """Get first rank in each node"""
    host_ranks = get_host_ranks()
    return np.array([ranks[0] for ranks in host_ranks.values()], dtype="int32")


def get_num_nodes():  # pragma: no cover
    """Get number of nodes"""
    return len(get_host_ranks())


# Use default number of iterations for sync if not specified by user
sync_iters = (
    bodo.default_stream_loop_sync_iters
    if bodo.stream_loop_sync_iters == -1
    else bodo.stream_loop_sync_iters
)


@numba.njit
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
delete_is_last_state = types.ExternalFunction(
    "delete_is_last_state", types.none(is_last_state_type)
)
# NOTE: using int32 types to avoid i1 vs i8 boolean errors in lowering
sync_is_last_non_blocking = types.ExternalFunction(
    "sync_is_last_non_blocking", types.int32(is_last_state_type, types.int32)
)
