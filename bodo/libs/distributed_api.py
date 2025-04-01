import atexit
import datetime
import sys
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
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.datetime_timedelta_ext import datetime_timedelta_array_type
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
    py_data_to_cpp_table,
    py_table_to_cpp_table,
    table_type,
)
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
from bodo.mpi4py import MPI
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    ColNamesMetaType,
    ExternalFunctionErrorChecked,
    MetaType,
    decode_if_dict_array,
    is_bodosql_context_type,
    is_overload_false,
    is_overload_none,
    is_str_arr_type,
)
from bodo.utils.utils import (
    CTypeEnum,
    bodo_exec,
    cached_call_internal,
    check_and_propagate_cpp_exception,
    empty_like_type,
    is_array_typ,
    is_distributable_typ,
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
ll.add_symbol("c_get_remote_size", hdist.dist_get_remote_size)
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
ll.add_symbol("gather_table_py_entry", hdist.gather_table_py_entry)
ll.add_symbol("gather_array_py_entry", hdist.gather_array_py_entry)
ll.add_symbol("get_cpu_id", hdist.get_cpu_id)


# get size dynamically from C code (mpich 3.2 is 4 bytes but openmpi 1.6 is 8)
mpi_req_numba_type = getattr(types, "int" + str(8 * hdist.mpi_req_num_bytes))

DEFAULT_ROOT = 0
ANY_SOURCE = np.int32(hdist.ANY_SOURCE)

# Wrapper for getting process rank from C (MPI rank currently)
get_rank = hdist.get_rank_py_wrapper


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
_get_cpu_id = types.ExternalFunction("get_cpu_id", types.int32())
get_remote_size = types.ExternalFunction("c_get_remote_size", types.int32(types.int64))


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


@numba.njit(cache=True)
def get_size():  # pragma: no cover
    """wrapper for getting number of processes (MPI COMM size currently)"""
    return _get_size()


@numba.njit(cache=True)
def barrier():  # pragma: no cover
    """wrapper for barrier (MPI barrier currently)"""
    _barrier()


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


_get_time = types.ExternalFunction("get_time", types.float64())
dist_time = types.ExternalFunction("dist_get_time", types.float64())


@infer_global(time.time)
class TimeInfer(ConcreteTemplate):
    cases = [signature(types.float64)]


@lower_builtin(time.time)
def lower_time_time(context, builder, sig, args):
    return cached_call_internal(context, builder, lambda: _get_time(), sig, args)


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

        loc_vars = {}
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


@numba.njit(cache=True)
def alltoall(send_arr, recv_arr, count):  # pragma: no cover
    # TODO: handle int64 counts
    assert count < INT_MAX
    type_enum = get_type_enum(send_arr)
    _alltoall(send_arr.ctypes, recv_arr.ctypes, np.int32(count), type_enum)


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


c_gather_scalar = types.ExternalFunction(
    "c_gather_scalar",
    types.void(
        types.voidptr, types.voidptr, types.int32, types.bool_, types.int32, types.int64
    ),
)


# sendbuf, sendcount, recvbuf, recv_counts, displs, dtype
c_gatherv = types.ExternalFunction(
    "c_gatherv",
    types.void(
        types.voidptr,
        types.int64,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int32,
        types.bool_,
        types.int32,
        types.int64,
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
        types.int64,
        types.int32,
        types.int32,
        types.int64,
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
    "dist_reduce",
    types.void(types.voidptr, types.voidptr, types.int32, types.int32, types.int64),
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


@numba.njit(cache=True)
def dist_reduce(value, reduce_op, comm=0):
    return dist_reduce_impl(value, reduce_op, comm)


@numba.generated_jit(nopython=True)
def dist_reduce_impl(value, reduce_op, comm):
    if isinstance(value, types.Array):
        typ_enum = np.int32(numba_to_c_type(value.dtype))

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
            bodo.datetime64ns,
            bodo.timedelta64ns,
            bodo.datetime_date_type,
            bodo.TimeType,
        ]

        if target_typ not in supported_typs and not isinstance(
            target_typ, (bodo.Decimal128Type, bodo.PandasTimestampType)
        ):  # pragma: no cover
            raise BodoError(f"argmin/argmax not supported for type {target_typ}")

    typ_enum = np.int32(numba_to_c_type(target_typ))

    if isinstance(target_typ, bodo.Decimal128Type):
        # For index-value types, the data pointed to has different amounts of padding depending on machine type.
        # as a workaround, we can pass the index separately.
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
            return bodo.TimestampTZ(pd.Timestamp(out_ts), out_offset)

        return impl

    def impl(value, reduce_op, comm):  # pragma: no cover
        in_ptr = value_to_ptr(value)
        out_ptr = value_to_ptr(value)
        _dist_reduce(in_ptr, out_ptr, reduce_op, typ_enum, comm)
        return load_val_ptr(out_ptr, value)

    return impl


_dist_exscan = types.ExternalFunction(
    "dist_exscan", types.void(types.voidptr, types.voidptr, types.int32, types.int32)
)


@numba.njit(cache=True)
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


_gather_table_py_entry = ExternalFunctionErrorChecked(
    "gather_table_py_entry",
    table_type(table_type, types.bool_, types.int32, types.int64),
)


_gather_array_py_entry = ExternalFunctionErrorChecked(
    "gather_array_py_entry",
    array_info_type(array_info_type, types.bool_, types.int32, types.int64),
)


def gatherv(data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=None):
    """Gathers data from all ranks to root."""
    from bodo.mpi4py import MPI

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


@overload(gatherv)
def gatherv_overload(
    data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
):
    """support gatherv inside jit functions"""

    return (
        lambda data,
        allgather=False,
        warn_if_rep=True,
        root=DEFAULT_ROOT,
        comm=0: gatherv_impl_jit(data, allgather, warn_if_rep, root, comm)
    )  # pragma: no cover


@numba.njit(cache=True)
def gatherv_impl_wrapper(
    data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
):
    return gatherv_impl_jit(data, allgather, warn_if_rep, root, comm)


@numba.generated_jit(nopython=True)
def gatherv_impl_jit(
    data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
):
    """gathers distributed data into rank 0 or all ranks if 'allgather' is set.
    'warn_if_rep' flag controls if a warning is raised if the input is replicated and
    gatherv has no effect (applicable only inside jit functions).
    """
    from bodo.libs.csr_matrix_ext import CSRMatrixType

    bodo.hiframes.pd_dataframe_ext.check_runtime_cols_unsupported(
        data, "bodo.gatherv()"
    )

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # Send name from workers to receiver in case of intercomm since not
            # available on receiver
            if comm != 0:
                bcast_root = MPI.PROC_NULL
                is_receiver = root == MPI.ROOT
                if is_receiver:
                    bcast_root = 0
                elif bodo.get_rank() == 0:
                    bcast_root = MPI.ROOT
                name = bcast_scalar(name, bcast_root, comm)
            # gather data
            out_arr = bodo.libs.distributed_api.gatherv(
                arr, allgather, warn_if_rep, root, comm
            )
            out_index = bodo.gatherv(index, allgather, warn_if_rep, root, comm)
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

        return impl

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):
        INT64_MAX = np.iinfo(np.int64).max
        INT64_MIN = np.iinfo(np.int64).min

        def impl_range_index(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            is_receiver = bodo.get_rank() == root
            if comm != 0:
                is_receiver = root == MPI.ROOT

            # NOTE: assuming processes have chunks of a global RangeIndex with equal
            # steps. using min/max reductions to get start/stop of global range
            start = data._start
            stop = data._stop
            step = data._step
            name = data._name
            # Send name and step from workers to receiver in case of intercomm since not
            # available on receiver
            if comm != 0:
                bcast_root = MPI.PROC_NULL
                if is_receiver:
                    bcast_root = 0
                elif bodo.get_rank() == 0:
                    bcast_root = MPI.ROOT
                name = bcast_scalar(name, bcast_root, comm)
                step = bcast_scalar(step, bcast_root, comm)

            # ignore empty ranges coming from slicing, see test_getitem_slice
            if len(data) == 0:
                start = INT64_MAX
                stop = INT64_MIN
            min_op = np.int32(Reduce_Type.Min.value)
            max_op = np.int32(Reduce_Type.Max.value)
            start = bodo.libs.distributed_api.dist_reduce(
                start, min_op if step > 0 else max_op, comm
            )
            stop = bodo.libs.distributed_api.dist_reduce(
                stop, max_op if step > 0 else min_op, comm
            )
            total_len = bodo.libs.distributed_api.dist_reduce(
                len(data), np.int32(Reduce_Type.Sum.value), comm
            )
            # output is empty if all range chunks are empty
            if start == INT64_MAX and stop == INT64_MIN:
                start = 0
                stop = 0

            # make sure global length is consistent in case the user passes in incorrect
            # RangeIndex chunks (e.g. trivial index in each chunk), see test_rebalance
            l = max(0, -(-(stop - start) // step))
            if l < total_len:
                stop = start + step * total_len

            # gatherv() of dataframe returns 0-length arrays so index should
            # be 0-length to match
            if not is_receiver and not allgather:
                start = 0
                stop = 0

            return bodo.hiframes.pd_index_ext.init_range_index(start, stop, step, name)

        return impl_range_index

    # Index types
    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):
        from bodo.hiframes.pd_index_ext import PeriodIndexType

        if isinstance(data, PeriodIndexType):
            freq = data.freq

            def impl_pd_index(
                data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
            ):  # pragma: no cover
                arr = bodo.libs.distributed_api.gatherv(
                    data._data, allgather, warn_if_rep, root, comm
                )
                # Send name from workers to receiver in case of intercomm since not
                # available on receiver
                name = data._name
                if comm != 0:
                    bcast_root = MPI.PROC_NULL
                    is_receiver = root == MPI.ROOT
                    if is_receiver:
                        bcast_root = 0
                    elif bodo.get_rank() == 0:
                        bcast_root = MPI.ROOT
                    name = bcast_scalar(name, bcast_root, comm)
                return bodo.hiframes.pd_index_ext.init_period_index(arr, name, freq)

        else:

            def impl_pd_index(
                data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
            ):  # pragma: no cover
                arr = bodo.libs.distributed_api.gatherv(
                    data._data, allgather, warn_if_rep, root, comm
                )
                # Send name from workers to receiver in case of intercomm since not
                # available on receiver
                name = data._name
                if comm != 0:
                    bcast_root = MPI.PROC_NULL
                    is_receiver = root == MPI.ROOT
                    if is_receiver:
                        bcast_root = 0
                    elif bodo.get_rank() == 0:
                        bcast_root = MPI.ROOT
                    name = bcast_scalar(name, bcast_root, comm)
                return bodo.utils.conversion.index_from_array(arr, name)

        return impl_pd_index

    # MultiIndex index
    if isinstance(data, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        # just gather the data arrays
        # TODO: handle `levels` and `codes` when available
        def impl_multi_index(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            all_data = bodo.gatherv(data._data, allgather, warn_if_rep, root, comm)
            # Send name from workers to receiver in case of intercomm since not
            # available on receiver
            name = data._name
            names = data._names
            if comm != 0:
                bcast_root = MPI.PROC_NULL
                is_receiver = root == MPI.ROOT
                if is_receiver:
                    bcast_root = 0
                elif bodo.get_rank() == 0:
                    bcast_root = MPI.ROOT
                name = bcast_scalar(name, bcast_root, comm)
                names = bcast_tuple(names, bcast_root, comm)
            return bodo.hiframes.pd_multi_index_ext.init_multi_index(
                all_data, names, name
            )

        return impl_multi_index

    if isinstance(data, bodo.hiframes.table.TableType):
        table_type = data
        n_table_cols = len(table_type.arr_types)
        in_col_inds = MetaType(tuple(range(n_table_cols)))
        out_cols_arr = np.array(range(n_table_cols), dtype=np.int64)

        def impl(data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0):
            cpp_table = py_data_to_cpp_table(data, (), in_col_inds, n_table_cols)
            out_cpp_table = _gather_table_py_entry(cpp_table, allgather, root, comm)
            ret = cpp_table_to_py_table(out_cpp_table, out_cols_arr, table_type, 0)
            delete_table(out_cpp_table)
            return ret

        return impl

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        # empty dataframe case
        if n_cols == 0:
            __col_name_meta_value_gatherv_no_cols = ColNamesMetaType(())

            def impl(
                data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
            ):  # pragma: no cover
                index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)
                g_index = bodo.gatherv(index, allgather, warn_if_rep, root, comm)
                return bodo.hiframes.pd_dataframe_ext.init_dataframe(
                    (), g_index, __col_name_meta_value_gatherv_no_cols
                )

            return impl

        data_args = ", ".join(f"g_data_{i}" for i in range(n_cols))

        func_text = f"def impl_df(data, allgather=False, warn_if_rep=True, root={DEFAULT_ROOT}, comm=0):\n"
        if data.is_table_format:
            data_args = "T2"
            func_text += (
                "  T = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(data)\n"
                "  T2 = bodo.gatherv(T, allgather, warn_if_rep, root, comm)\n"
            )
        else:
            for i in range(n_cols):
                func_text += f"  data_{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {i})\n"
                func_text += f"  g_data_{i} = bodo.gatherv(data_{i}, allgather, warn_if_rep, root, comm)\n"
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
            "  g_index = bodo.gatherv(index, allgather, warn_if_rep, root, comm)\n"
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

    # CSR Matrix
    if isinstance(data, CSRMatrixType):

        def impl_csr_matrix(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # gather local data
            all_data = bodo.gatherv(data.data, allgather, warn_if_rep, root, comm)
            all_col_inds = bodo.gatherv(
                data.indices, allgather, warn_if_rep, root, comm
            )
            all_indptr = bodo.gatherv(data.indptr, allgather, warn_if_rep, root, comm)
            all_local_rows = gather_scalar(
                data.shape[0], allgather, root=root, comm=comm
            )
            n_rows = all_local_rows.sum()
            n_cols = bodo.libs.distributed_api.dist_reduce(
                data.shape[1], np.int32(Reduce_Type.Max.value), comm
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
        func_text = f"def impl_tuple(data, allgather=False, warn_if_rep=True, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(
                f"bodo.gatherv(data[{i}], allgather, warn_if_rep, root, comm)"
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
            lambda data,
            allgather=False,
            warn_if_rep=True,
            root=DEFAULT_ROOT,
            comm=0: None
        )  # pragma: no cover

    if isinstance(data, types.Array) and data.ndim != 1:
        typ_val = numba_to_c_type(data.dtype)

        def gatherv_impl(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            data = np.ascontiguousarray(data)
            rank = bodo.get_rank()
            is_receiver = rank == root
            is_intercomm = comm != 0
            if is_intercomm:
                is_receiver = root == MPI.ROOT
            # size to handle multi-dim arrays
            n_loc = data.size
            recv_counts = gather_scalar(
                np.int64(n_loc), allgather, root=root, comm=comm
            )
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            displs = np.empty(1, np.int64)
            if is_receiver or allgather:
                displs = bodo.ir.join.calc_disp(recv_counts)
            c_gatherv(
                data.ctypes,
                np.int64(n_loc),
                all_data.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                np.int32(typ_val),
                allgather,
                np.int32(root),
                comm,
            )

            shape = data.shape
            # Send shape from workers to receiver in case of intercomm since not
            # available on receiver
            if is_intercomm:
                bcast_root = MPI.PROC_NULL
                if is_receiver:
                    bcast_root = 0
                elif rank == 0:
                    bcast_root = MPI.ROOT
                shape = bcast_tuple(shape, bcast_root, comm)

            # handle multi-dim case
            return all_data.reshape((-1,) + shape[1:])

        return gatherv_impl

    if isinstance(data, CategoricalArrayType):

        def impl_cat(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            codes = bodo.gatherv(data.codes, allgather, warn_if_rep, root, comm)
            return bodo.hiframes.pd_categorical_ext.init_categorical_array(
                codes, data.dtype
            )

        return impl_cat

    if isinstance(data, bodo.MatrixType):

        def impl_matrix(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            new_data = bodo.gatherv(data.data, allgather, warn_if_rep, root, comm)
            return bodo.libs.matrix_ext.init_np_matrix(new_data)

        return impl_matrix

    # DatetimeTimeDeltaArrayType (not supported by C++ below yet)
    if data == datetime_timedelta_array_type:
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_timedelta(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            days = bodo.gatherv(data._days_data, allgather, warn_if_rep, root, comm)
            seconds = bodo.gatherv(
                data._seconds_data, allgather, warn_if_rep, root, comm
            )
            microseconds = bodo.gatherv(
                data._microseconds_data, allgather, warn_if_rep, root, comm
            )

            rank = bodo.get_rank()
            is_receiver = rank == root
            if comm != 0:
                is_receiver = root == MPI.ROOT
            n_loc = len(data)
            n_bytes = (n_loc + 7) >> 3
            recv_counts = gather_scalar(
                np.int32(n_loc), allgather, warn_if_rep, root, comm
            )

            recv_counts_nulls = np.empty(1, np.int64)
            displs_nulls = np.empty(1, np.int64)
            tmp_null_bytes = np.empty(1, np.uint8)
            if is_receiver or allgather:
                recv_counts_nulls = np.empty(len(recv_counts), np.int64)
                for i in range(len(recv_counts)):
                    recv_counts_nulls[i] = (recv_counts[i] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)

            c_gatherv(
                data._null_bitmap.ctypes,
                np.int64(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_nulls.ctypes,
                displs_nulls.ctypes,
                char_typ_enum,
                allgather,
                np.int32(root),
                comm,
            )
            out_null_bitmap = np.empty(recv_counts.sum() >> 3, np.uint8)
            copy_gathered_null_bytes(
                out_null_bitmap.ctypes,
                tmp_null_bytes,
                recv_counts_nulls,
                recv_counts,
            )
            return bodo.hiframes.datetime_timedelta_ext.init_datetime_timedelta_array(
                days, seconds, microseconds, out_null_bitmap
            )

        return impl_timedelta

    if is_array_typ(data, False):
        dtype = data

        def impl(data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0):
            input_info = array_to_info(data)
            out_info = _gather_array_py_entry(input_info, allgather, root, comm)
            ret = info_to_array(out_info, dtype)
            delete_info(out_info)
            return ret

        return impl

    # List of distributable data
    if isinstance(data, types.List) and is_distributable_typ(data.dtype):

        def impl_list(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank = bodo.get_rank()
            is_receiver = rank == root
            is_intercomm = comm != 0
            if is_intercomm:
                is_receiver = root == MPI.ROOT

            length = len(data)
            # Send length from workers to receiver in case of intercomm since not
            # available on receiver
            if is_intercomm:
                bcast_root = MPI.PROC_NULL
                if is_receiver:
                    bcast_root = 0
                elif rank == 0:
                    bcast_root = MPI.ROOT
                length = bcast_scalar(length, bcast_root, comm)

            out = []
            for i in range(length):
                in_val = data[i] if not is_receiver else data[0]
                out.append(bodo.gatherv(in_val, allgather, warn_if_rep, root, comm))

            return out

        return impl_list

    # Dict of distributable data
    if isinstance(data, types.DictType) and is_distributable_typ(data.value_type):

        def impl_dict(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank = bodo.get_rank()
            is_receiver = rank == root
            is_intercomm = comm != 0
            if is_intercomm:
                is_receiver = root == MPI.ROOT

            length = len(data)
            # Send length from workers to receiver in case of intercomm since not
            # available on receiver
            if is_intercomm:
                bcast_root = MPI.PROC_NULL
                if is_receiver:
                    bcast_root = 0
                elif rank == 0:
                    bcast_root = MPI.ROOT
                length = bcast_scalar(length, bcast_root, comm)

            in_keys = list(data.keys())
            in_values = list(data.values())
            out = {}
            for i in range(length):
                key = in_keys[i] if not is_receiver else in_keys[0]
                if is_intercomm:
                    bcast_root = MPI.PROC_NULL
                    if is_receiver:
                        bcast_root = 0
                    elif rank == 0:
                        bcast_root = MPI.ROOT
                    key = bcast_scalar(key, bcast_root, comm)
                value = in_values[i] if not is_receiver else in_values[0]
                out[key] = bodo.gatherv(value, allgather, warn_if_rep, root, comm)

            return out

        return impl_dict

    if is_bodosql_context_type(data):
        import bodosql

        func_text = f"def impl_bodosql_context(data, allgather=False, warn_if_rep=True, root={DEFAULT_ROOT}, comm=0):\n"
        comma_sep_names = ", ".join([f"'{name}'" for name in data.names])
        comma_sep_dfs = ", ".join(
            [
                f"bodo.gatherv(data.dataframes[{i}], allgather, warn_if_rep, root, comm)"
                for i in range(len(data.dataframes))
            ]
        )
        func_text += f"  return bodosql.context_ext.init_sql_context(({comma_sep_names}, ), ({comma_sep_dfs}, ), data.catalog, None)\n"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)
        impl_bodosql_context = loc_vars["impl_bodosql_context"]
        return impl_bodosql_context

    if type(data).__name__ == "TablePathType":
        try:
            from bodosql import TablePathType
        except ImportError:  # pragma: no cover
            raise ImportError("Install bodosql to use gatherv() with TablePathType")
        assert isinstance(data, TablePathType)
        # Table Path info is all compile time so we return the same data.
        func_text = f"def impl_table_path(data, allgather=False, warn_if_rep=True, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  return data\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        impl_table_path = loc_vars["impl_table_path"]
        return impl_table_path

    raise BodoError(f"gatherv() not available for {data}")  # pragma: no cover


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


@numba.njit(cache=True)
def get_scatter_null_bytes_buff(
    null_bitmap_ptr, sendcounts, sendcounts_nulls, is_sender
):  # pragma: no cover
    """copy null bytes into a padded buffer for scatter.
    Padding is needed since processors receive whole bytes and data inside border bytes
    has to be split.
    Only the root rank has the input data and needs to create a valid send buffer.
    """
    # non-root ranks don't have scatter input
    if not is_sender:
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


def _bcast_dtype(data, root=DEFAULT_ROOT, comm=None):
    """broadcast data type from rank 0 using mpi4py"""
    try:
        from bodo.mpi4py import MPI
    except ImportError:  # pragma: no cover
        raise BodoError("mpi4py is required for scatterv")

    if comm is None:
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
        send_counts = np.empty(n_pes, np.int64)
        for i in range(n_pes):
            send_counts[i] = get_node_portion(n, n_pes, i)
        return send_counts

    return impl


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _scatterv_np(data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0):
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
        data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
    ):  # pragma: no cover
        rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)

        data_in = np.ascontiguousarray(data)
        data_ptr = data_in.ctypes

        # broadcast shape to all processors
        shape = zero_shape
        if is_sender:
            shape = data_in.shape
        shape = bcast_tuple(shape, root, comm)
        n_elem_per_row = get_tuple_prod(shape[1:])

        send_counts = _get_scatterv_send_counts(send_counts, n_pes, shape[0])
        send_counts *= n_elem_per_row

        # allocate output with total number of receive elements on this PE
        n_loc = 0 if (is_intercomm and is_sender) else send_counts[rank]
        recv_data = np.empty(n_loc, dtype)

        # displacements
        displs = bodo.ir.join.calc_disp(send_counts)

        c_scatterv(
            data_ptr,
            send_counts.ctypes,
            displs.ctypes,
            recv_data.ctypes,
            np.int64(n_loc),
            np.int32(typ_val),
            root,
            comm,
        )

        if is_intercomm and is_sender:
            shape = zero_shape

        # handle multi-dim case
        return recv_data.reshape((-1,) + shape[1:])

    return scatterv_arr_impl


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

    if dtype == bodo.dict_str_arr_type:
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
    if dtype == datetime_timedelta_array_type:
        return np.array([datetime.timedelta(33)])

    # Index types
    if bodo.hiframes.pd_index_ext.is_pd_index_type(dtype):
        name = get_value_for_type(dtype.name_typ)
        if isinstance(dtype, bodo.hiframes.pd_index_ext.RangeIndexType):
            return pd.RangeIndex(1, name=name)
        arr_type = bodo.utils.typing.get_index_data_arr_types(dtype)[0]
        arr = get_value_for_type(arr_type)
        if isinstance(dtype, bodo.PeriodIndexType):
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
    if isinstance(dtype, bodo.TableType):
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
        return pd.array([pd.Timestamp("2024/1/1", tz=dtype.tz)])

    # TimestampTZ array
    if dtype == bodo.timestamptz_array_type:
        return np.array([bodo.TimestampTZ(pd.Timestamp(0), 0)])

    # TimeArray
    if isinstance(dtype, TimeArrayType):
        precision = dtype.precision
        if use_arrow_time:
            assert precision == 9, (
                "get_value_for_type: only nanosecond precision is supported for nested data"
            )
            return pd.array(
                [bodo.Time(3, precision=precision)], pd.ArrowDtype(pa.time64("ns"))
            )
        return np.array([bodo.Time(3, precision=precision)], object)

    # NullArray
    if dtype == bodo.null_array_type:
        return pd.arrays.ArrowExtensionArray(pa.nulls(1))

    # StructArray
    if isinstance(dtype, bodo.StructArrayType):
        # Handle empty struct corner case which can have typing issues
        if dtype == bodo.StructArrayType((), ()):
            return pd.array([{}], pd.ArrowDtype(pa.struct([])))

        pa_arr = pa.StructArray.from_arrays(
            tuple(get_value_for_type(t, True) for t in dtype.data), dtype.names
        )
        return pd.arrays.ArrowExtensionArray(pa_arr)

    # TupleArray
    if isinstance(dtype, bodo.TupleArrayType):
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
    if isinstance(dtype, bodo.MapArrayType):
        pa_arr = pa.MapArray.from_arrays(
            [0, 1],
            get_value_for_type(dtype.key_arr_type, True),
            get_value_for_type(dtype.value_arr_type, True),
        )
        return pd.arrays.ArrowExtensionArray(pa_arr)

    # Numpy Matrix
    if isinstance(dtype, bodo.MatrixType):
        return np.asmatrix(
            get_value_for_type(types.Array(dtype.dtype, 2, dtype.layout))
        )

    if isinstance(dtype, types.List):
        return [get_value_for_type(dtype.dtype)]

    if isinstance(dtype, types.DictType):
        return {
            get_value_for_type(dtype.key_type): get_value_for_type(dtype.value_type)
        }

    if dtype == bodo.string_type:
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


@numba.njit(cache=True, no_cpython_wrapper=True)
def get_scatter_comm_info(root, comm):
    """Return communication attributes for scatterv based on root and intercomm"""
    is_intercomm = comm != 0
    rank = bodo.libs.distributed_api.get_rank()
    is_sender = rank == root
    if is_intercomm:
        is_sender = root == MPI.ROOT
    n_pes = (
        bodo.libs.distributed_api.get_size()
        if not (is_intercomm and is_sender)
        else bodo.libs.distributed_api.get_remote_size(comm)
    )
    return rank, is_intercomm, is_sender, n_pes


def scatterv(data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=None):
    """scatterv() distributes data from rank 0 to all ranks.
    Rank 0 passes the data but the other ranks should just pass None.
    """
    from bodo.mpi4py import MPI

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


@overload(scatterv)
def scatterv_overload(
    data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
):
    """support scatterv inside jit functions"""
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


@numba.njit(cache=True)
def scatterv_impl(data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0):
    return scatterv_impl_jit(data, send_counts, warn_if_dist, root, comm)


@numba.generated_jit(nopython=True)
def scatterv_impl_jit(
    data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
):
    """nopython implementation of scatterv()"""
    if isinstance(data, types.Array):
        return (
            lambda data,
            send_counts=None,
            warn_if_dist=True,
            root=DEFAULT_ROOT,
            comm=0: _scatterv_np(data, send_counts, warn_if_dist, root, comm)
        )  # pragma: no cover

    if data in (string_array_type, binary_array_type):
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))
        empty_int32_arr = np.array([], np.int32)

        if data == binary_array_type:
            alloc_fn = bodo.libs.binary_arr_ext.pre_alloc_binary_array
        else:
            alloc_fn = bodo.libs.str_arr_ext.pre_alloc_string_array

        def impl(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)

            n_all = bodo.libs.distributed_api.bcast_scalar(len(data), root, comm)

            # convert offsets to lengths of strings
            send_arr_lens = np.empty(
                len(data), np.uint32
            )  # XXX offset type is offset_type, lengths for comm are uint32
            for i in range(len(data)):
                send_arr_lens[i] = bodo.libs.str_arr_ext.get_str_arr_item_length(
                    data, i
                )

            # ------- calculate buffer counts -------

            send_counts = bodo.libs.distributed_api._get_scatterv_send_counts(
                send_counts, n_pes, n_all
            )

            # displacements
            displs = bodo.ir.join.calc_disp(send_counts)

            # compute send counts for characters
            send_counts_char = np.empty(n_pes, np.int64)
            if is_sender:
                curr_str = 0
                for i in range(n_pes):
                    c = 0
                    for _ in range(send_counts[i]):
                        c += send_arr_lens[curr_str]
                        curr_str += 1
                    send_counts_char[i] = c

            send_counts_char = bodo.libs.distributed_api.bcast(
                send_counts_char, empty_int32_arr, root, comm
            )

            # displacements for characters
            displs_char = bodo.ir.join.calc_disp(send_counts_char)

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int64)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            # allocate output with total number of receive elements on this PE
            n_loc = 0 if (is_intercomm and is_sender) else send_counts[rank]
            n_loc_char = 0 if (is_intercomm and is_sender) else send_counts_char[rank]
            recv_arr = alloc_fn(n_loc, n_loc_char)

            # ----- string lengths -----------

            recv_lens = np.empty(n_loc, np.uint32)
            bodo.libs.distributed_api.c_scatterv(
                send_arr_lens.ctypes,
                send_counts.ctypes,
                displs.ctypes,
                recv_lens.ctypes,
                np.int64(n_loc),
                int32_typ_enum,
                root,
                comm,
            )

            # TODO: don't hardcode offset type. Also, if offset is 32 bit we can
            # use the same buffer
            bodo.libs.str_arr_ext.convert_len_arr_to_offset(
                recv_lens.ctypes, bodo.libs.str_arr_ext.get_offset_ptr(recv_arr), n_loc
            )

            # ----- string characters -----------

            bodo.libs.distributed_api.c_scatterv(
                bodo.libs.str_arr_ext.get_data_ptr(data),
                send_counts_char.ctypes,
                displs_char.ctypes,
                bodo.libs.str_arr_ext.get_data_ptr(recv_arr),
                np.int64(n_loc_char),
                char_typ_enum,
                root,
                comm,
            )

            # ----------- null bitmap -------------

            n_recv_bytes = (n_loc + 7) >> 3

            send_null_bitmap = bodo.libs.distributed_api.get_scatter_null_bytes_buff(
                bodo.libs.str_arr_ext.get_null_bitmap_ptr(data),
                send_counts,
                send_counts_nulls,
                is_sender,
            )

            bodo.libs.distributed_api.c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                bodo.libs.str_arr_ext.get_null_bitmap_ptr(recv_arr),
                np.int64(n_recv_bytes),
                char_typ_enum,
                root,
                comm,
            )

            return recv_arr

        return impl

    if isinstance(data, ArrayItemArrayType):
        # Code adapted from the string code. Both the string and array(item) codes should be
        # refactored.
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))
        empty_int32_arr = np.array([], np.int32)

        def scatterv_array_item_impl(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            in_offsets_arr = bodo.libs.array_item_arr_ext.get_offsets(data)
            in_data_arr = bodo.libs.array_item_arr_ext.get_data(data)
            in_data_arr = in_data_arr[: in_offsets_arr[-1]]
            in_null_bitmap_arr = bodo.libs.array_item_arr_ext.get_null_bitmap(data)

            rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)
            n_all = bcast_scalar(len(data), root, comm)

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
            send_counts_item = np.empty(n_pes, np.int64)
            if is_sender:
                curr_item = 0
                for i in range(n_pes):
                    c = 0
                    for _ in range(send_counts[i]):
                        c += send_arr_lens[curr_item]
                        curr_item += 1
                    send_counts_item[i] = c

            send_counts_item = bodo.libs.distributed_api.bcast(
                send_counts_item, empty_int32_arr, root, comm
            )

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int64)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            # allocate output with total number of receive elements on this PE
            n_loc = 0 if (is_intercomm and is_sender) else send_counts[rank]
            recv_offsets_arr = np.empty(n_loc + 1, np_offset_type)

            recv_data_arr = bodo.libs.distributed_api.scatterv_impl(
                in_data_arr, send_counts_item, warn_if_dist, root, comm
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
                np.int64(n_loc),
                int32_typ_enum,
                root,
                comm,
            )

            # TODO: don't hardcode offset type. Also, if offset is 32 bit we can
            # use the same buffer
            convert_len_arr_to_offset(recv_lens.ctypes, recv_offsets_arr.ctypes, n_loc)

            # ----------- null bitmap -------------

            send_null_bitmap = get_scatter_null_bytes_buff(
                in_null_bitmap_arr.ctypes, send_counts, send_counts_nulls, is_sender
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                recv_null_bitmap_arr.ctypes,
                np.int64(n_recv_null_bytes),
                char_typ_enum,
                root,
                comm,
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
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)
            data_in = data._data
            null_bitmap = data._null_bitmap
            # Calculate the displacements for nulls and data, each of
            # which is a single bit.
            n_in = len(data)
            n_all = bcast_scalar(n_in, root, comm)

            send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)
            # Calculate number of local output elements
            n_loc = np.int64(0 if (is_intercomm and is_sender) else send_counts[rank])
            # compute send counts bytes
            send_counts_bytes = np.empty(n_pes, np.int64)
            for i in range(n_pes):
                send_counts_bytes[i] = (send_counts[i] + 7) >> 3

            displs_bytes = bodo.ir.join.calc_disp(send_counts_bytes)

            send_data_bitmap = get_scatter_null_bytes_buff(
                data_in.ctypes, send_counts, send_counts_bytes, is_sender
            )
            send_null_bitmap = get_scatter_null_bytes_buff(
                null_bitmap.ctypes, send_counts, send_counts_bytes, is_sender
            )
            # Allocate the output arrays
            n_recv_bytes = (
                0 if (is_intercomm and is_sender) else send_counts_bytes[rank]
            )
            data_recv = np.empty(n_recv_bytes, np.uint8)
            bitmap_recv = np.empty(n_recv_bytes, np.uint8)

            c_scatterv(
                send_data_bitmap.ctypes,
                send_counts_bytes.ctypes,
                displs_bytes.ctypes,
                data_recv.ctypes,
                np.int64(n_recv_bytes),
                char_typ_enum,
                root,
                comm,
            )
            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_bytes.ctypes,
                displs_bytes.ctypes,
                bitmap_recv.ctypes,
                np.int64(n_recv_bytes),
                char_typ_enum,
                root,
                comm,
            )
            return bodo.libs.bool_arr_ext.init_bool_array(data_recv, bitmap_recv, n_loc)

        return scatterv_impl_bool_arr

    if (
        isinstance(
            data,
            (
                IntegerArrayType,
                FloatingArrayType,
                DecimalArrayType,
                DatetimeArrayType,
                TimeArrayType,
            ),
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
        if isinstance(data, DatetimeArrayType):
            tz = data.tz
            init_func = numba.njit(no_cpython_wrapper=True)(
                lambda d, b: bodo.libs.pd_datetime_arr_ext.init_datetime_array(d, b, tz)
            )  # pragma: no cover
        if data == datetime_date_array_type:
            init_func = bodo.hiframes.datetime_date_ext.init_datetime_date_array
        if isinstance(data, TimeArrayType):
            precision = data.precision
            init_func = numba.njit(no_cpython_wrapper=True)(
                lambda d, b: bodo.hiframes.time_ext.init_time_array(
                    d, b, precision
                )  # pragma: no cover
            )

        def scatterv_impl_int_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            data_in = data._data
            null_bitmap = data._null_bitmap
            n_in = len(data_in)

            data_recv = _scatterv_np(data_in, send_counts, warn_if_dist, root, comm)
            out_null_bitmap = _scatterv_null_bitmap(
                null_bitmap, send_counts, n_in, root, comm
            )

            return init_func(data_recv, out_null_bitmap)

        return scatterv_impl_int_arr

    # interval array
    if isinstance(data, IntervalArrayType):
        # scatter the left/right arrays
        def impl_interval_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            left_chunk = bodo.libs.distributed_api.scatterv_impl(
                data._left, send_counts, warn_if_dist, root, comm
            )
            right_chunk = bodo.libs.distributed_api.scatterv_impl(
                data._right, send_counts, warn_if_dist, root, comm
            )
            return bodo.libs.interval_arr_ext.init_interval_array(
                left_chunk, right_chunk
            )

        return impl_interval_arr

    # DatetimeTimeDeltaArrayType
    if data == datetime_timedelta_array_type:

        def impl_timedelta_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            days = bodo.libs.distributed_api.scatterv_impl(
                data._days_data, send_counts, warn_if_dist, root, comm
            )
            seconds = bodo.libs.distributed_api.scatterv_impl(
                data._seconds_data, send_counts, warn_if_dist, root, comm
            )
            microseconds = bodo.libs.distributed_api.scatterv_impl(
                data._microseconds_data, send_counts, warn_if_dist, root, comm
            )
            out_null_bitmap = _scatterv_null_bitmap(
                data._null_bitmap, send_counts, len(data), root, comm
            )
            return bodo.hiframes.datetime_timedelta_ext.init_datetime_timedelta_array(
                days, seconds, microseconds, out_null_bitmap
            )

        return impl_timedelta_arr

    # NullArray
    if data == bodo.null_array_type:

        def impl_null_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            _, is_intercomm, is_sender, _ = get_scatter_comm_info(root, comm)
            n = bodo.libs.distributed_api.get_node_portion(
                bcast_scalar(len(data), root, comm), bodo.get_size(), bodo.get_rank()
            )
            if is_intercomm and is_sender:
                n = 0
            return bodo.libs.null_arr_ext.init_null_array(n)

        return impl_null_arr

    # TimestampTZ array
    if data == bodo.timestamptz_array_type:
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_timestamp_tz_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            _, _, is_sender, n_pes = get_scatter_comm_info(root, comm)

            data_ts_in = data.data_ts
            data_offset_in = data.data_offset
            null_bitmap = data._null_bitmap
            n_in = len(data_ts_in)

            data_ts_recv = _scatterv_np(
                data_ts_in, send_counts, warn_if_dist, root, comm
            )
            data_offset_recv = _scatterv_np(
                data_offset_in, send_counts, warn_if_dist, root, comm
            )

            n_all = bcast_scalar(n_in, root, comm)
            n_recv_bytes = (len(data_ts_recv) + 7) >> 3
            bitmap_recv = np.empty(n_recv_bytes, np.uint8)

            send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int64)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            send_null_bitmap = get_scatter_null_bytes_buff(
                null_bitmap.ctypes, send_counts, send_counts_nulls, is_sender
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                bitmap_recv.ctypes,
                np.int64(n_recv_bytes),
                char_typ_enum,
                root,
                comm,
            )
            return bodo.hiframes.timestamptz_ext.init_timestamptz_array(
                data_ts_recv, data_offset_recv, bitmap_recv
            )

        return impl_timestamp_tz_arr

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):
        # TODO: support send_counts
        def impl_range_index(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)

            start = data._start
            stop = data._stop
            step = data._step
            name = data._name

            name = bcast_scalar(name, root, comm)

            start = bcast_scalar(start, root, comm)
            stop = bcast_scalar(stop, root, comm)
            step = bcast_scalar(step, root, comm)
            n_items = bodo.libs.array_kernels.calc_nitems(start, stop, step)
            chunk_start = bodo.libs.distributed_api.get_start(n_items, n_pes, rank)
            chunk_count = bodo.libs.distributed_api.get_node_portion(
                n_items, n_pes, rank
            )
            new_start = start + step * chunk_start
            new_stop = start + step * (chunk_start + chunk_count)
            new_stop = min(new_stop, stop) if step > 0 else max(new_stop, stop)

            if is_intercomm and is_sender:
                new_start = new_stop = 0

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
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            data_in = data._data
            name = data._name
            name = bcast_scalar(name, root, comm)
            arr = bodo.libs.distributed_api.scatterv_impl(
                data_in, send_counts, warn_if_dist, root, comm
            )
            return bodo.hiframes.pd_index_ext.init_period_index(arr, name, freq)

        return impl_period_index

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):

        def impl_pd_index(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            data_in = data._data
            name = data._name
            name = bcast_scalar(name, root, comm)
            arr = bodo.libs.distributed_api.scatterv_impl(
                data_in, send_counts, warn_if_dist, root, comm
            )
            return bodo.utils.conversion.index_from_array(arr, name)

        return impl_pd_index

    # MultiIndex index
    if isinstance(data, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        # TODO: handle `levels` and `codes` when available
        def impl_multi_index(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            all_data = bodo.libs.distributed_api.scatterv_impl(
                data._data, send_counts, warn_if_dist, root, comm
            )
            name = bcast_scalar(data._name, root, comm)
            names = bcast_tuple(data._names, root, comm)
            return bodo.hiframes.pd_multi_index_ext.init_multi_index(
                all_data, names, name
            )

        return impl_multi_index

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl_series(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # scatter data
            out_name = bcast_scalar(name, root, comm)
            out_arr = bodo.libs.distributed_api.scatterv_impl(
                arr, send_counts, warn_if_dist, root, comm
            )
            out_index = bodo.libs.distributed_api.scatterv_impl(
                index, send_counts, warn_if_dist, root, comm
            )
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, out_name)

        return impl_series

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        __col_name_meta_scaterv_impl = ColNamesMetaType(data.columns)

        func_text = f"def impl_df(data, send_counts=None, warn_if_dist=True, root={DEFAULT_ROOT}, comm=0):\n"
        if data.is_table_format:
            func_text += (
                "  table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(data)\n"
            )
            func_text += "  g_table = bodo.libs.distributed_api.scatterv_impl(table, send_counts, warn_if_dist, root, comm)\n"
            data_args = "g_table"
        else:
            for i in range(n_cols):
                func_text += f"  data_{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {i})\n"
                func_text += f"  g_data_{i} = bodo.libs.distributed_api.scatterv_impl(data_{i}, send_counts, warn_if_dist, root, comm)\n"
            data_args = ", ".join(f"g_data_{i}" for i in range(n_cols))
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
        )
        func_text += "  g_index = bodo.libs.distributed_api.scatterv_impl(index, send_counts, warn_if_dist, root, comm)\n"
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
        func_text = f"def impl_table(data, send_counts=None, warn_if_dist=True, root={DEFAULT_ROOT}, comm=0):\n"
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
            func_text += f"    out_arr_{blk} = bodo.libs.distributed_api.scatterv_impl(arr_list_{blk}[i], send_counts, warn_if_dist, root, comm)\n"
            func_text += f"    out_arr_list_{blk}[i] = out_arr_{blk}\n"
            func_text += f"    l = len(out_arr_{blk})\n"
            func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"
        func_text += "  T2 = set_table_len(T2, l)\n"
        func_text += "  return T2\n"

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
        empty_int32_arr = np.array([], np.int32)

        def impl_dict_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # broadcast the dictionary data (string array)
            str_arr = bodo.libs.distributed_api.bcast(
                data._data, empty_int32_arr, root, comm
            )
            # scatter indices array
            new_indices = bodo.libs.distributed_api.scatterv_impl(
                data._indices, send_counts, warn_if_dist, root, comm
            )
            # the dictionary is global by construction (broadcast)
            return bodo.libs.dict_arr_ext.init_dict_arr(
                str_arr, new_indices, True, data._has_unique_local_dictionary, None
            )

        return impl_dict_arr

    if isinstance(data, CategoricalArrayType):

        def impl_cat(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            codes = bodo.libs.distributed_api.scatterv_impl(
                data.codes, send_counts, warn_if_dist, root, comm
            )
            return bodo.hiframes.pd_categorical_ext.init_categorical_array(
                codes, data.dtype
            )

        return impl_cat

    # Tuple of data containers
    if isinstance(data, types.BaseTuple):
        func_text = f"def impl_tuple(data, send_counts=None, warn_if_dist=True, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(
                f"bodo.libs.distributed_api.scatterv_impl(data[{i}], send_counts, warn_if_dist, root, comm)"
                for i in range(len(data))
            ),
            "," if len(data) > 0 else "",
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl_tuple = loc_vars["impl_tuple"]
        return impl_tuple

    # List of distributable data
    if isinstance(data, types.List) and is_distributable_typ(data.dtype):

        def impl_list(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            is_sender = rank == root
            if comm != 0:
                is_sender = root == MPI.ROOT

            length = bcast_scalar(len(data), root, comm)
            out = []
            for i in range(length):
                in_val = data[i] if is_sender else data[0]
                out.append(
                    bodo.libs.distributed_api.scatterv_impl(
                        in_val, send_counts, warn_if_dist, root, comm
                    )
                )

            return out

        return impl_list

    # Dictionary of distributable data
    if isinstance(data, types.DictType) and is_distributable_typ(data.value_type):

        def impl_dict(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            is_sender = rank == root
            if comm != 0:
                is_sender = root == MPI.ROOT

            length = bcast_scalar(len(data), root, comm)
            in_keys = list(data.keys())
            in_values = list(data.values())
            out = {}
            for i in range(length):
                key = in_keys[i] if is_sender else in_keys[0]
                value = in_values[i] if is_sender else in_values[0]
                out_key = bcast_scalar(key, root, comm)
                out[out_key] = bodo.libs.distributed_api.scatterv_impl(
                    value, send_counts, warn_if_dist, root, comm
                )

            return out

        return impl_dict

    # StructArray
    if isinstance(data, bodo.StructArrayType):
        n_fields = len(data.data)
        func_text = f"def impl_struct(data, send_counts=None, warn_if_dist=True, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  inner_data_arrs = bodo.libs.struct_arr_ext.get_data(data)\n"
        func_text += "  out_null_bitmap = _scatterv_null_bitmap(bodo.libs.struct_arr_ext.get_null_bitmap(data), send_counts, len(data), root, comm)\n"
        for i in range(n_fields):
            func_text += f"  new_inner_data_arr_{i} = bodo.libs.distributed_api.scatterv_impl(inner_data_arrs[{i}], send_counts, warn_if_dist, root, comm)\n"

        new_data_tuple_str = "({}{})".format(
            ", ".join([f"new_inner_data_arr_{i}" for i in range(n_fields)]),
            "," if n_fields > 0 else "",
        )
        field_names_tuple_str = "({}{})".format(
            ", ".join([f"'{f}'" for f in data.names]),
            "," if n_fields > 0 else "",
        )
        out_len = (
            "len(new_inner_data_arr_0)"
            if n_fields > 0
            else "bodo.libs.distributed_api.get_node_portion(bcast_scalar(len(data), root, comm), bodo.get_size(), bodo.get_rank())"
        )
        func_text += f"  return bodo.libs.struct_arr_ext.init_struct_arr({out_len}, {new_data_tuple_str}, out_null_bitmap, {field_names_tuple_str})\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "_scatterv_null_bitmap": _scatterv_null_bitmap,
                "bcast_scalar": bcast_scalar,
            },
            loc_vars,
        )
        impl_struct = loc_vars["impl_struct"]
        return impl_struct

    # MapArrayType
    if isinstance(data, bodo.MapArrayType):

        def impl(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # Call it recursively on the underlying ArrayItemArray array.
            new_underlying_data = bodo.libs.distributed_api.scatterv_impl(
                data._data, send_counts, warn_if_dist, root, comm
            )
            # Reconstruct the Map array from the new ArrayItemArray array.
            new_data = bodo.libs.map_arr_ext.init_map_arr(new_underlying_data)
            return new_data

        return impl

    # TupleArray
    if isinstance(data, bodo.TupleArrayType):

        def impl_tuple(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            new_underlying_data = bodo.libs.distributed_api.scatterv_impl(
                data._data, send_counts, warn_if_dist, root, comm
            )
            return bodo.libs.tuple_arr_ext.init_tuple_arr(new_underlying_data)

        return impl_tuple

    if data is types.none:  # pragma: no cover
        return (
            lambda data,
            send_counts=None,
            warn_if_dist=True,
            root=DEFAULT_ROOT,
            comm=0: None
        )

    if isinstance(data, bodo.MatrixType):

        def impl_matrix(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            new_underlying_data = bodo.libs.distributed_api.scatterv_impl(
                data.data, send_counts, warn_if_dist, root, comm
            )
            return bodo.libs.matrix_ext.init_np_matrix(new_underlying_data)

        return impl_matrix

    raise BodoError(f"scatterv() not available for {data}")  # pragma: no cover


char_typ_enum = np.int32(numba_to_c_type(types.uint8))


@numba.njit(cache=True, no_cpython_wrapper=True)
def _scatterv_null_bitmap(null_bitmap, send_counts, n_in, root, comm):
    """Scatter null bitmap for nullable arrays"""
    rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)

    n_all = bcast_scalar(n_in, root, comm)

    send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)
    n_loc = 0 if (is_intercomm and is_sender) else send_counts[rank]

    n_recv_bytes = (n_loc + 7) >> 3
    bitmap_recv = np.empty(n_recv_bytes, np.uint8)

    # compute send counts for nulls
    send_counts_nulls = np.empty(n_pes, np.int64)
    for i in range(n_pes):
        send_counts_nulls[i] = (send_counts[i] + 7) >> 3

    # displacements for nulls
    displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

    send_null_bitmap = get_scatter_null_bytes_buff(
        null_bitmap.ctypes, send_counts, send_counts_nulls, is_sender
    )

    c_scatterv(
        send_null_bitmap.ctypes,
        send_counts_nulls.ctypes,
        displs_nulls.ctypes,
        bitmap_recv.ctypes,
        np.int64(n_recv_bytes),
        char_typ_enum,
        root,
        comm,
    )
    return bitmap_recv


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


def gen_bcast_scalar_impl(val, root=DEFAULT_ROOT, comm=0):
    if val == types.none:
        return lambda val, root=DEFAULT_ROOT, comm=0: None

    if val == bodo.timestamptz_type:

        def impl(val, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
            updated_timestamp = bodo.libs.distributed_api.bcast_scalar(
                val.utc_timestamp, root, comm
            )
            updated_offset = bodo.libs.distributed_api.bcast_scalar(
                val.offset_minutes, root, comm
            )
            return bodo.TimestampTZ(updated_timestamp, updated_offset)

        return impl

    if val == bodo.datetime_date_type:
        c_type = numba_to_c_type(types.int32)

        # Note: There are issues calling this function with recursion.
        # As a result we just implement it directly.
        def impl(val, root=DEFAULT_ROOT, comm=0):  # pragma: no cover
            send = np.empty(1, np.int32)
            send[0] = bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int(val)
            c_bcast(send.ctypes, np.int32(1), np.int32(c_type), np.int32(root), comm)
            return bodo.hiframes.datetime_date_ext.cast_int_to_datetime_date(send[0])

        return impl

    if isinstance(val, bodo.PandasTimestampType):
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

    if val == bodo.string_type:
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
    # TODO: fix np.full and refactor
    func_text = (
        f"def bcast_scalar_impl(val, root={DEFAULT_ROOT}, comm=0):\n"
        "  send = np.empty(1, dtype)\n"
        "  send[0] = val\n"
        f"  c_bcast(send.ctypes, np.int32(1), np.int32({typ_val}), np.int32(root), comm)\n"
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


dummy_use = numba.njit(cache=True, no_cpython_wrapper=True)(lambda a: None)


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
            c_bcast(data_ptr, np.int32(l), char_typ_enum, np.int32(root), 0)
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
        return (
            lambda send_data,
            out_data,
            send_counts,
            recv_counts,
            send_disp,
            recv_disp: c_alltoallv(
                send_data._data.ctypes,
                out_data._data.ctypes,
                send_counts.ctypes,
                recv_counts.ctypes,
                send_disp.ctypes,
                recv_disp.ctypes,
                typ_enum,
            )
        )  # pragma: no cover

    if isinstance(send_data, bodo.CategoricalArrayType):
        return (
            lambda send_data,
            out_data,
            send_counts,
            recv_counts,
            send_disp,
            recv_disp: c_alltoallv(
                send_data.codes.ctypes,
                out_data.codes.ctypes,
                send_counts.ctypes,
                recv_counts.ctypes,
                send_disp.ctypes,
                recv_disp.ctypes,
                typ_enum,
            )
        )  # pragma: no cover

    return (
        lambda send_data,
        out_data,
        send_counts,
        recv_counts,
        send_disp,
        recv_disp: c_alltoallv(
            send_data.ctypes,
            out_data.ctypes,
            send_counts.ctypes,
            recv_counts.ctypes,
            send_disp.ctypes,
            recv_disp.ctypes,
            typ_enum,
        )
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

    func_text = "def bodo_alltoallv_tup(send_data, out_data, send_counts, recv_counts, send_disp, recv_disp):\n"
    for i in range(count):
        func_text += f"  alltoallv(send_data[{i}], out_data[{i}], send_counts, recv_counts, send_disp, recv_disp)\n"
    func_text += "  return\n"

    return bodo_exec(func_text, {"alltoallv": alltoallv}, {}, __name__)


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


_wait = types.ExternalFunction("dist_wait", types.void(mpi_req_numba_type, types.bool_))


@numba.generated_jit(nopython=True)
def wait(req, cond=True):
    """wait on MPI request"""
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


########### finalize MPI & s3_reader, disconnect hdfs when exiting ############


from bodo.io import hdfs_reader

finalize = hdist.finalize_py_wrapper
disconnect_hdfs_py_wrapper = hdfs_reader.disconnect_hdfs_py_wrapper

ll.add_symbol("disconnect_hdfs", hdfs_reader.disconnect_hdfs)
disconnect_hdfs = types.ExternalFunction("disconnect_hdfs", types.int32())


@numba.njit(cache=True)
def disconnect_hdfs_njit():  # pragma: no cover
    """
    Simple njit wrapper around disconnect_hdfs.
    This is useful for resetting the singleton
    hadoop filesystem instance. This is a NOP
    if the filesystem hasn't been initialized yet.
    """
    disconnect_hdfs()


def call_finalize():  # pragma: no cover
    finalize()
    disconnect_hdfs_py_wrapper()


def flush_stdout():
    # using a function since pytest throws an error sometimes
    # if flush function is passed directly to atexit
    if not sys.stdout.closed:
        sys.stdout.flush()


atexit.register(call_finalize)
# Flush output before finalize
atexit.register(flush_stdout)


ll.add_symbol("broadcast_array_py_entry", hdist.broadcast_array_py_entry)
c_broadcast_array = ExternalFunctionErrorChecked(
    "broadcast_array_py_entry",
    array_info_type(array_info_type, array_info_type, types.int32, types.int64),
)
ll.add_symbol("broadcast_table_py_entry", hdist.broadcast_table_py_entry)
c_broadcast_table = ExternalFunctionErrorChecked(
    "broadcast_table_py_entry",
    table_type(table_type, array_info_type, types.int32, types.int64),
)


def bcast(data, comm_ranks=None, root=DEFAULT_ROOT, comm=None):  # pragma: no cover
    """bcast() sends data from rank 0 to comm_ranks."""
    from bodo.mpi4py import MPI

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
delete_is_last_state = types.ExternalFunction(
    "delete_is_last_state", types.none(is_last_state_type)
)
# NOTE: using int32 types to avoid i1 vs i8 boolean errors in lowering
sync_is_last_non_blocking = types.ExternalFunction(
    "sync_is_last_non_blocking", types.int32(is_last_state_type, types.int32)
)
