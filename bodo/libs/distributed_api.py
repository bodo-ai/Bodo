# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
from enum import Enum
import math
import time
import pandas as pd
import numpy as np
import sys
import atexit
from decimal import Decimal
import datetime
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType, int128_type

import numba
from numba.core import types, cgutils, ir_utils
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.core.typing import signature
from numba.extending import models, register_model, intrinsic, overload

import bodo
from bodo.libs.str_arr_ext import (
    string_array_type,
    num_total_chars,
    pre_alloc_string_array,
    get_offset_ptr,
    get_null_bitmap_ptr,
    get_data_ptr,
    convert_len_arr_to_offset,
    getitem_str_bitmap,
    setitem_str_bitmap,
    set_bit_to,
    get_bit_bitmap,
)
from bodo.libs.int_arr_ext import IntegerArrayType, set_bit_to_arr
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.list_str_arr_ext import (
    list_string_array_type,
    pre_alloc_list_string_array,
)
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayType,
    pre_alloc_array_item_array,
)
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.hiframes.pd_categorical_ext import CategoricalArray
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.utils.utils import (
    debug_prints,
    empty_like_type,
    numba_to_c_type,
    unliteral_all,
    CTypeEnum,
    tuple_to_scalar,
)
from bodo.utils.typing import BodoError
from numba.core.typing.builtins import IndexValueType
from llvmlite import ir as lir
from bodo.libs import hdist

import llvmlite.binding as ll


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
ll.add_symbol("comm_req_alloc", hdist.comm_req_alloc)
ll.add_symbol("comm_req_dealloc", hdist.comm_req_dealloc)
ll.add_symbol("req_array_setitem", hdist.req_array_setitem)
ll.add_symbol("dist_waitall", hdist.dist_waitall)
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


# get size dynamically from C code (mpich 3.2 is 4 bytes but openmpi 1.6 is 8)
mpi_req_numba_type = getattr(types, "int" + str(8 * hdist.mpi_req_num_bytes))

MPI_ROOT = 0
ANY_SOURCE = np.int32(hdist.ANY_SOURCE)


class Reduce_Type(Enum):
    Sum = 0
    Prod = 1
    Min = 2
    Max = 3
    Argmin = 4
    Argmax = 5
    Or = 6


_get_rank = types.ExternalFunction("c_get_rank", types.int32())
_get_size = types.ExternalFunction("c_get_size", types.int32())
_barrier = types.ExternalFunction("c_barrier", types.int32())


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


@overload(time.time, no_unliteral=True)
def overload_time_time():
    return lambda: _get_time()


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
    if isinstance(arr, types.Array):

        def impl(arr, size, pe, tag, cond=True):  # pragma: no cover
            type_enum = get_type_enum(arr)
            return _isend(arr.ctypes, size, type_enum, pe, tag, cond)

        return impl

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


@numba.njit
def irecv(arr, size, pe, tag, cond=True):  # pragma: no cover
    type_enum = get_type_enum(arr)
    return _irecv(arr.ctypes, size, type_enum, pe, tag, cond)


_alltoall = types.ExternalFunction(
    "c_alltoall", types.void(types.voidptr, types.voidptr, types.int32, types.int32)
)


@numba.njit
def alltoall(send_arr, recv_arr, count):  # pragma: no cover
    # TODO: handle int64 counts
    assert count < INT_MAX
    type_enum = get_type_enum(send_arr)
    _alltoall(send_arr.ctypes, recv_arr.ctypes, np.int32(count), type_enum)


@numba.generated_jit(nopython=True)
def gather_scalar(data, allgather=False):
    data = types.unliteral(data)
    typ_val = numba_to_c_type(data)
    dtype = data

    def gather_scalar_impl(data, allgather=False):  # pragma: no cover
        n_pes = bodo.libs.distributed_api.get_size()
        rank = bodo.libs.distributed_api.get_rank()
        send = np.full(1, data, dtype)
        res_size = n_pes if (rank == MPI_ROOT or allgather) else 0
        res = np.empty(res_size, dtype)
        c_gather_scalar(send.ctypes, res.ctypes, np.int32(typ_val), allgather)
        return res

    return gather_scalar_impl


c_gather_scalar = types.ExternalFunction(
    "c_gather_scalar",
    types.void(types.voidptr, types.voidptr, types.int32, types.bool_),
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
    def codegen(context, builder, sig, args):
        ptr = cgutils.alloca_once(builder, args[0].type)
        builder.store(args[0], ptr)
        return builder.bitcast(ptr, lir.IntType(8).as_pointer())

    return types.voidptr(val_tp), codegen


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


@numba.generated_jit(nopython=True)
def dist_reduce(value, reduce_op):
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
        supported_typs = [types.int32, types.float32, types.float64]
        import sys

        if not sys.platform.startswith("win"):
            # long is 4 byte on Windows
            supported_typs.append(types.int64)
            supported_typs.append(types.NPDatetime("ns"))
        if target_typ not in supported_typs:  # pragma: no cover
            raise TypeError(
                "argmin/argmax not supported for type {}".format(target_typ)
            )

    typ_enum = np.int32(numba_to_c_type(target_typ))

    def impl(value, reduce_op):  # pragma: no cover
        in_ptr = value_to_ptr(value)
        out_ptr = value_to_ptr(value)
        _dist_reduce(in_ptr, out_ptr, reduce_op, typ_enum)
        return load_val_ptr(out_ptr, value)

    return impl


_dist_exscan = types.ExternalFunction(
    "dist_exscan", types.void(types.voidptr, types.voidptr, types.int32, types.int32)
)


@numba.generated_jit(nopython=True)
def dist_exscan(value, reduce_op):
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


@numba.generated_jit(nopython=True)
def gatherv(data, allgather=False):

    if isinstance(data, CategoricalArray):

        def impl_cat(data, allgather=False):  # pragma: no cover
            codes = bodo.gatherv(data._codes, allgather)
            return bodo.hiframes.pd_categorical_ext.init_categorical_array(
                codes, data.dtype
            )

        return impl_cat

    if isinstance(data, types.Array):
        typ_val = numba_to_c_type(data.dtype)

        def gatherv_impl(data, allgather=False):  # pragma: no cover
            data = np.ascontiguousarray(data)
            rank = bodo.libs.distributed_api.get_rank()
            # size to handle multi-dim arrays
            n_loc = data.size
            recv_counts = gather_scalar(np.int32(n_loc), allgather)
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            displs = np.empty(1, np.int32)
            if rank == MPI_ROOT or allgather:
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
            )
            # handle multi-dim case
            return all_data.reshape((-1,) + data.shape[1:])

        return gatherv_impl

    if data == string_array_type:
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def gatherv_str_arr_impl(data, allgather=False):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_all_chars = num_total_chars(data)

            # allocate send lens arrays
            send_arr_lens = np.empty(n_loc, np.uint32)  # XXX offset type is uint32
            send_data_ptr = get_data_ptr(data)
            send_null_bitmap_ptr = get_null_bitmap_ptr(data)
            n_bytes = (n_loc + 7) >> 3

            for i in range(n_loc):
                send_arr_lens[i] = bodo.libs.str_arr_ext.get_str_arr_item_length(
                    data, i
                )

            recv_counts = gather_scalar(np.int32(n_loc), allgather)
            recv_counts_char = gather_scalar(np.int32(n_all_chars), allgather)
            n_total = recv_counts.sum()
            n_total_char = recv_counts_char.sum()

            # displacements
            all_data = pre_alloc_string_array(0, 0)  # dummy arrays on non-root PEs
            displs = np.empty(1, np.int32)
            displs_char = np.empty(1, np.int32)
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)

            if rank == MPI_ROOT or allgather:
                all_data = pre_alloc_string_array(n_total, n_total_char)
                displs = bodo.ir.join.calc_disp(recv_counts)
                displs_char = bodo.ir.join.calc_disp(recv_counts_char)
                recv_counts_nulls = np.empty(len(recv_counts), np.int32)
                for i in range(len(recv_counts)):
                    recv_counts_nulls[i] = (recv_counts[i] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)

            offset_ptr = get_offset_ptr(all_data)
            data_ptr = get_data_ptr(all_data)
            null_bitmap_ptr = get_null_bitmap_ptr(all_data)

            c_gatherv(
                send_arr_lens.ctypes,
                np.int32(n_loc),
                offset_ptr,
                recv_counts.ctypes,
                displs.ctypes,
                int32_typ_enum,
                allgather,
            )
            c_gatherv(
                send_data_ptr,
                np.int32(n_all_chars),
                data_ptr,
                recv_counts_char.ctypes,
                displs_char.ctypes,
                char_typ_enum,
                allgather,
            )
            c_gatherv(
                send_null_bitmap_ptr,
                np.int32(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_nulls.ctypes,
                displs_nulls.ctypes,
                char_typ_enum,
                allgather,
            )

            convert_len_arr_to_offset(offset_ptr, n_total)
            copy_gathered_null_bytes(
                null_bitmap_ptr, tmp_null_bytes, recv_counts_nulls, recv_counts
            )
            return all_data

        return gatherv_str_arr_impl

    if isinstance(data, (IntegerArrayType, DecimalArrayType)) or data in (
        boolean_array,
        datetime_date_array_type,
    ):
        typ_val = numba_to_c_type(data.dtype)
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def gatherv_impl_int_arr(data, allgather=False):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_bytes = (n_loc + 7) >> 3
            recv_counts = gather_scalar(np.int32(n_loc), allgather)
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            displs = np.empty(1, np.int32)
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)
            if rank == MPI_ROOT or allgather:
                displs = bodo.ir.join.calc_disp(recv_counts)
                recv_counts_nulls = np.empty(len(recv_counts), np.int32)
                for i in range(len(recv_counts)):
                    recv_counts_nulls[i] = (recv_counts[i] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)
            #  print(rank, n_loc, n_total, recv_counts, displs)
            c_gatherv(
                data._data.ctypes,
                np.int32(n_loc),
                all_data._data.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                np.int32(typ_val),
                allgather,
            )
            c_gatherv(
                data._null_bitmap.ctypes,
                np.int32(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_nulls.ctypes,
                displs_nulls.ctypes,
                char_typ_enum,
                allgather,
            )
            copy_gathered_null_bytes(
                all_data._null_bitmap.ctypes,
                tmp_null_bytes,
                recv_counts_nulls,
                recv_counts,
            )
            return all_data

        return gatherv_impl_int_arr

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl(data, allgather=False):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # gather data
            out_arr = bodo.libs.distributed_api.gatherv(arr, allgather)
            out_index = bodo.gatherv(index, allgather)
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

        return impl

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):

        def impl_range_index(data, allgather=False):  # pragma: no cover
            # XXX: assuming global range starts from zero
            # and each process has a chunk, and step is 1
            local_n = data._stop - data._start
            n = bodo.libs.distributed_api.dist_reduce(
                local_n, np.int32(Reduce_Type.Sum.value)
            )
            # gatherv() of dataframe returns 0-length arrays so index should
            # be 0-length to match
            if bodo.get_rank() != 0 and not allgather:
                n = 0
            return bodo.hiframes.pd_index_ext.init_range_index(0, n, 1, data._name)

        return impl_range_index

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):
        from bodo.hiframes.pd_index_ext import PeriodIndexType

        if isinstance(data, PeriodIndexType):
            freq = data.freq

            def impl_pd_index(data, allgather=False):  # pragma: no cover
                arr = bodo.libs.distributed_api.gatherv(data._data, allgather)
                return bodo.hiframes.pd_index_ext.init_period_index(
                    arr, data._name, freq
                )

        else:

            def impl_pd_index(data, allgather=False):  # pragma: no cover
                arr = bodo.libs.distributed_api.gatherv(data._data, allgather)
                return bodo.utils.conversion.index_from_array(arr, data._name)

        return impl_pd_index

    # MultiIndex index
    if isinstance(data, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        # just gather the data arrays
        # TODO: handle `levels` and `codes` when available
        def impl_multi_index(data, allgather=False):  # pragma: no cover
            all_data = bodo.gatherv(data._data, allgather)
            return bodo.hiframes.pd_multi_index_ext.init_multi_index(
                all_data, data._names, data._name
            )

        return impl_multi_index

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        # empty dataframe case
        if n_cols == 0:
            return lambda data, allgather=False: bodo.hiframes.pd_dataframe_ext.init_dataframe(
                (), bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data), ()
            )

        data_args = ", ".join("g_data_{}".format(i) for i in range(n_cols))
        col_var = bodo.utils.transform.gen_const_tup(data.columns)

        func_text = "def impl_df(data, allgather=False):\n"
        for i in range(n_cols):
            func_text += "  data_{} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {})\n".format(
                i, i
            )
            func_text += "  g_data_{} = bodo.gatherv(data_{}, allgather)\n".format(i, i)
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
        )
        func_text += "  g_index = bodo.gatherv(index, allgather)\n"
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), g_index, {})\n".format(
            data_args, col_var
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl_df = loc_vars["impl_df"]
        return impl_df

    if data == list_string_array_type:
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def gatherv_list_str_arr_impl(data, allgather=False):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_all_strs = data._num_total_strings
            n_all_chars = data._num_total_chars

            # allocate buffer for sending lengths of lists and strings
            send_list_lens = np.empty(n_loc, np.uint32)  # XXX offset type is uint32
            send_str_lens = np.empty(n_all_strs, np.uint32)
            n_bytes = (n_loc + 7) >> 3

            for i in range(n_loc):
                send_list_lens[i] = data._index_offsets[i + 1] - data._index_offsets[i]
            for j in range(n_all_strs):
                send_str_lens[j] = data._data_offsets[j + 1] - data._data_offsets[j]

            recv_counts = gather_scalar(np.int32(n_loc), allgather)
            recv_counts_str = gather_scalar(np.int32(n_all_strs), allgather)
            recv_counts_char = gather_scalar(np.int32(n_all_chars), allgather)
            n_total = recv_counts.sum()
            n_total_str = recv_counts_str.sum()
            n_total_char = recv_counts_char.sum()

            # displacements
            all_data = pre_alloc_list_string_array(
                0, 0, 0
            )  # dummy arrays on non-root PEs
            displs = np.empty(1, np.int32)
            displs_str = np.empty(1, np.int32)
            displs_char = np.empty(1, np.int32)
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)

            if rank == MPI_ROOT or allgather:
                all_data = pre_alloc_list_string_array(
                    n_total, n_total_str, n_total_char
                )
                displs = bodo.ir.join.calc_disp(recv_counts)
                displs_str = bodo.ir.join.calc_disp(recv_counts_str)
                displs_char = bodo.ir.join.calc_disp(recv_counts_char)
                recv_counts_nulls = np.empty(len(recv_counts), np.int32)
                for k in range(len(recv_counts)):
                    recv_counts_nulls[k] = (recv_counts[k] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)

            # data
            c_gatherv(
                cptr_to_voidptr(data._data),
                np.int32(n_all_chars),
                cptr_to_voidptr(all_data._data),
                recv_counts_char.ctypes,
                displs_char.ctypes,
                char_typ_enum,
                allgather,
            )
            # data offset
            c_gatherv(
                send_str_lens.ctypes,
                np.int32(n_all_strs),
                cptr_to_voidptr(all_data._data_offsets),
                recv_counts_str.ctypes,
                displs_str.ctypes,
                int32_typ_enum,
                allgather,
            )
            # index offset
            c_gatherv(
                send_list_lens.ctypes,
                np.int32(n_loc),
                cptr_to_voidptr(all_data._index_offsets),
                recv_counts.ctypes,
                displs.ctypes,
                int32_typ_enum,
                allgather,
            )
            c_gatherv(
                cptr_to_voidptr(data._null_bitmap),
                np.int32(n_bytes),
                tmp_null_bytes.ctypes,
                recv_counts_nulls.ctypes,
                displs_nulls.ctypes,
                char_typ_enum,
                allgather,
            )
            dummy_use(data)  # needed?

            convert_len_arr_to_offset(
                cptr_to_voidptr(all_data._data_offsets), n_total_str
            )
            convert_len_arr_to_offset(cptr_to_voidptr(all_data._index_offsets), n_total)
            copy_gathered_null_bytes(
                cptr_to_voidptr(all_data._null_bitmap),
                tmp_null_bytes,
                recv_counts_nulls,
                recv_counts,
            )
            return all_data

        return gatherv_list_str_arr_impl

    # array(item) array
    if isinstance(data, ArrayItemArrayType):
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def gatherv_array_item_arr_impl(data, allgather=False):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            offsets_arr = bodo.libs.array_item_arr_ext.get_offsets(data)
            data_arr = bodo.libs.array_item_arr_ext.get_data(data)
            null_bitmap_arr = bodo.libs.array_item_arr_ext.get_null_bitmap(data)
            n_loc = len(data)

            # allocate buffer for sending lengths of lists
            send_list_lens = np.empty(n_loc, np.uint32)  # XXX offset type is uint32
            n_bytes = (n_loc + 7) >> 3

            for i in range(n_loc):
                send_list_lens[i] = offsets_arr[i + 1] - offsets_arr[i]

            recv_counts = gather_scalar(np.int32(n_loc), allgather)
            n_total = recv_counts.sum()

            # displacements
            displs = np.empty(1, np.int32)
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)

            if rank == MPI_ROOT or allgather:
                displs = bodo.ir.join.calc_disp(recv_counts)
                recv_counts_nulls = np.empty(len(recv_counts), np.int32)
                for k in range(len(recv_counts)):
                    recv_counts_nulls[k] = (recv_counts[k] + 7) >> 3
                displs_nulls = bodo.ir.join.calc_disp(recv_counts_nulls)
                tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)

            out_offsets_arr = np.empty(n_total + 1, np.uint32)
            out_data_arr = bodo.gatherv(data_arr)
            out_null_bitmap_arr = np.empty((n_total + 7) >> 3, np.uint8)

            # index offset
            c_gatherv(
                send_list_lens.ctypes,
                np.int32(n_loc),
                out_offsets_arr.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                int32_typ_enum,
                allgather,
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
            )
            dummy_use(data)  # needed?

            convert_len_arr_to_offset(out_offsets_arr.ctypes, n_total)
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

        def impl_struct_arr(data, allgather=False):
            data_arrs = bodo.libs.struct_arr_ext.get_data(data)
            null_bitmap = bodo.libs.struct_arr_ext.get_null_bitmap(data)
            out_data_arrs = bodo.gatherv(data_arrs, allgather=allgather)
            # gather the null bits similar to other arrays
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_bytes = (n_loc + 7) >> 3
            recv_counts = gather_scalar(np.int32(n_loc), allgather)
            n_total = recv_counts.sum()
            out_null_bitmap = np.empty((n_total + 7) >> 3, np.uint8)
            # displacements
            recv_counts_nulls = np.empty(1, np.int32)
            displs_nulls = np.empty(1, np.int32)
            tmp_null_bytes = np.empty(1, np.uint8)
            if rank == MPI_ROOT or allgather:
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
            )
            copy_gathered_null_bytes(
                out_null_bitmap.ctypes, tmp_null_bytes, recv_counts_nulls, recv_counts,
            )
            return bodo.libs.struct_arr_ext.init_struct_arr(
                out_data_arrs, out_null_bitmap, names
            )

        return impl_struct_arr

    # Tuple of data containers
    if isinstance(data, types.BaseTuple):
        func_text = "def impl_tuple(data, allgather=False):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(
                "bodo.gatherv(data[{}], allgather)".format(i) for i in range(len(data))
            ),
            "," if len(data) > 0 else "",
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl_tuple = loc_vars["impl_tuple"]
        return impl_tuple

    if data is types.none:
        return lambda data: None

    raise BodoError("gatherv() not available for {}".format(data))  # pragma: no cover


@numba.generated_jit(nopython=True)
def allgatherv(data):
    return lambda data: gatherv(data, True)


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


def _bcast_dtype(data):
    """broadcast data type from rank 0 using mpi4py
    """
    try:
        from mpi4py import MPI
    except:  # pragma: no cover
        raise BodoError("mpi4py is required for scatterv")

    comm = MPI.COMM_WORLD
    data = comm.bcast(data)
    return data


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _scatterv_np(data):
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

    def scatterv_arr_impl(data):  # pragma: no cover
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

        # compute send counts
        sendcounts = np.empty(n_pes, np.int32)
        for i in range(n_pes):
            sendcounts[i] = get_node_portion(shape[0], n_pes, i) * n_elem_per_row

        # allocate output
        n_loc = sendcounts[rank]  # total number of elements on this PE
        recv_data = np.empty(n_loc, dtype)

        # displacements
        displs = bodo.ir.join.calc_disp(sendcounts)

        c_scatterv(
            data_ptr,
            sendcounts.ctypes,
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
    """get a value for name of a Series/Index type
    """
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

    # Int array
    if isinstance(dtype, IntegerArrayType):
        pd_dtype = "{}Int{}".format(
            "" if dtype.dtype.signed else "U", dtype.dtype.bitwidth
        )
        return pd.array([3], pd_dtype)

    # bool array
    if dtype == boolean_array:
        return pd.array([True], "boolean")

    # Decimal array
    if isinstance(dtype, DecimalArrayType):
        return np.array([Decimal("32.1")])

    # date array
    if dtype == datetime_date_array_type:
        return np.array([datetime.date(2011, 8, 9)])

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
        name = _get_name_value_for_type(dtype.name_typ)
        names = tuple(_get_name_value_for_type(t) for t in dtype.names_typ)
        arrs = tuple(get_value_for_type(t) for t in dtype.array_types)
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

    if isinstance(dtype, CategoricalArray):
        return pd.Categorical.from_codes([0], dtype.dtype.categories)

    if dtype == list_string_array_type:
        return pd.Series([["AA", "B"],]).values

    if isinstance(dtype, types.BaseTuple):
        return tuple(get_value_for_type(t) for t in dtype.types)

    if isinstance(dtype, ArrayItemArrayType):
        if isinstance(dtype.dtype.dtype, types.Integer):
            return pd.Series([[1, 2], []]).values
        if isinstance(dtype.dtype.dtype, types.Float):
            return pd.Series([[1.0, 2.0], []]).values

    # TODO: Add missing data types
    raise BodoError("get_value_for_type(dtype): Missing data type")


def scatterv(data):
    """scatterv() distributes data from rank 0 to all ranks.
    Rank 0 passes the data but the other ranks just pass None.
    """
    rank = bodo.libs.distributed_api.get_rank()
    if rank != MPI_ROOT and data is not None:  # pragma: no cover
        raise BodoError(
            "bodo.scatterv() requires 'data' argument to be None on all ranks except rank 0."
        )

    # make sure all ranks receive the proper data type as input (instead of None)
    dtype = bodo.typeof(data)
    dtype = _bcast_dtype(dtype)
    if rank != MPI_ROOT:
        data = get_value_for_type(dtype)

    return scatterv_impl(data)


@numba.generated_jit(nopython=True)
def scatterv_impl(data):
    """nopython implementation of scatterv()
    """
    if isinstance(data, types.Array):
        return lambda data: _scatterv_np(data)  # pragma: no cover

    if data == string_array_type:
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def scatterv_str_arr_impl(data):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_pes = bodo.libs.distributed_api.get_size()
            n_all = bcast_scalar(len(data))

            # convert offsets to lengths of strings
            send_arr_lens = np.empty(len(data), np.uint32)  # XXX offset type is uint32
            for i in range(len(data)):
                send_arr_lens[i] = bodo.libs.str_arr_ext.get_str_arr_item_length(
                    data, i
                )

            # ------- calculate buffer counts -------

            # compute send counts
            sendcounts = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                sendcounts[i] = get_node_portion(n_all, n_pes, i)

            # displacements
            displs = bodo.ir.join.calc_disp(sendcounts)

            # compute send counts for characters
            sendcounts_char = np.empty(n_pes, np.int32)
            if rank == 0:
                curr_str = 0
                for i in range(n_pes):
                    c = 0
                    for _ in range(sendcounts[i]):
                        c += send_arr_lens[curr_str]
                        curr_str += 1
                    sendcounts_char[i] = c

            bcast(sendcounts_char)

            # displacements for characters
            displs_char = bodo.ir.join.calc_disp(sendcounts_char)

            # compute send counts for nulls
            sendcounts_nulls = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                sendcounts_nulls[i] = (sendcounts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(sendcounts_nulls)

            # alloc output array
            n_loc = sendcounts[rank]  # total number of elements on this PE
            n_loc_char = sendcounts_char[rank]
            recv_arr = pre_alloc_string_array(n_loc, n_loc_char)

            # ----- string lengths -----------

            c_scatterv(
                send_arr_lens.ctypes,
                sendcounts.ctypes,
                displs.ctypes,
                get_offset_ptr(recv_arr),
                np.int32(n_loc),
                int32_typ_enum,
            )

            convert_len_arr_to_offset(get_offset_ptr(recv_arr), n_loc)

            # ----- string characters -----------

            c_scatterv(
                get_data_ptr(data),
                sendcounts_char.ctypes,
                displs_char.ctypes,
                get_data_ptr(recv_arr),
                np.int32(n_loc_char),
                char_typ_enum,
            )

            # ----------- null bitmap -------------

            n_recv_bytes = (n_loc + 7) >> 3

            send_null_bitmap = get_scatter_null_bytes_buff(
                get_null_bitmap_ptr(data), sendcounts, sendcounts_nulls
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                sendcounts_nulls.ctypes,
                displs_nulls.ctypes,
                get_null_bitmap_ptr(recv_arr),
                np.int32(n_recv_bytes),
                char_typ_enum,
            )

            return recv_arr

        return scatterv_str_arr_impl

    if isinstance(data, ArrayItemArrayType):
        # Code adapted from the string code. Both the string and array(item) codes should be
        # refactored.
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        data_typ_enum = np.int32(numba_to_c_type(data.dtype.dtype))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))
        data_arr_type = data.dtype

        def scatterv_array_item_impl(data):
            in_offsets_arr = bodo.libs.array_item_arr_ext.get_offsets(data)
            in_data_arr = bodo.libs.array_item_arr_ext.get_data(data)
            in_null_bitmap_arr = bodo.libs.array_item_arr_ext.get_null_bitmap(data)
            #
            rank = bodo.libs.distributed_api.get_rank()
            n_pes = bodo.libs.distributed_api.get_size()
            n_all = bcast_scalar(len(data))

            # convert offsets to lengths of lists
            send_arr_lens = np.empty(len(data), np.uint32)  # XXX offset type is uint32
            for i in range(len(data)):
                send_arr_lens[i] = in_offsets_arr[i + 1] - in_offsets_arr[i]

            # ------- calculate buffer counts -------

            # compute send counts
            sendcounts = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                sendcounts[i] = get_node_portion(n_all, n_pes, i)

            # displacements
            displs = bodo.ir.join.calc_disp(sendcounts)

            # compute send counts for items
            sendcounts_item = np.empty(n_pes, np.int32)
            if rank == 0:
                curr_str = 0
                for i in range(n_pes):
                    c = 0
                    for _ in range(sendcounts[i]):
                        c += send_arr_lens[curr_str]
                        curr_str += 1
                    sendcounts_item[i] = c

            bcast(sendcounts_item)

            # displacements for items
            displs_item = bodo.ir.join.calc_disp(sendcounts_item)

            # compute send counts for nulls
            sendcounts_nulls = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                sendcounts_nulls[i] = (sendcounts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(sendcounts_nulls)

            # alloc output array
            n_loc = sendcounts[rank]  # total number of elements on this PE
            n_loc_item = sendcounts_item[rank]
            recv_arr = pre_alloc_array_item_array(
                n_loc, (n_loc_item,), data_arr_type
            )
            recv_offsets_arr = bodo.libs.array_item_arr_ext.get_offsets(recv_arr)
            recv_data_arr = bodo.libs.array_item_arr_ext.get_data(recv_arr)
            recv_null_bitmap_arr = bodo.libs.array_item_arr_ext.get_null_bitmap(
                recv_arr
            )

            # ----- list of item lengths -----------

            c_scatterv(
                send_arr_lens.ctypes,
                sendcounts.ctypes,
                displs.ctypes,
                recv_offsets_arr.ctypes,
                np.int32(n_loc),
                int32_typ_enum,
            )

            convert_len_arr_to_offset(recv_offsets_arr.ctypes, n_loc)

            # ----- items -----------

            c_scatterv(
                in_data_arr.ctypes,
                sendcounts_item.ctypes,
                displs_item.ctypes,
                recv_data_arr.ctypes,
                np.int32(n_loc_item),
                data_typ_enum,
            )

            # ----------- null bitmap -------------

            n_recv_bytes = (n_loc + 7) >> 3

            send_null_bitmap = get_scatter_null_bytes_buff(
                in_null_bitmap_arr.ctypes, sendcounts, sendcounts_nulls
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                sendcounts_nulls.ctypes,
                displs_nulls.ctypes,
                recv_null_bitmap_arr.ctypes,
                np.int32(n_recv_bytes),
                char_typ_enum,
            )

            return recv_arr

        return scatterv_array_item_impl

    if isinstance(data, (IntegerArrayType, DecimalArrayType)) or data in (
        boolean_array,
        datetime_date_array_type,
    ):
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        # these array need a data array and a null bitmap array to be initialized by
        # their init functions
        if isinstance(data, IntegerArrayType):
            init_func = bodo.libs.int_arr_ext.init_integer_array
        if isinstance(data, DecimalArrayType):
            precision = data.precision
            scale = data.scale
            init_func = numba.njit(no_cpython_wrapper=True)(
                lambda d, b: bodo.libs.decimal_arr_ext.init_decimal_array(
                    d, b, precision, scale
                )  # pragma: no cover
            )
        if data == boolean_array:
            init_func = bodo.libs.bool_arr_ext.init_bool_array
        if data == datetime_date_array_type:
            init_func = bodo.hiframes.datetime_date_ext.init_datetime_date_array

        def scatterv_impl_int_arr(data):  # pragma: no cover
            n_pes = bodo.libs.distributed_api.get_size()

            data_in = data._data
            null_bitmap = data._null_bitmap
            n_in = len(data_in)

            data_recv = _scatterv_np(data_in)

            n_all = bcast_scalar(n_in)
            n_recv_bytes = (len(data_recv) + 7) >> 3
            bitmap_recv = np.empty(n_recv_bytes, np.uint8)

            # compute send counts
            sendcounts = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                sendcounts[i] = get_node_portion(n_all, n_pes, i)

            # compute send counts for nulls
            sendcounts_nulls = np.empty(n_pes, np.int32)
            for i in range(n_pes):
                sendcounts_nulls[i] = (sendcounts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(sendcounts_nulls)

            send_null_bitmap = get_scatter_null_bytes_buff(
                null_bitmap.ctypes, sendcounts, sendcounts_nulls
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                sendcounts_nulls.ctypes,
                displs_nulls.ctypes,
                bitmap_recv.ctypes,
                np.int32(n_recv_bytes),
                char_typ_enum,
            )
            return init_func(data_recv, bitmap_recv)

        return scatterv_impl_int_arr

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):

        def impl_range_index(data):  # pragma: no cover
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

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):

        def impl_pd_index(data):  # pragma: no cover
            data_in = data._data
            name = data._name
            name = bcast_scalar(name)
            arr = bodo.libs.distributed_api.scatterv_impl(data_in)
            return bodo.utils.conversion.index_from_array(arr, name)

        return impl_pd_index

    # MultiIndex index
    if isinstance(data, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        # TODO: handle `levels` and `codes` when available
        def impl_multi_index(data):  # pragma: no cover
            all_data = bodo.libs.distributed_api.scatterv_impl(data._data)
            name = bcast_scalar(data._name)
            names = bcast_tuple(data._names)
            return bodo.hiframes.pd_multi_index_ext.init_multi_index(
                all_data, names, name
            )

        return impl_multi_index

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl_series(data):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # scatter data
            out_name = bcast_scalar(name)
            out_arr = bodo.libs.distributed_api.scatterv_impl(arr)
            out_index = bodo.libs.distributed_api.scatterv_impl(index)
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, out_name)

        return impl_series

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        data_args = ", ".join("g_data_{}".format(i) for i in range(n_cols))
        col_var = bodo.utils.transform.gen_const_tup(data.columns)

        func_text = "def impl_df(data):\n"
        for i in range(n_cols):
            func_text += "  data_{} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {})\n".format(
                i, i
            )
            func_text += "  g_data_{} = bodo.libs.distributed_api.scatterv_impl(data_{})\n".format(
                i, i
            )
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
        )
        func_text += "  g_index = bodo.libs.distributed_api.scatterv_impl(index)\n"
        func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), g_index, {})\n".format(
            data_args, col_var
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl_df = loc_vars["impl_df"]
        return impl_df

    if isinstance(data, CategoricalArray):

        def impl_cat(data):  # pragma: no cover
            codes = bodo.libs.distributed_api.scatterv_impl(data._codes)
            return bodo.hiframes.pd_categorical_ext.init_categorical_array(
                codes, data.dtype
            )

        return impl_cat

    if data == list_string_array_type:
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def scatterv_list_str_arr_impl(data):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_pes = bodo.libs.distributed_api.get_size()

            n_in = len(data)
            n_in_strs = data._num_total_strings

            n_all = bcast_scalar(n_in)

            # convert offsets to lengths of strings
            send_list_lens = np.empty(n_in, np.uint32)  # XXX offset type is uint32
            send_str_lens = np.empty(n_in_strs, np.uint32)

            for i in range(n_in):
                send_list_lens[i] = data._index_offsets[i + 1] - data._index_offsets[i]
            for j in range(n_in_strs):
                send_str_lens[j] = data._data_offsets[j + 1] - data._data_offsets[j]

            # ------- calculate buffer counts -------

            # compute send counts
            sendcounts = np.empty(n_pes, np.int32)
            for ii in range(n_pes):
                sendcounts[ii] = get_node_portion(n_all, n_pes, ii)

            # displacements
            displs = bodo.ir.join.calc_disp(sendcounts)

            # compute send counts for list lengths (number of strings)
            sendcounts_str = np.empty(n_pes, np.int32)
            if rank == 0:
                curr_str = 0
                for k in range(n_pes):
                    c = 0
                    for _ in range(sendcounts[k]):
                        c += send_list_lens[curr_str]
                        curr_str += 1
                    sendcounts_str[k] = c

            bcast(sendcounts_str)

            # displacements for characters
            displs_str = bodo.ir.join.calc_disp(sendcounts_str)

            # compute send counts for string lengths (number of characters)
            sendcounts_char = np.empty(n_pes, np.int32)
            if rank == 0:
                curr_str = 0
                for kk in range(n_pes):
                    c = 0
                    for _ in range(sendcounts_str[kk]):
                        c += send_str_lens[curr_str]
                        curr_str += 1
                    sendcounts_char[kk] = c

            bcast(sendcounts_char)

            # displacements for characters
            displs_char = bodo.ir.join.calc_disp(sendcounts_char)

            # compute send counts for nulls
            sendcounts_nulls = np.empty(n_pes, np.int32)
            for i_i in range(n_pes):
                sendcounts_nulls[i_i] = (sendcounts[i_i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(sendcounts_nulls)

            # alloc output array
            n_loc = sendcounts[rank]  # total number of elements on this PE
            n_loc_str = sendcounts_str[rank]
            n_loc_char = sendcounts_char[rank]
            recv_arr = pre_alloc_list_string_array(n_loc, n_loc_str, n_loc_char)

            # ----- list lengths -----------

            c_scatterv(
                send_list_lens.ctypes,
                sendcounts.ctypes,
                displs.ctypes,
                cptr_to_voidptr(recv_arr._index_offsets),
                np.int32(n_loc),
                int32_typ_enum,
            )

            convert_len_arr_to_offset(cptr_to_voidptr(recv_arr._index_offsets), n_loc)

            # ----- string lengths -----------

            c_scatterv(
                send_str_lens.ctypes,
                sendcounts_str.ctypes,
                displs_str.ctypes,
                cptr_to_voidptr(recv_arr._data_offsets),
                np.int32(n_loc_str),
                int32_typ_enum,
            )

            convert_len_arr_to_offset(
                cptr_to_voidptr(recv_arr._data_offsets), n_loc_str
            )

            # ----- string characters -----------

            c_scatterv(
                cptr_to_voidptr(data._data),
                sendcounts_char.ctypes,
                displs_char.ctypes,
                cptr_to_voidptr(recv_arr._data),
                np.int32(n_loc_char),
                char_typ_enum,
            )

            # ----------- null bitmap -------------

            n_recv_bytes = (n_loc + 7) >> 3

            send_null_bitmap = get_scatter_null_bytes_buff(
                cptr_to_voidptr(data._null_bitmap), sendcounts, sendcounts_nulls
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                sendcounts_nulls.ctypes,
                displs_nulls.ctypes,
                cptr_to_voidptr(recv_arr._null_bitmap),
                np.int32(n_recv_bytes),
                char_typ_enum,
            )
            dummy_use(data)  # needed?

            return recv_arr

        return scatterv_list_str_arr_impl

    # Tuple of data containers
    if isinstance(data, types.BaseTuple):
        func_text = "def impl_tuple(data):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(
                "bodo.libs.distributed_api.scatterv_impl(data[{}])".format(i)
                for i in range(len(data))
            ),
            "," if len(data) > 0 else "",
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl_tuple = loc_vars["impl_tuple"]
        return impl_tuple

    if data is types.none:  # pragma: no cover
        return lambda data: None

    raise BodoError("scatterv() not available for {}".format(data))  # pragma: no cover


@intrinsic
def cptr_to_voidptr(typingctx, cptr_tp=None):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], lir.IntType(8).as_pointer())

    return types.voidptr(cptr_tp), codegen


# TODO: test
# TODO: large BCast


def bcast(data):  # pragma: no cover
    return


@overload(bcast, no_unliteral=True)
def bcast_overload(data):
    if isinstance(data, types.Array):

        def bcast_impl(data):  # pragma: no cover
            typ_enum = get_type_enum(data)
            count = data.size
            assert count < INT_MAX
            c_bcast(data.ctypes, np.int32(count), typ_enum)
            return

        return bcast_impl

    if isinstance(data, DecimalArrayType):

        def bcast_decimal_arr(data):  # pragma: no cover
            count = data._data.size
            assert count < INT_MAX
            c_bcast(data._data.ctypes, np.int32(count), CTypeEnum.Int128.value)
            bcast(data._null_bitmap)
            return

        return bcast_decimal_arr

    if isinstance(data, IntegerArrayType) or data in (
        boolean_array,
        datetime_date_array_type,
    ):

        def bcast_impl_int_arr(data):  # pragma: no cover
            bcast(data._data)
            bcast(data._null_bitmap)
            return

        return bcast_impl_int_arr

    if data == string_array_type:
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def bcast_str_impl(data):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = len(data)
            n_all_chars = num_total_chars(data)
            assert n_loc < INT_MAX
            assert n_all_chars < INT_MAX

            offset_ptr = get_offset_ptr(data)
            data_ptr = get_data_ptr(data)
            null_bitmap_ptr = get_null_bitmap_ptr(data)
            n_bytes = (n_loc + 7) >> 3

            if rank == MPI_ROOT:
                send_arr_lens = np.empty(n_loc, np.uint32)  # XXX offset type is uint32
                for i in range(n_loc):
                    send_arr_lens[i] = bodo.libs.str_arr_ext.get_str_arr_item_length(
                        data, i
                    )

                c_bcast(send_arr_lens.ctypes, np.int32(n_loc), int32_typ_enum)
            else:
                c_bcast(offset_ptr, np.int32(n_loc), int32_typ_enum)

            c_bcast(data_ptr, np.int32(n_all_chars), char_typ_enum)
            c_bcast(null_bitmap_ptr, np.int32(n_bytes), char_typ_enum)
            if rank != MPI_ROOT:
                convert_len_arr_to_offset(offset_ptr, n_loc)

        return bcast_str_impl


# sendbuf, sendcount, dtype
c_bcast = types.ExternalFunction(
    "c_bcast", types.void(types.voidptr, types.int32, types.int32)
)


def bcast_scalar(val):  # pragma: no cover
    return val


# TODO: test
@overload(bcast_scalar, no_unliteral=True)
def bcast_scalar_overload(val):
    """broadcast for a scalar value
    """
    val = types.unliteral(val)
    # NOTE: scatterv() can call this with string on rank 0 and None on others, or an
    # Optional type
    assert (
        isinstance(val, (types.Integer, types.Float))
        or val == types.NPDatetime("ns")
        or val == bodo.string_type
        or val == types.none
    )

    if val == types.none:
        return lambda val: None

    if val == bodo.string_type:
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_str(val):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            if rank != MPI_ROOT:
                n_char = 0
                utf8_str = np.empty(0, np.uint8).ctypes
            else:
                utf8_str, n_char = bodo.libs.str_ext.unicode_to_utf8_and_len(val)
            n_char = bodo.libs.distributed_api.bcast_scalar(n_char)
            if rank != MPI_ROOT:
                utf8_str = np.empty(n_char, np.uint8).ctypes
            c_bcast(utf8_str, np.int32(n_char), char_typ_enum)
            return bodo.libs.str_arr_ext.decode_utf8(utf8_str, n_char)

        return impl_str

    # TODO: other types like boolean
    typ_val = numba_to_c_type(val)
    # TODO: fix np.full and refactor
    func_text = (
        "def bcast_scalar_impl(val):\n"
        "  send = np.empty(1, dtype)\n"
        "  send[0] = val\n"
        "  c_bcast(send.ctypes, np.int32(1), np.int32({}))\n"
        "  return send[0]\n"
    ).format(typ_val)

    dtype = numba.np.numpy_support.as_dtype(val)
    loc_vars = {}
    exec(
        func_text,
        {"bodo": bodo, "np": np, "c_bcast": c_bcast, "dtype": dtype},
        loc_vars,
    )
    bcast_scalar_impl = loc_vars["bcast_scalar_impl"]
    return bcast_scalar_impl


def bcast_tuple(val):  # pragma: no cover
    return val


@overload(bcast_tuple, no_unliteral=True)
def overload_bcast_tuple(val):
    """broadcast a tuple value (root == 0)
    calls bcast_scalar() on individual elements
    """
    assert isinstance(val, types.BaseTuple)
    n_elem = len(val)
    func_text = "def bcast_tuple_impl(val):\n"
    func_text += "  return ({}{})".format(
        ",".join("bcast_scalar(val[{}])".format(i) for i in range(n_elem)),
        "," if n_elem else "",
    )

    loc_vars = {}
    exec(
        func_text, {"bcast_scalar": bcast_scalar}, loc_vars,
    )
    bcast_tuple_impl = loc_vars["bcast_tuple_impl"]
    return bcast_tuple_impl


# if arr is string array, pre-allocate on non-root the same size as root
def prealloc_str_for_bcast(arr):  # pragma: no cover
    return arr


@overload(prealloc_str_for_bcast, no_unliteral=True)
def prealloc_str_for_bcast_overload(arr):
    if arr == string_array_type:

        def prealloc_impl(arr):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = bcast_scalar(len(arr))
            n_all_char = bcast_scalar(np.int64(num_total_chars(arr)))
            if rank != MPI_ROOT:
                arr = pre_alloc_string_array(n_loc, n_all_char)
            return arr

        return prealloc_impl

    return lambda arr: arr


def slice_getitem(arr, slice_index, arr_start, total_len, is_1D):  # pragma: no cover
    return arr[slice_index]


@overload(slice_getitem, no_unliteral=True)
def slice_getitem_overload(arr, slice_index, arr_start, total_len, is_1D):
    def getitem_impl(arr, slice_index, arr_start, total_len, is_1D):  # pragma: no cover
        # normalize slice
        slice_index = numba.cpython.unicode._normalize_slice(slice_index, total_len)
        start = slice_index.start
        step = slice_index.step

        # just broadcast from rank 0 in the common case A[:k] (for S.head())
        n_pes = bodo.libs.distributed_api.get_size()
        rank_0_portion = bodo.libs.distributed_api.get_node_portion(
            total_len, n_pes, np.int32(0)
        )
        if start == 0 and step == 1 and is_1D and rank_0_portion >= slice_index.stop:
            return slice_getitem_from_start(arr, slice_index)

        offset = (
            0
            if step == 1 or start > arr_start
            else (abs(step - (arr_start % step)) % step)
        )
        new_start = max(arr_start, slice_index.start) - arr_start + offset
        new_stop = max(slice_index.stop - arr_start, 0)
        my_arr = arr[new_start:new_stop:step]
        return bodo.libs.distributed_api.allgatherv(my_arr)

    return getitem_impl


# assuming start and step are None
def slice_getitem_from_start(arr, slice_index):  # pragma: no cover
    return arr[slice_index]


@overload(slice_getitem_from_start, no_unliteral=True)
def slice_getitem_from_start_overload(arr, slice_index):

    if arr == bodo.hiframes.datetime_date_ext.datetime_date_array_type:

        def getitem_datetime_date_impl(arr, slice_index):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            k = slice_index.stop
            A = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(k)
            if rank == 0:
                A = arr[:k]
            bodo.libs.distributed_api.bcast(A)
            return A

        return getitem_datetime_date_impl

    if isinstance(arr.dtype, Decimal128Type):
        precision = arr.dtype.precision
        scale = arr.dtype.scale

        def getitem_decimal_impl(arr, slice_index):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            k = slice_index.stop
            A = bodo.libs.decimal_arr_ext.alloc_decimal_array(k, precision, scale)
            if rank == 0:
                for i in range(k):
                    A._data[i] = arr._data[i]
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(arr._null_bitmap, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, bit)
            bodo.libs.distributed_api.bcast(A)
            return A

        return getitem_decimal_impl

    if arr == string_array_type:

        def getitem_str_impl(arr, slice_index):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            k = slice_index.stop
            # get total characters for allocation
            n_chars = np.uint64(0)
            if rank == 0:
                out_arr = arr[:k]
                n_chars = num_total_chars(out_arr)
            n_chars = bcast_scalar(n_chars)
            if rank != 0:
                out_arr = pre_alloc_string_array(k, n_chars)

            # actual communication
            bodo.libs.distributed_api.bcast(out_arr)
            return out_arr

        return getitem_str_impl

    arr_type = arr

    def getitem_impl(arr, slice_index):  # pragma: no cover
        rank = bodo.libs.distributed_api.get_rank()
        k = slice_index.stop
        out_arr = bodo.utils.utils.alloc_type(
            tuple_to_scalar((k,) + arr.shape[1:]), arr_type
        )
        if rank == 0:
            out_arr = arr[:k]
        bodo.libs.distributed_api.bcast(out_arr)
        return out_arr

    return getitem_impl


dummy_use = numba.njit(lambda a: None)


def int_getitem(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
    return arr[ind]


@overload(int_getitem, no_unliteral=True)
def int_getitem_overload(arr, ind, arr_start, total_len, is_1D):
    if arr == string_array_type:
        # TODO: other kinds, unicode
        kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def str_getitem_impl(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
            if ind >= total_len:
                raise IndexError("index out of bounds")

            # normalize negative slice
            ind = ind % total_len
            # TODO: avoid sending to root in case of 1D since position can be
            # calculated

            # send data to rank 0 and broadcast
            root = np.int32(0)
            size_tag = np.int32(10)
            tag = np.int32(11)
            send_size = np.zeros(1, np.int64)
            send_val = ""
            if arr_start <= ind < (arr_start + len(arr)):
                send_val = arr[ind - arr_start]
                send_size[0] = len(send_val)
                isend(send_size, np.int32(1), root, size_tag, True)
                isend(send_val._data, np.int32(len(send_val)), root, tag, True)

            rank = bodo.libs.distributed_api.get_rank()
            val = ""
            l = 0
            if rank == root:
                l = recv(np.int64, ANY_SOURCE, size_tag)
                val = numba.cpython.unicode._empty_string(kind, l, 1)
                _recv(val._data, np.int32(l), char_typ_enum, ANY_SOURCE, tag)

            dummy_use(send_size)
            dummy_use(send_val)
            l = bcast_scalar(l)
            if rank != root:
                val = numba.cpython.unicode._empty_string(kind, l, 1)
            # TODO: unicode fix?
            c_bcast(val._data, np.int32(l), char_typ_enum)
            return val

        return str_getitem_impl

    def getitem_impl(arr, ind, arr_start, total_len, is_1D):  # pragma: no cover
        # TODO: multi-dim array support

        if ind >= total_len:
            raise IndexError("index out of bounds")

        # normalize negative slice
        ind = ind % total_len
        # TODO: avoid sending to root in case of 1D since position can be
        # calculated

        # send data to rank 0 and broadcast
        root = np.int32(0)
        tag = np.int32(11)
        send_arr = np.zeros(1, arr.dtype)
        if arr_start <= ind < (arr_start + len(arr)):
            data = arr[ind - arr_start]
            send_arr = np.full(1, data)
            isend(send_arr, np.int32(1), root, tag, True)

        rank = bodo.libs.distributed_api.get_rank()
        val = np.zeros(1, arr.dtype)[0]  # TODO: better way to get zero of type
        if rank == root:
            val = recv(arr.dtype, ANY_SOURCE, tag)

        dummy_use(send_arr)
        val = bcast_scalar(val)
        return val

    return getitem_impl


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

    if isinstance(send_data, (IntegerArrayType, DecimalArrayType)) or send_data in (
        boolean_array,
        datetime_date_array_type,
    ):
        return lambda send_data, out_data, send_counts, recv_counts, send_disp, recv_disp: c_alltoallv(
            send_data._data.ctypes,
            out_data._data.ctypes,
            send_counts.ctypes,
            recv_counts.ctypes,
            send_disp.ctypes,
            recv_disp.ctypes,
            typ_enum,
        )

    return lambda send_data, out_data, send_counts, recv_counts, send_disp, recv_disp: c_alltoallv(
        send_data.ctypes,
        out_data.ctypes,
        send_counts.ctypes,
        recv_counts.ctypes,
        send_disp.ctypes,
        recv_disp.ctypes,
        typ_enum,
    )


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


@numba.generated_jit(nopython=True)
def dist_cumsum(in_arr, out_arr):
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


def dist_cumprod(arr):  # pragma: no cover
    """dummy to implement cumprod"""
    return arr


def dist_setitem(arr, index, val):  # pragma: no cover
    return 0


_allgather = types.ExternalFunction(
    "allgather", types.void(types.voidptr, types.int32, types.voidptr, types.int32)
)


@numba.njit
def allgather(arr, val):  # pragma: no cover
    type_enum = get_type_enum(arr)
    _allgather(arr.ctypes, 1, value_to_ptr(val), type_enum)


def dist_return(A):  # pragma: no cover
    return A


def threaded_return(A):  # pragma: no cover
    return A


def rebalance_array(A):
    return A


@numba.njit
def rebalance_array_parallel(in_arr, count):  # pragma: no cover
    n_pes = bodo.libs.distributed_api.get_size()
    my_rank = bodo.libs.distributed_api.get_rank()
    out_arr = np.empty((count,) + in_arr.shape[1:], in_arr.dtype)
    # copy old data
    old_len = len(in_arr)
    out_ind = 0
    for i in range(min(old_len, count)):
        out_arr[i] = in_arr[i]
        out_ind += 1
    # get diff data for all procs
    my_diff = old_len - count
    all_diffs = np.empty(n_pes, np.int64)
    bodo.libs.distributed_api.allgather(all_diffs, my_diff)
    # alloc comm requests
    comm_req_ind = 0
    comm_reqs = bodo.libs.distributed_api.comm_req_alloc(n_pes)
    # for each potential receiver
    for i in range(n_pes):
        # if receiver
        if all_diffs[i] < 0:
            # for each potential sender
            for j in range(n_pes):
                # if sender
                if all_diffs[j] > 0:
                    send_size = min(all_diffs[j], -all_diffs[i])
                    # if I'm receiver
                    if my_rank == i:
                        buff = out_arr[out_ind : (out_ind + send_size)]
                        comm_reqs[comm_req_ind] = bodo.libs.distributed_api.irecv(
                            buff, np.int32(buff.size), np.int32(j), np.int32(9)
                        )
                        comm_req_ind += 1
                        out_ind += send_size
                    # if I'm sender
                    if my_rank == j:
                        buff = np.ascontiguousarray(
                            in_arr[out_ind : (out_ind + send_size)]
                        )
                        comm_reqs[comm_req_ind] = bodo.libs.distributed_api.isend(
                            buff, np.int32(buff.size), np.int32(i), np.int32(9)
                        )
                        comm_req_ind += 1
                        out_ind += send_size
                    # update sender and receivers remaining counts
                    all_diffs[i] += send_size
                    all_diffs[j] -= send_size
                    # if receiver is done, stop sender search
                    if all_diffs[i] == 0:
                        break
    bodo.libs.distributed_api.waitall(np.int32(comm_req_ind), comm_reqs)
    bodo.libs.distributed_api.comm_req_dealloc(comm_reqs)
    return out_arr


# dummy function to set a distributed array without changing the index in distributed
# pass
@numba.njit
def set_arr_local(arr, ind, val):
    arr[ind] = val


# dummy function to specify local allocation size, to enable bypassing distributed
# transformations
@numba.njit
def local_alloc_size(n, in_arr):
    return n


@overload(rebalance_array, no_unliteral=True)
def dist_return_overload(A):
    return dist_return


# TODO: move other funcs to old API?
@infer_global(threaded_return)
@infer_global(dist_return)
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


wait = types.ExternalFunction("dist_wait", types.void(mpi_req_numba_type, types.bool_))


# @infer_global(dist_setitem)
# class DistSetitem(AbstractTemplate):
#     def generic(self, args, kws):
#         assert not kws
#         assert len(args)==5
#         return signature(types.int32, *unliteral_all(args))


class ReqArrayType(types.Type):
    def __init__(self):
        super(ReqArrayType, self).__init__(name="ReqArrayType()")


req_array_type = ReqArrayType()
register_model(ReqArrayType)(models.OpaqueModel)
waitall = types.ExternalFunction(
    "dist_waitall", types.void(types.int32, req_array_type)
)
comm_req_alloc = types.ExternalFunction("comm_req_alloc", req_array_type(types.int32))
comm_req_dealloc = types.ExternalFunction(
    "comm_req_dealloc", types.void(req_array_type)
)
req_array_setitem = types.ExternalFunction(
    "req_array_setitem", types.void(req_array_type, types.int64, mpi_req_numba_type)
)


@overload(operator.setitem, no_unliteral=True)
def overload_req_arr_setitem(A, idx, val):
    if A == req_array_type:
        assert val == mpi_req_numba_type
        return lambda A, idx, val: req_array_setitem(A, idx, val)


# find overlapping range of an input range (start:stop) and a chunk range
# (chunk_start:chunk_start+chunk_count). Inputs are assumed positive.
# output is set to empty range of local range goes out of bounds
@numba.njit
def _get_local_range(start, stop, chunk_start, chunk_count):  # pragma: no cover
    assert start >= 0 and stop > 0
    new_start = max(start, chunk_start)
    new_stop = min(stop, chunk_start + chunk_count)
    loc_start = new_start - chunk_start
    loc_stop = new_stop - chunk_start
    if loc_start < 0 or loc_stop < 0:
        loc_start = 1
        loc_stop = 0
    return loc_start, loc_stop


@numba.njit
def _set_if_in_range(A, val, index, chunk_start, chunk_count):  # pragma: no cover
    if index >= chunk_start and index < chunk_start + chunk_count:
        A[index - chunk_start] = val


@numba.njit
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
)

oneD_reshape_shuffle = types.ExternalFunction("oneD_reshape_shuffle", sig)


@numba.njit(no_cpython_wrapper=True, cache=True)
def dist_oneD_reshape_shuffle(lhs, in_arr, new_dim0_global_len):  # pragma: no cover
    """shuffles the data for ndarray reshape to fill the output array properly
    """
    c_in_arr = np.ascontiguousarray(in_arr)
    in_lower_dims_size = get_tuple_prod(c_in_arr.shape[1:])
    out_lower_dims_size = get_tuple_prod(lhs.shape[1:])

    dtype_size = bodo.io.np_io.get_dtype_size(in_arr.dtype)
    oneD_reshape_shuffle(
        lhs.ctypes,
        c_in_arr.ctypes,
        new_dim0_global_len,
        len(in_arr),
        dtype_size * out_lower_dims_size,
        dtype_size * in_lower_dims_size,
    )


permutation_int = types.ExternalFunction(
    "permutation_int", types.void(types.voidptr, types.intp)
)


@numba.njit
def dist_permutation_int(lhs, n):  # pragma: no cover
    permutation_int(lhs.ctypes, n)


permutation_array_index = types.ExternalFunction(
    "permutation_array_index",
    types.void(
        types.voidptr, types.intp, types.intp, types.voidptr, types.voidptr, types.intp
    ),
)


@numba.njit
def dist_permutation_array_index(
    lhs, lhs_len, dtype_size, rhs, p, p_len
):  # pragma: no cover
    c_rhs = np.ascontiguousarray(rhs)
    lower_dims_size = get_tuple_prod(c_rhs.shape[1:])
    elem_size = dtype_size * lower_dims_size
    permutation_array_index(
        lhs.ctypes, lhs_len, elem_size, c_rhs.ctypes, p.ctypes, p_len
    )


########### finalize MPI & s3_reader, disconnect hdfs when exiting ############


from bodo.io import s3_reader
from bodo.io import hdfs_reader

ll.add_symbol("finalize", hdist.finalize)
finalize = types.ExternalFunction("finalize", types.int32())

ll.add_symbol("finalize_s3", s3_reader.finalize_s3)
finalize_s3 = types.ExternalFunction("finalize_s3", types.int32())

ll.add_symbol("disconnect_hdfs", hdfs_reader.disconnect_hdfs)
disconnect_hdfs = types.ExternalFunction("disconnect_hdfs", types.int32())


@numba.njit
def call_finalize():  # pragma: no cover
    finalize()
    finalize_s3()
    disconnect_hdfs()


def flush_stdout():
    # using a function since pytest throws an error sometimes
    # if flush function is passed directly to atexit
    if not sys.stdout.closed:
        sys.stdout.flush()


atexit.register(call_finalize)
# flush output before finalize
atexit.register(flush_stdout)
