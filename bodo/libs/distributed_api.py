# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
from enum import Enum
import math
import time
import numpy as np

import numba
from numba import types
from numba.typing.templates import infer_global, AbstractTemplate, infer
from numba.typing import signature
from numba.extending import models, register_model, intrinsic, overload

import bodo
from bodo.libs.str_arr_ext import (
    string_array_type,
    num_total_chars,
    StringArray,
    pre_alloc_string_array,
    get_offset_ptr,
    get_null_bitmap_ptr,
    get_data_ptr,
    convert_len_arr_to_offset,
    getitem_str_bitmap,
    setitem_str_bitmap,
    set_bit_to,
)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.hiframes.pd_categorical_ext import CategoricalArray
from bodo.utils.utils import (
    debug_prints,
    empty_like_type,
    _numba_to_c_type_map,
    unliteral_all,
)
from llvmlite import ir as lir
from bodo.libs import hdist

import llvmlite.binding as ll

ll.add_symbol("c_get_rank", hdist.hpat_dist_get_rank)
ll.add_symbol("c_get_size", hdist.hpat_dist_get_size)
ll.add_symbol("c_barrier", hdist.hpat_barrier)
ll.add_symbol("c_alltoall", hdist.c_alltoall)
ll.add_symbol("c_gather_scalar", hdist.c_gather_scalar)
ll.add_symbol("c_gatherv", hdist.c_gatherv)
ll.add_symbol("c_allgatherv", hdist.c_allgatherv)
ll.add_symbol("c_bcast", hdist.c_bcast)
ll.add_symbol("c_recv", hdist.hpat_dist_recv)
ll.add_symbol("c_send", hdist.hpat_dist_send)


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
def get_rank():
    """wrapper for getting process rank (MPI rank currently)"""
    return _get_rank()


@numba.njit
def get_size():
    """wrapper for getting number of processes (MPI COMM size currently)"""
    return _get_size()


@numba.njit
def barrier():
    """wrapper for barrier (MPI barrier currently)"""
    return _barrier()


@numba.generated_jit(nopython=True)
def get_type_enum(arr):
    arr = arr.instance_type if isinstance(arr, types.TypeRef) else arr
    dtype = arr.dtype
    if isinstance(dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
        dtype = bodo.hiframes.pd_categorical_ext.get_categories_int_type(dtype)

    typ_val = _numba_to_c_type_map[dtype]
    return lambda arr: np.int32(typ_val)


INT_MAX = np.iinfo(np.int32).max

_send = types.ExternalFunction(
    "c_send",
    types.void(types.voidptr, types.int32, types.int32, types.int32, types.int32),
)


@numba.njit
def send(val, rank, tag):
    # dummy array for val
    send_arr = np.full(1, val)
    type_enum = get_type_enum(send_arr)
    _send(send_arr.ctypes, 1, type_enum, rank, tag)


_recv = types.ExternalFunction(
    "c_recv",
    types.void(types.voidptr, types.int32, types.int32, types.int32, types.int32),
)


@numba.njit
def recv(dtype, rank, tag):
    # dummy array for val
    recv_arr = np.empty(1, dtype)
    type_enum = get_type_enum(recv_arr)
    _recv(recv_arr.ctypes, 1, type_enum, rank, tag)
    return recv_arr[0]


_alltoall = types.ExternalFunction(
    "c_alltoall", types.void(types.voidptr, types.voidptr, types.int32, types.int32)
)


@numba.njit
def alltoall(send_arr, recv_arr, count):
    # TODO: handle int64 counts
    assert count < INT_MAX
    type_enum = get_type_enum(send_arr)
    _alltoall(send_arr.ctypes, recv_arr.ctypes, np.int32(count), type_enum)


@numba.generated_jit
def gather_scalar(data, allgather=False):
    data = types.unliteral(data)
    typ_val = _numba_to_c_type_map[data]
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


# from GetBit() in Arrow
@numba.njit
def get_bit(bits, i):
    return (bits[i >> 3] >> (i & 0x07)) & 1


@numba.njit
def copy_gathered_null_bytes(
    null_bitmap_ptr, tmp_null_bytes, recv_counts_nulls, recv_counts
):
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

        def impl_cat(data, allgather=False):
            int_arr = bodo.hiframes.pd_categorical_ext.cat_array_to_int(data)
            return bodo.hiframes.pd_categorical_ext.set_cat_dtype(
                bodo.gatherv(int_arr, allgather), data
            )

        return impl_cat

    if isinstance(data, types.Array):
        typ_val = _numba_to_c_type_map[data.dtype]

        def gatherv_impl(data, allgather=False):
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
            #  print(rank, n_loc, n_total, recv_counts, displs)
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
        int32_typ_enum = np.int32(_numba_to_c_type_map[types.int32])
        char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])

        def gatherv_str_arr_impl(data, allgather=False):
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
            all_data = StringArray()  # dummy arrays on non-root PEs
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

    if isinstance(data, IntegerArrayType) or data == boolean_array:
        typ_val = _numba_to_c_type_map[data.dtype]
        char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])

        def gatherv_impl_int_arr(data, allgather=False):
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

        def impl(data, allgather=False):
            # get data and index arrays
            arr = bodo.hiframes.api.get_series_data(data)
            index = bodo.hiframes.api.get_series_index(data)
            name = bodo.hiframes.api.get_series_name(data)
            # gather data
            out_arr = bodo.libs.distributed_api.gatherv(arr, allgather)
            out_index = bodo.gatherv(index, allgather)
            # create output Series
            return bodo.hiframes.api.init_series(out_arr, out_index, name)

        return impl

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):

        def impl_range_index(data, allgather=False):
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
            return bodo.hiframes.pd_index_ext.init_range_index(0, n, 1)

        return impl_range_index

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):

        def impl_pd_index(data, allgather=False):
            arr = bodo.libs.distributed_api.gatherv(data._data, allgather)
            return bodo.utils.conversion.index_from_array(arr, data._name)

        return impl_pd_index

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        data_args = ", ".join("g_data_{}".format(i) for i in range(n_cols))
        col_var = "bodo.utils.typing.add_consts_to_type([{0}], {0})".format(
            ", ".join("'{}'".format(c) for c in data.columns)
        )

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

    if data is types.none:
        return lambda data: None

    raise NotImplementedError("gatherv() not available for {}".format(data))


@numba.generated_jit(nopython=True)
def allgatherv(data):
    return lambda data: gatherv(data, True)


# TODO: test
# TODO: large BCast


def bcast(data):  # pragma: no cover
    return


@overload(bcast)
def bcast_overload(data):
    if isinstance(data, types.Array):

        def bcast_impl(data):
            typ_enum = get_type_enum(data)
            count = data.size
            assert count < INT_MAX
            c_bcast(data.ctypes, np.int32(count), typ_enum)
            return

        return bcast_impl

    if isinstance(data, IntegerArrayType) or data == boolean_array:

        def bcast_impl_int_arr(data):
            bcast(data._data)
            bcast(data._null_bitmap)
            return

        return bcast_impl_int_arr

    if data == string_array_type:
        int32_typ_enum = np.int32(_numba_to_c_type_map[types.int32])
        char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])

        def bcast_str_impl(data):
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
@overload(bcast_scalar)
def bcast_scalar_overload(val):
    assert isinstance(val, (types.Integer, types.Float)) or val == types.NPDatetime(
        "ns"
    )
    # TODO: other types like boolean
    typ_val = _numba_to_c_type_map[val]
    # TODO: fix np.full and refactor
    func_text = (
        "def bcast_scalar_impl(val):\n"
        "  send = np.empty(1, dtype)\n"
        "  send[0] = val\n"
        "  c_bcast(send.ctypes, np.int32(1), np.int32({}))\n"
        "  return send[0]\n"
    ).format(typ_val)

    dtype = numba.numpy_support.as_dtype(val)
    loc_vars = {}
    exec(
        func_text,
        {"bodo": bodo, "np": np, "c_bcast": c_bcast, "dtype": dtype},
        loc_vars,
    )
    bcast_scalar_impl = loc_vars["bcast_scalar_impl"]
    return bcast_scalar_impl


# if arr is string array, pre-allocate on non-root the same size as root
def prealloc_str_for_bcast(arr):
    return arr


@overload(prealloc_str_for_bcast)
def prealloc_str_for_bcast_overload(arr):
    if arr == string_array_type:

        def prealloc_impl(arr):
            rank = bodo.libs.distributed_api.get_rank()
            n_loc = bcast_scalar(len(arr))
            n_all_char = bcast_scalar(np.int64(num_total_chars(arr)))
            if rank != MPI_ROOT:
                arr = pre_alloc_string_array(n_loc, n_all_char)
            return arr

        return prealloc_impl

    return lambda arr: arr


def slice_getitem(arr, slice_index, arr_start, total_len, is_1D):
    return arr[slice_index]


@overload(slice_getitem)
def slice_getitem_overload(arr, slice_index, arr_start, total_len, is_1D):
    def getitem_impl(arr, slice_index, arr_start, total_len, is_1D):
        # normalize slice
        slice_index = numba.unicode._normalize_slice(slice_index, total_len)
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
def slice_getitem_from_start(arr, slice_index):
    return arr[slice_index]


@overload(slice_getitem_from_start)
def slice_getitem_from_start_overload(arr, slice_index):
    if arr == string_array_type:

        def getitem_str_impl(arr, slice_index):
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

    def getitem_impl(arr, slice_index):
        rank = bodo.libs.distributed_api.get_rank()
        k = slice_index.stop
        out_arr = bodo.utils.utils.alloc_type((k,) + arr.shape[1:], arr_type)
        if rank == 0:
            out_arr = arr[:k]
        bodo.libs.distributed_api.bcast(out_arr)
        return out_arr

    return getitem_impl


dummy_use = numba.njit(lambda a: None)


def int_getitem(arr, ind, arr_start, total_len, is_1D):
    return arr[ind]


@overload(int_getitem)
def int_getitem_overload(arr, ind, arr_start, total_len, is_1D):
    if arr == string_array_type:
        # TODO: fix and test. It fails with weird error on mpich 3.2.1:
        # MPIDI_CH3U_Buffer_copy(64): Message truncated; 4 bytes received but
        # buffer size is 4
        # TODO: other kinds, unicode
        kind = numba.unicode.PY_UNICODE_1BYTE_KIND
        char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])

        def str_getitem_impl(arr, ind, arr_start, total_len, is_1D):
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
                val = numba.unicode._empty_string(kind, l, 1)
                _recv(val._data, np.int32(l), char_typ_enum, ANY_SOURCE, tag)

            dummy_use(send_size)
            dummy_use(send_val)
            l = bcast_scalar(l)
            if rank != root:
                val = numba.unicode._empty_string(kind, l, 1)
            # TODO: unicode fix?
            c_bcast(val._data, np.int32(l), char_typ_enum)
            return val

        return str_getitem_impl

    def getitem_impl(arr, ind, arr_start, total_len, is_1D):
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

    if isinstance(send_data, IntegerArrayType) or send_data == boolean_array:
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


@overload(alltoallv_tup)
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
def get_start_count(n):
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    start = bodo.libs.distributed_api.get_start(n, n_pes, rank)
    count = bodo.libs.distributed_api.get_node_portion(n, n_pes, rank)
    return start, count


def remove_dist_calls(rhs, lives, call_list):
    if call_list == ["dist_reduce", "distributed_api", "libs", bodo]:
        return True
    if call_list == [dist_reduce]:
        return True
    return False


numba.ir_utils.remove_call_handlers.append(remove_dist_calls)


@numba.njit
def get_start(total_size, pes, rank):
    """get start index in 1D distribution"""
    chunk = math.ceil(total_size / pes)
    return min(total_size, rank * chunk)


@numba.njit
def get_end(total_size, pes, rank):
    """get end point of range for parfor division"""
    chunk = math.ceil(total_size / pes)
    return min(total_size, (rank + 1) * chunk)


@numba.njit
def get_node_portion(total_size, pes, rank):
    """get portion of size for alloc division"""
    chunk = math.ceil(total_size / pes)
    return min(total_size, (rank + 1) * chunk) - min(total_size, rank * chunk)


def dist_reduce(value, op):  # pragma: no cover
    """dummy to implement simple reductions"""
    return value


def dist_arr_reduce(arr):  # pragma: no cover
    """dummy to implement array reductions"""
    return -1


def dist_cumsum(arr):  # pragma: no cover
    """dummy to implement cumsum"""
    return arr


def dist_cumprod(arr):  # pragma: no cover
    """dummy to implement cumprod"""
    return arr


def dist_exscan(value):  # pragma: no cover
    """dummy to implement simple exscan"""
    return value


def dist_setitem(arr, index, val):  # pragma: no cover
    return 0


def allgather(arr, val):  # pragma: no cover
    arr[0] = val


def dist_time():  # pragma: no cover
    return time.time()


def dist_return(A):  # pragma: no cover
    return A


def threaded_return(A):  # pragma: no cover
    return A


def rebalance_array(A):
    return A


def rebalance_array_parallel(A):
    return A


@overload(rebalance_array)
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
def parallel_print(s):
    print(s)


@numba.njit
def single_print(*args):
    if bodo.libs.distributed_api.get_rank() == 0:
        print(*args)


def irecv():  # pragma: no cover
    return 0


def isend():  # pragma: no cover
    return 0


def wait():  # pragma: no cover
    return 0


def waitall():  # pragma: no cover
    return 0


@infer_global(allgather)
class DistAllgather(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2  # array and val
        return signature(types.none, *unliteral_all(args))


@infer_global(rebalance_array_parallel)
class DistRebalanceParallel(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2  # array and count
        return signature(args[0], *unliteral_all(args))


@infer_global(dist_reduce)
class DistReduce(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2  # value and reduce_op
        return signature(args[0], *unliteral_all(args))


@infer_global(dist_exscan)
class DistExscan(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(args[0], *unliteral_all(args))


@infer_global(dist_arr_reduce)
class DistArrReduce(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2  # value and reduce_op
        return signature(types.int32, *unliteral_all(args))


@infer_global(time.time)
class DistTime(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 0
        return signature(types.float64, *unliteral_all(args))


@infer_global(dist_time)
class DistDistTime(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 0
        return signature(types.float64, *unliteral_all(args))


@infer_global(dist_cumsum)
@infer_global(dist_cumprod)
class DistCumsumprod(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.int32, *unliteral_all(args))


@infer_global(irecv)
@infer_global(isend)
class DistIRecv(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) in [4, 5]
        return signature(mpi_req_numba_type, *unliteral_all(args))


@infer_global(wait)
class DistWait(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(types.int32, *unliteral_all(args))


@infer_global(waitall)
class DistWaitAll(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2 and args == (types.int32, req_array_type)
        return signature(types.none, *unliteral_all(args))


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


def comm_req_alloc():
    return 0


def comm_req_dealloc():
    return 0


@infer_global(comm_req_alloc)
class DistCommReqAlloc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1 and args[0] == types.int32
        return signature(req_array_type, *unliteral_all(args))


@infer_global(comm_req_dealloc)
class DistCommReqDeAlloc(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1 and args[0] == req_array_type
        return signature(types.none, *unliteral_all(args))


@infer_global(operator.setitem)
class SetItemReqArray(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [ary, idx, val] = args
        if (
            isinstance(ary, ReqArrayType)
            and idx == types.intp
            and val == mpi_req_numba_type
        ):
            return signature(types.none, *unliteral_all(args))
