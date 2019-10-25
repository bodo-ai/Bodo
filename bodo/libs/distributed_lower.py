# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import numpy as np
import time
import sys
import atexit

from numba import types, cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
from numba.extending import overload
import numba.targets.arrayobj
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed
from numba.typing.builtins import IndexValueType

import bodo
from bodo.libs import distributed_api
from bodo.utils.utils import _numba_to_c_type_map
from bodo.libs.distributed_api import mpi_req_numba_type, ReqArrayType, req_array_type
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

# get size dynamically from C code
mpi_req_llvm_type = lir.IntType(8 * hdist.mpi_req_num_bytes)


@lower_builtin(distributed_api.comm_req_dealloc, req_array_type)
def lower_dist_comm_req_dealloc(context, builder, sig, args):
    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="comm_req_dealloc")
    builder.call(fn, args)
    return context.get_dummy_value()


@lower_builtin(operator.setitem, ReqArrayType, types.intp, mpi_req_numba_type)
def setitem_req_array(context, builder, sig, args):
    fnty = lir.FunctionType(
        lir.VoidType(),
        [lir.IntType(8).as_pointer(), lir.IntType(64), mpi_req_llvm_type],
    )
    fn = builder.module.get_or_insert_function(fnty, name="req_array_setitem")
    return builder.call(fn, args)


# @lower_builtin(distributed_api.dist_setitem, types.Array, types.Any, types.Any,
#     types.intp, types.intp)
# def dist_setitem_array(context, builder, sig, args):
#     """add check for access to be in processor bounds of the array"""
#     # TODO: replace array shape if array is small
#     #  (processor chuncks smaller than setitem range causes index normalization)
#     # remove start and count args to call regular get_item_pointer2
#     count = args.pop()
#     start = args.pop()
#     sig.args = tuple([sig.args[0], sig.args[1], sig.args[2]])
#     regular_get_item_pointer2 = cgutils.get_item_pointer2
#     # add bounds check for distributed access,
#     # return a dummy pointer if out of bounds
#     def dist_get_item_pointer2(builder, data, shape, strides, layout, inds,
#                       wraparound=False):
#         # get local index or -1 if out of bounds
#         fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(64), lir.IntType(64), lir.IntType(64)])
#         fn = builder.module.get_or_insert_function(fnty, name="dist_get_item_pointer")
#         first_ind = builder.call(fn, [inds[0], start, count])
#         inds = tuple([first_ind, *inds[1:]])
#         # regular local pointer with new indices
#         in_ptr = regular_get_item_pointer2(builder, data, shape, strides, layout, inds, wraparound)
#         ret_ptr = cgutils.alloca_once(builder, in_ptr.type)
#         builder.store(in_ptr, ret_ptr)
#         not_inbound = builder.icmp_signed('==', first_ind, lir.Constant(lir.IntType(64), -1))
#         # get dummy pointer
#         dummy_fnty  = lir.FunctionType(lir.IntType(8).as_pointer(), [])
#         dummy_fn = builder.module.get_or_insert_function(dummy_fnty, name="get_dummy_ptr")
#         dummy_ptr = builder.bitcast(builder.call(dummy_fn, []), in_ptr.type)
#         with builder.if_then(not_inbound, likely=True):
#             builder.store(dummy_ptr, ret_ptr)
#         return builder.load(ret_ptr)
#
#     # replace inner array access call for setitem generation
#     cgutils.get_item_pointer2 = dist_get_item_pointer2
#     numba.targets.arrayobj.setitem_array(context, builder, sig, args)
#     cgutils.get_item_pointer2 = regular_get_item_pointer2
#     return lir.Constant(lir.IntType(32), 0)

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
    if distributed_api.get_rank() == 0:
        return old_val
    return new_val


def get_tuple_prod(t):
    return np.prod(t)


@overload(get_tuple_prod)
def get_tuple_prod_overload(t):
    # handle empty tuple seperately since empty getiter doesn't work
    if t == numba.types.containers.Tuple(()):
        return lambda t: 1

    def get_tuple_prod_impl(t):
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


@numba.njit
def dist_oneD_reshape_shuffle(
    lhs, in_arr, new_0dim_global_len, old_0dim_global_len, dtype_size
):  # pragma: no cover
    c_in_arr = np.ascontiguousarray(in_arr)
    in_lower_dims_size = get_tuple_prod(c_in_arr.shape[1:])
    out_lower_dims_size = get_tuple_prod(lhs.shape[1:])
    # print(c_in_arr)
    # print(new_0dim_global_len, old_0dim_global_len, out_lower_dims_size, in_lower_dims_size)
    oneD_reshape_shuffle(
        lhs.ctypes,
        c_in_arr.ctypes,
        new_0dim_global_len,
        old_0dim_global_len,
        dtype_size * out_lower_dims_size,
        dtype_size * in_lower_dims_size,
    )
    # print(in_arr)


permutation_int = types.ExternalFunction(
    "permutation_int", types.void(types.voidptr, types.intp)
)


@numba.njit
def dist_permutation_int(lhs, n):
    permutation_int(lhs.ctypes, n)


permutation_array_index = types.ExternalFunction(
    "permutation_array_index",
    types.void(
        types.voidptr, types.intp, types.intp, types.voidptr, types.voidptr, types.intp
    ),
)


@numba.njit
def dist_permutation_array_index(lhs, lhs_len, dtype_size, rhs, p, p_len):
    c_rhs = np.ascontiguousarray(rhs)
    lower_dims_size = get_tuple_prod(c_rhs.shape[1:])
    elem_size = dtype_size * lower_dims_size
    permutation_array_index(
        lhs.ctypes, lhs_len, elem_size, c_rhs.ctypes, p.ctypes, p_len
    )


########### finalize MPI when exiting ####################


def finalize():
    return 0


from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature


@infer_global(finalize)
class FinalizeInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 0
        return signature(types.int32, *args)


ll.add_symbol("finalize", hdist.finalize)


@lower_builtin(finalize)
def lower_hpat_finalize(context, builder, sig, args):
    fnty = lir.FunctionType(lir.IntType(32), [])
    fn = builder.module.get_or_insert_function(fnty, name="finalize")
    return builder.call(fn, args)


@numba.njit
def call_finalize():
    finalize()


def flush_stdout():
    # using a function since pytest throws an error sometimes
    # if flush function is passed directly to atexit
    if not sys.stdout.closed:
        sys.stdout.flush()


atexit.register(call_finalize)
# flush output before finalize
atexit.register(flush_stdout)
