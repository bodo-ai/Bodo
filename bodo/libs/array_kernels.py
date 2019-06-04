"""
Implements array kernels such as median and quantile.
"""
import pandas as pd
import numpy as np

import numba
from numba.extending import overload
from numba import types, cgutils
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import lower_builtin, impl_ret_untracked
from numba.targets.arrayobj import make_array

import bodo
from bodo.utils.utils import _numba_to_c_type_map, unliteral_all

import llvmlite.llvmpy.core as lc
from llvmlite import ir as lir
from bodo.libs import quantile_alg
import llvmlite.binding as ll
ll.add_symbol('quantile_parallel', quantile_alg.quantile_parallel)
ll.add_symbol('nth_sequential', quantile_alg.nth_sequential)
ll.add_symbol('nth_parallel', quantile_alg.nth_parallel)


nth_sequential = types.ExternalFunction("nth_sequential",
    types.void(types.voidptr, types.voidptr, types.int64, types.int64, types.int32))

nth_parallel = types.ExternalFunction("nth_parallel",
    types.void(types.voidptr, types.voidptr, types.int64, types.int64, types.int32))

MPI_ROOT = 0
sum_op = bodo.libs.distributed_api.Reduce_Type.Sum.value


################################ median ####################################


@numba.njit
def nth_element(arr, k, parallel=False):
    res = np.empty(1, arr.dtype)
    type_enum = bodo.libs.distributed_api.get_type_enum(arr)
    if parallel:
        nth_parallel(res.ctypes, arr.ctypes, len(arr), k, type_enum)
    else:
        nth_sequential(res.ctypes, arr.ctypes, len(arr), k, type_enum)
    return res[0]


@numba.njit
def median(arr, parallel=False):
    # similar to numpy/lib/function_base.py:_median
    # TODO: check return types, e.g. float32 -> float32
    n = len(arr)
    if parallel:
        n = bodo.libs.distributed_api.dist_reduce(n, np.int32(sum_op))
    k = n // 2

    # odd length case
    if n % 2 == 1:
        return nth_element(arr, k, parallel)

    v1 = nth_element(arr, k-1, parallel)
    v2 = nth_element(arr, k, parallel)
    return (v1 + v2) / 2


################################ quantile ####################################


def quantile(A, q):  # pragma: no cover
    return 0


def quantile_parallel(A, q):  # pragma: no cover
    return 0


@infer_global(quantile)
@infer_global(quantile_parallel)
class QuantileType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) in [2, 3]
        return signature(types.float64, *unliteral_all(args))


@lower_builtin(quantile, types.npytypes.Array, types.float64)
@lower_builtin(quantile_parallel, types.npytypes.Array, types.float64, types.intp)
def lower_dist_quantile(context, builder, sig, args):

    # store an int to specify data type
    typ_enum = _numba_to_c_type_map[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(
        builder, lir.Constant(lir.IntType(32), typ_enum))
    assert sig.args[0].ndim == 1

    arr = make_array(sig.args[0])(context, builder, args[0])
    local_size = builder.extract_value(arr.shape, 0)

    if len(args) == 3:
        total_size = args[2]
    else:
        # sequential case
        total_size = local_size

    call_args = [builder.bitcast(arr.data, lir.IntType(8).as_pointer()),
                 local_size, total_size, args[1], builder.load(typ_arg)]

    # array, size, total_size, quantile, type enum
    arg_typs = [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(64),
                lir.DoubleType(), lir.IntType(32)]
    fnty = lir.FunctionType(lir.DoubleType(), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="quantile_parallel")
    return builder.call(fn, call_args)


################################ nlargest ####################################


@numba.njit
def min_heapify(arr, ind_arr, n, start, cmp_f):
    min_ind = start
    left = 2 * start + 1
    right = 2 * start + 2

    if left < n and not cmp_f(arr[left], arr[min_ind]):  # < for nlargest
        min_ind = left

    if right < n and not cmp_f(arr[right], arr[min_ind]):
        min_ind = right

    if min_ind != start:
        arr[start], arr[min_ind] = arr[min_ind], arr[start]  # swap
        ind_arr[start], ind_arr[min_ind] = ind_arr[min_ind], ind_arr[start]
        min_heapify(arr, ind_arr, n, min_ind, cmp_f)


def select_k_nonan(A, index_arr, m, k):  # pragma: no cover
    return A[:k]


@overload(select_k_nonan)
def select_k_nonan_overload(A, index_arr, m, k):
    dtype = A.dtype
    # TODO: other types like strings and categoricals
    # TODO: handle NA in integer
    if isinstance(dtype, types.Integer):
        # ints don't have nans
        return lambda A, index_arr, m, k: (
            A[:k].copy(), index_arr[:k].copy(), k)


    def select_k_nonan_float(A, index_arr, m, k):
        # select the first k elements but ignore NANs
        min_heap_vals = np.empty(k, A.dtype)
        min_heap_inds = np.empty(k, index_arr.dtype)
        i = 0
        ind = 0
        while i < m and ind < k:
            if not bodo.hiframes.api.isna(A, i):
                min_heap_vals[ind] = A[i]
                min_heap_inds[ind] = index_arr[i]
                ind += 1
            i += 1

        # if couldn't fill with k values
        if ind < k:
            min_heap_vals = min_heap_vals[:ind]
            min_heap_inds = min_heap_inds[:ind]

        return min_heap_vals, min_heap_inds, i

    return select_k_nonan_float


@numba.njit
def nlargest(A, index_arr, k, is_largest, cmp_f):
    # algorithm: keep a min heap of k largest values, if a value is greater
    # than the minimum (root) in heap, replace the minimum and rebuild the heap
    m = len(A)

    # if all of A, just sort and reverse
    if k >= m:
        B = np.sort(A)
        out_index = index_arr[np.argsort(A)]
        mask = pd.Series(B).notna().values
        B = B[mask]
        out_index = out_index[mask]
        if is_largest:
            B = B[::-1]
            out_index = out_index[::-1]
        return np.ascontiguousarray(B), np.ascontiguousarray(out_index)

    # create min heap but
    min_heap_vals, min_heap_inds, start = select_k_nonan(A, index_arr, m, k)
    # heapify k/2-1 to 0 instead of sort?
    min_heap_inds = min_heap_inds[min_heap_vals.argsort()]
    min_heap_vals.sort()
    if not is_largest:
        min_heap_vals = np.ascontiguousarray(min_heap_vals[::-1])
        min_heap_inds = np.ascontiguousarray(min_heap_inds[::-1])

    for i in range(start, m):
        if cmp_f(A[i], min_heap_vals[0]):  # > for nlargest
            min_heap_vals[0] = A[i]
            min_heap_inds[0] = index_arr[i]
            min_heapify(min_heap_vals, min_heap_inds, k, 0, cmp_f)

    # sort and return the heap values
    min_heap_inds = min_heap_inds[min_heap_vals.argsort()]
    min_heap_vals.sort()
    if is_largest:
        min_heap_vals = min_heap_vals[::-1]
        min_heap_inds = min_heap_inds[::-1]
    return (np.ascontiguousarray(min_heap_vals),
            np.ascontiguousarray(min_heap_inds))


@numba.njit
def nlargest_parallel(A, I, k, is_largest, cmp_f):
    # parallel algorithm: assuming k << len(A), just call nlargest on chunks
    # of A, gather the result and return the largest k
    # TODO: support cases where k is not too small
    my_rank = bodo.libs.distributed_api.get_rank()
    local_res, local_res_ind = nlargest(A, I, k, is_largest, cmp_f)
    all_largest = bodo.libs.distributed_api.gatherv(local_res)
    all_largest_ind = bodo.libs.distributed_api.gatherv(local_res_ind)

    # TODO: handle len(res) < k case
    if my_rank == MPI_ROOT:
        res, res_ind = nlargest(
            all_largest, all_largest_ind, k, is_largest, cmp_f)
    else:
        res = np.empty(k, A.dtype)
        res_ind = np.empty(k, I.dtype)  # TODO: string array
    bodo.libs.distributed_api.bcast(res)
    bodo.libs.distributed_api.bcast(res_ind)
    return res, res_ind
