# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implements array kernels such as median and quantile.
"""
import pandas as pd
import numpy as np
import math
from math import sqrt

import numba
from numba.extending import overload
from numba import types, cgutils
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import lower_builtin, impl_ret_untracked
from numba.targets.arrayobj import make_array
from numba.numpy_support import as_dtype

import bodo
from bodo.utils.utils import _numba_to_c_type_map, unliteral_all
from bodo.libs.str_arr_ext import (
    string_array_type,
    pre_alloc_string_array,
    get_str_arr_item_length,
)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import BooleanArrayType, boolean_array
from bodo.utils.shuffle import getitem_arr_tup_single
from bodo.utils.utils import build_set
from bodo.ir.sort import (
    alltoallv_tup,
    finalize_shuffle_meta,
    update_shuffle_meta,
    alloc_pre_shuffle_metadata,
)
from bodo.ir.join import write_send_buff
from bodo.libs.list_str_arr_ext import list_string_array_type
from bodo.hiframes.split_impl import string_array_split_view_type
from bodo.hiframes.datetime_date_ext import datetime_date_array_type

import llvmlite.llvmpy.core as lc
from llvmlite import ir as lir
from bodo.libs import quantile_alg
import llvmlite.binding as ll

from bodo.libs.array_tools import (
    array_to_info,
    arr_info_list_to_table,
    shuffle_table,
    drop_duplicates_table_outplace,
    info_from_table,
    info_to_array,
    delete_table,
)

ll.add_symbol("quantile_sequential", quantile_alg.quantile_sequential)
ll.add_symbol("quantile_parallel", quantile_alg.quantile_parallel)
ll.add_symbol("nth_sequential", quantile_alg.nth_sequential)
ll.add_symbol("nth_parallel", quantile_alg.nth_parallel)


nth_sequential = types.ExternalFunction(
    "nth_sequential",
    types.void(types.voidptr, types.voidptr, types.int64, types.int64, types.int32),
)

nth_parallel = types.ExternalFunction(
    "nth_parallel",
    types.void(types.voidptr, types.voidptr, types.int64, types.int64, types.int32),
)

MPI_ROOT = 0
sum_op = np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)


def isna(arr, i):  # pragma: no cover
    return False


@overload(isna)
def isna_overload(arr, i):
    # String array
    if arr == string_array_type:
        return lambda arr, i: bodo.libs.str_arr_ext.str_arr_is_na(arr, i)

    # masked Integer array, boolean array
    if isinstance(arr, IntegerArrayType) or arr == boolean_array:
        return lambda arr, i: not bodo.libs.int_arr_ext.get_bit_bitmap_arr(
            arr._null_bitmap, i
        )

    if arr == list_string_array_type:
        # reuse string array function
        return lambda arr, i: bodo.libs.str_arr_ext.str_arr_is_na(arr, i)

    # TODO: support NAs in split view
    if arr == string_array_split_view_type:
        return lambda arr, i: False

    # TODO: support NAs in datetime.date array
    if arr == datetime_date_array_type:
        return lambda arr, i: False

    # TODO: extend to other types
    assert isinstance(arr, types.Array)
    dtype = arr.dtype
    if isinstance(dtype, types.Float):
        return lambda arr, i: np.isnan(arr[i])

    # NaT for dt64
    if isinstance(dtype, (types.NPDatetime, types.NPTimedelta)):
        nat = dtype("NaT")
        # TODO: replace with np.isnat
        return lambda arr, i: arr[i] == nat

    # XXX integers don't have nans, extend to boolean
    return lambda arr, i: False


################################ median ####################################


@numba.njit
def nth_element(arr, k, parallel=False):  # pragma: no cover
    res = np.empty(1, arr.dtype)
    type_enum = bodo.libs.distributed_api.get_type_enum(arr)
    if parallel:
        nth_parallel(res.ctypes, arr.ctypes, len(arr), k, type_enum)
    else:
        nth_sequential(res.ctypes, arr.ctypes, len(arr), k, type_enum)
    return res[0]


@numba.njit
def median(arr, parallel=False):  # pragma: no cover
    # similar to numpy/lib/function_base.py:_median
    # TODO: check return types, e.g. float32 -> float32
    n = len(arr)
    if parallel:
        n = bodo.libs.distributed_api.dist_reduce(n, np.int32(sum_op))
    k = n // 2

    # odd length case
    if n % 2 == 1:
        return nth_element(arr, k, parallel)

    v1 = nth_element(arr, k - 1, parallel)
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


@lower_builtin(quantile, types.Array, types.float64)
@lower_builtin(quantile, IntegerArrayType, types.float64)
@lower_builtin(quantile, BooleanArrayType, types.float64)
def lower_dist_quantile_seq(context, builder, sig, args):

    # store an int to specify data type
    typ_enum = _numba_to_c_type_map[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(
        builder, lir.Constant(lir.IntType(32), typ_enum)
    )

    arr_val = args[0]
    arr_typ = sig.args[0]
    if isinstance(arr_typ, (IntegerArrayType, BooleanArrayType)):
        arr_val = cgutils.create_struct_proxy(arr_typ)(context, builder, arr_val).data
        arr_typ = types.Array(arr_typ.dtype, 1, "C")

    assert arr_typ.ndim == 1

    arr = make_array(arr_typ)(context, builder, arr_val)
    local_size = builder.extract_value(arr.shape, 0)

    call_args = [
        builder.bitcast(arr.data, lir.IntType(8).as_pointer()),
        local_size,
        args[1],
        builder.load(typ_arg),
    ]

    # array, size,  quantile, type enum
    arg_typs = [
        lir.IntType(8).as_pointer(),
        lir.IntType(64),
        lir.DoubleType(),
        lir.IntType(32),
    ]
    fnty = lir.FunctionType(lir.DoubleType(), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="quantile_sequential")
    return builder.call(fn, call_args)


@lower_builtin(quantile_parallel, types.Array, types.float64, types.intp)
@lower_builtin(quantile_parallel, IntegerArrayType, types.float64, types.intp)
@lower_builtin(quantile_parallel, BooleanArrayType, types.float64, types.intp)
def lower_dist_quantile_parallel(context, builder, sig, args):

    # store an int to specify data type
    typ_enum = _numba_to_c_type_map[sig.args[0].dtype]
    typ_arg = cgutils.alloca_once_value(
        builder, lir.Constant(lir.IntType(32), typ_enum)
    )

    arr_val = args[0]
    arr_typ = sig.args[0]
    if isinstance(arr_typ, (IntegerArrayType, BooleanArrayType)):
        arr_val = cgutils.create_struct_proxy(arr_typ)(context, builder, arr_val).data
        arr_typ = types.Array(arr_typ.dtype, 1, "C")

    assert arr_typ.ndim == 1

    arr = make_array(arr_typ)(context, builder, arr_val)
    local_size = builder.extract_value(arr.shape, 0)

    if len(args) == 3:
        total_size = args[2]
    else:
        # sequential case
        total_size = local_size

    call_args = [
        builder.bitcast(arr.data, lir.IntType(8).as_pointer()),
        local_size,
        total_size,
        args[1],
        builder.load(typ_arg),
    ]

    # array, size, total_size, quantile, type enum
    arg_typs = [
        lir.IntType(8).as_pointer(),
        lir.IntType(64),
        lir.IntType(64),
        lir.DoubleType(),
        lir.IntType(32),
    ]
    fnty = lir.FunctionType(lir.DoubleType(), arg_typs)
    fn = builder.module.get_or_insert_function(fnty, name="quantile_parallel")
    return builder.call(fn, call_args)


################################ nlargest ####################################


@numba.njit
def min_heapify(arr, ind_arr, n, start, cmp_f):  # pragma: no cover
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
        return lambda A, index_arr, m, k: (A[:k].copy(), index_arr[:k].copy(), k)

    def select_k_nonan_float(A, index_arr, m, k):  # pragma: no cover
        # select the first k elements but ignore NANs
        min_heap_vals = np.empty(k, A.dtype)
        min_heap_inds = np.empty(k, index_arr.dtype)
        i = 0
        ind = 0
        while i < m and ind < k:
            if not bodo.libs.array_kernels.isna(A, i):
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
def nlargest(A, index_arr, k, is_largest, cmp_f):  # pragma: no cover
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
    return (np.ascontiguousarray(min_heap_vals), np.ascontiguousarray(min_heap_inds))


@numba.njit
def nlargest_parallel(A, I, k, is_largest, cmp_f):  # pragma: no cover
    # parallel algorithm: assuming k << len(A), just call nlargest on chunks
    # of A, gather the result and return the largest k
    # TODO: support cases where k is not too small
    my_rank = bodo.libs.distributed_api.get_rank()
    local_res, local_res_ind = nlargest(A, I, k, is_largest, cmp_f)
    all_largest = bodo.libs.distributed_api.gatherv(local_res)
    all_largest_ind = bodo.libs.distributed_api.gatherv(local_res_ind)

    # TODO: handle len(res) < k case
    if my_rank == MPI_ROOT:
        res, res_ind = nlargest(all_largest, all_largest_ind, k, is_largest, cmp_f)
    else:
        res = np.empty(k, A.dtype)
        res_ind = np.empty(k, I.dtype)  # TODO: string array
    bodo.libs.distributed_api.bcast(res)
    bodo.libs.distributed_api.bcast(res_ind)
    return res, res_ind


# adapted from pandas/_libs/algos.pyx/nancorr()
@numba.njit(no_cpython_wrapper=True, cache=True)
def nancorr(mat, cov=0, minpv=1, parallel=False):  # pragma: no cover
    N, K = mat.shape
    result = np.empty((K, K), dtype=np.float64)

    for xi in range(K):
        for yi in range(xi + 1):
            nobs = 0
            sumxx = sumyy = sumx = sumy = 0.0
            for i in range(N):
                if np.isfinite(mat[i, xi]) and np.isfinite(mat[i, yi]):
                    vx = mat[i, xi]
                    vy = mat[i, yi]
                    nobs += 1
                    sumx += vx
                    sumy += vy

            if parallel:
                nobs = bodo.libs.distributed_api.dist_reduce(nobs, sum_op)
                sumx = bodo.libs.distributed_api.dist_reduce(sumx, sum_op)
                sumy = bodo.libs.distributed_api.dist_reduce(sumy, sum_op)

            if nobs < minpv:
                result[xi, yi] = result[yi, xi] = np.nan
            else:
                meanx = sumx / nobs
                meany = sumy / nobs

                # now the cov numerator
                sumx = 0.0

                for i in range(N):
                    if np.isfinite(mat[i, xi]) and np.isfinite(mat[i, yi]):
                        vx = mat[i, xi] - meanx
                        vy = mat[i, yi] - meany

                        sumx += vx * vy
                        sumxx += vx * vx
                        sumyy += vy * vy

                if parallel:
                    sumx = bodo.libs.distributed_api.dist_reduce(sumx, sum_op)
                    sumxx = bodo.libs.distributed_api.dist_reduce(sumxx, sum_op)
                    sumyy = bodo.libs.distributed_api.dist_reduce(sumyy, sum_op)

                divisor = (nobs - 1.0) if cov else sqrt(sumxx * sumyy)

                if divisor != 0.0:
                    result[xi, yi] = result[yi, xi] = sumx / divisor
                else:
                    result[xi, yi] = result[yi, xi] = np.nan

    return result


@numba.njit(no_cpython_wrapper=True)
def duplicated(data, ind_arr, parallel=False):  # pragma: no cover
    # TODO: inline for optimization?
    # TODO: handle NAs better?

    if parallel:
        data, (ind_arr,) = bodo.ir.join.parallel_shuffle(data, (ind_arr,))

    # XXX: convert StringArray to list of strings due to strange error with set
    # TODO: debug StringArray issue on test_df_duplicated with multiple pes
    data = bodo.libs.str_arr_ext.to_string_list(data)

    n = len(data[0])
    out = np.empty(n, np.bool_)
    # uniqs = set()
    uniqs = dict()

    for i in range(n):
        val = getitem_arr_tup_single(data, i)
        if val in uniqs:
            out[i] = True
        else:
            out[i] = False
            # uniqs.add(val)
            uniqs[val] = 0

    return out, ind_arr


def drop_duplicates(data, ind_arr, parallel=False):  # pragma: no cover
    return data, ind_arr


@overload(drop_duplicates)
def overload_drop_duplicates(data, ind_arr, parallel=False):

    # TODO: inline for optimization?
    # TODO: handle NAs better?
    count = len(data)

    func_text = "def impl(data, ind_arr, parallel=False):\n"
    func_text += "  if parallel:\n"
    key_names = tuple(["data[" + str(i) + "]" for i in range(len(data))])
    func_text += bodo.ir.join._gen_par_shuffle(
        key_names, ("ind_arr",), "data", "data_ind", data.types, data.types
    )
    func_text += "    (ind_arr,) = data_ind\n"
    if not bodo.use_cpp_drop_duplicates:
        func_text += "  n = len(data[0])\n"
        for i in range(count):
            if data.types[i] == string_array_type:
                func_text += "  out_arr_{0} = pre_alloc_string_array(n, data[{0}]._num_total_chars)\n".format(
                    i
                )
            else:
                func_text += "  out_arr_{0} = bodo.utils.utils.alloc_type(n, data[{0}])\n".format(
                    i
                )
        if ind_arr == string_array_type:
            func_text += "  out_arr_index = pre_alloc_string_array(n, ind_arr._num_total_chars)\n"
        else:
            func_text += "  out_arr_index = bodo.utils.utils.alloc_type(n, ind_arr)\n"
        # func_text += "  uniqs = set()\n"
        func_text += "  uniqs = dict()\n"
        func_text += "  w_ind = 0\n"
        func_text += "  for i in range(n):\n"
        func_text += "    val = getitem_arr_tup_single(data, i)\n"
        func_text += "    if val in uniqs:\n"
        func_text += "      continue\n"
        # func_text += "    uniqs.add(val)\n"
        func_text += "    uniqs[val] = 0\n"
        for i in range(count):
            func_text += "    out_arr_{0}[w_ind] = data[{0}][i]\n".format(i)
        func_text += "    out_arr_index[w_ind] = ind_arr[i]\n"
        func_text += "    w_ind += 1\n"
        for i in range(count):
            func_text += "  out_arr_{0} = trim_arr(out_arr_{0}, w_ind)\n".format(i)
        func_text += "  out_arr_index = trim_arr(out_arr_index, w_ind)\n"
        func_text += "  return ({},), out_arr_index\n".format(
            ", ".join("out_arr_{}".format(i) for i in range(count))
        )
    else:
        func_text += "  info_list_total = [{}, array_to_info(ind_arr)]\n".format(
            ", ".join("array_to_info(data[{}])".format(x) for x in range(count))
        )
        func_text += "  table_total = arr_info_list_to_table(info_list_total)\n"
        # All are keys except the last one which is for index
        func_text += "  subset_vect = np.array([1] * {} + [0])\n".format(count)
        # We keep the first entry in the drop_duplicates
        func_text += "  keep_i = 0\n"
        func_text += "  out_table = drop_duplicates_table_outplace(table_total, subset_vect.ctypes, keep_i)\n"
        for i_col in range(count):
            func_text += "  out_arr_{} = info_to_array(info_from_table(out_table, {}), data[{}])\n".format(
                i_col, i_col, i_col
            )
        func_text += "  out_arr_index = info_to_array(info_from_table(out_table, {}), ind_arr)\n".format(
            count
        )
        func_text += "  delete_table(out_table)\n"
        func_text += "  delete_table(table_total)\n"
        func_text += "  return ({},), out_arr_index\n".format(
            ", ".join("out_arr_{}".format(i) for i in range(count))
        )

    #    print("array_kernels : func_text=", func_text)
    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "bodo": bodo,
            "pre_alloc_string_array": pre_alloc_string_array,
            "getitem_arr_tup_single": getitem_arr_tup_single,
            "get_str_arr_item_length": get_str_arr_item_length,
            "trim_arr": bodo.ir.join.trim_arr,
            "array_to_info": array_to_info,
            "drop_duplicates_table_outplace": drop_duplicates_table_outplace,
            "arr_info_list_to_table": arr_info_list_to_table,
            "shuffle_table": shuffle_table,
            "info_from_table": info_from_table,
            "info_to_array": info_to_array,
            "delete_table": delete_table,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


def concat(arr_list):  # pragma: no cover
    return pd.concat(arr_list)


@overload(concat)
def concat_overload(arr_list):
    # all string input case
    # TODO: handle numerics to string casting case
    if isinstance(arr_list, types.UniTuple) and arr_list.dtype == string_array_type:

        def string_concat_impl(arr_list):  # pragma: no cover
            # preallocate the output
            num_strs = 0
            num_chars = 0
            for A in arr_list:
                arr = A
                num_strs += len(arr)
                num_chars += bodo.libs.str_arr_ext.num_total_chars(arr)
            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(num_strs, num_chars)
            # copy data to output
            curr_str_ind = 0
            curr_chars_ind = 0
            for A in arr_list:
                arr = A
                bodo.libs.str_arr_ext.set_string_array_range(
                    out_arr, arr, curr_str_ind, curr_chars_ind
                )
                curr_str_ind += len(arr)
                curr_chars_ind += bodo.libs.str_arr_ext.num_total_chars(arr)
            return out_arr

        return string_concat_impl

    if isinstance(arr_list, types.UniTuple) and isinstance(
        arr_list.dtype, IntegerArrayType
    ):
        return lambda arr_list: bodo.libs.int_arr_ext.init_integer_array(
            np.concatenate(bodo.libs.int_arr_ext.get_int_arr_data_tup(arr_list)),
            bodo.libs.int_arr_ext.concat_bitmap_tup(arr_list),
        )

    if isinstance(arr_list, types.UniTuple) and arr_list.dtype == boolean_array:
        # reusing int arr concat functions
        # TODO: test
        return lambda arr_list: bodo.libs.bool_arr_ext.init_bool_array(
            np.concatenate(bodo.libs.int_arr_ext.get_int_arr_data_tup(arr_list)),
            bodo.libs.int_arr_ext.concat_bitmap_tup(arr_list),
        )

    for typ in arr_list:
        if not isinstance(typ, types.Array):
            raise ValueError("concat supports only numerical and string arrays")
    # numerical input
    return lambda arr_list: np.concatenate(arr_list)


def nunique(A):  # pragma: no cover
    return len(set(A))


def nunique_parallel(A):  # pragma: no cover
    return len(set(A))


@overload(nunique)
def nunique_overload(A):
    if A == boolean_array:
        return lambda A: len(A.unique())
    # TODO: extend to other types like datetime?
    def nunique_seq(A):
        return len(build_set(A))

    return nunique_seq


@overload(nunique_parallel)
def nunique_overload_parallel(A):
    sum_op = bodo.libs.distributed_api.Reduce_Type.Sum.value

    def nunique_par(A):  # pragma: no cover
        uniq_A = bodo.libs.array_kernels.unique_parallel(A)
        loc_nuniq = len(uniq_A)
        return bodo.libs.distributed_api.dist_reduce(loc_nuniq, np.int32(sum_op))

    return nunique_par


def unique(A):  # pragma: no cover
    return np.array([a for a in set(A)]).astype(A.dtype)


def unique_parallel(A):  # pragma: no cover
    return np.array([a for a in set(A)]).astype(A.dtype)


@overload(unique)
def unique_overload(A):
    # TODO: extend to other types like datetime?
    def unique_seq(A):
        return bodo.utils.utils.unique(A)

    return unique_seq


@overload(unique_parallel)
def unique_overload_parallel(A):
    def unique_par(A):  # pragma: no cover
        uniq_A = bodo.utils.utils.unique(A)
        key_arrs = (uniq_A,)
        n = len(uniq_A)
        node_ids = np.empty(n, np.int32)

        n_pes = bodo.libs.distributed_api.get_size()
        pre_shuffle_meta = alloc_pre_shuffle_metadata(key_arrs, (), n_pes, False)

        # calc send/recv counts
        for i in range(n):
            val = uniq_A[i]
            node_id = hash(val) % n_pes
            node_ids[i] = node_id
            update_shuffle_meta(pre_shuffle_meta, node_id, i, key_arrs, (), False)

        shuffle_meta = finalize_shuffle_meta(
            key_arrs, (), pre_shuffle_meta, n_pes, False
        )

        # write send buffers
        for i in range(n):
            node_id = node_ids[i]
            write_send_buff(shuffle_meta, node_id, i, key_arrs, ())
            # update last since it is reused in data
            shuffle_meta.tmp_offset[node_id] += 1

        # shuffle
        (out_arr,) = alltoallv_tup(key_arrs, shuffle_meta, ())

        return bodo.utils.utils.unique(out_arr)

    return unique_par


# np.arange implementation is copied from parfor.py and range length
# calculation is replaced with explicit call for easier matching
# (e.g. for handling 1D_Var RangeIndex)
# TODO: move this to upstream Numba
@numba.njit
def calc_nitems(start, stop, step):  # pragma: no cover
    nitems_r = math.ceil((stop - start) / step)
    return int(max(nitems_r, 0))


def arange_parallel_impl(return_type, *args):
    dtype = as_dtype(return_type.dtype)

    def arange_1(stop):  # pragma: no cover
        return np.arange(0, stop, 1, dtype)

    def arange_2(start, stop):  # pragma: no cover
        return np.arange(start, stop, 1, dtype)

    def arange_3(start, stop, step):  # pragma: no cover
        return np.arange(start, stop, step, dtype)

    if any(isinstance(a, types.Complex) for a in args):

        def arange_4(start, stop, step, dtype):  # pragma: no cover
            numba.parfor.init_prange()
            nitems_c = (stop - start) / step
            nitems_r = math.ceil(nitems_c.real)
            nitems_i = math.ceil(nitems_c.imag)
            nitems = int(max(min(nitems_i, nitems_r), 0))
            arr = np.empty(nitems, dtype)
            for i in numba.parfor.internal_prange(nitems):
                arr[i] = start + i * step
            return arr

    else:

        def arange_4(start, stop, step, dtype):  # pragma: no cover
            numba.parfor.init_prange()
            nitems = bodo.libs.array_kernels.calc_nitems(start, stop, step)
            arr = np.empty(nitems, dtype)
            for i in numba.parfor.internal_prange(nitems):
                arr[i] = start + i * step
            return arr

    if len(args) == 1:
        return arange_1
    elif len(args) == 2:
        return arange_2
    elif len(args) == 3:
        return arange_3
    elif len(args) == 4:
        return arange_4
    else:
        raise ValueError("parallel arange with types {}".format(args))


numba.parfor.replace_functions_map[("arange", "numpy")] = arange_parallel_impl
