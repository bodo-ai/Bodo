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
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.core.imputils import lower_builtin
from numba.np.arrayobj import make_array
from numba.np.numpy_support import as_dtype
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.utils.utils import numba_to_c_type, unliteral_all
from bodo.libs.str_arr_ext import (
    string_array_type,
    pre_alloc_string_array,
    get_str_arr_item_length,
)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import BooleanArrayType, boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.utils.shuffle import getitem_arr_tup_single
from bodo.utils.utils import build_set
from bodo.ir.sort import (
    alltoallv_tup,
    finalize_shuffle_meta,
    update_shuffle_meta,
    alloc_pre_shuffle_metadata,
)
from bodo.libs.list_str_arr_ext import list_string_array_type
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.hiframes.split_impl import string_array_split_view_type
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.utils.indexing import init_nested_counts, add_nested_counts

from llvmlite import ir as lir
from bodo.libs import quantile_alg
import llvmlite.binding as ll

from bodo.libs.array import (
    array_to_info,
    arr_info_list_to_table,
    shuffle_table,
    drop_duplicates_table,
    info_from_table,
    info_to_array,
    delete_table,
)
from bodo.utils.typing import (
    BodoError,
    get_overload_const_list,
    get_overload_const_str,
    is_overload_none,
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


@overload(isna, no_unliteral=True)
def overload_isna(arr, i):
    i = types.unliteral(i)
    # String array
    if arr in (string_array_type, list_string_array_type):
        return lambda arr, i: bodo.libs.str_arr_ext.str_arr_is_na(
            arr, i
        )  # pragma: no cover

    # masked Integer array, boolean array
    if isinstance(arr, (IntegerArrayType, DecimalArrayType)) or arr in (
        boolean_array,
        datetime_date_array_type,
        string_array_split_view_type,
    ):
        return lambda arr, i: not bodo.libs.int_arr_ext.get_bit_bitmap_arr(
            arr._null_bitmap, i
        )  # pragma: no cover

    # array(item) array
    if isinstance(arr, ArrayItemArrayType):
        return lambda arr, i: not bodo.libs.int_arr_ext.get_bit_bitmap_arr(
            bodo.libs.array_item_arr_ext.get_null_bitmap(arr), i
        )  # pragma: no cover

    # struct array
    if isinstance(arr, StructArrayType):
        return lambda arr, i: not bodo.libs.int_arr_ext.get_bit_bitmap_arr(
            bodo.libs.struct_arr_ext.get_null_bitmap(arr), i
        )  # pragma: no cover

    # TODO: extend to other types (which ones are missing?)
    assert isinstance(arr, types.Array)
    dtype = arr.dtype
    if isinstance(dtype, types.Float):
        return lambda arr, i: np.isnan(arr[i])  # pragma: no cover

    # NaT for dt64
    if isinstance(dtype, (types.NPDatetime, types.NPTimedelta)):
        return lambda arr, i: np.isnat(arr[i])  # pragma: no cover

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
    typ_enum = numba_to_c_type(sig.args[0].dtype)
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
    typ_enum = numba_to_c_type(sig.args[0].dtype)
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


@overload(select_k_nonan, no_unliteral=True)
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


@overload(drop_duplicates, no_unliteral=True)
def overload_drop_duplicates(data, ind_arr, parallel=False):

    # TODO: inline for optimization?
    # TODO: handle NAs better?
    count = len(data)

    func_text = "def impl(data, ind_arr, parallel=False):\n"
    func_text += "  info_list_total = [{}, array_to_info(ind_arr)]\n".format(
        ", ".join("array_to_info(data[{}])".format(x) for x in range(count))
    )
    func_text += "  table_total = arr_info_list_to_table(info_list_total)\n"
    # We keep the first entry in the drop_duplicates
    func_text += "  keep_i = 0\n"
    func_text += "  out_table = drop_duplicates_table(table_total, parallel, {}, keep_i)\n".format(
        count
    )
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
            "drop_duplicates_table": drop_duplicates_table,
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


def dropna(data, how, thresh, subset, parallel=False):  # pragma: no cover
    return data


@overload(dropna, no_unliteral=True)
def overload_dropna(data, how, thresh, subset):
    """drop NA rows in tuple of arrays 'data'. 'subset' is the index numbers of arrays
    to consider for NA check. 'how' and 'thresh' are the same as df.dropna().
    """

    n_data_arrs = len(data.types)
    out_names = ["out" + str(i) for i in range(n_data_arrs)]
    subset_inds = get_overload_const_list(subset)
    how = get_overload_const_str(how)

    # gen NA check code
    isna_calls = ["isna(data[{}], i)".format(i) for i in subset_inds]
    isna_check = "not ({})".format(" or ".join(isna_calls))
    if not is_overload_none(thresh):
        isna_check = "(({}) <= ({}) - thresh)".format(
            " + ".join(isna_calls), n_data_arrs - 1
        )
    elif how == "all":
        isna_check = "not ({})".format(" and ".join(isna_calls))

    # count the number of elements in output arrays, allocate output arrays, fill data
    func_text = "def _dropna_imp(data, how, thresh, subset):\n"
    func_text += "  old_len = len(data[0])\n"
    func_text += "  new_len = 0\n"
    for i in range(n_data_arrs):
        func_text += "  nested_counts_{0} = init_nested_counts(d{0})\n".format(i)
    func_text += "  for i in range(old_len):\n"
    func_text += "    if {}:\n".format(isna_check)
    for i in range(n_data_arrs):
        func_text += "      if not isna(data[{}], i):\n".format(i)
        func_text += "        nested_counts_{0} = add_nested_counts(nested_counts_{0}, data[{0}][i])\n".format(
            i
        )
    func_text += "      new_len += 1\n"
    # allocate new arrays
    for i, out in enumerate(out_names):
        func_text += "  {0} = bodo.utils.utils.alloc_type(new_len, t{1}, nested_counts_{1})\n".format(
            out, i
        )
    func_text += "  curr_ind = 0\n"
    func_text += "  for i in range(old_len):\n"
    func_text += "    if {}:\n".format(isna_check)
    for i in range(n_data_arrs):
        func_text += "      if isna(data[{}], i):\n".format(i)
        func_text += "        bodo.ir.join.setitem_arr_nan({}, curr_ind)\n".format(
            out_names[i]
        )
        func_text += "      else:\n"
        func_text += "        {}[curr_ind] = data[{}][i]\n".format(out_names[i], i)
    func_text += "      curr_ind += 1\n"
    func_text += "  return {}\n".format(", ".join(out_names))
    loc_vars = {}
    # pass data types to generated code
    _globals = {"t{}".format(i): t for i, t in enumerate(data.types)}
    _globals.update({"d{}".format(i): t.dtype for i, t in enumerate(data.types)})
    _globals.update(
        {
            "isna": isna,
            "init_nested_counts": bodo.utils.indexing.init_nested_counts,
            "add_nested_counts": bodo.utils.indexing.add_nested_counts,
            "bodo": bodo,
        }
    )
    exec(func_text, _globals, loc_vars)
    _dropna_imp = loc_vars["_dropna_imp"]
    return _dropna_imp


def concat(arr_list):  # pragma: no cover
    return pd.concat(arr_list)


@overload(concat, no_unliteral=True)
def concat_overload(arr_list):
    # all string input case
    # TODO: handle numerics to string casting case

    if isinstance(arr_list, types.UniTuple) and isinstance(
        arr_list.dtype, ArrayItemArrayType
    ):
        data_arr_type = arr_list.dtype.dtype

        def array_item_concat_impl(arr_list):  # pragma: no cover
            # preallocate the output
            num_lists = 0
            data_arrs = []
            for A in arr_list:
                n_lists = len(A)
                data_arrs.append(bodo.libs.array_item_arr_ext.get_data(A))
                num_lists += n_lists
            out_offsets = np.empty(num_lists + 1, np.uint32)
            out_data = bodo.libs.array_kernels.concat(data_arrs)
            out_null_bitmap = np.empty((num_lists + 7) >> 3, np.uint8)
            # copy data to output
            curr_list = 0
            curr_item = 0
            for A in arr_list:
                in_offsets = bodo.libs.array_item_arr_ext.get_offsets(A)
                in_null_bitmap = bodo.libs.array_item_arr_ext.get_null_bitmap(A)
                n_lists = len(A)
                n_items = in_offsets[n_lists]
                # copying of index
                for i in range(n_lists):
                    out_offsets[i + curr_list] = in_offsets[i] + curr_item
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(in_null_bitmap, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        out_null_bitmap, i + curr_list, bit
                    )
                # shifting indexes
                curr_list += n_lists
                curr_item += n_items
            out_offsets[curr_list] = curr_item
            out_arr = bodo.libs.array_item_arr_ext.init_array_item_array(
                num_lists, out_data, out_offsets, out_null_bitmap
            )
            return out_arr

        return array_item_concat_impl

    if (
        isinstance(arr_list, types.UniTuple)
        and arr_list.dtype == datetime_date_array_type
    ):

        def datetime_date_array_concat_impl(arr_list):  # pragma: no cover
            tot_len = 0
            for A in arr_list:
                tot_len += len(A)
            Aret = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(tot_len)
            curr_pos = 0
            for A in arr_list:
                for i in range(len(A)):
                    Aret._data[i + curr_pos] = A._data[i]
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(A._null_bitmap, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        Aret._null_bitmap, i + curr_pos, bit
                    )
                curr_pos += len(A)

            return Aret

        return datetime_date_array_concat_impl

    if isinstance(arr_list, types.UniTuple) and isinstance(
        arr_list.dtype, DecimalArrayType
    ):
        precision = arr_list.dtype.precision
        scale = arr_list.dtype.scale

        def decimal_array_concat_impl(arr_list):  # pragma: no cover
            tot_len = 0
            for A in arr_list:
                tot_len += len(A)
            Aret = bodo.libs.decimal_arr_ext.alloc_decimal_array(
                tot_len, precision, scale
            )
            curr_pos = 0
            for A in arr_list:
                for i in range(len(A)):
                    Aret._data[i + curr_pos] = A._data[i]
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(A._null_bitmap, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        Aret._null_bitmap, i + curr_pos, bit
                    )
                curr_pos += len(A)

            return Aret

        return decimal_array_concat_impl

    if (
        isinstance(arr_list, types.UniTuple)
        and arr_list.dtype == list_string_array_type
    ):

        def list_string_array_concat_impl(arr_list):  # pragma: no cover
            # preallocate the output
            num_lists = 0
            num_strs = 0
            num_chars = 0
            for A in arr_list:
                arr = A
                n_list = len(arr)
                n_str = arr._index_offsets[
                    n_list
                ]  # We use here that the first index is 0
                n_char = arr._data_offsets[n_str]
                num_lists += n_list
                num_strs += n_str
                num_chars += n_char
            out_arr = bodo.libs.list_str_arr_ext.pre_alloc_list_string_array(
                num_lists, num_strs, num_chars
            )
            curr_lists = 0
            curr_strs = 0
            curr_chars = 0
            for A in arr_list:
                n_list = len(A)
                n_str = A._index_offsets[n_list]
                n_char = A._data_offsets[n_str]
                # adjusting the indexes
                for i in range(n_list):
                    out_arr._index_offsets[curr_lists + i] = (
                        A._index_offsets[i] + curr_strs
                    )
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(A._null_bitmap, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        out_arr._null_bitmap, i + curr_lists, bit
                    )
                for i in range(n_str):
                    out_arr._data_offsets[curr_strs + i] = (
                        A._data_offsets[i] + curr_chars
                    )
                # Copying the characters
                in_ptr = bodo.hiframes.split_impl.get_c_arr_ptr(A._data, 0)
                out_ptr = bodo.hiframes.split_impl.get_c_arr_ptr(
                    out_arr._data, curr_chars
                )
                bodo.libs.str_arr_ext._memcpy(out_ptr, in_ptr, n_char, 1)
                # Updating the shifts
                curr_lists += n_list
                curr_strs += n_str
                curr_chars += n_char

            out_arr._index_offsets[num_lists] = num_strs
            out_arr._data_offsets[num_strs] = num_chars
            return out_arr

        return list_string_array_concat_impl

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

    # list of nullable int arrays
    if isinstance(arr_list, types.List) and isinstance(
        arr_list.dtype, IntegerArrayType
    ):

        def impl_int_arr_list(arr_list):
            all_data = []
            n_all = 0
            for A in arr_list:
                all_data.append(A._data)
                n_all += len(A)
            out_data = bodo.libs.array_kernels.concat(all_data)
            n_bytes = (n_all + 7) >> 3
            new_mask = np.empty(n_bytes, np.uint8)
            curr_bit = 0
            for A in arr_list:
                old_mask = A._null_bitmap
                for j in range(len(A)):
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, j)
                    bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
                    curr_bit += 1
            return bodo.libs.int_arr_ext.init_integer_array(out_data, new_mask,)

        return impl_int_arr_list

    if isinstance(arr_list, types.UniTuple) and arr_list.dtype == boolean_array:
        # reusing int arr concat functions
        # TODO: test
        return lambda arr_list: bodo.libs.bool_arr_ext.init_bool_array(
            np.concatenate(bodo.libs.int_arr_ext.get_int_arr_data_tup(arr_list)),
            bodo.libs.int_arr_ext.concat_bitmap_tup(arr_list),
        )

    # list of 1D np arrays
    if (
        isinstance(arr_list, types.List)
        and isinstance(arr_list.dtype, types.Array)
        and arr_list.dtype.ndim == 1
    ):
        dtype = arr_list.dtype.dtype

        def impl_np_arr_list(arr_list):
            n_all = 0
            for A in arr_list:
                n_all += len(A)
            out_arr = np.empty(n_all, dtype)
            curr_val = 0
            for A in arr_list:
                n = len(A)
                out_arr[curr_val : curr_val + n] = A
                curr_val += n
            return out_arr

        return impl_np_arr_list

    # arrays of int/float mix need conversion to all-float before concat
    if (
        isinstance(arr_list, types.BaseTuple)
        and any(
            isinstance(t, (types.Array, IntegerArrayType))
            and isinstance(t.dtype, types.Integer)
            for t in arr_list.types
        )
        and any(
            isinstance(t, types.Array) and isinstance(t.dtype, types.Float)
            for t in arr_list.types
        )
    ):
        return lambda arr_list: np.concatenate(astype_float_tup(arr_list))

    for typ in arr_list:
        if not isinstance(typ, types.Array):
            raise BodoError("concat supports only numerical and string arrays")
    # numerical input
    return lambda arr_list: np.concatenate(arr_list)


def astype_float_tup(arr_tup):
    return tuple(t.astype(np.float64) for t in arr_tup)


@overload(astype_float_tup, no_unliteral=True)
def overload_astype_float_tup(arr_tup):
    """converts a tuple of arrays to float arrays using array.astype(np.float64)
    """
    assert isinstance(arr_tup, types.BaseTuple)
    count = len(arr_tup.types)

    func_text = "def f(arr_tup):\n"
    func_text += "  return ({}{})\n".format(
        ",".join("arr_tup[{}].astype(np.float64)".format(i) for i in range(count)),
        "," if count == 1 else "",
    )  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {"np": np}, loc_vars)
    astype_impl = loc_vars["f"]
    return astype_impl


def nunique(A):  # pragma: no cover
    return len(set(A))


def nunique_parallel(A):  # pragma: no cover
    return len(set(A))


@overload(nunique, no_unliteral=True)
def nunique_overload(A):
    if A == boolean_array:
        return lambda A: len(A.unique())
    # TODO: extend to other types like datetime?
    def nunique_seq(A):
        return len(build_set(A))

    return nunique_seq


@overload(nunique_parallel, no_unliteral=True)
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


@overload(unique, no_unliteral=True)
def unique_overload(A):
    # TODO: extend to other types like datetime?
    def unique_seq(A):
        return bodo.utils.utils.unique(A)

    return unique_seq


@overload(unique_parallel, no_unliteral=True)
def unique_overload_parallel(A):
    def unique_par(A):  # pragma: no cover
        input_table = arr_info_list_to_table([array_to_info(A)])
        n_key = 1
        keep_i = 0
        out_table = drop_duplicates_table(input_table, True, n_key, keep_i)
        out_arr = info_to_array(info_from_table(out_table, 0), A)
        delete_table(input_table)
        delete_table(out_table)
        return bodo.utils.utils.unique(out_arr)

    return unique_par


def explode(arr, index_arr):  # pragma: no cover
    return pd.Series(arr, index_arr).explode()


@overload(explode, no_unliteral=True)
def overload_explode(arr, index_arr):
    """
    Internal kernel for Series.explode(). Transforms each item in array(item) array into
    its own row, replicating the index values. Each empty array will have an NA in
    output.
    """
    assert isinstance(arr, ArrayItemArrayType)
    data_arr_type = arr.dtype
    index_arr_type = index_arr
    index_dtype = index_arr_type.dtype

    def impl(arr, index_arr):
        n = len(arr)
        nested_counts = init_nested_counts(data_arr_type)
        nested_index_counts = init_nested_counts(index_dtype)
        for i in range(n):
            ind_val = index_arr[i]
            if isna(arr, i):
                nested_counts = (nested_counts[0] + 1,) + nested_counts[1:]
                nested_index_counts = add_nested_counts(nested_index_counts, ind_val)
                continue
            arr_item = arr[i]
            if len(arr_item) == 0:
                nested_counts = (nested_counts[0] + 1,) + nested_counts[1:]
                nested_index_counts = add_nested_counts(nested_index_counts, ind_val)
                continue
            nested_counts = add_nested_counts(nested_counts, arr_item)
            for _ in range(len(arr_item)):
                nested_index_counts = add_nested_counts(nested_index_counts, ind_val)
        out_arr = bodo.utils.utils.alloc_type(
            nested_counts[0], data_arr_type, nested_counts[1:]
        )
        out_index_arr = bodo.utils.utils.alloc_type(
            nested_counts[0], index_arr_type, nested_index_counts
        )

        curr_item = 0
        for i in range(n):
            if isna(arr, i):
                bodo.ir.join.setitem_arr_nan(out_arr, curr_item)
                out_index_arr[curr_item] = index_arr[i]
                curr_item += 1
                continue
            arr_item = arr[i]
            n_items = len(arr_item)
            if n_items == 0:
                bodo.ir.join.setitem_arr_nan(out_arr, curr_item)
                out_index_arr[curr_item] = index_arr[i]
                curr_item += 1
                continue
            out_arr[curr_item : curr_item + n_items] = arr_item
            out_index_arr[curr_item : curr_item + n_items] = index_arr[i]
            curr_item += n_items

        return out_arr, out_index_arr

    return impl


def gen_na_array(n, arr):  # pragma: no cover
    return np.full(n, np.nan)


@overload(gen_na_array, no_unliteral=True)
def overload_gen_na_array(n, arr):
    """
    generate an array full of NA values with the same type as 'arr'
    """
    # TODO: support all array types

    if isinstance(arr, types.TypeRef):
        dtype = arr.instance_type
    else:
        dtype = arr.dtype

    if isinstance(arr, ArrayItemArrayType):
        data_arr_type = arr.dtype

        def array_item_impl(n, arr):  # pragma: no cover
            # preallocate the output
            num_lists = n
            num_items = 0
            in_data = bodo.libs.array_item_arr_ext.get_data(arr)
            out_arr = bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
                num_lists, (num_items,), data_arr_type
            )
            out_offsets = bodo.libs.array_item_arr_ext.get_offsets(out_arr)
            out_data = bodo.libs.array_item_arr_ext.get_data(out_arr)
            out_null_bitmap = bodo.libs.array_item_arr_ext.get_null_bitmap(out_arr)
            for i in range(n + 1):
                out_offsets[i] = 0
            for i in range(n):
                bodo.libs.int_arr_ext.set_bit_to_arr(out_null_bitmap, i, 0)
            return out_arr

        return array_item_impl

    if arr == datetime_date_array_type:

        def impl_datetime_date(n, arr):  # pragma: no cover
            A = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)
            for i in range(n):
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 0)
            return A

        return impl_datetime_date

    if arr == boolean_array:

        def impl_boolean(n, arr):  # pragma: no cover
            A = bodo.libs.bool_arr_ext.alloc_bool_array(n)
            for i in range(n):
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 0)
            return A

        return impl_boolean

    if isinstance(arr, DecimalArrayType):
        precision = arr.dtype.precision
        scale = arr.dtype.scale

        def impl_decimal(n, arr):  # pragma: no cover
            A = bodo.libs.decimal_arr_ext.alloc_decimal_array(n, precision, scale)
            for i in range(n):
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 0)
            return A

        return impl_decimal

    if arr == list_string_array_type:

        def impl_list_string(n, arr):  # pragma: no cover
            # preallocate the output
            num_lists = n
            num_strs = 0
            num_chars = 0
            out_arr = bodo.libs.list_str_arr_ext.pre_alloc_list_string_array(
                num_lists, num_strs, num_chars
            )

            for i in numba.parfors.parfor.internal_prange(n):
                out_arr._index_offsets[i] = 0
                bit = 0
                bodo.libs.int_arr_ext.set_bit_to_arr(out_arr._null_bitmap, i, bit)

            return out_arr

        return impl_list_string

    # array of np.nan values if 'arr' is float or int Numpy array
    # TODO: use nullable int array
    if isinstance(dtype, (types.Integer, types.Float)):
        dtype = dtype if isinstance(dtype, types.Float) else types.float64

        def impl_float(n, arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            out_arr = np.empty(n, dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                out_arr[i] = np.nan
            return out_arr

        return impl_float

    if isinstance(dtype, (types.NPDatetime, types.NPTimedelta)):
        nat = dtype("NaT")
        if dtype == types.NPDatetime("ns"):
            dtype = np.dtype("datetime64[ns]")
        elif dtype == types.NPTimedelta("ns"):
            dtype = np.dtype("timedelta64[ns]")

        def impl_dt64(n, arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            out_arr = np.empty(n, dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                out_arr[i] = nat
            return out_arr

        return impl_dt64

    if dtype == bodo.string_type:

        def impl_str(n, arr):  # pragma: no cover
            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, 0)
            for j in numba.parfors.parfor.internal_prange(n):
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            return out_arr

        return impl_str

    # TODO: support all array types
    raise BodoError(
        "array type {} not supported for all NA generation in gen_na_array() yet".format(
            arr
        )
    )  # pragma: no cover


def gen_na_array_equiv(self, scope, equiv_set, loc, args, kws):  # pragma: no cover
    """Array analysis function for gen_na_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert not kws
    return args[0], []


ArrayAnalysis._analyze_op_call_bodo_libs_array_kernels_gen_na_array = gen_na_array_equiv


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
            numba.parfors.parfor.init_prange()
            nitems_c = (stop - start) / step
            nitems_r = math.ceil(nitems_c.real)
            nitems_i = math.ceil(nitems_c.imag)
            nitems = int(max(min(nitems_i, nitems_r), 0))
            arr = np.empty(nitems, dtype)
            for i in numba.parfors.parfor.internal_prange(nitems):
                arr[i] = start + i * step
            return arr

    else:

        def arange_4(start, stop, step, dtype):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            nitems = bodo.libs.array_kernels.calc_nitems(start, stop, step)
            arr = np.empty(nitems, dtype)
            for i in numba.parfors.parfor.internal_prange(nitems):
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


numba.parfors.parfor.replace_functions_map[("arange", "numpy")] = arange_parallel_impl
