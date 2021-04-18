# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implements array kernels such as median and quantile.
"""
import math
import operator
import re
from math import sqrt

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types, typing
from numba.core.imputils import lower_builtin
from numba.core.typing import signature
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.extending import overload, overload_attribute, register_jitable
from numba.np.arrayobj import make_array
from numba.np.numpy_support import as_dtype
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.datetime_timedelta_ext import datetime_timedelta_array_type
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    init_categorical_array,
)
from bodo.hiframes.split_impl import string_array_split_view_type
from bodo.libs import quantile_alg
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    delete_info_decref_array,
    delete_table,
    delete_table_decref_arrays,
    drop_duplicates_table,
    info_from_table,
    info_to_array,
    sample_table,
)
from bodo.libs.array_item_arr_ext import ArrayItemArrayType, offset_type
from bodo.libs.bool_arr_ext import BooleanArrayType, boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.distributed_api import Reduce_Type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import str_arr_set_na, string_array_type
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.indexing import add_nested_counts, init_nested_counts
from bodo.utils.shuffle import getitem_arr_tup_single
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    find_common_np_dtype,
    get_overload_const_list,
    get_overload_const_str,
    is_overload_none,
    raise_bodo_error,
)
from bodo.utils.utils import (
    build_set,
    check_and_propagate_cpp_exception,
    numba_to_c_type,
    unliteral_all,
)

ll.add_symbol("quantile_sequential", quantile_alg.quantile_sequential)
ll.add_symbol("quantile_parallel", quantile_alg.quantile_parallel)

MPI_ROOT = 0
sum_op = np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)


def isna(arr, i):  # pragma: no cover
    return False


@overload(isna, no_unliteral=True)
def overload_isna(arr, i):
    i = types.unliteral(i)
    # String array
    if arr == string_array_type:
        return lambda arr, i: bodo.libs.str_arr_ext.str_arr_is_na(
            arr, i
        )  # pragma: no cover

    # masked Integer array, boolean array
    if isinstance(arr, (IntegerArrayType, DecimalArrayType)) or arr in (
        boolean_array,
        datetime_date_array_type,
        datetime_timedelta_array_type,
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

    # tuple array
    if isinstance(arr, TupleArrayType):
        return lambda arr, i: bodo.libs.array_kernels.isna(
            arr._data, i
        )  # pragma: no cover

    # Categorical Array
    if isinstance(arr, bodo.hiframes.pd_categorical_ext.CategoricalArrayType):
        return lambda arr, i: arr.codes[i] == -1

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


def setna(arr, ind, int_nan_const=0):  # pragma: no cover
    arr[ind] = np.nan


@overload(setna, no_unliteral=True)
def setna_overload(arr, ind, int_nan_const=0):
    if isinstance(arr.dtype, types.Float):
        return setna

    if isinstance(arr.dtype, (types.NPDatetime, types.NPTimedelta)):
        nat = arr.dtype("NaT")

        def _setnan_impl(arr, ind, int_nan_const=0):  # pragma: no cover
            arr[ind] = nat

        return _setnan_impl

    if arr == string_array_type:

        def impl(arr, ind, int_nan_const=0):
            # set empty string to set offsets properly
            arr[ind] = ""
            str_arr_set_na(arr, ind)

        return impl

    if isinstance(arr, (IntegerArrayType, DecimalArrayType)) or arr == boolean_array:
        return lambda arr, ind, int_nan_const=0: bodo.libs.int_arr_ext.set_bit_to_arr(
            arr._null_bitmap, ind, 0
        )  # pragma: no cover

    if isinstance(arr, bodo.libs.array_item_arr_ext.ArrayItemArrayType):

        def impl_arr_item(arr, ind, int_nan_const=0):  # pragma: no cover
            # set offset
            offsets = bodo.libs.array_item_arr_ext.get_offsets(arr)
            offsets[ind + 1] = offsets[ind]
            # set NA bitmask
            bodo.libs.int_arr_ext.set_bit_to_arr(
                bodo.libs.array_item_arr_ext.get_null_bitmap(arr), ind, 0
            )

        return impl_arr_item

    if isinstance(arr, bodo.libs.struct_arr_ext.StructArrayType):

        def impl(arr, ind, int_nan_const=0):  # pragma: no cover
            bodo.libs.int_arr_ext.set_bit_to_arr(
                bodo.libs.struct_arr_ext.get_null_bitmap(arr), ind, 0
            )
            # set all data values to NA for this index
            data = bodo.libs.struct_arr_ext.get_data(arr)
            setna_tup(data, ind)

        return impl

    # tuple array
    if isinstance(arr, TupleArrayType):

        def impl(arr, ind, int_nan_const=0):  # pragma: no cover
            bodo.libs.array_kernels.setna(arr._data, ind)

        return impl

    # TODO: support strings, bools, etc.
    # XXX: set NA values in bool arrays to False
    # FIXME: replace with proper NaN
    if arr.dtype == types.bool_:

        def b_set(arr, ind, int_nan_const=0):  # pragma: no cover
            arr[ind] = False

        return b_set

    if isinstance(arr, bodo.hiframes.pd_categorical_ext.CategoricalArrayType):

        def setna_cat(arr, ind, int_nan_const=0):  # pragma: no cover
            arr.codes[ind] = -1

        return setna_cat

    # XXX set integer NA to 0 to avoid unexpected errors
    # TODO: convert integer to float if nan
    if isinstance(arr.dtype, types.Integer):

        def setna_int(arr, ind, int_nan_const=0):  # pragma: no cover
            arr[ind] = int_nan_const

        return setna_int

    # Add support for datetime.date array. This checks that the value in the
    # array won't cause a runtime error in getitem.
    if arr == datetime_date_array_type:

        def setna_datetime_date(arr, ind, int_nan_const=0):  # pragma: no cover
            # Set the actual value to a valid date (i.e. 1970, 1, 1)
            arr._data[ind] = (1970 << 32) + (1 << 16) + 1
            bodo.libs.int_arr_ext.set_bit_to_arr(arr._null_bitmap, ind, 0)

        return setna_datetime_date

    # Add support for datetime.timedelta array
    if arr == datetime_timedelta_array_type:

        def setna_datetime_timedelta(arr, ind, int_nan_const=0):  # pragma: no cover
            bodo.libs.array_kernels.setna(arr._days_data, ind)
            bodo.libs.array_kernels.setna(arr._seconds_data, ind)
            bodo.libs.array_kernels.setna(arr._microseconds_data, ind)
            bodo.libs.int_arr_ext.set_bit_to_arr(arr._null_bitmap, ind, 0)

        return setna_datetime_timedelta

    return lambda arr, ind, int_nan_const=0: None  # pragma: no cover


def setna_tup(arr_tup, ind, int_nan_const=0):  # pragma: no cover
    for arr in arr_tup:
        arr[ind] = np.nan


@overload(setna_tup, no_unliteral=True)
def overload_setna_tup(arr_tup, ind, int_nan_const=0):
    count = arr_tup.count

    func_text = "def f(arr_tup, ind, int_nan_const=0):\n"
    for i in range(count):
        func_text += "  setna(arr_tup[{}], ind, int_nan_const)\n".format(i)
    func_text += "  return\n"

    loc_vars = {}
    exec(func_text, {"setna": setna}, loc_vars)
    impl = loc_vars["f"]
    return impl


def setna_slice(arr, s):  # pragma: no cover
    arr[s] = np.nan


@overload(setna_slice, no_unliteral=True)
def overload_setna_slice(arr, s):
    """set all elements of array slice to NA"""

    def impl(arr, s):  # pragma: no cover
        arr_slice = numba.cpython.unicode._normalize_slice(s, len(arr))
        for i in range(arr_slice.start, arr_slice.stop, arr_slice.step):
            setna(arr, i)

    return impl


################################ median ####################################


ll.add_symbol("median_series_computation", quantile_alg.median_series_computation)


_median_series_computation = types.ExternalFunction(
    "median_series_computation",
    types.void(
        types.voidptr, bodo.libs.array.array_info_type, types.bool_, types.bool_
    ),
)


@numba.njit
def median_series_computation(res, arr, is_parallel, skipna):  # pragma: no cover
    arr_info = array_to_info(arr)
    _median_series_computation(res, arr_info, is_parallel, skipna)
    check_and_propagate_cpp_exception()
    delete_info_decref_array(arr_info)


@numba.njit
def median(arr, skipna=True, parallel=False):  # pragma: no cover
    # TODO: check return types, e.g. float32 -> float32
    res = np.empty(1, types.float64)
    median_series_computation(res.ctypes, arr, parallel, skipna)
    return res[0]


################################ autocorr ####################################

ll.add_symbol("autocorr_series_computation", quantile_alg.autocorr_series_computation)


_autocorr_series_computation = types.ExternalFunction(
    "autocorr_series_computation",
    types.void(
        types.voidptr, bodo.libs.array.array_info_type, types.int64, types.bool_
    ),
)


@numba.njit
def autocorr_series_computation(res, arr, lag, is_parallel):  # pragma: no cover
    arr_info = array_to_info(arr)
    _autocorr_series_computation(res, arr_info, lag, is_parallel)
    check_and_propagate_cpp_exception()
    delete_info_decref_array(arr_info)


@numba.njit
def autocorr(arr, lag=1, parallel=False):  # pragma: no cover
    # TODO: check return types, e.g. float32 -> float32
    res = np.empty(1, types.float64)
    autocorr_series_computation(res.ctypes, arr, lag, parallel)
    return res[0]


####################### series monotonicity ####################################

ll.add_symbol("compute_series_monotonicity", quantile_alg.compute_series_monotonicity)


_compute_series_monotonicity = types.ExternalFunction(
    "compute_series_monotonicity",
    types.void(
        types.voidptr, bodo.libs.array.array_info_type, types.int64, types.bool_
    ),
)


@numba.njit
def series_monotonicity_call(res, arr, inc_dec, is_parallel):  # pragma: no cover
    arr_info = array_to_info(arr)
    _compute_series_monotonicity(res, arr_info, inc_dec, is_parallel)
    check_and_propagate_cpp_exception()
    delete_info_decref_array(arr_info)


@numba.njit
def series_monotonicity(arr, inc_dec, parallel=False):  # pragma: no cover
    # TODO: check return types, e.g. float32 -> float32
    res = np.empty(1, types.float64)
    series_monotonicity_call(res.ctypes, arr, inc_dec, parallel)
    is_correct = res[0] > 0.5
    return is_correct


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
    ret = builder.call(fn, call_args)
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return ret


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
    ret = builder.call(fn, call_args)
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return ret


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
    # TODO [BE-414]: Move to C++ and handle NA in each val
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


def sample_table_operation(data, ind_arr, n, frac, replace, parallel=False):
    return data, ind_arr


@overload(sample_table_operation, no_unliteral=True)
def overload_sample_table_operation(data, ind_arr, n, frac, replace, parallel=False):
    """This is the code calling the C++ function for the sampling procedure.
    Parameters passed in argument are:
    ---the number of rows.
    ---The fraction used (-1 if the number of rows is used.
    ---Whether to allow collision of values."""
    count = len(data)

    func_text = "def impl(data, ind_arr, n, frac, replace, parallel=False):\n"
    func_text += "  info_list_total = [{}, array_to_info(ind_arr)]\n".format(
        ", ".join("array_to_info(data[{}])".format(x) for x in range(count))
    )
    func_text += "  table_total = arr_info_list_to_table(info_list_total)\n"
    func_text += (
        "  out_table = sample_table(table_total, n, frac, replace, parallel)\n".format(
            count
        )
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
    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "bodo": bodo,
            "array_to_info": array_to_info,
            "sample_table": sample_table,
            "arr_info_list_to_table": arr_info_list_to_table,
            "info_from_table": info_from_table,
            "info_to_array": info_to_array,
            "delete_table": delete_table,
            "delete_table_decref_arrays": delete_table_decref_arrays,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


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
    func_text += "  out_table = drop_duplicates_table(table_total, parallel, {}, keep_i, -1)\n".format(
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
    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "bodo": bodo,
            "array_to_info": array_to_info,
            "drop_duplicates_table": drop_duplicates_table,
            "arr_info_list_to_table": arr_info_list_to_table,
            "info_from_table": info_from_table,
            "info_to_array": info_to_array,
            "delete_table": delete_table,
            "delete_table_decref_arrays": delete_table_decref_arrays,
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
    func_text += "  for i in range(old_len):\n"
    func_text += "    if {}:\n".format(isna_check)
    func_text += "      new_len += 1\n"
    # allocate new arrays
    for i, out in enumerate(out_names):
        # Add a check for categorical, if so use data[{i}].dtype
        if isinstance(data[i], bodo.CategoricalArrayType):
            func_text += "  {0} = bodo.utils.utils.alloc_type(new_len, data[{1}], (-1,))\n".format(
                out, i
            )
        else:
            func_text += (
                "  {0} = bodo.utils.utils.alloc_type(new_len, t{1}, (-1,))\n".format(
                    out, i
                )
            )
    func_text += "  curr_ind = 0\n"
    func_text += "  for i in range(old_len):\n"
    func_text += "    if {}:\n".format(isna_check)
    for i in range(n_data_arrs):
        func_text += "      if isna(data[{}], i):\n".format(i)
        func_text += "        setna({}, curr_ind)\n".format(out_names[i])
        func_text += "      else:\n"
        func_text += "        {}[curr_ind] = data[{}][i]\n".format(out_names[i], i)
    func_text += "      curr_ind += 1\n"
    func_text += "  return {}\n".format(", ".join(out_names))
    loc_vars = {}
    # pass data types to generated code
    _globals = {"t{}".format(i): t for i, t in enumerate(data.types)}
    _globals.update(
        {
            "isna": isna,
            "setna": setna,
            "init_nested_counts": bodo.utils.indexing.init_nested_counts,
            "add_nested_counts": bodo.utils.indexing.add_nested_counts,
            "bodo": bodo,
        }
    )
    exec(func_text, _globals, loc_vars)
    _dropna_imp = loc_vars["_dropna_imp"]
    return _dropna_imp


def get(arr, ind):  # pragma: no cover
    return pd.Series(arr).str.get(ind)


@overload(get, no_unliteral=True)
def overload_get(arr, ind):

    if isinstance(arr, ArrayItemArrayType):
        arr_typ = arr.dtype
        out_dtype = arr_typ.dtype

        def get_arr_item(arr, ind):
            n = len(arr)
            nested_counts = init_nested_counts(out_dtype)
            for k in range(n):
                if bodo.libs.array_kernels.isna(arr, k):
                    continue
                val = arr[k]
                if not (len(val) > ind >= -len(val)) or bodo.libs.array_kernels.isna(
                    val, ind
                ):
                    continue
                nested_counts = add_nested_counts(nested_counts, val[ind])
            out_arr = bodo.utils.utils.alloc_type(n, arr_typ, nested_counts)
            for j in range(n):
                if bodo.libs.array_kernels.isna(arr, j):
                    setna(out_arr, j)
                    continue
                val = arr[j]
                if not (len(val) > ind >= -len(val)) or bodo.libs.array_kernels.isna(
                    val, ind
                ):
                    setna(out_arr, j)
                    continue

                out_arr[j] = val[ind]
            return out_arr

        return get_arr_item


def concat(arr_list):  # pragma: no cover
    return pd.concat(arr_list)


@overload(concat, no_unliteral=True)
def concat_overload(arr_list):

    # array(item) arrays
    if isinstance(arr_list, (types.UniTuple, types.List)) and isinstance(
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
            out_offsets = np.empty(num_lists + 1, offset_type)
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

    # datetime.date array
    if (
        isinstance(arr_list, (types.UniTuple, types.List))
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

    # datetime.timedelta array
    if (
        isinstance(arr_list, (types.UniTuple, types.List))
        and arr_list.dtype == datetime_timedelta_array_type
    ):

        def datetime_timedelta_array_concat_impl(arr_list):  # pragma: no cover
            tot_len = 0
            for A in arr_list:
                tot_len += len(A)
            Aret = bodo.hiframes.datetime_timedelta_ext.alloc_datetime_timedelta_array(
                tot_len
            )
            curr_pos = 0
            for A in arr_list:
                for i in range(len(A)):
                    Aret._days_data[i + curr_pos] = A._days_data[i]
                    Aret._seconds_data[i + curr_pos] = A._seconds_data[i]
                    Aret._microseconds_data[i + curr_pos] = A._microseconds_data[i]
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(A._null_bitmap, i)
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        Aret._null_bitmap, i + curr_pos, bit
                    )
                curr_pos += len(A)

            return Aret

        return datetime_timedelta_array_concat_impl

    # Decimal array
    if isinstance(arr_list, (types.UniTuple, types.List)) and isinstance(
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

    # all string input case
    # TODO: handle numerics to string casting case
    if (
        isinstance(arr_list, (types.UniTuple, types.List))
        and arr_list.dtype == string_array_type
    ):

        def string_concat_impl(arr_list):  # pragma: no cover
            # preallocate the output
            num_strs = 0
            num_chars = 0
            for A in arr_list:
                arr = A
                num_strs += len(arr)
                num_chars += bodo.libs.str_arr_ext.num_total_chars(arr)
            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(num_strs, num_chars)
            bodo.libs.str_arr_ext.set_null_bits_to_value(out_arr, -1)
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

    # Integer array input, or mix of Integer array and Numpy int array
    if (
        isinstance(arr_list, (types.UniTuple, types.List))
        and isinstance(arr_list.dtype, IntegerArrayType)
        or (
            isinstance(arr_list, types.BaseTuple)
            and all(isinstance(t.dtype, types.Integer) for t in arr_list.types)
            and any(isinstance(t, IntegerArrayType) for t in arr_list.types)
        )
    ):

        def impl_int_arr_list(arr_list):
            arr_list_converted = convert_to_nullable_tup(arr_list)
            all_data = []
            n_all = 0
            for A in arr_list_converted:
                all_data.append(A._data)
                n_all += len(A)
            out_data = bodo.libs.array_kernels.concat(all_data)
            n_bytes = (n_all + 7) >> 3
            new_mask = np.empty(n_bytes, np.uint8)
            curr_bit = 0
            for A in arr_list_converted:
                old_mask = A._null_bitmap
                for j in range(len(A)):
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, j)
                    bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
                    curr_bit += 1
            return bodo.libs.int_arr_ext.init_integer_array(
                out_data,
                new_mask,
            )

        return impl_int_arr_list

    # Boolean array input, or mix of Numpy and nullable boolean
    if (
        isinstance(arr_list, (types.UniTuple, types.List))
        and arr_list.dtype == boolean_array
        or (
            isinstance(arr_list, types.BaseTuple)
            and all(t.dtype == types.bool_ for t in arr_list.types)
            and any(t == boolean_array for t in arr_list.types)
        )
    ):
        # TODO: refactor to avoid duplication with integer array
        def impl_bool_arr_list(arr_list):
            arr_list_converted = convert_to_nullable_tup(arr_list)
            all_data = []
            n_all = 0
            for A in arr_list_converted:
                all_data.append(A._data)
                n_all += len(A)
            out_data = bodo.libs.array_kernels.concat(all_data)
            n_bytes = (n_all + 7) >> 3
            new_mask = np.empty(n_bytes, np.uint8)
            curr_bit = 0
            for A in arr_list_converted:
                old_mask = A._null_bitmap
                for j in range(len(A)):
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, j)
                    bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
                    curr_bit += 1
            return bodo.libs.bool_arr_ext.init_bool_array(out_data, new_mask)

        return impl_bool_arr_list

    # categorical arrays
    if isinstance(arr_list, (types.UniTuple, types.List)) and isinstance(
        arr_list.dtype, CategoricalArrayType
    ):

        def cat_array_concat_impl(arr_list):  # pragma: no cover
            new_code_arrs = []
            for A in arr_list:
                new_code_arrs.append(A.codes)
            return init_categorical_array(
                bodo.libs.array_kernels.concat(new_code_arrs), arr_list[0].dtype
            )

        return cat_array_concat_impl

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
            raise_bodo_error("concat of array types {} not supported".format(arr_list))

    # numpy array input
    return lambda arr_list: np.concatenate(arr_list)


def astype_float_tup(arr_tup):
    return tuple(t.astype(np.float64) for t in arr_tup)


@overload(astype_float_tup, no_unliteral=True)
def overload_astype_float_tup(arr_tup):
    """converts a tuple of arrays to float arrays using array.astype(np.float64)"""
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


def convert_to_nullable_tup(arr_tup):
    return arr_tup


@overload(convert_to_nullable_tup, no_unliteral=True)
def overload_convert_to_nullable_tup(arr_tup):
    """converts a tuple of integer/bool arrays to nullable integer/bool arrays with
    common dtype
    """
    # no need for conversion if already nullable int
    if isinstance(arr_tup, (types.UniTuple, types.List)) and isinstance(
        arr_tup.dtype, (IntegerArrayType, BooleanArrayType)
    ):
        return lambda arr_tup: arr_tup  # pragma: no cover

    assert isinstance(arr_tup, types.BaseTuple)
    count = len(arr_tup.types)
    comm_dtype = find_common_np_dtype(arr_tup.types)
    out_dtype = None
    astype_str = ""
    if isinstance(comm_dtype, types.Integer):
        out_dtype = bodo.libs.int_arr_ext.IntDtype(comm_dtype)
        astype_str = ".astype(out_dtype, False)"

    func_text = "def f(arr_tup):\n"
    func_text += "  return ({}{})\n".format(
        ",".join(
            "bodo.utils.conversion.coerce_to_array(arr_tup[{}], use_nullable_array=True){}".format(
                i, astype_str
            )
            for i in range(count)
        ),
        "," if count == 1 else "",
    )  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "out_dtype": out_dtype}, loc_vars)
    convert_impl = loc_vars["f"]
    return convert_impl


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
        uniq_A = bodo.libs.array_kernels.unique(A, parallel=True)
        loc_nuniq = len(uniq_A)
        return bodo.libs.distributed_api.dist_reduce(loc_nuniq, np.int32(sum_op))

    return nunique_par


def unique(A, parallel=False):  # pragma: no cover
    return np.array([a for a in set(A)]).astype(A.dtype)


def cummin(A):  # pragma: no cover
    return A


@overload(cummin, no_unliteral=True)
def cummin_overload(A):
    if isinstance(A.dtype, types.Float):
        neutral_val = np.finfo(A.dtype(1).dtype).max
    else:  # TODO: Add support for dates
        neutral_val = np.iinfo(A.dtype(1).dtype).max
    # No parallel code here. This cannot be done via parfor usual stuff but instead
    # by the more complicated mpi_exscan

    def impl(A):
        n = len(A)
        out_arr = np.empty(n, A.dtype)
        curr_cumulative = neutral_val
        for i in range(n):
            curr_cumulative = min(curr_cumulative, A[i])
            out_arr[i] = curr_cumulative
        return out_arr

    return impl


def cummax(A):  # pragma: no cover
    return A


@overload(cummax, no_unliteral=True)
def cummax_overload(A):
    if isinstance(A.dtype, types.Float):
        neutral_val = np.finfo(A.dtype(1).dtype).min
    else:  # TODO: Add support for dates
        neutral_val = np.iinfo(A.dtype(1).dtype).min
    # No parallel code here. This cannot be done via parfor usual stuff but instead
    # by the more complicated mpi_exscan

    def impl(A):
        n = len(A)
        out_arr = np.empty(n, A.dtype)
        curr_cumulative = neutral_val
        for i in range(n):
            curr_cumulative = max(curr_cumulative, A[i])
            out_arr[i] = curr_cumulative
        return out_arr

    return impl


@overload(unique, no_unliteral=True)
def unique_overload(A, parallel=False):
    def unique_impl(A, parallel=False):  # pragma: no cover
        input_table = arr_info_list_to_table([array_to_info(A)])
        n_key = 1
        keep_i = 0
        out_table = drop_duplicates_table(input_table, parallel, n_key, keep_i, -1)
        out_arr = info_to_array(info_from_table(out_table, 0), A)
        delete_table(input_table)
        delete_table(out_table)
        return out_arr

    return unique_impl


def explode(arr, index_arr):  # pragma: no cover
    return pd.Series(arr, index_arr).explode()


@overload(explode, no_unliteral=True)
def overload_explode(arr, index_arr):
    """
    Internal kernel for Series.explode(). Transforms each item in array(item) array into
    its own row, replicating the index values. Each empty array will have an NA in
    output.
    """
    assert isinstance(arr, ArrayItemArrayType) or arr == string_array_split_view_type
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
                setna(out_arr, curr_item)
                out_index_arr[curr_item] = index_arr[i]
                curr_item += 1
                continue
            arr_item = arr[i]
            n_items = len(arr_item)
            if n_items == 0:
                setna(out_arr, curr_item)
                out_index_arr[curr_item] = index_arr[i]
                curr_item += 1
                continue
            out_arr[curr_item : curr_item + n_items] = arr_item
            out_index_arr[curr_item : curr_item + n_items] = index_arr[i]
            curr_item += n_items

        return out_arr, out_index_arr

    return impl


def explode_str_split(arr, pat, n, index_arr):  # pragma: no cover
    return pd.Series(arr, index_arr).str.split(pat, n).explode()


@overload(explode_str_split, no_unliteral=True)
def overload_explode_str_split(arr, pat, n, index_arr):
    """
    Internal kernel for optimizing Series.str.split().explode(). Splits
    each string and assigns each portion to its own row, replicating the
    index values for a given split.
    """
    assert arr == string_array_type
    index_arr_type = index_arr
    index_dtype = index_arr_type.dtype

    def impl(arr, pat, n, index_arr):  # pragma: no cover
        is_regex = pat is not None and len(pat) > 1
        if is_regex:
            compiled_pat = re.compile(pat)
            if n == -1:
                n = 0
        else:
            if n == 0:
                n = -1
        l = len(arr)
        num_strs = 0
        num_chars = 0
        nested_index_counts = init_nested_counts(index_dtype)
        for i in range(l):
            ind_val = index_arr[i]
            if bodo.libs.array_kernels.isna(arr, i):
                num_strs += 1
                nested_index_counts = add_nested_counts(nested_index_counts, ind_val)
                continue
            if is_regex:
                vals = compiled_pat.split(arr[i], maxsplit=n)
            else:
                vals = arr[i].split(pat, n)
            num_strs += len(vals)
            for s in vals:
                nested_index_counts = add_nested_counts(nested_index_counts, ind_val)
                num_chars += bodo.libs.str_arr_ext.get_utf8_size(s)
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(num_strs, num_chars)
        out_index_arr = bodo.utils.utils.alloc_type(
            num_strs, index_arr_type, nested_index_counts
        )

        # Calculate the index offset from j
        idx = 0
        for j in range(l):
            if isna(arr, j):
                out_arr[idx] = ""
                bodo.libs.array_kernels.setna(out_arr, idx)
                out_index_arr[idx] = index_arr[j]
                idx += 1
                continue
            if is_regex:
                vals = compiled_pat.split(arr[j], maxsplit=n)
            else:
                vals = arr[j].split(pat, n)
            n_str = len(vals)
            out_arr[idx : idx + n_str] = vals
            out_index_arr[idx : idx + n_str] = index_arr[j]
            idx += n_str

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
        arr = arr.instance_type

    dtype = arr.dtype

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

    def impl(n, arr):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        out_arr = bodo.utils.utils.alloc_type(n, arr, (0,))
        for i in numba.parfors.parfor.internal_prange(n):
            setna(out_arr, i)
        return out_arr

    return impl


def gen_na_array_equiv(self, scope, equiv_set, loc, args, kws):  # pragma: no cover
    """Array analysis function for gen_na_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_array_kernels_gen_na_array = gen_na_array_equiv


def resize_and_copy(A, new_len):  # pragma: no cover
    return A


@overload(resize_and_copy, no_unliteral=True)
def overload_resize_and_copy(A, old_size, new_len):
    """allocate a new array (same type as 'A') and copy data of array 'A'"""

    _dtype = A

    # faster version for characters of string arrays
    if A == types.Array(types.uint8, 1, "C"):

        def impl_char(A, old_size, new_len):  # pragma: no cover
            out_arr = bodo.utils.utils.alloc_type(new_len, _dtype)
            # NOTE: direct memcpy using str_copy_ptr is slightly faster than slice copy
            # out_arr[:old_size] = A[:old_size]
            bodo.libs.str_arr_ext.str_copy_ptr(out_arr.ctypes, 0, A.ctypes, old_size)
            return out_arr

        return impl_char

    # generic copy for const sized arrays
    def impl(A, old_size, new_len):  # pragma: no cover
        out_arr = bodo.utils.utils.alloc_type(new_len, _dtype, (-1,))
        out_arr[:old_size] = A[:old_size]
        return out_arr

    return impl


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
        raise BodoError("parallel arange with types {}".format(args))


numba.parfors.parfor.replace_functions_map[("arange", "numpy")] = arange_parallel_impl


def sort(arr, ascending, inplace):  # pragma: no cover
    return np.sort(arr)


# Our sort implementation for series and dataframe relies on the inclusion
# of a sort IR node. This sort kernel is for use when that is not appropriate.
# For example: Inside of sklearn functions.
@overload(sort, no_unliteral=True)
def overload_sort(arr, ascending, inplace):
    def impl(arr, ascending, inplace):  # pragma: no cover
        n = len(arr)
        data = (np.arange(n),)
        key_arrs = (arr,)
        if not inplace:
            key_arrs = (arr.copy(),)

        l_key_arrs = bodo.libs.str_arr_ext.to_string_list(key_arrs)
        l_data = bodo.libs.str_arr_ext.to_string_list(data, True)
        bodo.libs.timsort.sort(l_key_arrs, 0, n, l_data)
        if not ascending:
            bodo.libs.timsort.reverseRange(l_key_arrs, 0, n, l_data)
        bodo.libs.str_arr_ext.cp_str_list_to_array(key_arrs, l_key_arrs)
        return key_arrs[0]

    return impl


###### Overloads of np array operations on our array types. ######


@overload(np.max, inline="always", no_unliteral=True)
@overload(max, inline="always", no_unliteral=True)
def overload_array_max(A):
    if isinstance(A, IntegerArrayType) or A == boolean_array:

        def impl(A):  # pragma: no cover
            return pd.Series(A).max()

        return impl


@overload(np.min, inline="always", no_unliteral=True)
@overload(min, inline="always", no_unliteral=True)
def overload_array_min(A):
    if isinstance(A, IntegerArrayType) or A == boolean_array:

        def impl(A):  # pragma: no cover
            return pd.Series(A).min()

        return impl


@overload(np.sum, inline="always", no_unliteral=True)
@overload(sum, inline="always", no_unliteral=True)
def overload_array_sum(A):
    if isinstance(A, IntegerArrayType) or A == boolean_array:

        def impl(A):  # pragma: no cover
            return pd.Series(A).sum()

    return impl


@overload(np.prod, inline="always", no_unliteral=True)
def overload_array_prod(A):
    if isinstance(A, IntegerArrayType) or A == boolean_array:

        def impl(A):  # pragma: no cover
            return pd.Series(A).prod()

    return impl


def nonzero(arr):
    return (arr,)


@overload(nonzero, no_unliteral=True)
def nonzero_overload(A, parallel=False):
    if not bodo.utils.utils.is_array_typ(A, False):  # pragma: no cover
        return

    def impl(A, parallel=False):  # pragma: no cover
        n = len(A)
        if parallel:
            offset = bodo.libs.distributed_api.dist_exscan(n, Reduce_Type.Sum.value)
        else:
            offset = 0
        result = []
        for i in range(n):
            if A[i]:
                result.append(i + offset)
        return (np.array(result, np.int64),)

    return impl


@overload(np.sort, inline="always", no_unliteral=True)
def np_sort(A, axis=-1, kind=None, order=None):
    if not bodo.utils.utils.is_array_typ(A, False) or isinstance(
        A, types.Array
    ):  # pragma: no cover
        return
    args_dict = {
        "axis": axis,
        "kind": kind,
        "order": order,
    }
    args_default_dict = {
        "axis": -1,
        "kind": None,
        "order": None,
    }
    check_unsupported_args("np.sort", args_dict, args_default_dict, "numpy")

    def impl(A, axis=-1, kind=None, order=None):  # pragma: no cover

        return pd.Series(A).sort_values().values

    return impl


def repeat_kernel(A, repeats):
    return A


@overload(repeat_kernel, no_unliteral=True)
def repeat_kernel_overload(A, repeats):
    """kernel for repeating array values (for Series.repeat)"""
    _dtype = A

    # int case
    if isinstance(repeats, types.Integer):

        def impl_int(A, repeats):  # pragma: no cover
            # TODO(Nick): Add a check that repeats > 0
            l = len(A)
            out_arr = bodo.utils.utils.alloc_type(l * repeats, _dtype, (-1,))
            for i in range(l):
                idx = i * repeats
                if bodo.libs.array_kernels.isna(A, i):
                    for j in range(repeats):
                        bodo.libs.array_kernels.setna(out_arr, idx + j)
                else:
                    out_arr[idx : idx + repeats] = A[i]
            return out_arr

        return impl_int

    # array case
    def impl_arr(A, repeats):  # pragma: no cover
        l = len(A)
        # TODO(ehsan): Add a check to ensure non-negative repeat values
        out_arr = bodo.utils.utils.alloc_type(repeats.sum(), _dtype, (-1,))
        idx = 0
        for i in range(l):
            r = repeats[i]
            if bodo.libs.array_kernels.isna(A, i):
                for j in range(r):
                    bodo.libs.array_kernels.setna(out_arr, idx + j)
            else:
                out_arr[idx : idx + r] = A[i]
            idx += r
        return out_arr

    return impl_arr


@overload(np.repeat, inline="always", no_unliteral=True)
def np_repeat(A, repeats):
    if not bodo.utils.utils.is_array_typ(A, False) or isinstance(
        A, types.Array
    ):  # pragma: no cover
        return
    if not isinstance(repeats, types.Integer):  # pragma: no cover
        raise BodoError("Only integer type supported for repeats in np.repeat()")

    def impl(A, repeats):  # pragma: no cover
        return bodo.libs.array_kernels.repeat_kernel(A, repeats)

    return impl


@overload(np.unique, inline="always", no_unliteral=True)
def np_unique(A):
    if not bodo.utils.utils.is_array_typ(A, False) or isinstance(
        A, types.Array
    ):  # pragma: no cover
        return

    def impl(A):  # pragma: no cover
        arr_q = bodo.libs.array_kernels.unique(A)
        return bodo.allgatherv(arr_q, False)

    return impl


@overload(np.union1d, inline="always", no_unliteral=True)
def overload_union1d(A1, A2):
    if not bodo.utils.utils.is_array_typ(
        A1, False
    ) or not bodo.utils.utils.is_array_typ(
        A2, False
    ):  # pragma: no cover
        return

    # TODO(Nick): Fix this to be proper typechecking.
    # For example Union(IntArray(32) and IntArray(64)) should be ok
    if A1 != A2:  # pragma: no cover
        raise BodoError("Both arrays must be the same type in np.union1d()")

    def impl(A1, A2):  # pragma: no cover
        merged_array = bodo.libs.array_kernels.concat([A1, A2])
        unique_array = bodo.libs.array_kernels.unique(merged_array)
        return pd.Series(unique_array).sort_values().values

    return impl


# TODO(Nick): Add support for a parallel implementation
@overload(np.intersect1d, inline="always", no_unliteral=True)
def overload_intersect1d(A1, A2, assume_unique=False, return_indices=False):
    if not bodo.utils.utils.is_array_typ(
        A1, False
    ) or not bodo.utils.utils.is_array_typ(
        A2, False
    ):  # pragma: no cover
        return

    args_dict = {
        "assume_unique": assume_unique,
        "return_indices": return_indices,
    }
    args_default_dict = {
        "assume_unique": False,
        "return_indices": False,
    }
    check_unsupported_args("np.intersect1d", args_dict, args_default_dict, "numpy")

    # TODO(Nick): Fix this to be proper typechecking.
    # For example Intersect(IntArray(32) and IntArray(64)) should be ok
    if A1 != A2:  # pragma: no cover
        raise BodoError("Both arrays must be the same type in np.intersect1d()")
    if A1.ndim != 1 or A2.ndim != 1:
        raise BodoError("Only 1D arrays supported in np.intersect1d()")

    def impl(A1, A2, assume_unique=False, return_indices=False):  # pragma: no cover
        unique_A1 = bodo.libs.array_kernels.unique(A1)
        unique_A2 = bodo.libs.array_kernels.unique(A2)
        merged_array = bodo.libs.array_kernels.concat([unique_A1, unique_A2])
        sorted_array = pd.Series(merged_array).sort_values().values
        # Compare each element to the element in front of it. Duplicates means
        # that element is in both A1 and A2
        # TODO(Nick): check for nulls (errors in numpy due to equality)
        return slice_array_intersect1d(sorted_array)

    return impl


@register_jitable
def slice_array_intersect1d(arr):  # pragma: no cover
    mask = arr[1:] == arr[:-1]
    return arr[:-1][mask]


@overload(np.setdiff1d, inline="always", no_unliteral=True)
def overload_setdiff1d(A1, A2, assume_unique=False):
    if not bodo.utils.utils.is_array_typ(
        A1, False
    ) or not bodo.utils.utils.is_array_typ(
        A2, False
    ):  # pragma: no cover
        return

    args_dict = {
        "assume_unique": assume_unique,
    }
    args_default_dict = {
        "assume_unique": False,
    }
    check_unsupported_args("np.setdiff1d", args_dict, args_default_dict, "numpy")

    # TODO(Nick): Fix this to be proper typechecking.
    # For example Intersect(IntArray(32) and IntArray(64)) should be ok
    if A1 != A2:  # pragma: no cover
        raise BodoError("Both arrays must be the same type in np.setdiff1d()")
    if A1.ndim != 1 or A2.ndim != 1:
        raise BodoError("Only 1D arrays supported in np.setdiff1d()")

    def impl(A1, A2, assume_unique=False):  # pragma: no cover
        unique_A1 = bodo.libs.array_kernels.unique(A1)
        unique_A2 = bodo.libs.array_kernels.unique(A2)
        # TODO(Nick): Add a sorting option to match numpy, which
        # only does the sequential search if a heuristic is met.
        mask = calculate_mask_setdiff1d(unique_A1, unique_A2)
        return pd.Series(unique_A1[mask]).sort_values().values

    return impl


@register_jitable
def calculate_mask_setdiff1d(A1, A2):  # pragma: no cover
    mask = np.ones(len(A1), np.bool_)
    for i in range(len(A2)):
        mask &= A1 != A2[i]
    return mask


@overload(np.linspace, inline="always", no_unliteral=True)
def np_linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    # This kernel is also supported in Numba but without kwargs. Based on our tests,
    # whenever any kwargs are passed in this kernel is selected. This kernel also seems to
    # be selected when only start, stop, and num are included, but that may not be reliable.

    args_dict = {"retstep": retstep, "axis": axis}
    args_default_dict = {"retstep": False, "axis": 0}
    check_unsupported_args("np.linspace", args_dict, args_default_dict, "numpy")
    # Numpy infers the dtype if one isn't provided. We perform the same check here.
    # https://github.com/numpy/numpy/blob/92ebe1e9a6aeb47a881a1226b08218175776f9ea/numpy/core/function_base.py#L123
    is_int_impl = False
    if is_overload_none(dtype):
        # Check start, stop types and compare to float64 type for step
        _dtype = np.promote_types(
            np.promote_types(
                numba.np.numpy_support.as_dtype(start),
                numba.np.numpy_support.as_dtype(stop),
            ),
            numba.np.numpy_support.as_dtype(types.float64),
        ).type
    else:
        # Providing an int dtype has different behavior.
        # See 1.20 changes in: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace
        if isinstance(dtype.dtype, types.Integer):
            is_int_impl = True
        _dtype = numba.np.numpy_support.as_dtype(dtype).type

    # TODO(Nick): Add more detailed type checking
    if is_int_impl:

        def impl_int(
            start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
        ):  # pragma: no cover
            step_size = np_linspace_get_stepsize(start, stop, num, endpoint)
            numba.parfors.parfor.init_prange()
            out_arr = np.empty(num, _dtype)
            for i in numba.parfors.parfor.internal_prange(num):
                out_arr[i] = _dtype(np.floor(start + i * step_size))
            return out_arr

        return impl_int

    else:

        def impl(
            start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
        ):  # pragma: no cover
            step_size = np_linspace_get_stepsize(start, stop, num, endpoint)
            numba.parfors.parfor.init_prange()
            out_arr = np.empty(num, _dtype)
            for i in numba.parfors.parfor.internal_prange(num):
                out_arr[i] = _dtype(start + i * step_size)
            return out_arr

        return impl


def np_linspace_get_stepsize(start, stop, num, endpoint):  # pragma: no cover
    """Helper kernel to keep the if statements from inlining"""
    return 0


@overload(np_linspace_get_stepsize, no_unliteral=True)
def overload_np_linspace_get_stepsize(start, stop, num, endpoint):
    def impl(start, stop, num, endpoint):  # pragma: no cover
        if num < 0:
            raise ValueError("np.linspace() Num must be >= 0")
        if endpoint:
            num -= 1
        if num > 1:
            return (stop - start) / num
        return 0

    return impl


@overload(operator.contains, no_unliteral=True)
def arr_contains(A, val):
    # TODO: Add support for types with different width. i.e. int64 and int16
    if not (
        bodo.utils.utils.is_array_typ(A, False) and A.dtype == types.unliteral(val)
    ):  # pragma: no cover
        return

    def impl(A, val):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        count = 0
        n = len(A)
        for i in numba.parfors.parfor.internal_prange(n):
            if not bodo.libs.array_kernels.isna(A, i):
                count += A[i] == val
        return count > 0

    return impl


@overload(np.any, inline="always", no_unliteral=True)
def np_any(A, axis=None, out=None, keepdims=None):
    if not (
        bodo.utils.utils.is_array_typ(A, False) and A.ndim == 1
    ):  # pragma: no cover
        return

    args_dict = {"axis": axis, "out": out, "keepdims": keepdims}
    args_default_dict = {"axis": None, "out": None, "keepdims": None}
    check_unsupported_args("np.any", args_dict, args_default_dict, "numpy")

    def impl(A, axis=None, out=None, keepdims=None):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        count = 0
        n = len(A)
        for i in numba.parfors.parfor.internal_prange(n):
            if not bodo.libs.array_kernels.isna(A, i):
                count += int(bool(A[i]))
        return count > 0

    return impl


@overload(np.all, inline="always", no_unliteral=True)
def np_all(A, axis=None, out=None, keepdims=None):
    if not (
        bodo.utils.utils.is_array_typ(A, False) and A.ndim == 1
    ):  # pragma: no cover
        return

    args_dict = {"axis": axis, "out": out, "keepdims": keepdims}
    args_default_dict = {"axis": None, "out": None, "keepdims": None}
    check_unsupported_args("np.any", args_dict, args_default_dict, "numpy")

    def impl(A, axis=None, out=None, keepdims=None):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        count = 0
        n = len(A)
        for i in numba.parfors.parfor.internal_prange(n):
            if not bodo.libs.array_kernels.isna(A, i):
                count += int(bool(A[i]))
        return count == n

    return impl


@overload(np.cbrt, inline="always", no_unliteral=True)
def np_cbrt(
    A, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    """cbrt implementation on 1D Numeric arrays and our Integer Array."""
    # TODO: Remove when Numba ufunc gets merged https://github.com/numba/numba/pull/6075
    # This kernel should no longer be necessary because it should come with ufunc support

    if not (
        isinstance(A, types.Number)
        or (
            bodo.utils.utils.is_array_typ(A, False)
            and A.ndim == 1
            and isinstance(A.dtype, types.Number)
        )
    ):  # pragma: no cover
        return

    args_dict = {
        "out": out,
        "where": where,
        "casting": casting,
        "order": order,
        "dtype": dtype,
        "subok": subok,
    }
    args_default_dict = {
        "out": None,
        "where": True,
        "casting": "same_kind",
        "order": "K",
        "dtype": None,
        "subok": True,
    }
    check_unsupported_args("np.cbrt", args_dict, args_default_dict, "numpy")

    if bodo.utils.utils.is_array_typ(A, False):
        # Reuse scalar implementation on each element

        # TODO: Small integer types are not currently correct, because Numpy will use a smaller
        # float type (float16) when possible
        _out_dtype = np.promote_types(
            numba.np.numpy_support.as_dtype(A.dtype),
            numba.np.numpy_support.as_dtype(types.float32),
        ).type

        def impl_arr(
            A,
            out=None,
            where=True,
            casting="same_kind",
            order="K",
            dtype=None,
            subok=True,
        ):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            n = len(A)
            out_arr = np.empty(n, _out_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(A, i):
                    bodo.libs.array_kernels.setna(out_arr, i)
                    continue
                out_arr[i] = np_cbrt_scalar(A[i], _out_dtype)
            return out_arr

        return impl_arr

    # TODO: Small integer types are not currently correct, because Numpy will use a smaller
    # float type (float16) when possible
    _out_dtype = np.promote_types(
        numba.np.numpy_support.as_dtype(A),
        numba.np.numpy_support.as_dtype(types.float32),
    ).type

    def impl_scalar(
        A, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
    ):  # pragma: no cover
        # Call a helper function to avoid inlining control flow
        return np_cbrt_scalar(A, _out_dtype)

    return impl_scalar


# TODO: Remove when Numba ufunc gets merged https://github.com/numba/numba/pull/6075
# Numpy implementation
# https://github.com/numpy/numpy/blob/31ffdecf07d18ed4dbb66b171cb0f998d4b190fa/numpy/core/src/npymath/npy_math_internal.h.src#L513
@register_jitable
def np_cbrt_scalar(x, float_dtype):  # pragma: no cover
    if np.isnan(x):
        return np.nan
    flag = x < 0
    if flag:
        x = -x
    res = np.power(float_dtype(x), 1.0 / 3.0)
    if flag:
        return -res
    return res


# inlining manually instead of inline="always" since Numba's
# np.hstack overload for tuples fails.
# TODO(Ehsan): Fix in Numba
@overload(np.hstack, no_unliteral=True)
def np_hstack(tup):
    # Verify that arr_iter is a tuple, list of arrays, or Series of Arrays
    is_sequence = isinstance(tup, (types.BaseTuple, types.List))
    is_series = isinstance(
        tup, (bodo.SeriesType, bodo.hiframes.pd_series_ext.HeterogeneousSeriesType)
    ) and isinstance(tup.data, (types.BaseTuple, types.List))
    if isinstance(tup, types.BaseTuple):
        # Determine that each type is an array type
        for typ in tup.types:
            # TODO: Add proper checking for if the arrays can be merged
            is_sequence = is_sequence and bodo.utils.utils.is_array_typ(typ, False)
    elif isinstance(tup, types.List):
        is_sequence = bodo.utils.utils.is_array_typ(tup.dtype, False)
    elif is_series:
        for typ in tup.data.types:
            # TODO: Add proper checking for if the arrays can be merged
            is_series = is_series and bodo.utils.utils.is_array_typ(typ, False)

    if not (is_sequence or is_series):  # pragma: no cover
        return

    # If its a series return the implementation on the data
    if is_series:

        def impl_series(tup):  # pragma: no cover
            arr_tup = bodo.hiframes.pd_series_ext.get_series_data(tup)
            return bodo.libs.array_kernels.concat(arr_tup)

        return impl_series

    def impl(tup):  # pragma: no cover
        return bodo.libs.array_kernels.concat(tup)

    return impl


# Source code: https://github.com/numpy/numpy/blob/04b58d3ffbd2c8d30c36ae6ed6366f1069136c43/numpy/random/mtrand.pyx#L4060
@overload(np.random.multivariate_normal, inline="always", no_unliteral=True)
def np_random_multivariate_normal(mean, cov, size=None, check_valid="warn", tol=1e-8):
    args_dict = {"check_valid": check_valid, "tol": tol}
    args_default_dict = {"check_valid": "warn", "tol": 1e-8}
    check_unsupported_args(
        "np.random.multivariate_normal", args_dict, args_default_dict, "numpy"
    )

    # TODO: Numpy seems to support tuple as well
    # TODO: All None. This fails in the distributed analysis step for np.dot
    # (produces a replicated result)
    if not (isinstance(size, types.Integer)):  # pragma: no cover
        raise BodoError(
            "np.random.multivariate_normal() size argument is required and must be an integer"
        )

    # TODO: Should we support Pandas arrays.

    # Check the input types and shapes
    if not (
        bodo.utils.utils.is_array_typ(mean, False) and mean.ndim == 1
    ):  # pragma: no cover
        raise BodoError(
            "np.random.multivariate_normal() mean must be a 1 dimensional numpy array"
        )
    if not (
        bodo.utils.utils.is_array_typ(cov, False) and cov.ndim == 2
    ):  # pragma: no cover
        raise BodoError(
            "np.random.multivariate_normal() cov must be a 2 dimensional square, numpy array"
        )

    def impl(mean, cov, size=None, check_valid="warn", tol=1e-8):  # pragma: no cover
        # Check that cov is square. Moved to separate function for inlining
        # TODO: Replace with array analysis
        _validate_multivar_norm(cov)
        # Calculate the output shape
        N = mean.shape[0]
        # TODO: Support None and array like inputs
        final_shape = (size, N)
        standard_normal = np.random.standard_normal(final_shape)

        # Convert cov float64 to make tol meaningful
        # https://github.com/numpy/numpy/blob/04b58d3ffbd2c8d30c36ae6ed6366f1069136c43/numpy/random/mtrand.pyx#L4099
        cov = cov.astype(np.float64)

        # Compute the svd
        (u, s, v) = np.linalg.svd(cov)

        # TODO: Handle check_valid options by ensuring cov is positive-semidefinite
        # https://github.com/numpy/numpy/blob/04b58d3ffbd2c8d30c36ae6ed6366f1069136c43/numpy/random/mtrand.pyx#L4099

        res = np.dot(standard_normal, np.sqrt(s).reshape(N, 1) * v)
        output = res + mean
        # Do we need to set the shape? Not sure why Numpy does that.
        # https://github.com/numpy/numpy/blob/04b58d3ffbd2c8d30c36ae6ed6366f1069136c43/numpy/random/mtrand.pyx#L4120
        return output

    return impl


def _validate_multivar_norm(cov):  # pragma: no cover
    # Dummy function for overload
    return


@overload(_validate_multivar_norm, no_unliteral=True)
def _overload_validate_multivar_norm(cov):
    def impl(cov):  # pragma: no cover
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(
                "np.random.multivariate_normal() cov must be a 2 dimensional square, numpy array"
            )

    return impl


def _nan_argmin(arr):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(_nan_argmin, inline="always", no_unliteral=True)
def _overload_nan_argmin(arr):
    """
    Argmin function used on Bodo Array types for idxmin
    """
    # We check just the dtype because the previous function ensures
    # we are operating on 1D arrays

    if (
        isinstance(arr, IntegerArrayType)
        or arr in [boolean_array, datetime_date_array_type]
        or arr.dtype == bodo.timedelta64ns
    ):

        def impl_bodo_arr(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            init_val = bodo.hiframes.series_kernels._get_type_max_value(arr)
            ival = typing.builtins.IndexValue(-1, init_val)
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                if bodo.libs.array_kernels.isna(arr, i):
                    continue
                curr_ival = typing.builtins.IndexValue(i, arr[i])
                ival = min(ival, curr_ival)
            return ival.index

        return impl_bodo_arr

    if isinstance(arr, CategoricalArrayType):
        assert (
            arr.dtype.ordered
        ), "Categorical Array must be ordered to select an argmin"
        elem_width = bodo.hiframes.pd_categorical_ext.get_categories_int_type(arr.dtype)

        def impl_cat_arr(arr):  # pragma: no cover
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(arr)
            numba.parfors.parfor.init_prange()
            init_val = elem_width(len(arr.dtype.categories) + 1)
            ival = typing.builtins.IndexValue(-1, init_val)
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                if bodo.libs.array_kernels.isna(arr, i):
                    continue
                curr_ival = typing.builtins.IndexValue(i, codes[i])
                ival = min(ival, curr_ival)
            return ival.index

        return impl_cat_arr

    return lambda arr: arr.argmin()  # pragma: no cover


def _nan_argmax(arr):  # pragma: no cover
    # Dummy function used for overload
    return


# TODO: Figure out how if this should be inlined.
@overload(_nan_argmax, no_unliteral=True, inline="always")
def _overload_nan_argmax(arr):
    """
    Argmax function used on Bodo Array types for idxmax
    """
    # We check just the dtype because the previous function ensures
    # we are operating on 1D arrays

    if (
        isinstance(arr, IntegerArrayType)
        or arr in [boolean_array, datetime_date_array_type]
        or arr.dtype == bodo.timedelta64ns
    ):

        def impl_bodo_arr(arr):  # pragma: no cover
            n = len(arr)
            numba.parfors.parfor.init_prange()
            init_val = bodo.hiframes.series_kernels._get_type_min_value(arr)
            ival = typing.builtins.IndexValue(-1, init_val)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(arr, i):
                    continue
                curr_ival = typing.builtins.IndexValue(i, arr[i])
                ival = max(ival, curr_ival)
            return ival.index

        return impl_bodo_arr

    if isinstance(arr, CategoricalArrayType):
        assert (
            arr.dtype.ordered
        ), "Categorical Array must be ordered to select an argmin"
        elem_width = bodo.hiframes.pd_categorical_ext.get_categories_int_type(arr.dtype)

        def impl_cat_arr(arr):  # pragma: no cover
            n = len(arr)
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(arr)
            numba.parfors.parfor.init_prange()
            init_val = elem_width(-1)
            ival = typing.builtins.IndexValue(-1, init_val)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(arr, i):
                    continue
                curr_ival = typing.builtins.IndexValue(i, codes[i])
                ival = max(ival, curr_ival)
            return ival.index

        return impl_cat_arr

    # TODO: Does this need a parallel implementation?
    return lambda arr: arr.argmax()  # pragma: no cover


@overload_attribute(types.Array, "nbytes", inline="always")
def overload_dataframe_index(A):
    """get number of bytes in Numpy array"""
    # TODO(ehsan): contribute to Numba
    return lambda A: A.size * bodo.io.np_io.get_dtype_size(A.dtype)  # pragma: no cover
