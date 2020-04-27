# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Some kernels for Series related functions. This is a legacy file that needs to be
refactored.
"""
import numpy as np

import numba
from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.int_arr_ext import IntDtype


# TODO: series index and name
# float columns can have regular np.nan
def _column_filter_impl(B, ind):  # pragma: no cover
    dtype = bodo.hiframes.rolling.shift_dtype(B.dtype)
    A = np.empty(len(B), dtype)
    for i in numba.parfors.parfor.internal_prange(len(A)):
        if ind[i]:
            A[i] = B[i]
        else:
            bodo.ir.join.setitem_arr_nan(A, i)
    return A


def _column_count_impl(A):  # pragma: no cover
    numba.parfors.parfor.init_prange()
    count = 0
    for i in numba.parfors.parfor.internal_prange(len(A)):
        if not bodo.libs.array_kernels.isna(A, i):
            count += 1

    res = count
    return res


def _column_fillna_impl(A, B, fill):  # pragma: no cover
    for i in numba.parfors.parfor.internal_prange(len(A)):
        s = B[i]
        if bodo.libs.array_kernels.isna(B, i):
            s = fill
        A[i] = s


# using njit since 1D_var is broken for alloc when there is calculation of len
@numba.njit(no_cpython_wrapper=True)
def _series_dropna_str_alloc_impl_inner(B):  # pragma: no cover
    # TODO: test
    # TODO: generalize
    old_len = len(B)
    na_count = 0
    for i in range(len(B)):
        if bodo.libs.str_arr_ext.str_arr_is_na(B, i):
            na_count += 1
    # TODO: more efficient null counting
    new_len = old_len - na_count
    num_chars = bodo.libs.str_arr_ext.num_total_chars(B)
    A = bodo.libs.str_arr_ext.pre_alloc_string_array(new_len, num_chars)
    bodo.libs.str_arr_ext.copy_non_null_offsets(A, B)
    bodo.libs.str_arr_ext.copy_data(A, B)
    return A


# return the nan value for the type (handle dt64)
def _get_nan(val):  # pragma: no cover
    return np.nan


@overload(_get_nan)
def _get_nan_overload(val):
    """get NA value with same type as val
    """
    if isinstance(val, (types.NPDatetime, types.NPTimedelta)):
        nat = val("NaT")
        return lambda val: nat  # pragma: no cover

    if isinstance(val, types.Float):
        return lambda val: np.nan  # pragma: no cover

    # Just return same value for other types that don't have NA sentinel
    # This makes sure output type of Series.min/max with integer values don't get
    # converted to float unnecessarily
    return lambda val: val  # pragma: no cover


def _get_type_max_value(dtype):  # pragma: no cover
    return 0


@overload(_get_type_max_value)
def _get_type_max_value_overload(dtype):
    # pd.Int64Dtype(), etc.
    if isinstance(dtype, IntDtype):
        _dtype = dtype.dtype
        return lambda dtype: numba.cpython.builtins.get_type_max_value(_dtype)

    # dt64/td64
    if isinstance(dtype.dtype, (types.NPDatetime, types.NPTimedelta)):
        return lambda dtype: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
            numba.cpython.builtins.get_type_max_value(numba.core.types.int64)
        )

    if dtype.dtype == types.bool_:
        return lambda dtype: True

    return lambda dtype: numba.cpython.builtins.get_type_max_value(dtype)


def _get_type_min_value(dtype):  # pragma: no cover
    return 0


@overload(_get_type_min_value)
def _get_type_min_value_overload(dtype):
    # pd.Int64Dtype(), etc.
    if isinstance(dtype, IntDtype):
        _dtype = dtype.dtype
        return lambda dtype: numba.cpython.builtins.get_type_min_value(_dtype)

    # dt64/td64
    if isinstance(dtype.dtype, (types.NPDatetime, types.NPTimedelta)):
        return lambda dtype: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
            numba.cpython.builtins.get_type_min_value(numba.core.types.int64)
        )

    if dtype.dtype == types.bool_:
        return lambda dtype: False

    return lambda dtype: numba.cpython.builtins.get_type_min_value(dtype)


@overload(min)
def indval_min(a1, a2):
    if a1 == types.bool_ and a2 == types.bool_:

        def min_impl(a1, a2):  # pragma: no cover
            if a1 > a2:
                return a2
            return a1

        return min_impl


@overload(max)
def indval_max(a1, a2):
    if a1 == types.bool_ and a2 == types.bool_:

        def max_impl(a1, a2):  # pragma: no cover
            if a2 > a1:
                return a2
            return a1

        return max_impl


@numba.njit
def _sum_handle_nan(s, count):  # pragma: no cover
    if not count:
        s = bodo.hiframes.series_kernels._get_nan(s)
    return s


@numba.generated_jit
def get_float_nan(s):
    nan = np.nan
    if s == types.float32:
        nan = np.float32("nan")
    return lambda s: nan


@numba.njit
def _mean_handle_nan(s, count):  # pragma: no cover
    if not count:
        s = get_float_nan(s)
    else:
        s = s / count
    return s


@numba.njit
def _var_handle_nan(s, count):  # pragma: no cover
    if count <= 1:
        s = np.nan
    else:
        s = s / (count - 1)
    return s


@numba.njit
def lt_f(a, b):  # pragma: no cover
    return a < b


@numba.njit
def gt_f(a, b):  # pragma: no cover
    return a > b
