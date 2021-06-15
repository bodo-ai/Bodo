# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Implements array operations for usage by DataFrames and Series
such as count and max.
"""

import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.extending import overload

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.utils.typing import (
    element_type,
    is_hashable_type,
    is_iterable_type,
    is_overload_true,
    is_overload_zero,
)


@numba.njit
def array_op_median(arr, skipna=True, parallel=False):  # pragma: no cover
    # TODO: check return types, e.g. float32 -> float32
    res = np.empty(1, types.float64)
    bodo.libs.array_kernels.median_series_computation(res.ctypes, arr, parallel, skipna)
    return res[0]


def array_op_isna(arr):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_isna)
def overload_array_op_isna(arr):
    def impl(arr):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        n = len(arr)
        out_arr = np.empty(n, np.bool_)
        for i in numba.parfors.parfor.internal_prange(n):
            out_arr[i] = bodo.libs.array_kernels.isna(arr, i)
        return out_arr

    return impl


def array_op_count(arr):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_count)
def overload_array_op_count(arr):
    def impl(arr):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(arr)):
            count_val = 0
            if not bodo.libs.array_kernels.isna(arr, i):
                count_val = 1
            count += count_val

        res = count
        return res

    return impl


def array_op_describe(arr):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_describe)
def overload_array_op_describe(arr):
    # Pandas doesn't return std for describe of datetime64 data
    # https://github.com/pandas-dev/pandas/blob/059c8bac51e47d6eaaa3e36d6a293a22312925e6/pandas/core/describe.py#L328
    if arr.dtype == bodo.datetime64ns:

        def impl_dt(arr):  # pragma: no cover
            a_count = array_op_count(arr)
            a_min = array_op_min(arr)
            a_max = array_op_max(arr)
            a_mean = array_op_mean(arr)
            q25 = array_op_quantile(arr, 0.25)
            q50 = array_op_quantile(arr, 0.5)
            q75 = array_op_quantile(arr, 0.75)
            return (a_count, a_mean, a_min, q25, q50, q75, a_max)

        return impl_dt

    def impl(arr):  # pragma: no cover
        a_count = array_op_count(arr)
        a_min = array_op_min(arr)
        a_max = array_op_max(arr)
        a_mean = array_op_mean(arr)
        a_std = array_op_std(arr)
        q25 = array_op_quantile(arr, 0.25)
        q50 = array_op_quantile(arr, 0.5)
        q75 = array_op_quantile(arr, 0.75)
        return (a_count, a_mean, a_std, a_min, q25, q50, q75, a_max)

    return impl


def array_op_min(arr):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_min)
def overload_array_op_min(arr):
    if arr.dtype == bodo.timedelta64ns:

        def impl_td64(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_max_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(arr[i])
                    count_val = 1
                s = min(s, val)
                count += count_val
            return bodo.hiframes.pd_index_ext._tdi_val_finalize(s, count)

        return impl_td64

    if arr.dtype == bodo.datetime64ns:

        def impl_dt64(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_max_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    count_val = 1
                s = min(s, val)
                count += count_val
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

        return impl_dt64

    # categorical case
    if isinstance(arr, CategoricalArrayType):

        def impl_cat(arr):  # pragma: no cover
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(arr)
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_max_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(codes)):
                v = codes[i]
                if v == -1:
                    continue
                s = min(s, v)
                count += 1

            res = bodo.hiframes.series_kernels._box_cat_val(s, arr.dtype, count)
            return res

        return impl_cat

    # TODO: Setup datetime_date_array.dtype() so we can reuse impl
    if arr == datetime_date_array_type:

        def impl_date(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = bodo.hiframes.series_kernels._get_date_max_value()
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = arr[i]
                    count_val = 1
                s = min(s, val)
                count += count_val
            res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
            return res

        return impl_date

    def impl(arr):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        s = bodo.hiframes.series_kernels._get_type_max_value(arr.dtype)
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(arr)):
            val = s
            count_val = 0
            if not bodo.libs.array_kernels.isna(arr, i):
                val = arr[i]
                count_val = 1
            s = min(s, val)
            count += count_val
        res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
        return res

    return impl


def array_op_max(arr):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_max)
def overload_array_op_max(arr):
    if arr.dtype == bodo.timedelta64ns:

        def impl_td64(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_min_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(arr[i])
                    count_val = 1
                s = max(s, val)
                count += count_val
            return bodo.hiframes.pd_index_ext._tdi_val_finalize(s, count)

        return impl_td64

    if arr.dtype == bodo.datetime64ns:

        def impl_dt64(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_min_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    count_val = 1
                s = max(s, val)
                count += count_val
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

        return impl_dt64

    # categorical case
    if isinstance(arr, CategoricalArrayType):

        def impl_cat(arr):  # pragma: no cover
            codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(arr)
            numba.parfors.parfor.init_prange()
            s = -1
            # keeping track of NAs is not necessary for max since all valid codes are
            # greater than -1
            for i in numba.parfors.parfor.internal_prange(len(codes)):
                s = max(s, codes[i])

            res = bodo.hiframes.series_kernels._box_cat_val(s, arr.dtype, 1)
            return res

        return impl_cat

    # TODO: Setup datetime_date_array.dtype() so we can reuse impl
    if arr == datetime_date_array_type:

        def impl_date(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = bodo.hiframes.series_kernels._get_date_min_value()
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = arr[i]
                    count_val = 1
                s = max(s, val)
                count += count_val
            res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
            return res

        return impl_date

    def impl(arr):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        s = bodo.hiframes.series_kernels._get_type_min_value(arr.dtype)
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(arr)):
            val = s
            count_val = 0
            if not bodo.libs.array_kernels.isna(arr, i):
                val = arr[i]
                count_val = 1
            s = max(s, val)
            count += count_val
        res = bodo.hiframes.series_kernels._sum_handle_nan(s, count)
        return res

    return impl


def array_op_mean(arr):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_mean)
def overload_array_op_mean(arr):

    # datetime
    if arr.dtype == bodo.datetime64ns:

        def impl(arr):  # pragma: no cover
            return pd.Timestamp(
                types.int64(bodo.libs.array_ops.array_op_mean(arr.view(np.int64)))
            )

        return impl
    # see core/nanops.py/nanmean() for output types
    # TODO: more accurate port of dtypes from pandas
    sum_dtype = types.float64
    count_dtype = types.float64
    if isinstance(arr, types.Array) and arr.dtype == types.float32:
        sum_dtype = types.float32
        count_dtype = types.float32

    val_0 = sum_dtype(0)
    count_0 = count_dtype(0)
    count_1 = count_dtype(1)

    def impl(arr):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        s = val_0
        count = count_0
        for i in numba.parfors.parfor.internal_prange(len(arr)):
            val = val_0
            count_val = count_0
            if not bodo.libs.array_kernels.isna(arr, i):
                val = arr[i]
                count_val = count_1
            s += val
            count += count_val

        res = bodo.hiframes.series_kernels._mean_handle_nan(s, count)
        return res

    return impl


def array_op_var(arr, skipna, ddof):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_var)
def overload_array_op_var(arr, skipna, ddof):
    def impl(arr, skipna, ddof):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        first_moment = 0.0
        second_moment = 0.0
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(arr)):
            val = 0.0
            count_val = 0
            if not bodo.libs.array_kernels.isna(arr, i) or not skipna:
                val = arr[i]
                count_val = 1
            first_moment += val
            second_moment += val * val
            count += count_val

        s = second_moment - first_moment * first_moment / count
        res = bodo.hiframes.series_kernels._handle_nan_count_ddof(s, count, ddof)
        return res

    return impl


def array_op_std(arr, skipna=True, ddof=1):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_std)
def overload_array_op_std(arr, skipna=True, ddof=1):
    # datetime
    if arr.dtype == bodo.datetime64ns:

        def impl_dt64(arr, skipna=True, ddof=1):  # pragma: no cover
            return pd.Timedelta(
                types.int64(array_op_var(arr.view(np.int64), skipna, ddof) ** 0.5)
            )

        return impl_dt64
    return (
        lambda arr, skipna=True, ddof=1: array_op_var(arr, skipna, ddof) ** 0.5
    )  # pragma: no cover


def array_op_quantile(arr, q):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_quantile)
def overload_array_op_quantile(arr, q):
    if is_iterable_type(q):

        if arr.dtype == bodo.datetime64ns:

            def _impl_list_dt(arr, q):  # pragma: no cover
                out_arr = np.empty(len(q), np.int64)
                for i in range(len(q)):
                    q_val = np.float64(q[i])
                    out_arr[i] = bodo.libs.array_kernels.quantile(
                        arr.view(np.int64), q_val
                    )
                return out_arr.view(np.dtype("datetime64[ns]"))

            return _impl_list_dt

        def impl_list(arr, q):  # pragma: no cover
            out_arr = np.empty(len(q), np.float64)
            for i in range(len(q)):
                q_val = np.float64(q[i])
                out_arr[i] = bodo.libs.array_kernels.quantile(arr, q_val)
            return out_arr

        return impl_list

    if arr.dtype == bodo.datetime64ns:

        def _impl_dt(arr, q):  # pragma: no cover
            return pd.Timestamp(
                bodo.libs.array_kernels.quantile(arr.view(np.int64), np.float64(q))
            )

        return _impl_dt

    def impl(arr, q):  # pragma: no cover
        return bodo.libs.array_kernels.quantile(arr, np.float64(q))

    return impl


def array_op_sum(arr, skipna, min_count):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_sum, no_unliteral=True)
def overload_array_op_sum(arr, skipna, min_count):
    # TODO: arr that have different underlying data type than dtype
    # like records/tuples
    if isinstance(arr.dtype, types.Integer):
        retty = types.intp
    elif arr.dtype == types.bool_:
        retty = np.int64
    else:
        retty = arr.dtype
    val_zero = retty(0)

    # For integer array we cannot handle the missing values because
    # we cannot mix np.nan with integers
    if isinstance(arr.dtype, types.Float) and (
        not is_overload_true(skipna) or not is_overload_zero(min_count)
    ):

        def impl(arr, skipna, min_count):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = val_zero
            n = len(arr)
            count = 0
            for i in numba.parfors.parfor.internal_prange(n):
                val = val_zero
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i) or not skipna:
                    val = arr[i]
                    count_val = 1
                s += val
                count += count_val
            res = bodo.hiframes.series_kernels._var_handle_mincount(s, count, min_count)
            return res

    else:

        def impl(arr, skipna, min_count):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = val_zero
            n = len(arr)
            for i in numba.parfors.parfor.internal_prange(n):
                val = val_zero
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = arr[i]
                s += val
            return s

    return impl


def array_op_prod(arr, skipna, min_count):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_prod)
def overload_array_op_prod(arr, skipna, min_count):
    val_one = arr.dtype(1)
    # Using True fails for some reason in test_dataframe.py::test_df_prod"[df_value2]"
    # with Bodo inliner
    if arr.dtype == types.bool_:
        val_one = 1

    # For integer array we cannot handle the missing values because
    # we cannot mix np.nan with integers
    if isinstance(arr.dtype, types.Float):

        def impl(arr, skipna, min_count):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = val_one
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = val_one
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i) or not skipna:
                    val = arr[i]
                    count_val = 1
                count += count_val
                s *= val
            res = bodo.hiframes.series_kernels._var_handle_mincount(s, count, min_count)
            return res

    else:

        def impl(arr, skipna, min_count):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = val_one
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = val_one
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = arr[i]
                s *= val
            return s

    return impl


def array_op_idxmax(arr, index):  # pragma: no cover
    pass


@overload(array_op_idxmax, inline="always")
def overload_array_op_idxmax(arr, index):
    # TODO: Make sure -1 is replaced with np.nan
    def impl(arr, index):  # pragma: no cover
        i = bodo.libs.array_kernels._nan_argmax(arr)
        return index[i]

    return impl


def array_op_idxmin(arr, index):  # pragma: no cover
    pass


@overload(array_op_idxmin, inline="always")
def overload_array_op_idxmin(arr, index):
    # TODO: Make sure -1 is replaced with np.nan
    def impl(arr, index):  # pragma: no cover
        i = bodo.libs.array_kernels._nan_argmin(arr)
        return index[i]

    return impl


def _convert_isin_values(values, use_hash_impl):  # pragma: no cover
    pass


@overload(_convert_isin_values, no_unliteral=True)
def overload_convert_isin_values(values, use_hash_impl):
    if is_overload_true(use_hash_impl):

        def impl(values, use_hash_impl):  # pragma: no cover
            values_d = {}
            for k in values:
                values_d[bodo.utils.conversion.box_if_dt64(k)] = 0
            return values_d

        return impl
    else:

        def impl(values, use_hash_impl):  # pragma: no cover
            return values

        return impl


def array_op_isin(arr, values):  # pragma: no cover
    pass


@overload(array_op_isin, inline="always")
def overload_array_op_isin(arr, values):

    # For now we're only using the hash implementation when the dtypes of values
    # and the series are the same, and they are hashable.
    # TODO Optimize this further by casting values to a common dtype if possible
    # and optimal
    use_hash_impl = (element_type(values) == element_type(arr)) and is_hashable_type(
        element_type(values)
    )

    def impl(arr, values):  # pragma: no cover
        values = bodo.libs.array_ops._convert_isin_values(values, use_hash_impl)
        numba.parfors.parfor.init_prange()
        n = len(arr)
        out_arr = np.empty(n, np.bool_)
        for i in numba.parfors.parfor.internal_prange(n):
            # TODO: avoid Timestamp conversion for date comparisons if possible
            out_arr[i] = bodo.utils.conversion.box_if_dt64(arr[i]) in values
        return out_arr

    return impl
