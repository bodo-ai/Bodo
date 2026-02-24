"""
Implements array operations for usage by DataFrames and Series
such as count and max.
"""

import numba
import numpy as np
import pandas as pd
from numba import generated_jit
from numba.core import types
from numba.extending import overload

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.hiframes.time_ext import TimeArrayType
from bodo.utils import tracing
from bodo.utils.typing import (
    element_type,
    is_hashable_type,
    is_iterable_type,
    is_overload_true,
    is_overload_zero,
    is_str_arr_type,
)


def array_op_any(arr, skipna=True):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


# TODO: implement skipna
@overload(array_op_any, jit_options={"cache": True})
def overload_array_op_any(A, skipna=True):
    """Returns whether an array contains any truthy values.

    Args:
        A (np.ndarray): an array of values that can be integers, booleans,
        strings or bytes.
        skipna (bool, optional): not supported. Defaults to True.

    Raises:
        BodoError: if an unsupported array type is provided.

    Returns:
        boolean: whether the array contains at least 1 truthy value.
    """
    if (
        isinstance(A, types.Array) and isinstance(A.dtype, types.Integer)
    ) or isinstance(A, bodo.libs.int_arr_ext.IntegerArrayType):
        zero_value = 0
    elif (isinstance(A, bodo.libs.bool_arr_ext.BooleanArrayType)) or (
        isinstance(A, types.Array) and A.dtype == types.bool_
    ):
        zero_value = False
    elif A == bodo.types.string_array_type:
        zero_value = ""
    elif A == bodo.types.binary_array_type:
        zero_value = b""
    else:
        raise bodo.utils.typing.BodoError(
            f"Cannot perform any with this array type: {A}"
        )

    def impl(A, skipna=True):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            if not bodo.libs.array_kernels.isna(A, i):
                if A[i] != zero_value:
                    count += 1
        return count != 0

    return impl


def array_op_all(arr, skipna=True):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


# TODO: implement skipna
@overload(array_op_all, jit_options={"cache": True})
def overload_array_op_all(A, skipna=True):
    """Returns whether an array contains only truthy values.

    Args:
        A (np.ndarray): an array of values that can be integers, booleans,
        strings or bytes.
        skipna (bool, optional): not supported. Defaults to True.

    Raises:
        BodoError: if an unsupported array type is provided.

    Returns:
        boolean: whether the array contains at only truthy value.
    """
    if (
        isinstance(A, types.Array) and isinstance(A.dtype, types.Integer)
    ) or isinstance(A, bodo.libs.int_arr_ext.IntegerArrayType):
        zero_value = 0
    elif (isinstance(A, bodo.libs.bool_arr_ext.BooleanArrayType)) or (
        isinstance(A, types.Array) and A.dtype == types.bool_
    ):
        zero_value = False
    elif A == bodo.types.string_array_type:
        zero_value = ""
    elif A == bodo.types.binary_array_type:
        zero_value = b""
    else:
        raise bodo.utils.typing.BodoError(
            f"Cannot perform all with this array type: {A}"
        )

    def impl(A, skipna=True):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(A)):
            if not bodo.libs.array_kernels.isna(A, i):
                if A[i] == zero_value:
                    count += 1
        return count == 0

    return impl


@numba.njit
def array_op_median(arr, skipna=True, parallel=False):  # pragma: no cover
    # TODO: check return types, e.g. float32 -> float32
    res = np.empty(1, types.float64)
    bodo.libs.array_kernels.median_series_computation(res.ctypes, arr, parallel, skipna)
    return res[0]


def array_op_isna(arr):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_isna, jit_options={"cache": True})
def overload_array_op_isna(arr):
    def impl(arr):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        n = len(arr)
        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        for i in numba.parfors.parfor.internal_prange(n):
            out_arr[i] = bodo.libs.array_kernels.isna(arr, i)
        return out_arr

    return impl


def drop_duplicates_local_dictionary_if_dict(arr):  # pragma: no cover
    """
    Used with bodo.utils.table_utils.generate_mappable_table_func to
    drop duplicates (and NAs) from dictionaries of all dictionary-encoded
    arrays in a dataframe using table-format.
    """


@overload(drop_duplicates_local_dictionary_if_dict, jit_options={"cache": True})
def overload_drop_duplicates_local_dictionary_if_dict(arr):
    if arr == bodo.types.dict_str_arr_type:
        return lambda arr: bodo.libs.array.drop_duplicates_local_dictionary(
            arr, False
        )  # lambda: no cover
    return lambda arr: arr  # lambda: no cover


def array_op_count(arr):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_count, jit_options={"cache": True})
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


def array_op_describe_impl(arr):  # pragma: no cover
    a_count = array_op_count(arr)
    a_min = array_op_min(arr)
    a_max = array_op_max(arr)
    a_mean = array_op_mean(arr)
    a_std = array_op_std(arr)
    q25 = array_op_quantile(arr, 0.25)
    q50 = array_op_quantile(arr, 0.5)
    q75 = array_op_quantile(arr, 0.75)
    return (a_count, a_mean, a_std, a_min, q25, q50, q75, a_max)


def array_op_describe_dt_impl(arr):  # pragma: no cover
    # Pandas doesn't return std for describe of datetime64 data
    # https://github.com/pandas-dev/pandas/blob/059c8bac51e47d6eaaa3e36d6a293a22312925e6/pandas/core/describe.py#L328
    a_count = array_op_count(arr)
    a_min = array_op_min(arr)
    a_max = array_op_max(arr)
    a_mean = array_op_mean(arr)
    q25 = array_op_quantile(arr, 0.25)
    q50 = array_op_quantile(arr, 0.5)
    q75 = array_op_quantile(arr, 0.75)
    return (a_count, a_mean, a_min, q25, q50, q75, a_max)


@overload(array_op_describe, jit_options={"cache": True})
def overload_array_op_describe(arr):
    # Pandas doesn't return std for describe of datetime64 data
    # https://github.com/pandas-dev/pandas/blob/059c8bac51e47d6eaaa3e36d6a293a22312925e6/pandas/core/describe.py#L328
    if arr.dtype == bodo.types.datetime64ns or isinstance(
        arr, bodo.types.DatetimeArrayType
    ):
        return array_op_describe_dt_impl

    return array_op_describe_impl


@generated_jit(nopython=True)
def array_op_nbytes(arr):
    return array_op_nbytes_impl


def array_op_nbytes_impl(arr):  # pragma: no cover
    return arr.nbytes


def array_op_min(arr):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_min)
def overload_array_op_min(arr):
    if arr.dtype in (bodo.types.timedelta64ns, bodo.types.pd_timedelta_type):

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

    if arr.dtype == bodo.types.datetime64ns:

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

    if isinstance(arr, bodo.types.DatetimeArrayType):

        def impl_dt64(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_max_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i].value)
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

    if isinstance(arr, TimeArrayType):

        def impl_time(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = bodo.hiframes.series_kernels._get_time_max_value()
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

        return impl_time

    if is_str_arr_type(arr):
        min_or_max = bodo.libs.str_arr_ext.MinOrMax.Min.value

        def impl_str_arr_min(arr):  # pragma: no cover
            return bodo.libs.str_arr_ext.str_arr_min_max(arr, min_or_max)

        return impl_str_arr_min

    def impl(arr):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        s = bodo.hiframes.series_kernels._get_type_max_value(arr)
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


@overload(array_op_max, jit_options={"cache": True})
def overload_array_op_max(arr):
    if arr.dtype in (bodo.types.timedelta64ns, bodo.types.pd_timedelta_type):

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

    if arr.dtype == bodo.types.datetime64ns:

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

    if isinstance(arr, bodo.types.DatetimeArrayType):

        def impl_dt64(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = numba.cpython.builtins.get_type_min_value(np.int64)
            count = 0
            for i in numba.parfors.parfor.internal_prange(len(arr)):
                val = s
                count_val = 0
                if not bodo.libs.array_kernels.isna(arr, i):
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i].value)
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

    if isinstance(arr, TimeArrayType):

        def impl_time(arr):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            s = bodo.hiframes.series_kernels._get_time_min_value()
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

        return impl_time

    if is_str_arr_type(arr):
        min_or_max = bodo.libs.str_arr_ext.MinOrMax.Max.value

        def impl_str_arr_max(arr):  # pragma: no cover
            return bodo.libs.str_arr_ext.str_arr_min_max(arr, min_or_max)

        return impl_str_arr_max

    def impl(arr):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        s = bodo.hiframes.series_kernels._get_type_min_value(arr)
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


@overload(array_op_mean, jit_options={"cache": True})
def overload_array_op_mean(arr):
    # datetime
    if arr.dtype == bodo.types.datetime64ns:

        def impl(arr):  # pragma: no cover
            return pd.Timestamp(
                types.int64(bodo.libs.array_ops.array_op_mean(arr.view(np.int64)))
            )

        return impl

    if isinstance(arr, bodo.types.DatetimeArrayType):

        def impl(arr):  # pragma: no cover
            return pd.Timestamp(
                types.int64(bodo.libs.array_ops.array_op_mean(arr._data.view(np.int64)))
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


@overload(array_op_var, jit_options={"cache": True})
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

        res = bodo.hiframes.series_kernels._compute_var_nan_count_ddof(
            first_moment, second_moment, count, ddof
        )
        return res

    return impl


def array_op_std(arr, skipna=True, ddof=1):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_std, jit_options={"cache": True})
def overload_array_op_std(arr, skipna=True, ddof=1):
    # datetime
    if arr.dtype == bodo.types.datetime64ns:

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


@overload(array_op_quantile, jit_options={"cache": True})
def overload_array_op_quantile(arr, q):
    if is_iterable_type(q):
        if arr.dtype == bodo.types.datetime64ns:

            def _impl_list_dt(arr, q):  # pragma: no cover
                out_arr = np.empty(len(q), np.int64)
                for i in range(len(q)):
                    q_val = np.float64(q[i])
                    out_arr[i] = bodo.libs.array_kernels.quantile(
                        arr.view(np.int64), q_val
                    )
                return out_arr.view(np.dtype("datetime64[ns]"))

            return _impl_list_dt

        if isinstance(arr, bodo.types.DatetimeArrayType):
            tz = arr.tz

            def _impl_list_dt_tz(arr, q):  # pragma: no cover
                out_arr = bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(
                    len(q), tz
                )
                for i in range(len(q)):
                    q_val = np.float64(q[i])
                    out_arr[i] = pd.Timestamp(
                        bodo.libs.array_kernels.quantile(
                            arr._data.view(np.int64), q_val
                        ),
                        tz=tz,
                    )
                return out_arr

            return _impl_list_dt_tz

        def impl_list(arr, q):  # pragma: no cover
            out_arr = np.empty(len(q), np.float64)
            for i in range(len(q)):
                q_val = np.float64(q[i])
                out_arr[i] = bodo.libs.array_kernels.quantile(arr, q_val)
            return out_arr

        return impl_list

    if arr.dtype == bodo.types.datetime64ns:

        def _impl_dt(arr, q):  # pragma: no cover
            return pd.Timestamp(
                bodo.libs.array_kernels.quantile(arr.view(np.int64), np.float64(q))
            )

        return _impl_dt

    if isinstance(arr, bodo.types.DatetimeArrayType):
        tz = arr.tz

        def _impl_dt_tz(arr, q):  # pragma: no cover
            return pd.Timestamp(
                bodo.libs.array_kernels.quantile(
                    arr._data.view(np.int64), np.float64(q)
                ),
                tz=tz,
            )

        return _impl_dt_tz

    def impl(arr, q):  # pragma: no cover
        return bodo.libs.array_kernels.quantile(arr, np.float64(q))

    return impl


def array_op_sum(arr, skipna, min_count):  # pragma: no cover
    # Create an overload for manual inlining in Series pass.
    pass


@overload(array_op_sum, no_unliteral=True, jit_options={"cache": True})
def overload_array_op_sum(arr, skipna, min_count):
    if isinstance(arr, bodo.types.DecimalArrayType):

        def impl(arr, skipna, min_count):  # pragma: no cover
            return bodo.libs.decimal_arr_ext.sum_decimal_array(arr)

        return impl

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


@overload(array_op_prod, jit_options={"cache": True})
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


@overload(array_op_idxmax, inline="always", jit_options={"cache": True})
def overload_array_op_idxmax(arr, index):
    # TODO: Make sure -1 is replaced with np.nan
    def impl(arr, index):  # pragma: no cover
        i = bodo.libs.array_kernels._nan_argmax(arr)
        return index[i]

    return impl


def array_op_idxmin(arr, index):  # pragma: no cover
    pass


@overload(array_op_idxmin, inline="always", jit_options={"cache": True})
def overload_array_op_idxmin(arr, index):
    # TODO: Make sure -1 is replaced with np.nan
    def impl(arr, index):  # pragma: no cover
        i = bodo.libs.array_kernels._nan_argmin(arr)
        return index[i]

    return impl


def _convert_isin_values(values, use_hash_impl):  # pragma: no cover
    pass


@overload(_convert_isin_values, no_unliteral=True, jit_options={"cache": True})
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


@overload(array_op_isin, inline="always", jit_options={"cache": True})
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
        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        for i in numba.parfors.parfor.internal_prange(n):
            # TODO: avoid Timestamp conversion for date comparisons if possible
            # TODO: handle None/nan/NA values properly
            out_arr[i] = bodo.utils.conversion.box_if_dt64(arr[i]) in values
        return out_arr

    return impl


@generated_jit(nopython=True)
def array_unique_vector_map(in_arr_tup):
    """
    Computes the unique values of a tuple of arrays on a given
    rank. This returns two values:
        - An tuple of arrays with the unique values, out_arr_tup
        - An array that maps each value in the original
          in_arr_tup to its row in out_arr_tup
    """
    # Avoid using tuple when you have 1 element to avoid nullable tuples.
    use_tuple = len(in_arr_tup) != 1
    arr_typ_list = list(in_arr_tup.types)
    func_text = "def impl(in_arr_tup):\n"
    func_text += "  ev = tracing.Event('array_unique_vector_map', is_parallel=False)\n"
    func_text += "  n = len(in_arr_tup[0])\n"
    if use_tuple:
        # If we use a tuple then we generate code per element in the tuple.
        elem_list = ", ".join(
            [f"in_arr_tup[{i}][unused]" for i in range(len(in_arr_tup))]
        )
        null_bitmap = ", ".join(["False" for _ in range(len(in_arr_tup))])
        func_text += f"  arr_map = {{bodo.libs.nullable_tuple_ext.build_nullable_tuple(({elem_list},), ({null_bitmap},)): 0 for unused in range(0)}}\n"
        # Use a dummy dictionary comprehension to type the
        # dictionary. See the list example in:
        # https://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#my-code-has-an-untyped-list-problem
        func_text += "  map_vector = np.empty(n, np.int64)\n"
        for i, in_arr in enumerate(arr_typ_list):
            func_text += f"  in_lst_{i} = []\n"
            # If the array is a string type, compute the output size while computing unique
            if is_str_arr_type(in_arr):
                func_text += f"  total_len_{i} = 0\n"
            func_text += f"  null_in_lst_{i} = []\n"
        func_text += "  for i in range(n):\n"
        # If we have a tuple create a nullable tuple
        data_code = ", ".join([f"in_arr_tup[{i}][i]" for i in range(len(arr_typ_list))])
        null_code = ", ".join(
            [
                f"bodo.libs.array_kernels.isna(in_arr_tup[{i}], i)"
                for i in range(len(arr_typ_list))
            ]
        )
        func_text += f"    data_val = bodo.libs.nullable_tuple_ext.build_nullable_tuple(({data_code},), ({null_code},))\n"
        func_text += "    if data_val not in arr_map:\n"
        func_text += "      set_val = len(arr_map)\n"
        # Add the data to index info
        func_text += "      values_tup = data_val._data\n"
        func_text += "      nulls_tup = data_val._null_values\n"
        for i, in_arr in enumerate(arr_typ_list):
            func_text += f"      in_lst_{i}.append(values_tup[{i}])\n"
            func_text += f"      null_in_lst_{i}.append(nulls_tup[{i}])\n"
            if is_str_arr_type(in_arr):
                # If the data is null nulls_tup[i] == 0, so we multiply here.
                # To avoid extra string allocations to convert to utf8
                # we use the offset in the original array.
                func_text += f"      total_len_{i}  += nulls_tup[{i}] * bodo.libs.str_arr_ext.get_str_arr_item_length(in_arr_tup[{i}], i)\n"
        func_text += "      arr_map[data_val] = len(arr_map)\n"
        func_text += "    else:\n"
        func_text += "      set_val = arr_map[data_val]\n"
        func_text += "    map_vector[i] = set_val\n"
        # Compute the output arrays for the index.
        func_text += "  n_rows = len(arr_map)\n"
        for i, in_arr in enumerate(arr_typ_list):
            if is_str_arr_type(in_arr):
                func_text += f"  out_arr_{i} = bodo.libs.str_arr_ext.pre_alloc_string_array(n_rows, total_len_{i})\n"
            else:
                func_text += f"  out_arr_{i} = bodo.utils.utils.alloc_type(n_rows, in_arr_tup[{i}], (-1,))\n"
        # Convert the lists to arrays
        func_text += "  for j in range(len(arr_map)):\n"
        for i in range(len(arr_typ_list)):
            func_text += f"    if null_in_lst_{i}[j]:\n"
            func_text += f"      bodo.libs.array_kernels.setna(out_arr_{i}, j)\n"
            func_text += "    else:\n"
            func_text += f"      out_arr_{i}[j] = in_lst_{i}[j]\n"
        ret_arrs = ", ".join([f"out_arr_{i}" for i in range(len(arr_typ_list))])
        func_text += "  ev.add_attribute('n_map_entries', n_rows)\n"
        func_text += "  ev.finalize()\n"
        func_text += f"  return ({ret_arrs},), map_vector\n"
    else:
        # If we have a single array, extract it and generate code for 1 array.
        func_text += "  in_arr = in_arr_tup[0]\n"
        # Use a dummy dictionary comprehension to type the
        # dictionary. See the list example in:
        # https://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#my-code-has-an-untyped-list-problem
        func_text += "  arr_map = {in_arr[unused]: 0 for unused in range(0)}\n"
        func_text += "  map_vector = np.empty(n, np.int64)\n"
        func_text += "  is_na = 0\n"
        func_text += "  in_lst = []\n"
        func_text += "  na_idxs = []\n"
        if is_str_arr_type(arr_typ_list[0]):
            func_text += "  total_len = 0\n"
        func_text += "  for i in range(n):\n"
        func_text += "    if bodo.libs.array_kernels.isna(in_arr, i):\n"
        func_text += "      is_na = 1\n"
        func_text += "      # Always put NA in the last location.\n"
        func_text += "      # We use -1 as a placeholder\n"
        func_text += "      set_val = -1\n"
        func_text += "      na_idxs.append(i)\n"
        func_text += "    else:\n"
        func_text += "      data_val = in_arr[i]\n"
        func_text += "      if data_val not in arr_map:\n"
        func_text += "        set_val = len(arr_map)\n"
        # Add the data to index info
        func_text += "        in_lst.append(data_val)\n"
        if is_str_arr_type(arr_typ_list[0]):
            # To avoid extra string allocations to convert to utf8
            # we use the offset in the original array.
            func_text += "        total_len += bodo.libs.str_arr_ext.get_str_arr_item_length(in_arr, i)\n"
        func_text += "        arr_map[data_val] = len(arr_map)\n"
        func_text += "      else:\n"
        func_text += "        set_val = arr_map[data_val]\n"
        func_text += "    map_vector[i] = set_val\n"
        # Replace -1 with the actual row value.
        func_text += "  map_vector[na_idxs] = len(arr_map)\n"
        # Compute the output arrays for the index.
        func_text += "  n_rows = len(arr_map) + is_na\n"
        if is_str_arr_type(arr_typ_list[0]):
            func_text += "  out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n_rows, total_len)\n"
        else:
            func_text += (
                "  out_arr = bodo.utils.utils.alloc_type(n_rows, in_arr, (-1,))\n"
            )
        # Convert the list to an array
        func_text += "  for j in range(len(arr_map)):\n"
        func_text += "    out_arr[j] = in_lst[j]\n"
        func_text += "  if is_na:\n"
        func_text += "    bodo.libs.array_kernels.setna(out_arr, n_rows - 1)\n"
        func_text += "  ev.add_attribute('n_map_entries', n_rows)\n"
        func_text += "  ev.finalize()\n"
        func_text += "  return (out_arr,), map_vector\n"

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np, "tracing": tracing}, loc_vars)
    impl = loc_vars["impl"]
    return impl
