# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Some kernels for Series related functions. This is a legacy file that needs to be
refactored.
"""
import datetime

import numba
import numpy as np
from numba.core import types
from numba.extending import overload, register_jitable

import bodo
from bodo.libs.int_arr_ext import IntDtype
from bodo.utils.typing import decode_if_dict_array


# TODO[BE-476]: refactor dataframe column filtering
# TODO: series index and name
# float columns can have regular np.nan
def _column_filter_impl(B, ind):  # pragma: no cover
    A = bodo.hiframes.rolling.alloc_shift(len(B), B, (-1,))
    for i in numba.parfors.parfor.internal_prange(len(A)):
        if ind[i]:
            A[i] = B[i]
        else:
            bodo.libs.array_kernels.setna(A, i)
    return A


# using njit since 1D_var is broken for alloc when there is calculation of len
@numba.njit(no_cpython_wrapper=True)
def _series_dropna_str_alloc_impl_inner(B):  # pragma: no cover
    # TODO: test
    # TODO: generalize
    B = decode_if_dict_array(B)
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
    bodo.libs.str_arr_ext.set_null_bits_to_value(A, -1)
    return A


# return the nan value for the type (handle dt64)
def _get_nan(val):  # pragma: no cover
    return np.nan


@overload(_get_nan, no_unliteral=True)
def _get_nan_overload(val):
    """get NA value with same type as val"""
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


@overload(_get_type_max_value, inline="always", no_unliteral=True)
def _get_type_max_value_overload(dtype):
    # pd.Int64Dtype(), pd.IntegerArray, etc.
    if isinstance(dtype, (bodo.IntegerArrayType, IntDtype)):
        _dtype = dtype.dtype
        return lambda dtype: numba.cpython.builtins.get_type_max_value(
            _dtype
        )  # pragma: no cover

    # datetime.date array
    if dtype == bodo.datetime_date_array_type:
        return lambda dtype: _get_date_max_value()  # pragma: no cover

    # dt64
    if isinstance(dtype.dtype, types.NPDatetime):
        return lambda dtype: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
            numba.cpython.builtins.get_type_max_value(numba.core.types.int64)
        )  # pragma: no cover

    # td64
    if isinstance(dtype.dtype, types.NPTimedelta):
        return lambda dtype: bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
            numba.cpython.builtins.get_type_max_value(numba.core.types.int64)
        )  # pragma: no cover

    if dtype.dtype == types.bool_:
        return lambda dtype: True  # pragma: no cover

    return lambda dtype: numba.cpython.builtins.get_type_max_value(
        dtype
    )  # pragma: no cover


@register_jitable
def _get_date_max_value():  # pragma: no cover
    return datetime.date(datetime.MAXYEAR, 12, 31)


def _get_type_min_value(dtype):  # pragma: no cover
    return 0


@overload(_get_type_min_value, inline="always", no_unliteral=True)
def _get_type_min_value_overload(dtype):

    # pd.Int64Dtype(), pd.IntegerArray, etc.
    if isinstance(dtype, (bodo.IntegerArrayType, IntDtype)):
        _dtype = dtype.dtype
        return lambda dtype: numba.cpython.builtins.get_type_min_value(
            _dtype
        )  # pragma: no cover

    # datetime.date array
    if dtype == bodo.datetime_date_array_type:
        return lambda dtype: _get_date_min_value()  # pragma: no cover

    # dt64
    if isinstance(dtype.dtype, types.NPDatetime):
        return lambda dtype: bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
            numba.cpython.builtins.get_type_min_value(numba.core.types.int64)
        )  # pragma: no cover

    # td64
    if isinstance(dtype.dtype, types.NPTimedelta):
        # int64 seems to get converted to NAT, but uint64 isn't, so we use uint64
        return lambda dtype: bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
            numba.cpython.builtins.get_type_min_value(numba.core.types.uint64)
        )  # pragma: no cover

    if dtype.dtype == types.bool_:
        return lambda dtype: False  # pragma: no cover

    return lambda dtype: numba.cpython.builtins.get_type_min_value(
        dtype
    )  # pragma: no cover


@register_jitable
def _get_date_min_value():  # pragma: no cover
    return datetime.date(datetime.MINYEAR, 1, 1)


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


@numba.njit
def _box_cat_val(s, cat_dtype, count):  # pragma: no cover
    """box categorical code into actual value"""
    if s == -1 or count == 0:
        return bodo.hiframes.series_kernels._get_nan(cat_dtype.categories[0])
    return cat_dtype.categories[s]


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
def _var_handle_mincount(s, count, min_count):  # pragma: no cover
    if count < min_count:
        res = np.nan
    else:
        res = s
    return res


@numba.njit
def _compute_var_nan_count_ddof(
    first_moment, second_moment, count, ddof
):  # pragma: no cover
    if count == 0 or count <= ddof:
        s = np.nan
    else:
        s = second_moment - first_moment * first_moment / count
        s = s / (count - ddof)
    return s


@numba.njit
def _sem_handle_nan(res, count):  # pragma: no cover
    if count < 1:
        res_out = np.nan
    else:
        res_out = (res / count) ** 0.5
    return res_out


@numba.njit
def lt_f(a, b):  # pragma: no cover
    return a < b


@numba.njit
def gt_f(a, b):  # pragma: no cover
    return a > b


@numba.njit
def compute_skew(first_moment, second_moment, third_moment, count):  # pragma: no cover
    if count < 3:
        return np.nan
    mu = first_moment / count
    numerator = third_moment - 3 * second_moment * mu + 2 * count * mu**3
    denominator = second_moment - mu * first_moment
    s = (
        (count * (count - 1) ** (1.5) / (count - 2))
        * numerator
        / (denominator ** (1.5))
    )
    s = s / (count - 1)
    return s


@numba.njit
def compute_kurt(
    first_moment, second_moment, third_moment, fourth_moment, count
):  # pragma: no cover
    if count < 4:
        return np.nan
    mu = first_moment / count
    m4 = (
        fourth_moment
        - 4 * third_moment * mu
        + 6 * second_moment * mu**2
        - 3 * count * mu**4
    )
    m2 = second_moment - mu * first_moment
    adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
    numer = count * (count + 1) * (count - 1) * m4
    denom = (count - 2) * (count - 3) * m2**2
    s = (count - 1) * (numer / denom - adj)
    s = s / (count - 1)
    return s
