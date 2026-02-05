"""Some kernels for Series related functions. This is a legacy file that needs to be
refactored.
"""

import datetime

import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.extending import overload, register_jitable

import bodo
from bodo.hiframes.time_ext import TimeArrayType
from bodo.libs.decimal_arr_ext import _str_to_decimal_scalar
from bodo.libs.float_arr_ext import FloatDtype, FloatingArrayType
from bodo.libs.int_arr_ext import IntDtype
from bodo.utils.typing import decode_if_dict_array


# TODO[BE-476]: refactor dataframe column filtering
# TODO: series index and name
# float columns can have regular np.nan
def _column_filter_impl(B, ind):  # pragma: no cover
    # I'm not entirly certain why, but we need to specify the optional argument here,
    # or else we get Numba bugs.
    A = bodo.hiframes.rolling.alloc_shift(len(B), B, (-1,), None)
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
    # nullable float and int data
    if isinstance(
        dtype, (bodo.types.IntegerArrayType, IntDtype, FloatingArrayType, FloatDtype)
    ):
        _dtype = dtype.dtype
        return lambda dtype: numba.cpython.builtins.get_type_max_value(
            _dtype
        )  # pragma: no cover

    # datetime.date array
    if dtype == bodo.types.datetime_date_array_type:
        return lambda dtype: _get_date_max_value()  # pragma: no cover

    # bodo.types.Time array
    if dtype == TimeArrayType:
        return lambda dtype: _get_time_max_value()  # pragma: no cover

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

    # tz-aware timestamp array
    if isinstance(dtype, bodo.types.DatetimeArrayType):
        tz = dtype.tz

        def impl(dtype):  # pragma: no cover
            int_value = numba.cpython.builtins.get_type_max_value(
                numba.core.types.int64
            )
            result = bodo.hiframes.pd_timestamp_ext.convert_val_to_timestamp(
                int_value, tz
            )
            return result

        return impl

    # timedelta array
    if dtype == bodo.types.timedelta_array_type:

        def impl(dtype):  # pragma: no cover
            int_value = numba.cpython.builtins.get_type_max_value(
                numba.core.types.int64
            )
            return bodo.hiframes.datetime_timedelta_ext.init_pd_timedelta(int_value)

        return impl

    # timestamptz array
    if dtype == bodo.types.timestamptz_array_type:
        return lambda dtype: _get_timestamptz_max_value()

    if dtype.dtype == types.bool_:
        return lambda dtype: True  # pragma: no cover

    if isinstance(dtype, bodo.types.DecimalArrayType):
        scale = dtype.dtype.scale
        precision = dtype.dtype.precision

        def impl(dtype):
            decimal_string = "9" * (precision - scale) + "." + "9" * scale
            decimal_val, _ = _str_to_decimal_scalar(decimal_string, precision, scale)
            return decimal_val

        return impl

    _dtype = dtype.dtype
    return lambda dtype: numba.cpython.builtins.get_type_max_value(
        _dtype
    )  # pragma: no cover


@register_jitable
def _get_date_max_value():  # pragma: no cover
    return datetime.date(datetime.MAXYEAR, 12, 31)


@register_jitable
def _get_time_max_value():  # pragma: no cover
    return bodo.types.Time(23, 59, 59, 999, 999, 999)


@register_jitable
def _get_timestamptz_max_value():  # pragma: no cover
    return bodo.types.TimestampTZ(
        pd.Timestamp(numba.cpython.builtins.get_type_max_value(numba.core.types.int64)),
        0,
    )


def _get_type_min_value(dtype):  # pragma: no cover
    return 0


@overload(_get_type_min_value, inline="always", no_unliteral=True)
def _get_type_min_value_overload(dtype):
    # nullable float and int data
    if isinstance(
        dtype, (bodo.types.IntegerArrayType, IntDtype, FloatingArrayType, FloatDtype)
    ):
        _dtype = dtype.dtype
        return lambda dtype: numba.cpython.builtins.get_type_min_value(
            _dtype
        )  # pragma: no cover

    # datetime.date array
    if dtype == bodo.types.datetime_date_array_type:
        return lambda dtype: _get_date_min_value()  # pragma: no cover

    # bodo.types.Time array
    if dtype == TimeArrayType:
        return lambda dtype: _get_time_min_value()  # pragma: no cover

    # Datetime array
    if isinstance(
        dtype, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType
    ):  # pragma: no cover
        tz = dtype.tz

        def impl(dtype):  # pragma: no cover
            int_value = numba.cpython.builtins.get_type_min_value(
                numba.core.types.int64
            )
            result = bodo.hiframes.pd_timestamp_ext.convert_val_to_timestamp(
                int_value, tz
            )
            return result

        return impl

    # timedelta array
    if dtype == bodo.types.timedelta_array_type:

        def impl(dtype):  # pragma: no cover
            int_value = numba.cpython.builtins.get_type_min_value(
                numba.core.types.int64
            )
            return bodo.hiframes.datetime_timedelta_ext.init_pd_timedelta(int_value)

        return impl

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

    # timestamptz array
    if dtype == bodo.types.timestamptz_array_type:
        return lambda dtype: _get_timestamptz_min_value()

    if dtype.dtype == types.bool_:
        return lambda dtype: False  # pragma: no cover

    if isinstance(dtype, bodo.types.DecimalArrayType):
        scale = dtype.dtype.scale
        precision = dtype.dtype.precision

        def impl(dtype):
            decimal_string = "-" + "9" * (precision - scale) + "." + "9" * scale
            decimal_val, _ = _str_to_decimal_scalar(decimal_string, precision, scale)
            return decimal_val

        return impl

    _dtype = dtype.dtype
    return lambda dtype: numba.cpython.builtins.get_type_min_value(
        _dtype
    )  # pragma: no cover


@register_jitable
def _get_date_min_value():  # pragma: no cover
    return datetime.date(datetime.MINYEAR, 1, 1)


@register_jitable
def _get_time_min_value():  # pragma: no cover
    return bodo.types.Time()


@register_jitable
def _get_timestamptz_min_value():  # pragma: no cover
    return bodo.types.TimestampTZ(
        pd.Timestamp(numba.cpython.builtins.get_type_min_value(numba.core.types.int64)),
        0,
    )


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
    denominator = (second_moment - mu * first_moment) ** 1.5
    # If the denominator is deemed sufficiently close to zero, return zero by default.
    # These constants were derived from trial and error to correctly flag values
    # that should be zero without causing any of the other tests to be falsely
    # flagged just because they are near zero.
    if (
        numerator == 0.0
        or np.abs(denominator) < 1e-14
        or np.log2(np.abs(denominator)) - np.log2(np.abs(numerator)) < -20
    ):
        return 0.0
    s = (count * (count - 1) ** (1.5) / (count - 2)) * numerator / (denominator)
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
    numer = count * (count + 1) * (count - 1) * m4
    denom = (count - 2) * (count - 3) * m2**2
    # If the denominator is deemed sufficiently close to zero, return zero by default.
    # These constants were derived from trial and error to correctly flag values
    # that should be zero without causing any of the other tests to be falsely
    # flagged just because they are near zero.
    if (
        numer == 0.0
        or np.abs(denom) < 1e-14
        or np.log2(np.abs(denom)) - np.log2(np.abs(numer)) < -20
    ):
        return 0.0
    adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
    s = (count - 1) * (numer / denom - adj)
    s = s / (count - 1)
    return s
