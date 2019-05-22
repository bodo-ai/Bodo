"""
Utility functions for conversion of data such as list to array.
Need to be inlined for better optimization.
"""
import pandas as pd
import numpy as np
import numba
from numba import types
from numba.extending import overload
import bodo

NS_DTYPE = np.dtype('M8[ns]')  # similar to Pandas


# TODO: use generated_jit with IR inlining
def coerce_to_ndarray(data):
    return data


@overload(coerce_to_ndarray)
def overload_coerce_to_ndarray(data):
    # TODO: other cases handled by this function in Pandas like scalar
    """
    Coerces data to ndarray. Data should be numeric.
    """
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.hiframes.pd_index_ext import (RangeIndexType, NumericIndexType,
        DatetimeIndexType, TimedeltaIndexType)

    if isinstance(data, types.Array):
        return lambda data: data

    if isinstance(data, (types.List, types.Tuple)):
        # TODO: check homogenous for tuple
        return lambda data: np.asarray(data)

    if isinstance(data, SeriesType):
        return lambda data: bodo.hiframes.api.get_series_data(data)

    # index types
    if isinstance(data, (NumericIndexType, DatetimeIndexType,
                         TimedeltaIndexType)):
        return lambda data: bodo.hiframes.api.get_index_data(data)

    if isinstance(data, RangeIndexType):
        return lambda data: np.arange(data._start, data._stop, data._step)

    raise TypeError("cannot coerce {} to array".format(data))


# TODO: use generated_jit with IR inlining
def coerce_to_array(data):
    return data


@overload(coerce_to_array)
def overload_coerce_to_array(data):
    """
    convert data to bodo arrays.
    """
    from bodo.hiframes.pd_series_ext import is_str_series_typ

    # string series
    if is_str_series_typ(data):
        return lambda data: bodo.hiframes.api.get_series_data(data)

    # string array
    if data == bodo.string_array_type:
        return lambda data: data

    # string list
    if isinstance(data, types.List) and data.dtype == bodo.string_type:
        return lambda data: bodo.libs.str_arr_ext.StringArray(data)

    # string tuple
    if isinstance(data, types.UniTuple) and data.dtype == bodo.string_type:
        return lambda data: bodo.libs.str_arr_ext.StringArray(list(data))

    # assuming can be ndarray
    return lambda data: bodo.utils.conversion.coerce_to_ndarray(data)


# TODO: use generated_jit with IR inlining
def fix_arr_dtype(data, new_dtype):
    return data


@overload(fix_arr_dtype)
def overload_fix_arr_dtype(data, new_dtype):
    assert isinstance(data, types.Array)
    assert isinstance(new_dtype, types.DType)

    if data.dtype != new_dtype.dtype:
        return lambda data, new_dtype: data.astype(new_dtype)

    return lambda data, new_dtype: data


# TODO: use generated_jit with IR inlining
def parse_datetimes_from_strings(data):
    return data


@overload(parse_datetimes_from_strings)
def overload_parse_datetimes_from_strings(data):
    assert data == bodo.string_array_type

    def parse_impl(data):
        numba.parfor.init_prange()
        n = len(data)
        S = np.empty(n, bodo.utils.conversion.NS_DTYPE)
        for i in numba.parfor.internal_prange(n):
            S[i] = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(data[i])
        return S

    return parse_impl


# TODO: use generated_jit with IR inlining
def convert_to_dt64ns(data):
    return data


@overload(convert_to_dt64ns)
def overload_convert_to_dt64ns(data):
    """Converts data formats like int64 and arrays of strings to dt64ns
    """
    # see pd.core.arrays.datetimes.sequence_to_dt64ns for constructor types
    # TODO: support datetime.date, datetime.datetime
    # TODO: support dayfirst, yearfirst, tz
    if data == types.Array(types.int64, 1, 'C'):
        return lambda data: data.view(bodo.utils.conversion.NS_DTYPE)

    if data == types.Array(types.NPDatetime('ns'), 1, 'C'):
        return lambda data: data

    if data == bodo.string_array_type:
        return (lambda data:
                bodo.utils.conversion.parse_datetimes_from_strings(data))

    raise TypeError("invalid data type {} for dt64 conversion".format(data))


def convert_to_index(data):
    return data


@overload(convert_to_index)
def overload_convert_to_index(data):
    """
    convert data to Index object if necessary.
    """
    from bodo.hiframes.pd_index_ext import (RangeIndexType, NumericIndexType,
        DatetimeIndexType, TimedeltaIndexType)

    # already Index
    if isinstance(data, (RangeIndexType, NumericIndexType, DatetimeIndexType,
                         TimedeltaIndexType, types.NoneType)):
        return lambda data: data

    def impl(data):
        data_arr = bodo.utils.conversion.coerce_to_array(data)
        return bodo.utils.conversion.index_from_array(data_arr)

    return impl


def index_from_array(data):
    return data


@overload(index_from_array)
def overload_index_from_array(data):
    """
    convert data array to Index object.
    """
    if data == bodo.string_array_type:
        return lambda data: data  # TODO: String index

    assert isinstance(data, types.Array)
    if data.dtype == types.NPDatetime('ns'):
        return lambda data: pd.DatetimeIndex(data)

    if isinstance(data.dtype, types.Integer):
        if not data.dtype.signed:
            return lambda data: pd.UInt64Index(data)
        else:
            return lambda data: pd.Int64Index(data)

    if isinstance(data.dtype, types.Float):
        return lambda data: pd.Float64Index(data)

    raise TypeError("invalid index type {}".format(data))


def index_to_array(data):
    return data


@overload(index_to_array)
def overload_index_to_array(I):
    """
    convert Index object to data array.
    """
    from bodo.hiframes.pd_index_ext import (RangeIndexType, NumericIndexType,
        DatetimeIndexType, TimedeltaIndexType)

    if isinstance(I, RangeIndexType):
        return lambda I: np.arange(I._start, I._stop, I._step)

    if I == bodo.string_array_type:
        return lambda I: I  # TODO: String index

    # other indices have data
    return lambda I: bodo.hiframes.api.get_index_data(I)
