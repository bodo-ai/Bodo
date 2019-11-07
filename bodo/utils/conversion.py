# Copyright (C) 2019 Bodo Inc. All rights reserved.
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
from bodo.utils.typing import is_overload_none, is_overload_true


NS_DTYPE = np.dtype("M8[ns]")  # similar pandas/_libs/tslibs/conversion.pyx
TD_DTYPE = np.dtype("m8[ns]")


# TODO: use generated_jit with IR inlining
def coerce_to_ndarray(data, error_on_nonarray=True, bool_arr_convert=None):
    return data


@overload(coerce_to_ndarray)
def overload_coerce_to_ndarray(data, error_on_nonarray=True, bool_arr_convert=None):
    # TODO: other cases handled by this function in Pandas like scalar
    """
    Coerces data to ndarray. Data should be numeric.
    """
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.hiframes.pd_index_ext import (
        RangeIndexType,
        NumericIndexType,
        DatetimeIndexType,
        TimedeltaIndexType,
    )

    # TODO: handle NAs?
    if isinstance(data, bodo.libs.int_arr_ext.IntegerArrayType):
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.libs.int_arr_ext.get_int_arr_data(
            data
        )

    if data == bodo.libs.bool_arr_ext.boolean_array:
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.libs.bool_arr_ext.get_bool_arr_data(
            data
        )

    if isinstance(data, types.Array):
        if not is_overload_none(bool_arr_convert) and data.dtype == types.bool_:
            return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.libs.bool_arr_ext.init_bool_array(
                data, np.full((len(data) + 7) >> 3, 255, np.uint8)
            )
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: data

    if isinstance(data, (types.List, types.UniTuple)):
        # convert Timestamp() back to dt64
        if data.dtype == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type:

            def impl(data, error_on_nonarray=True, bool_arr_convert=None):
                vals = []
                for d in data:
                    vals.append(
                        bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                            bodo.hiframes.pd_timestamp_ext.convert_timestamp_to_datetime64(
                                d
                            )
                        )
                    )
                return np.asarray(vals)

            return impl
        if not is_overload_none(bool_arr_convert) and data.dtype == types.bool_:
            return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.libs.bool_arr_ext.init_bool_array(
                np.asarray(data), np.full((len(data) + 7) >> 3, 255, np.uint8)
            )
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: np.asarray(
            data
        )

    if isinstance(data, SeriesType):
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.hiframes.pd_series_ext.get_series_data(
            data
        )

    # index types
    if isinstance(data, (NumericIndexType, DatetimeIndexType, TimedeltaIndexType)):
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.hiframes.pd_index_ext.get_index_data(
            data
        )

    if isinstance(data, RangeIndexType):
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: np.arange(
            data._start, data._stop, data._step
        )

    if is_overload_true(error_on_nonarray):
        raise TypeError("cannot coerce {} to array".format(data))

    return lambda data, error_on_nonarray=True, bool_arr_convert=None: data


# TODO: use generated_jit with IR inlining
def coerce_to_array(data, error_on_nonarray=True, bool_arr_convert=None):
    return data


@overload(coerce_to_array)
def overload_coerce_to_array(data, error_on_nonarray=True, bool_arr_convert=None):
    """
    convert data to bodo arrays.
    bool_arr_convert=True converts boolean arrays to nullable BooleanArray
    instead of Numpy arrays.
    """
    from bodo.hiframes.pd_series_ext import is_str_series_typ
    from bodo.hiframes.pd_index_ext import StringIndexType

    # string series
    if is_str_series_typ(data):
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.hiframes.pd_series_ext.get_series_data(
            data
        )

    if isinstance(data, StringIndexType):
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.hiframes.pd_index_ext.get_index_data(
            data
        )

    # string array
    if data == bodo.string_array_type:
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: data

    # string list
    if isinstance(data, types.List) and data.dtype == bodo.string_type:
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.libs.str_arr_ext.StringArray(
            data
        )

    # string tuple
    if isinstance(data, types.UniTuple) and data.dtype == bodo.string_type:
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.libs.str_arr_ext.StringArray(
            list(data)
        )

    if data == bodo.libs.bool_arr_ext.boolean_array:
        return lambda data, error_on_nonarray=True, bool_arr_convert=None: data

    # assuming can be ndarray
    return lambda data, error_on_nonarray=True, bool_arr_convert=None: bodo.utils.conversion.coerce_to_ndarray(
        data, error_on_nonarray, bool_arr_convert
    )


# TODO: use generated_jit with IR inlining
def fix_arr_dtype(data, new_dtype, copy=None):
    return data


@overload(fix_arr_dtype)
def overload_fix_arr_dtype(data, new_dtype, copy=None):
    """convert data to new_dtype, copy if copy parameter is not None
    """
    # TODO: support copy=True and copy=False when literals are passed reliably
    do_copy = not is_overload_none(copy)

    if is_overload_none(new_dtype):
        if do_copy:
            return lambda data, new_dtype, copy=None: data.copy()
        return lambda data, new_dtype, copy=None: data

    nb_dtype = bodo.utils.typing.parse_dtype(new_dtype)

    # nullable int array case
    if isinstance(nb_dtype, bodo.libs.int_arr_ext.IntDtype):
        _dtype = nb_dtype.dtype
        if isinstance(data.dtype, types.Float):

            def impl_float(data, new_dtype, copy=None):
                n = len(data)
                n_bytes = (n + 7) >> 3
                arr = np.empty(n, _dtype)
                bitmap = np.empty(n_bytes, np.uint8)
                for i in numba.parfor.internal_prange(n):
                    arr[i] = data[i]
                    bodo.libs.int_arr_ext.set_bit_to_arr(
                        bitmap, i, not np.isnan(data[i])
                    )
                return bodo.libs.int_arr_ext.init_integer_array(arr, bitmap)

            return impl_float
        else:

            def impl(data, new_dtype, copy=None):
                n = len(data)
                n_bytes = (n + 7) >> 3
                bitmap = np.empty(n_bytes, np.uint8)
                for i in numba.parfor.internal_prange(n):
                    # TODO: use simple set_bit
                    bodo.libs.int_arr_ext.set_bit_to_arr(bitmap, i, 1)
                return bodo.libs.int_arr_ext.init_integer_array(
                    data.astype(_dtype), bitmap
                )

            return impl

    # Array case
    if do_copy or data.dtype != nb_dtype:
        return lambda data, new_dtype, copy=None: data.astype(nb_dtype)

    return lambda data, new_dtype, copy=None: data


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
    if data == types.Array(types.int64, 1, "C"):
        return lambda data: data.view(bodo.utils.conversion.NS_DTYPE)

    if data == types.Array(types.NPDatetime("ns"), 1, "C"):
        return lambda data: data

    if data == bodo.string_array_type:
        return lambda data: bodo.utils.conversion.parse_datetimes_from_strings(data)

    raise TypeError("invalid data type {} for dt64 conversion".format(data))


# TODO: use generated_jit with IR inlining
def convert_to_td64ns(data):
    return data


@overload(convert_to_td64ns)
def overload_convert_to_td64ns(data):
    """Converts data formats like int64 to timedelta64ns
    """
    # TODO: array of strings
    # see pd.core.arrays.timedeltas.sequence_to_td64ns for constructor types
    # TODO: support datetime.timedelta
    if data == types.Array(types.int64, 1, "C"):
        return lambda data: data.view(bodo.utils.conversion.TD_DTYPE)

    if data == types.Array(types.NPTimedelta("ns"), 1, "C"):
        return lambda data: data

    if data == bodo.string_array_type:
        # TODO: support
        raise ValueError("conversion to timedelta from string not supported yet")

    raise TypeError("invalid data type {} for dt64 conversion".format(data))


def convert_to_index(data):
    return data


@overload(convert_to_index)
def overload_convert_to_index(data):
    """
    convert data to Index object if necessary.
    """
    from bodo.hiframes.pd_index_ext import (
        RangeIndexType,
        NumericIndexType,
        DatetimeIndexType,
        TimedeltaIndexType,
        StringIndexType,
    )

    # already Index
    if isinstance(
        data,
        (
            RangeIndexType,
            NumericIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
            StringIndexType,
            types.NoneType,
        ),
    ):
        return lambda data: data

    def impl(data):
        data_arr = bodo.utils.conversion.coerce_to_array(data)
        return bodo.utils.conversion.index_from_array(data_arr)

    return impl


def force_convert_index(I1, I2):
    return I2


@overload(force_convert_index)
def overload_force_convert_index(I1, I2):
    """
    Convert I1 to type of I2, with possible loss of data. TODO: remove this
    """
    from bodo.hiframes.pd_index_ext import RangeIndexType

    if isinstance(I2, RangeIndexType):
        return lambda I1, I2: pd.RangeIndex(len(I1._data))

    return lambda I1, I2: I1


def index_from_array(data, name=None):
    return data


@overload(index_from_array)
def overload_index_from_array(data, name=None):
    """
    convert data array to Index object.
    """
    if data == bodo.string_array_type:
        return lambda data, name=None: bodo.hiframes.pd_index_ext.init_string_index(
            data, name
        )

    assert isinstance(data, (types.Array, bodo.libs.int_arr_ext.IntegerArrayType))

    if data.dtype == types.NPDatetime("ns"):
        return lambda data, name=None: pd.DatetimeIndex(data, name=name)

    if isinstance(data.dtype, types.Integer):
        if not data.dtype.signed:
            return lambda data, name=None: pd.UInt64Index(data, name=name)
        else:
            return lambda data, name=None: pd.Int64Index(data, name=name)

    if isinstance(data.dtype, types.Float):
        return lambda data, name=None: pd.Float64Index(data, name=name)

    # TODO: timedelta, period
    raise TypeError("invalid index type {}".format(data))


def index_to_array(data, l):
    return data


@overload(index_to_array)
def overload_index_to_array(I, l=0):
    """
    convert Index object to data array.
    """
    from bodo.hiframes.pd_index_ext import RangeIndexType

    if is_overload_none(I):
        # return lambda I, l=0: np.arange(l)
        # XXX use implementation of arange directly to avoid calc_nitems calls
        # TODO: remove calc_nitems() with trivial input
        def impl(I, l=0):
            numba.parfor.init_prange()
            arr = np.empty(l, np.int64)
            for i in numba.parfor.internal_prange(l):
                arr[i] = i
            return arr

        return impl

    if isinstance(I, RangeIndexType):
        return lambda I, l=0: np.arange(I._start, I._stop, I._step)

    # other indices have data
    return lambda I, l=0: bodo.hiframes.pd_index_ext.get_index_data(I)


def extract_name_if_none(data, name):
    return name


@overload(extract_name_if_none)
def overload_extract_name_if_none(data, name):
    """Extract name if `data` is has name (Series/Index) and `name` is None
    """
    from bodo.hiframes.pd_index_ext import (
        RangeIndexType,
        NumericIndexType,
        DatetimeIndexType,
        TimedeltaIndexType,
        PeriodIndexType,
    )
    from bodo.hiframes.pd_series_ext import SeriesType

    if not is_overload_none(name):
        return lambda data, name: name

    # Index type, TODO: other indices like Range?
    if isinstance(
        data, (NumericIndexType, DatetimeIndexType, TimedeltaIndexType, PeriodIndexType)
    ):
        return lambda data, name: bodo.hiframes.pd_index_ext.get_index_name(data)

    if isinstance(data, SeriesType):
        return lambda data, name: bodo.hiframes.pd_series_ext.get_series_name(data)

    return lambda data, name: name


def extract_index_if_none(data, index):
    return index


@overload(extract_index_if_none)
def overload_extract_index_if_none(data, index):
    """Extract index if `data` is Series and `index` is None
    """
    from bodo.hiframes.pd_series_ext import SeriesType

    if not is_overload_none(index):
        return lambda data, index: index

    if isinstance(data, SeriesType):
        return lambda data, index: bodo.hiframes.pd_series_ext.get_series_index(data)

    return lambda data, index: index


def box_if_dt64(val):
    return val


@overload(box_if_dt64)
def overload_box_if_dt64(val):
    """If 'val' is dt64, box it to Timestamp otherwise just return 'val'
    """
    if val == types.NPDatetime("ns"):
        return lambda val: bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
            np.int64(val)
        )

    return lambda val: val


def get_array_if_series_or_index(data):
    return data


@overload(get_array_if_series_or_index)
def overload_get_array_if_series_or_index(data):
    from bodo.hiframes.pd_series_ext import SeriesType

    if isinstance(data, SeriesType):
        return lambda data: bodo.hiframes.pd_series_ext.get_series_data(data)

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):
        return lambda data: bodo.hiframes.pd_index_ext.get_index_data(data)

    return lambda data: data


def fix_none_index(I, n):
    return I


@overload(fix_none_index)
def overload_fix_none_index(I, n):
    """Used for converting None index of Series to RangeIndex.
    If I is None, RangeIndex is created. Otherwise, I is returned.
    """
    if is_overload_none(I):
        return lambda I, n: pd.RangeIndex(n)

    return lambda I, n: I


def extract_index_array(A):
    return np.arange(len(A))


@overload(extract_index_array)
def overload_extract_index_array(A):
    """Returns an index array for Series or array.
    if Series, return it's index array. Otherwise, create an index array.
    """
    from bodo.hiframes.pd_series_ext import SeriesType

    if isinstance(A, SeriesType):

        def impl(A):
            index = bodo.hiframes.pd_series_ext.get_series_index(A)
            index_t = bodo.utils.conversion.fix_none_index(index, len(A))
            index_arr = bodo.utils.conversion.coerce_to_array(index_t)
            return index_arr

        return impl

    return lambda A: np.arange(len(A))


def extract_index_array_tup(series_tup):
    return tuple(extract_index_array(s) for s in series_tup)


@overload(extract_index_array_tup)
def overload_extract_index_array_tup(series_tup):
    n_series = len(series_tup.types)
    func_text = "def f(series_tup):\n"
    res = ",".join(
        "bodo.utils.conversion.extract_index_array(series_tup[{}])".format(i)
        for i in range(n_series)
    )
    func_text += "  return ({}{})\n".format(res, "," if n_series == 1 else "")
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["f"]
    return impl


# def to_bool_array_if_np_bool(A):
#     return A


# @overload(to_bool_array_if_np_bool)
# def overload_to_bool_array_if_np_bool(A):
#     """Returns a nullable BooleanArray if input is bool ndarray. Otherwise,
#     just returns the input.
#     """
#     if A == types.Array(types.bool_, 1, 'C'):
#         return lambda A: bodo.libs.bool_arr_ext.init_bool_array(
#                          A, np.full((len(A) + 7) >> 3, 255, np.uint8))

#     return lambda A: A
