"""
Implementation of Series attributes and methods.
"""
import numba
from numba import types
from numba.extending import overload, overload_attribute, overload_method
import bodo
from bodo.hiframes.pd_series_ext import SeriesType


@overload_attribute(SeriesType, 'index')
def overload_series_index(s):
    # None should be range type
    if s.index == types.none:
        return lambda s: bodo.hiframes.pd_index_ext.init_range_index(
            0, len(bodo.hiframes.api.get_series_data(s)), 1)

    return lambda s: bodo.hiframes.api.get_series_index(s)


@overload_attribute(SeriesType, 'values')
def overload_series_values(s):
    return lambda s: bodo.hiframes.api.get_series_data(s)


@overload_attribute(SeriesType, 'dtype')
def overload_series_dtype(s):
    # TODO: check other dtypes like tuple, etc.
    if s.dtype == bodo.string_type:
        raise ValueError("Series.dtype not supported for string Series yet")

    return lambda s: bodo.hiframes.api.get_series_data(s).dtype


@overload(len)
def overload_series_len(S):
    if isinstance(S, SeriesType):
        return lambda S: len(bodo.hiframes.api.get_series_data(S))
