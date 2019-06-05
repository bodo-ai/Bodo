"""
Implementation of DataFrame attributes and methods using overload.
"""
import operator
import numpy as np
import pandas as pd
import numba
from numba import types
from numba.extending import overload, overload_attribute, overload_method
import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.utils.typing import (is_overload_none, is_overload_true,
    is_overload_false, is_overload_zero)


@overload_attribute(DataFrameType, 'index')
def overload_dataframe_index(df):
    # None index means full RangeIndex
    if df.index == types.none:
        return lambda df: bodo.hiframes.pd_index_ext.init_range_index(
            0, len(df), 1, None)

    return lambda df: bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)


@overload_attribute(DataFrameType, 'columns')
def overload_dataframe_columns(df):
    str_arr = "bodo.utils.conversion.coerce_to_array({})".format(df.columns)
    func_text = "def impl(df):\n"
    func_text += "  return bodo.hiframes.pd_index_ext.init_string_index({})\n".format(
        str_arr)
    loc_vars = {}
    exec(func_text, {'bodo': bodo}, loc_vars)
    # print(func_text)
    impl = loc_vars['impl']
    return impl


@overload_attribute(DataFrameType, 'values')
def overload_dataframe_values(df):
    n_cols = len(df.columns)
    data_args = ", ".join(
        'bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})'.format(i)
        for i in range(n_cols))
    func_text = "def f(df):\n".format()
    func_text += "    return np.stack(({},), 1)\n".format(data_args)

    loc_vars = {}
    exec(func_text, {'bodo': bodo, 'np': np}, loc_vars)
    f = loc_vars['f']
    return f


@overload_method(DataFrameType, 'get_values')
def overload_dataframe_get_values(df):
    def impl(df):
        return df.values
    return impl


@overload_attribute(DataFrameType, 'ndim')
def overload_dataframe_ndim(df):
    return lambda df: 2


@overload_attribute(DataFrameType, 'size')
def overload_dataframe_size(df):
    ncols = len(df.columns)
    return lambda df: ncols * len(df)


@overload_attribute(DataFrameType, 'shape')
def overload_dataframe_shape(df):
    ncols = len(df.columns)
    return lambda df: (len(df), ncols)


@overload_attribute(DataFrameType, 'empty')
def overload_dataframe_empty(df):
    if len(df.columns) == 0:
        return lambda df: True
    return lambda df: len(df) == 0
