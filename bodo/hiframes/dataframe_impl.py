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
