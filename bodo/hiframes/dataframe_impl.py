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
