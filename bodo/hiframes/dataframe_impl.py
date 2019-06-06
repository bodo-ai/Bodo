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


@overload_method(DataFrameType, 'astype')
def overload_dataframe_astype(df, dtype, copy=True, errors='raise'):
    # just call astype() on all column Series
    # TODO: support categorical, dt64, etc.

    data_args = ", ".join("df['{}'].astype(dtype).values".format(c)
        for c in df.columns)
    header = "def impl(df, dtype, copy=True, errors='raise'):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, 'copy')
def overload_dataframe_copy(df, deep=True):
    # just call copy() on all arrays
    data_outs = []
    for i in range(len(df.columns)):
        arr = "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})".format(i)
        if is_overload_true(deep):
            data_outs.append(arr + ".copy()")
        elif is_overload_false(deep):
            data_outs.append(arr)
        else:
            data_outs.append("{arr}.copy() if deep else {arr}".format(arr=arr))

    header = "def impl(df, deep=True):\n"
    return _gen_init_df(header, df.columns, ", ".join(data_outs))


@overload_method(DataFrameType, 'isna')
@overload_method(DataFrameType, 'isnull')
def overload_dataframe_isna(df):
    # call isna() on column Series
    data_args = ", ".join("df['{}'].isna().values".format(c)
        for c in df.columns)
    header = "def impl(df):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, 'notna')
def overload_dataframe_notna(df):
    # call notna() on column Series
    data_args = ", ".join("df['{}'].notna().values".format(c)
        for c in df.columns)
    header = "def impl(df):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, 'head')
def overload_dataframe_head(df, n=5):
    # call head() on column Series
    data_args = ", ".join("df['{}'].head(n).values".format(c)
        for c in df.columns)
    header = "def impl(df, n=5):\n"
    index = ("bodo.utils.conversion.fix_none_index("
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df), len(df))[:n]")
    return _gen_init_df(header, df.columns, data_args, index)


@overload_method(DataFrameType, 'isin')
def overload_dataframe_isin(df, values):
    # TODO: call isin on Series
    # TODO: make sure df indices match?
    # TODO: dictionary case
    other_colmap = {}
    df_case = False
    # dataframe case
    if isinstance(values, DataFrameType):
        df_case = True
        other_colmap = {c: "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(values, {})".format(values.columns.index(c))
                        for c in df.columns if c in values.columns}
    else:
        # general iterable (e.g. list, set) case
        # TODO: handle passed in dict case (pass colname to func?)
        other_colmap = {c: "values" for c in df.columns}

    data = [
        'bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})'.format(i)
        for i in range(len(df.columns))]

    isin_func = "bodo.hiframes.api.df_isin({}, {})"
    isin_vals_func = "bodo.hiframes.api.df_isin_vals({}, {})"
    bool_arr_func = "np.zeros(len(df), np.bool_)"
    out_vars = []
    for cname, in_var in zip(df.columns, data):
        if cname in other_colmap:
            if df_case:
                func = isin_func
            else:
                func = isin_vals_func
            other_col_var = other_colmap[cname]
            func = func.format(in_var, other_col_var)
        else:
            func = bool_arr_func
        out_vars.append(func)

    data_args = ", ".join(out_vars)
    header = "def impl(df, values):\n"
    return _gen_init_df(header, df.columns, data_args)


def _gen_init_df(header, columns, data_args, index=None):
    if index is None:
        index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)"

    # using add_consts_to_type with list to avoid const tuple problems
    # TODO: fix type inference for const str
    col_seq = ", ".join("'{}'".format(c) for c in columns)
    col_var = "bodo.utils.typing.add_consts_to_type([{}], {})".format(
        col_seq, col_seq)

    func_text = "{}  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), {}, {})\n".format(
        header, data_args, index, col_var)
    # print(func_text)
    loc_vars = {}
    exec(func_text, {'bodo': bodo, 'np': np}, loc_vars)
    impl = loc_vars['impl']
    return impl
