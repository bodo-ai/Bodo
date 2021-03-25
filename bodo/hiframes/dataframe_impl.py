# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implementation of DataFrame attributes and methods using overload.
"""
import operator
from collections import namedtuple

import llvmlite.llvmpy.core as lc
import numba
import numpy as np
import pandas as pd
from numba.core import cgutils, ir, types
from numba.core.imputils import (
    RefType,
    impl_ret_borrowed,
    impl_ret_new_ref,
    iternext_impl,
    lower_builtin,
)
from numba.core.ir_utils import mk_unique_var, next_label
from numba.core.typing import signature
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.extending import (
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import (
    DataFrameType,
    handle_inplace_df_type_change,
)
from bodo.hiframes.pd_series_ext import (
    SeriesType,
    _get_series_array_type,
    if_series_to_array_type,
)
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.utils.transform import gen_const_tup
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    get_overload_const_int,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_constant_dict,
    is_common_scalar_dtype,
    is_overload_constant_bool,
    is_overload_constant_dict,
    is_overload_constant_int,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_false,
    is_overload_int,
    is_overload_none,
    is_overload_true,
    parse_dtype,
    raise_bodo_error,
    scalar_to_array_type,
    unliteral_val,
)
from bodo.utils.utils import is_array_typ


@overload_attribute(DataFrameType, "index", inline="always")
def overload_dataframe_index(df):
    return lambda df: bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)


@overload_attribute(DataFrameType, "columns", inline="always")
def overload_dataframe_columns(df):
    func_text = "def impl(df):\n"
    if all(isinstance(a, str) for a in df.columns):
        str_arr = f"bodo.utils.conversion.coerce_to_array({df.columns})"
        func_text += (
            f"  return bodo.hiframes.pd_index_ext.init_string_index({str_arr})\n"
        )
    elif all(isinstance(a, (int, float)) for a in df.columns):  # pragma: no cover
        # TODO(ehsan): test
        arr = f"bodo.utils.conversion.coerce_to_array({df.columns})"
        func_text += f"  return bodo.hiframes.pd_index_ext.init_numeric_index({arr})\n"
    else:
        func_text += (
            f"  return bodo.hiframes.pd_index_ext.init_heter_index({df.columns})\n"
        )
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_attribute(DataFrameType, "values")
def overload_dataframe_values(df):
    # TODO: error checking to make sure df only has numerical values
    n_cols = len(df.columns)
    # convert nullable int columns to float to match Pandas behavior
    nullable_int_cols = set(
        i for i in range(n_cols) if isinstance(df.data[i], IntegerArrayType)
    )
    data_args = ", ".join(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}){}".format(
            i, ".astype(float)" if i in nullable_int_cols else ""
        )
        for i in range(n_cols)
    )
    func_text = "def f(df):\n".format()
    func_text += "    return np.stack(({},), 1)\n".format(data_args)

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    f = loc_vars["f"]
    return f


@overload_method(DataFrameType, "to_numpy", inline="always", no_unliteral=True)
def overload_dataframe_to_numpy(df, dtype=None, copy=False):
    # The copy argument can be ignored here since we always copy the data
    # (our underlying structures are fully columnar which should be copied to get a
    # matrix). This is consistent with Pandas since copy=False doesn't guarantee it
    # won't be copied.

    args_dict = {
        "dtype": dtype,
    }

    args_default_dict = {
        "dtype": None,
    }

    check_unsupported_args("to_numpy", args_dict, args_default_dict)

    def impl(df, dtype=None, copy=False):  # pragma: no cover
        return df.values

    return impl


@overload_attribute(DataFrameType, "ndim", inline="always")
def overload_dataframe_ndim(df):
    return lambda df: 2


@overload_attribute(DataFrameType, "size")
def overload_dataframe_size(df):
    ncols = len(df.columns)
    return lambda df: ncols * len(df)


@overload_attribute(DataFrameType, "shape")
def overload_dataframe_shape(df):
    ncols = len(df.columns)
    # using types.int64 due to lowering error (a Numba tuple handling bug)
    return lambda df: (len(df), types.int64(ncols))


@overload_attribute(DataFrameType, "empty")
def overload_dataframe_empty(df):
    if len(df.columns) == 0:
        return lambda df: True
    return lambda df: len(df) == 0


@overload_method(DataFrameType, "assign", no_unliteral=True)
def overload_dataframe_assign(df, **kwargs):
    # raise error to let typing pass handle it, since **kwargs is not supported in
    # overload
    raise_bodo_error("Invalid df.assign() call")


def _get_dtype_str(dtype):
    """return string representation of dtype value"""
    # function cases like str
    if isinstance(dtype, types.Function):  # pragma: no cover
        if dtype.key[0] == str:
            return "'str'"
        elif dtype.key[0] == float:
            return "float"
        elif dtype.key[0] == int:
            return "int"
        elif dtype.key[0] == np.bool:
            return "bool"
        else:
            raise BodoError(f"invalid dtype: {dtype}")
    # cases like np.float32
    if isinstance(dtype, types.functions.NumberClass):
        return f"'{dtype.key}'"
    return f"'{dtype}'"


@overload_method(DataFrameType, "astype", inline="always", no_unliteral=True)
def overload_dataframe_astype(df, dtype, copy=True, errors="raise"):
    # check unsupported arguments
    args_dict = {
        "copy": copy,
        "errors": errors,
    }
    args_default_dict = {"copy": True, "errors": "raise"}
    check_unsupported_args("df.astype", args_dict, args_default_dict)

    # just call astype() on all column Series
    # TODO: support categorical, dt64, etc.
    if is_overload_constant_dict(dtype):
        dtype_const = get_overload_constant_dict(dtype)
        data_args = ", ".join(
            f"df.iloc[:, {i}].astype({_get_dtype_str(dtype_const[c])}).values"
            if c in dtype_const
            else f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})"
            for i, c in enumerate(df.columns)
        )
    else:
        data_args = ", ".join(
            f"df.iloc[:, {i}].astype(dtype).values" for i in range(len(df.columns))
        )

    header = "def impl(df, dtype, copy=True, errors='raise'):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "copy", inline="always", no_unliteral=True)
def overload_dataframe_copy(df, deep=True):
    # just call copy() on all arrays
    data_outs = []
    for i in range(len(df.columns)):
        arr = f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})"
        if is_overload_true(deep):
            data_outs.append(arr + ".copy()")
        elif is_overload_false(deep):
            data_outs.append(arr)
        else:
            data_outs.append(f"{arr}.copy() if deep else {arr}")

    header = "def impl(df, deep=True):\n"
    return _gen_init_df(header, df.columns, ", ".join(data_outs))


@overload_method(DataFrameType, "rename", inline="always", no_unliteral=True)
def overload_dataframe_rename(
    df,
    mapper=None,
    index=None,
    columns=None,
    axis=None,
    copy=True,
    inplace=False,
    level=None,
    errors="ignore",
    _bodo_transformed=False,
):

    handle_inplace_df_type_change(inplace, _bodo_transformed, "rename")

    # check unsupported arguments
    args_dict = {
        "index": index,
        "level": level,
        "errors": errors,
    }
    args_default_dict = {"index": None, "level": None, "errors": "ignore"}

    check_unsupported_args("DataFrame.rename", args_dict, args_default_dict)

    if not (is_overload_constant_bool(inplace)):
        raise BodoError(
            "DataFrame.rename(): 'inplace' keyword only supports boolean constant assignment"
        )

    # columns should be constant dictionary
    if not is_overload_none(mapper):
        if not is_overload_none(columns):
            raise BodoError(
                "DataFrame.rename(): Cannot specify both 'mapper' and 'columns'"
            )
        if not (is_overload_constant_int(axis) and get_overload_const_int(axis) == 1):
            raise BodoError("DataFrame.rename(): 'mapper' only supported with axis=1")
        if not is_overload_constant_dict(mapper):
            raise_bodo_error(
                "'mapper' argument to DataFrame.rename() should be a constant dictionary"
            )

        col_map = get_overload_constant_dict(mapper)

    elif not is_overload_none(columns):
        if not is_overload_none(axis):
            raise BodoError(
                "DataFrame.rename(): Cannot specify both 'axis' and 'columns'"
            )
        if not is_overload_constant_dict(columns):
            raise_bodo_error(
                "'columns' argument to DataFrame.rename() should be a constant dictionary"
            )

        col_map = get_overload_constant_dict(columns)
    else:
        raise_bodo_error(
            "DataFrame.rename(): must pass columns either via 'mapper' and 'axis'=1 or 'columns'"
        )
    new_cols = [
        col_map.get(df.columns[i], df.columns[i]) for i in range(len(df.columns))
    ]

    data_outs = []
    for i in range(len(df.columns)):
        arr = f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})"
        if is_overload_true(copy):
            data_outs.append(arr + ".copy()")
        elif is_overload_false(copy):
            data_outs.append(arr)
        else:
            data_outs.append(f"{arr}.copy() if copy else {arr}")

    header = (
        "def impl(df, mapper=None, index=None, columns=None, axis=None, "
        "copy=True, inplace=False, level=None, errors='ignore', _bodo_transformed=False):\n"
    )
    return _gen_init_df(header, new_cols, ", ".join(data_outs))


@overload_method(DataFrameType, "isna", inline="always", no_unliteral=True)
@overload_method(DataFrameType, "isnull", inline="always", no_unliteral=True)
def overload_dataframe_isna(df):
    # call isna() on column Series
    data_args = ", ".join(
        f"df.iloc[:, {i}].isna().values" for i in range(len(df.columns))
    )
    header = "def impl(df):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "select_dtypes", inline="always", no_unliteral=True)
def overload_dataframe_select_dtypes(df, include=None, exclude=None):
    # Check that at least one of include or exclude exists
    include_none = is_overload_none(include)
    exclude_none = is_overload_none(exclude)

    if include_none and exclude_none:
        raise_bodo_error(
            "DataFrame.select_dtypes() At least one of include or exclude must not be none"
        )

    def is_legal_input(elem):
        # TODO(Nick): Replace with the correct type check
        return (
            is_overload_constant_str(elem)
            or isinstance(elem, types.DTypeSpec)
            or isinstance(elem, types.Function)
        )

    if not include_none:
        # If the input is a list process each elem in the list
        if is_overload_constant_list(include):
            include = get_overload_const_list(include)
            include_types = [
                _get_series_array_type(parse_dtype(elem)) for elem in include
            ]
        # If its a scalar then just make it a list of 1 element
        elif is_legal_input(include):
            include_types = [_get_series_array_type(parse_dtype(include))]
        else:
            raise_bodo_error(
                "DataFrame.select_dtypes() only supports constant strings or types as arguments"
            )
        # Filter columns to those with a matching datatype
        # TODO(Nick): Add more general support for type rules:
        # ex. np.number for all numeric types, np.object for all obj types,
        # "string" for all string types
        chosen_columns = tuple(
            c for i, c in enumerate(df.columns) if df.data[i] in include_types
        )
    else:
        chosen_columns = df.columns
    if not exclude_none:
        # If the input is a list process each elem in the list
        if is_overload_constant_list(exclude):
            exclude = get_overload_const_list(exclude)
            exclude_types = [
                _get_series_array_type(parse_dtype(elem)) for elem in exclude
            ]
        # If its a scalar then just make it a list of 1 element
        elif is_legal_input(exclude):
            exclude_types = [_get_series_array_type(parse_dtype(exclude))]
        else:
            raise_bodo_error(
                "DataFrame.select_dtypes() only supports constant strings or types as arguments"
            )
        # Filter columns to those without a matching datatype
        # TODO(Nick): Add more general support for type rules:
        # ex. np.number for all numeric types, np.object for all obj types,
        # "string" for all string types
        chosen_columns = tuple(
            c for i, c in enumerate(chosen_columns) if df.data[i] not in exclude_types
        )

    data_args = ", ".join(
        f"df.iloc[:, {df.columns.index(c)}].values" for c in chosen_columns
    )
    # Define our function
    header = "def impl(df, include=None, exclude=None):\n"

    return _gen_init_df(header, chosen_columns, data_args)


@overload_method(DataFrameType, "notna", inline="always", no_unliteral=True)
@overload_method(DataFrameType, "notnull", inline="always", no_unliteral=True)
def overload_dataframe_notna(df):
    # call notna() on column Series
    data_args = ", ".join(
        f"df.iloc[:, {i}].notna().values" for i in range(len(df.columns))
    )
    header = "def impl(df):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "head", inline="always", no_unliteral=True)
def overload_dataframe_head(df, n=5):
    # n must be an integer for indexing.
    if not is_overload_int(n):
        raise BodoError("Dataframe.head(): 'n' must be an Integer")

    # call head() on column Series
    data_args = ", ".join(
        f"df.iloc[:, {i}].head(n).values" for i in range(len(df.columns))
    )
    header = "def impl(df, n=5):\n"
    index = "bodo.allgatherv(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[:n], False)"
    return _gen_init_df(header, df.columns, data_args, index)


@overload_method(DataFrameType, "tail", inline="always", no_unliteral=True)
def overload_dataframe_tail(df, n=5):
    # n must be an integer for indexing.
    if not is_overload_int(n):
        raise BodoError("Dataframe.tail(): 'n' must be an Integer")
    # call tail() on column Series
    data_args = ", ".join(
        f"df.iloc[:, {i}].tail(n).values" for i in range(len(df.columns))
    )
    header = "def impl(df, n=5):\n"
    index = "bodo.allgatherv(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[-n:], False)"
    return _gen_init_df(header, df.columns, data_args, index)


@overload_method(DataFrameType, "to_string", no_unliteral=True)
def to_string_overload(
    df,
    buf=None,
    columns=None,
    col_space=None,
    header=True,
    index=True,
    na_rep="NaN",
    formatters=None,
    float_format=None,
    sparsify=None,
    index_names=True,
    justify=None,
    max_rows=None,
    min_rows=None,
    max_cols=None,
    show_dimensions=False,
    decimal=".",
    line_width=None,
    max_colwidth=None,
    encoding=None,
):
    def impl(
        df,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        max_rows=None,
        min_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal=".",
        line_width=None,
        max_colwidth=None,
        encoding=None,
    ):  # pragma: no cover
        with numba.objmode(res="string"):
            res = df.to_string(
                buf=buf,
                columns=columns,
                col_space=col_space,
                header=header,
                index=index,
                na_rep=na_rep,
                formatters=formatters,
                float_format=float_format,
                sparsify=sparsify,
                index_names=index_names,
                justify=justify,
                max_rows=max_rows,
                min_rows=min_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                decimal=decimal,
                line_width=line_width,
                max_colwidth=max_colwidth,
                encoding=encoding,
            )
        return res

    return impl


@overload_method(DataFrameType, "isin", inline="always", no_unliteral=True)
def overload_dataframe_isin(df, values):
    # TODO: call isin on Series
    # TODO: make sure df indices match?
    # TODO: dictionary case

    func_text = "def impl(df, values):\n"

    other_colmap = {}
    df_case = False
    # dataframe case
    if isinstance(values, DataFrameType):
        df_case = True
        for i, c in enumerate(df.columns):
            if c in values.columns:
                v_name = "val{}".format(i)
                func_text += "  {} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(values, {})\n".format(
                    v_name, values.columns.index(c)
                )
                other_colmap[c] = v_name
    else:
        # general iterable (e.g. list, set) case
        # TODO: handle passed in dict case (pass colname to func?)
        other_colmap = {c: "values" for c in df.columns}

    data = []
    for i in range(len(df.columns)):
        v_name = "data{}".format(i)
        func_text += (
            "  {} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})\n".format(
                v_name, i
            )
        )
        data.append(v_name)

    out_data = ["out{}".format(i) for i in range(len(df.columns))]

    isin_func = """
  numba.parfors.parfor.init_prange()
  n = len({0})
  m = len({1})
  {2} = np.empty(n, np.bool_)
  for i in numba.parfors.parfor.internal_prange(n):
    {2}[i] = {0}[i] == {1}[i] if i < m else False
"""

    isin_vals_func = """
  numba.parfors.parfor.init_prange()
  n = len({0})
  {2} = np.empty(n, np.bool_)
  for i in numba.parfors.parfor.internal_prange(n):
    {2}[i] = {0}[i] in {1}
"""
    bool_arr_func = "  {} = np.zeros(len(df), np.bool_)\n"
    for i, (cname, in_var) in enumerate(zip(df.columns, data)):
        if cname in other_colmap:
            other_col_var = other_colmap[cname]
            if df_case:
                func_text += isin_func.format(in_var, other_col_var, out_data[i])
            else:
                func_text += isin_vals_func.format(in_var, other_col_var, out_data[i])
        else:
            func_text += bool_arr_func.format(out_data[i])

    return _gen_init_df(func_text, df.columns, ",".join(out_data))


@overload_method(DataFrameType, "abs", inline="always", no_unliteral=True)
def overload_dataframe_abs(df):
    # only works for numerical data and Timedelta
    # TODO: handle timedelta

    n_cols = len(df.columns)
    data_args = ", ".join(
        "np.abs(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}))".format(i)
        for i in range(n_cols)
    )
    header = "def impl(df):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "corr", inline="always", no_unliteral=True)
def overload_dataframe_corr(df, method="pearson", min_periods=1):

    unsupported_args = dict(method=method)
    arg_defaults = dict(method="pearson")
    check_unsupported_args("DataFrame.corr", unsupported_args, arg_defaults)

    numeric_cols = [
        c for c, d in zip(df.columns, df.data) if _is_numeric_dtype(d.dtype)
    ]
    # TODO: support empty dataframe
    assert len(numeric_cols) != 0

    # convert input matrix to float64 if necessary
    typ_conv = ""
    if not any(d == types.float64 for d in df.data):
        typ_conv = ".astype(np.float64)"

    arr_args = ", ".join(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}){}".format(
            df.columns.index(c),
            ".astype(np.float64)"
            if (
                isinstance(df.data[df.columns.index(c)], IntegerArrayType)
                or df.data[df.columns.index(c)] == boolean_array
            )
            else "",
        )
        for c in numeric_cols
    )
    mat = "np.stack(({},), 1){}".format(arr_args, typ_conv)

    data_args = ", ".join("res[:,{}]".format(i) for i in range(len(numeric_cols)))
    index = f"pd.Index({numeric_cols})\n"

    header = "def impl(df, method='pearson', min_periods=1):\n"
    header += "  mat = {}\n".format(mat)
    header += "  res = bodo.libs.array_kernels.nancorr(mat, 0, min_periods)\n"
    return _gen_init_df(header, numeric_cols, data_args, index)


@overload_method(DataFrameType, "cov", inline="always", no_unliteral=True)
def overload_dataframe_cov(df, min_periods=None, ddof=1):

    unsupported_args = dict(ddof=ddof)
    arg_defaults = dict(ddof=1)
    check_unsupported_args("DataFrame.cov", unsupported_args, arg_defaults)

    # TODO: support calling np.cov() when there is no NA
    minpv = "1" if is_overload_none(min_periods) else "min_periods"

    numeric_cols = [
        c for c, d in zip(df.columns, df.data) if _is_numeric_dtype(d.dtype)
    ]
    # TODO: support empty dataframe
    assert len(numeric_cols) != 0

    # convert input matrix to float64 if necessary
    typ_conv = ""
    if not any(d == types.float64 for d in df.data):
        typ_conv = ".astype(np.float64)"

    arr_args = ", ".join(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}){}".format(
            df.columns.index(c),
            ".astype(np.float64)"
            if (
                isinstance(df.data[df.columns.index(c)], IntegerArrayType)
                or df.data[df.columns.index(c)] == boolean_array
            )
            else "",
        )
        for c in numeric_cols
    )
    mat = "np.stack(({},), 1){}".format(arr_args, typ_conv)

    data_args = ", ".join("res[:,{}]".format(i) for i in range(len(numeric_cols)))
    index = f"pd.Index({numeric_cols})\n"

    header = "def impl(df, min_periods=None, ddof=1):\n"
    header += "  mat = {}\n".format(mat)
    header += "  res = bodo.libs.array_kernels.nancorr(mat, 1, {})\n".format(minpv)
    return _gen_init_df(header, numeric_cols, data_args, index)


@overload_method(DataFrameType, "count", inline="always", no_unliteral=True)
def overload_dataframe_count(df, axis=0, level=None, numeric_only=False):
    # TODO: numeric_only flag
    data_args = ", ".join(f"df.iloc[:, {i}].count()" for i in range(len(df.columns)))

    func_text = "def impl(df, axis=0, level=None, numeric_only=False):\n"
    func_text += "  data = np.array([{}])\n".format(data_args)
    func_text += "  return bodo.hiframes.pd_series_ext.init_series(data, df.columns)\n"
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(DataFrameType, "nunique", inline="always", no_unliteral=True)
def overload_dataframe_nunique(df, axis=0, dropna=True):
    unsupported_args = dict(axis=0, dropna=dropna)
    arg_defaults = dict(axis=0, dropna=True)
    check_unsupported_args("DataFrame.nunique", unsupported_args, arg_defaults)
    data_args = ", ".join(f"df.iloc[:, {i}].nunique()" for i in range(len(df.columns)))
    func_text = "def impl(df, axis=0, dropna=True):\n"
    func_text += "  data = np.asarray(({},))\n".format(data_args)
    func_text += "  return bodo.hiframes.pd_series_ext.init_series(data, df.columns)\n"
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(DataFrameType, "prod", inline="always", no_unliteral=True)
@overload_method(DataFrameType, "product", inline="always", no_unliteral=True)
def overload_dataframe_prod(
    df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0
):

    unsupported_args = dict(
        skipna=skipna, level=level, numeric_only=numeric_only, min_count=min_count
    )
    arg_defaults = dict(skipna=None, level=None, numeric_only=None, min_count=0)
    check_unsupported_args("DataFrame.prod/product", unsupported_args, arg_defaults)

    return _gen_reduce_impl(df, "prod", axis=axis)


@overload_method(DataFrameType, "sum", inline="always", no_unliteral=True)
def overload_dataframe_sum(
    df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0
):

    unsupported_args = dict(
        skipna=skipna, level=level, numeric_only=numeric_only, min_count=min_count
    )
    arg_defaults = dict(skipna=None, level=None, numeric_only=None, min_count=0)
    check_unsupported_args("DataFrame.sum", unsupported_args, arg_defaults)

    return _gen_reduce_impl(df, "sum", axis=axis)


@overload_method(DataFrameType, "max", inline="always", no_unliteral=True)
def overload_dataframe_max(df, axis=None, skipna=None, level=None, numeric_only=None):

    unsupported_args = dict(skipna=skipna, level=level, numeric_only=numeric_only)
    arg_defaults = dict(skipna=None, level=None, numeric_only=None)
    check_unsupported_args("DataFrame.max", unsupported_args, arg_defaults)

    return _gen_reduce_impl(df, "max", axis=axis)


@overload_method(DataFrameType, "min", inline="always", no_unliteral=True)
def overload_dataframe_min(df, axis=None, skipna=None, level=None, numeric_only=None):

    unsupported_args = dict(skipna=skipna, level=level, numeric_only=numeric_only)
    arg_defaults = dict(skipna=None, level=None, numeric_only=None)
    check_unsupported_args("DataFrame.min", unsupported_args, arg_defaults)

    return _gen_reduce_impl(df, "min", axis=axis)


@overload_method(DataFrameType, "mean", inline="always", no_unliteral=True)
def overload_dataframe_mean(df, axis=None, skipna=None, level=None, numeric_only=None):

    unsupported_args = dict(skipna=skipna, level=level, numeric_only=numeric_only)
    arg_defaults = dict(skipna=None, level=None, numeric_only=None)
    check_unsupported_args("DataFrame.mean", unsupported_args, arg_defaults)

    return _gen_reduce_impl(df, "mean", axis=axis)


@overload_method(DataFrameType, "var", inline="always", no_unliteral=True)
def overload_dataframe_var(
    df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None
):

    unsupported_args = dict(
        skipna=skipna, level=level, ddof=ddof, numeric_only=numeric_only
    )
    arg_defaults = dict(skipna=None, level=None, ddof=1, numeric_only=None)
    check_unsupported_args("DataFrame.mean", unsupported_args, arg_defaults)

    return _gen_reduce_impl(df, "var", axis=axis)


@overload_method(DataFrameType, "std", inline="always", no_unliteral=True)
def overload_dataframe_std(
    df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None
):

    unsupported_args = dict(
        skipna=skipna, level=level, ddof=ddof, numeric_only=numeric_only
    )
    arg_defaults = dict(skipna=None, level=None, ddof=1, numeric_only=None)
    check_unsupported_args("DataFrame.std", unsupported_args, arg_defaults)

    return _gen_reduce_impl(df, "std", axis=axis)


@overload_method(DataFrameType, "median", inline="always", no_unliteral=True)
def overload_dataframe_median(
    df, axis=None, skipna=None, level=None, numeric_only=None
):

    unsupported_args = dict(skipna=skipna, level=level, numeric_only=numeric_only)
    arg_defaults = dict(skipna=None, level=None, numeric_only=None)
    check_unsupported_args("DataFrame.median", unsupported_args, arg_defaults)

    return _gen_reduce_impl(df, "median", axis=axis)


@overload_method(DataFrameType, "quantile", inline="always", no_unliteral=True)
def overload_dataframe_quantile(
    df, q=0.5, axis=0, numeric_only=True, interpolation="linear"
):

    unsupported_args = dict(numeric_only=numeric_only, interpolation=interpolation)
    arg_defaults = dict(numeric_only=True, interpolation="linear")
    check_unsupported_args("DataFrame.quantile", unsupported_args, arg_defaults)

    # TODO: name is str(q)
    return _gen_reduce_impl(df, "quantile", "q", axis=axis)


@overload_method(DataFrameType, "idxmax", inline="always", no_unliteral=True)
def overload_dataframe_idxmax(df, axis=0, skipna=True):

    # TODO: [BE-281] Support idxmax with axis=1
    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=0, skipna=True)
    check_unsupported_args("DataFrame.idxmax", unsupported_args, arg_defaults)

    # Pandas restrictions:
    # Only supported for numeric types with numpy arrays
    # - int, floats, bool, dt64, td64. (maybe complex)
    # We also support categorical and nullable arrays
    for coltype in df.data:
        if not (
            bodo.utils.utils.is_np_array_typ(coltype)
            and (
                coltype.dtype in [bodo.datetime64ns, bodo.timedelta64ns]
                or isinstance(coltype.dtype, (types.Number, types.Boolean))
            )
            or isinstance(coltype, (bodo.IntegerArrayType, bodo.CategoricalArray))
            or coltype in [bodo.boolean_array, bodo.datetime_date_array_type]
        ):
            raise BodoError(
                f"DataFrame.idxmax() only supported for numeric column types. Column type: {coltype} not supported."
            )
        if isinstance(coltype, bodo.CategoricalArray) and not coltype.dtype.ordered:
            raise BodoError("DataFrame.idxmax(): categorical columns must be ordered")

    return _gen_reduce_impl(df, "idxmax", axis=axis)


@overload_method(DataFrameType, "idxmin", inline="always", no_unliteral=True)
def overload_dataframe_idxmin(df, axis=0, skipna=True):

    # TODO: [BE-281] Support idxmin with axis=1
    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=0, skipna=True)
    check_unsupported_args("DataFrame.idxmin", unsupported_args, arg_defaults)

    # Pandas restrictions:
    # Only supported for numeric types with numpy arrays
    # - int, floats, bool, dt64, td64. (maybe complex)
    # We also support categorical and nullable arrays
    for coltype in df.data:
        if not (
            bodo.utils.utils.is_np_array_typ(coltype)
            and (
                coltype.dtype in [bodo.datetime64ns, bodo.timedelta64ns]
                or isinstance(coltype.dtype, (types.Number, types.Boolean))
            )
            or isinstance(coltype, (bodo.IntegerArrayType, bodo.CategoricalArray))
            or coltype in [bodo.boolean_array, bodo.datetime_date_array_type]
        ):
            raise BodoError(
                f"DataFrame.idxmin() only supported for numeric column types. Column type: {coltype} not supported."
            )
        if isinstance(coltype, bodo.CategoricalArray) and not coltype.dtype.ordered:
            raise BodoError("DataFrame.idxmin(): categorical columns must be ordered")

    return _gen_reduce_impl(df, "idxmin", axis=axis)


def _gen_reduce_impl(df, func_name, args=None, axis=None):
    """generate implementation for dataframe reduction functions like min, max, sum, ..."""
    args = "" if is_overload_none(args) else args

    # axis is 0 by default. Some reduce functions have None as the default value.
    if is_overload_none(axis):
        axis = 0
    else:
        axis = get_overload_const_int(axis)

    assert axis in (0, 1), "invalid axis argument for DataFrame.{}".format(func_name)

    if func_name in ("idxmax", "idxmin"):
        out_colnames = df.columns
    else:
        # TODO: numeric_only=None tries its best: core/frame.py/DataFrame/_reduce
        numeric_cols = tuple(
            c for c, d in zip(df.columns, df.data) if _is_numeric_dtype(d.dtype)
        )
        out_colnames = numeric_cols

    # TODO: support empty dataframe
    assert len(out_colnames) != 0

    # Ensure that the dtypes can be supported and raise a BodoError if
    # they cannot be combined. This is a safety net and should generally
    # be handled by the individual functions.
    try:
        # TODO: Only generate common types when necessary to prevent errors.
        if func_name in ("idxmax", "idxmin") and axis == 0:
            comm_dtype = None
        else:
            dtypes = [
                numba.np.numpy_support.as_dtype(df.data[df.columns.index(c)].dtype)
                for c in out_colnames
            ]
            # TODO: Determine what possible exceptions this might raise.
            comm_dtype = numba.np.numpy_support.from_dtype(
                np.find_common_type(dtypes, [])
            )
    except NotImplementedError:
        raise BodoError(
            f"Dataframe.{func_name}() with column types: {df.data} could not be merged to a common type."
        )

    # generate function signature
    minc = ""
    if func_name in ("sum", "prod"):
        minc = ", min_count=0"

    ddof = ""
    if func_name in ("var", "std"):
        ddof = "ddof=1, "

    func_text = "def impl(df, axis=None, skipna=None, level=None,{} numeric_only=None{}):\n".format(
        ddof, minc
    )
    if func_name == "quantile":
        func_text = (
            "def impl(df, q=0.5, axis=0, numeric_only=True, interpolation='linear'):\n"
        )
    if func_name in ("idxmax", "idxmin"):
        func_text = "def impl(df, axis=0, skipna=True):\n"

    if axis == 0:
        func_text += _gen_reduce_impl_axis0(
            df, func_name, out_colnames, comm_dtype, args
        )
    else:
        func_text += _gen_reduce_impl_axis1(func_name, out_colnames, comm_dtype, df)

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np, "pd": pd, "numba": numba}, loc_vars)
    impl = loc_vars["impl"]
    return impl


def _gen_reduce_impl_axis0(df, func_name, out_colnames, comm_dtype, args):
    """generate function body for dataframe reduction across rows"""
    # XXX: use common type for min/max to avoid float for ints due to NaN
    # TODO: handle NaN for ints better
    typ_cast = ""
    if func_name in ("min", "max"):
        typ_cast = ", dtype=np.{}".format(comm_dtype)

    # XXX pandas combines all column values so int8/float32 results in float32
    # not float64
    if comm_dtype == types.float32 and func_name in (
        "sum",
        "prod",
        "mean",
        "var",
        "std",
        "median",
    ):
        typ_cast = ", dtype=np.float32"

    data_args = ", ".join(
        f"df.iloc[:, {df.columns.index(c)}].{func_name}({args})" for c in out_colnames
    )

    func_text = ""
    # data conversion
    if func_name in ("idxmax", "idxmin"):
        # idxmax/idxmin don't cast type since just index value is produced
        # but need to convert tuple of Timestamp to dt64 array
        # see idxmax test numeric_df_value[6]
        func_text += "  data = bodo.utils.conversion.coerce_to_array(({},))\n".format(
            data_args
        )
    else:
        func_text += "  data = np.asarray(({},){})\n".format(data_args, typ_cast)
    func_text += f"  return bodo.hiframes.pd_series_ext.init_series(data, pd.Index({out_colnames}))\n"
    return func_text


def _gen_reduce_impl_axis1(func_name, out_colnames, comm_dtype, df_type):
    """generate function body for dataframe reduction across columns"""
    col_inds = [df_type.columns.index(c) for c in out_colnames]
    index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)"
    data_args = "\n    ".join(
        "arr_{0} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {0})".format(i)
        for i in col_inds
    )
    data_accesses = "\n        ".join(
        f"row[{i}] = arr_{col_inds[i]}[i]" for i in range(len(out_colnames))
    )
    # TODO: support empty dataframes
    assert len(data_args) > 0, f"empty dataframe in DataFrame.{func_name}()"
    df_len = f"len(arr_{col_inds[0]})"

    func_np_func_map = {
        "max": "np.nanmax",
        "min": "np.nanmin",
        "sum": "np.nansum",
        "prod": "np.nanprod",
        "mean": "np.nanmean",
        "median": "np.nanmedian",
        # TODO: Handle these cases. Numba doesn't support the
        # ddof argument and pd & np
        # implementations vary (sample vs population)
        #'var': '(lambda A: np.nanvar(A, ddof=1))',
        "var": "bodo.utils.utils.nanvar_ddof1",
        #'std': '(lambda A: np.nanstd(A, ddof=1))',
        "std": "bodo.utils.utils.nanstd_ddof1",
    }

    if func_name in func_np_func_map:
        np_func = func_np_func_map[func_name]
        # NOTE: Pandas outputs float64 output even for int64 dataframes
        # when using df.mean() and df.median()
        # TODO: More sophisticated manner of computing this output_dtype
        output_dtype = (
            "float64" if func_name in ["mean", "median", "std", "var"] else comm_dtype
        )
        func_text = f"""
    {data_args}
    numba.parfors.parfor.init_prange()
    n = {df_len}
    row = np.empty({len(out_colnames)}, np.{comm_dtype})
    A = np.empty(n, np.{output_dtype})
    for i in numba.parfors.parfor.internal_prange(n):
        {data_accesses}
        A[i] = {np_func}(row)
    return bodo.hiframes.pd_series_ext.init_series(A, {index})
"""
        return func_text
    else:
        # Prevent internal error from func_text not existing
        raise BodoError(f"DataFrame.{func_name}(): Not supported for axis=1")


@overload_method(DataFrameType, "pct_change", inline="always", no_unliteral=True)
def overload_dataframe_pct_change(
    df, periods=1, fill_method="pad", limit=None, freq=None
):

    unsupported_args = dict(fill_method=fill_method, limit=limit, freq=freq)
    arg_defaults = dict(fill_method="pad", limit=None, freq=None)
    check_unsupported_args("DataFrame.pct_change", unsupported_args, arg_defaults)

    data_args = ", ".join(
        f"df.iloc[:, {i}].pct_change(periods).values" for i in range(len(df.columns))
    )
    header = "def impl(df, periods=1, fill_method='pad', limit=None, freq=None):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "cumprod", inline="always", no_unliteral=True)
def overload_dataframe_cumprod(df, axis=None, skipna=True):
    unsupported_args = dict(skipna=skipna)
    arg_defaults = dict(skipna=True)
    check_unsupported_args("DataFrame.cumprod", unsupported_args, arg_defaults)

    data_args = ", ".join(
        f"df.iloc[:, {i}].values.cumprod()" for i in range(len(df.columns))
    )
    header = "def impl(df, axis=None, skipna=True):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "cumsum", inline="always", no_unliteral=True)
def overload_dataframe_cumsum(df, axis=None, skipna=True):
    unsupported_args = dict(skipna=skipna)
    arg_defaults = dict(skipna=True)
    check_unsupported_args("DataFrame.cumsum", unsupported_args, arg_defaults)

    data_args = ", ".join(
        f"df.iloc[:, {i}].values.cumsum()" for i in range(len(df.columns))
    )

    header = "def impl(df, axis=None, skipna=True):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "describe", inline="always", no_unliteral=True)
def overload_dataframe_describe(
    df, percentiles=None, include=None, exclude=None, datetime_is_numeric=False
):

    unsupported_args = dict(
        percentiles=percentiles,
        include=include,
        exclude=exclude,
        datetime_is_numeric=datetime_is_numeric,
    )
    arg_defaults = dict(
        percentiles=None, include=None, exclude=None, datetime_is_numeric=False
    )
    check_unsupported_args("DataFrame.describe", unsupported_args, arg_defaults)

    header = "def impl(df, percentiles=None, include=None, exclude=None, datetime_is_numeric=False):\n"
    data_args = ", ".join(
        f"df.iloc[:, {i}].describe().values" for i in range(len(df.columns))
    )
    index = (
        "bodo.utils.conversion.convert_to_index(['count', 'mean', 'std', "
        "'min', '25%', '50%', '75%', 'max'])"
    )
    return _gen_init_df(header, df.columns, data_args, index)


@overload_method(DataFrameType, "take", inline="always", no_unliteral=True)
def overload_dataframe_take(df, indices, axis=0, convert=None, is_copy=True):

    unsupported_args = dict(axis=axis, convert=convert, is_copy=is_copy)
    arg_defaults = dict(axis=0, convert=None, is_copy=True)
    check_unsupported_args("DataFrame.take", unsupported_args, arg_defaults)

    data_args = ", ".join(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})[indices_t]".format(i)
        for i in range(len(df.columns))
    )
    header = "def impl(df, indices, axis=0, convert=None, is_copy=True):\n"
    header += "  indices_t = bodo.utils.conversion.coerce_to_ndarray(indices)\n"
    index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)" "[indices_t]"
    return _gen_init_df(header, df.columns, data_args, index)


@overload_method(DataFrameType, "shift", inline="always", no_unliteral=True)
def overload_dataframe_shift(df, periods=1, freq=None, axis=0, fill_value=None):
    # TODO: handle fill_value, freq, int NA
    # TODO: Support nullable integer/float types
    unsupported_args = dict(freq=freq, axis=axis, fill_value=fill_value)
    arg_defaults = dict(freq=None, axis=0, fill_value=None)
    check_unsupported_args("DataFrame.shift", unsupported_args, arg_defaults)

    # Bodo specific limitations for supported types
    # Currently only float (not nullable), int (not nullable), and dt64 are supported
    for column_type in df.data:
        if not (
            isinstance(column_type, types.Array)
            and (
                isinstance(column_type.dtype, (types.Number))
                or column_type.dtype == bodo.datetime64ns
            )
        ):
            # TODO: Link to supported Column input types.
            raise BodoError(
                f"Dataframe.shift() column input type {column_type.dtype} not supported."
            )

    # Ensure period is int
    if not is_overload_int(periods):
        raise BodoError("DataFrame.shift(): 'periods' input must be an integer.")

    data_args = ", ".join(
        "bodo.hiframes.rolling.shift(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}), periods, False)".format(
            i
        )
        for i in range(len(df.columns))
    )
    header = "def impl(df, periods=1, freq=None, axis=0, fill_value=None):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "diff", inline="always", no_unliteral=True)
def overload_dataframe_diff(df, periods=1, axis=0):
    """DataFrame.diff() support which is the same as df - df.shift(periods)"""
    # TODO: Support nullable integer/float types
    unsupported_args = dict(axis=axis)
    arg_defaults = dict(axis=0)
    check_unsupported_args("DataFrame.diff", unsupported_args, arg_defaults)

    # Bodo specific limitations for supported types
    # Currently only float (not nullable), int (not nullable), and dt64 are supported
    for column_type in df.data:
        if not (
            isinstance(column_type, types.Array)
            and (
                isinstance(column_type.dtype, (types.Number))
                or column_type.dtype == bodo.datetime64ns
            )
        ):
            # TODO: Link to supported Column input types.
            raise BodoError(
                f"DataFrame.diff() column input type {column_type.dtype} not supported."
            )

    # Ensure period is int
    if not is_overload_int(periods):
        raise BodoError("DataFrame.diff(): 'periods' input must be an integer.")

    header = "def impl(df, periods=1, axis= 0):\n"
    for i in range(len(df.columns)):
        header += (
            f"  data_{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})\n"
        )
    data_args = ", ".join(
        # NOTE: using our sub function for dt64 due to bug in Numba (TODO: fix)
        f"bodo.hiframes.series_impl.dt64_arr_sub(data_{i}, bodo.hiframes.rolling.shift(data_{i}, periods, False))"
        if df.data[i] == types.Array(bodo.datetime64ns, 1, "C")
        else f"data_{i} - bodo.hiframes.rolling.shift(data_{i}, periods, False)"
        for i in range(len(df.columns))
    )
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "set_index", inline="always", no_unliteral=True)
def overload_dataframe_set_index(
    df, keys, drop=True, append=False, inplace=False, verify_integrity=False
):
    args_dict = {
        "inplace": inplace,
        "append": append,
        "verify_integrity": verify_integrity,
    }
    args_default_dict = {"inplace": False, "append": False, "verify_integrity": False}

    check_unsupported_args("DataFrame.set_index", args_dict, args_default_dict)

    # Column name only supproted on constant string
    if not is_overload_constant_str(keys):
        raise_bodo_error("DataFrame.set_index(): 'keys' must be a constant string")
    col_name = get_overload_const_str(keys)
    col_ind = df.columns.index(col_name)
    if isinstance(df.data[col_ind], bodo.CategoricalArray):
        raise BodoError("DataFrame.set_index(): Not supported for categorical columns.")
    if len(df.columns) == 1:
        raise BodoError(
            "DataFrame.set_index(): Not supported on single column DataFrames."
        )

    data_args = ", ".join(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})".format(i)
        for i in range(len(df.columns))
        if i != col_ind
    )
    header = "def impl(df, keys, drop=True, append=False, inplace=False, verify_integrity=False):\n"
    columns = tuple(c for c in df.columns if c != col_name)
    index = (
        "bodo.utils.conversion.index_from_array("
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}), {})"
    ).format(col_ind, f"'{col_name}'" if isinstance(col_name, str) else col_name)
    return _gen_init_df(header, columns, data_args, index)


@overload_method(DataFrameType, "query", no_unliteral=True)
def overload_dataframe_query(df, expr, inplace=False):
    """Support query only for the case where expr is a constant string and expr output
    is a 1D boolean array.
    Refering to named index by name is not supported.
    Series.dt.* is not supported. issue #451
    """
    # check unsupported "inplace"
    args_dict = {
        "inplace": inplace,
    }
    args_default_dict = {
        "inplace": False,
    }

    check_unsupported_args("query", args_dict, args_default_dict)

    # check expr is of type string
    if not isinstance(expr, (types.StringLiteral, types.UnicodeType)):
        raise BodoError("query(): expr argument should be a string")

    # TODO: support df.loc for normal case and getitem for multi-dim case similar to
    # Pandas
    def impl(df, expr, inplace=False):  # pragma: no cover
        b = bodo.hiframes.pd_dataframe_ext.query_dummy(df, expr)
        return df[b]

    return impl


@overload_method(DataFrameType, "duplicated", inline="always", no_unliteral=True)
def overload_dataframe_duplicated(df, subset=None, keep="first"):
    # TODO: support subset and first
    args_dict = {
        "subset": subset,
        "keep": keep,
    }
    args_default_dict = {
        "subset": None,
        "keep": "first",
    }

    check_unsupported_args("DataFrame.duplicated", args_dict, args_default_dict)

    n_cols = len(df.columns)

    func_text = "def impl(df, subset=None, keep='first'):\n"
    for i in range(n_cols):
        func_text += "  data_{0} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {0})\n".format(
            i
        )
    index = "bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df))"
    func_text += "  duplicated, index_arr = bodo.libs.array_kernels.duplicated(({},), {})\n".format(
        ", ".join("data_{}".format(i) for i in range(n_cols)), index
    )
    func_text += "  index = bodo.utils.conversion.index_from_array(index_arr)\n"
    func_text += "  return bodo.hiframes.pd_series_ext.init_series(duplicated, index)\n"
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(DataFrameType, "drop_duplicates", inline="always", no_unliteral=True)
def overload_dataframe_drop_duplicates(
    df, subset=None, keep="first", inplace=False, ignore_index=False
):
    # TODO: support inplace
    args_dict = {
        "keep": keep,
        "inplace": inplace,
        "subset": subset,
        "ignore_index": ignore_index,
    }
    args_default_dict = {
        "keep": "first",
        "inplace": False,
        "subset": None,
        "ignore_index": False,
    }

    check_unsupported_args("DataFrame.drop_duplicates", args_dict, args_default_dict)

    # XXX: can't reuse duplicated() here since it shuffles data and chunks
    # may not match

    n_cols = len(df.columns)

    data_args = ", ".join("data_{}".format(i) for i in range(n_cols))

    func_text = (
        "def impl(df, subset=None, keep='first', inplace=False, ignore_index=False):\n"
    )
    for i in range(n_cols):
        func_text += "  data_{0} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {0})\n".format(
            i
        )
    index = "bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df))"
    func_text += "  ({0},), index_arr = bodo.libs.array_kernels.drop_duplicates(({0},), {1})\n".format(
        data_args, index
    )
    func_text += "  index = bodo.utils.conversion.index_from_array(index_arr)\n"
    return _gen_init_df(func_text, df.columns, data_args, "index")


def _gen_init_df(header, columns, data_args, index=None, extra_globals=None):
    if index is None:
        index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)"

    if extra_globals is None:
        extra_globals = {}

    col_var = gen_const_tup(columns)
    data_args = "({}{})".format(data_args, "," if len(columns) == 1 else "")

    func_text = (
        "{}  return bodo.hiframes.pd_dataframe_ext.init_dataframe({}, {}, {})\n".format(
            header, data_args, index, col_var
        )
    )
    loc_vars = {}
    _global = {"bodo": bodo, "np": np, "pd": pd, "numba": numba}
    _global.update(extra_globals)
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


def _is_numeric_dtype(dtype):
    # Pandas considers bool numeric as well: core/internals/blocks
    return isinstance(dtype, types.Number) or dtype == types.bool_


############################ binary operators #############################


def create_binary_op_overload(op):
    def overload_dataframe_binary_op(left, right):
        op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
        # Handle equality specially because we can determine the result
        # when there are mismatched types.
        eq_ops = (operator.eq, operator.ne)
        if isinstance(left, DataFrameType):
            # df/df case
            if isinstance(right, DataFrameType):
                if left != right:
                    raise TypeError(
                        "Inconsistent dataframe schemas in binary operator {} ({} and {})".format(
                            op, left, right
                        )
                    )

                data_args = ", ".join(
                    (
                        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(left, {0}) {1}"
                        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(right, {0})"
                    ).format(i, op_str)
                    for i in range(len(left.columns))
                )
                header = "def impl(left, right):\n"
                index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(left)"
                return _gen_init_df(header, left.columns, data_args, index)

            # scalar case, TODO: Proper error handling for all operators
            # TODO: Test with ArrayItemArrayType
            data_impl = []
            # For each array with a different type, we generate an array
            # of all True/False using a prange. This is because np.full
            # support can have parallelism issues in Numba.
            diff_types = []
            # TODO: What is the best way to place these constants in the code.
            if op in eq_ops:
                for i, col in enumerate(left.data):
                    # If the types don't match, generate an array of False/True values
                    if is_common_scalar_dtype([col.dtype, right]):
                        data_impl.append(
                            f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(left, {i}) {op_str} right"
                        )
                    else:
                        arr_name = f"arr{i}"
                        diff_types.append(arr_name)
                        data_impl.append(arr_name)
                data_args = ", ".join(data_impl)
            else:
                data_args = ", ".join(
                    "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(left, {0}) {1} right".format(
                        i, op_str
                    )
                    for i in range(len(left.columns))
                )

            header = "def impl(left, right):\n"
            if len(diff_types) > 0:
                header += "  numba.parfors.parfor.init_prange()\n"
                header += "  n = len(left)\n"
                header += "".join(
                    "  {0} = np.empty(n, dtype=np.bool_)\n".format(arr_name)
                    for arr_name in diff_types
                )
                header += "  for i in numba.parfors.parfor.internal_prange(n):\n"
                header += "".join(
                    "    {0}[i] = {1}\n".format(arr_name, op == operator.ne)
                    for arr_name in diff_types
                )
            index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(left)"
            return _gen_init_df(header, left.columns, data_args, index)

        if isinstance(right, DataFrameType):
            # scalar case, TODO: Proper error handling for all operators
            # TODO: Test with ArrayItemArrayType
            data_impl = []
            # For each array with a different type, we generate an array
            # of all True/False using a prange. This is because np.full
            # support can have parallelism issues in Numba.
            diff_types = []
            # TODO: What is the best way to place these constants in the code.
            if op in eq_ops:
                for i, col in enumerate(right.data):
                    # If the types don't match, generate an array of False/True values
                    if is_common_scalar_dtype([left, col.dtype]):
                        data_impl.append(
                            f"left {op_str} bodo.hiframes.pd_dataframe_ext.get_dataframe_data(right, {i})"
                        )
                    else:
                        arr_name = f"arr{i}"
                        diff_types.append(arr_name)
                        data_impl.append(arr_name)
                data_args = ", ".join(data_impl)
            else:
                data_args = ", ".join(
                    "left {1} bodo.hiframes.pd_dataframe_ext.get_dataframe_data(right, {0})".format(
                        i, op_str
                    )
                    for i in range(len(right.columns))
                )
            header = "def impl(left, right):\n"
            if len(diff_types) > 0:
                header += "  numba.parfors.parfor.init_prange()\n"
                header += "  n = len(right)\n"
                header += "".join(
                    "  {0} = np.empty(n, dtype=np.bool_)\n".format(arr_name)
                    for arr_name in diff_types
                )
                header += "  for i in numba.parfors.parfor.internal_prange(n):\n"
                header += "".join(
                    "    {0}[i] = {1}\n".format(arr_name, op == operator.ne)
                    for arr_name in diff_types
                )
            index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(right)"
            return _gen_init_df(header, right.columns, data_args, index)

    return overload_dataframe_binary_op


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in bodo.hiframes.pd_series_ext.series_binary_ops:
        overload_impl = create_binary_op_overload(op)
        overload(op, no_unliteral=True)(overload_impl)


_install_binary_ops()


####################### binary inplace operators #############################


def create_inplace_binary_op_overload(op):
    def overload_dataframe_inplace_binary_op(left, right):
        if isinstance(left, DataFrameType):
            op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
            if isinstance(right, DataFrameType):
                if left != right:
                    raise TypeError(
                        "Inconsistent dataframe schemas in binary operator {} ({} and {})".format(
                            op, left, right
                        )
                    )

                func_text = "def impl(left, right):\n"
                for i in range(len(left.columns)):
                    func_text += "  df_arr{0} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(left, {0})\n".format(
                        i
                    )
                    func_text += "  df_arr{0} {1} bodo.hiframes.pd_dataframe_ext.get_dataframe_data(right, {0})\n".format(
                        i, op_str
                    )
                data_args = ", ".join(
                    ("df_arr{}").format(i) for i in range(len(left.columns))
                )
                index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(left)"
                return _gen_init_df(func_text, left.columns, data_args, index)

            # scalar case
            func_text = "def impl(left, right):\n"
            for i in range(len(left.columns)):
                func_text += "  df_arr{0} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(left, {0})\n".format(
                    i
                )
                func_text += "  df_arr{0} {1} right\n".format(i, op_str)
            data_args = ", ".join(
                ("df_arr{}").format(i) for i in range(len(left.columns))
            )
            index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(left)"
            return _gen_init_df(func_text, left.columns, data_args, index)

    return overload_dataframe_inplace_binary_op


def _install_inplace_binary_ops():
    # install inplace binary ops such as iadd, isub, ...
    for op in bodo.hiframes.pd_series_ext.series_inplace_binary_ops:
        overload_impl = create_inplace_binary_op_overload(op)
        overload(op, no_unliteral=True)(overload_impl)


_install_inplace_binary_ops()


########################## unary operators ###############################


def create_unary_op_overload(op):
    def overload_dataframe_unary_op(df):
        if isinstance(df, DataFrameType):
            op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
            data_args = ", ".join(
                "{1} bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {0})".format(
                    i, op_str
                )
                for i in range(len(df.columns))
            )
            header = "def impl(df):\n"
            return _gen_init_df(header, df.columns, data_args)

    return overload_dataframe_unary_op


def _install_unary_ops():
    # install unary operators: ~, -, +
    for op in bodo.hiframes.pd_series_ext.series_unary_ops:
        overload_impl = create_unary_op_overload(op)
        overload(op, no_unliteral=True)(overload_impl)


_install_unary_ops()


# TODO: move to other file
########### top level functions ###############


# inline IR for parallelizable data structures, but don't inline for scalars since we
# pattern match pd.isna(A[i]) in SeriesPass to handle it properly
@overload(pd.isna, inline="always", no_unliteral=True)
@overload(pd.isnull, inline="always", no_unliteral=True)
def overload_isna(obj):
    # DataFrame, Series, Index
    if isinstance(
        obj, (DataFrameType, SeriesType)
    ) or bodo.hiframes.pd_index_ext.is_pd_index_type(obj):
        return lambda obj: obj.isna()

    # Bodo arrays, use array_kernels.isna()
    if is_array_typ(obj):

        def impl(obj):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            n = len(obj)
            out_arr = np.empty(n, np.bool_)
            for i in numba.parfors.parfor.internal_prange(n):
                out_arr[i] = bodo.libs.array_kernels.isna(obj, i)
            return out_arr

        return impl


@overload(pd.isna, no_unliteral=True)
@overload(pd.isnull, no_unliteral=True)
def overload_isna_scalar(obj):
    # ignore cases handled above
    if (
        isinstance(obj, (DataFrameType, SeriesType))
        or bodo.hiframes.pd_index_ext.is_pd_index_type(obj)
        or is_array_typ(obj)
    ):
        return

    # array-like: list, tuple
    if isinstance(obj, (types.List, types.UniTuple)):
        # no reuse of array implementation to avoid prange (unexpected threading etc.)
        def impl(obj):
            n = len(obj)
            out_arr = np.empty(n, np.bool_)
            for i in range(n):
                out_arr[i] = pd.isna(obj[i])
            return out_arr

        return impl

    # scalars
    # using unliteral_val() to avoid literal type in output type since we may replace
    # this call in Series pass with array_kernels.isna()
    obj = types.unliteral(obj)
    if obj == bodo.string_type:
        return lambda obj: unliteral_val(False)  # pragma: no cover
    if isinstance(obj, types.Integer):
        return lambda obj: unliteral_val(False)  # pragma: no cover
    if isinstance(obj, types.Float):
        return lambda obj: np.isnan(obj)  # pragma: no cover
    if isinstance(obj, (types.NPDatetime, types.NPTimedelta)):
        return lambda obj: np.isnat(obj)  # pragma: no cover
    if obj == types.none:
        return lambda obj: unliteral_val(True)
    if obj == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type:
        return lambda obj: np.isnat(
            bodo.hiframes.pd_timestamp_ext.integer_to_dt64(obj.value)
        )  # pragma: no cover
    if obj == bodo.hiframes.datetime_timedelta_ext.pd_timedelta_type:
        return lambda obj: np.isnat(
            bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(obj.value)
        )  # pragma: no cover

    # TODO: catch other cases
    return lambda obj: unliteral_val(False)  # pragma: no cover


# support A[i] = None array setitem using our array NA setting function
# TODO: inline when supported in Numba
@overload(operator.setitem, no_unliteral=True)
def overload_setitem_arr_none(A, idx, val):
    if is_array_typ(A, False) and isinstance(idx, types.Integer) and val == types.none:
        return lambda A, idx, val: bodo.libs.array_kernels.setna(
            A, idx
        )  # pragma: no cover


@overload(pd.notna, inline="always", no_unliteral=True)
@overload(pd.notnull, inline="always", no_unliteral=True)
def overload_notna(obj):
    # non-scalars
    # TODO: ~pd.isna(obj) implementation fails for some reason in
    # test_dataframe.py::test_pd_notna[na_test_obj7] with 1D_Var input
    if isinstance(obj, DataFrameType):
        return lambda obj: obj.notna()  # pragma: no cover
    if (
        isinstance(obj, (SeriesType, types.Array, types.List, types.UniTuple))
        or bodo.hiframes.pd_index_ext.is_pd_index_type(obj)
        or obj == bodo.string_array_type
    ):
        return lambda obj: ~pd.isna(obj)  # pragma: no cover

    # scalars
    return lambda obj: not pd.isna(obj)  # pragma: no cover


def _get_pd_dtype_str(t):
    """return dtype string for 'dtype' values in read_excel()
    it's not fully consistent for read_csv(), since datetime64 requires 'datetime64[ns]'
    instead of 'str'
    """
    if t.dtype == types.NPDatetime("ns"):
        return "'datetime64[ns]'"

    return bodo.ir.csv_ext._get_pd_dtype_str(t)


@overload_method(DataFrameType, "replace", inline="always", no_unliteral=True)
def replace_overload(
    df,
    to_replace=None,
    value=None,
    inplace=False,
    limit=None,
    regex=False,
    method="pad",
):

    # Check that to_replace is never none
    if is_overload_none(to_replace):
        raise BodoError("replace(): to_replace value of None is not supported")

    # TODO: Add error checking to ensure this is only called on supported types.e

    # Handle type error checking for defaults only supported
    # Right now this will be everything except to_replace
    # and value
    args_dict = {
        "inplace": inplace,
        "limit": limit,
        "regex": regex,
        "method": method,
    }
    args_default_dict = {
        "inplace": False,
        "limit": None,
        "regex": False,
        "method": "pad",
    }

    check_unsupported_args("replace", args_dict, args_default_dict)

    data_args = ", ".join(
        f"df.iloc[:, {i}].replace(to_replace, value).values"
        for i in range(len(df.columns))
    )
    header = "def impl(df, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload(pd.read_excel, no_unliteral=True)
def overload_read_excel(
    io,
    sheet_name=0,
    header=0,
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skiprows=None,
    nrows=None,
    na_values=None,
    keep_default_na=True,
    na_filter=True,
    verbose=False,
    parse_dates=False,
    date_parser=None,
    thousands=None,
    comment=None,
    skipfooter=0,
    convert_float=True,
    mangle_dupe_cols=True,
    _bodo_df_type=None,
):
    # implement pd.read_excel() by just calling Pandas
    # utyped pass adds _bodo_df_type argument which is a TypeRef of output type
    df_type = _bodo_df_type.instance_type

    # add output type to numba 'types' module with a unique name, needed for objmode
    t_name = "read_excel_df{}".format(next_label())
    setattr(types, t_name, df_type)

    # objmode doesn't allow lists, embed 'parse_dates' as a constant inside objmode
    parse_dates_const = False
    if is_overload_constant_list(parse_dates):
        parse_dates_const = get_overload_const_list(parse_dates)

    # embed dtype since objmode doesn't allow list/dict
    pd_dtype_strs = ", ".join(
        [
            "'{}':{}".format(cname, _get_pd_dtype_str(t))
            for cname, t in zip(df_type.columns, df_type.data)
        ]
    )

    func_text = """
def impl(
    io,
    sheet_name=0,
    header=0,
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skiprows=None,
    nrows=None,
    na_values=None,
    keep_default_na=True,
    na_filter=True,
    verbose=False,
    parse_dates=False,
    date_parser=None,
    thousands=None,
    comment=None,
    skipfooter=0,
    convert_float=True,
    mangle_dupe_cols=True,
    _bodo_df_type=None,
):
    with numba.objmode(df="{}"):
        df = pd.read_excel(
            io,
            sheet_name,
            header,
            {},
            index_col,
            usecols,
            squeeze,
            {{{}}},
            engine,
            converters,
            true_values,
            false_values,
            skiprows,
            nrows,
            na_values,
            keep_default_na,
            na_filter,
            verbose,
            {},
            date_parser,
            thousands,
            comment,
            skipfooter,
            convert_float,
            mangle_dupe_cols,
        )
    return df
    """.format(
        t_name, list(df_type.columns), pd_dtype_strs, parse_dates_const
    )
    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    impl = loc_vars["impl"]
    return impl


def set_df_col(df, cname, arr, inplace):  # pragma: no cover
    df[cname] = arr


@infer_global(set_df_col)
class SetDfColInfer(AbstractTemplate):
    def generic(self, args, kws):
        from bodo.hiframes.pd_dataframe_ext import DataFrameType

        assert not kws
        assert len(args) == 4
        assert isinstance(args[1], types.Literal)
        target = args[0]
        ind = args[1].literal_value
        val = args[2]
        ret = target

        if isinstance(target, DataFrameType):
            index = target.index
            # empty df index is updated based on new column
            if len(target.columns) == 0:
                index = bodo.hiframes.pd_index_ext.RangeIndexType(types.none)
            if isinstance(val, SeriesType):
                if len(target.columns) == 0:
                    index = val.index
                val = val.data

            if isinstance(val, types.List):
                val = scalar_to_array_type(val.dtype)
            if not is_array_typ(val):
                val = scalar_to_array_type(val)
            if ind in target.columns:
                # set existing column, with possibly a new array type
                new_cols = target.columns
                col_id = target.columns.index(ind)
                new_typs = list(target.data)
                new_typs[col_id] = val
                new_typs = tuple(new_typs)
            else:
                # set a new column
                new_cols = target.columns + (ind,)
                new_typs = target.data + (val,)
            ret = DataFrameType(new_typs, index, new_cols)

        return ret(*args)


class DataFrameTupleIterator(types.SimpleIteratorType):
    """
    Type class for itertuples of dataframes.
    """

    def __init__(self, col_names, arr_typs):
        self.array_types = arr_typs
        self.col_names = col_names
        name_args = [
            "{}={}".format(col_names[i], arr_typs[i]) for i in range(len(col_names))
        ]
        name = "itertuples({})".format(",".join(name_args))
        py_ntup = namedtuple("Pandas", col_names)
        yield_type = types.NamedTuple([_get_series_dtype(a) for a in arr_typs], py_ntup)
        super(DataFrameTupleIterator, self).__init__(name, yield_type)


def _get_series_dtype(arr_typ):
    # values of datetimeindex are extracted as Timestamp
    if arr_typ == types.Array(types.NPDatetime("ns"), 1, "C"):
        return pandas_timestamp_type
    return arr_typ.dtype


def get_itertuples():  # pragma: no cover
    pass


@infer_global(get_itertuples)
class TypeIterTuples(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) % 2 == 0, "name and column pairs expected"
        col_names = [a.literal_value for a in args[: len(args) // 2]]
        arr_types = [if_series_to_array_type(a) for a in args[len(args) // 2 :]]
        # XXX index handling, assuming implicit index
        assert "Index" not in col_names[0]
        col_names = ["Index"] + col_names
        arr_types = [types.Array(types.int64, 1, "C")] + arr_types
        iter_typ = DataFrameTupleIterator(col_names, arr_types)
        return iter_typ(*args)


@register_model(DataFrameTupleIterator)
class DataFrameTupleIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        # XXX array_types[0] is implicit index
        members = [("index", types.EphemeralPointer(types.uintp))] + [
            ("array{}".format(i), arr) for i, arr in enumerate(fe_type.array_types[1:])
        ]
        super(DataFrameTupleIteratorModel, self).__init__(dmm, fe_type, members)

    def from_return(self, builder, value):
        # dummy to avoid lowering error for itertuples_overload
        # TODO: remove when overload_method can avoid lowering or avoid cpython
        # wrapper
        return value


@lower_builtin(get_itertuples, types.VarArg(types.Any))
def get_itertuples_impl(context, builder, sig, args):
    arrays = args[len(args) // 2 :]
    array_types = sig.args[len(sig.args) // 2 :]

    iterobj = context.make_helper(builder, sig.return_type)

    zero = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once_value(builder, zero)

    iterobj.index = indexptr

    for i, arr in enumerate(arrays):
        setattr(iterobj, "array{}".format(i), arr)

    # Incref arrays
    for arr, arr_typ in zip(arrays, array_types):
        context.nrt.incref(builder, arr_typ, arr)

    res = iterobj._getvalue()

    # Note: a decref on the iterator will dereference all internal MemInfo*
    return impl_ret_new_ref(context, builder, sig.return_type, res)


@lower_builtin("getiter", DataFrameTupleIterator)
def getiter_itertuples(context, builder, sig, args):
    # simply return the iterator
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# similar to iternext of ArrayIterator
@lower_builtin("iternext", DataFrameTupleIterator)
@iternext_impl(RefType.UNTRACKED)
def iternext_itertuples(context, builder, sig, args, result):
    # TODO: refcount issues?
    (iterty,) = sig.args
    (it,) = args

    # TODO: support string arrays
    iterobj = context.make_helper(builder, iterty, value=it)
    # first array type is implicit int index
    # use len() to support string arrays
    len_sig = signature(types.intp, iterty.array_types[1])
    nitems = context.compile_internal(
        builder, lambda a: len(a), len_sig, [iterobj.array0]
    )
    # ary = make_array(iterty.array_types[1])(context, builder, value=iterobj.array0)
    # nitems, = cgutils.unpack_tuple(builder, ary.shape, count=1)

    index = builder.load(iterobj.index)
    is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        values = [index]  # XXX implicit int index
        for i, arr_typ in enumerate(iterty.array_types[1:]):
            arr_ptr = getattr(iterobj, "array{}".format(i))

            if arr_typ == types.Array(types.NPDatetime("ns"), 1, "C"):
                getitem_sig = signature(pandas_timestamp_type, arr_typ, types.intp)
                val = context.compile_internal(
                    builder,
                    lambda a, i: bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
                        np.int64(a[i])
                    ),
                    getitem_sig,
                    [arr_ptr, index],
                )
            else:
                getitem_sig = signature(arr_typ.dtype, arr_typ, types.intp)
                val = context.compile_internal(
                    builder, lambda a, i: a[i], getitem_sig, [arr_ptr, index]
                )
            # arr = make_array(arr_typ)(context, builder, value=arr_ptr)
            # val = _getitem_array1d(context, builder, arr_typ, arr, index,
            #                      wraparound=False)
            values.append(val)

        value = context.make_tuple(builder, iterty.yield_type, values)
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)


# TODO: move this to array analysis
# the namedtuples created by get_itertuples-iternext-pair_first don't have
# shapes created in array analysis
# def _analyze_op_static_getitem(self, scope, equiv_set, expr):
#     var = expr.value
#     typ = self.typemap[var.name]
#     if not isinstance(typ, types.BaseTuple):
#         return self._index_to_shape(scope, equiv_set, expr.value, expr.index_var)
#     try:
#         shape = equiv_set._get_shape(var)
#         require(isinstance(expr.index, int) and expr.index < len(shape))
#         return shape[expr.index], []
#     except:
#         pass

#     return None

# numba.parfors.array_analysis.ArrayAnalysis._analyze_op_static_getitem = _analyze_op_static_getitem

# FIXME: fix array analysis for tuples in general
def _analyze_op_pair_first(self, scope, equiv_set, expr, lhs):
    # TODO(ehsan): Numba 0.53 adds lhs so this code should be refactored
    # make dummy lhs since we don't have access to lhs
    typ = self.typemap[expr.value.name].first_type
    if not isinstance(typ, types.NamedTuple):
        return None
    lhs = ir.Var(scope, mk_unique_var("tuple_var"), expr.loc)
    self.typemap[lhs.name] = typ
    rhs = ir.Expr.pair_first(expr.value, expr.loc)
    lhs_assign = ir.Assign(rhs, lhs, expr.loc)
    # (shape, post) = self._gen_shape_call(equiv_set, lhs, typ.count, )
    var = lhs
    out = []
    size_vars = []
    ndims = typ.count
    for i in range(ndims):
        # get size: Asize0 = A_sh_attr[0]
        size_var = ir.Var(
            var.scope, mk_unique_var("{}_size{}".format(var.name, i)), var.loc
        )
        getitem = ir.Expr.static_getitem(lhs, i, None, var.loc)
        self.calltypes[getitem] = None
        out.append(ir.Assign(getitem, size_var, var.loc))
        self._define(equiv_set, size_var, types.intp, getitem)
        size_vars.append(size_var)
    shape = tuple(size_vars)
    return numba.parfors.array_analysis.ArrayAnalysis.AnalyzeResult(
        shape=shape, pre=[lhs_assign] + out
    )


numba.parfors.array_analysis.ArrayAnalysis._analyze_op_pair_first = (
    _analyze_op_pair_first
)
