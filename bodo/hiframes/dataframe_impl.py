# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implementation of DataFrame attributes and methods using overload.
"""
from collections import namedtuple
import numpy as np
import pandas as pd
import numba
from numba.core import types, cgutils
from numba.extending import overload, overload_attribute, overload_method
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.core.typing import signature
from numba.core.imputils import (
    impl_ret_new_ref,
    impl_ret_borrowed,
    iternext_impl,
    RefType,
)
from numba.core import ir
from numba.core.ir_utils import mk_unique_var, next_label
import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType, handle_inplace_df_type_change
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.utils.typing import (
    is_overload_none,
    is_overload_true,
    is_overload_false,
    is_overload_zero,
    get_overload_const_str,
    is_overload_constant_str,
    is_overload_constant_bool,
    BodoError,
    ConstDictType,
    scalar_to_array_type,
    raise_bodo_error,
    get_registry_consts,
)
from bodo.utils.transform import gen_const_tup
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
from numba.core.imputils import lower_builtin
from bodo.hiframes.pd_series_ext import if_series_to_array_type
from numba.extending import register_model, models
import llvmlite.llvmpy.core as lc
from bodo.libs.array import (
    array_to_info,
    arr_info_list_to_table,
    info_from_table,
    info_to_array,
    delete_table,
)


@overload_attribute(DataFrameType, "index", inline="always")
def overload_dataframe_index(df):
    return lambda df: bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)


@overload_attribute(DataFrameType, "columns", inline="always")
def overload_dataframe_columns(df):
    str_arr = "bodo.utils.conversion.coerce_to_array({})".format(df.columns)
    func_text = "def impl(df):\n"
    func_text += "  return bodo.hiframes.pd_index_ext.init_string_index({})\n".format(
        str_arr
    )
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    # print(func_text)
    impl = loc_vars["impl"]
    return impl


@overload_attribute(DataFrameType, "values")
def overload_dataframe_values(df):
    n_cols = len(df.columns)
    data_args = ", ".join(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})".format(i)
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
    if not is_overload_none(dtype):
        raise BodoError("'dtype' argument of to_numpy() not supported yet")

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


@overload_method(DataFrameType, "astype", inline="always", no_unliteral=True)
def overload_dataframe_astype(df, dtype, copy=True, errors="raise"):
    # just call astype() on all column Series
    # TODO: support categorical, dt64, etc.

    data_args = ", ".join("df['{}'].astype(dtype).values".format(c) for c in df.columns)
    header = "def impl(df, dtype, copy=True, errors='raise'):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "copy", inline="always", no_unliteral=True)
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
    if not (
        is_overload_none(mapper)
        and is_overload_none(index)
        and is_overload_none(axis)
        and is_overload_constant_bool(inplace)
        and is_overload_none(level)
        and is_overload_constant_str(errors)
        and get_overload_const_str(errors) == "ignore"
    ):
        raise BodoError("Only 'columns' and copy arguments of df.rename() supported")

    # columns should be constant dictionary
    if not isinstance(columns, ConstDictType):
        raise BodoError(
            "'columns' argument to df.rename() should be a constant dictionary"
        )

    columns_consts = get_registry_consts(columns.const_no)

    col_map = {
        columns_consts[2 * i]: columns_consts[2 * i + 1]
        for i in range(len(columns_consts) // 2)
    }
    new_cols = [
        col_map.get(df.columns[i], df.columns[i]) for i in range(len(df.columns))
    ]

    data_outs = []
    for i in range(len(df.columns)):
        arr = "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})".format(i)
        if is_overload_true(copy):
            data_outs.append(arr + ".copy()")
        elif is_overload_false(copy):
            data_outs.append(arr)
        else:
            data_outs.append("{arr}.copy() if copy else {arr}".format(arr=arr))

    header = (
        "def impl(df, mapper=None, index=None, columns=None, axis=None, "
        "copy=True, inplace=False, level=None, errors='ignore', _bodo_transformed=False):\n"
    )
    return _gen_init_df(header, new_cols, ", ".join(data_outs))


@overload_method(DataFrameType, "isna", inline="always", no_unliteral=True)
@overload_method(DataFrameType, "isnull", inline="always", no_unliteral=True)
def overload_dataframe_isna(df):
    # call isna() on column Series
    data_args = ", ".join("df['{}'].isna().values".format(c) for c in df.columns)
    header = "def impl(df):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "notna", inline="always", no_unliteral=True)
def overload_dataframe_notna(df):
    # call notna() on column Series
    data_args = ", ".join("df['{}'].notna().values".format(c) for c in df.columns)
    header = "def impl(df):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "head", inline="always", no_unliteral=True)
def overload_dataframe_head(df, n=5):
    # call head() on column Series
    data_args = ", ".join("df['{}'].head(n).values".format(c) for c in df.columns)
    header = "def impl(df, n=5):\n"
    index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[:n]"
    return _gen_init_df(header, df.columns, data_args, index)


@overload_method(DataFrameType, "tail", inline="always", no_unliteral=True)
def overload_dataframe_tail(df, n=5):
    # call tail() on column Series
    data_args = ", ".join("df['{}'].tail(n).values".format(c) for c in df.columns)
    header = "def impl(df, n=5):\n"
    index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[-n:]"
    return _gen_init_df(header, df.columns, data_args, index)


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
        func_text += "  {} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})\n".format(
            v_name, i
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

    # XXX: Pandas pass a single array to Numpy and therefore, casts the input
    # columns to the same dtype! We simulate this behavior here.
    extra = ""
    first_col_dtype = df.data[0] if len(df.data) > 0 else None
    if not all(c == first_col_dtype for c in df.data):
        dtypes = [numba.np.numpy_support.as_dtype(d.dtype) for d in df.data]
        out_dtype = numba.np.numpy_support.from_dtype(np.find_common_type(dtypes, []))
        extra = ".astype(np.{})".format(out_dtype)

    n_cols = len(df.columns)
    data_args = ", ".join(
        "np.abs(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}){})".format(
            i, extra
        )
        for i in range(n_cols)
    )
    header = "def impl(df):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "corr", inline="always", no_unliteral=True)
def overload_dataframe_corr(df, method="pearson", min_periods=1):

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

    str_arr = "bodo.utils.conversion.coerce_to_array({})".format(numeric_cols)
    index = "bodo.hiframes.pd_index_ext.init_string_index({})\n".format(str_arr)

    header = "def impl(df, method='pearson', min_periods=1):\n"
    header += "  mat = {}\n".format(mat)
    header += "  res = bodo.libs.array_kernels.nancorr(mat, 0, min_periods)\n"
    return _gen_init_df(header, numeric_cols, data_args, index)


@overload_method(DataFrameType, "cov", inline="always", no_unliteral=True)
def overload_dataframe_cov(df, min_periods=None):
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

    str_arr = "bodo.utils.conversion.coerce_to_array({})".format(numeric_cols)
    index = "bodo.hiframes.pd_index_ext.init_string_index({})\n".format(str_arr)

    header = "def impl(df, min_periods=None):\n"
    header += "  mat = {}\n".format(mat)
    header += "  res = bodo.libs.array_kernels.nancorr(mat, 1, {})\n".format(minpv)
    return _gen_init_df(header, numeric_cols, data_args, index)


@overload_method(DataFrameType, "count", no_unliteral=True)
def overload_dataframe_count(df, axis=0, level=None, numeric_only=False):
    # TODO: numeric_only flag
    data_args = ", ".join("df['{}'].count()".format(c) for c in df.columns)

    str_arr = "bodo.utils.conversion.coerce_to_array({})".format(df.columns)
    index = "bodo.hiframes.pd_index_ext.init_string_index({})\n".format(str_arr)

    func_text = "def impl(df, axis=0, level=None, numeric_only=False):\n"
    func_text += "  data = np.array([{}])\n".format(data_args)
    func_text += "  return bodo.hiframes.pd_series_ext.init_series(data, {})\n".format(
        index
    )
    # print(func_text)
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(DataFrameType, "nunique", inline="always", no_unliteral=True)
def overload_dataframe_nunique(df, axis=0, dropna=True):
    data_args = ", ".join("df['{}'].nunique()".format(c) for c in df.columns)

    str_arr = "bodo.utils.conversion.coerce_to_array({})".format(df.columns)
    index = "bodo.hiframes.pd_index_ext.init_string_index({})\n".format(str_arr)

    func_text = "def impl(df, axis=0, dropna=True):\n"
    func_text += "  data = np.asarray(({},))\n".format(data_args)
    func_text += "  return bodo.hiframes.pd_series_ext.init_series(data, {})\n".format(
        index
    )
    # print(func_text)
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(DataFrameType, "prod", inline="always", no_unliteral=True)
@overload_method(DataFrameType, "product", inline="always", no_unliteral=True)
def overload_dataframe_prod(
    df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0
):
    return _gen_reduce_impl(df, "prod")


@overload_method(DataFrameType, "sum", inline="always", no_unliteral=True)
def overload_dataframe_sum(
    df, axis=None, skipna=None, level=None, numeric_only=None, min_count=0
):
    return _gen_reduce_impl(df, "sum")


@overload_method(DataFrameType, "max", inline="always", no_unliteral=True)
def overload_dataframe_max(df, axis=None, skipna=None, level=None, numeric_only=None):
    return _gen_reduce_impl(df, "max")


@overload_method(DataFrameType, "min", inline="always", no_unliteral=True)
def overload_dataframe_min(df, axis=None, skipna=None, level=None, numeric_only=None):
    return _gen_reduce_impl(df, "min")


@overload_method(DataFrameType, "mean", inline="always", no_unliteral=True)
def overload_dataframe_mean(df, axis=None, skipna=None, level=None, numeric_only=None):
    return _gen_reduce_impl(df, "mean")


@overload_method(DataFrameType, "var", inline="always", no_unliteral=True)
def overload_dataframe_var(
    df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None
):
    return _gen_reduce_impl(df, "var")


@overload_method(DataFrameType, "std", inline="always", no_unliteral=True)
def overload_dataframe_std(
    df, axis=None, skipna=None, level=None, ddof=1, numeric_only=None
):
    return _gen_reduce_impl(df, "std")


@overload_method(DataFrameType, "median", inline="always", no_unliteral=True)
def overload_dataframe_median(
    df, axis=None, skipna=None, level=None, numeric_only=None
):
    return _gen_reduce_impl(df, "median")


@overload_method(DataFrameType, "quantile", inline="always", no_unliteral=True)
def overload_dataframe_quantile(
    df, q=0.5, axis=0, numeric_only=True, interpolation="linear"
):
    # TODO: name is str(q)
    return _gen_reduce_impl(df, "quantile", "q")


@overload_method(DataFrameType, "idxmax", inline="always", no_unliteral=True)
def overload_dataframe_idxmax(df, axis=0, skipna=True):
    return _gen_reduce_impl(df, "idxmax")


@overload_method(DataFrameType, "idxmin", inline="always", no_unliteral=True)
def overload_dataframe_idxmin(df, axis=0, skipna=True):
    return _gen_reduce_impl(df, "idxmin")


def _gen_reduce_impl(df, func_name, args=None):
    args = "" if args is None else args

    if func_name in ("idxmax", "idxmin"):
        out_colnames = df.columns
    else:
        # TODO: numeric_only=None tries its best: core/frame.py/DataFrame/_reduce
        numeric_cols = [
            c for c, d in zip(df.columns, df.data) if _is_numeric_dtype(d.dtype)
        ]
        out_colnames = numeric_cols

    # TODO: support empty dataframe
    assert len(out_colnames) != 0

    dtypes = [
        numba.np.numpy_support.as_dtype(df.data[df.columns.index(c)].dtype)
        for c in out_colnames
    ]
    comm_dtype = numba.np.numpy_support.from_dtype(np.find_common_type(dtypes, []))

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
        "df['{}'].{}({})".format(c, func_name, args) for c in out_colnames
    )

    str_arr = "bodo.utils.conversion.coerce_to_array({})".format(out_colnames)
    index = "bodo.hiframes.pd_index_ext.init_string_index({})\n".format(str_arr)

    minc = ""
    if func_name in ("sum", "prod"):
        minc = ", min_count=0"

    ddof = ""
    if func_name in ("var", "std"):
        ddof = "ddof=1, "

    # function signature
    func_text = "def impl(df, axis=None, skipna=None, level=None,{} numeric_only=None{}):\n".format(
        ddof, minc
    )
    if func_name == "quantile":
        func_text = (
            "def impl(df, q=0.5, axis=0, numeric_only=True, interpolation='linear'):\n"
        )
    if func_name in ("idxmax", "idxmin"):
        func_text = "def impl(df, axis=0, skipna=True):\n"

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
    func_text += "  return bodo.hiframes.pd_series_ext.init_series(data, {})\n".format(
        index
    )
    # print(func_text)
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(DataFrameType, "pct_change", inline="always", no_unliteral=True)
def overload_dataframe_pct_change(
    df, periods=1, fill_method="pad", limit=None, freq=None
):
    data_args = ", ".join(
        "df['{}'].pct_change(periods).values".format(c) for c in df.columns
    )
    header = "def impl(df, periods=1, fill_method='pad', limit=None, freq=None):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "cumprod", inline="always", no_unliteral=True)
def overload_dataframe_cumprod(df, axis=None, skipna=True):
    data_args = ", ".join("df['{}'].values.cumprod()".format(c) for c in df.columns)
    header = "def impl(df, axis=None, skipna=True):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "cumsum", inline="always", no_unliteral=True)
def overload_dataframe_cumsum(df, axis=None, skipna=True):
    data_args = ", ".join("df['{}'].values.cumsum()".format(c) for c in df.columns)
    header = "def impl(df, axis=None, skipna=True):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "describe", inline="always", no_unliteral=True)
def overload_dataframe_describe(df, percentiles=None, include=None, exclude=None):
    data_args = ", ".join("df['{}'].describe().values".format(c) for c in df.columns)
    header = "def impl(df, percentiles=None, include=None, exclude=None):\n"
    index = (
        "bodo.utils.conversion.convert_to_index(['count', 'mean', 'std', "
        "'min', '25%', '50%', '75%', 'max'])"
    )
    return _gen_init_df(header, df.columns, data_args, index)


@overload_method(DataFrameType, "take", inline="always", no_unliteral=True)
def overload_dataframe_take(df, indices, axis=0, convert=None, is_copy=True):
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
    data_args = ", ".join(
        "bodo.hiframes.rolling.shift(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}), periods, False)".format(
            i
        )
        for i in range(len(df.columns))
    )
    header = "def impl(df, periods=1, freq=None, axis=0, fill_value=None):\n"
    return _gen_init_df(header, df.columns, data_args)


@overload_method(DataFrameType, "set_index", inline="always", no_unliteral=True)
def overload_dataframe_set_index(
    df, keys, drop=True, append=False, inplace=False, verify_integrity=False
):
    if not is_overload_false(inplace):
        raise BodoError("set_index() inplace argument not supported yet")

    col_name = get_overload_const_str(keys)
    col_ind = df.columns.index(col_name)

    data_args = ", ".join(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})".format(i)
        for i in range(len(df.columns))
        if i != col_ind
    )
    header = "def impl(df, keys, drop=True, append=False, inplace=False, verify_integrity=False):\n"
    columns = tuple(c for c in df.columns if c != col_name)
    index = (
        "bodo.utils.conversion.index_from_array("
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}), '{}')"
    ).format(col_ind, col_name)
    return _gen_init_df(header, columns, data_args, index)


@overload_method(DataFrameType, "query", no_unliteral=True)
def overload_dataframe_query(df, expr, inplace=False):
    """Support query only for the case where expr is a constant string and expr output
    is a 1D boolean array. 
    Refering to named index by name is not supported.
    Series.dt.* is not supported. issue #451
    """
    # check unsupported "inplace"
    if not is_overload_false(inplace):
        raise BodoError("query(): inplace argument not supported yet")

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
    if not is_overload_none(subset):
        raise ValueError("duplicated() subset argument not supported yet")

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
    # print(func_text)
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(DataFrameType, "drop_duplicates", inline="always", no_unliteral=True)
def overload_dataframe_drop_duplicates(df, subset=None, keep="first", inplace=False):
    # TODO: support inplace
    if not is_overload_none(subset):
        raise BodoError("drop_duplicates() subset argument not supported yet")

    if is_overload_true(inplace):
        raise BodoError("drop_duplicates() inplace argument not supported yet")

    # XXX: can't reuse duplicated() here since it shuffles data and chunks
    # may not match

    n_cols = len(df.columns)

    data_args = ", ".join("data_{}".format(i) for i in range(n_cols))

    func_text = "def impl(df, subset=None, keep='first', inplace=False):\n"
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
    data_args = "({},)".format(data_args)

    func_text = "{}  return bodo.hiframes.pd_dataframe_ext.init_dataframe({}, {}, {})\n".format(
        header, data_args, index, col_var
    )
    loc_vars = {}
    _global = {"bodo": bodo, "np": np, "numba": numba}
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

            # scalar case, TODO: check
            data_args = ", ".join(
                (
                    "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(left, {0}) {1}"
                    "right"
                ).format(i, op_str)
                for i in range(len(left.columns))
            )
            header = "def impl(left, right):\n"
            index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(left)"
            return _gen_init_df(header, left.columns, data_args, index)

        if isinstance(right, DataFrameType):
            data_args = ", ".join(
                "left {1} bodo.hiframes.pd_dataframe_ext.get_dataframe_data(right, {0})".format(
                    i, op_str
                )
                for i in range(len(right.columns))
            )
            header = "def impl(left, right):\n"
            index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(right)"
            return _gen_init_df(header, right.columns, data_args, index)

    return overload_dataframe_binary_op


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in bodo.hiframes.pd_series_ext.series_binary_ops:
        overload_impl = create_binary_op_overload(op)
        overload(op)(overload_impl)


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
                # print(func_text)
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
        overload(op)(overload_impl)


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
        overload(op)(overload_impl)


_install_unary_ops()


# TODO: move to other file
########### top level functions ###############


@overload(pd.isna, inline="always", no_unliteral=True)
@overload(pd.isnull, inline="always", no_unliteral=True)
def overload_isna(obj):
    # DataFrame, Series, Index
    if isinstance(
        obj, (DataFrameType, SeriesType)
    ) or bodo.hiframes.pd_index_ext.is_pd_index_type(obj):
        return lambda obj: obj.isna()

    # arrays
    if isinstance(obj, types.Array):
        return lambda obj: np.isnan(obj)

    # array of strings
    # TODO: other string array data structures
    if obj == bodo.string_array_type:

        def impl(obj):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            n = len(obj)
            out_arr = np.empty(n, np.bool_)
            for i in numba.parfors.parfor.internal_prange(n):
                out_arr[i] = bodo.libs.array_kernels.isna(obj, i)
            return out_arr

        return impl

    # array-like: list, tuple
    if isinstance(obj, (types.List, types.UniTuple)):
        return lambda obj: pd.isna(bodo.utils.conversion.coerce_to_array(obj))

    # scalars
    obj = types.unliteral(obj)
    if obj == bodo.string_type:
        return lambda obj: False
    if isinstance(obj, types.Integer):
        return lambda obj: False
    if isinstance(obj, types.Float):
        return lambda obj: np.isnan(obj)
    # TODO: NaT

    # TODO: catch other cases
    return lambda obj: False


@overload(pd.notna, inline="always", no_unliteral=True)
@overload(pd.notnull, inline="always", no_unliteral=True)
def overload_notna(obj):
    # non-scalars
    # TODO: ~pd.isna(obj) implementation fails for some reason in
    # test_dataframe.py::test_pd_notna[na_test_obj7] with 1D_Var input
    if isinstance(obj, DataFrameType):
        return lambda obj: obj.notna()
    if (
        isinstance(obj, (SeriesType, types.Array, types.List, types.UniTuple))
        or bodo.hiframes.pd_index_ext.is_pd_index_type(obj)
        or obj == bodo.string_array_type
    ):
        return lambda obj: ~pd.isna(obj)

    # scalars
    return lambda obj: not pd.isna(obj)


def _get_pd_dtype_str(t):
    """return dtype string for 'dtype' values in read_excel()
    it's not fully consistent for read_csv(), since datetime64 requires 'datetime64[ns]'
    instead of 'str'
    """
    if t.dtype == types.NPDatetime("ns"):
        return "'datetime64[ns]'"

    return bodo.ir.csv_ext._get_pd_dtype_str(t)


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
    if isinstance(parse_dates, bodo.utils.typing.ConstList):
        parse_dates_const = list(get_registry_consts(parse_dates.const_no))

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
    exec(func_text, {"numba": numba, "pd": pd, "bodo": bodo, "np": np}, loc_vars)
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
            if not bodo.utils.utils.is_array_typ(val):
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
            ret = DataFrameType(new_typs, index, new_cols, target.has_parent)

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
    if context.enable_nrt:
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
def _analyze_op_pair_first(self, scope, equiv_set, expr):
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
    return shape, [lhs_assign] + out


numba.parfors.array_analysis.ArrayAnalysis._analyze_op_pair_first = (
    _analyze_op_pair_first
)
