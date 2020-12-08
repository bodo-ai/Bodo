# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""typing for rolling window functions
"""
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.core.typing.templates import AbstractTemplate, signature
from numba.extending import (
    infer,
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload_method,
    register_model,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.rolling import supported_rolling_funcs
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    get_literal_value,
    raise_const_error,
)


class RollingType(types.Type):
    """Rolling objects from df.rolling() or Series.rolling() calls"""

    def __init__(self, obj_type, window_type, on, selection, explicit_select=False):
        # obj_type can be either Series or DataFrame
        self.obj_type = obj_type
        self.window_type = window_type
        self.on = on
        self.selection = selection
        self.explicit_select = explicit_select

        super(RollingType, self).__init__(
            name=f"RollingType({obj_type}, {window_type}, {on}, {selection}, {explicit_select})"
        )

    def copy(self):
        return RollingType(
            self.obj_type,
            self.window_type,
            self.on,
            self.selection,
            self.explicit_select,
        )


@register_model(RollingType)
class RollingModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("obj", fe_type.obj_type),
            ("window", fe_type.window_type),
            ("center", types.bool_),
        ]
        super(RollingModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(RollingType, "obj", "obj")
make_attribute_wrapper(RollingType, "window", "window")
make_attribute_wrapper(RollingType, "center", "center")


@overload_method(DataFrameType, "rolling", inline="always", no_unliteral=True)
def df_rolling_overload(
    df,
    window,
    min_periods=None,
    center=False,
    win_type=None,
    on=None,
    axis=0,
    closed=None,
):
    unsupported_args = dict(
        min_periods=min_periods, win_type=win_type, axis=axis, closed=closed
    )
    arg_defaults = dict(min_periods=None, win_type=None, axis=0, closed=None)
    check_unsupported_args("DataFrame.rolling", unsupported_args, arg_defaults)

    def _impl(
        df,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
    ):  # pragma: no cover
        return bodo.hiframes.pd_rolling_ext.init_rolling(df, window, center, on)

    return _impl


@overload_method(SeriesType, "rolling", inline="always", no_unliteral=True)
def overload_series_rolling(
    df,
    window,
    min_periods=None,
    center=False,
    win_type=None,
    on=None,
    axis=0,
    closed=None,
):
    unsupported_args = dict(
        min_periods=min_periods, win_type=win_type, axis=axis, closed=closed
    )
    arg_defaults = dict(min_periods=None, win_type=None, axis=0, closed=None)
    check_unsupported_args("Series.rolling", unsupported_args, arg_defaults)

    def impl(
        df,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
    ):  # pragma: no cover
        return bodo.hiframes.pd_rolling_ext.init_rolling(df, window, center, on)

    return impl


@intrinsic
def init_rolling(typingctx, obj_type, window_type, center_type, on_type=None):
    """initialize rolling object"""

    def codegen(context, builder, signature, args):
        (obj_val, window_val, center_val, _) = args
        rolling_type = signature.return_type

        rolling_val = cgutils.create_struct_proxy(rolling_type)(context, builder)
        rolling_val.obj = obj_val
        rolling_val.window = window_val
        rolling_val.center = center_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)
        context.nrt.incref(builder, signature.args[1], window_val)
        context.nrt.incref(builder, signature.args[2], center_val)

        return rolling_val._getvalue()

    on = get_literal_value(on_type)
    selection = None if isinstance(obj_type, SeriesType) else obj_type.columns
    rolling_type = RollingType(obj_type, window_type, on, selection, False)
    return rolling_type(obj_type, window_type, center_type, on_type), codegen


def _gen_df_rolling_out_data(rolling):
    """gen code for output data columns of Rolling calls"""
    is_variable_win = not isinstance(rolling.window_type, types.Integer)
    ftype = "variable" if is_variable_win else "fixed"
    on_arr = "None"
    if is_variable_win:
        on_arr = (
            "bodo.utils.conversion.index_to_array(index)"
            if rolling.on is None
            else f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {rolling.obj_type.columns.index(rolling.on)})"
        )
    data_args = []
    on_arr_arg = "on_arr, " if is_variable_win else ""

    if isinstance(rolling.obj_type, SeriesType):
        return (
            f"bodo.hiframes.rolling.rolling_{ftype}(bodo.hiframes.pd_series_ext.get_series_data(df), {on_arr_arg}index_arr, window, center, func, raw)",
            on_arr,
        )

    for c in rolling.selection:
        c_ind = rolling.obj_type.columns.index(c)
        if c == rolling.on:
            # avoid adding 'on' column if output will be Series (just ignored in Pandas)
            if len(rolling.selection) == 2 and rolling.explicit_select:
                continue
            out = f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {c_ind})"
        else:
            out = f"bodo.hiframes.rolling.rolling_{ftype}(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {c_ind}), {on_arr_arg}index_arr, window, center, func, raw)"
        data_args.append(out)

    return ", ".join(data_args), on_arr


@overload_method(RollingType, "apply", inline="always", no_unliteral=True)
def overload_rolling_apply(
    rolling, func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None
):
    return _gen_rolling_impl(rolling, "apply")


def _gen_rolling_impl(rolling, fname, other=None):
    """generates an implementation function for rolling overloads"""
    is_series = isinstance(rolling.obj_type, SeriesType)
    if fname in ("corr", "cov"):
        out_cols = None if is_series else _get_corr_cov_out_cols(rolling, other, fname)
        df_cols = None if is_series else rolling.obj_type.columns
        other_cols = None if is_series else other.columns
        data_args, on_arr = _gen_corr_cov_out_data(
            out_cols, df_cols, other_cols, rolling.window_type, fname
        )
    else:
        out_cols = rolling.selection
        data_args, on_arr = _gen_df_rolling_out_data(rolling)

    # NOTE: 'on' column is discarded and output is a Series if there is only one data
    # column with explicit column selection
    is_out_series = is_series or (
        len(out_cols) == (1 if rolling.on is None else 2) and rolling.explicit_select
    )

    if fname == "apply":
        header = "def impl(rolling, func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None):\n"
    elif fname == "corr":
        header = "def impl(rolling, other=None, pairwise=None):\n"
    elif fname == "cov":
        header = "def impl(rolling, other=None, pairwise=None, ddof=1):\n"
    else:
        header = "def impl(rolling):\n"
    header += "  df = rolling.obj\n"
    header += "  index = {}\n".format(
        "bodo.hiframes.pd_series_ext.get_series_index(df)"
        if is_series
        else "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)"
    )
    name = "None"
    if is_series:
        name = "bodo.hiframes.pd_series_ext.get_series_name(df)"
    elif is_out_series:
        # name of the only output column (excluding 'on' column)
        c = (set(out_cols) - set([rolling.on])).pop()
        name = f"'{c}'" if isinstance(c, str) else str(c)
    header += f"  name = {name}\n"
    header += "  window = rolling.window\n"
    header += "  center = rolling.center\n"
    header += f"  on_arr = {on_arr}\n"
    if fname == "apply":
        header += f"  index_arr = bodo.utils.conversion.index_to_array(index)\n"
    else:
        header += f"  func = '{fname}'\n"
        # no need to pass index array
        header += f"  index_arr = None\n"
        header += f"  raw = False\n"

    if is_out_series:
        header += f"  return bodo.hiframes.pd_series_ext.init_series({data_args}, index, name)"
        loc_vars = {}
        _global = {"bodo": bodo}
        exec(header, _global, loc_vars)
        impl = loc_vars["impl"]
        return impl
    return bodo.hiframes.dataframe_impl._gen_init_df(header, out_cols, data_args)


def create_rolling_overload(fname):
    """creates overloads for simple rolling functions (e.g. sum)"""

    def overload_rolling_func(rolling):
        return _gen_rolling_impl(rolling, fname)

    return overload_rolling_func


def _install_rolling_methods():
    """install overloads for simple rolling functions (e.g. sum)"""
    for fname in supported_rolling_funcs:
        if fname in ("apply", "corr", "cov"):
            continue  # handled separately
        overload_impl = create_rolling_overload(fname)
        overload_method(RollingType, fname, inline="always", no_unliteral=True)(
            overload_impl
        )


_install_rolling_methods()


def _get_corr_cov_out_cols(rolling, other, func_name):
    """get output column names for Rolling.corr/cov calls"""
    # TODO(ehsan): support other=None
    # XXX pandas only accepts variable window cov/corr
    # when both inputs have time index
    columns = rolling.selection
    if rolling.on is not None:
        raise BodoError(f"variable window rolling {func_name} not supported yet.")
    # df on df cov/corr returns output on common columns only (without
    # pairwise flag), rest are NaNs
    # TODO: support pairwise arg
    out_cols = tuple(sorted(set(columns) | set(other.columns), key=lambda k: str(k)))
    return out_cols


def _gen_corr_cov_out_data(out_cols, df_cols, other_cols, window_type, func_name):
    """gen code for output data columns of Rolling.corr/cov calls"""
    is_variable_win = not isinstance(window_type, types.Integer)
    on_arr = "None"
    if is_variable_win:
        on_arr = "bodo.utils.conversion.index_to_array(index)"
    on_arr_arg = "on_arr, " if is_variable_win else ""
    data_args = []

    # Series case
    if out_cols is None:
        return (
            f"bodo.hiframes.rolling.rolling_{func_name}(bodo.hiframes.pd_series_ext.get_series_data(df), bodo.hiframes.pd_series_ext.get_series_data(other), {on_arr_arg}window, center)",
            on_arr,
        )

    for c in out_cols:
        # non-common columns are just NaN values
        if c in df_cols and c in other_cols:
            i = df_cols.index(c)
            j = other_cols.index(c)
            out = f"bodo.hiframes.rolling.rolling_{func_name}(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i}), bodo.hiframes.pd_dataframe_ext.get_dataframe_data(other, {j}), {on_arr_arg}window, center)"
        else:
            out = "np.full(len(df), np.nan)"
        data_args.append(out)

    return ", ".join(data_args), on_arr


@overload_method(RollingType, "corr", inline="always", no_unliteral=True)
def overload_rolling_corr(rolling, other=None, pairwise=None):
    return _gen_rolling_impl(rolling, "corr", other)


@overload_method(RollingType, "cov", inline="always", no_unliteral=True)
def overload_rolling_cov(rolling, other=None, pairwise=None, ddof=1):
    return _gen_rolling_impl(rolling, "cov", other)


@infer
class GetItemDataFrameRolling2(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        rolling, idx = args
        # df.rolling('A')['B', 'C']
        if isinstance(rolling, RollingType):
            if isinstance(idx, (tuple, list)):
                if (
                    len(set(idx).difference(set(rolling.obj_type.columns))) > 0
                ):  # pragma: no cover
                    raise_const_error(
                        "rolling: selected column {} not found in dataframe".format(
                            set(idx).difference(set(rolling.obj_type.columns))
                        )
                    )
                selection = list(idx)
            else:
                if idx not in rolling.obj_type.columns:  # pragma: no cover
                    raise_const_error(
                        "rolling: selected column {} not found in dataframe".format(idx)
                    )
                selection = [idx]
            if rolling.on is not None:
                selection.append(rolling.on)
            ret_rolling = RollingType(
                rolling.obj_type,
                rolling.window_type,
                rolling.on,
                tuple(selection),
                True,
            )
            return signature(ret_rolling, *args)


@lower_builtin("static_getitem", RollingType, types.Any)
def static_getitem_df_groupby(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])
