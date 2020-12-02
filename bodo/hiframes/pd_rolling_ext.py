# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""typing for rolling window functions
"""
from numba.core import cgutils, types
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    bound_function,
    signature,
)
from numba.extending import (
    infer,
    infer_getattr,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload_method,
    register_model,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.rolling import supported_rolling_funcs
from bodo.utils.typing import BodoError, get_literal_value


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
    selection = None if isinstance(obj_type, SeriesType) else list(obj_type.columns)
    rolling_type = RollingType(obj_type, window_type, on, tuple(selection), False)
    return rolling_type(obj_type, window_type, center_type, on_type), codegen


def _gen_df_rolling_out_data(rolling):
    """gen code for output data columns of Rolling calls"""
    is_variable_win = not isinstance(rolling.window_type, types.Integer)
    ftype = "variable" if is_variable_win else "fixed"
    on_arr = "None"
    if is_variable_win:
        on_arr = (
            "bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df))"
            if rolling.on is None
            else f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {rolling.obj_type.columns.index(rolling.on)})"
        )
    data_args = []
    on_arr_arg = "on_arr, " if is_variable_win else ""
    for c in rolling.selection:
        c_ind = rolling.obj_type.columns.index(c)
        if c == rolling.on:
            out = f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {c_ind})"
        else:
            out = f"bodo.hiframes.rolling.rolling_{ftype}(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {c_ind}), {on_arr_arg}window, center, False, func)"
        data_args.append(out)

    return ", ".join(data_args), on_arr


@overload_method(RollingType, "apply", inline="always", no_unliteral=True)
def overload_rolling_apply(
    rolling, func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None
):
    data_args, on_arr = _gen_df_rolling_out_data(rolling)
    header = "def impl(rolling, func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None):\n"
    header += "  df = rolling.obj\n"
    header += "  window = rolling.window\n"
    header += "  center = rolling.center\n"
    header += f"  on_arr = {on_arr}\n"
    return bodo.hiframes.dataframe_impl._gen_init_df(
        header, rolling.selection, data_args
    )


def create_rolling_overload(fname):
    """creates overloads for simple rolling functions (e.g. sum)"""

    def overload_rolling_func(rolling):
        data_args, on_arr = _gen_df_rolling_out_data(rolling)

        header = "def impl(rolling):\n"
        header += "  df = rolling.obj\n"
        header += "  window = rolling.window\n"
        header += "  center = rolling.center\n"
        header += f"  on_arr = {on_arr}\n"
        header += f"  func = '{fname}'\n"
        return bodo.hiframes.dataframe_impl._gen_init_df(
            header, rolling.selection, data_args
        )

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
        on_arr = "bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df))"
    on_arr_arg = "on_arr, " if is_variable_win else ""
    data_args = []
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
    out_cols = _get_corr_cov_out_cols(rolling, other, "corr")
    data_args, on_arr = _gen_corr_cov_out_data(
        out_cols, rolling.obj_type.columns, other.columns, rolling.window_type, "corr"
    )
    header = "def impl(rolling, other=None, pairwise=None):\n"
    header += "  df = rolling.obj\n"
    header += "  window = rolling.window\n"
    header += "  center = rolling.center\n"
    header += f"  on_arr = {on_arr}\n"
    return bodo.hiframes.dataframe_impl._gen_init_df(header, out_cols, data_args)


@overload_method(RollingType, "cov", inline="always", no_unliteral=True)
def overload_rolling_cov(rolling, other=None, pairwise=None, ddof=1):
    out_cols = _get_corr_cov_out_cols(rolling, other, "cov")
    data_args, on_arr = _gen_corr_cov_out_data(
        out_cols, rolling.obj_type.columns, other.columns, rolling.window_type, "cov"
    )
    header = "def impl(rolling, other=None, pairwise=None, ddof=1):\n"
    header += "  df = rolling.obj\n"
    header += "  window = rolling.window\n"
    header += "  center = rolling.center\n"
    header += f"  on_arr = {on_arr}\n"
    return bodo.hiframes.dataframe_impl._gen_init_df(header, out_cols, data_args)


@infer
class GetItemDataFrameRolling2(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        rolling, idx = args
        # df.rolling('A')['B', 'C']
        if isinstance(rolling, RollingType):
            if isinstance(idx, tuple):
                assert all(isinstance(c, str) for c in idx)
                selection = idx
            elif isinstance(idx, str):
                selection = (idx,)
            else:
                raise BodoError("invalid rolling selection {}".format(idx))
            ret_rolling = RollingType(rolling.df_type, rolling.on, selection, True)
            return signature(ret_rolling, *args)


class SeriesRollingType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = "SeriesRollingType({})".format(stype)
        super(SeriesRollingType, self).__init__(name)


@infer_getattr
class SeriesRollingAttribute(AttributeTemplate):
    key = SeriesRollingType

    @bound_function("rolling.apply", no_unliteral=True)
    def resolve_apply(self, ary, args, kws):
        # result is always float64 (see Pandas window.pyx:roll_generic)
        return signature(
            SeriesType(
                types.float64, index=ary.stype.index, name_typ=ary.stype.name_typ
            ),
            *args,
        )

    @bound_function("rolling.cov", no_unliteral=True)
    def resolve_cov(self, ary, args, kws):
        return signature(
            SeriesType(
                types.float64, index=ary.stype.index, name_typ=ary.stype.name_typ
            ),
            *args,
        )

    @bound_function("rolling.corr", no_unliteral=True)
    def resolve_corr(self, ary, args, kws):
        return signature(
            SeriesType(
                types.float64, index=ary.stype.index, name_typ=ary.stype.name_typ
            ),
            *args,
        )


# similar to install_array_method in arraydecl.py
def install_rolling_method(name):
    def rolling_attribute_attachment(self, ary):
        def rolling_generic(self, args, kws):
            # output is always float64
            return signature(
                SeriesType(
                    types.float64, index=ary.stype.index, name_typ=ary.stype.name_typ
                ),
                *args,
            )

        my_attr = {"key": "rolling." + name, "generic": rolling_generic}
        temp_class = type("Rolling_" + name, (AbstractTemplate,), my_attr)
        return types.BoundFunction(temp_class, ary)

    setattr(SeriesRollingAttribute, "resolve_" + name, rolling_attribute_attachment)


for fname in supported_rolling_funcs:
    install_rolling_method(fname)
