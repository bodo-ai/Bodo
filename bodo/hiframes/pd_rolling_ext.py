"""typing for rolling window functions"""

from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    signature,
)
from numba.extending import (
    infer,
    infer_getattr,
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload,
    overload_method,
    register_model,
)

import bodo
from bodo.hiframes.datetime_timedelta_ext import (
    datetime_timedelta_type,
    pd_timedelta_type,
)
from bodo.hiframes.pd_dataframe_ext import (
    DataFrameType,
    check_runtime_cols_unsupported,
)
from bodo.hiframes.pd_groupby_ext import DataFrameGroupByType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.rolling import (
    supported_rolling_funcs,
    unsupported_rolling_methods,
)
from bodo.ir.unsupported_method_template import (
    overload_unsupported_method,
)
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    get_literal_value,
    is_const_func_type,
    is_literal_type,
    is_overload_bool,
    is_overload_constant_str,
    is_overload_int,
    is_overload_none,
    raise_bodo_error,
)


class RollingType(types.Type):
    """Rolling objects from df.rolling() or Series.rolling() calls"""

    def __init__(
        self,
        obj_type,
        window_type,
        on,
        selection,
        explicit_select=False,
        series_select=False,
    ):
        # obj_type can be either Series or DataFrame
        self.obj_type = obj_type
        self.window_type = window_type
        self.on = on
        self.selection = selection
        self.explicit_select = explicit_select
        self.series_select = series_select

        super().__init__(
            name=f"RollingType({obj_type}, {window_type}, {on}, {selection}, {explicit_select}, {series_select})"
        )

    def copy(self):
        return RollingType(
            self.obj_type,
            self.window_type,
            self.on,
            self.selection,
            self.explicit_select,
            self.series_select,
        )

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@register_model(RollingType)
class RollingModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("obj", fe_type.obj_type),
            ("window", fe_type.window_type),
            ("min_periods", types.int64),
            ("center", types.bool_),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(RollingType, "obj", "obj")
make_attribute_wrapper(RollingType, "window", "window")
make_attribute_wrapper(RollingType, "center", "center")
make_attribute_wrapper(RollingType, "min_periods", "min_periods")


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
    check_runtime_cols_unsupported(df, "DataFrame.rolling()")
    unsupported_args = {"win_type": win_type, "axis": axis, "closed": closed}
    arg_defaults = {"win_type": None, "axis": 0, "closed": None}
    check_unsupported_args(
        "DataFrame.rolling",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Window",
    )
    _validate_rolling_args(df, window, min_periods, center, on)

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
        min_periods = _handle_default_min_periods(min_periods, window)
        return bodo.hiframes.pd_rolling_ext.init_rolling(
            df, window, min_periods, center, on
        )

    return impl


@overload_method(SeriesType, "rolling", inline="always", no_unliteral=True)
def overload_series_rolling(
    S,
    window,
    min_periods=None,
    center=False,
    win_type=None,
    on=None,
    axis=0,
    closed=None,
):
    unsupported_args = {"win_type": win_type, "axis": axis, "closed": closed}
    arg_defaults = {"win_type": None, "axis": 0, "closed": None}
    check_unsupported_args(
        "Series.rolling",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Window",
    )
    _validate_rolling_args(S, window, min_periods, center, on)

    def impl(
        S,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
    ):  # pragma: no cover
        min_periods = _handle_default_min_periods(min_periods, window)
        return bodo.hiframes.pd_rolling_ext.init_rolling(
            S, window, min_periods, center, on
        )

    return impl


@intrinsic(prefer_literal=True)
def init_rolling(
    typingctx, obj_type, window_type, min_periods_type, center_type, on_type
):
    """Initialize a rolling object. The data object inside can be a DataFrame, Series,
    or GroupBy."""

    def codegen(context, builder, signature, args):
        (obj_val, window_val, min_periods_val, center_val, _) = args
        rolling_type = signature.return_type

        rolling_val = cgutils.create_struct_proxy(rolling_type)(context, builder)
        rolling_val.obj = obj_val
        rolling_val.window = window_val
        rolling_val.min_periods = min_periods_val
        rolling_val.center = center_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)
        context.nrt.incref(builder, signature.args[1], window_val)
        context.nrt.incref(builder, signature.args[2], min_periods_val)
        context.nrt.incref(builder, signature.args[3], center_val)

        return rolling_val._getvalue()

    on = get_literal_value(on_type)
    if isinstance(obj_type, SeriesType):
        selection = None
    elif isinstance(obj_type, DataFrameType):
        selection = obj_type.columns
    else:
        assert isinstance(obj_type, DataFrameGroupByType), (
            f"invalid obj type for rolling: {obj_type}"
        )
        selection = obj_type.selection
    rolling_type = RollingType(obj_type, window_type, on, selection, False)
    return (
        rolling_type(obj_type, window_type, min_periods_type, center_type, on_type),
        codegen,
    )


def _handle_default_min_periods(min_periods, window):  # pragma: no cover
    return min_periods


@overload(_handle_default_min_periods)
def overload_handle_default_min_periods(min_periods, window):
    """handle default values for the min_periods kwarg."""
    if is_overload_none(min_periods):
        # return win_size if fixed, or 1 if win_type is variable
        if isinstance(window, types.Integer):
            return lambda min_periods, window: window  # pragma: no cover
        else:
            return lambda min_periods, window: 1  # pragma: no cover
    else:
        return lambda min_periods, window: min_periods  # pragma: no cover


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
            f"bodo.hiframes.rolling.rolling_{ftype}(bodo.hiframes.pd_series_ext.get_series_data(df), {on_arr_arg}index_arr, window, minp, center, func, raw)",
            on_arr,
            rolling.selection,
        )

    assert isinstance(rolling.obj_type, DataFrameType), "expected df in rolling obj"
    data_types = rolling.obj_type.data
    out_cols = []
    for c in rolling.selection:
        c_ind = rolling.obj_type.columns.index(c)
        if c == rolling.on:
            # avoid adding 'on' column if output will be Series (just ignored in Pandas)
            if len(rolling.selection) == 2 and rolling.series_select:
                continue
            out = f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {c_ind})"
            out_cols.append(c)
        else:
            # skip non-numeric data columns
            if not isinstance(data_types[c_ind].dtype, (types.Boolean, types.Number)):
                continue
            out = f"bodo.hiframes.rolling.rolling_{ftype}(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {c_ind}), {on_arr_arg}index_arr, window, minp, center, func, raw)"
            out_cols.append(c)
        data_args.append(out)

    return ", ".join(data_args), on_arr, tuple(out_cols)


@overload_method(RollingType, "apply", inline="always", no_unliteral=True)
def overload_rolling_apply(
    rolling, func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None
):
    unsupported_args = {
        "engine": engine,
        "engine_kwargs": engine_kwargs,
        "args": args,
        "kwargs": kwargs,
    }
    arg_defaults = {"engine": None, "engine_kwargs": None, "args": None, "kwargs": None}
    check_unsupported_args(
        "Rolling.apply",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Window",
    )

    # func should be function
    if not is_const_func_type(func):
        raise BodoError(
            f"Rolling.apply(): 'func' parameter must be a function, not {func} (builtin functions not supported yet)."
        )

    # raw should be bool
    if not is_overload_bool(raw):
        raise BodoError(f"Rolling.apply(): 'raw' parameter must be bool, not {raw}.")

    return _gen_rolling_impl(rolling, "apply")


@overload_method(DataFrameGroupByType, "rolling", inline="always", no_unliteral=True)
def groupby_rolling_overload(
    grp,
    window,
    min_periods=None,
    center=False,
    win_type=None,
    on=None,
    axis=0,
    closed=None,
    method="single",
):
    unsupported_args = {
        "win_type": win_type,
        "axis": axis,
        "closed": closed,
        "method": method,
    }
    arg_defaults = {"win_type": None, "axis": 0, "closed": None, "method": "single"}
    check_unsupported_args(
        "GroupBy.rolling",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Window",
    )
    _validate_rolling_args(grp, window, min_periods, center, on)

    def _impl(
        grp,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
        method="single",
    ):  # pragma: no cover
        min_periods = _handle_default_min_periods(min_periods, window)
        return bodo.hiframes.pd_rolling_ext.init_rolling(
            grp, window, min_periods, center, on
        )

    return _impl


def _gen_rolling_impl(rolling, fname, other=None):
    """generates an implementation function for rolling overloads"""
    # Support df.groupby().rolling().func() using
    # df.groupby().apply(lambda df: df.rolling().func())
    if isinstance(rolling.obj_type, DataFrameGroupByType):
        func_text = f"def impl(rolling, {_get_rolling_func_args(fname)}):\n"
        on_arg = f"'{rolling.on}'" if isinstance(rolling.on, str) else f"{rolling.on}"
        selection = ""
        if rolling.explicit_select:
            selection = "[{}]".format(
                ", ".join(
                    f"'{a}'" if isinstance(a, str) else f"{a}"
                    for a in rolling.selection
                    if a != rolling.on
                )
            )
        call_args = f_args = ""
        if fname == "apply":
            call_args = "func, raw, args, kwargs"
            f_args = "func, raw, None, None, args, kwargs"
        if fname == "corr":
            call_args = f_args = "other, pairwise"
        if fname == "cov":
            call_args = f_args = "other, pairwise, ddof"
        udf = f"lambda df, window, minp, center, {call_args}: bodo.hiframes.pd_rolling_ext.init_rolling(df, window, minp, center, {on_arg}){selection}.{fname}({f_args})"
        func_text += f"  return rolling.obj.apply({udf}, rolling.window, rolling.min_periods, rolling.center, {call_args})\n"
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl = loc_vars["impl"]
        return impl

    is_series = isinstance(rolling.obj_type, SeriesType)
    if fname in ("corr", "cov"):
        out_cols = None if is_series else _get_corr_cov_out_cols(rolling, other, fname)
        df_cols = None if is_series else rolling.obj_type.columns
        other_cols = None if is_series else other.columns
        data_args, on_arr = _gen_corr_cov_out_data(
            out_cols, df_cols, other_cols, rolling.window_type, fname
        )
    else:
        data_args, on_arr, out_cols = _gen_df_rolling_out_data(rolling)

    # NOTE: 'on' column is discarded and output is a Series if there is only one data
    # column with explicit column selection
    is_out_series = is_series or (
        len(rolling.selection) == (1 if rolling.on is None else 2)
        and rolling.series_select
    )

    header = f"def impl(rolling, {_get_rolling_func_args(fname)}):\n"
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
        c = (set(out_cols) - {rolling.on}).pop()
        name = f"'{c}'" if isinstance(c, str) else str(c)
    header += f"  name = {name}\n"
    header += "  window = rolling.window\n"
    header += "  center = rolling.center\n"
    header += "  minp = rolling.min_periods\n"
    header += f"  on_arr = {on_arr}\n"
    if fname == "apply":
        header += "  index_arr = bodo.utils.conversion.index_to_array(index)\n"
    else:
        header += f"  func = '{fname}'\n"
        # no need to pass index array
        header += "  index_arr = None\n"
        header += "  raw = False\n"

    if is_out_series:
        header += f"  return bodo.hiframes.pd_series_ext.init_series({data_args}, index, name)"
        loc_vars = {}
        _global = {"bodo": bodo}
        exec(header, _global, loc_vars)
        impl = loc_vars["impl"]
        return impl
    return bodo.hiframes.dataframe_impl._gen_init_df(header, out_cols, data_args)


def _get_rolling_func_args(fname):
    """returns the extra argument signature for rolling function with name 'fname'"""
    if fname == "apply":
        return (
            "func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None\n"
        )
    elif fname == "corr":
        return "other=None, pairwise=None, ddof=1\n"
    elif fname == "cov":
        return "other=None, pairwise=None, ddof=1\n"
    return ""


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


def _install_rolling_unsupported_methods():
    """install unsupported overloads for rolling functions"""
    for fname in unsupported_rolling_methods:
        full_name = f"pandas.core.window.rolling.Rolling.{fname}"
        overload_unsupported_method(RollingType, fname, full_name)


_install_rolling_methods()

_install_rolling_unsupported_methods()


def _get_corr_cov_out_cols(rolling, other, func_name):
    """get output column names for Rolling.corr/cov calls"""
    # TODO(ehsan): support other=None
    # XXX pandas only accepts variable window cov/corr
    # when both inputs have time index
    # TODO: Support Mixing DataFrame and Series
    if not isinstance(other, DataFrameType):
        raise_bodo_error(
            f"DataFrame.rolling.{func_name}(): requires providing a DataFrame for 'other'"
        )
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
            f"bodo.hiframes.rolling.rolling_{func_name}(bodo.hiframes.pd_series_ext.get_series_data(df), bodo.hiframes.pd_series_ext.get_series_data(other), {on_arr_arg}window, minp, center)",
            on_arr,
        )

    for c in out_cols:
        # non-common columns are just NaN values
        if c in df_cols and c in other_cols:
            i = df_cols.index(c)
            j = other_cols.index(c)
            out = f"bodo.hiframes.rolling.rolling_{func_name}(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i}), bodo.hiframes.pd_dataframe_ext.get_dataframe_data(other, {j}), {on_arr_arg}window, minp, center)"
        else:
            out = "np.full(len(df), np.nan)"
        data_args.append(out)

    return ", ".join(data_args), on_arr


@overload_method(RollingType, "corr", inline="always", no_unliteral=True)
def overload_rolling_corr(rolling, other=None, pairwise=None, ddof=1):
    args_dict = {
        "pairwise": pairwise,
        "ddof": ddof,
    }

    args_default_dict = {"pairwise": None, "ddof": 1}
    check_unsupported_args(
        "pandas.core.window.rolling.Rolling.corr",
        args_dict,
        args_default_dict,
        package_name="pandas",
        module_name="Window",
    )

    return _gen_rolling_impl(rolling, "corr", other)


@overload_method(RollingType, "cov", inline="always", no_unliteral=True)
def overload_rolling_cov(rolling, other=None, pairwise=None, ddof=1):
    args_dict = {
        "ddof": ddof,
        "pairwise": pairwise,
    }

    args_default_dict = {
        "ddof": 1,
        "pairwise": None,
    }
    check_unsupported_args(
        "pandas.core.window.rolling.Rolling.cov",
        args_dict,
        args_default_dict,
        package_name="pandas",
        module_name="Window",
    )

    return _gen_rolling_impl(rolling, "cov", other)


@infer
class GetItemDataFrameRolling2(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        rolling, idx = args
        # df.rolling('A')['B', 'C']
        if isinstance(rolling, RollingType):
            columns = (
                rolling.obj_type.selection
                if isinstance(rolling.obj_type, DataFrameGroupByType)
                else rolling.obj_type.columns
            )
            series_select = False
            if isinstance(idx, (tuple, list)):
                if len(set(idx).difference(set(columns))) > 0:  # pragma: no cover
                    raise_bodo_error(
                        f"rolling: selected column {set(idx).difference(set(columns))} not found in dataframe"
                    )
                selection = list(idx)
            else:
                if idx not in columns:  # pragma: no cover
                    raise_bodo_error(
                        f"rolling: selected column {idx} not found in dataframe"
                    )
                selection = [idx]
                series_select = True
            if rolling.on is not None:
                selection.append(rolling.on)
            ret_rolling = RollingType(
                rolling.obj_type,
                rolling.window_type,
                rolling.on,
                tuple(selection),
                True,
                series_select,
            )
            return signature(ret_rolling, *args)


@lower_builtin("static_getitem", RollingType, types.Any)
def static_getitem_df_groupby(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


@infer_getattr
class RollingAttribute(AttributeTemplate):
    key = RollingType

    def generic_resolve(self, rolling, attr):
        """handle df.rolling().B.func() case"""
        columns = ()
        if isinstance(rolling.obj_type, DataFrameGroupByType):
            columns = rolling.obj_type.selection
        if isinstance(rolling.obj_type, DataFrameType):
            columns = rolling.obj_type.columns
        if attr in columns:
            return RollingType(
                rolling.obj_type,
                rolling.window_type,
                rolling.on,
                # 'on' column is always kept in selected columns
                (attr,) if rolling.on is None else (attr, rolling.on),
                True,
                True,
            )


def _validate_rolling_args(obj, window, min_periods, center, on):
    """Validate argument types of DataFrame/Series/DataFrameGroupBy.rolling() calls"""
    # similar to argument validation in Pandas:
    # https://github.com/pandas-dev/pandas/blob/93d46cfc76f939ec5e2148c35728fad4e2389c90/pandas/core/window/rolling.py#L196
    # https://github.com/pandas-dev/pandas/blob/93d46cfc76f939ec5e2148c35728fad4e2389c90/pandas/core/window/rolling.py#L1393
    assert isinstance(obj, (SeriesType, DataFrameType, DataFrameGroupByType)), (
        "invalid rolling obj"
    )
    func_name = (
        "Series"
        if isinstance(obj, SeriesType)
        else "DataFrame"
        if isinstance(obj, DataFrameType)
        else "DataFrameGroupBy"
    )

    # window should be integer or time offset (str, timedelta)
    # TODO(ehsan): support offset types like Week
    if not (
        is_overload_int(window)
        or is_overload_constant_str(window)
        or window == bodo.types.string_type
        or window in (pd_timedelta_type, datetime_timedelta_type)
    ):
        raise BodoError(
            f"{func_name}.rolling(): 'window' should be int or time offset (str, pd.Timedelta, datetime.timedelta), not {window}"
        )

    # center should be bool
    if not is_overload_bool(center):
        raise BodoError(
            f"{func_name}.rolling(): center must be a boolean, not {center}"
        )

    # min_periods should be None or int
    if not (is_overload_none(min_periods) or isinstance(min_periods, types.Integer)):
        raise BodoError(
            f"{func_name}.rolling(): min_periods must be an integer, not {min_periods}"
        )

    # 'on' not supported for Series yet (TODO: support)
    if isinstance(obj, SeriesType) and not is_overload_none(on):
        raise BodoError(
            f"{func_name}.rolling(): 'on' not supported for Series yet (can use a DataFrame instead)."
        )

    col_names = (
        obj.columns
        if isinstance(obj, DataFrameType)
        else obj.df_type.columns
        if isinstance(obj, DataFrameGroupByType)
        else []
    )
    data_types = (
        [obj.data]
        if isinstance(obj, SeriesType)
        else obj.data
        if isinstance(obj, DataFrameType)
        else obj.df_type.data
    )

    # 'on' should be in column names
    if not is_overload_none(on) and (
        not is_literal_type(on) or get_literal_value(on) not in col_names
    ):
        raise BodoError(
            f"{func_name}.rolling(): 'on' should be a constant column name."
        )

    # 'on' column should be datetime
    if not is_overload_none(on):
        on_data_type = data_types[col_names.index(get_literal_value(on))]
        if (
            not isinstance(on_data_type, types.Array)
            or on_data_type.dtype != bodo.types.datetime64ns
        ) and on_data_type != bodo.types.DatetimeArrayType(None):
            raise BodoError(
                f"{func_name}.rolling(): 'on' column should have datetime64 data."
            )

    # input should have numeric data types
    if not any(isinstance(A.dtype, (types.Boolean, types.Number)) for A in data_types):
        raise BodoError(f"{func_name}.rolling(): No numeric types to aggregate")
