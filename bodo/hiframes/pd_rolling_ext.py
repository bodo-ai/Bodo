# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""typing for rolling window functions
"""
from numba.core import types
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    bound_function,
    infer_global,
    signature,
)
from numba.extending import (
    infer,
    infer_getattr,
    lower_builtin,
    models,
    overload_method,
    register_model,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.rolling import supported_rolling_funcs
from bodo.utils.typing import BodoError


class RollingType(types.Type):
    """Temporary type class for RollingType objects before transformation
    to rolling node.
    """

    def __init__(self, df_type, on, selection, explicit_select=False):

        self.df_type = df_type
        self.on = on
        self.selection = selection
        self.explicit_select = explicit_select

        super(RollingType, self).__init__(
            name="RollingType({}, {}, {}, {})".format(
                df_type, on, selection, explicit_select
            )
        )

    def copy(self):
        # XXX is copy necessary?
        # TODO: key attribute?
        return RollingType(self.df_type, self.on, self.selection, self.explicit_select)


# dummy model since info is kept in type
# TODO: add df object and win/center vals to allow control flow?
register_model(RollingType)(models.OpaqueModel)


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
        return bodo.hiframes.pd_rolling_ext.rolling_dummy(df, window, center, on)

    return _impl


# a dummy rolling function that will be replace in dataframe_pass
def rolling_dummy(df, window, center, on):  # pragma: no cover
    return 0


@infer_global(rolling_dummy)
class RollingTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, window, center, on = args

        if on == types.none:
            on = None
        else:
            if not isinstance(on, types.StringLiteral):
                raise BodoError("'on' argument to rolling() should be constant string")
            on = on.literal_value

        selection = list(df.columns)
        if on is not None:
            selection.remove(on)

        out_typ = RollingType(df, on, tuple(selection), False)
        return signature(out_typ, *args)


RollingTyper._no_unliteral = True


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(rolling_dummy, types.VarArg(types.Any))
def lower_rolling_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


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


@infer_getattr
class DataframeRollingAttribute(AttributeTemplate):
    key = RollingType

    def generic_resolve(self, rolling, func_name):
        if func_name not in supported_rolling_funcs:
            raise BodoError(
                "only ({}) supported in rolling".format(
                    ", ".join(supported_rolling_funcs)
                )
            )
        template_key = "rolling." + func_name
        # output is always float64
        out_arr = types.Array(types.float64, 1, "C")

        # TODO: handle Series case (explicit select)
        columns = rolling.selection

        # handle 'on' case
        if rolling.on is not None:
            columns = columns + (rolling.on,)
        # Pandas sorts the output column names _flex_binary_moment
        # line: res_columns = arg1.columns.union(arg2.columns)
        columns = tuple(sorted(columns))
        n_out_cols = len(columns)
        out_data = [out_arr] * n_out_cols
        if rolling.on is not None:
            # offset key's data type is preserved
            out_ind = columns.index(rolling.on)
            in_ind = rolling.df_type.columns.index(rolling.on)
            out_data[out_ind] = rolling.df_type.data[in_ind]
        index = rolling.df_type.index
        out_typ = DataFrameType(tuple(out_data), index, columns)

        class MethodTemplate(AbstractTemplate):
            key = template_key

            def generic(self, args, kws):
                if func_name in ("cov", "corr"):
                    if len(args) != 1:
                        raise BodoError(
                            "rolling {} requires one argument (other)".format(func_name)
                        )
                    # XXX pandas only accepts variable window cov/corr
                    # when both inputs have time index
                    if rolling.on is not None:
                        raise BodoError(
                            "variable window rolling {} not supported yet.".format(
                                func_name
                            )
                        )
                    # TODO: support variable window rolling cov/corr which is only
                    # possible in pandas with time index
                    other = args[0]
                    # df on df cov/corr returns common columns only (without
                    # pairwise flag)
                    # TODO: support pairwise arg
                    out_cols = tuple(
                        sorted(set(columns) | set(other.columns), key=lambda k: str(k))
                    )
                    return signature(
                        DataFrameType((out_arr,) * len(out_cols), index, out_cols),
                        *args,
                    )
                return signature(out_typ, *args)

        return types.BoundFunction(MethodTemplate, rolling)


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
