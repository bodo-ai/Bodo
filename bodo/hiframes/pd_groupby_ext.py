# Copyright (C) 2019 Bodo Inc. All rights reserved.
import numba
from numba.core import types
from numba.extending import (
    models,
    register_model,
    lower_cast,
    infer_getattr,
    type_callable,
    infer,
    overload,
    make_attribute_wrapper,
    intrinsic,
    lower_builtin,
    overload_method,
)
from numba.core.typing.templates import (
    infer_global,
    AbstractTemplate,
    signature,
    AttributeTemplate,
    bound_function,
)
from bodo.libs.int_arr_ext import IntegerArrayType, IntDtype
import bodo
from bodo.hiframes.pd_series_ext import SeriesType, _get_series_array_type
from bodo.libs.str_ext import string_type
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import RangeIndexType
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.ir.aggregate import get_agg_func
from bodo.utils.typing import (
    list_cumulative,
    BodoError,
    raise_const_error,
    is_overload_none,
    get_const_str_list,
    is_overload_true,
    is_overload_false,
    is_overload_zero,
    is_overload_constant_bool,
    is_overload_constant_str,
    is_overload_constant_str_list,
    is_dtype_nullable,
    ConstDictType,
    get_overload_const_func,
    get_overload_const_str,
)
from bodo.utils.transform import get_const_func_output_type
from bodo.utils.utils import is_expr


class DataFrameGroupByType(types.Type):  # TODO: IterableType over groups
    """Temporary type class for DataFrameGroupBy objects before transformation
    to aggregate node.
    """

    def __init__(self, df_type, keys, selection, as_index, explicit_select=False):

        self.df_type = df_type
        self.keys = keys
        self.selection = selection
        self.as_index = as_index
        self.explicit_select = explicit_select

        super(DataFrameGroupByType, self).__init__(
            name="DataFrameGroupBy({}, {}, {}, {}, {})".format(
                df_type, keys, selection, as_index, explicit_select
            )
        )

    def copy(self):
        # XXX is copy necessary?
        return DataFrameGroupByType(
            self.df_type, self.keys, self.selection, self.as_index, self.explicit_select
        )


# dummy model since info is kept in type
# TODO: add df object to allow control flow?
register_model(DataFrameGroupByType)(models.OpaqueModel)


@overload_method(DataFrameType, "groupby", no_unliteral=True)
def df_groupby_overload(
    df,
    by=None,
    axis=0,
    level=None,
    as_index=True,
    sort=False,
    group_keys=True,
    squeeze=False,
    observed=False,
):

    validate_groupby_spec(
        df, by, axis, level, as_index, sort, group_keys, squeeze, observed
    )

    def _impl(
        df,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=False,
        group_keys=True,
        squeeze=False,
        observed=False,
    ):  # pragma: no cover
        return bodo.hiframes.pd_groupby_ext.groupby_dummy(df, by, as_index)

    return _impl


def validate_groupby_spec(
    df, by, axis, level, as_index, sort, group_keys, squeeze, observed
):
    """
    validate df.groupby() specifications: In addition to consistent error checking
    with pandas, we also check for unsupported specs.

    An error is raised if the spec is invalid.
    """

    # make sure 'by' is supplied
    if is_overload_none(by):
        raise BodoError("groupby(): 'by' must be supplied.")

    # make sure axis has default value 0
    if not is_overload_zero(axis):
        raise BodoError("groupby(): 'axis' parameter only supports integer value 0.")

    # make sure level is not specified
    if not is_overload_none(level):
        raise BodoError(
            "groupby(): 'level' is not supported since MultiIndex is not supported."
        )

    # make sure by is a const str list
    if not is_overload_constant_str(by) and not is_overload_constant_str_list(by):
        raise_const_error(
            "groupby(): 'by' parameter only supports a constant column label or column labels."
        )

    # make sure by has valid label(s)
    if len(set(get_const_str_list(by)).difference(set(df.columns))) > 0:
        raise_const_error(
            "groupby(): invalid key {} for 'by' (not available in columns {}).".format(
                get_const_str_list(by), df.columns
            )
        )

    # make sure as_index is of type bool
    if not is_overload_constant_bool(as_index):
        raise BodoError(
            "groupby(): 'as_index' parameter must be of type bool, ",
            "not {}.".format(as_index),
        )

    # make sure sort is the default value, sort=True not supported
    if not is_overload_false(sort):
        raise BodoError("groupby(): 'sort' parameter only supports default value False")

    # make sure group_keys has default value True
    if not is_overload_true(group_keys):
        raise BodoError(
            "groupby(): 'group_keys' parameter only supports default value True."
        )

    # make sure squeeze has default value False
    if not is_overload_false(squeeze):
        raise BodoError(
            "groupby(): 'squeeze' parameter only supports default value False."
        )

    # make sure observed has default value False
    if not is_overload_false(observed):
        raise BodoError(
            "groupby(): 'observed' parameter only supports default value False."
        )


def validate_udf(func_name, func):
    if not isinstance(
        func,
        (
            types.functions.MakeFunctionLiteral,
            bodo.utils.typing.FunctionLiteral,
            types.Dispatcher,
        ),
    ):
        raise BodoError(
            "Groupby.{}: 'func' must be user defined function".format(func_name)
        )


# a dummy groupby function that will be replace in dataframe_pass
def groupby_dummy(df, by, as_index):  # pragma: no cover
    return 0


@infer_global(groupby_dummy)
class GroupbyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, by, as_index = args

        if is_overload_constant_str(by):
            keys = (get_overload_const_str(by),)
        elif is_overload_constant_str_list(by):
            keys = tuple(get_const_str_list(by))

        selection = list(df.columns)
        for k in keys:
            selection.remove(k)

        if isinstance(as_index, bodo.utils.typing.BooleanLiteral):
            as_index = as_index.literal_value
        else:
            # XXX as_index type is just bool when value not passed. Therefore,
            # we assume the default True value.
            # TODO: more robust fix or just check
            as_index = True

        out_typ = DataFrameGroupByType(df, keys, tuple(selection), as_index, False)
        return signature(out_typ, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(groupby_dummy, types.VarArg(types.Any))
def lower_groupby_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


@infer
class GetItemDataFrameGroupBy(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        grpby, idx = args
        # df.groupby('A')['B', 'C']
        if isinstance(grpby, DataFrameGroupByType):
            if isinstance(idx, (tuple, list)):
                assert all(isinstance(c, str) for c in idx)
                if len(set(idx).difference(set(grpby.df_type.columns))) > 0:
                    raise_const_error(
                        "groupby: selected column {} not found in dataframe".format(
                            set(idx).difference(set(grpby.df_type.columns))
                        )
                    )
                selection = idx
            elif isinstance(idx, str):
                if idx not in grpby.df_type.columns:
                    raise_const_error(
                        "groupby: selected column {} not found in dataframe".format(idx)
                    )
                selection = (idx,)
            else:
                raise ValueError("invalid groupby selection {}".format(idx))
            ret_grp = DataFrameGroupByType(
                grpby.df_type, grpby.keys, selection, grpby.as_index, True
            )
            return signature(ret_grp, *args)


def get_groupby_output_dtype(arr_type, func_name):
    """
    Return output dtype for groupby aggregation function based on the
    function and the input array type and dtype.
    """
    in_dtype = arr_type.dtype
    if (
        func_name == "median"
        and not isinstance(in_dtype, types.Float)
        and not isinstance(in_dtype, types.Integer)
    ):
        raise BodoError("For median, only column of integer or float type are allowed")
    if (
        not isinstance(in_dtype, types.Integer)
        and not isinstance(in_dtype, types.Float)
        and not isinstance(in_dtype, types.Boolean)
    ):
        is_list_string = isinstance(in_dtype, numba.core.types.containers.List) and in_dtype.dtype == types.unicode_type
        if is_list_string or in_dtype == types.unicode_type:
            if func_name not in {"count", "nunique", "min", "max", "sum", "first", "last"}:
                raise BodoError(
                    "column type of strings or list of strings is not supported in groupby built-in function {}".format(
                        func_name
                    )
                )
        else:
            if func_name not in {"count", "nunique", "min", "max", "first", "last"}:
                raise BodoError(
                    "column type of {} is not supported in groupby built-in function {}".format(
                        in_dtype, func_name
                    )
                )
    if isinstance(in_dtype, types.Boolean) and func_name in {"cumsum", "sum"}:
        raise BodoError(
            "groupby built-in functions {}"
            " does not support boolean column".format(func_name)
        )
    if func_name in {"count", "nunique"}:
        return types.int64
    elif func_name in {"median", "mean", "var", "std"}:
        return types.float64
    else:
        if isinstance(arr_type, IntegerArrayType):
            return IntDtype(in_dtype)
        return in_dtype  # default: return same dtype as input


@infer_getattr
class DataframeGroupByAttribute(AttributeTemplate):
    key = DataFrameGroupByType

    def _get_keys_not_as_index(
        self, grp, out_columns, out_data, multi_level_names=False
    ):
        """ Add groupby keys to output columns (to be used when
            as_index=False) """
        for k in grp.keys:
            if multi_level_names:
                out_columns.append((k, ""))
            else:
                out_columns.append(k)
            ind = grp.df_type.columns.index(k)
            out_data.append(grp.df_type.data[ind])

    def _get_agg_typ(self, grp, args, func_name, func=None):
        index = RangeIndexType(types.none)
        out_data = []
        out_columns = []
        # add key columns of not as_index
        if not grp.as_index:
            self._get_keys_not_as_index(grp, out_columns, out_data)
        else:
            if len(grp.keys) > 1:
                key_col_inds = tuple(
                    grp.df_type.columns.index(grp.keys[i]) for i in range(len(grp.keys))
                )
                arr_types = tuple(grp.df_type.data[ind] for ind in key_col_inds)
                index = MultiIndexType(
                    arr_types, tuple(types.StringLiteral(k) for k in grp.keys)
                )
            else:
                ind = grp.df_type.columns.index(grp.keys[0])
                index = bodo.hiframes.pd_index_ext.array_typ_to_index(
                    grp.df_type.data[ind], types.StringLiteral(grp.keys[0])
                )

        # get output type for each selected column
        for c in grp.selection:
            out_columns.append(c)
            ind = grp.df_type.columns.index(c)
            data = grp.df_type.data[ind]

            if func_name == "agg":
                try:
                    # input to UDFs is a Series
                    in_series_typ = SeriesType(data.dtype, data, None, string_type)
                    out_dtype = get_const_func_output_type(
                        func, (in_series_typ,), self.context
                    )
                except:
                    raise BodoError(
                        "Groupy.agg()/Groupy.aggregate(): column {col} of type {type} "
                        "is unsupported/not a valid input type for user defined function".format(
                            col=c, type=data.dtype
                        )
                    )
            else:
                out_dtype = get_groupby_output_dtype(data, func_name)

            out_arr = _get_series_array_type(out_dtype)
            out_data.append(out_arr)

        out_res = DataFrameType(tuple(out_data), index, tuple(out_columns))
        # XXX output becomes series if single output and explicitly selected
        if len(grp.selection) == 1 and grp.explicit_select and grp.as_index:
            if isinstance(out_data[0], IntegerArrayType):
                dtype = IntDtype(out_data[0].dtype)
            else:
                dtype = out_data[0].dtype
            out_res = SeriesType(
                dtype, index=index, name_typ=types.StringLiteral(grp.selection[0])
            )
        return signature(out_res, *args)

    def _get_agg_funcname_and_outtyp(self, grp, args, col, f_val):
        """ Get function name and output type for a function used in
            groupby.agg(), given by f_val (can be a string constant or
            user-defined function) applied to column col """
        is_udf = True  # is user-defined function
        if isinstance(f_val, str):
            is_udf = False
            f_name = f_val
        elif is_overload_constant_str(f_val):
            is_udf = False
            f_name = get_overload_const_str(f_val)
        if not is_udf:
            if f_name not in bodo.ir.aggregate.supported_agg_funcs[:-1]:
                raise BodoError("unsupported aggregate function {}".format(f_name))
            # run typer on a groupby with just column col
            ret_grp = DataFrameGroupByType(
                grp.df_type, grp.keys, (col,), grp.as_index, True
            )
            out_tp = self._get_agg_typ(ret_grp, args, f_name).return_type
        else:
            # assume udf
            if is_expr(f_val, "make_function"):
                f = types.functions.MakeFunctionLiteral(f_val)
            else:
                assert isinstance(f_val, (types.MakeFunctionLiteral, types.Dispatcher))
                f = f_val
            validate_udf("agg", f)
            func = get_overload_const_func(f)
            code = func.code if hasattr(func, "code") else func.__code__
            f_name = code.co_name
            # run typer on a groupby with just column col
            ret_grp = DataFrameGroupByType(
                grp.df_type, grp.keys, (col,), grp.as_index, True
            )
            out_tp = self._get_agg_typ(ret_grp, args, "agg", f).return_type
        return f_name, out_tp

    def _resolve_agg(self, grp, args, kws):
        if len(args) == 0:
            raise BodoError("Groupby.agg()/aggregate(): Must provide 'func'")

        func = args[0]
        has_cumulative_ops = False

        def _append_out_type(grp, out_data, out_tp):
            if grp.as_index is False:
                # _get_agg_typ also returns the index (keys) as part of
                # out_tp, but we already added them at the beginning
                # (by calling _get_keys_not_as_index), so we skip them
                out_data.append(out_tp.data[len(grp.keys)])
            else:
                # out_tp is assumed to be a SeriesType (see _get_agg_typ)
                out_data.append(out_tp.data)

        # multi-function constant dictionary case
        if isinstance(func, ConstDictType):
            # get mapping of column names to functions:
            # string -> string or tuple of strings (tuple when multiple
            # functions are applied to a column)
            col_map = {
                func.consts[2 * i]: func.consts[2 * i + 1]
                for i in range(len(func.consts) // 2)
            }

            # make sure selected columns exist in dataframe
            if any(c not in grp.selection for c in col_map.keys()):
                raise_const_error(
                    "Selected column names {} not all available in dataframe column names {}".format(
                        tuple(col_map.keys()), grp.selection
                    )
                )

            # if a list/tuple of functions is applied to any column, have to use
            # MultiLevel for every column (even if list/tuple length is one)
            multi_level_names = any(
                isinstance(f_val, (tuple, list)) for f_val in col_map.values()
            )

            # get output names and output types
            out_columns = []
            out_data = []
            if not grp.as_index:
                self._get_keys_not_as_index(
                    grp, out_columns, out_data, multi_level_names=multi_level_names
                )
            for col_name, f_val in col_map.items():
                if isinstance(f_val, (tuple, list)):
                    # TODO tuple containing function objects (not just strings)
                    for f in f_val:
                        f_name, out_tp = self._get_agg_funcname_and_outtyp(
                            grp, args, col_name, f
                        )
                        has_cumulative_ops = f_name in list_cumulative
                        # TODO f_name == "<lambda>"
                        # output column name is 2-level (col_name, func_name)
                        # This happens, for example, with
                        # df.groupby(...).agg({"A": [f1, f2]})
                        out_columns.append((col_name, f_name))
                        _append_out_type(grp, out_data, out_tp)
                else:
                    f_name, out_tp = self._get_agg_funcname_and_outtyp(
                        grp, args, col_name, f_val
                    )
                    has_cumulative_ops = f_name in list_cumulative
                    if multi_level_names:
                        out_columns.append((col_name, f_name))
                    else:
                        out_columns.append(col_name)
                    _append_out_type(grp, out_data, out_tp)

            if has_cumulative_ops:
                # result of groupby.cumsum, etc. doesn't have a group index
                # So instead we set from the input index
                index = grp.df_type.index
            else:
                index = out_tp.index
            out_res = DataFrameType(tuple(out_data), index, tuple(out_columns))
            return signature(out_res, *args)

        # multi-function tuple case
        if isinstance(func, types.BaseTuple):
            if not (len(grp.selection) == 1 and grp.explicit_select):
                raise BodoError(
                    "Groupby.agg()/aggregate(): must select exactly one column when more than one functions supplied"
                )
            assert len(func) > 0
            out_data = []
            out_columns = []
            lambda_count = 0
            if not grp.as_index:
                self._get_keys_not_as_index(grp, out_columns, out_data)
            for f_val in func.types:
                f_name, out_tp = self._get_agg_funcname_and_outtyp(
                    grp, args, grp.selection[0], f_val
                )
                has_cumulative_ops = f_name in list_cumulative
                # if tuple has lambdas they will be named <lambda_0>,
                # <lambda_1>, ... in output
                if f_name == "<lambda>":
                    f_name = "<lambda_" + str(lambda_count) + ">"
                    lambda_count += 1
                out_columns.append(f_name)
                _append_out_type(grp, out_data, out_tp)
            if has_cumulative_ops:
                # result of groupby.cumsum, etc. doesn't have a group index
                index = grp.df_type.index
            else:
                index = out_tp.index
            out_res = DataFrameType(tuple(out_data), index, tuple(out_columns))
            return signature(out_res, *args)

        validate_udf("agg", func)
        return self._get_agg_typ(grp, args, "agg", func)

    @bound_function("groupby.agg")
    def resolve_agg(self, grp, args, kws):
        return self._resolve_agg(grp, args, kws)

    @bound_function("groupby.aggregate")
    def resolve_aggregate(self, grp, args, kws):
        return self._resolve_agg(grp, args, kws)

    @bound_function("groupby.sum")
    def resolve_sum(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "sum")

    @bound_function("groupby.count")
    def resolve_count(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "count")

    @bound_function("groupby.nunique")
    def resolve_nunique(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "nunique")

    @bound_function("groupby.median")
    def resolve_median(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "median")

    @bound_function("groupby.mean")
    def resolve_mean(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "mean")

    @bound_function("groupby.min")
    def resolve_min(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "min")

    @bound_function("groupby.max")
    def resolve_max(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "max")

    @bound_function("groupby.prod")
    def resolve_prod(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "prod")

    @bound_function("groupby.var")
    def resolve_var(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "var")

    @bound_function("groupby.std")
    def resolve_std(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "std")

    @bound_function("groupby.first")
    def resolve_first(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "first")

    @bound_function("groupby.last")
    def resolve_last(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "last")

    def resolve_cumulative(self, grp, args, kws, msg, is_minmax):
        """For datetime and timedelta datatypes, we can support cummin / cummax,
        but not cumsum / cumprod. Hence the is_minmax entry"""
        index = grp.df_type.index
        out_columns = []
        out_data = []
        for c in grp.selection:
            out_columns.append(c)
            ind = grp.df_type.columns.index(c)
            data = grp.df_type.data[ind]
            if not is_minmax:
                if not isinstance(data.dtype, (types.Integer, types.Float)):
                    raise BodoError(msg)
            else:
                if not isinstance(data.dtype, types.Integer) and not is_dtype_nullable(
                    data.dtype
                ):
                    raise BodoError(msg)
            out_data.append(data)
        out_res = DataFrameType(tuple(out_data), index, tuple(out_columns))
        # XXX output becomes series if single output and explicitly selected
        if len(grp.selection) == 1 and grp.explicit_select and grp.as_index:
            out_res = SeriesType(
                out_data[0].dtype,
                data=out_data[0],
                index=index,
                name_typ=types.StringLiteral(grp.selection[0]),
            )
        return signature(out_res, *args)

    @bound_function("groupby.cumsum")
    def resolve_cumsum(self, grp, args, kws):
        msg = "Groupby.cumsum() only supports columns of types integer, float, string or liststring"
        return self.resolve_cumulative(grp, args, kws, msg, False)

    @bound_function("groupby.cumprod")
    def resolve_cumprod(self, grp, args, kws):
        msg = "Groupby.cumprod() only supports columns of types integer and float"
        return self.resolve_cumulative(grp, args, kws, msg, False)

    @bound_function("groupby.cummin")
    def resolve_cummin(self, grp, args, kws):
        msg = "Groupby.cummin() only supports columns of types integer, float, string, liststring, date, datetime or timedelta"
        return self.resolve_cumulative(grp, args, kws, msg, True)

    @bound_function("groupby.cummax")
    def resolve_cummax(self, grp, args, kws):
        msg = "Groupby.cummax() only supports columns of types integer, float, string, liststring, date, datetime or timedelta"
        return self.resolve_cumulative(grp, args, kws, msg, True)

    def generic_resolve(self, grpby, attr):
        if attr not in grpby.df_type.columns:
            raise_const_error(
                "groupby: invalid attribute {} (column not found in dataframe or unsupported function)".format(
                    attr
                )
            )
        return DataFrameGroupByType(
            grpby.df_type, grpby.keys, (attr,), grpby.as_index, True
        )


# a dummy pivot_table function that will be replace in dataframe_pass
def pivot_table_dummy(
    df, values, index, columns, aggfunc, _pivot_values
):  # pragma: no cover
    return 0


@infer_global(pivot_table_dummy)
class PivotTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, values, index, columns, aggfunc, _pivot_values = args

        if not (
            is_overload_constant_str(values)
            and is_overload_constant_str(index)
            and is_overload_constant_str(columns)
        ):
            raise BodoError(
                "pivot_table() only support string constants for "
                "'values', 'index' and 'columns' arguments"
            )

        values = values.literal_value
        index = index.literal_value
        columns = columns.literal_value

        # get output data type
        data = df.data[df.columns.index(values)]
        func = get_agg_func(None, aggfunc.literal_value, None)
        f_ir = numba.core.ir_utils.get_ir_of_code(func.__globals__, func.__code__)
        in_series_typ = SeriesType(data.dtype, data, None, string_type)
        bodo.ir.aggregate.replace_closures(f_ir, func.__closure__, func.__code__)
        _, out_dtype, _ = numba.core.typed_passes.type_inference_stage(
            self.context, f_ir, (in_series_typ,), None
        )
        out_arr_typ = _get_series_array_type(out_dtype)

        pivot_vals = _pivot_values.meta
        n_vals = len(pivot_vals)
        df_index = RangeIndexType(types.none)
        out_df = DataFrameType((out_arr_typ,) * n_vals, df_index, tuple(pivot_vals))

        return signature(out_df, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(pivot_table_dummy, types.VarArg(types.Any))
def lower_pivot_table_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


# a dummy crosstab function that will be replace in dataframe_pass
def crosstab_dummy(index, columns, _pivot_values):  # pragma: no cover
    return 0


@infer_global(crosstab_dummy)
class CrossTabTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        index, columns, _pivot_values = args

        # TODO: support agg func other than frequency
        out_arr_typ = types.Array(types.int64, 1, "C")

        pivot_vals = _pivot_values.meta
        n_vals = len(pivot_vals)
        df_index = RangeIndexType(types.none)
        out_df = DataFrameType((out_arr_typ,) * n_vals, df_index, tuple(pivot_vals))

        return signature(out_df, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(crosstab_dummy, types.VarArg(types.Any))
def lower_crosstab_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)
