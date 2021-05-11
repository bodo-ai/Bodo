# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Support for Pandas Groupby operations
"""
from enum import Enum

import numba
import numpy as np
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.core.registry import CPUDispatcher
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
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload,
    overload_method,
    register_model,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType
from bodo.libs.array import (
    arr_info_list_to_table,
    array_to_info,
    delete_table,
    delete_table_decref_arrays,
    get_groupby_labels,
    get_shuffle_info,
    info_from_table,
    info_to_array,
    reverse_shuffle_table,
    shuffle_table,
)
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.decimal_arr_ext import Decimal128Type
from bodo.libs.int_arr_ext import IntDtype, IntegerArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.transform import (
    gen_const_tup,
    get_call_expr_arg,
    get_const_func_output_type,
)
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    create_unsupported_overload,
    dtype_to_array_type,
    get_index_data_arr_types,
    get_index_name_types,
    get_literal_value,
    get_overload_const_func,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_constant_dict,
    get_udf_error_msg,
    get_udf_out_arr_type,
    is_dtype_nullable,
    is_literal_type,
    is_overload_constant_bool,
    is_overload_constant_dict,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_none,
    is_overload_true,
    list_cumulative,
    raise_bodo_error,
    raise_const_error,
)
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


@register_model(DataFrameGroupByType)
class GroupbyModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("obj", fe_type.df_type),
        ]
        super(GroupbyModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameGroupByType, "obj", "obj")


def validate_udf(func_name, func):
    if not isinstance(
        func,
        (
            types.functions.MakeFunctionLiteral,
            bodo.utils.typing.FunctionLiteral,
            types.Dispatcher,
            CPUDispatcher,
        ),
    ):
        raise_const_error(f"Groupby.{func_name}: 'func' must be user defined function")


@intrinsic
def init_groupby(typingctx, obj_type, by_type, as_index_type=None):
    """Initialize a groupby object. The data object inside can be a DataFrame"""

    def codegen(context, builder, signature, args):
        obj_val = args[0]
        groupby_type = signature.return_type
        groupby_val = cgutils.create_struct_proxy(groupby_type)(context, builder)
        groupby_val.obj = obj_val
        context.nrt.incref(builder, signature.args[0], obj_val)
        return groupby_val._getvalue()

    # get groupby key column names
    if is_overload_constant_list(by_type):
        keys = tuple(get_overload_const_list(by_type))
    elif is_literal_type(by_type):
        keys = (get_literal_value(by_type),)

    selection = list(obj_type.columns)
    for k in keys:
        selection.remove(k)

    if is_overload_constant_bool(as_index_type):
        as_index = is_overload_true(as_index_type)
    else:
        # XXX as_index type is just bool when value not passed. Therefore,
        # we assume the default True value.
        # TODO: more robust fix or just check
        as_index = True

    groupby_type = DataFrameGroupByType(
        obj_type, keys, tuple(selection), as_index, False
    )
    return groupby_type(obj_type, by_type, as_index_type), codegen


# dummy lowering for groupby.size since it is used in Series.value_counts()
# groupby.apply is used in groupby.rolling
@lower_builtin("groupby.count", types.VarArg(types.Any))
@lower_builtin("groupby.size", types.VarArg(types.Any))
@lower_builtin("groupby.apply", types.VarArg(types.Any))
def lower_groupby_count_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


@infer
class GetItemDataFrameGroupBy(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        grpby, idx = args
        # df.groupby('A')['B', 'C']
        if isinstance(grpby, DataFrameGroupByType):
            if isinstance(idx, (tuple, list)):
                if len(set(idx).difference(set(grpby.df_type.columns))) > 0:
                    raise_const_error(
                        "groupby: selected column {} not found in dataframe".format(
                            set(idx).difference(set(grpby.df_type.columns))
                        )
                    )
                selection = idx
            else:
                if idx not in grpby.df_type.columns:
                    raise_const_error(
                        "groupby: selected column {} not found in dataframe".format(idx)
                    )
                selection = (idx,)
            ret_grp = DataFrameGroupByType(
                grpby.df_type, grpby.keys, selection, grpby.as_index, True
            )
            return signature(ret_grp, *args)


# dummy lowering for groupby getitem to avoid errors (e.g. test_series_groupby_arr)
@lower_builtin("static_getitem", DataFrameGroupByType, types.Any)
def static_getitem_df_groupby(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


def get_groupby_output_dtype(arr_type, func_name, index_type=None):
    """
    Return output dtype for groupby aggregation function based on the
    function and the input array type and dtype.
    If the operation is not feasible (e.g. summing dates) then an error message
    is passed upward to be decided according to the context.
    """
    is_list_string = arr_type == ArrayItemArrayType(string_array_type)
    in_dtype = arr_type.dtype
    # Bodo don't support DatetimeTimeDeltaType. use (timedelta64 instead)
    if isinstance(in_dtype, bodo.hiframes.datetime_timedelta_ext.DatetimeTimeDeltaType):
        raise BodoError(
            "column type of {} is not supported in groupby built-in function {}.\n \
            If you are trying to set NULL values for timedelta64 in regular Python, \
            consider using np.timedelta64('nat') instead of None".format(
                in_dtype, func_name
            )
        )
    if func_name == "median" and not isinstance(
        in_dtype, (Decimal128Type, types.Float, types.Integer)
    ):
        return (
            None,
            "For median, only column of integer, float or Decimal type are allowed",
        )
    # [BE-416] Support with list
    # [BE-433] Support with tuples
    if (func_name in ("first", "last", "sum", "prod", "min", "max", "count")) and (
        isinstance(arr_type, (TupleArrayType, ArrayItemArrayType))
    ):
        return (
            None,
            "column type of list of {} is not supported in groupby built-in function {}".format(
                in_dtype, func_name
            ),
        )

    if (func_name in {"median", "mean", "var", "std"}) and isinstance(
        in_dtype, (Decimal128Type, types.Integer, types.Float)
    ):
        return types.float64, "ok"
    if not isinstance(in_dtype, (types.Integer, types.Float, types.Boolean)):
        if is_list_string or in_dtype == types.unicode_type:
            if func_name not in {
                "count",
                "nunique",
                "min",
                "max",
                "sum",
                "first",
                "last",
            }:
                return (
                    None,
                    f"column type of strings or list of strings is not supported in groupby built-in function {func_name}",
                )
        else:
            if isinstance(in_dtype, bodo.PDCategoricalDtype):
                if func_name in ("min", "max") and not in_dtype.ordered:
                    return (
                        None,
                        f"categorical column must be ordered in groupby built-in function {func_name}",
                    )
            if func_name not in {
                "count",
                "nunique",
                "min",
                "max",
                "first",
                "last",
            }:
                return (
                    None,
                    f"column type of {in_dtype} is not supported in groupby built-in function {func_name}",
                )

    if isinstance(in_dtype, types.Boolean) and func_name in {
        "cumsum",
        "sum",
        "mean",
        "std",
        "var",
        # See: [BE-549]
        "idxmin",
        "idxmax",
    }:
        return (
            None,
            f"groupby built-in functions {func_name} does not support boolean column",
        )
    # See: [BE-549]
    if func_name in {"idxmin", "idxmax"} and isinstance(arr_type, IntegerArrayType):
        return (
            None,
            f"column type of nullable integer is not supported in groupby built-in function {func_name}",
        )
    if func_name in {"idxmin", "idxmax"}:
        return get_index_data_arr_types(index_type)[0].dtype, "ok"
    if func_name in {"count", "nunique"}:
        return types.int64, "ok"
    else:
        if isinstance(arr_type, IntegerArrayType):
            return IntDtype(in_dtype), "ok"
        elif is_list_string:
            return arr_type, "ok"
        return in_dtype, "ok"  # default: return same dtype as input


def get_pivot_output_dtype(arr_type, func_name, index_type=None):
    """
    Return output dtype for groupby aggregation function based on the
    function and the input array type and dtype.
    If the operation is not feasible (e.g. summing dates) then an error message
    is passed upward to be decided according to the context.
    """
    in_dtype = arr_type.dtype
    if func_name in {"count"}:
        return IntDtype(types.int64)
    if func_name in {"sum", "prod", "min", "max"}:
        if func_name in {"sum", "prod"} and not isinstance(
            in_dtype, (types.Integer, types.Float)
        ):
            raise BodoError(
                "pivot_table(): sum and prod operations require integer or float input"
            )
        if isinstance(in_dtype, types.Integer):
            return IntDtype(in_dtype)
        return in_dtype
    if func_name in {"mean", "var", "std"}:
        return types.float64
    raise BodoError("invalid pivot operation")


def check_args_kwargs(func_name, len_args, args, kws):
    """ Check for extra incorrect arguments """
    if len(kws) > 0:
        bad_key = list(kws.keys())[0]
        raise BodoError(
            f"Groupby.{func_name}() got an unexpected keyword argument '{bad_key}'."
        )
    elif len(args) > len_args:
        raise BodoError(
            f"Groupby.{func_name}() takes {len_args+1} positional argument but {len(args)} were given."
        )


class ColumnType(Enum):
    KeyColumn = 0
    NumericalColumn = 1
    NonNumericalColumn = 2


@infer_getattr
class DataframeGroupByAttribute(AttributeTemplate):
    key = DataFrameGroupByType

    def _get_keys_not_as_index(
        self, grp, out_columns, out_data, out_column_type, multi_level_names=False
    ):
        """Add groupby keys to output columns (to be used when
        as_index=False)"""
        for k in grp.keys:
            if multi_level_names:
                e_col = (k, "")
            else:
                e_col = k
            ind = grp.df_type.columns.index(k)
            data = grp.df_type.data[ind]
            out_columns.append(e_col)
            out_data.append(data)
            out_column_type.append(ColumnType.KeyColumn.value)

    def _get_agg_typ(self, grp, args, func_name, func=None, kws=None):
        """Get output signature for a groupby function"""
        index = RangeIndexType(types.none)
        out_data = []
        out_columns = []
        out_column_type = []
        if not grp.as_index:
            self._get_keys_not_as_index(grp, out_columns, out_data, out_column_type)
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
                # Bodo Limitation. Categorical Key not yet supported.
                # TODO [BE-403]: Support Categorical Key
                if isinstance(grp.df_type.data[ind], bodo.CategoricalArrayType):
                    raise BodoError("Groupby with Categorical key not supported.")
                index = bodo.hiframes.pd_index_ext.array_type_to_index(
                    grp.df_type.data[ind], types.StringLiteral(grp.keys[0])
                )

        # size always produces one integer output and doesn't depend on any input
        if func_name == "size":
            out_data.append(types.Array(types.int64, 1, "C"))
            out_columns.append("size")

        # get output type for each selected column
        list_err_msg = []
        for c in grp.selection:
            if func_name in ("size", "count"):
                kws = dict(kws) if kws else {}
                check_args_kwargs(func_name, 0, args, kws)

            if func_name == "size":
                break
            ind = grp.df_type.columns.index(c)
            data = grp.df_type.data[ind]
            e_column_type = ColumnType.NonNumericalColumn.value
            if isinstance(data, (types.Array, IntegerArrayType)) and isinstance(
                data.dtype, (types.Integer, types.Float)
            ):
                e_column_type = ColumnType.NumericalColumn.value

            if func_name == "agg":
                try:
                    # input to UDFs is a Series
                    in_series_typ = SeriesType(data.dtype, data, None, string_type)
                    out_dtype = get_const_func_output_type(
                        func, (in_series_typ,), {}, self.context
                    )
                    err_msg = "ok"
                except:
                    raise_bodo_error(
                        "Groupy.agg()/Groupy.aggregate(): column {col} of type {type} "
                        "is unsupported/not a valid input type for user defined function".format(
                            col=c, type=data.dtype
                        )
                    )
            else:
                if func_name in ("first", "last", "min", "max"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws
                    # or from args or assign default values
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    numeric_only = (
                        args[0] if len(args) > 0 else kws.pop("numeric_only", False)
                    )
                    min_count = args[1] if len(args) > 1 else kws.pop("min_count", -1)
                    unsupported_args = dict(
                        numeric_only=numeric_only, min_count=min_count
                    )
                    arg_defaults = dict(numeric_only=False, min_count=-1)
                    check_unsupported_args(
                        f"Groupby.{func_name}", unsupported_args, arg_defaults
                    )

                elif func_name in ("sum", "prod"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args if any
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    numeric_only = (
                        args[0] if len(args) > 0 else kws.pop("numeric_only", True)
                    )
                    min_count = args[1] if len(args) > 1 else kws.pop("min_count", 0)
                    unsupported_args = dict(
                        numeric_only=numeric_only, min_count=min_count
                    )
                    arg_defaults = dict(numeric_only=True, min_count=0)
                    check_unsupported_args(
                        f"Groupby.{func_name}", unsupported_args, arg_defaults
                    )
                elif func_name in ("mean", "median"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    numeric_only = (
                        args[0] if len(args) > 0 else kws.pop("numeric_only", True)
                    )
                    unsupported_args = dict(numeric_only=numeric_only)
                    arg_defaults = dict(numeric_only=True)
                    check_unsupported_args(
                        f"Groupby.{func_name}", unsupported_args, arg_defaults
                    )
                elif func_name in ("idxmin", "idxmax"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    axis = args[0] if len(args) > 0 else kws.pop("axis", 0)
                    skipna = args[1] if len(args) > 1 else kws.pop("skipna", True)
                    unsupported_args = dict(axis=axis, skipna=skipna)
                    arg_defaults = dict(axis=0, skipna=True)
                    check_unsupported_args(
                        f"Groupby.{func_name}", unsupported_args, arg_defaults
                    )
                elif func_name in ("var", "std"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    ddof = args[0] if len(args) > 0 else kws.pop("ddof", 1)
                    unsupported_args = dict(ddof=ddof)
                    arg_defaults = dict(ddof=1)
                    check_unsupported_args(
                        f"Groupby.{func_name}", unsupported_args, arg_defaults
                    )
                elif func_name == "nunique":
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    dropna = args[0] if len(args) > 0 else kws.pop("dropna", 1)
                    if len(kws) > 0:
                        bad_key = list(kws.keys())[0]
                        raise BodoError(
                            f"Groupby.{func_name} got an unexpected keyword argument '{bad_key}'"
                        )
                out_dtype, err_msg = get_groupby_output_dtype(
                    data, func_name, grp.df_type.index
                )

            if err_msg == "ok":
                if out_dtype != ArrayItemArrayType(string_array_type):
                    out_arr = dtype_to_array_type(out_dtype)
                else:
                    out_arr = out_dtype
                out_data.append(out_arr)
                out_columns.append(c)
                out_column_type.append(e_column_type)
            else:
                list_err_msg.append(err_msg)

        if func_name == "sum":
            has_numeric = any(
                [x == ColumnType.NumericalColumn.value for x in out_column_type]
            )
            if has_numeric:
                out_data = [
                    x
                    for x, y in zip(out_data, out_column_type)
                    if y != ColumnType.NonNumericalColumn.value
                ]
                out_columns = [
                    x
                    for x, y in zip(out_columns, out_column_type)
                    if y != ColumnType.NonNumericalColumn.value
                ]
        nb_drop = len(list_err_msg)
        if len(out_data) == 0:
            if nb_drop == 0:
                raise BodoError("No columns in output.")
            else:
                raise BodoError(
                    "No columns in output. {} column{} dropped for following reasons: {}".format(
                        nb_drop,
                        " was" if nb_drop == 1 else "s were",
                        ",".join(list_err_msg),
                    )
                )

        out_res = DataFrameType(tuple(out_data), index, tuple(out_columns))
        # XXX output becomes series if single output and explicitly selected
        if (len(grp.selection) == 1 and grp.explicit_select and grp.as_index) or (
            func_name == "size" and grp.as_index
        ):
            if isinstance(out_data[0], IntegerArrayType):
                dtype = IntDtype(out_data[0].dtype)
            else:
                dtype = out_data[0].dtype
            name_type = (
                types.none
                if func_name == "size"
                else types.StringLiteral(grp.selection[0])
            )
            out_res = SeriesType(dtype, index=index, name_typ=name_type)
        return signature(out_res, *args)

    def _get_agg_funcname_and_outtyp(self, grp, col, f_val):
        """Get function name and output type for a function used in
        groupby.agg(), given by f_val (can be a string constant or
        user-defined function) applied to column col"""
        is_udf = True  # is user-defined function
        if isinstance(f_val, str):
            is_udf = False
            f_name = f_val
        elif is_overload_constant_str(f_val):
            is_udf = False
            f_name = get_overload_const_str(f_val)
        if not is_udf:
            if f_name not in bodo.ir.aggregate.supported_agg_funcs[:-1]:
                raise BodoError(f"unsupported aggregate function {f_name}")
            # run typer on a groupby with just column col
            ret_grp = DataFrameGroupByType(
                grp.df_type, grp.keys, (col,), grp.as_index, True
            )
            out_tp = self._get_agg_typ(ret_grp, (), f_name).return_type
        else:
            # assume udf
            if is_expr(f_val, "make_function"):
                f = types.functions.MakeFunctionLiteral(f_val)
            else:
                f = f_val
            validate_udf("agg", f)
            func = get_overload_const_func(f)
            code = func.code if hasattr(func, "code") else func.__code__
            f_name = code.co_name
            # run typer on a groupby with just column col
            ret_grp = DataFrameGroupByType(
                grp.df_type, grp.keys, (col,), grp.as_index, True
            )
            out_tp = self._get_agg_typ(ret_grp, (), "agg", f).return_type
        return f_name, out_tp

    def _resolve_agg(self, grp, args, kws):
        """infer groupby output type for agg/aggregate"""
        # NamedAgg case has func=None
        # e.g. df.groupby("A").agg(C=pd.NamedAgg(column="B", aggfunc="sum"))
        func = get_call_expr_arg("agg", args, dict(kws), 0, "func", default=types.none)
        # untyped pass converts NamedAgg to regular tuple (equivalent in Pandas) to
        # enable typing.
        # This check is same as Pandas:
        # https://github.com/pandas-dev/pandas/blob/64027e60eead00d5ccccc5c7cddc9493a186aa95/pandas/core/aggregation.py#L129
        relabeling = kws and all(
            isinstance(v, types.Tuple) and len(v) == 2 for v in kws.values()
        )

        if is_overload_none(func) and not relabeling:
            raise_bodo_error("Groupby.agg()/aggregate(): Must provide 'func'")
        if len(args) > 1 or (kws and not relabeling):
            raise_bodo_error(
                "Groupby.agg()/aggregate(): passing extra arguments to functions not supported yet."
            )
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
        if relabeling or is_overload_constant_dict(func):
            # get mapping of column names to functions:
            # string -> string or tuple of strings (tuple when multiple
            # functions are applied to a column)
            if relabeling:
                # not using a col_map dictionary since input columns could be repeated
                in_col_names = [
                    get_literal_value(in_col) for (in_col, _) in kws.values()
                ]
                f_vals = [get_literal_value(col_func) for (_, col_func) in kws.values()]
            else:
                col_map = get_overload_constant_dict(func)
                in_col_names = tuple(col_map.keys())
                f_vals = tuple(col_map.values())

            # make sure selected columns exist in dataframe
            if any(c not in grp.selection and c not in grp.keys for c in in_col_names):
                raise_const_error(
                    f"Selected column names {in_col_names} not all available in dataframe column names {grp.selection}"
                )

            # if a list/tuple of functions is applied to any column, have to use
            # MultiLevel for every column (even if list/tuple length is one)
            multi_level_names = any(
                isinstance(f_val, (tuple, list)) for f_val in f_vals
            )

            # NamedAgg case in Pandas doesn't support multiple functions
            if relabeling and multi_level_names:
                raise_bodo_error(
                    "Groupby.agg()/aggregate(): cannot pass multiple functions in a single pd.NamedAgg()"
                )

            # get output names and output types
            out_columns = []
            out_data = []
            out_column_type = []
            if not grp.as_index:
                self._get_keys_not_as_index(
                    grp,
                    out_columns,
                    out_data,
                    out_column_type,
                    multi_level_names=multi_level_names,
                )
            for col_name, f_val in zip(in_col_names, f_vals):
                if isinstance(f_val, (tuple, list)):
                    lambda_count = 0
                    for f in f_val:
                        f_name, out_tp = self._get_agg_funcname_and_outtyp(
                            grp, col_name, f
                        )
                        has_cumulative_ops = f_name in list_cumulative
                        if f_name == "<lambda>" and len(f_val) > 1:
                            f_name = "<lambda_" + str(lambda_count) + ">"
                            lambda_count += 1
                        # output column name is 2-level (col_name, func_name)
                        # This happens, for example, with
                        # df.groupby(...).agg({"A": [f1, f2]})
                        out_columns.append((col_name, f_name))
                        _append_out_type(grp, out_data, out_tp)
                else:
                    f_name, out_tp = self._get_agg_funcname_and_outtyp(
                        grp, col_name, f_val
                    )
                    has_cumulative_ops = f_name in list_cumulative
                    if multi_level_names:
                        out_columns.append((col_name, f_name))
                    elif not relabeling:
                        out_columns.append(col_name)
                    _append_out_type(grp, out_data, out_tp)

            # user specifies output names as kws in NamedAgg case
            if relabeling:
                out_columns += list(kws.keys())

            if has_cumulative_ops:
                # result of groupby.cumsum, etc. doesn't have a group index
                # So instead we set from the input index
                index = grp.df_type.index
            else:
                index = out_tp.index

            out_res = DataFrameType(tuple(out_data), index, tuple(out_columns))
            return signature(out_res, *args)

        # multi-function tuple case
        if isinstance(func, types.BaseTuple) and not isinstance(
            func, types.LiteralStrKeyDict
        ):
            if not (len(grp.selection) == 1 and grp.explicit_select):
                raise_bodo_error(
                    "Groupby.agg()/aggregate(): must select exactly one column when more than one functions supplied"
                )
            assert len(func) > 0
            out_data = []
            out_columns = []
            out_column_type = []
            lambda_count = 0
            if not grp.as_index:
                self._get_keys_not_as_index(grp, out_columns, out_data, out_column_type)
            for f_val in func.types:
                f_name, out_tp = self._get_agg_funcname_and_outtyp(
                    grp, grp.selection[0], f_val
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

    @bound_function("groupby.agg", no_unliteral=True)
    def resolve_agg(self, grp, args, kws):
        return self._resolve_agg(grp, args, kws)

    @bound_function("groupby.aggregate", no_unliteral=True)
    def resolve_aggregate(self, grp, args, kws):
        return self._resolve_agg(grp, args, kws)

    @bound_function("groupby.sum", no_unliteral=True)
    def resolve_sum(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "sum", kws=kws)

    @bound_function("groupby.count", no_unliteral=True)
    def resolve_count(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "count", kws=kws)

    @bound_function("groupby.nunique", no_unliteral=True)
    def resolve_nunique(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "nunique", kws=kws)

    @bound_function("groupby.median", no_unliteral=True)
    def resolve_median(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "median", kws=kws)

    @bound_function("groupby.mean", no_unliteral=True)
    def resolve_mean(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "mean", kws=kws)

    @bound_function("groupby.min", no_unliteral=True)
    def resolve_min(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "min", kws=kws)

    @bound_function("groupby.max", no_unliteral=True)
    def resolve_max(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "max", kws=kws)

    @bound_function("groupby.prod", no_unliteral=True)
    def resolve_prod(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "prod", kws=kws)

    @bound_function("groupby.var", no_unliteral=True)
    def resolve_var(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "var", kws=kws)

    @bound_function("groupby.std", no_unliteral=True)
    def resolve_std(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "std", kws=kws)

    @bound_function("groupby.first", no_unliteral=True)
    def resolve_first(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "first", kws=kws)

    @bound_function("groupby.last", no_unliteral=True)
    def resolve_last(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "last", kws=kws)

    @bound_function("groupby.idxmin", no_unliteral=True)
    def resolve_idxmin(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "idxmin", kws=kws)

    @bound_function("groupby.idxmax", no_unliteral=True)
    def resolve_idxmax(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "idxmax", kws=kws)

    @bound_function("groupby.size", no_unliteral=True)
    def resolve_size(self, grp, args, kws):
        return self._get_agg_typ(grp, args, "size", kws=kws)

    def resolve_transformative(self, grp, args, kws, msg, name_operation):
        """For datetime and timedelta datatypes, we can support cummin / cummax,
        but not cumsum / cumprod. Hence the is_minmax entry"""
        index = grp.df_type.index
        out_columns = []
        out_data = []
        if name_operation in list_cumulative:
            kws = dict(kws) if kws else {}
            # pop arguments from kws or args
            # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
            axis = args[0] if len(args) > 0 else kws.pop("axis", 0)
            numeric_only = args[1] if len(args) > 1 else kws.pop("numeric_only", False)
            skipna = args[2] if len(args) > 2 else kws.pop("skipna", 1)
            unsupported_args = dict(axis=axis, numeric_only=numeric_only)
            arg_defaults = dict(axis=0, numeric_only=False)
            check_unsupported_args(
                f"Groupby.{name_operation}", unsupported_args, arg_defaults
            )
            check_args_kwargs(name_operation, 3, args, kws)

        for c in grp.selection:
            out_columns.append(c)
            ind = grp.df_type.columns.index(c)
            data = grp.df_type.data[ind]
            if name_operation == "cumprod":
                if not isinstance(data.dtype, (types.Integer, types.Float)):
                    raise BodoError(msg)
            if name_operation == "cumsum":
                if (
                    data.dtype != types.unicode_type
                    and data != ArrayItemArrayType(string_array_type)
                    and not isinstance(data.dtype, (types.Integer, types.Float))
                ):
                    raise BodoError(msg)
            if name_operation in ("cummin", "cummax"):
                if not isinstance(data.dtype, types.Integer) and not is_dtype_nullable(
                    data.dtype
                ):
                    raise BodoError(msg)

            out_data.append(data)
        if len(out_data) == 0:
            raise BodoError("No columns in output.")
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

    @bound_function("groupby.cumsum", no_unliteral=True)
    def resolve_cumsum(self, grp, args, kws):
        msg = "Groupby.cumsum() only supports columns of types integer, float, string or liststring"
        return self.resolve_transformative(grp, args, kws, msg, "cumsum")

    @bound_function("groupby.cumprod", no_unliteral=True)
    def resolve_cumprod(self, grp, args, kws):
        msg = "Groupby.cumprod() only supports columns of types integer and float"
        return self.resolve_transformative(grp, args, kws, msg, "cumprod")

    @bound_function("groupby.cummin", no_unliteral=True)
    def resolve_cummin(self, grp, args, kws):
        msg = "Groupby.cummin() only supports columns of types integer, float, string, liststring, date, datetime or timedelta"
        return self.resolve_transformative(grp, args, kws, msg, "cummin")

    @bound_function("groupby.cummax", no_unliteral=True)
    def resolve_cummax(self, grp, args, kws):
        msg = "Groupby.cummax() only supports columns of types integer, float, string, liststring, date, datetime or timedelta"
        return self.resolve_transformative(grp, args, kws, msg, "cummax")

    @bound_function("groupby.shift", no_unliteral=True)
    def resolve_shift(self, grp, args, kws):
        msg = "Groupby.shift() doesn't support timedelta"  # HA: for now, until its error is fixed.
        return self.resolve_transformative(grp, args, kws, msg, "shift")

    @bound_function("groupby.pipe", no_unliteral=True)
    def resolve_pipe(self, grp, args, kws):
        return resolve_obj_pipe(self, grp, args, kws, "GroupBy")

    @bound_function("groupby.apply", no_unliteral=True)
    def resolve_apply(self, grp, args, kws):
        kws = dict(kws)
        # pop apply() arguments from kws so only UDF kws remain
        func = args[0] if len(args) > 0 else kws.pop("func", None)
        f_args = tuple(args[1:]) if len(args) > 0 else ()

        f_return_type = _get_groupby_apply_udf_out_type(
            func, grp, f_args, kws, self.context
        )

        # TODO: check output data type to array-compatible scalar, Series or DataFrame

        # whether UDF returns a single row of output
        single_row_output = (
            isinstance(f_return_type, (SeriesType, HeterogeneousSeriesType))
            and f_return_type.const_info is not None
        ) or not isinstance(f_return_type, (SeriesType, DataFrameType))

        # get Index type
        if single_row_output:
            out_data = []
            out_columns = []
            out_column_type = []  # unused
            if not grp.as_index:
                # for as_index=False, index arrays become regular columns
                self._get_keys_not_as_index(grp, out_columns, out_data, out_column_type)
                # group number is assigned to output
                out_index_type = NumericIndexType(types.int64, types.none)
            else:
                if len(grp.keys) > 1:
                    key_col_inds = tuple(
                        grp.df_type.columns.index(grp.keys[i])
                        for i in range(len(grp.keys))
                    )
                    arr_types = tuple(grp.df_type.data[ind] for ind in key_col_inds)
                    out_index_type = MultiIndexType(
                        arr_types, tuple(types.literal(k) for k in grp.keys)
                    )
                else:
                    ind = grp.df_type.columns.index(grp.keys[0])
                    out_index_type = bodo.hiframes.pd_index_ext.array_type_to_index(
                        grp.df_type.data[ind], types.literal(grp.keys[0])
                    )
            out_data = tuple(out_data)
            out_columns = tuple(out_columns)
        else:
            key_arr_types = tuple(
                grp.df_type.data[grp.df_type.columns.index(c)] for c in grp.keys
            )
            index_names = tuple(
                types.literal(v) for v in grp.keys
            ) + get_index_name_types(f_return_type.index)
            if not grp.as_index:
                key_arr_types = (types.Array(types.int64, 1, "C"),)
                index_names = (types.none,) + get_index_name_types(f_return_type.index)
            out_index_type = MultiIndexType(
                key_arr_types + get_index_data_arr_types(f_return_type.index),
                index_names,
            )

        # const Series output returns a DataFrame
        # NOTE: get_const_func_output_type() adds const_info attribute for const Series
        # output
        if single_row_output:
            if isinstance(f_return_type, HeterogeneousSeriesType):
                _, index_vals = f_return_type.const_info
                arrs = tuple(dtype_to_array_type(t) for t in f_return_type.data.types)
                ret_type = DataFrameType(
                    out_data + arrs, out_index_type, out_columns + index_vals
                )
            elif isinstance(f_return_type, SeriesType):
                n_cols, index_vals = f_return_type.const_info
                arrs = tuple(
                    dtype_to_array_type(f_return_type.dtype) for _ in range(n_cols)
                )
                ret_type = DataFrameType(
                    out_data + arrs, out_index_type, out_columns + index_vals
                )
            else:  # scalar case
                data_arr = get_udf_out_arr_type(f_return_type)
                if not grp.as_index:
                    # TODO: Pandas sets NaN for data column
                    ret_type = DataFrameType(
                        out_data + (data_arr,), out_index_type, out_columns + ("",)
                    )
                else:
                    ret_type = SeriesType(
                        data_arr.dtype, data_arr, out_index_type, None
                    )
        elif isinstance(f_return_type, SeriesType):
            ret_type = SeriesType(
                f_return_type.dtype,
                f_return_type.data,
                out_index_type,
                f_return_type.name_typ,
            )
        else:
            ret_type = DataFrameType(
                f_return_type.data, out_index_type, f_return_type.columns
            )

        pysig = gen_apply_pysig(len(f_args), kws.keys())
        new_args = (func, *f_args) + tuple(kws.values())
        return signature(ret_type, *new_args).replace(pysig=pysig)

    def generic_resolve(self, grpby, attr):
        if attr in groupby_unsupported or attr in ("rolling", "value_counts"):
            return
        if attr not in grpby.df_type.columns:
            raise_const_error(
                "groupby: invalid attribute {} (column not found in dataframe or unsupported function)".format(
                    attr
                )
            )
        return DataFrameGroupByType(
            grpby.df_type, grpby.keys, (attr,), grpby.as_index, True
        )


def _get_groupby_apply_udf_out_type(func, grp, f_args, kws, context):
    """get output type for UDF used in groupby apply()"""

    # NOTE: without explicit column selection, Pandas passes key columns also for
    # some reason (as of Pandas 1.1.5)
    in_df_type = grp.df_type
    if grp.explicit_select:
        # input to UDF is a Series if only one column is explicitly selected
        if len(grp.selection) == 1:
            col_name = grp.selection[0]
            data_arr = in_df_type.data[in_df_type.columns.index(col_name)]
            in_data_type = SeriesType(
                data_arr.dtype, data_arr, in_df_type.index, types.literal(col_name)
            )
        else:
            in_data = tuple(
                in_df_type.data[in_df_type.columns.index(c)] for c in grp.selection
            )
            in_data_type = DataFrameType(
                in_data, in_df_type.index, tuple(grp.selection)
            )
    else:
        in_data_type = in_df_type

    arg_typs = (in_data_type,)
    arg_typs += tuple(f_args)
    try:
        f_return_type = get_const_func_output_type(func, arg_typs, kws, context)
    except Exception as e:
        raise_bodo_error(
            get_udf_error_msg("GroupBy.apply()", e), getattr(e, "loc", None)
        )
    return f_return_type


def resolve_obj_pipe(self, grp, args, kws, obj_name):
    """handle groupyby/dataframe/series.pipe in low-level API since it requires
    **kwargs which is not supported in overloads yet.
    Transform: grp.pipe(f, args) -> f(grp, args)
    """
    kws = dict(kws)
    func = args[0] if len(args) > 0 else kws.pop("func", None)
    f_args = tuple(args[1:]) if len(args) > 0 else ()

    arg_typs = (grp,) + f_args
    try:
        f_return_type = get_const_func_output_type(
            func, arg_typs, kws, self.context, False
        )
    except Exception as e:
        raise_bodo_error(
            get_udf_error_msg(f"{obj_name}.pipe()", e), getattr(e, "loc", None)
        )

    pysig = gen_apply_pysig(len(f_args), kws.keys())
    new_args = (func, *f_args) + tuple(kws.values())
    return signature(f_return_type, *new_args).replace(pysig=pysig)


def gen_apply_pysig(n_args, kws):
    """generate pysignature object for apply/pipe"""
    arg_names = ", ".join(f"arg{i}" for i in range(n_args))
    arg_names = arg_names + ", " if arg_names else ""
    # add dummy default value for UDF kws to avoid errors
    kw_names = ", ".join(f"{a} = ''" for a in kws)
    func_text = f"def apply_stub(func, {arg_names}{kw_names}):\n"
    func_text += "    pass\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    apply_stub = loc_vars["apply_stub"]

    return numba.core.utils.pysignature(apply_stub)


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
        out_dtype = get_pivot_output_dtype(data, aggfunc.literal_value)
        out_arr_typ = dtype_to_array_type(out_dtype)

        pivot_vals = _pivot_values.meta
        n_vals = len(pivot_vals)

        ind = df.columns.index(index)
        index_typ = bodo.hiframes.pd_index_ext.array_type_to_index(
            df.data[ind], types.StringLiteral(index)
        )

        out_df = DataFrameType((out_arr_typ,) * n_vals, index_typ, tuple(pivot_vals))

        return signature(out_df, *args)


# don't convert literal types to non-literal and rerun the typing template
PivotTyper._no_unliteral = True


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
        index_typ = bodo.hiframes.pd_index_ext.array_type_to_index(
            index.data, types.StringLiteral("index")
        )
        out_df = DataFrameType((out_arr_typ,) * n_vals, index_typ, tuple(pivot_vals))

        return signature(out_df, *args)


# don't convert literal types to non-literal and rerun the typing template
CrossTabTyper._no_unliteral = True


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(crosstab_dummy, types.VarArg(types.Any))
def lower_crosstab_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


def get_group_indices(keys):  # pragma: no cover
    return np.arange(len(keys))


@overload(get_group_indices)
def get_group_indices_overload(keys):
    """get group indices (labels) for a tuple of key arrays."""
    func_text = "def impl(keys):\n"
    # convert arrays to table
    func_text += "    info_list = [{}]\n".format(
        ", ".join(f"array_to_info(keys[{i}])" for i in range(len(keys.types))),
    )
    func_text += "    table = arr_info_list_to_table(info_list)\n"
    func_text += "    group_labels = np.empty(len(keys[0]), np.int64)\n"
    func_text += "    ngroups = get_groupby_labels(table, group_labels.ctypes)\n"
    func_text += "    delete_table_decref_arrays(table)\n"
    func_text += "    return group_labels, ngroups\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "np": np,
            "get_groupby_labels": get_groupby_labels,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "delete_table_decref_arrays": delete_table_decref_arrays,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


@numba.njit(no_cpython_wrapper=True)
def generate_slices(labels, ngroups):  # pragma: no cover
    """same as:
    https://github.com/pandas-dev/pandas/blob/53d1622eebb8fc46e90f131a559d32f42babd858/pandas/_libs/lib.pyx#L845
    """

    n = len(labels)

    starts = np.zeros(ngroups, dtype=np.int64)
    ends = np.zeros(ngroups, dtype=np.int64)

    start = 0
    group_size = 0
    for i in range(n):
        lab = labels[i]
        if lab < 0:
            start += 1
        else:
            group_size += 1
            if i == n - 1 or lab != labels[i + 1]:
                starts[lab] = start
                ends[lab] = start + group_size
                start += group_size
                group_size = 0

    return starts, ends


def shuffle_dataframe(df, keys):  # pragma: no cover
    return df, keys


@overload(shuffle_dataframe)
def overload_shuffle_dataframe(df, keys):
    """shuffle a dataframe using a tuple of key arrays."""
    n_cols = len(df.columns)
    n_keys = len(keys.types)
    data_args = ", ".join("data_{}".format(i) for i in range(n_cols))

    func_text = "def impl(df, keys):\n"
    # create C++ table from input arrays
    for i in range(n_cols):
        func_text += f"  in_arr{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})\n"

    func_text += f"  in_index_arr = bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df))\n"

    func_text += "  info_list = [{}, {}, {}]\n".format(
        ", ".join(f"array_to_info(keys[{i}])" for i in range(n_keys)),
        ", ".join(f"array_to_info(in_arr{i})" for i in range(n_cols)),
        "array_to_info(in_index_arr)",
    )
    func_text += "  table = arr_info_list_to_table(info_list)\n"
    func_text += f"  out_table = shuffle_table(table, {n_keys}, 1)\n"

    # extract arrays from C++ table
    for i in range(n_keys):
        func_text += f"  out_key{i} = info_to_array(info_from_table(out_table, {i}), keys[{i}])\n"

    for i in range(n_cols):
        func_text += f"  out_arr{i} = info_to_array(info_from_table(out_table, {i+n_keys}), in_arr{i})\n"

    func_text += f"  out_arr_index = info_to_array(info_from_table(out_table, {n_keys + n_cols}), in_index_arr)\n"

    func_text += "  shuffle_info = get_shuffle_info(out_table)\n"
    func_text += "  delete_table(out_table)\n"
    func_text += "  delete_table(table)\n"

    out_data = ", ".join(f"out_arr{i}" for i in range(n_cols))
    func_text += "  out_index = bodo.utils.conversion.index_from_array(out_arr_index)\n"
    func_text += f"  out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(({out_data},), out_index, {gen_const_tup(df.columns)})\n"

    func_text += "  return out_df, ({},), shuffle_info\n".format(
        ", ".join(f"out_key{i}" for i in range(n_keys))
    )

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "shuffle_table": shuffle_table,
            "info_from_table": info_from_table,
            "info_to_array": info_to_array,
            "delete_table": delete_table,
            "get_shuffle_info": get_shuffle_info,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


def reverse_shuffle(data, shuffle_info):  # pragma: no cover
    return data


@overload(reverse_shuffle)
def overload_reverse_shuffle(data, shuffle_info):
    """Reverse a previous shuffle of 'data' with 'shuffle_info'"""

    # MultiIndex
    if isinstance(data, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        n_fields = len(data.array_types)
        func_text = "def impl(data, shuffle_info):\n"
        func_text += "  info_list = [{}]\n".format(
            ", ".join(f"array_to_info(data._data[{i}])" for i in range(n_fields)),
        )
        func_text += "  table = arr_info_list_to_table(info_list)\n"
        func_text += "  out_table = reverse_shuffle_table(table, shuffle_info)\n"
        for i in range(n_fields):
            func_text += f"  out_arr{i} = info_to_array(info_from_table(out_table, {i}), data._data[{i}])\n"
        func_text += "  delete_table(out_table)\n"
        func_text += "  delete_table(table)\n"
        func_text += (
            "  return init_multi_index(({},), data._names, data._name)\n".format(
                ", ".join(f"out_arr{i}" for i in range(n_fields))
            )
        )
        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "array_to_info": array_to_info,
                "arr_info_list_to_table": arr_info_list_to_table,
                "reverse_shuffle_table": reverse_shuffle_table,
                "info_from_table": info_from_table,
                "info_to_array": info_to_array,
                "delete_table": delete_table,
                "init_multi_index": bodo.hiframes.pd_multi_index_ext.init_multi_index,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    # Index types
    if bodo.hiframes.pd_index_ext.is_index_type(data):

        def impl_index(data, shuffle_info):  # pragma: no cover
            in_arr = bodo.utils.conversion.index_to_array(data)
            out_arr = reverse_shuffle(in_arr, shuffle_info)
            return bodo.utils.conversion.index_from_array(out_arr)

        return impl_index

    # arrays
    def impl_arr(data, shuffle_info):  # pragma: no cover
        info_list = [array_to_info(data)]
        table = arr_info_list_to_table(info_list)
        out_table = reverse_shuffle_table(table, shuffle_info)
        out_arr = info_to_array(info_from_table(out_table, 0), data)
        delete_table(out_table)
        delete_table(table)
        return out_arr

    return impl_arr


@overload_method(
    DataFrameGroupByType, "value_counts", inline="always", no_unliteral=True
)
def groupby_value_counts(
    grp, normalize=False, sort=True, ascending=False, bins=None, dropna=True
):

    unsupported_args = dict(normalize=normalize, sort=sort, bins=bins, dropna=dropna)
    arg_defaults = dict(normalize=False, sort=True, bins=None, dropna=True)
    check_unsupported_args("Groupby.value_counts", unsupported_args, arg_defaults)

    # Pandas restriction: value_counts work on SeriesGroupBy only so only one column selection is allowed
    if (len(grp.selection) > 1) or (not grp.as_index):
        raise BodoError("'DataFrameGroupBy' object has no attribute 'value_counts'")

    # Series.value_counts set its index name to `None`
    # Here, last index name in MultiIndex will have series name.
    name = grp.selection[0]

    # df.groupby("X")["Y"].value_counts() => df.groupby("X")["Y"].apply(lambda S : S.value_counts())
    func_text = f"def impl(grp, normalize=False, sort=True, ascending=False, bins=None, dropna=True):\n"
    # TODO: [BE-635] Use S.rename_axis
    udf = f"lambda S : S.value_counts(ascending={ascending}, _index_name='{name}')"
    func_text += f"    return grp.apply({udf})\n"
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


groupby_unsupported = {
    "all",
    "any",
    "backfill",
    "bfill",
    "boxplot",
    "corr",
    "corrwith",
    "cumcount",
    "cummax",
    "cov",
    "diff",
    "fillna",
    "hist",
    "idxmin",
    "mad",
    "skew",
    "take",
    "cummin",
    "cumprod",
    "describe",
    "ffill",
    "filter",
    "get_group",
    "head",
    "ngroup",
    "nth",
    "ohlc",
    "pad",
    "pct_change",
    "plot",
    "quantile",
    "rank",
    "resample",
    "sample",
    "sem",
    "tail",
    "transform",
    "tshift",
}


def _install_groupy_unsupported():
    """install an overload that raises BodoError for unsupported methods of GroupBy,
    DataFrameGroupBy, and SeriesGroupBy types
    """

    for fname in groupby_unsupported:
        overload_method(DataFrameGroupByType, fname, no_unliteral=True)(
            create_unsupported_overload(f"DataFrameGroupByType: '{fname}'")
        )


_install_groupy_unsupported()
