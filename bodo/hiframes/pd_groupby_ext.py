"""Support for Pandas Groupby operations"""

import operator
from enum import Enum

import numba
import numpy as np
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.core.registry import CPUDispatcher
from numba.core.typing.templates import (
    AbstractTemplate,
    Signature,
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
    array_from_cpp_table,
    array_to_info,
    delete_table,
    get_groupby_labels,
    get_null_shuffle_info,
    get_shuffle_info,
    reverse_shuffle_table,
    shuffle_table,
)
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.decimal_arr_ext import DECIMAL128_MAX_PRECISION, Decimal128Type
from bodo.libs.float_arr_ext import FloatDtype, FloatingArrayType
from bodo.libs.int_arr_ext import IntDtype, IntegerArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.templates import OverloadedKeyAttributeTemplate
from bodo.utils.transform import get_call_expr_arg, get_const_func_output_type
from bodo.utils.typing import (
    BodoError,
    ColNamesMetaType,
    assert_bodo_error,
    check_unsupported_args,
    dtype_to_array_type,
    get_index_data_arr_types,
    get_index_name_types,
    get_literal_value,
    get_overload_const_bool,
    get_overload_const_func,
    get_overload_const_int,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_const_tuple,
    get_overload_constant_dict,
    get_udf_error_msg,
    get_udf_out_arr_type,
    is_dtype_nullable,
    is_literal_type,
    is_overload_constant_bool,
    is_overload_constant_dict,
    is_overload_constant_int,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_false,
    is_overload_none,
    is_overload_true,
    list_cumulative,
    raise_bodo_error,
    to_nullable_type,
    to_numeric_index_if_range_index,
    to_str_arr_if_dict_array,
)
from bodo.utils.utils import dict_add_multimap, dt_err, is_expr


class DataFrameGroupByType(types.Type):  # TODO: IterableType over groups
    """Temporary type class for DataFrameGroupBy objects before transformation
    to aggregate node.
    """

    def __init__(
        self,
        df_type,
        keys,
        selection,
        as_index,
        dropna=True,
        explicit_select=False,
        series_select=False,
        _num_shuffle_keys=-1,
        _use_sql_rules=False,
    ):
        # TODO [BE-3982]: Ensure full groupby support with tz-aware keys and data.
        self.df_type = df_type
        self.keys = keys
        self.selection = selection
        self.as_index = as_index
        self.dropna = dropna
        self.explicit_select = explicit_select
        self.series_select = series_select
        # How many of the keys to use as the keys to the shuffle. If -1
        # we shuffle on all keys.
        self._num_shuffle_keys = _num_shuffle_keys
        # Should we use SQL or Pandas rules
        self._use_sql_rules = _use_sql_rules

        super().__init__(
            name=f"DataFrameGroupBy({df_type}, {keys}, {selection}, {as_index}, {dropna}, {explicit_select}, {series_select}, {_num_shuffle_keys})"
        )

    def copy(self):
        # XXX is copy necessary?
        return DataFrameGroupByType(
            self.df_type,
            self.keys,
            self.selection,
            self.as_index,
            self.dropna,
            self.explicit_select,
            self.series_select,
            self._num_shuffle_keys,
            self._use_sql_rules,
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


@register_model(DataFrameGroupByType)
class GroupbyModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("obj", fe_type.df_type),
        ]
        super().__init__(dmm, fe_type, members)


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
        raise_bodo_error(f"Groupby.{func_name}: 'func' must be user defined function")


@intrinsic(prefer_literal=True)
def init_groupby(
    typingctx,
    obj_type,
    by_type,
    as_index_type,
    dropna_type,
    _num_shuffle_keys,
    _is_bodosql,
):
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
    else:
        assert False, (
            "Reached unreachable code in init_groupby; there is an validate_groupby_spec"
        )

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

    if is_overload_constant_bool(dropna_type):
        dropna = is_overload_true(dropna_type)
    else:
        dropna = True
    if is_overload_constant_int(_num_shuffle_keys):
        shuffle_keys = get_overload_const_int(_num_shuffle_keys)
    else:  # pragma: no cover
        shuffle_keys = -1
    if is_overload_constant_bool(_is_bodosql):
        _use_sql_rules = get_overload_const_bool(_is_bodosql)
    else:
        _use_sql_rules = False
    groupby_type = DataFrameGroupByType(
        obj_type,
        keys,
        tuple(selection),
        as_index,
        dropna,
        False,
        _num_shuffle_keys=shuffle_keys,
        _use_sql_rules=_use_sql_rules,
    )
    return (
        groupby_type(
            obj_type,
            by_type,
            as_index_type,
            dropna_type,
            _num_shuffle_keys,
            _is_bodosql,
        ),
        codegen,
    )


# dummy lowering for groupby.size since it is used in Series.value_counts()
# groupby.apply is used in groupby.rolling
@lower_builtin("groupby.count", types.VarArg(types.Any))
@lower_builtin("groupby.size", types.VarArg(types.Any))
@lower_builtin("groupby.apply", types.VarArg(types.Any))
# groupby.agg is used in pivot_table
@lower_builtin("groupby.agg", types.VarArg(types.Any))
def lower_groupby_count_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


@infer
class StaticGetItemDataFrameGroupBy(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        grpby, idx = args
        # df.groupby('A')['B', 'C']
        if isinstance(grpby, DataFrameGroupByType):
            series_select = False
            if isinstance(idx, (tuple, list)):
                if len(set(idx).difference(set(grpby.df_type.columns))) > 0:
                    raise_bodo_error(
                        f"groupby: selected column {set(idx).difference(set(grpby.df_type.columns))} not found in dataframe"
                    )
                selection = idx
            else:
                if idx not in grpby.df_type.columns:
                    raise_bodo_error(
                        f"groupby: selected column {idx} not found in dataframe"
                    )
                selection = (idx,)
                series_select = True
            ret_grp = DataFrameGroupByType(
                grpby.df_type,
                grpby.keys,
                selection,
                grpby.as_index,
                grpby.dropna,
                True,
                series_select,
                _num_shuffle_keys=grpby._num_shuffle_keys,
                _use_sql_rules=grpby._use_sql_rules,
            )
            return signature(ret_grp, *args)


@infer_global(operator.getitem)
class GetItemDataFrameGroupBy(AbstractTemplate):
    """typing pass may force getitem index on df groupby value to be constant, but the
    getitem type won't change to 'static_getitem' so needs handling here.
    """

    def generic(self, args, kws):
        grpby, idx = args
        # df.groupby('A')['B', 'C']
        if isinstance(grpby, DataFrameGroupByType) and is_literal_type(idx):
            # just call typer for 'static_getitem'
            ret_grp = StaticGetItemDataFrameGroupBy.generic(
                self, (grpby, get_literal_value(idx)), {}
            ).return_type
            return signature(ret_grp, *args)


GetItemDataFrameGroupBy.prefer_literal = True


# dummy lowering for groupby getitem to avoid errors (e.g. test_series_groupby_arr)
@lower_builtin("static_getitem", DataFrameGroupByType, types.Any)
@lower_builtin(operator.getitem, DataFrameGroupByType, types.Any)
def static_getitem_df_groupby(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


def get_groupby_output_dtype(arr_type, func_name, index_type=None, other_args=None):
    """
    Return output array dtype for groupby aggregation function based on the
    function and the input array type and dtype.
    If the operation is not feasible (e.g. summing dates) then an error message
    is passed upward to be decided according to the context.
    """
    is_list_string = arr_type == ArrayItemArrayType(string_array_type)
    in_dtype = arr_type.dtype
    # Bodo don't support DatetimeTimeDeltaType. use (timedelta64 instead)
    if isinstance(in_dtype, bodo.hiframes.datetime_timedelta_ext.DatetimeTimeDeltaType):
        raise BodoError(
            f"column type of {in_dtype} is not supported in groupby built-in function {func_name}.\n{dt_err}"
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
    elif (
        func_name in {"last", "sum", "prod", "min", "max", "nunique", "head"}
    ) and isinstance(
        arr_type, (ArrayItemArrayType, TupleArrayType)
    ):  # pragma: no cover
        return (
            None,
            f"column type of {arr_type} of {in_dtype} is not supported in groupby built-in function {func_name}",
        )
    elif func_name in {"count"} and isinstance(
        arr_type, TupleArrayType
    ):  # pragma: no cover
        return (
            None,
            f"column type of {arr_type} of {in_dtype} is not supported in groupby built-in function {func_name}",
        )

    elif func_name == "size" or func_name == "grouping":
        return dtype_to_array_type(types.int64), "ok"
    elif func_name == "sum" and isinstance(in_dtype, Decimal128Type):
        # Use maximum precision since sum can produce large output values
        # TODO[BSE-3224] Support changing decimal representation in runtime to handle
        # overflows.
        out_dtype = Decimal128Type(DECIMAL128_MAX_PRECISION, in_dtype.scale)
        return dtype_to_array_type(out_dtype), "ok"
    elif (
        (func_name == "sum")
        and isinstance(in_dtype, types.Integer)
        and (in_dtype.bitwidth <= 64)
    ):
        # Upcast output integer to the 64-bit variant to prevent overflow.
        out_dtype = types.int64 if in_dtype.signed else types.uint64
        if isinstance(arr_type, types.Array):
            # If regular numpy (i.e. non-nullable)
            return dtype_to_array_type(out_dtype), "ok"
        else:
            # Nullable:
            return dtype_to_array_type(out_dtype, convert_nullable=True), "ok"

    # These functions have a dedicated decimal implementation.
    elif func_name in {"median", "percentile_cont"} and isinstance(
        in_dtype, Decimal128Type
    ):
        # MEDIAN / PERCENTILE_CONT
        # For median, the input precision and scale are increased by 3,
        # while precision remains capped at 38.
        # This means that an input scale of 36 or more is invalid.
        # (Based off of Snowflake's MEDIAN)
        new_scale = in_dtype.scale + 3
        new_precision = min(38, in_dtype.precision + 3)
        if new_scale > 38:
            raise BodoError(
                f"Input scale of {in_dtype.precision} too large for MEDIAN operation"
            )
        return bodo.DecimalArrayType(new_precision, new_scale), "ok"
    elif func_name == "percentile_disc" and isinstance(in_dtype, Decimal128Type):
        # PERCENTILE_DISC
        # Maintain the same precision and scale as the input.
        return bodo.DecimalArrayType(in_dtype.precision, in_dtype.scale), "ok"

    elif (
        func_name
        in {
            "median",
            "mean",
            "var_pop",
            "std_pop",
            "var",
            "std",
            "kurtosis",
            "skew",
            "percentile_cont",
            "percentile_disc",
        }
    ) and isinstance(in_dtype, (Decimal128Type, types.Integer, types.Float)):
        # TODO: Only make the output nullable if the input is nullable?
        return to_nullable_type(dtype_to_array_type(types.float64)), "ok"
    elif func_name in {"boolor_agg", "booland_agg", "boolxor_agg"}:
        if isinstance(
            in_dtype, (Decimal128Type, types.Integer, types.Float, types.Boolean)
        ):
            return bodo.boolean_array_type, "ok"
        return (
            None,
            f"For {func_name}, only columns of type integer, float, Decimal, or boolean type are allowed",
        )
    elif func_name == "mode":
        return arr_type, "ok"
    elif func_name in {"bitor_agg", "bitand_agg", "bitxor_agg"}:
        if isinstance(in_dtype, types.Integer):
            return to_nullable_type(dtype_to_array_type(in_dtype)), "ok"
        elif isinstance(in_dtype, (types.Float)) or in_dtype == types.unicode_type:
            return to_nullable_type(dtype_to_array_type(types.int64)), "ok"
        else:
            return (
                None,
                f"For {func_name}, only columns of type integer, float, or strings (that evaluate to numbers) are allowed",
            )
    elif func_name == "count_if":
        if in_dtype == types.boolean:
            return types.Array(types.int64, 1, "C"), "ok"
        return (
            None,
            "For count_if, only boolean columns are allowed",
        )
    elif func_name == "listagg":
        # For listagg, output is always string, even if the input is dict encoded.
        if in_dtype == string_type:
            return string_array_type, "ok"
        return (
            None,
            "For listagg, only string columns are allowed",
        )
    elif func_name in {"array_agg", "array_agg_distinct"}:
        # For array_agg, output is a nested array where the internal arrays' dtypes
        # are the same as the original input.
        return ArrayItemArrayType(arr_type), "ok"
    elif func_name == "object_agg":
        # For object_agg, output is a map array where the keys are the
        # first argument (which must be a string dtype) and the values
        # are the second argument
        key_arr_type = other_args[0]
        if key_arr_type.dtype != string_type:  # pragma: no cover
            return (
                None,
                f"Unsupported array type for {func_name}: {arr_type}",
            )
        return bodo.MapArrayType(key_arr_type, arr_type), "ok"

    if not isinstance(in_dtype, (types.Integer, types.Float, types.Boolean)):
        if (
            is_list_string
            or in_dtype == types.unicode_type
            or arr_type == bodo.binary_array_type
        ):
            if func_name not in {
                "count",
                "nunique",
                "min",
                "max",
                "sum",
                "first",
                "last",
                "head",
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
                "head",
            }:
                return (
                    None,
                    f"column type of {in_dtype} is not supported in groupby built-in function {func_name}",
                )

    if isinstance(in_dtype, types.Boolean) and func_name in {
        "cumsum",
        "mean",
        "sum",
        "std",
        "var",
    }:
        if func_name in {"sum"}:
            return to_nullable_type(dtype_to_array_type(types.int64)), "ok"
        return (
            None,
            f"groupby built-in functions {func_name} does not support boolean column",
        )
    elif func_name in {"idxmin", "idxmax"}:
        return dtype_to_array_type(get_index_data_arr_types(index_type)[0].dtype), "ok"
    elif func_name in {"count", "nunique"}:
        return dtype_to_array_type(types.int64), "ok"
    else:
        # default: return same dtype as input
        return arr_type, "ok"


def check_args_kwargs(func_name, len_args, args, kws):
    """Check for extra incorrect arguments"""
    if len(kws) > 0:
        bad_key = list(kws.keys())[0]
        raise BodoError(
            f"Groupby.{func_name}() got an unexpected keyword argument '{bad_key}'."
        )
    elif len(args) > len_args:
        raise BodoError(
            f"Groupby.{func_name}() takes {len_args + 1} positional argument but {len(args)} were given."
        )


class ColumnType(Enum):
    KeyColumn = 0
    NumericalColumn = 1
    NonNumericalColumn = 2


def get_keys_not_as_index(
    grp, out_columns, out_data, out_column_type, multi_level_names=False
):
    """Add groupby keys to output columns (to be used when as_index=False)"""
    for k in grp.keys:
        if multi_level_names:
            e_col = (k, "")
        else:
            e_col = k
        ind = grp.df_type.column_index[k]
        data = grp.df_type.data[ind]
        out_columns.append(e_col)
        out_data.append(data)
        out_column_type.append(ColumnType.KeyColumn.value)


def get_agg_typ(
    grp,
    args,
    func_name,
    typing_context,
    target_context,
    func=None,
    kws=None,
    raise_on_any_error=False,
):
    """Get output signature for a groupby function"""
    # grp: DataFrameGroupByType instance
    # args: arguments to xxx in df.groupby().xxx()
    # func_name: name of function to apply. "agg" if UDF
    # func: function if func_name=="agg"

    # NOTE: for groupby.agg, where multiple different functions can be
    # applied to different input columns, resolve_agg uses this function
    # as helper and can call it repeatedly by passing a DataFrameGroupByType
    # instance with only one input column
    index = RangeIndexType(types.none)  # groupby output index type
    out_data = []  # type of output columns (array type)
    out_columns = []  # name of output columns
    out_column_type = []  # ColumnType of output columns (see ColumnType Enum above)

    if func_name in ("head", "ngroup"):
        # Per Pandas documentation as_index flag is ignored
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.GroupBy.head.html
        # NOTE: ngroup by testing found this applies. Not mentioned in docs.
        # The closest I could find is that index is set here (https://github.com/pandas-dev/pandas/blob/v1.4.3/pandas/core/groupby/groupby.py#L3041)
        grp.as_index = True
    if not grp.as_index:
        get_keys_not_as_index(grp, out_columns, out_data, out_column_type)
    elif func_name in ("head", "ngroup"):
        # TODO: clarify this case with examples
        # Regardless of number of keys, index is always NumericIndex
        # unless it's explicitly assigned
        if grp.df_type.index == index:
            index = NumericIndexType(types.int64, types.none)
        else:
            index = grp.df_type.index
    else:
        if len(grp.keys) > 1:
            key_col_inds = tuple(
                grp.df_type.column_index[grp.keys[i]] for i in range(len(grp.keys))
            )
            arr_types = tuple(grp.df_type.data[ind] for ind in key_col_inds)
            index = MultiIndexType(
                arr_types, tuple(types.StringLiteral(k) for k in grp.keys)
            )
        else:
            ind = grp.df_type.column_index[grp.keys[0]]
            ind_arr_t = grp.df_type.data[ind]
            index = bodo.hiframes.pd_index_ext.array_type_to_index(
                ind_arr_t, types.StringLiteral(grp.keys[0])
            )

    # gb_info maps (in_cols, additional_args, func_name) -> out_col
    # where in_cols is a tuple of input column names
    gb_info = {}
    list_err_msg = []
    if func_name in ("size", "count"):
        kws = dict(kws) if kws else {}
        check_args_kwargs(func_name, 0, args, kws)
    if func_name == "size":
        # size always produces one integer output and doesn't depend on any input
        out_data.append(types.Array(types.int64, 1, "C"))
        out_columns.append("size")
        dict_add_multimap(gb_info, ((), (), "size"), "size")
    elif func_name == "ngroup":
        # ngroup always produces one integer output and doesn't depend on any input
        out_data.append(types.Array(types.int64, 1, "C"))
        out_columns.append("ngroup")
        dict_add_multimap(gb_info, ((), (), "ngroup"), "ngroup")
        # arguments passed to ngroup(ascending=True)
        kws = dict(kws) if kws else {}
        ascending = args[0] if len(args) > 0 else kws.pop("ascending", True)
        unsupported_args = {"ascending": ascending}
        arg_defaults = {"ascending": True}
        check_unsupported_args(
            f"Groupby.{func_name}",
            unsupported_args,
            arg_defaults,
            package_name="pandas",
            module_name="GroupBy",
        )
        check_args_kwargs(func_name, 1, args, kws)

    else:
        # gb.head() w/o explicit select, has all columns in output (including keys)
        columns = (
            grp.selection
            if func_name != "head" or grp.explicit_select
            else grp.df_type.columns
        )
        # get output type for each selected column
        for c in columns:
            ind = grp.df_type.column_index[c]
            data = grp.df_type.data[ind]  # type of input column
            # for sum and cumsum we opted to return regular string since
            # repeated strings are unlikely and this simplifies code on C++ side
            if func_name in ("sum", "cumsum"):
                data = to_str_arr_if_dict_array(data)
            e_column_type = ColumnType.NonNumericalColumn.value
            if isinstance(
                data, (types.Array, IntegerArrayType, FloatingArrayType)
            ) and isinstance(data.dtype, (types.Integer, types.Float)):
                e_column_type = ColumnType.NumericalColumn.value

            if func_name == "agg":
                try:
                    # input to UDFs is a Series
                    in_series_typ = SeriesType(data.dtype, data, None, string_type)
                    out_dtype = get_const_func_output_type(
                        func, (in_series_typ,), {}, typing_context, target_context
                    )
                    # Is this check still necessary or should we always wrap
                    # the result in an array because its a UDF?
                    if out_dtype != ArrayItemArrayType(string_array_type):
                        out_dtype = dtype_to_array_type(out_dtype)
                    err_msg = "ok"
                except Exception:
                    raise_bodo_error(
                        f"Groupy.agg()/Groupy.aggregate(): column {c} of type {data.dtype} "
                        "is unsupported/not a valid input type for user defined function"
                    )
            else:
                other_args = None
                if func_name in ("first", "last", "min", "max"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws
                    # or from args or assign default values
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    numeric_only = (
                        args[0] if len(args) > 0 else kws.pop("numeric_only", False)
                    )
                    min_count = args[1] if len(args) > 1 else kws.pop("min_count", -1)
                    unsupported_args = {
                        "numeric_only": numeric_only,
                        "min_count": min_count,
                    }
                    arg_defaults = {"numeric_only": False, "min_count": -1}
                    check_unsupported_args(
                        f"Groupby.{func_name}",
                        unsupported_args,
                        arg_defaults,
                        package_name="pandas",
                        module_name="GroupBy",
                    )

                elif func_name in ("sum", "prod"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args if any
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    numeric_only = (
                        args[0] if len(args) > 0 else kws.pop("numeric_only", True)
                    )
                    min_count = args[1] if len(args) > 1 else kws.pop("min_count", 0)
                    unsupported_args = {
                        "numeric_only": numeric_only,
                        "min_count": min_count,
                    }
                    arg_defaults = {"numeric_only": True, "min_count": 0}
                    check_unsupported_args(
                        f"Groupby.{func_name}",
                        unsupported_args,
                        arg_defaults,
                        package_name="pandas",
                        module_name="GroupBy",
                    )
                elif func_name in ("mean", "median"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    numeric_only = (
                        args[0] if len(args) > 0 else kws.pop("numeric_only", True)
                    )
                    unsupported_args = {"numeric_only": numeric_only}
                    arg_defaults = {"numeric_only": True}
                    check_unsupported_args(
                        f"Groupby.{func_name}",
                        unsupported_args,
                        arg_defaults,
                        package_name="pandas",
                        module_name="GroupBy",
                    )
                elif func_name in ("idxmin", "idxmax"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    axis = args[0] if len(args) > 0 else kws.pop("axis", 0)
                    skipna = args[1] if len(args) > 1 else kws.pop("skipna", True)
                    unsupported_args = {"axis": axis, "skipna": skipna}
                    arg_defaults = {"axis": 0, "skipna": True}
                    check_unsupported_args(
                        f"Groupby.{func_name}",
                        unsupported_args,
                        arg_defaults,
                        package_name="pandas",
                        module_name="GroupBy",
                    )
                elif func_name in ("var_pop", "std_pop", "var", "std"):
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    ddof = args[0] if len(args) > 0 else kws.pop("ddof", 1)
                    unsupported_args = {"ddof": ddof}
                    arg_defaults = {"ddof": 1}
                    check_unsupported_args(
                        f"Groupby.{func_name}",
                        unsupported_args,
                        arg_defaults,
                        package_name="pandas",
                        module_name="GroupBy",
                    )
                elif func_name == "nunique":
                    kws = dict(kws) if kws else {}
                    # pop arguments from kws or args
                    # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
                    if len(args) == 0:
                        kws.pop("dropna", None)
                    check_args_kwargs(func_name, 1, args, kws)
                elif func_name == "head":
                    # pop arguments from kws or args
                    if len(args) == 0:
                        kws.pop("n", None)
                elif func_name == "object_agg":
                    key_col = args[0]
                    key_col_idx = grp.df_type.column_index[key_col]
                    key_col_type = grp.df_type.data[key_col_idx]
                    other_args = (key_col_type,)
                out_dtype, err_msg = get_groupby_output_dtype(
                    data, func_name, grp.df_type.index, other_args
                )
            if err_msg == "ok":
                out_dtype = (
                    to_str_arr_if_dict_array(out_dtype)
                    if func_name in ("sum", "cumsum")
                    else out_dtype
                )
                out_data.append(out_dtype)
                out_columns.append(c)
                if func_name == "agg":
                    # XXX Can we merge with get_const_func_output_type above?
                    udf_name = bodo.ir.aggregate._get_udf_name(
                        bodo.ir.aggregate._get_const_agg_func(func, None)
                    )
                    dict_add_multimap(gb_info, ((c,), (), udf_name), c)
                else:
                    dict_add_multimap(gb_info, ((c,), (), func_name), c)
                out_column_type.append(e_column_type)
            else:
                if raise_on_any_error:
                    raise BodoError(
                        f"Groupby with function {func_name} not supported. Error message: {err_msg}"
                    )
                else:
                    list_err_msg.append(err_msg)

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

    out_res = DataFrameType(
        tuple(out_data), index, tuple(out_columns), is_table_format=True
    )
    # XXX output becomes series if single output and explicitly selected
    # or size with as_index=True
    # or ngroup
    if (
        (len(grp.selection) == 1 and grp.series_select and grp.as_index)
        or (func_name == "size" and grp.as_index)
        or (func_name == "ngroup")
    ):
        if isinstance(out_data[0], IntegerArrayType):
            dtype = IntDtype(out_data[0].dtype)
        elif isinstance(out_data[0], FloatingArrayType):
            dtype = FloatDtype(out_data[0].dtype)
        else:
            dtype = out_data[0].dtype
        name_type = (
            types.none
            if func_name in ("size", "ngroup")
            else types.StringLiteral(grp.selection[0])
        )
        out_res = SeriesType(dtype, data=out_data[0], index=index, name_typ=name_type)
    return signature(out_res, *args), gb_info


def get_agg_name_for_numpy_method(method_name):
    """Takes in the name of a numpy method that is supported for groupby
    aggregation (e.g. var, std) and returns the corresponding name used
    to describe it in the groupby aggregation internals (e.g. "var_pop",
    "std_pop")"""
    method_to_agg_names = {
        "var": "var_pop",
        "std": "std_pop",
    }
    if method_name not in method_to_agg_names:
        raise BodoError(
            f"unsupported numpy method for use as an aggregate function np.{method_name}"
        )
    return method_to_agg_names[method_name]


def get_agg_funcname_and_outtyp(
    grp, col, f_val, additional_args, typing_context, target_context, raise_on_any_error
):
    """
    Get function name and output type for a function used in
    groupby.agg(), given by f_val (can be a string constant or
    user-defined function) applied to column col

    Additional_args is a tuple of additional arguments to the aggregation call.
    Empty tuple for aggregation functions which do not require
    additional arguments.
    """

    is_udf = True  # is user-defined function
    if isinstance(f_val, str):
        is_udf = False
        f_name = f_val
    elif is_overload_constant_str(f_val):
        is_udf = False
        f_name = get_overload_const_str(f_val)
    elif bodo.utils.typing.is_builtin_function(f_val):
        # Builtin functions like
        is_udf = False
        f_name = bodo.utils.typing.get_builtin_function_name(f_val)
    elif bodo.utils.typing.is_numpy_function(f_val):
        is_udf = False
        method_name = bodo.utils.typing.get_builtin_function_name(f_val)
        f_name = get_agg_name_for_numpy_method(method_name)

    if not is_udf:
        if f_name not in bodo.ir.aggregate.supported_agg_funcs[:-1]:
            raise BodoError(f"unsupported aggregate function {f_name}")

        if (
            f_name not in bodo.ir.aggregate.supported_extended_agg_funcs
            and len(additional_args) != 0
        ):
            raise BodoError(
                f"Internal error: aggregate function {f_name} does not support additional arguments and should not be used with bodo.utils.utils.ExtendedNamedAgg"
            )

        # run typer on a groupby with just column col
        ret_grp = DataFrameGroupByType(
            grp.df_type,
            grp.keys,
            (col,),
            grp.as_index,
            grp.dropna,
            True,
            True,
            _num_shuffle_keys=grp._num_shuffle_keys,
            _use_sql_rules=grp._use_sql_rules,
        )
        out_tp = get_agg_typ(
            ret_grp,
            additional_args,
            f_name,
            typing_context,
            target_context,
            raise_on_any_error=raise_on_any_error,
        )[0].return_type
    else:
        # assume udf
        if is_expr(f_val, "make_function"):
            f = types.functions.MakeFunctionLiteral(f_val)
        else:
            f = f_val
        validate_udf("agg", f)
        func = get_overload_const_func(f, None)
        code = func.code if hasattr(func, "code") else func.__code__
        f_name = code.co_name
        # run typer on a groupby with just column col
        ret_grp = DataFrameGroupByType(
            grp.df_type,
            grp.keys,
            (col,),
            grp.as_index,
            grp.dropna,
            True,
            True,
            _num_shuffle_keys=grp._num_shuffle_keys,
            _use_sql_rules=grp._use_sql_rules,
        )
        # out_tp is series because we are passing only one input column
        out_tp = get_agg_typ(
            ret_grp,
            additional_args,
            "agg",
            typing_context,
            target_context,
            f,
            raise_on_any_error=raise_on_any_error,
        )[0].return_type
    return f_name, out_tp


def handle_extended_named_agg_input_cols(
    data_col_name, f_name, args
):  # pragma: no cover
    assert len(args) == 3, (
        "Internal error in handle_extended_named_agg_input_cols: args length does not equal 3"
    )
    assert_bodo_error(
        is_literal_type(args[0]),
        "Internal error in handle_extended_named_agg_input_cols: data column name is not a literal value",
    )
    assert get_literal_value(args[0]) == data_col_name, (
        f"Internal error in handle_extended_named_agg_input_cols: data column name mismatch: {data_col_name} and {get_literal_value(args[0])}"
    )

    additional_args = args[2]
    additional_args_made_literal = get_overload_const_tuple(additional_args)

    if f_name not in bodo.ir.aggregate.supported_extended_agg_funcs:
        raise RuntimeError(
            f"Internal error: aggregate function {f_name} does not support additional arguments and should not be used with bodo.utils.utils.ExtendedNamedAgg"
        )

    if f_name == "listagg":
        assert get_literal_value(args[1]) == "listagg", (
            "Internal error in resolve_listagg_func_inputs: Called on not listagg function."
        )
        return resolve_listagg_func_inputs(data_col_name, additional_args_made_literal)
    if f_name in {"array_agg", "array_agg_distinct"}:
        assert get_literal_value(args[1]) == f_name, (
            "Internal error in resolve_array_agg_func_inputs: Called on not array_agg function."
        )
        return resolve_array_agg_func_inputs(
            data_col_name, additional_args_made_literal
        )
    if f_name in {"percentile_cont", "percentile_disc"}:
        assert get_literal_value(args[1]) == f_name, (
            f"Internal error in resolve_listagg_func_inputs: Called on not {f_name} function."
        )
        percentile = additional_args_made_literal[0]
        if not (isinstance(percentile, str)):
            raise_bodo_error(
                f"Groupby.{f_name}: 'percentile' should be a string of a single column name."
            )
        return (data_col_name, additional_args_made_literal[0]), ()
    if f_name == "object_agg":
        assert get_literal_value(args[1]) == f_name, (
            "Internal error in resolve_array_agg_func_inputs: Called on not object_agg function."
        )
        key_col_name = additional_args_made_literal[0]
        if not (isinstance(key_col_name, str)):
            raise_bodo_error(
                f"Groupby.{f_name}: 'key_col_name' should be a string of a single column name."
            )
        return (key_col_name, data_col_name), ()

    raise RuntimeError(
        f"Internal error in handle_extended_named_agg_input_cols: Unsupported function name: {f_name}"
    )


def resolve_listagg_func_inputs(data_col_name, additional_args) -> tuple:
    sep = additional_args[0]
    order_by = additional_args[1]
    ascending = additional_args[2]
    na_position = additional_args[3]

    if not (isinstance(sep, str)):
        raise_bodo_error(
            "Groupby.listagg: 'sep' string  of a single column name if provided."
        )
    if not (
        isinstance(order_by, tuple)
        and all(isinstance(col_name, str) for col_name in order_by)
    ):
        raise_bodo_error(
            "Groupby.listagg: 'order_by' argument must be a tuple of column names if provided."
        )
    if not (
        isinstance(ascending, tuple) and all(isinstance(val, bool) for val in ascending)
    ):
        raise_bodo_error(
            "Groupby.listagg: 'ascending' argument must be a tuple of booleans if provided."
        )

    if not (
        isinstance(na_position, tuple)
        and all(isinstance(val, str) for val in na_position)
    ):
        raise_bodo_error(
            "Groupby.listagg: 'na_position' argument must be a tuple of 'first' or 'last' if provided."
        )

    if len(order_by) != len(ascending) or len(ascending) != len(na_position):
        raise_bodo_error(
            "Groupby.listagg: 'order_by', 'ascending', and 'na_position' arguments must have the same length."
        )

    # orderby is the only columnar input that should be included in the input columns
    input_cols = (data_col_name,) + (sep,) + order_by
    additional_args = (ascending, na_position)
    return input_cols, additional_args


def resolve_array_agg_func_inputs(data_col_name, additional_args) -> tuple:
    order_by = additional_args[0]
    ascending = additional_args[1]
    na_position = additional_args[2]

    if not (
        isinstance(order_by, tuple)
        and all(isinstance(col_name, str) for col_name in order_by)
    ):  # pragma: no cover
        raise_bodo_error(
            "Groupby.array_agg: 'order_by' argument must be a tuple of column names if provided."
        )
    if not (
        isinstance(ascending, tuple) and all(isinstance(val, bool) for val in ascending)
    ):  # pragma: no cover
        raise_bodo_error(
            "Groupby.array_agg: 'ascending' argument must be a tuple of booleans if provided."
        )

    if not (
        isinstance(na_position, tuple)
        and all(isinstance(val, str) for val in na_position)
    ):  # pragma: no cover
        raise_bodo_error(
            "Groupby.array_agg: 'na_position' argument must be a tuple of 'first' or 'last' if provided."
        )

    if len(order_by) != len(ascending) or len(ascending) != len(
        na_position
    ):  # pragma: no cover
        raise_bodo_error(
            "Groupby.array_agg: 'order_by', 'ascending', and 'na_position' arguments must have the same length."
        )

    # orderby is the only columnar input that should be included in the input columns
    input_cols = (data_col_name,) + order_by
    additional_args = (ascending, na_position)
    return input_cols, additional_args


def resolve_named_agg_literals(kws):
    in_col_names = []
    f_vals = []
    additional_args = []

    for out_tuple in kws.values():
        in_col_names.append(get_literal_value(out_tuple[0]))
        f_vals.append(get_literal_value(out_tuple[1]))
        if len(out_tuple) == 2:
            additional_args.append(())
        else:
            additional_args.append(get_literal_value(out_tuple[2]))

    return in_col_names, f_vals, additional_args


def resolve_agg(grp, args, kws, typing_context, target_context):
    """infer groupby output type for agg/aggregate/extended agg"""
    # NamedAgg case has func=None
    # e.g. df.groupby("A").agg(C=pd.NamedAgg(column="B", aggfunc="sum"))
    func = get_call_expr_arg("agg", args, dict(kws), 0, "func", default=types.none)
    # untyped pass converts NamedAgg to regular tuple (equivalent in Pandas) to
    # enable typing.
    # This check is same as Pandas:
    # https://github.com/pandas-dev/pandas/blob/64027e60eead00d5ccccc5c7cddc9493a186aa95/pandas/core/aggregation.py#L129
    relabeling = kws and all(
        isinstance(v, types.Tuple) and len(v) in (2, 3) for v in kws.values()
    )
    # Should we raise an immediate exception when a particular column cannot be
    # used in an aggregation. If a column is explicitly selected then yes.
    raise_on_any_error = relabeling

    if is_overload_none(func) and not relabeling:
        raise_bodo_error("Groupby.agg()/aggregate(): Must provide 'func'")
    if len(args) > 1 or (kws and not relabeling):
        raise_bodo_error(
            "Groupby.agg()/aggregate(): passing extra arguments to functions not supported yet."
        )
    has_cumulative_ops = False

    def _append_out_type(grp, out_data, out_tp):
        if grp.as_index is False:
            # get_agg_typ also returns the index (keys) as part of
            # out_tp, but we already added them at the beginning
            # (by calling get_keys_not_as_index), so we skip them
            out_data.append(out_tp.data[len(grp.keys)])
        else:
            # out_tp is assumed to be a SeriesType (see get_agg_typ)
            out_data.append(out_tp.data)

    # multi-function constant dictionary case
    if relabeling or is_overload_constant_dict(func):
        # get mapping of column names to functions:
        # string -> string or tuple of strings (tuple when multiple
        # functions are applied to a column)
        if relabeling:
            # not using a col_map dictionary since input columns could be repeated
            in_col_names, f_vals, additional_args_list = resolve_named_agg_literals(kws)
        else:
            col_map = get_overload_constant_dict(func)
            in_col_names = tuple(col_map.keys())
            f_vals = tuple(col_map.values())
            additional_args_list = [()] * len(col_map)
        for fn in ("head", "ngroup"):
            if fn in f_vals:
                raise BodoError(
                    f"Groupby.agg()/aggregate(): {fn} cannot be mixed with other groupby operations."
                )

        # make sure selected columns exist in dataframe
        if any(c not in grp.selection and c not in grp.keys for c in in_col_names):
            raise_bodo_error(
                f"Selected column names {in_col_names} not all available in dataframe column names {grp.selection}"
            )

        # if a list/tuple of functions is applied to any column, have to use
        # MultiLevel for every column (even if list/tuple length is one)
        multi_level_names = any(isinstance(f_val, (tuple, list)) for f_val in f_vals)

        # NamedAgg case in Pandas doesn't support multiple functions
        if relabeling and multi_level_names:
            raise_bodo_error(
                "Groupby.agg()/aggregate(): cannot pass multiple functions in a single pd.NamedAgg()"
            )

        # get output names and output types
        # gb_info maps (in_cols, additional_args, func_name) -> out_col
        # where in_cols is a tuple of input column names
        gb_info = {}
        out_columns = []
        out_data = []
        out_column_type = []
        f_names = []
        if not grp.as_index:
            get_keys_not_as_index(
                grp,
                out_columns,
                out_data,
                out_column_type,
                multi_level_names=multi_level_names,
            )
        for col_name, f_val, additional_args in zip(
            in_col_names, f_vals, additional_args_list
        ):
            if isinstance(f_val, (tuple, list)):
                lambda_count = 0
                for f in f_val:
                    f_name, out_tp = get_agg_funcname_and_outtyp(
                        grp,
                        col_name,
                        f,
                        additional_args,
                        typing_context,
                        target_context,
                        raise_on_any_error,
                    )
                    has_cumulative_ops = f_name in list_cumulative
                    if f_name == "<lambda>" and len(f_val) > 1:
                        f_name = "<lambda_" + str(lambda_count) + ">"
                        lambda_count += 1
                    # output column name is 2-level (col_name, func_name)
                    # This happens, for example, with
                    # df.groupby(...).agg({"A": [f1, f2]})
                    out_columns.append((col_name, f_name))
                    dict_add_multimap(
                        gb_info, ((col_name,), (), f_name), (col_name, f_name)
                    )
                    _append_out_type(grp, out_data, out_tp)
            else:
                f_name, out_tp = get_agg_funcname_and_outtyp(
                    grp,
                    col_name,
                    f_val,
                    additional_args,
                    typing_context,
                    target_context,
                    raise_on_any_error,
                )
                has_cumulative_ops = f_name in list_cumulative
                if multi_level_names:
                    out_columns.append((col_name, f_name))
                    dict_add_multimap(
                        gb_info, ((col_name,), (), f_name), (col_name, f_name)
                    )
                elif not relabeling:
                    out_columns.append(col_name)
                    dict_add_multimap(gb_info, ((col_name,), (), f_name), col_name)
                elif relabeling:
                    f_names.append(f_name)
                _append_out_type(grp, out_data, out_tp)

        # user specifies output names as kws in NamedAgg case
        if relabeling:
            for i, out_col in enumerate(kws.keys()):
                out_columns.append(out_col)

                if len(kws[out_col]) == 3:
                    input_cols, additional_args = handle_extended_named_agg_input_cols(
                        in_col_names[i], f_names[i], kws[out_col]
                    )
                else:
                    input_cols, additional_args = (in_col_names[i],), ()
                dict_add_multimap(
                    gb_info, (input_cols, additional_args, f_names[i]), out_col
                )

        if has_cumulative_ops:
            # result of groupby.cumsum, etc. doesn't have a group index
            # So instead we set from the input index
            index = grp.df_type.index
        else:
            index = out_tp.index

        out_res = DataFrameType(
            tuple(out_data), index, tuple(out_columns), is_table_format=True
        )
        return signature(out_res, *args), gb_info

    # multi-function tuple or list case
    if (
        isinstance(func, types.BaseTuple)
        and not isinstance(func, types.LiteralStrKeyDict)
    ) or is_overload_constant_list(func):
        if not (len(grp.selection) == 1 and grp.explicit_select):
            raise_bodo_error(
                "Groupby.agg()/aggregate(): must select exactly one column when more than one function is supplied"
            )
        if is_overload_constant_list(func):
            # Lists find functions through their initial/literal values
            func_vals = get_overload_const_list(func)
        else:
            # Tuples can find functions through their types
            func_vals = func.types
        if len(func_vals) == 0:
            raise_bodo_error(
                "Groupby.agg()/aggregate(): List of functions must contain at least 1 function"
            )
        out_data = []
        out_columns = []
        out_column_type = []
        lambda_count = 0
        if not grp.as_index:
            get_keys_not_as_index(grp, out_columns, out_data, out_column_type)
        # gb_info maps (in_cols, additional_args, func_name) -> out_col
        # where in_cols is a tuple of input column names
        gb_info = {}
        in_col_name = grp.selection[0]
        for f_val in func_vals:
            f_name, out_tp = get_agg_funcname_and_outtyp(
                grp,
                in_col_name,
                f_val,
                (),
                typing_context,
                target_context,
                raise_on_any_error,
            )
            has_cumulative_ops = f_name in list_cumulative
            # if tuple has lambdas they will be named <lambda_0>,
            # <lambda_1>, ... in output
            if f_name == "<lambda>" and len(func_vals) > 1:
                f_name = "<lambda_" + str(lambda_count) + ">"
                lambda_count += 1
            out_columns.append(f_name)
            dict_add_multimap(gb_info, ((in_col_name,), (), f_name), f_name)
            _append_out_type(grp, out_data, out_tp)
        if has_cumulative_ops:
            # result of groupby.cumsum, etc. doesn't have a group index
            index = grp.df_type.index
        else:
            index = out_tp.index
        out_res = DataFrameType(
            tuple(out_data), index, tuple(out_columns), is_table_format=True
        )
        return signature(out_res, *args), gb_info

    f_name = ""

    # String case
    if types.unliteral(func) == types.unicode_type:
        # If we have a single string function, we apply the function to the
        # whole dataframe
        f_name = get_overload_const_str(func)

    # Builtin function case, for example df.groupby("B").agg(sum)
    if bodo.utils.typing.is_builtin_function(func):
        f_name = bodo.utils.typing.get_builtin_function_name(func)

    if f_name:
        # Remove func from args.
        args = args[1:]
        kws.pop("func", None)
        return get_agg_typ(grp, args, f_name, typing_context, kws)

    validate_udf("agg", func)
    return get_agg_typ(grp, args, "agg", typing_context, target_context, func)


def resolve_transformative(grp, args, kws, msg, name_operation):
    """For datetime and timedelta datatypes, we can support cummin / cummax,
    but not cumsum / cumprod. Hence the is_minmax entry"""
    index = to_numeric_index_if_range_index(grp.df_type.index)
    if isinstance(index, MultiIndexType):
        raise_bodo_error(
            f"Groupby.{name_operation}: MultiIndex input not supported for groupby operations that use input Index"
        )

    out_columns = []
    out_data = []
    if name_operation in list_cumulative:
        kws = dict(kws) if kws else {}
        # pop arguments from kws or args
        # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
        axis = args[0] if len(args) > 0 else kws.pop("axis", 0)
        numeric_only = args[1] if len(args) > 1 else kws.pop("numeric_only", False)
        if len(args) <= 2:
            kws.pop("skipna", None)
        unsupported_args = {"axis": axis, "numeric_only": numeric_only}
        arg_defaults = {"axis": 0, "numeric_only": False}
        check_unsupported_args(
            f"Groupby.{name_operation}",
            unsupported_args,
            arg_defaults,
            package_name="pandas",
            module_name="GroupBy",
        )
        check_args_kwargs(name_operation, 3, args, kws)
    elif name_operation == "shift":
        # pop arguments from kws or args
        # TODO: [BE-475] Throw an error if both args and kws are passed for same argument
        if len(args) == 0:
            kws.pop("periods", None)
        freq = args[1] if len(args) > 1 else kws.pop("freq", None)
        axis = args[2] if len(args) > 2 else kws.pop("axis", 0)
        fill_value = args[3] if len(args) > 3 else kws.pop("fill_value", None)
        unsupported_args = {"freq": freq, "axis": axis, "fill_value": fill_value}
        arg_defaults = {"freq": None, "axis": 0, "fill_value": None}
        check_unsupported_args(
            f"Groupby.{name_operation}",
            unsupported_args,
            arg_defaults,
            package_name="pandas",
            module_name="GroupBy",
        )
        check_args_kwargs(name_operation, 4, args, kws)
    elif name_operation == "transform":
        kws = dict(kws)
        # pop transform() unsupported keyword arguments from kws
        transform_func = args[0] if len(args) > 0 else kws.pop("func", None)
        transform_func = get_literal_value(transform_func)
        if bodo.utils.typing.is_builtin_function(transform_func):
            # Builtin function case (e.g. df.groupby("B").transform(sum))
            transform_func = bodo.utils.typing.get_builtin_function_name(transform_func)

        engine = kws.pop("engine", None)
        engine_kwargs = kws.pop("engine_kwargs", None)
        unsupported_args = {"engine": engine, "engine_kwargs": engine_kwargs}
        arg_defaults = {"engine": None, "engine_kwargs": None}
        check_unsupported_args(
            "Groupby.transform",
            unsupported_args,
            arg_defaults,
            package_name="pandas",
            module_name="GroupBy",
        )

    # gb_info maps (in_cols, additional_args, func_name) -> out_col
    # where in_cols is a tuple of input column names
    gb_info = {}
    for c in grp.selection:
        out_columns.append(c)
        dict_add_multimap(gb_info, ((c,), (), name_operation), c)
        ind = grp.df_type.column_index[c]
        data = grp.df_type.data[ind]
        operation = name_operation if name_operation != "transform" else transform_func
        if operation in ("sum", "cumsum"):
            data = to_str_arr_if_dict_array(data)
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
        if name_operation == "shift":
            if isinstance(data, (TupleArrayType, ArrayItemArrayType)):
                raise BodoError(msg)
            if isinstance(
                data.dtype,
                bodo.hiframes.datetime_timedelta_ext.DatetimeTimeDeltaType,
            ):
                raise BodoError(
                    f"column type of {data.dtype} is not supported in groupby built-in function shift.\n{dt_err}"
                )
        # Column output depends on the operation in transform.
        if name_operation == "transform":
            out_dtype, err_msg = get_groupby_output_dtype(
                data, transform_func, grp.df_type.index
            )

            if err_msg == "ok":
                data = out_dtype
            else:
                raise BodoError(
                    f"column type of {data.dtype} is not supported by {args[0]} yet.\n"
                )
        out_data.append(data)

    if len(out_data) == 0:
        raise BodoError("No columns in output.")
    out_res = DataFrameType(
        tuple(out_data), index, tuple(out_columns), is_table_format=True
    )
    # XXX output becomes series if single output and explicitly selected
    if len(grp.selection) == 1 and grp.series_select and grp.as_index:
        out_res = SeriesType(
            out_data[0].dtype,
            data=out_data[0],
            index=index,
            name_typ=types.StringLiteral(grp.selection[0]),
        )
    return signature(out_res, *args), gb_info


def extract_window_args(
    func_name: str, func_args: tuple[str]
) -> tuple[list[str], list[str]]:
    """
    Processes a function name and tuple of argument strings corresponding to a window function
    inside of a groupby.window term. Verifies that the number of arguments is correct for
    the input function, and returns the scalar and vector arguments in separate lists

    Args:

        func_name: the string name of which window aggregation is being used.
        func_args: the tuple of string representations of any arguments to the function.

    Returns:

        A list of all scalar arguments as string literals and list of all vector argument names.
    """
    # Map each function to the pattern of expected arguments (default is no arguments)
    func_arg_typs = {
        "ntile": ["scalar"],
        "ratio_to_report": ["vector"],
        "conditional_true_event": ["vector"],
        "conditional_change_event": ["vector"],
        "size": ["scalar", "scalar"],
        "count": ["vector", "scalar", "scalar"],
        "count_if": ["vector", "scalar", "scalar"],
        "var": ["vector", "scalar", "scalar"],
        "var_pop": ["vector", "scalar", "scalar"],
        "std": ["vector", "scalar", "scalar"],
        "std_pop": ["vector", "scalar", "scalar"],
        "mean": ["vector", "scalar", "scalar"],
        "any_value": ["vector"],
        "first": ["vector", "scalar", "scalar"],
        "last": ["vector", "scalar", "scalar"],
    }
    func_arg_typ = func_arg_typs.get(func_name, [])
    # Verify that the input tuple has the correct length
    if len(func_args) != len(func_arg_typ):
        raise_bodo_error(
            f"groupby.window: {func_name} expects {len(func_arg_typ)} arguments, received {len(func_args)}"
        )
    # Append each argument to the correct list
    scalar_args = []
    vector_args = []
    for i in range(len(func_arg_typ)):
        if func_arg_typ[i] == "vector":
            vector_args.append(func_args[i])
        else:
            scalar_args.append(func_args[i])
    return scalar_args, vector_args


def get_window_func_types():
    """
    Return a mapping from function name to an expected output row type.
    This may return None if the output type depends on the input type.
    TODO: Add the input type as an argument.
    """
    window_func_types = {
        "row_number": dtype_to_array_type(types.uint64),
        "rank": dtype_to_array_type(types.uint64),
        "dense_rank": dtype_to_array_type(types.uint64),
        "ntile": dtype_to_array_type(types.uint64),
        "ratio_to_report": to_nullable_type(dtype_to_array_type(types.float64)),
        "conditional_true_event": dtype_to_array_type(types.uint64),
        "conditional_change_event": dtype_to_array_type(types.uint64),
        "size": dtype_to_array_type(types.uint64),
        "count": dtype_to_array_type(types.uint64),
        "count_if": dtype_to_array_type(types.uint64),
        "percent_rank": dtype_to_array_type(types.float64),
        "cume_dist": dtype_to_array_type(types.float64),
        "var": to_nullable_type(dtype_to_array_type(types.float64)),
        "var_pop": to_nullable_type(dtype_to_array_type(types.float64)),
        "std": to_nullable_type(dtype_to_array_type(types.float64)),
        "std_pop": to_nullable_type(dtype_to_array_type(types.float64)),
        "mean": to_nullable_type(dtype_to_array_type(types.float64)),
        "min_row_number_filter": bodo.boolean_array_type,
        "booland_agg": bodo.boolean_array_type,
        "boolor_agg": bodo.boolean_array_type,
        # None = output dtype matches input dtype
        "any_value": None,
        "first": None,
        "last": None,
        "max": None,
        "min": None,
        "sum": None,
        "bitand_agg": None,
        "bitor_agg": None,
        "bitxor_agg": None,
        "lead": None,
        "lag": None,
    }
    return window_func_types


def resolve_window_funcs(
    grp: DataFrameGroupByType,
    args: tuple,
    kws: tuple[tuple[str, types.Type]] | dict[str, types.Type],
) -> tuple[Signature, dict]:
    """
    Output the Numba function signature and groupby information for window functions.
    The groupby information maps the used columns and functions to the output columns.

    Args:
        grp (DataFrameGroupByType): _description_
        args (Tuple): N-Tuple of argument types passed to Numba.
        kws (Union[Tuple[Tuple[str, types.Type]], Dict[str, types.Type]]):
            Either a N-tuple of pairs or Dict mapping kwargs passed to
            Numba to their types.
        msg (str): Error message to output on failure.

    Returns:
        Tuple[Signature, Dict]: The output signature for this function
        and a dictionary mapping the used columns in the input to the output
        and the used function.
    """
    # This operation isn't shared with Pandas so we can reduce
    # memory by just generating a range index.
    index = RangeIndexType(types.none)
    out_columns = []
    out_data = []
    kws = dict(kws)
    default_tuple = types.Tuple([])
    # Extract the relevant arguments from kws or args
    window_funcs = get_literal_value(
        args[0] if len(args) > 0 else kws.pop("funcs", default_tuple)
    )
    order_by = get_literal_value(
        args[1] if len(args) > 1 else kws.pop("order_by", default_tuple)
    )
    ascending = get_literal_value(
        args[2] if len(args) > 2 else kws.pop("ascending", default_tuple)
    )
    na_position = get_literal_value(
        args[3] if len(args) > 3 else kws.pop("na_position", default_tuple)
    )
    # We currently require only a single order by column as that satisfies the initial

    if not (
        isinstance(order_by, tuple)
        and all(isinstance(col_name, str) for col_name in order_by)
    ):
        raise_bodo_error(
            "Groupby.window: 'order_by' argument must be a tuple of column names if provided."
        )
    if not (
        isinstance(ascending, tuple) and all(isinstance(val, bool) for val in ascending)
    ):
        raise_bodo_error(
            "Groupby.window: 'ascending' argument must be a tuple of booleans if provided."
        )
    if not (
        isinstance(na_position, tuple)
        and all(isinstance(val, str) for val in na_position)
    ):
        raise_bodo_error(
            "Groupby.window: 'na_position' argument must be a tuple of 'first' or 'last' if provided."
        )

    # Verify that every order_by column exists in the
    if any(col_name not in grp.df_type.column_index for col_name in order_by):
        raise_bodo_error(
            f"Groupby.window: Column '{order_by}' does not exist in the input dataframe."
        )
    # Reuse the name currently generated by BodoSQL
    out_columns = [f"AGG_OUTPUT_{i}" for i in range(len(window_funcs))]
    out_data = []
    in_cols = list(order_by)

    for window_func in window_funcs:
        if len(window_func) == 0:
            raise_bodo_error(
                "Invalid groupby.window term: argument tuples cannot have length zero"
            )
        func_name = window_func[0]
        func_args = window_func[1:]
        window_func_types = get_window_func_types()
        if func_name not in window_func_types:
            raise_bodo_error(f"Unrecognized window function {func_name}")
        _, vector_args = extract_window_args(func_name, func_args)
        in_cols.extend(vector_args)
        out_dtype = window_func_types[func_name]
        # None = output dtype matches input dtype
        if out_dtype is None:
            ind = grp.df_type.column_index[vector_args[0]]
            in_arr_type = grp.df_type.data[ind]
            out_dtype = in_arr_type
            # If the function allows frames, the output can be nullable
            # even if the input is not
            if func_name in {"first", "last"}:
                out_dtype = to_nullable_type(out_dtype)
        out_data.append(out_dtype)

    # Generate the gb_info
    # gb_info maps (in_cols, additional_args, func_name) -> out_col
    # where in_cols is a tuple of input column names
    gb_info = {(tuple(in_cols), (), "window"): [out_columns[0]]}

    out_res = DataFrameType(
        tuple(out_data), index, tuple(out_columns), is_table_format=True
    )
    # XXX output becomes series if single output and explicitly selected
    # TODO: Drop the keys for this groupby and just work as a series output
    if len(grp.selection) == 1 and grp.series_select and grp.as_index:
        out_res = SeriesType(
            out_data[0].dtype,
            data=out_data[0],
            index=index,
            name_typ=types.StringLiteral(grp.selection[0]),
        )
    return signature(out_res, *args), gb_info


def resolve_gb(grp, args, kws, func_name, typing_context, target_context, err_msg=""):
    """Given a groupby function returns 2-tuple with output signature
    and dict with mapping of (in_col, func_name) -> [out_col_1, out_col_2, ...]
    """

    if func_name in set(list_cumulative) | {"shift", "transform"}:
        return resolve_transformative(grp, args, kws, err_msg, func_name)
    elif func_name == "window":
        return resolve_window_funcs(grp, args, kws)
    elif func_name in {"agg", "aggregate"}:
        return resolve_agg(grp, args, kws, typing_context, target_context)
    else:
        return get_agg_typ(
            grp, args, func_name, typing_context, target_context, kws=kws
        )


@infer_getattr
class DataframeGroupByAttribute(OverloadedKeyAttributeTemplate):
    key = DataFrameGroupByType
    # Set of attribute names stored for caching.
    _attr_set = None

    # NOTE the resolve functions return output signature of groupby to Numba
    # typer, so we return the first value returned by `resolve_gb`

    @bound_function("groupby.agg", no_unliteral=True)
    def resolve_agg(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "agg",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.aggregate", no_unliteral=True)
    def resolve_aggregate(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "agg",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.sum", no_unliteral=True)
    def resolve_sum(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "sum",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.count", no_unliteral=True)
    def resolve_count(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "count",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.nunique", no_unliteral=True)
    def resolve_nunique(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "nunique",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.median", no_unliteral=True)
    def resolve_median(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "median",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.mean", no_unliteral=True)
    def resolve_mean(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "mean",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.min", no_unliteral=True)
    def resolve_min(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "min",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.max", no_unliteral=True)
    def resolve_max(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "max",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.std", no_unliteral=True)
    def resolve_std(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "std",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.prod", no_unliteral=True)
    def resolve_prod(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "prod",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.var", no_unliteral=True)
    def resolve_var(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "var",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.kurtosis", no_unliteral=True)
    def resolve_kurtosis(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "kurtosis",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.skew", no_unliteral=True)
    def resolve_skew(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "skew",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.kurtosis", no_unliteral=True)
    def resolve_kurtosis(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "kurtosis",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.first", no_unliteral=True)
    def resolve_first(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "first",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.last", no_unliteral=True)
    def resolve_last(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "last",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.idxmin", no_unliteral=True)
    def resolve_idxmin(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "idxmin",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.idxmax", no_unliteral=True)
    def resolve_idxmax(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "idxmax",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.size", no_unliteral=True)
    def resolve_size(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "size",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.cumsum", no_unliteral=True)
    def resolve_cumsum(self, grp, args, kws):
        msg = "Groupby.cumsum() only supports columns of types integer, float, string or liststring"
        return resolve_gb(
            grp,
            args,
            kws,
            "cumsum",
            self.context,
            numba.core.registry.cpu_target.target_context,
            err_msg=msg,
        )[0]

    @bound_function("groupby.cumprod", no_unliteral=True)
    def resolve_cumprod(self, grp, args, kws):
        msg = "Groupby.cumprod() only supports columns of types integer and float"
        return resolve_gb(
            grp,
            args,
            kws,
            "cumprod",
            self.context,
            numba.core.registry.cpu_target.target_context,
            err_msg=msg,
        )[0]

    @bound_function("groupby.cummin", no_unliteral=True)
    def resolve_cummin(self, grp, args, kws):
        msg = "Groupby.cummin() only supports columns of types integer, float, string, liststring, date, datetime or timedelta"
        return resolve_gb(
            grp,
            args,
            kws,
            "cummin",
            self.context,
            numba.core.registry.cpu_target.target_context,
            err_msg=msg,
        )[0]

    @bound_function("groupby.cummax", no_unliteral=True)
    def resolve_cummax(self, grp, args, kws):
        msg = "Groupby.cummax() only supports columns of types integer, float, string, liststring, date, datetime or timedelta"
        return resolve_gb(
            grp,
            args,
            kws,
            "cummax",
            self.context,
            numba.core.registry.cpu_target.target_context,
            err_msg=msg,
        )[0]

    @bound_function("groupby.shift", no_unliteral=True)
    def resolve_shift(self, grp, args, kws):
        msg = "Column type of list/tuple is not supported in groupby built-in function shift"
        return resolve_gb(
            grp,
            args,
            kws,
            "shift",
            self.context,
            numba.core.registry.cpu_target.target_context,
            err_msg=msg,
        )[0]

    @bound_function("groupby.pipe", no_unliteral=True)
    def resolve_pipe(self, grp, args, kws):
        return resolve_obj_pipe(self, grp, args, kws, "GroupBy")

    @bound_function("groupby.transform", no_unliteral=True)
    def resolve_transform(self, grp, args, kws):
        msg = "Groupby.transform() only supports sum, count, min, max, mean, and std operations"
        return resolve_gb(
            grp,
            args,
            kws,
            "transform",
            self.context,
            numba.core.registry.cpu_target.target_context,
            err_msg=msg,
        )[0]

    @bound_function("groupby.window", no_unliteral=True)
    def resolve_window(self, grp, args, kws):
        return resolve_gb(
            grp,
            args,
            kws,
            "window",
            self.context,
            numba.core.registry.cpu_target.target_context,
        )[0]

    @bound_function("groupby.head", no_unliteral=True)
    def resolve_head(self, grp, args, kws):
        msg = "Unsupported Groupby head operation.\n"
        return resolve_gb(
            grp,
            args,
            kws,
            "head",
            self.context,
            numba.core.registry.cpu_target.target_context,
            err_msg=msg,
        )[0]

    @bound_function("groupby.ngroup", no_unliteral=True)
    def resolve_ngroup(self, grp, args, kws):
        msg = "Unsupported Gropupby head operation.\n"
        return resolve_gb(
            grp,
            args,
            kws,
            "ngroup",
            self.context,
            numba.core.registry.cpu_target.target_context,
            err_msg=msg,
        )[0]

    @bound_function("groupby.apply", no_unliteral=True)
    def resolve_apply(self, grp, args, kws):
        kws = dict(kws)
        # pop apply() arguments from kws so only UDF kws remain
        func = args[0] if len(args) > 0 else kws.pop("func", None)
        f_args = tuple(args[1:]) if len(args) > 0 else ()

        f_return_type = _get_groupby_apply_udf_out_type(
            func,
            grp,
            f_args,
            kws,
            self.context,
            numba.core.registry.cpu_target.target_context,
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
                get_keys_not_as_index(grp, out_columns, out_data, out_column_type)
                # group number is assigned to output
                out_index_type = NumericIndexType(types.int64, types.none)
            else:
                if len(grp.keys) > 1:
                    key_col_inds = tuple(
                        grp.df_type.column_index[grp.keys[i]]
                        for i in range(len(grp.keys))
                    )
                    arr_types = tuple(grp.df_type.data[ind] for ind in key_col_inds)
                    out_index_type = MultiIndexType(
                        arr_types, tuple(types.literal(k) for k in grp.keys)
                    )
                else:
                    ind = grp.df_type.column_index[grp.keys[0]]
                    ind_arr_t = grp.df_type.data[ind]
                    out_index_type = bodo.hiframes.pd_index_ext.array_type_to_index(
                        ind_arr_t, types.literal(grp.keys[0])
                    )
            out_data = tuple(out_data)
            out_columns = tuple(out_columns)
        else:
            key_arr_types = tuple(
                grp.df_type.data[grp.df_type.column_index[c]] for c in grp.keys
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
                assert_bodo_error(f_return_type.const_info is not None)
                _, index_vals = f_return_type.const_info
                # Heterogenous Series should always return a Nullable Tuple in the output type,
                if isinstance(
                    f_return_type.data, bodo.libs.nullable_tuple_ext.NullableTupleType
                ):
                    scalar_types = f_return_type.data.tuple_typ.types
                elif isinstance(f_return_type.data, types.Tuple):
                    # TODO: Confirm if this path ever taken? It shouldn't be.
                    scalar_types = f_return_type.data.types
                # NOTE: nullable is determined at runtime, so by default always assume nullable type
                # TODO: Support for looking at constant values.
                arrs = tuple(
                    to_nullable_type(dtype_to_array_type(t)) for t in scalar_types
                )
                ret_type = DataFrameType(
                    out_data + arrs,
                    out_index_type,
                    out_columns + index_vals,
                )
            elif isinstance(f_return_type, SeriesType):
                n_cols, index_vals = f_return_type.const_info
                # Note: For homogenous Series we return a regular tuple, so
                # convert to nullable.
                # NOTE: nullable is determined at runtime, so by default always assume nullable type
                # TODO: Support for looking at constant values.
                arrs = tuple(
                    to_nullable_type(dtype_to_array_type(f_return_type.dtype))
                    for _ in range(n_cols)
                )
                ret_type = DataFrameType(
                    out_data + arrs,
                    out_index_type,
                    out_columns + index_vals,
                )
            else:  # scalar case
                data_arr = get_udf_out_arr_type(f_return_type)
                if not grp.as_index:
                    # TODO: Pandas sets NaN for data column
                    ret_type = DataFrameType(
                        out_data + (data_arr,),
                        out_index_type,
                        out_columns + ("",),
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
                f_return_type.data,
                out_index_type,
                f_return_type.columns,
            )

        pysig = gen_apply_pysig(len(f_args), kws.keys())
        new_args = (func, *f_args) + tuple(kws.values())
        return signature(ret_type, *new_args).replace(pysig=pysig)

    def generic_resolve(self, grpby, attr):
        if self._is_existing_attr(attr):
            return
        if attr not in grpby.df_type.columns:
            raise_bodo_error(
                f"groupby: invalid attribute {attr} (column not found in dataframe or unsupported function)"
            )
        return DataFrameGroupByType(
            grpby.df_type,
            grpby.keys,
            (attr,),
            grpby.as_index,
            grpby.dropna,
            True,
            True,
            _num_shuffle_keys=grpby._num_shuffle_keys,
            _use_sql_rules=grpby._use_sql_rules,
        )


def _get_groupby_apply_udf_out_type(
    func, grp, f_args, kws, typing_context, target_context
):
    """get output type for UDF used in groupby apply()"""

    # NOTE: without explicit column selection, Pandas passes key columns also for
    # some reason (as of Pandas 1.1.5)
    in_df_type = grp.df_type
    if grp.explicit_select:
        # input to UDF is a Series if only one column is explicitly selected
        if len(grp.selection) == 1:
            col_name = grp.selection[0]
            data_arr = in_df_type.data[in_df_type.column_index[col_name]]
            in_data_type = SeriesType(
                data_arr.dtype, data_arr, in_df_type.index, types.literal(col_name)
            )
        else:
            in_data = tuple(
                in_df_type.data[in_df_type.column_index[c]] for c in grp.selection
            )
            in_data_type = DataFrameType(
                in_data, in_df_type.index, tuple(grp.selection)
            )
    else:
        in_data_type = in_df_type

    arg_typs = (in_data_type,)
    arg_typs += tuple(f_args)
    try:
        f_return_type = get_const_func_output_type(
            func, arg_typs, kws, typing_context, target_context
        )
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
            func,
            arg_typs,
            kws,
            self.context,
            numba.core.registry.cpu_target.target_context,
            False,
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
            index.data,
            types.StringLiteral("index"),
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


# _is_parallel: Bodo flag for tracing
def get_group_indices(keys, dropna, _is_parallel):  # pragma: no cover
    return np.arange(len(keys))


@overload(get_group_indices)
def get_group_indices_overload(keys, dropna, _is_parallel):
    """get group indices (labels) for a tuple of key arrays."""
    func_text = "def impl(keys, dropna, _is_parallel):\n"
    # convert arrays to table
    func_text += (
        "    ev = bodo.utils.tracing.Event('get_group_indices', _is_parallel)\n"
    )
    func_text += "    info_list = [{}]\n".format(
        ", ".join(f"array_to_info(keys[{i}])" for i in range(len(keys.types))),
    )
    func_text += "    table = arr_info_list_to_table(info_list)\n"
    func_text += "    group_labels = np.empty(len(keys[0]), np.int64)\n"
    func_text += "    sort_idx = np.empty(len(keys[0]), np.int64)\n"
    func_text += "    ngroups = get_groupby_labels(table, group_labels.ctypes, sort_idx.ctypes, dropna, _is_parallel)\n"
    func_text += "    ev.finalize()\n"
    func_text += "    return sort_idx, group_labels, ngroups\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "np": np,
            "get_groupby_labels": get_groupby_labels,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
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


def shuffle_dataframe(df, keys, _is_parallel):  # pragma: no cover
    return df, keys, _is_parallel


@overload(shuffle_dataframe, prefer_literal=True)
def overload_shuffle_dataframe(df, keys, _is_parallel):
    impl, _ = gen_shuffle_dataframe(df, keys, _is_parallel)
    return impl


def gen_shuffle_dataframe(df, keys, _is_parallel):
    """shuffle a dataframe using a tuple of key arrays."""
    n_cols = len(df.columns)
    n_keys = len(keys.types)
    assert is_overload_constant_bool(_is_parallel), (
        "shuffle_dataframe: _is_parallel is not a constant"
    )

    func_text = "def impl(df, keys, _is_parallel):\n"

    # generating code based on _is_parallel flag statically instead of runtime 'if'
    # check to avoid Numba's refcount pruning bugs.
    # See https://bodo.atlassian.net/browse/BE-974
    if is_overload_false(_is_parallel):
        func_text += "  return df, keys, get_null_shuffle_info()\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "get_null_shuffle_info": get_null_shuffle_info,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    # create C++ table from input arrays
    for i in range(n_cols):
        func_text += f"  in_arr{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})\n"

    func_text += "  in_index_arr = bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df))\n"

    func_text += "  info_list = [{}, {}, {}]\n".format(
        ", ".join(f"array_to_info(keys[{i}])" for i in range(n_keys)),
        ", ".join(f"array_to_info(in_arr{i})" for i in range(n_cols)),
        "array_to_info(in_index_arr)",
    )
    func_text += "  table = arr_info_list_to_table(info_list)\n"
    # NOTE: C++ will delete table pointer
    func_text += f"  out_table = shuffle_table(table, {n_keys}, _is_parallel, 1)\n"

    # extract arrays from C++ table
    for i in range(n_keys):
        func_text += (
            f"  out_key{i} = array_from_cpp_table(out_table, {i}, keys{i}_typ)\n"
        )

    for i in range(n_cols):
        func_text += f"  out_arr{i} = array_from_cpp_table(out_table, {i + n_keys}, in_arr{i}_typ)\n"

    func_text += f"  out_arr_index = array_from_cpp_table(out_table, {n_keys + n_cols}, ind_arr_typ)\n"

    func_text += "  shuffle_info = get_shuffle_info(out_table)\n"
    func_text += "  delete_table(out_table)\n"

    out_data = ", ".join(f"out_arr{i}" for i in range(n_cols))
    func_text += "  out_index = bodo.utils.conversion.index_from_array(out_arr_index)\n"
    func_text += f"  out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe(({out_data},), out_index, __col_name_meta_value_df_shuffle)\n"

    func_text += "  return out_df, ({},), shuffle_info\n".format(
        ", ".join(f"out_key{i}" for i in range(n_keys))
    )

    glbls = {
        "bodo": bodo,
        "array_to_info": array_to_info,
        "arr_info_list_to_table": arr_info_list_to_table,
        "shuffle_table": shuffle_table,
        "array_from_cpp_table": array_from_cpp_table,
        "delete_table": delete_table,
        "get_shuffle_info": get_shuffle_info,
        "__col_name_meta_value_df_shuffle": ColNamesMetaType(df.columns),
        "ind_arr_typ": (
            types.Array(types.int64, 1, "C")
            if isinstance(df.index, RangeIndexType)
            else df.index.data
        ),
    }
    glbls.update({f"keys{i}_typ": keys.types[i] for i in range(n_keys)})
    glbls.update({f"in_arr{i}_typ": df.data[i] for i in range(n_cols)})

    loc_vars = {}
    exec(
        func_text,
        glbls,
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl, glbls


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
        # NOTE: C++ will delete table pointer
        func_text += "  out_table = reverse_shuffle_table(table, shuffle_info)\n"
        for i in range(n_fields):
            func_text += f"  out_arr{i} = array_from_cpp_table(out_table, {i}, data._data[{i}])\n"
        func_text += "  delete_table(out_table)\n"
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
                "array_from_cpp_table": array_from_cpp_table,
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
        # NOTE: C++ will delete table pointer
        out_table = reverse_shuffle_table(table, shuffle_info)
        out_arr = array_from_cpp_table(out_table, 0, data)
        delete_table(out_table)
        return out_arr

    return impl_arr


@overload_method(
    DataFrameGroupByType, "value_counts", inline="always", no_unliteral=True
)
def groupby_value_counts(
    grp, normalize=False, sort=True, ascending=False, bins=None, dropna=True
):
    unsupported_args = {
        "normalize": normalize,
        "sort": sort,
        "bins": bins,
        "dropna": dropna,
    }
    arg_defaults = {"normalize": False, "sort": True, "bins": None, "dropna": True}
    check_unsupported_args(
        "Groupby.value_counts",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="GroupBy",
    )

    # Pandas restriction: value_counts work on SeriesGroupBy only so only one column selection is allowed
    if (len(grp.selection) > 1) or (not grp.as_index):
        raise BodoError("'DataFrameGroupBy' object has no attribute 'value_counts'")

    if not is_overload_constant_bool(ascending):
        raise BodoError("Groupby.value_counts() ascending must be a constant boolean")

    ascending_val = get_overload_const_bool(ascending)

    # df.groupby("X")["Y"].value_counts() => df.groupby("X")["Y"].apply(lambda S : S.value_counts())
    func_text = "def impl(grp, normalize=False, sort=True, ascending=False, bins=None, dropna=True):\n"
    # TODO: [BE-635] Use S.rename_axis
    udf = f"lambda S: S.value_counts(ascending={ascending_val})"
    func_text += f"    return grp.apply({udf})\n"
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


# Unsupported general Groupby attributes
groupby_unsupported_attr = {
    # Indexing, Iteration
    "groups",
    "indices",
}

# Unsupported general Groupby operations
groupby_unsupported = {
    # Indexing, Iteration
    "__iter__",
    "get_group",
    # Computation/ descriptive stats GroupBy
    "all",
    "any",
    "bfill",
    "backfill",
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "ffill",
    "nth",
    "ohlc",
    "pad",
    "rank",
    "pct_change",
    "sem",
    "tail",
    # DataFrame section (excluding anything already in GroupBy)
    "corr",
    "cov",
    "describe",
    "diff",
    "fillna",
    "filter",
    "hist",
    "plot",
    "quantile",
    "resample",
    "sample",
    "take",
    "tshift",
}

# Attributes/Methods exclusive to SeriesGroupBy
series_only_unsupported_attrs = {
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
}

series_only_unsupported = {"nlargest", "nsmallest", "unique"}

# Attributes/Methods exclusive to DataFrameGroupBy
dataframe_only_unsupported = {
    "corrwith",
    "boxplot",
}


def _install_groupby_unsupported():
    """install an overload that raises BodoError for unsupported methods of GroupBy,
    DataFrameGroupBy, and SeriesGroupBy types
    """

    for attr_name in groupby_unsupported_attr:
        bodo.overload_unsupported_attribute(
            DataFrameGroupByType, attr_name, f"DataFrameGroupBy.{attr_name}"
        )

    for fname in groupby_unsupported:
        bodo.overload_unsupported_method(
            DataFrameGroupByType, fname, f"DataFrameGroupBy.{fname}"
        )

    # TODO: Replace DataFrameGroupByType with SeriesGroupByType once we
    # have separate types.
    for attr_name in series_only_unsupported_attrs:
        bodo.overload_unsupported_attribute(
            DataFrameGroupByType, attr_name, f"SeriesGroupBy.{attr_name}"
        )

    for fname in series_only_unsupported:
        bodo.overload_unsupported_method(
            DataFrameGroupByType, fname, f"SeriesGroupBy.{fname}"
        )

    for fname in dataframe_only_unsupported:
        bodo.overload_unsupported_method(
            DataFrameGroupByType, fname, f"DataFrameGroupBy.{fname}"
        )


_install_groupby_unsupported()
