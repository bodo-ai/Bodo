# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Helper functions to enable typing.
"""
import operator
import itertools
import types as pytypes
import numpy as np
import pandas as pd
import numba
from numba.core import types, cgutils, ir_utils
from numba.extending import (
    register_model,
    models,
    overload,
    register_jitable,
    lower_cast,
    typeof_impl,
    unbox,
    NativeValue,
)
from numba.core.typing.templates import infer_global, AbstractTemplate, CallableTemplate
from numba.core.typing import signature
from numba.core.imputils import lower_builtin, impl_ret_borrowed, impl_ret_new_ref
from numba.core.registry import CPUDispatcher
import bodo


# sentinel string used in typing pass that specifies a const tuple as a const dict.
# const tuple is used since there is no literal type for dict
CONST_DICT_SENTINEL = "$_bodo_const_dict_$"


list_cumulative = {"cumsum", "cumprod", "cummin", "cummax"}


def is_dtype_nullable(in_dtype):
    """checks whether 'in_dtype' has sentinel NA values (as opposed to bitmap)"""
    return isinstance(in_dtype, (types.Float, types.NPDatetime, types.NPTimedelta))


class BodoError(BaseException):
    """Bodo error class that inherits from BaseException instead of Exception to avoid
    numba's error catching, which enables raising simpler error directly to the user.
    TODO: change to Exception when possible.
    """

    def __init__(self, msg, is_new=True):
        self.is_new = is_new
        highlight = numba.core.errors.termcolor().errmsg
        super(BodoError, self).__init__(highlight(msg))


class BodoException(Exception):
    """Bodo exception that inherits from Exception to allow typing pass to catch it
    and potentially transform the IR.
    """

    pass


class BodoNotConstError(Exception):
    """Indicates that a constant value is expected but non-constant is provided.
    Only used in partial typing pass to enable IR transformation. Inherits from regular
    Exception class for this purpose.
    """


class BodoConstUpdatedError(Exception):
    """Indicates that a constant value is expected but the input list/dict/set is
    updated in place. Only used in partial typing pass to enable error checking.
    """


def raise_const_error(msg):
    """raises an error indicating that a constant value is expected with the given
    message 'msg'.
    Raises BodoNotConstError during partial type inference, and BodoError otherwise.
    """
    if bodo.transforms.typing_pass.in_partial_typing:
        bodo.transforms.typing_pass.typing_transform_required = True
        raise BodoNotConstError(msg)
    else:
        raise BodoError(msg)


def raise_bodo_error(msg):
    """Raises BodoException during partial typing in case typing transforms can handle
    the issue. Otherwise, raises BodoError.
    """
    if bodo.transforms.typing_pass.in_partial_typing:
        bodo.transforms.typing_pass.typing_transform_required = True
        raise BodoException(msg)
    else:
        raise BodoError(msg)


class BodoWarning(Warning):
    """
    Warning class for Bodo-related potential issues such as prevention of
    parallelization by unsupported functions.
    """


# sentinel value representing non-constant values
class NotConstant:
    pass


NOT_CONSTANT = NotConstant()


def is_overload_none(val):
    return val is None or val == types.none or getattr(val, "value", False) is None


def is_overload_constant_bool(val):
    return (
        isinstance(val, bool)
        or isinstance(val, bodo.utils.typing.BooleanLiteral)
        or ((isinstance(val, types.Omitted) and isinstance(val.value, bool)))
    )


def is_overload_bool(val):
    return isinstance(val, types.Boolean) or is_overload_constant_bool(val)


def is_overload_constant_str(val):
    return (
        isinstance(val, str)
        or (isinstance(val, types.StringLiteral) and isinstance(val.literal_value, str))
        or ((isinstance(val, types.Omitted) and isinstance(val.value, str)))
    )


def is_overload_constant_list(val):
    return (
        isinstance(val, types.BaseTuple)
        and all(isinstance(t, types.Literal) for t in val.types)
        # avoid const dict values stored as const tuple
        and (not val.types or val.types[0] != types.StringLiteral(CONST_DICT_SENTINEL))
    ) or (isinstance(val, bodo.utils.typing.ListLiteral))


def is_overload_constant_tuple(val):
    return isinstance(val, tuple) or (
        isinstance(val, types.Omitted) and isinstance(val.value, tuple)
    )


def is_overload_constant_dict(val):
    """const dict values are stored as a const tuple with a sentinel
    """
    return (
        isinstance(val, types.BaseTuple)
        and val.types
        and val.types[0] == types.StringLiteral(CONST_DICT_SENTINEL)
    )


def is_overload_constant_int(val):
    return (
        isinstance(val, int)
        or (
            isinstance(val, types.IntegerLiteral) and isinstance(val.literal_value, int)
        )
        or ((isinstance(val, types.Omitted) and isinstance(val.value, int)))
    )


def is_overload_bool_list(val):
    return (
        isinstance(val, numba.core.types.List)
        and isinstance(val.dtype, types.Boolean)
        or (
            isinstance(val, types.BaseTuple)
            and all(isinstance(v, bodo.utils.typing.BooleanLiteral) for v in val.types)
        )
    )


def is_overload_true(val):
    return (
        val == True
        or val == bodo.utils.typing.BooleanLiteral(True)
        or getattr(val, "value", False) is True
    )


def is_overload_false(val):
    return (
        val == False
        or val == bodo.utils.typing.BooleanLiteral(False)
        or getattr(val, "value", True) is False
    )


def is_overload_zero(val):
    return val == 0 or val == types.IntegerLiteral(0) or getattr(val, "value", -1) == 0


def is_overload_str(val, const):
    return (
        val == const
        or val == types.StringLiteral(const)
        or getattr(val, "value", -1) == const
    )


def get_overload_const(val):
    """Get constant value for overload input. Returns NOT_CONSTANT if not constant.
    'val' can be a python value, an Omitted type, a literal type, or other Numba type
    (in case of non-constant).
    Supports None, bool, int, str, and tuple values.
    """
    # sometimes Dispatcher objects become TypeRef, see test_groupby_agg_const_dict
    if isinstance(val, types.TypeRef):
        val = val.instance_type
    # actual value
    if val is None or isinstance(val, (bool, int, str, tuple)):
        return val
    # Omitted case
    if isinstance(val, types.Omitted):
        return val.value
    # Literal value
    if isinstance(val, types.Literal):
        return val.literal_value
    if isinstance(val, types.Dispatcher):
        return val
    if isinstance(val, types.BaseTuple):
        return [get_overload_const(v) for v in val.types]
    return NOT_CONSTANT


# string representation of basic types for printing
_const_type_repr = {str: "string", bool: "boolean", int: "integer"}


def ensure_constant_arg(fname, arg_name, val, const_type):
    """Make sure argument 'val' to overload of function 'fname' is a constant of type
    'const_type'. Otherwise, raise BodoError.
    """
    const_val = get_overload_const(val)
    const_type_name = _const_type_repr.get(const_type, str(const_type))

    if not isinstance(const_val, const_type):
        raise BodoError(
            f"{fname}(): argument '{arg_name}' should be a constant "
            f"{const_type_name} not {val}"
        )


def ensure_constant_values(fname, arg_name, val, const_values):
    """Make sure argument 'val' to overload of function 'fname' is one of the values in
    'const_values'. Otherwise, raise BodoError.
    """
    const_val = get_overload_const(val)

    if const_val not in const_values:
        raise BodoError(
            f"{fname}(): argument '{arg_name}' should be a constant value in "
            f"{const_values} not '{const_val}'"
        )


def check_unsupported_args(fname, args_dict, arg_defaults_dict):
    """Check for unsupported arguments for function 'fname', and raise an error if any
    value other than the default is provided.
    'args_dict' is a dictionary of provided arguments in overload.
    'arg_defaults_dict' is a dictionary of default values for unsupported arguments.
    """
    assert len(args_dict) == len(arg_defaults_dict)
    for a in args_dict:
        v1 = get_overload_const(args_dict[a])
        v2 = arg_defaults_dict[a]
        if (
            v1 is NOT_CONSTANT
            or (v1 is not None and v2 is None)
            or (v1 is None and v2 is not None)
            or v1 != v2
        ):
            raise BodoError(
                f"{fname}(): {a} parameter only supports default value {v2}"
            )


def get_overload_const_tuple(val):
    if isinstance(val, tuple):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, tuple)
        return val.value


def get_overload_constant_dict(val):
    """get constant dict values from literal type (stored as const tuple)
    """
    assert (
        isinstance(val, types.BaseTuple)
        and val.types
        and val.types[0] == types.StringLiteral(CONST_DICT_SENTINEL)
    )
    # get values excluding sentinel
    items = [get_overload_const(v) for v in val.types[1:]]
    # create dict and return
    return {items[2 * i]: items[2 * i + 1] for i in range(len(items) // 2)}


def get_overload_const_str_len(val):
    if isinstance(val, str):
        return len(val)
    if isinstance(val, types.StringLiteral) and isinstance(val.literal_value, str):
        return len(val.literal_value)
    if isinstance(val, types.Omitted) and isinstance(val.value, str):
        return len(val.value)


def get_overload_const_list(val):
    """returns a constant list from type 'val', which could be a single value
    literal, a constant list or a constant tuple.
    """
    if isinstance(val, bodo.utils.typing.ListLiteral):
        return val.literal_value
    if isinstance(val, types.Omitted):
        return [val.value]
    # literal case
    if isinstance(val, types.Literal):
        return [val.literal_value]
    if isinstance(val, types.BaseTuple) and all(
        isinstance(t, types.Literal) for t in val.types
    ):
        return tuple(t.literal_value for t in val.types)


def get_overload_const_str(val):
    if isinstance(val, str):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, str)
        return val.value
    # literal case
    if isinstance(val, types.StringLiteral):
        assert isinstance(val.literal_value, str)
        return val.literal_value
    raise BodoError("{} not constant string".format(val))


def get_overload_const_int(val):
    if isinstance(val, int):
        return val
    if isinstance(val, types.Omitted):
        assert isinstance(val.value, int)
        return val.value
    # literal case
    if isinstance(val, types.IntegerLiteral):
        assert isinstance(val.literal_value, int)
        return val.literal_value
    raise BodoError("{} not constant integer".format(val))


def is_const_func_type(t):
    """check if 't' is a constant function type
    """
    return isinstance(
        t,
        (
            types.MakeFunctionLiteral,
            bodo.utils.typing.FunctionLiteral,
            types.Dispatcher,
        ),
    )


def get_overload_const_func(val):
    """get constant function object or ir.Expr.make_function from function type
    """
    if isinstance(val, (types.MakeFunctionLiteral, bodo.utils.typing.FunctionLiteral)):
        return val.literal_value
    if isinstance(val, types.Dispatcher):
        return val.dispatcher.py_func
    if isinstance(val, CPUDispatcher):
        return val.py_func
    raise BodoError("'{}' not a constant function type".format(val))


# TODO: move to Numba
def parse_dtype(dtype):
    if isinstance(dtype, types.DTypeSpec):
        return dtype.dtype

    try:
        d_str = get_overload_const_str(dtype)
        if d_str.startswith("Int") or d_str.startswith("UInt"):
            return bodo.libs.int_arr_ext.typeof_pd_int_dtype(
                pd.api.types.pandas_dtype(d_str), None
            )
        return numba.np.numpy_support.from_dtype(np.dtype(d_str))
    except:
        pass
    raise BodoError("invalid dtype {}".format(dtype))


def is_list_like_index_type(t):
    """Types that can be similar to list for indexing Arrays, Series, etc.
    Tuples are excluded due to indexing semantics.
    """
    from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.libs.bool_arr_ext import boolean_array

    # TODO: include datetimeindex/timedeltaindex?

    return (
        isinstance(t, types.List)
        or (isinstance(t, types.Array) and t.ndim == 1)
        or isinstance(t, (NumericIndexType, RangeIndexType))
        or isinstance(t, SeriesType)
        # or isinstance(t, IntegerArrayType)  # TODO: is this necessary?
        or t == boolean_array
    )


def get_index_names(t, func_name, default_name):
    """get name(s) of index type 't', assuming constant string literal name(s) are used.
    otherwise, throw error.
    """
    from bodo.hiframes.pd_multi_index_ext import MultiIndexType

    err_msg = "{}: index name should be a constant string".format(func_name)

    # MultIndex has multiple names
    if isinstance(t, MultiIndexType):
        names = []
        for i, n_typ in enumerate(t.names_typ):
            if n_typ == types.none:
                names.append("level_{}".format(i))
                continue
            if not is_overload_constant_str(n_typ):
                raise BodoError(err_msg)
            names.append(get_overload_const_str(n_typ))
        return tuple(names)

    # other indices have a single name
    if t.name_typ == types.none:
        return (default_name,)
    if not is_overload_constant_str(t.name_typ):
        raise BodoError(err_msg)
    return (get_overload_const_str(t.name_typ),)


def get_index_data_arr_types(t):
    from bodo.hiframes.pd_index_ext import (
        NumericIndexType,
        RangeIndexType,
        StringIndexType,
        DatetimeIndexType,
        TimedeltaIndexType,
        PeriodIndexType,
    )
    from bodo.hiframes.pd_multi_index_ext import MultiIndexType

    if isinstance(t, MultiIndexType):
        return tuple(t.array_types)

    if isinstance(t, (RangeIndexType, PeriodIndexType)):
        return (types.Array(types.int64, 1, "C"),)

    if isinstance(t, NumericIndexType):
        return (types.Array(t.dtype, 1, "C"),)

    if isinstance(t, StringIndexType):
        return (bodo.string_array_type,)

    if isinstance(t, DatetimeIndexType):
        return (bodo.hiframes.pd_index_ext._dt_index_data_typ,)

    if isinstance(t, TimedeltaIndexType):
        return (bodo.hiframes.pd_index_ext._timedelta_index_data_typ,)

    raise BodoError("Invalid index type {}".format(t))


def get_val_type_maybe_str_literal(value):
    """Get type of value, using StringLiteral if possible
    """
    t = numba.typeof(value)
    if isinstance(value, str):
        t = types.StringLiteral(value)
    return t


# TODO: move to Numba
class BooleanLiteral(types.Literal, types.Boolean):
    def can_convert_to(self, typingctx, other):
        # similar to IntegerLiteral
        conv = typingctx.can_convert(self.literal_type, other)
        if conv is not None:
            return max(conv, types.Conversion.promote)


types.Literal.ctor_map[bool] = BooleanLiteral

register_model(BooleanLiteral)(models.BooleanModel)


@lower_cast(BooleanLiteral, types.Boolean)
def literal_bool_cast(context, builder, fromty, toty, val):
    lit = context.get_constant_generic(
        builder, fromty.literal_type, fromty.literal_value
    )
    return context.cast(builder, lit, fromty.literal_type, toty)


class ListLiteral(types.Literal):
    """class for literal lists, only used when Bodo forces an argument to be a literal
    list (e.g. in typing pass for groupby/join/sort_values).
    """


types.Literal.ctor_map[list] = ListLiteral
register_model(ListLiteral)(models.OpaqueModel)


@unbox(ListLiteral)
def unbox_list_literal(typ, obj, c):
    # A list literal is a dummy value
    return NativeValue(c.context.get_dummy_value())


# literal type for functions (to handle function arguments to map/apply methods)
# TODO: update when Numba's #4967 is merged
# similar to MakeFunctionLiteral
class FunctionLiteral(types.Literal, types.Opaque):
    """Literal type for function objects (i.e. pytypes.FunctionType)
    """


@typeof_impl.register(pytypes.FunctionType)
def typeof_function(val, c):
    """Assign literal type to constant functions that are not overloaded in numba.
    """
    if not numba.core.registry.cpu_target.typing_context._get_global_type(val):
        return FunctionLiteral(val)


register_model(FunctionLiteral)(models.OpaqueModel)


# dummy unbox to avoid errors when function is passed as argument
@unbox(FunctionLiteral)
def unbox_func_literal(typ, obj, c):
    return NativeValue(obj)


# groupby.agg() can take a constant dictionary with a UDF in values. Typer of Numba's
# typed.Dict trys to get the type of the UDF value, which is not possible. This hack
# makes a dummy type available to Numba so that type inference works.
types.MakeFunctionLiteral._literal_type_cache = types.MakeFunctionLiteral(lambda: 0)


# type used to pass metadata to type inference functions
# see untyped_pass.py and df.pivot_table()
class MetaType(types.Type):
    def __init__(self, meta):
        self.meta = meta
        super(MetaType, self).__init__("MetaType({})".format(meta))

    def can_convert_from(self, typingctx, other):
        return True

    @property
    def key(self):
        # XXX this is needed for _TypeMetaclass._intern to return the proper
        # cached instance in case meta is changed
        # (e.g. TestGroupBy -k pivot -k cross)
        return tuple(self.meta)


register_model(MetaType)(models.OpaqueModel)


# convert const tuple expressions or const list to tuple statically
def to_const_tuple(arrs):  # pragma: no cover
    return tuple(arrs)


@infer_global(to_const_tuple)
class ToConstTupleTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        arr = args[0]
        ret_typ = arr
        # XXX: returns a dummy type that should be fixed in series_pass
        if isinstance(arr, types.List):
            ret_typ = types.Tuple((arr.dtype,))
        return signature(ret_typ, arr)


# dummy lowerer
@lower_builtin(to_const_tuple, types.Any)
def lower_to_const_tuple(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


def is_literal_type(t):
    return (
        isinstance(t, (types.Literal, types.Omitted))
        or t == types.none  # None type is always literal since single value
        or isinstance(t, types.Dispatcher)
        or (isinstance(t, types.BaseTuple) and all(is_literal_type(v) for v in t.types))
    )


def get_literal_value(t):
    assert is_literal_type(t)
    if isinstance(t, types.Literal):
        return t.literal_value
    if isinstance(t, types.Omitted):
        return t.value
    if isinstance(t, types.BaseTuple):
        return tuple(get_literal_value(v) for v in t.types)
    if isinstance(t, types.Dispatcher):
        return t


def can_literalize_type(t):
    """return True if type 't' can have literal values
    """
    return t in (bodo.string_type, types.bool_) or isinstance(
        t, (types.Integer, types.List, types.SliceType)
    )


def scalar_to_array_type(t):
    """convert scalar type "t" to array of "t" values
    """
    if isinstance(t, (types.UnicodeType, types.StringLiteral)):
        return bodo.string_array_type

    # decimal arrays are a little different
    if isinstance(t, bodo.libs.decimal_arr_ext.Decimal128Type):
        precision = t.precision
        scale = t.scale
        return bodo.libs.decimal_arr_ext.DecimalArrayType(precision, scale)

    # datetime.date values are stored as date arrays
    if t == bodo.hiframes.datetime_date_ext.datetime_date_type:
        return bodo.hiframes.datetime_date_ext.datetime_date_array_type

    # datetime.timedelta values are stored as td64 arrays
    if t == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type:
        return types.Array(types.NPTimedelta("ns"), 1, "C")

    # datetime.datetime values are stored as dt64 arrays
    if t == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type:
        return types.Array(types.NPDatetime("ns"), 1, "C")

    # Timestamp values are stored as dt64 arrays
    if t == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type:
        return types.Array(types.NPDatetime("ns"), 1, "C")

    # TODO: make sure t is a Numpy dtype
    return types.Array(t, 1, "C")


# dummy empty itertools implementation to avoid typing errors for series str
# flatten case
@overload(itertools.chain, no_unliteral=True)
def chain_overload():
    return lambda: [0]


@register_jitable
def from_iterable_impl(A):  # pragma: no cover
    """Internal call to support itertools.chain.from_iterable().
    Untyped pass replaces itertools.chain.from_iterable() with this call since class
    methods are not supported in Numba's typing
    """
    return bodo.utils.conversion.flatten_array(bodo.utils.conversion.coerce_to_array(A))


def convert_tup_to_rec(val):  # pragma: no cover
    return val


@infer_global(convert_tup_to_rec)
class ConvertTupRecType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        in_dtype = args[0]
        out_dtype = in_dtype

        if isinstance(in_dtype, types.BaseTuple):
            np_dtype = np.dtype(",".join(str(t) for t in in_dtype.types), align=True)
            out_dtype = numba.np.numpy_support.from_dtype(np_dtype)

        return signature(out_dtype, in_dtype)


@lower_builtin(convert_tup_to_rec, types.Any)
def lower_convert_impl(context, builder, sig, args):
    (val,) = args
    in_typ = sig.args[0]
    rec_typ = sig.return_type

    if not isinstance(in_typ, types.BaseTuple):
        return impl_ret_borrowed(context, builder, sig.return_type, val)

    res = cgutils.alloca_once(builder, context.get_data_type(rec_typ))

    func_text = "def _set_rec(r, val):\n"
    for i in range(len(rec_typ.members)):
        func_text += "  r.f{} = val[{}]\n".format(i, i)

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    set_rec = loc_vars["_set_rec"]

    context.compile_internal(builder, set_rec, types.void(rec_typ, in_typ), [res, val])
    return impl_ret_new_ref(context, builder, sig.return_type, res)


def convert_rec_to_tup(val):  # pragma: no cover
    return val


@infer_global(convert_rec_to_tup)
class ConvertRecTupType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        in_dtype = args[0]
        out_dtype = in_dtype

        if isinstance(in_dtype, types.Record):
            out_dtype = types.Tuple([m[1] for m in in_dtype.members])

        return signature(out_dtype, in_dtype)


@lower_builtin(convert_rec_to_tup, types.Any)
def lower_convert_rec_tup_impl(context, builder, sig, args):
    (val,) = args
    rec_typ = sig.args[0]
    tup_typ = sig.return_type

    if not isinstance(rec_typ, types.Record):
        return impl_ret_borrowed(context, builder, sig.return_type, val)

    n_fields = len(rec_typ.members)

    func_text = "def _rec_to_tup(r):\n"
    func_text += "  return ({},)\n".format(
        ", ".join("r.f{}".format(i) for i in range(n_fields))
    )

    loc_vars = {}
    exec(func_text, {}, loc_vars)
    _rec_to_tup = loc_vars["_rec_to_tup"]

    res = context.compile_internal(builder, _rec_to_tup, tup_typ(rec_typ), [val])
    return impl_ret_borrowed(context, builder, sig.return_type, res)


def create_unsupported_overload(fname):
    """Create an overload for unsupported function 'fname' that raises BodoError
    """

    def overload_f(*a, **kws):
        raise BodoError("{} not supported yet".format(fname))

    return overload_f
