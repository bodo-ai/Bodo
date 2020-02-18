# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Helper functions to enable typing.
"""
import itertools
import types as pytypes
import numpy as np
import pandas as pd
import numba
from numba import types, cgutils
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
from numba.typing.templates import infer_global, AbstractTemplate, CallableTemplate
from numba.typing import signature
from numba.targets.imputils import lower_builtin, impl_ret_borrowed, impl_ret_new_ref
import bodo


# error used to avoid numba's error checking
class BodoError(BaseException):
    pass


class BodoNotConstError(Exception):
    """Indicates that a constant value is expected but non-constant is provided.
    Only used in partial typing pass to enable IR transformation. Inherits from regular
    Exception class for this purpose.
    """
    pass


def raise_const_error(msg):
    """raises an error indicating that a constant value is expected with the given
    message 'msg'.
    Raises BodoNotConstError during partial type inference, and BodoError otherwise.
    """
    if bodo.transforms.typing_pass.in_partial_typing:
        raise BodoNotConstError(msg)
    else:
        raise BodoError(msg)


class BodoWarning(Warning):
    """
    Warning class for Bodo-related potential issues such as prevention of
    parallelization by unsupported functions.
    """

    pass


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


def is_overload_constant_str_list(val):
    return (
        isinstance(val, (bodo.utils.typing.ConstList, bodo.utils.typing.ConstUniTuple))
        and isinstance(val.consts, tuple)
        and isinstance(val.consts[0], str)
    ) or (
        isinstance(val, types.BaseTuple)
        and all(isinstance(t, types.StringLiteral) for t in val.types)
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
    return isinstance(val, numba.types.List) and isinstance(val.dtype, types.Boolean)


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


def get_overload_const_str_len(val):
    if isinstance(val, str):
        return len(val)
    if isinstance(val, types.StringLiteral) and isinstance(val.literal_value, str):
        return len(val.literal_value)
    if isinstance(val, types.Omitted) and isinstance(val.value, str):
        return len(val.value)


def get_const_str_list(val):
    # 'ommited' case
    if getattr(val, "value", None) is not None:
        return [val.value]
    # literal case
    if hasattr(val, "literal_value"):
        return [val.literal_value]
    if hasattr(val, "consts"):
        return val.consts
    if isinstance(val, types.BaseTuple) and all(
        isinstance(t, types.StringLiteral) for t in val.types
    ):
        return [t.literal_value for t in val.types]


def get_overload_const_str(val):
    if isinstance(val, str):
        return val
    # 'ommited' case
    if getattr(val, "value", None) is not None:
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
    # 'ommited' case
    if getattr(val, "value", None) is not None:
        assert isinstance(val.value, int)
        return val.value
    # literal case
    if isinstance(val, types.IntegerLiteral):
        assert isinstance(val.literal_value, int)
        return val.literal_value
    raise ValueError("{} not constant integer".format(val))


def get_overload_const_func(val):
    """get constant function object or ir.Expr.make_function from function type
    """
    if isinstance(val, (types.MakeFunctionLiteral, bodo.utils.typing.FunctionLiteral)):
        return val.literal_value
    if isinstance(val, types.Dispatcher):
        return val.dispatcher.py_func
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
        return numba.numpy_support.from_dtype(np.dtype(d_str))
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
    from bodo.libs.int_arr_ext import IntegerArrayType

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
    # TODO: remove when none index is gone
    if t == types.none:
        t = bodo.hiframes.pd_index_ext.RangeIndexType(types.none)

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

    # TODO: remove when none index is gone
    if t == types.none:
        t = RangeIndexType(types.none)

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


# literal type for functions (to handle function arguments to map/apply methods)
# TODO: update when Numba's #4967 is merged
# similar to MakeFunctionLiteral
class FunctionLiteral(types.Literal, types.Opaque):
    """Literal type for function objects (i.e. pytypes.FunctionType)
    """

    pass


@typeof_impl.register(pytypes.FunctionType)
def typeof_function(val, c):
    """Assign literal type to constant functions that are not overloaded in numba.
    """
    if not numba.targets.registry.cpu_target.typing_context._get_global_type(val):
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


# Type used to add constant values to constant lists to enable typing
class ConstList(types.List):
    def __init__(self, dtype, consts):
        dtype = types.unliteral(dtype)
        self.dtype = dtype
        self.reflected = False
        self.consts = consts
        cls_name = "list[{}]".format(consts)
        name = "%s(%s)" % (cls_name, self.dtype)
        super(types.List, self).__init__(name=name)

    def copy(self, dtype=None, reflected=None):
        if dtype is None:
            dtype = self.dtype
        return ConstList(dtype, self.consts)

    def unify(self, typingctx, other):
        if isinstance(other, types.List):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            reflected = self.reflected or other.reflected
            if dtype is not None:
                if isinstance(other, ConstList) and self.consts == other.consts:
                    return ConstList(dtype, reflected)
                else:
                    return types.List(dtype, reflected)

    @property
    def key(self):
        return self.dtype, self.reflected, self.consts


@register_model(ConstList)
class ConstListModel(models.ListModel):
    def __init__(self, dmm, fe_type):
        l_type = types.List(fe_type.dtype)
        super(ConstListModel, self).__init__(dmm, l_type)


def is_literal_type(t):
    return (
        isinstance(t, (types.Literal, types.Omitted))
        or isinstance(t, types.Dispatcher)
        or (isinstance(t, types.Tuple) and all(is_literal_type(v) for v in t.types))
    )


def get_literal_value(t):
    assert is_literal_type(t)
    if isinstance(t, types.Literal):
        return t.literal_value
    if isinstance(t, types.Omitted):
        return t.value
    if isinstance(t, types.Tuple):
        return tuple(get_literal_value(v) for v in t.types)
    if isinstance(t, types.Dispatcher):
        return t


# add constant metadata to list or tuple type, see untyped_pass.py
def add_consts_to_type(a, *args):
    return a


@infer_global(add_consts_to_type)
class AddConstsTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ret_typ = args[0]
        # TODO: FloatLiteral e.g. test_fillna
        if all(is_literal_type(v) for v in args[1:]):
            consts = tuple(get_literal_value(v) for v in args[1:])
            if isinstance(ret_typ, types.DictType):
                ret_typ = ConstDictType(ret_typ.key_type, ret_typ.value_type, consts)
            elif isinstance(ret_typ, types.UniTuple):
                assert ret_typ.count == len(consts)
                ret_typ = ConstUniTuple(ret_typ.dtype, ret_typ.count, consts)
            else:
                ret_typ = ConstList(ret_typ.dtype, consts)
        return signature(ret_typ, *args)


@lower_builtin(add_consts_to_type, types.VarArg(types.Any))
def lower_add_consts_to_type(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


class ConstDictType(types.DictType):
    """Dictionary type with constant keys and values
    """

    def __init__(self, keyty, valty, consts):
        keyty = types.unliteral(keyty)
        valty = types.unliteral(valty)
        self.key_type = keyty
        self.value_type = valty
        self.keyvalue_type = types.Tuple([keyty, valty])
        self.consts = consts
        name = "{}[{},{}][{}]".format(self.__class__.__name__, keyty, valty, consts)
        super(types.DictType, self).__init__(name)


@register_model(ConstDictType)
class ConstDictModel(numba.dictobject.DictModel):
    def __init__(self, dmm, fe_type):
        l_type = types.DictType(fe_type.key_type, fe_type.value_type)
        super(ConstDictModel, self).__init__(dmm, l_type)


# Type used to add constant values to constant uniform tuples to enable typing of calls
# such as df.merge() with tuple of str as "on" argument
class ConstUniTuple(types.UniTuple):
    def __init__(self, dtype, count, consts):
        dtype = types.unliteral(dtype)
        self.dtype = dtype
        self.count = count
        self.consts = consts
        cls_name = "tuple[{}]".format(consts)
        name = "%s(%s x %d)" % (cls_name, self.dtype, self.count)
        super(types.UniTuple, self).__init__(name=name)

    def copy(self):
        return ConstUniTuple(self.dtype, self.count, self.consts)

    def unify(self, typingctx, other):
        if isinstance(other, ConstUniTuple) and self.consts == other.consts:
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            if dtype is not None:
                return ConstUniTuple(dtype, self.count, self.consts)

    @property
    def key(self):
        return self.dtype, self.count, self.consts


@register_model(ConstUniTuple)
class ConstUniTupleModel(models.UniTupleModel):
    def __init__(self, dmm, fe_type):
        l_type = types.UniTuple(fe_type.dtype, fe_type.count)
        super(ConstUniTupleModel, self).__init__(dmm, l_type)


# dummy empty itertools implementation to avoid typing errors for series str
# flatten case
@overload(itertools.chain)
def chain_overload():
    return lambda: [0]


@register_jitable
def from_iterable_impl(A):  # pragma: no cover
    """Internal call to support itertools.chain.from_iterable().
    Untyped pass replaces itertools.chain.from_iterable() with this call since class
    methods are not supported in Numba's typing
    """
    return bodo.utils.conversion.flatten_array(bodo.utils.conversion.coerce_to_array(A))


# taken from numba/typing/listdecl.py
@infer_global(sorted)
class SortedBuiltinLambda(CallableTemplate):
    def generic(self):
        # TODO: reverse=None
        def typer(iterable, key=None):
            if not isinstance(iterable, types.IterableType):
                return
            return types.List(iterable.iterator_type.yield_type)

        return typer


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
            out_dtype = numba.numpy_support.from_dtype(np_dtype)

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


def convert_rec_to_tup(val):
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
