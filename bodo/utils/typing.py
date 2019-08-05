"""
Helper functions to enable typing.
"""
from numba import types
from numba.extending import register_model, models, overload
from numba.typing.templates import (infer_global, AbstractTemplate,
    CallableTemplate)
from numba.typing import signature
from numba.targets.imputils import lower_builtin, impl_ret_borrowed
import bodo


def is_overload_none(val):
    return (val is None or val == types.none
            or getattr(val, 'value', False) is None)


def is_overload_true(val):
    return (val == True or val == bodo.utils.utils.BooleanLiteral(True)
            or getattr(val, 'value', False) is True)


def is_overload_false(val):
    return (val == False or val == bodo.utils.utils.BooleanLiteral(False)
            or getattr(val, 'value', True) is False)


def is_overload_zero(val):
    return (val == 0 or val == types.IntegerLiteral(0)
            or getattr(val, 'value', -1) == 0)


def is_overload_str(val, const):
    return (val == const or val == types.StringLiteral(const)
            or getattr(val, 'value', -1) == const)


def get_const_str_list(val):
    # 'ommited' case
    if getattr(val, 'value', None) is not None:
        return [val.value]
    # literal case
    if hasattr(val, 'literal_value'):
        return [val.literal_value]
    if hasattr(val, 'consts'):
        return val.consts


def get_overload_const_str(val):
    if isinstance(val, str):
        return val
    # 'ommited' case
    if getattr(val, 'value', None) is not None:
        assert isinstance(val.value, str)
        return val.value
    # literal case
    if isinstance(val, types.StringLiteral):
        assert isinstance(val.literal_value, str)
        return val.literal_value
    raise ValueError("{} not constant string".format(val))


def is_list_like_index_type(t):
    """Types that can be similar to list for indexing Arrays, Series, etc.
    Tuples are excluded due to indexing semantics.
    """
    from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType
    from bodo.hiframes.pd_series_ext import SeriesType
    # TODO: include datetimeindex/timedeltaindex?

    return (isinstance(t, types.List)
        or (isinstance(t, types.Array) and t.ndim == 1)
        or isinstance(t, (NumericIndexType, RangeIndexType))
        or isinstance(t, SeriesType))


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


# dummy function to enable series flattening, replaced in series_pass
def flatten_to_series(A):  # pragma: no cover
    return A


@infer_global(flatten_to_series)
class FlattenTyp(AbstractTemplate):
    def generic(self, args, kws):
        from bodo.hiframes.pd_series_ext import SeriesType
        assert not kws
        assert len(args) == 1
        # only list of lists supported
        assert isinstance(args[0], (types.List, SeriesType))
        l_dtype = args[0].dtype
        assert isinstance(l_dtype, types.List)
        dtype = l_dtype.dtype
        return signature(
            SeriesType(dtype), *bodo.utils.utils.unliteral_all(args))


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
        if isinstance(other, ConstList) and self.consts == other.consts:
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            reflected = self.reflected or other.reflected
            if dtype is not None:
                return ConstList(dtype, reflected)

    @property
    def key(self):
        return self.dtype, self.reflected, self.consts


@register_model(ConstList)
class ConstListModel(models.ListModel):
    def __init__(self, dmm, fe_type):
        l_type = types.List(fe_type.dtype)
        super(ConstListModel, self).__init__(dmm, l_type)


# add constant metadata to list or tuple type, see untyped_pass.py
def add_consts_to_type(a, *args):
    return a


@infer_global(add_consts_to_type)
class AddConstsTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ret_typ = args[0]
        # assert isinstance(ret_typ, types.List)  # TODO: other types
        # TODO: FloatLiteral e.g. test_fillna
        if all(isinstance(v, types.Literal) for v in args[1:]):
            consts = tuple(v.literal_value for v in args[1:])
            ret_typ = ConstList(ret_typ.dtype, consts)
        return signature(ret_typ, *args)


@lower_builtin(add_consts_to_type, types.VarArg(types.Any))
def lower_add_consts_to_type(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# dummy empty itertools implementation to avoid typing errors for series str
# flatten case
import itertools
@overload(itertools.chain)
def chain_overload():
    return lambda: [0]


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
