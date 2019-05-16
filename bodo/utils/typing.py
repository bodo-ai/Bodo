"""
Helper functions to enable typing.
"""
from numba import types
from numba.extending import register_model, models
from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature

from bodo.utils.utils import unliteral_all


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
        return signature(SeriesType(dtype), *unliteral_all(args))
