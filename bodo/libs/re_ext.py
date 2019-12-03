"""Support re module using object mode of Numba
"""
import re
import numba
from numba import types
from numba.extending import (
    box,
    unbox,
    register_model,
    models,
    NativeValue,
    overload,
    overload_method,
)


class RePatternType(types.Opaque):
    def __init__(self):
        super(RePatternType, self).__init__(name="RePatternType")


re_pattern_type = RePatternType()
types.re_pattern_type = re_pattern_type

register_model(RePatternType)(models.OpaqueModel)


@box(RePatternType)
def box_re_pattern(typ, val, c):
    # TODO: fix
    c.pyapi.incref(val)
    return val


@unbox(RePatternType)
def unbox_re_pattern(typ, obj, c):
    # TODO: fix
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(re.compile)
def re_compile_overload(pattern, flags=0):
    def _re_compile_impl(pattern, flags=0):
        with numba.objmode(pat="re_pattern_type"):
            pat = re.compile(pattern, flags)
        return pat

    return _re_compile_impl


@overload_method(RePatternType, "sub")
def re_sub_overload(p, repl, string, count=0):
    def _re_sub_impl(p, repl, string, count=0):
        with numba.objmode(out="unicode_type"):
            out = p.sub(repl, string, count)
        return out

    return _re_sub_impl
