"""Support re module using object mode of Numba
"""
import re
import numba
from numba import types, cgutils
from numba.extending import (
    box,
    unbox,
    register_model,
    models,
    NativeValue,
    overload,
    overload_method,
    intrinsic,
    typeof_impl,
)
from llvmlite import ir as lir


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


# data type for storing re.Match objects or None
# handling None is required since functions like re.seach() return either Match object
# or None (when there is no match)
class ReMatchType(types.Type):
    def __init__(self):
        super(ReMatchType, self).__init__(name="ReMatchType")


re_match_type = ReMatchType()
types.re_match_type = re_match_type


register_model(ReMatchType)(models.OpaqueModel)


@typeof_impl.register(re.Match)
def typeof_pd_dataframe(val, c):
    return re_match_type


@box(ReMatchType)
def box_re_match(typ, val, c):
    # TODO: fix refcounting
    c.pyapi.incref(val)
    return val


@unbox(ReMatchType)
def unbox_re_match(typ, obj, c):
    # TODO: fix refcounting
    c.pyapi.incref(obj)
    return NativeValue(obj)


@overload(re.search)
def overload_re_search(pattern, string, flags=0):
    def _re_search_impl(pattern, string, flags=0):
        with numba.objmode(m="re_match_type"):
            m = re.search(pattern, string, flags)
        return m
    return _re_search_impl


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
