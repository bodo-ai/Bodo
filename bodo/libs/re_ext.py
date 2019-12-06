"""Support re module using object mode of Numba
"""
import operator
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
    lower_cast,
)
from bodo.libs.str_ext import string_type
from llvmlite import ir as lir


class RePatternType(types.Opaque):
    def __init__(self):
        super(RePatternType, self).__init__(name="RePatternType")


re_pattern_type = RePatternType()
types.re_pattern_type = re_pattern_type

register_model(RePatternType)(models.OpaqueModel)


@typeof_impl.register(re.Pattern)
def typeof_re_pattern(val, c):
    return re_pattern_type


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
types.list_str_type = types.List(string_type)


register_model(ReMatchType)(models.OpaqueModel)


@typeof_impl.register(re.Match)
def typeof_re_match(val, c):
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


# implement casting to boolean to support conditions like "if match:" which are
# commonly used to see if there are matches.
# NOTE: numba may need operator.truth and bool() at some point
@lower_cast(ReMatchType, types.Boolean)
def cast_match_obj_bool(context, builder, fromty, toty, val):
    """cast match object (which could be None also) to boolean.
    Output is False if match object is actually a None object, otherwise True.
    """
    out = cgutils.alloca_once_value(builder, context.get_constant(types.bool_, True))
    pyapi = context.get_python_api(builder)
    # check for None, equality is enough for None since it is singleton
    is_none = builder.icmp_signed('==', val, pyapi.borrow_none())
    with builder.if_then(is_none):
        builder.store(context.get_constant(types.bool_, False), out)
    return builder.load(out)


@overload(re.search)
def overload_re_search(pattern, string, flags=0):
    def _re_search_impl(pattern, string, flags=0):
        with numba.objmode(m="re_match_type"):
            m = re.search(pattern, string, flags)
        return m
    return _re_search_impl


@overload(re.match)
def overload_re_match(pattern, string, flags=0):
    def _re_match_impl(pattern, string, flags=0):
        with numba.objmode(m="re_match_type"):
            m = re.match(pattern, string, flags)
        return m
    return _re_match_impl


@overload(re.fullmatch)
def overload_re_fullmatch(pattern, string, flags=0):
    def _re_fullmatch_impl(pattern, string, flags=0):
        with numba.objmode(m="re_match_type"):
            m = re.fullmatch(pattern, string, flags)
        return m
    return _re_fullmatch_impl


@overload(re.split)
def overload_re_split(pattern, string, maxsplit=0, flags=0):
    def _re_split_impl(pattern, string, maxsplit=0, flags=0):
        with numba.objmode(m="list_str_type"):
            m = re.split(pattern, string, maxsplit, flags)
        return m
    return _re_split_impl


@overload(re.findall)
def overload_re_findall(pattern, string, flags=0):
    def _re_findall_impl(pattern, string, flags=0):
        with numba.objmode(m="list_str_type"):
            m = re.findall(pattern, string, flags)
        return m
    return _re_findall_impl


@overload(re.sub)
def overload_re_sub(pattern, repl, string, count=0, flags=0):
    def _re_sub_impl(pattern, repl, string, count=0, flags=0):
        with numba.objmode(m="unicode_type"):
            m = re.sub(pattern, repl, string, count, flags)
        return m
    return _re_sub_impl


@overload(re.subn)
def overload_re_subn(pattern, repl, string, count=0, flags=0):
    def _re_subn_impl(pattern, repl, string, count=0, flags=0):
        with numba.objmode(m="unicode_type", s="int64"):
            m, s = re.subn(pattern, repl, string, count, flags)
        return m, s
    return _re_subn_impl


@overload(re.escape)
def overload_re_escape(pattern):
    def _re_escape_impl(pattern):
        with numba.objmode(m="unicode_type"):
            m = re.escape(pattern)
        return m
    return _re_escape_impl


@overload(re.purge)
def overload_re_purge():
    def _re_purge_impl():
        with numba.objmode():
            re.purge()
        return
    return _re_purge_impl


@overload(re.compile)
def re_compile_overload(pattern, flags=0):
    def _re_compile_impl(pattern, flags=0):
        with numba.objmode(pat="re_pattern_type"):
            pat = re.compile(pattern, flags)
        return pat

    return _re_compile_impl


@overload_method(RePatternType, "search")
def overload_pat_search(p, string, pos=0, endpos=9223372036854775807):
    def _pat_search_impl(p, string, pos=0, endpos=9223372036854775807):
        with numba.objmode(m="re_match_type"):
            m = p.search(string, pos, endpos)
        return m
    return _pat_search_impl


@overload_method(RePatternType, "sub")
def re_sub_overload(p, repl, string, count=0):
    def _re_sub_impl(p, repl, string, count=0):
        with numba.objmode(out="unicode_type"):
            out = p.sub(repl, string, count)
        return out

    return _re_sub_impl
