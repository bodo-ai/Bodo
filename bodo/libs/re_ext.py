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
    overload_attribute,
    intrinsic,
    typeof_impl,
    lower_cast,
    lower_builtin,
)
from bodo.libs.str_ext import string_type
from llvmlite import ir as lir


# re.Pattern and re.Match classes are exposed starting Python 3.7, so we get the type
# class using type() call to support <=3.6
_dummy_pat = "_BODO_DUMMY_PATTERN_"
Pattern = type(re.compile(_dummy_pat))
Match = type(re.match(_dummy_pat, _dummy_pat))


class RePatternType(types.Opaque):
    def __init__(self):
        super(RePatternType, self).__init__(name="RePatternType")


re_pattern_type = RePatternType()
types.re_pattern_type = re_pattern_type

register_model(RePatternType)(models.OpaqueModel)


@typeof_impl.register(Pattern)
def typeof_re_pattern(val, c):
    return re_pattern_type


@box(RePatternType)
def box_re_pattern(typ, val, c):
    # NOTE: we can't just let Python steal a reference since boxing can happen at any
    # point and even in a loop, which can make refcount invalid.
    # see implementation of str.contains and test_contains_regex
    # TODO: investigate refcount semantics of boxing in Numba when variable is returned
    # from function versus not returned
    c.pyapi.incref(val)
    return val


@unbox(RePatternType)
def unbox_re_pattern(typ, obj, c):
    # borrow a reference from Python
    c.pyapi.incref(obj)
    return NativeValue(obj)


# data type for storing re.Match objects or None
# handling None is required since functions like re.seach() return either Match object
# or None (when there is no match)
class ReMatchType(types.Type):
    def __init__(self):
        super(ReMatchType, self).__init__(name="ReMatchType")


re_match_type = ReMatchType()
# TODO: avoid setting attributes to "types" when object mode can handle actual types
types.re_match_type = re_match_type
types.list_str_type = types.List(string_type)


register_model(ReMatchType)(models.OpaqueModel)


@typeof_impl.register(Match)
def typeof_re_match(val, c):
    return re_match_type


@box(ReMatchType)
def box_re_match(typ, val, c):
    c.pyapi.incref(val)
    return val


@unbox(ReMatchType)
def unbox_re_match(typ, obj, c):
    # borrow a reference from Python
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
    is_none = builder.icmp_signed("==", val, pyapi.borrow_none())
    with builder.if_then(is_none):
        builder.store(context.get_constant(types.bool_, False), out)
    return builder.load(out)


@lower_builtin(operator.is_, ReMatchType, types.NoneType)
def lower_match_is_none(context, builder, sig, args):
    """
    implementation for "match is None"
    """
    match = args[0]
    # reuse cast to bool implementation
    return builder.not_(
        cast_match_obj_bool(context, builder, sig.args[0], sig.args[1], match)
    )


@overload(re.search)
def overload_re_search(pattern, string, flags=0):
    def _re_search_impl(pattern, string, flags=0):  # pragma: no cover
        with numba.objmode(m="re_match_type"):
            m = re.search(pattern, string, flags)
        return m

    return _re_search_impl


@overload(re.match)
def overload_re_match(pattern, string, flags=0):
    def _re_match_impl(pattern, string, flags=0):  # pragma: no cover
        with numba.objmode(m="re_match_type"):
            m = re.match(pattern, string, flags)
        return m

    return _re_match_impl


@overload(re.fullmatch)
def overload_re_fullmatch(pattern, string, flags=0):
    def _re_fullmatch_impl(pattern, string, flags=0):  # pragma: no cover
        with numba.objmode(m="re_match_type"):
            m = re.fullmatch(pattern, string, flags)
        return m

    return _re_fullmatch_impl


@overload(re.split)
def overload_re_split(pattern, string, maxsplit=0, flags=0):
    def _re_split_impl(pattern, string, maxsplit=0, flags=0):  # pragma: no cover
        with numba.objmode(m="list_str_type"):
            m = re.split(pattern, string, maxsplit, flags)
        return m

    return _re_split_impl


@overload(re.findall)
def overload_re_findall(pattern, string, flags=0):
    def _re_findall_impl(pattern, string, flags=0):  # pragma: no cover
        with numba.objmode(m="list_str_type"):
            m = re.findall(pattern, string, flags)
        return m

    return _re_findall_impl


@overload(re.sub)
def overload_re_sub(pattern, repl, string, count=0, flags=0):
    def _re_sub_impl(pattern, repl, string, count=0, flags=0):  # pragma: no cover
        with numba.objmode(m="unicode_type"):
            m = re.sub(pattern, repl, string, count, flags)
        return m

    return _re_sub_impl


@overload(re.subn)
def overload_re_subn(pattern, repl, string, count=0, flags=0):
    def _re_subn_impl(pattern, repl, string, count=0, flags=0):  # pragma: no cover
        with numba.objmode(m="unicode_type", s="int64"):
            m, s = re.subn(pattern, repl, string, count, flags)
        return m, s

    return _re_subn_impl


@overload(re.escape)
def overload_re_escape(pattern):
    def _re_escape_impl(pattern):  # pragma: no cover
        with numba.objmode(m="unicode_type"):
            m = re.escape(pattern)
        return m

    return _re_escape_impl


@overload(re.purge)
def overload_re_purge():
    def _re_purge_impl():  # pragma: no cover
        with numba.objmode():
            re.purge()
        return

    return _re_purge_impl


@overload(re.compile)
def re_compile_overload(pattern, flags=0):
    def _re_compile_impl(pattern, flags=0):  # pragma: no cover
        with numba.objmode(pat="re_pattern_type"):
            pat = re.compile(pattern, flags)
        return pat

    return _re_compile_impl


@overload_method(RePatternType, "search")
def overload_pat_search(p, string, pos=0, endpos=9223372036854775807):
    def _pat_search_impl(
        p, string, pos=0, endpos=9223372036854775807
    ):  # pragma: no cover
        with numba.objmode(m="re_match_type"):
            m = p.search(string, pos, endpos)
        return m

    return _pat_search_impl


@overload_method(RePatternType, "match")
def overload_pat_match(p, string, pos=0, endpos=9223372036854775807):
    def _pat_match_impl(
        p, string, pos=0, endpos=9223372036854775807
    ):  # pragma: no cover
        with numba.objmode(m="re_match_type"):
            m = p.match(string, pos, endpos)
        return m

    return _pat_match_impl


@overload_method(RePatternType, "fullmatch")
def overload_pat_fullmatch(p, string, pos=0, endpos=9223372036854775807):
    def _pat_fullmatch_impl(
        p, string, pos=0, endpos=9223372036854775807
    ):  # pragma: no cover
        with numba.objmode(m="re_match_type"):
            m = p.fullmatch(string, pos, endpos)
        return m

    return _pat_fullmatch_impl


@overload_method(RePatternType, "split")
def overload_pat_split(pattern, string, maxsplit=0):
    def _pat_split_impl(pattern, string, maxsplit=0):  # pragma: no cover
        with numba.objmode(m="list_str_type"):
            m = pattern.split(string, maxsplit)
        return m

    return _pat_split_impl


@overload_method(RePatternType, "findall")
def overload_pat_findall(p, string, pos=0, endpos=9223372036854775807):
    def _pat_findall_impl(
        p, string, pos=0, endpos=9223372036854775807
    ):  # pragma: no cover
        with numba.objmode(m="list_str_type"):
            m = p.findall(string, pos, endpos)
        return m

    return _pat_findall_impl


@overload_method(RePatternType, "sub")
def re_sub_overload(p, repl, string, count=0):
    def _re_sub_impl(p, repl, string, count=0):  # pragma: no cover
        with numba.objmode(out="unicode_type"):
            out = p.sub(repl, string, count)
        return out

    return _re_sub_impl


@overload_method(RePatternType, "subn")
def re_subn_overload(p, repl, string, count=0):
    def _re_subn_impl(p, repl, string, count=0):  # pragma: no cover
        with numba.objmode(out="unicode_type", s="int64"):
            out, s = p.subn(repl, string, count)
        return out, s

    return _re_subn_impl


@overload_attribute(RePatternType, "flags")
def overload_pattern_flags(p):
    def _pat_flags_impl(p):  # pragma: no cover
        with numba.objmode(flags="int64"):
            flags = p.flags
        return flags

    return _pat_flags_impl


@overload_attribute(RePatternType, "groups")
def overload_pattern_groups(p):
    def _pat_groups_impl(p):  # pragma: no cover
        with numba.objmode(groups="int64"):
            groups = p.groups
        return groups

    return _pat_groups_impl


@overload_attribute(RePatternType, "groupindex")
def overload_pattern_groupindex(p):
    """overload Pattern.groupindex. Python returns mappingproxy object but Bodo returns
    a Numba TypedDict with essentially the same functionality
    """
    types.dict_string_int = types.DictType(string_type, types.int64)

    def _pat_groupindex_impl(p):  # pragma: no cover
        with numba.objmode(d="dict_string_int"):
            groupindex = dict(p.groupindex)
            d = numba.typed.Dict.empty(
                key_type=numba.types.unicode_type, value_type=numba.int64
            )
            d.update(groupindex)
        return d

    return _pat_groupindex_impl


@overload_attribute(RePatternType, "pattern")
def overload_pattern_pattern(p):
    def _pat_pattern_impl(p):  # pragma: no cover
        with numba.objmode(pattern="unicode_type"):
            pattern = p.pattern
        return pattern

    return _pat_pattern_impl


@overload_method(ReMatchType, "expand")
def overload_match_expand(m, template):
    def _match_expand_impl(m, template):  # pragma: no cover
        with numba.objmode(out="unicode_type"):
            out = m.expand(template)
        return out

    return _match_expand_impl


@overload_method(ReMatchType, "group")
def overload_match_group(m, *args):
    # TODO: support cases where a group is not matched and None should be returned
    # for example: re.match(r"(\w+)? (\w+) (\w+)", " words word")
    # NOTE: using *args in implementation throws an error in Numba lowering
    # TODO: use simpler implementation when Numba is fixed
    # def _match_group_impl(m, *args):
    #     with numba.objmode(out="unicode_type"):
    #         out = m.group(*args)
    #     return out

    # instead of the argument types, Numba passes a tuple with a StarArgTuple type at
    # some point during lowering
    if len(args) == 1 and isinstance(
            args[0], (types.StarArgTuple, types.StarArgUniTuple)):
        args = args[0].types

    # no argument case returns a string
    if len(args) == 0:

        def _match_group_impl_zero(m, *args):  # pragma: no cover
            with numba.objmode(out="unicode_type"):
                out = m.group()
            return out

        return _match_group_impl_zero

    # one argument case returns a string
    if len(args) == 1:

        def _match_group_impl_one(m, *args):  # pragma: no cover
            group1 = args[0]
            with numba.objmode(out="unicode_type"):
                out = m.group(group1)
            return out

        return _match_group_impl_one

    # multi-argument case returns a tuple of strings
    # TODO: avoid setting attributes to "types" when object mode can handle actual types
    type_name = "tuple_str_{}".format(len(args))
    setattr(types, type_name, types.Tuple([string_type] * len(args)))
    arg_names = ", ".join("group{}".format(i + 1) for i in range(len(args)))
    func_text = "def _match_group_impl(m, *args):\n"
    func_text += "  ({}) = args\n".format(arg_names)
    func_text += "  with numba.objmode(out='{}'):\n".format(type_name)
    func_text += "    out = m.group({})\n".format(arg_names)
    func_text += "  return out\n"

    loc_vars = {}
    exec(func_text, {"numba": numba}, loc_vars)
    impl = loc_vars["_match_group_impl"]
    return impl


@overload(operator.getitem)
def overload_match_getitem(m, ind):
    if m == re_match_type:
        return lambda m, ind: m.group(ind)


@overload_method(ReMatchType, "groups")
def overload_match_groups(m, default=None):
    # TODO: support cases where a group is not matched and None should be returned
    # for example: re.match(r"(\w+)? (\w+) (\w+)", " words word")

    # NOTE: Python returns tuple of strings, but we don't know the length in advance
    # which makes it not compilable. We return a list which is similar to tuple
    def _match_groups_impl(m, default=None):  # pragma: no cover
        with numba.objmode(out="list_str_type"):
            out = list(m.groups(default))
        return out

    return _match_groups_impl


@overload_method(ReMatchType, "groupdict")
def overload_match_groupdict(m, default=None):
    # TODO: support cases where a group is not matched and None should be returned
    # for example: re.match(r"(?P<AA>\w+)? (\w+) (\w+)", " words word")

    types.dict_string_string = types.DictType(string_type, string_type)

    def _match_groupdict_impl(m, default=None):  # pragma: no cover
        with numba.objmode(d="dict_string_string"):
            out = m.groupdict(default)
            d = numba.typed.Dict.empty(
                key_type=numba.types.unicode_type, value_type=numba.types.unicode_type
            )
            d.update(out)
        return d

    return _match_groupdict_impl


@overload_method(ReMatchType, "start")
def overload_match_start(m, group=0):
    def _match_start_impl(m, group=0):  # pragma: no cover
        with numba.objmode(out="int64"):
            out = m.start(group)
        return out

    return _match_start_impl


@overload_method(ReMatchType, "end")
def overload_match_end(m, group=0):
    def _match_end_impl(m, group=0):  # pragma: no cover
        with numba.objmode(out="int64"):
            out = m.end(group)
        return out

    return _match_end_impl


@overload_method(ReMatchType, "span")
def overload_match_span(m, group=0):

    # span() returns a tuple of int
    types.tuple_int64_2 = types.Tuple([types.int64, types.int64])

    def _match_span_impl(m, group=0):  # pragma: no cover
        with numba.objmode(out="tuple_int64_2"):
            out = m.span(group)
        return out

    return _match_span_impl


@overload_attribute(ReMatchType, "pos")
def overload_match_pos(p):
    def _match_pos_impl(p):  # pragma: no cover
        with numba.objmode(pos="int64"):
            pos = p.pos
        return pos

    return _match_pos_impl


@overload_attribute(ReMatchType, "endpos")
def overload_match_endpos(p):
    def _match_endpos_impl(p):  # pragma: no cover
        with numba.objmode(endpos="int64"):
            endpos = p.endpos
        return endpos

    return _match_endpos_impl


@overload_attribute(ReMatchType, "lastindex")
def overload_match_lastindex(p):
    # TODO: support returning None if no group was matched
    def _match_lastindex_impl(p):  # pragma: no cover
        with numba.objmode(lastindex="int64"):
            lastindex = p.lastindex
        return lastindex

    return _match_lastindex_impl


@overload_attribute(ReMatchType, "lastgroup")
def overload_match_lastgroup(p):
    # TODO: support returning None if last group didn't have a name or no group was
    # matched
    def _match_lastgroup_impl(p):  # pragma: no cover
        with numba.objmode(lastgroup="unicode_type"):
            lastgroup = p.lastgroup
        return lastgroup

    return _match_lastgroup_impl


@overload_attribute(ReMatchType, "re")
def overload_match_re(m):
    def _match_re_impl(m):  # pragma: no cover
        with numba.objmode(m_re="re_pattern_type"):
            m_re = m.re
        return m_re

    return _match_re_impl


@overload_attribute(ReMatchType, "string")
def overload_match_string(m):
    def _match_string_impl(m):  # pragma: no cover
        with numba.objmode(out="unicode_type"):
            out = m.string
        return out

    return _match_string_impl
