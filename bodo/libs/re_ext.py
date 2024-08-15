"""Support re module using object mode of Numba
"""
import operator
import re

import numba
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed, lower_constant
from numba.core.typing.templates import (
    ConcreteTemplate,
    infer_global,
    signature,
)
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_builtin,
    lower_cast,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.libs.str_ext import re_escape_len, re_escape_with_output, string_type
from bodo.utils.typing import (
    BodoError,
    gen_objmode_func_overload,
    gen_objmode_method_overload,
    get_overload_const_str,
    is_overload_constant_str,
)


class RePatternType(types.Opaque):
    def __init__(self, pat_const=None):
        # keep pattern string if it is a constant value
        # useful for findall() to handle multi-group case
        self.pat_const = pat_const
        super(RePatternType, self).__init__(name=f"RePatternType({pat_const})")

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


re_pattern_type = RePatternType()
types.re_pattern_type = re_pattern_type

register_model(RePatternType)(models.OpaqueModel)


@typeof_impl.register(re.Pattern)
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


@lower_constant(RePatternType)
def pattern_constant(context, builder, ty, pyval):
    """
    get LLVM constant by serializing the Python value.
    """
    pyapi = context.get_python_api(builder)
    return pyapi.unserialize(pyapi.serialize_object(pyval))


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


@typeof_impl.register(re.Match)
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


# TODO(ehsan): remove RegexFlagsType when we have IntFlag support since RegexFlags
# is a subclass of IntFlag
# https://bodo.atlassian.net/browse/BE-1791
# https://github.com/python/cpython/blob/86f42851c050d756679ae7797f8720adaef381c4/Lib/re.py#L147
# https://docs.python.org/3/library/enum.html#enum.IntFlag
class RegexFlagsType(types.Type):
    """Type for re.RegexFlags values like re.IGNORECASE"""

    def __init__(self):
        super().__init__(name="RegexFlagsType()")


regex_flags_type = RegexFlagsType()
types.regex_flags_type = regex_flags_type


@register_model(RegexFlagsType)
class RegexFlagsModel(models.ProxyModel):
    """The underlying data model is just int64"""

    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type)
        self._proxied_model = dmm.lookup(types.int64)


@typeof_impl.register(re.RegexFlag)
def typeof_re_flags(val, c):
    return regex_flags_type


@box(RegexFlagsType)
def box_regex_flag(typ, val, c):
    """box RegexFlagsType by calling re.RegexFlag class with the value object"""
    valobj = c.box(types.int64, val)
    cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(re.RegexFlag))
    return c.pyapi.call_function_objargs(cls_obj, (valobj,))


@unbox(RegexFlagsType)
def unbox_regex_flag(typ, obj, c):
    """unbox RegexFlags by getting its integer 'value' attribute"""
    valobj = c.pyapi.object_getattr_string(obj, "value")
    return c.unbox(types.int64, valobj)


@lower_constant(RegexFlagsType)
def regex_flags_constant(context, builder, ty, pyval):
    """
    get LLVM constant by getting its integer 'value' attribute
    """
    return context.get_constant_generic(builder, types.int64, pyval.value)


@infer_global(operator.or_)
class RegexFlagsOR(ConcreteTemplate):
    """
    'or' operation for RegexFlags (combines flags)
    """

    key = operator.or_
    cases = [signature(regex_flags_type, regex_flags_type, regex_flags_type)]


@lower_builtin(operator.or_, regex_flags_type, regex_flags_type)
def re_flags_or_impl(context, builder, sig, args):
    """simply 'or' the int values"""
    return builder.or_(args[0], args[1])


# implement casting to boolean to support conditions like "if match:" which are
# commonly used to see if there are matches.
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


@intrinsic
def match_obj_is_none(typingctx, match_typ):
    assert match_typ == re_match_type

    def codegen(context, builder, sig, args):
        return cast_match_obj_bool(
            context, builder, re_match_type, types.bool_, args[0]
        )

    return types.bool_(match_typ), codegen


@overload(bool)
def overload_bool_re_match(val):
    if val == re_match_type:
        return lambda val: match_obj_is_none(val)  # pragma: no cover


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


gen_objmode_func_overload(re.search, "re_match_type")
gen_objmode_func_overload(re.match, "re_match_type")
gen_objmode_func_overload(re.fullmatch, "re_match_type")
gen_objmode_func_overload(re.split, "list_str_type")
gen_objmode_func_overload(re.sub, "unicode_type")


@overload(re.escape)
def overload_re_escape(pattern):
    """Implementation of re.escape that works by calling C++
    kernels equivalent to the Cpython re.escape implementation.
    All allocations are done via JIT/Numpy.

    Args:
        pattern (types.unicode_type): String that needs to be escaped.

    Returns: The escaped pattern.
    """

    def impl(pattern):  # pragma: no cover
        new_length = re_escape_len(pattern)
        # Allocate the output string.
        out_str = numba.cpython.unicode._empty_string(
            pattern._kind, new_length, pattern._is_ascii
        )
        re_escape_with_output(pattern, out_str)
        return out_str

    return impl


@overload(re.findall, no_unliteral=True)
def overload_re_findall(pattern, string, flags=0):
    # reusing the Pattern.findall() implementation to check for non-constant pattern
    # with multiple groups properly (which causes typing issues)
    def _re_findall_impl(pattern, string, flags=0):  # pragma: no cover
        p = re.compile(pattern, flags)
        return p.findall(string)

    return _re_findall_impl


@overload(re.subn, no_unliteral=True)
def overload_re_subn(pattern, repl, string, count=0, flags=0):
    def _re_subn_impl(pattern, repl, string, count=0, flags=0):  # pragma: no cover
        with bodo.objmode(m="unicode_type", s="int64"):
            m, s = re.subn(pattern, repl, string, count, flags)
        return m, s

    return _re_subn_impl


@overload(re.purge, no_unliteral=True)
def overload_re_purge():
    def _re_purge_impl():  # pragma: no cover
        with bodo.objmode():
            re.purge()
        return

    return _re_purge_impl


@intrinsic(prefer_literal=True)
def init_const_pattern(typingctx, pat, pat_const):
    """dummy intrinsic to add constant pattern string to Pattern data type"""
    pat_const_str = get_overload_const_str(pat_const)

    def codegen(context, builder, sig, args):
        return impl_ret_borrowed(context, builder, sig.return_type, args[0])

    return RePatternType(pat_const_str)(pat, pat_const), codegen


@overload(re.compile, no_unliteral=True)
def re_compile_overload(pattern, flags=0):
    # if pattern string is constant, add it to data type to enable findall()
    if is_overload_constant_str(pattern):
        pat_const = get_overload_const_str(pattern)

        def _re_compile_const_impl(pattern, flags=0):  # pragma: no cover
            with bodo.objmode(pat="re_pattern_type"):
                pat = re.compile(pattern, flags)
            return init_const_pattern(pat, pat_const)

        return _re_compile_const_impl

    def _re_compile_impl(pattern, flags=0):  # pragma: no cover
        with bodo.objmode(pat="re_pattern_type"):
            pat = re.compile(pattern, flags)
        return pat

    return _re_compile_impl


gen_objmode_method_overload(RePatternType, "search", re.Pattern.search, "re_match_type")
gen_objmode_method_overload(RePatternType, "match", re.Pattern.match, "re_match_type")
gen_objmode_method_overload(
    RePatternType, "fullmatch", re.Pattern.fullmatch, "re_match_type"
)
gen_objmode_method_overload(RePatternType, "split", re.Pattern.split, "list_str_type")
gen_objmode_method_overload(RePatternType, "sub", re.Pattern.sub, "unicode_type")


@overload_method(RePatternType, "findall", no_unliteral=True)
def overload_pat_findall(p, string, pos=0, endpos=9223372036854775807):
    # if pattern string is constant, we can handle multi-group case since we know the
    # number of groups here
    if p.pat_const:
        n_groups = re.compile(p.pat_const).groups
        typ = types.List(string_type)
        if n_groups > 1:
            typ = types.List(types.Tuple([string_type] * n_groups))

        def _pat_findall_const_impl(
            p, string, pos=0, endpos=9223372036854775807
        ):  # pragma: no cover
            with bodo.objmode(m=typ):
                m = p.findall(string, pos, endpos)
            return m

        return _pat_findall_const_impl

    def _pat_findall_impl(
        p, string, pos=0, endpos=9223372036854775807
    ):  # pragma: no cover
        if p.groups > 1:
            raise BodoError(
                "pattern string should be constant for 'findall' with multiple groups"
            )
        with bodo.objmode(m="list_str_type"):
            m = p.findall(string, pos, endpos)
        return m

    return _pat_findall_impl


def re_count(p, string):  # pragma: no cover
    pass


@overload(re_count)
def overload_regexp_count(p, string):
    """Count the number of regex matches, used in BodoSQL regexp_count() kernel"""

    def impl(p, string):  # pragma: no cover
        with bodo.objmode(m="int64"):
            m = len(p.findall(string))
        return m

    return impl


@overload_method(RePatternType, "subn", no_unliteral=True)
def re_subn_overload(p, repl, string, count=0):
    def _re_subn_impl(p, repl, string, count=0):  # pragma: no cover
        with bodo.objmode(out="unicode_type", s="int64"):
            out, s = p.subn(repl, string, count)
        return out, s

    return _re_subn_impl


@overload_attribute(RePatternType, "flags")
def overload_pattern_flags(p):
    def _pat_flags_impl(p):  # pragma: no cover
        with bodo.objmode(flags="int64"):
            flags = p.flags
        return flags

    return _pat_flags_impl


@overload_attribute(RePatternType, "groups")
def overload_pattern_groups(p):
    def _pat_groups_impl(p):  # pragma: no cover
        with bodo.objmode(groups="int64"):
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
        with bodo.objmode(d="dict_string_int"):
            groupindex = dict(p.groupindex)
            d = numba.typed.Dict.empty(
                key_type=numba.core.types.unicode_type, value_type=numba.int64
            )
            d.update(groupindex)
        return d

    return _pat_groupindex_impl


@overload_attribute(RePatternType, "pattern")
def overload_pattern_pattern(p):
    def _pat_pattern_impl(p):  # pragma: no cover
        with bodo.objmode(pattern="unicode_type"):
            pattern = p.pattern
        return pattern

    return _pat_pattern_impl


gen_objmode_method_overload(ReMatchType, "expand", re.Match.expand, "unicode_type")


@overload_method(ReMatchType, "group", no_unliteral=True)
def overload_match_group(m, *args):
    # NOTE: using *args in implementation throws an error in Numba lowering
    # TODO: use simpler implementation when Numba is fixed
    # def _match_group_impl(m, *args):
    #     with bodo.objmode(out="unicode_type"):
    #         out = m.group(*args)
    #     return out

    # instead of the argument types, Numba passes a tuple with a StarArgTuple type at
    # some point during lowering
    if len(args) == 1 and isinstance(
        args[0], (types.StarArgTuple, types.StarArgUniTuple)
    ):
        args = args[0].types

    # no argument case returns a string
    if len(args) == 0:

        def _match_group_impl_zero(m, *args):  # pragma: no cover
            with bodo.objmode(out="unicode_type"):
                out = m.group()
            return out

        return _match_group_impl_zero

    # using optional(str) type instead of just string to support cases where a group is
    # not matched and None should be returned
    # for example: re.match(r"(\w+)? (\w+) (\w+)", " words word")
    optional_str = types.optional(string_type)

    # one argument case returns a string
    if len(args) == 1:

        def _match_group_impl_one(m, *args):  # pragma: no cover
            group1 = args[0]
            with bodo.objmode(out=optional_str):
                out = m.group(group1)
            return out

        return _match_group_impl_one

    # multi-argument case returns a tuple of strings
    # TODO: avoid setting attributes to "types" when object mode can handle actual types
    type_name = "tuple_str_{}".format(len(args))
    setattr(types, type_name, types.Tuple([optional_str] * len(args)))
    arg_names = ", ".join("group{}".format(i + 1) for i in range(len(args)))
    func_text = "def _match_group_impl(m, *args):\n"
    func_text += "  ({}) = args\n".format(arg_names)
    func_text += "  with bodo.objmode(out='{}'):\n".format(type_name)
    func_text += "    out = m.group({})\n".format(arg_names)
    func_text += "  return out\n"

    loc_vars = {}
    exec(func_text, globals(), loc_vars)
    impl = loc_vars["_match_group_impl"]
    return impl


@overload(operator.getitem, no_unliteral=True)
def overload_match_getitem(m, ind):
    if m == re_match_type:
        return lambda m, ind: m.group(ind)


@overload_method(ReMatchType, "groups", no_unliteral=True)
def overload_match_groups(m, default=None):
    # using optional(str) type instead of just string to support cases where a group is
    # not matched and None should be returned
    # for example: re.match(r"(\w+)? (\w+) (\w+)", " words word")
    out_type = types.List(types.optional(string_type))

    # NOTE: Python returns tuple of strings, but we don't know the length in advance
    # which makes it not compilable. We return a list which is similar to tuple
    def _match_groups_impl(m, default=None):  # pragma: no cover
        with bodo.objmode(out=out_type):
            out = list(m.groups(default))
        return out

    return _match_groups_impl


@overload_method(ReMatchType, "groupdict", no_unliteral=True)
def overload_match_groupdict(m, default=None):
    # TODO: support cases where a group is not matched and None should be returned
    # for example: re.match(r"(?P<AA>\w+)? (\w+) (\w+)", " words word")

    types.dict_string_string = types.DictType(string_type, string_type)

    # Numba's Dict doesn't support Optional, so make sure output does not have None
    # TODO(ehsan): support Optional in Dict [BE-1815]
    def _check_dict_none(out):
        if any(v is None for v in out.values()):
            raise BodoError(
                "Match.groupdict() does not support default=None "
                "for groups that did not participate in the match"
            )

    def _match_groupdict_impl(m, default=None):  # pragma: no cover
        with bodo.objmode(d="dict_string_string"):
            out = m.groupdict(default)
            _check_dict_none(out)
            d = numba.typed.Dict.empty(
                key_type=numba.core.types.unicode_type,
                value_type=numba.core.types.unicode_type,
            )
            d.update(out)
        return d

    return _match_groupdict_impl


gen_objmode_method_overload(ReMatchType, "start", re.Match.start, "int64")
gen_objmode_method_overload(ReMatchType, "end", re.Match.end, "int64")


@overload_method(ReMatchType, "span", no_unliteral=True)
def overload_match_span(m, group=0):
    # span() returns a tuple of int
    types.tuple_int64_2 = types.Tuple([types.int64, types.int64])

    def _match_span_impl(m, group=0):  # pragma: no cover
        with bodo.objmode(out="tuple_int64_2"):
            out = m.span(group)
        return out

    return _match_span_impl


@overload_attribute(ReMatchType, "pos")
def overload_match_pos(p):
    def _match_pos_impl(p):  # pragma: no cover
        with bodo.objmode(pos="int64"):
            pos = p.pos
        return pos

    return _match_pos_impl


@overload_attribute(ReMatchType, "endpos")
def overload_match_endpos(p):
    def _match_endpos_impl(p):  # pragma: no cover
        with bodo.objmode(endpos="int64"):
            endpos = p.endpos
        return endpos

    return _match_endpos_impl


@overload_attribute(ReMatchType, "lastindex")
def overload_match_lastindex(p):
    # Optional to support returning None if no group was matched
    typ = types.Optional(types.int64)

    def _match_lastindex_impl(p):  # pragma: no cover
        with bodo.objmode(lastindex=typ):
            lastindex = p.lastindex
        return lastindex

    return _match_lastindex_impl


@overload_attribute(ReMatchType, "lastgroup")
def overload_match_lastgroup(p):
    # Optional to support returning None if last group didn't have a name or no group
    # was matched
    optional_str = types.optional(string_type)

    def _match_lastgroup_impl(p):  # pragma: no cover
        with bodo.objmode(lastgroup=optional_str):
            lastgroup = p.lastgroup
        return lastgroup

    return _match_lastgroup_impl


@overload_attribute(ReMatchType, "re")
def overload_match_re(m):
    def _match_re_impl(m):  # pragma: no cover
        with bodo.objmode(m_re="re_pattern_type"):
            m_re = m.re
        return m_re

    return _match_re_impl


@overload_attribute(ReMatchType, "string")
def overload_match_string(m):
    def _match_string_impl(m):  # pragma: no cover
        with bodo.objmode(out="unicode_type"):
            out = m.string
        return out

    return _match_string_impl
