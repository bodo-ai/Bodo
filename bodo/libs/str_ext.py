# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import re
from collections import namedtuple
import ctypes
import numpy as np
import numba
from numba.extending import (
    box,
    unbox,
    typeof_impl,
    register_model,
    models,
    NativeValue,
    lower_builtin,
    lower_cast,
    overload,
    type_callable,
    overload_method,
    intrinsic,
)
from numba.targets.imputils import lower_constant, impl_ret_new_ref, impl_ret_untracked
from numba import types, typing
from numba.typing.templates import (
    signature,
    AbstractTemplate,
    infer,
    infer_getattr,
    ConcreteTemplate,
    AttributeTemplate,
    bound_function,
    infer_global,
)
from numba import cgutils
import llvmlite.llvmpy.core as lc
from llvmlite import ir as lir
import llvmlite.binding as ll
import bodo

# from bodo.utils.utils import unliteral_all
# TODO: resolve import conflict
def unliteral_all(args):
    return tuple(types.unliteral(a) for a in args)


from bodo.libs import hstr_ext

ll.add_symbol("del_str", hstr_ext.del_str)
ll.add_symbol("unicode_to_utf8", hstr_ext.unicode_to_utf8)


string_type = types.unicode_type


# XXX setting hash secret for hash(unicode_type) to be consistent across
# processes. Other wise, shuffle operators like unique_str_parallel will fail.
# TODO: use a seperate implementation?
# TODO: make sure hash(str) is not already instantiated in overloads
# def _rm_hash_str_overload():
#     try:
#         fn = numba.targets.registry.cpu_target.typing_context.resolve_value_type(hash)
#         sig = signature(types.int64, types.unicode_type)
#         key = fn.get_impl_key(sig)
#         numba.targets.registry.cpu_target.target_context._defns[key]._cache.pop(sig.args)
#     except:
#         pass

# _rm_hash_str_overload()

# HACK: copied from Numba and modified, TODO: use a better solution
import numba.targets.hashing

_hashsecret_entry = namedtuple("_hashsecret_entry", ["symbol", "value"])


def _build_hashsecret():
    info = {}

    def inject(name, val):
        symbol_name = "_numba_hashsecret2_{}".format(name)
        val = ctypes.c_uint64(val)
        addr = ctypes.addressof(val)
        ll.add_symbol(symbol_name, addr)
        info[name] = _hashsecret_entry(symbol=symbol_name, value=val)

    inject("djbx33a_suffix", 11094168454019993442)
    inject("siphash_k0", 14115454492144758974)
    inject("siphash_k1", 1895071979704503752)
    return info


numba.targets.hashing._hashsecret = _build_hashsecret()


## use objmode for string methods for now

# string methods that just return another string
# 'lstrip', 'rstrip', 'strip' are in Numba now
str2str_methods = ("capitalize", "casefold", "lower", "swapcase", "title", "upper")


for method in str2str_methods:
    func_text = "def str_overload(in_str):\n"
    func_text += "  def _str_impl(in_str):\n"
    func_text += "    with numba.objmode(out='unicode_type'):\n"
    func_text += "      out = in_str.{}()\n".format(method)
    func_text += "    return out\n"
    func_text += "  return _str_impl\n"
    loc_vars = {}
    exec(func_text, {"numba": numba}, loc_vars)
    str_overload = loc_vars["str_overload"]
    overload_method(types.UnicodeType, method)(str_overload)


# string methods that just return another boolean
# 'isupper' is in numba
str2bool_methods = (
    "isalnum",
    "isalpha",
    "isdigit",
    "isspace",
    "islower",
    "istitle",
    "isnumeric",
    "isdecimal",
)


for method in str2bool_methods:
    func_text = "def str_overload(in_str):\n"
    func_text += "  def _str_impl(in_str):\n"
    func_text += "    with numba.objmode(out='boolean'):\n"
    func_text += "      out = in_str.{}()\n".format(method)
    func_text += "    return out\n"
    func_text += "  return _str_impl\n"
    loc_vars = {}
    exec(func_text, {"numba": numba}, loc_vars)
    str_overload = loc_vars["str_overload"]
    overload_method(types.UnicodeType, method)(str_overload)


@overload_method(types.UnicodeType, "replace")
def str_replace_overload(in_str, old, new, count=-1):
    def _str_replace_impl(in_str, old, new, count=-1):
        with numba.objmode(out="unicode_type"):
            out = in_str.replace(old, new, count)
        return out

    return _str_replace_impl


@overload_method(types.UnicodeType, "rfind")
def str_rfind_overload(in_str, sub, start=0, end=None):
    def _str_rfind_impl(in_str, sub, start=0, end=None):
        with numba.objmode(out="int64"):
            out = in_str.rfind(sub, start, end)
        return out

    return _str_rfind_impl


@intrinsic
def string_to_char_ptr(typingctx, str_tp=None):
    assert str_tp == string_type or isinstance(str_tp, types.StringLiteral)

    def codegen(context, builder, sig, args):

        if str_tp == string_type:
            uni_str = cgutils.create_struct_proxy(str_tp)(
                context, builder, value=args[0]
            )
            return uni_str.data
        else:
            # TODO: what about unicode strings?
            ptr = context.insert_const_string(builder.module, str_tp.literal_value)
            return ptr

    return types.voidptr(str_tp), codegen


#####################  re support  ###################


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


# jitoptions until numba #4020 is resolved
@overload(re.compile, jit_options={"no_cpython_wrapper": False})
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


@numba.njit
def contains_regex(e, in_str):
    with numba.objmode(res="bool_"):
        res = bool(e.search(in_str))
    return res


@numba.generated_jit
def str_findall_count(regex, in_str):
    def _str_findall_count_impl(regex, in_str):
        with numba.objmode(res="int64"):
            res = len(regex.findall(in_str))
        return res

    return _str_findall_count_impl


utf8_str_type = types.ArrayCTypes(types.Array(types.uint8, 1, "C"))


@intrinsic
def unicode_to_utf8_and_len(typingctx, str_typ=None):
    """convert unicode string to utf8 string and return its utf8 length.
    If input is ascii, just wrap its data and meminfo. Otherwise, allocate
    a new buffer and call encoder.
    """
    assert str_typ == string_type
    ret_typ = types.Tuple([utf8_str_type, types.int64])

    def codegen(context, builder, sig, args):
        str_in, = args

        uni_str = cgutils.create_struct_proxy(string_type)(
            context, builder, value=str_in
        )
        utf8_str = cgutils.create_struct_proxy(utf8_str_type)(context, builder)

        out_tup = cgutils.create_struct_proxy(ret_typ)(context, builder)

        is_ascii = builder.icmp_unsigned(
            "==", uni_str.is_ascii, lir.Constant(uni_str.is_ascii.type, 1)
        )

        with builder.if_else(is_ascii) as (then, orelse):
            # ascii case
            with then:
                # TODO: check refcount
                if context.enable_nrt:
                    context.nrt.incref(builder, string_type, str_in)
                utf8_str.data = uni_str.data
                utf8_str.meminfo = uni_str.meminfo
                out_tup.f1 = uni_str.length
            # non-ascii case
            with orelse:
                # add null padding character
                # XXX: assuming kind is same as number of bytes per char
                nbytes_val = builder.mul(
                    context.cast(builder, uni_str.kind, types.uint32, types.intp),
                    builder.add(uni_str.length, lir.Constant(uni_str.length.type, 1)),
                )
                utf8_str.meminfo = context.nrt.meminfo_alloc_aligned(
                    builder, size=nbytes_val, align=32
                )

                utf8_str.data = context.nrt.meminfo_data(builder, utf8_str.meminfo)

                fnty = lir.FunctionType(
                    lir.IntType(64),
                    [
                        lir.IntType(8).as_pointer(),
                        lir.IntType(8).as_pointer(),
                        lir.IntType(64),
                        lir.IntType(32),
                    ],
                )
                fn_encode = builder.module.get_or_insert_function(
                    fnty, name="unicode_to_utf8"
                )
                utf8_len = builder.call(
                    fn_encode,
                    [utf8_str.data, uni_str.data, uni_str.length, uni_str.kind],
                )
                out_tup.f1 = utf8_len

        out_tup.f0 = utf8_str._getvalue()

        return out_tup._getvalue()

    return ret_typ(string_type), codegen


#######################  type for std string pointer  ########################
# Some support for std::string since it is used in some C++ extension code.


class StdStringType(types.Opaque):
    def __init__(self):
        super(StdStringType, self).__init__(name="StdStringType")


std_str_type = StdStringType()
register_model(StdStringType)(models.OpaqueModel)


del_str = types.ExternalFunction("del_str", types.void(std_str_type))
get_c_str = types.ExternalFunction("get_c_str", types.voidptr(std_str_type))


@overload(int)
def int_str_overload(in_str):
    if in_str == string_type:

        def _str_to_int_impl(in_str):
            return _str_to_int64(in_str._data, in_str._length)

        return _str_to_int_impl


# @infer_global(int)
# class StrToInt(AbstractTemplate):
#     def generic(self, args, kws):
#         assert not kws
#         [arg] = args
#         if isinstance(arg, StdStringType):
#             return signature(types.intp, arg)
#         # TODO: implement int(str) in Numba
#         if arg == string_type:
#             return signature(types.intp, arg)


@infer_global(float)
class StrToFloat(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        [arg] = args
        if isinstance(arg, StdStringType):
            return signature(types.float64, arg)
        # TODO: implement int(str) in Numba
        if arg == string_type:
            return signature(types.float64, arg)


# TODO: handle str() in Numba
@infer_global(str)
class StrConstInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        assert args[0] in [
            types.int32,
            types.int64,
            types.float32,
            types.float64,
            string_type,
        ]
        return signature(string_type, *args)


@overload(str)
def overload_str(val):
    if val in (
        types.int8,
        types.int16,
        types.uint8,
        types.uint16,
        types.uint32,
        types.uint64,
    ):
        return lambda val: str(np.int64(val))


class RegexType(types.Opaque):
    def __init__(self):
        super(RegexType, self).__init__(name="RegexType")


regex_type = RegexType()

register_model(RegexType)(models.OpaqueModel)

ll.add_symbol("init_string_const", hstr_ext.init_string_const)
ll.add_symbol("get_c_str", hstr_ext.get_c_str)
ll.add_symbol("str_to_int64", hstr_ext.str_to_int64)
ll.add_symbol("str_to_float64", hstr_ext.str_to_float64)
ll.add_symbol("get_str_len", hstr_ext.get_str_len)
ll.add_symbol("str_from_int32", hstr_ext.str_from_int32)
ll.add_symbol("str_from_int64", hstr_ext.str_from_int64)
ll.add_symbol("str_from_float32", hstr_ext.str_from_float32)
ll.add_symbol("str_from_float64", hstr_ext.str_from_float64)

get_std_str_len = types.ExternalFunction(
    "get_str_len", signature(types.intp, std_str_type)
)
init_string_from_chars = types.ExternalFunction(
    "init_string_const", std_str_type(types.voidptr, types.intp)
)

_str_to_int64 = types.ExternalFunction(
    "str_to_int64", signature(types.intp, types.voidptr, types.intp)
)


def gen_unicode_to_std_str(context, builder, unicode_val):
    #
    uni_str = cgutils.create_struct_proxy(string_type)(
        context, builder, value=unicode_val
    )
    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer(), lir.IntType(64)]
    )
    fn = builder.module.get_or_insert_function(fnty, name="init_string_const")
    return builder.call(fn, [uni_str.data, uni_str.length])


def gen_std_str_to_unicode(context, builder, std_str_val, del_str=False):
    kind = numba.unicode.PY_UNICODE_1BYTE_KIND

    def _std_str_to_unicode(std_str):
        length = bodo.libs.str_ext.get_std_str_len(std_str)
        ret = numba.unicode._empty_string(kind, length)
        bodo.libs.str_arr_ext._memcpy(
            ret._data, bodo.libs.str_ext.get_c_str(std_str), length, 1
        )
        if del_str:
            bodo.libs.str_ext.del_str(std_str)
        return ret

    val = context.compile_internal(
        builder,
        _std_str_to_unicode,
        string_type(bodo.libs.str_ext.std_str_type),
        [std_str_val],
    )
    return val


def gen_get_unicode_chars(context, builder, unicode_val):
    uni_str = cgutils.create_struct_proxy(string_type)(
        context, builder, value=unicode_val
    )
    return uni_str.data


def unicode_to_char_ptr(in_str):
    return in_str


@overload(unicode_to_char_ptr)
def unicode_to_char_ptr_overload(a):
    # str._data is not safe since str might be literal
    # overload resolves str literal to unicode_type
    if a == string_type:
        return lambda a: a._data


@intrinsic
def unicode_to_std_str(typingctx, unicode_t=None):
    def codegen(context, builder, sig, args):
        return gen_unicode_to_std_str(context, builder, args[0])

    return std_str_type(string_type), codegen


@intrinsic
def std_str_to_unicode(typingctx, unicode_t=None):
    def codegen(context, builder, sig, args):
        return gen_std_str_to_unicode(context, builder, args[0], True)

    return string_type(std_str_type), codegen


@intrinsic
def alloc_str_list(typingctx, n_t=None):
    def codegen(context, builder, sig, args):
        nitems = args[0]
        list_type = types.List(string_type)
        l = numba.targets.listobj.ListInstance.allocate(
            context, builder, list_type, nitems
        )
        l.size = nitems
        return impl_ret_new_ref(context, builder, list_type, l.value)

    return types.List(string_type)(types.intp), codegen


@lower_builtin(str, types.Any)
def string_from_impl(context, builder, sig, args):
    in_typ = sig.args[0]
    if in_typ == string_type:
        return args[0]
    ll_in_typ = context.get_value_type(sig.args[0])
    fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [ll_in_typ])
    fn = builder.module.get_or_insert_function(fnty, name="str_from_" + str(in_typ))
    std_str = builder.call(fn, args)
    return gen_std_str_to_unicode(context, builder, std_str)


@lower_cast(StdStringType, types.float64)
def cast_str_to_float64(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.DoubleType(), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_to_float64")
    return builder.call(fn, (val,))


# XXX handle unicode until Numba supports float(str)
@lower_cast(string_type, types.float64)
def cast_unicode_str_to_float64(context, builder, fromty, toty, val):
    std_str = gen_unicode_to_std_str(context, builder, val)
    return cast_str_to_float64(context, builder, std_str_type, toty, std_str)
