import operator
import re

import llvmlite.binding as ll
import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    bound_function,
    infer_getattr,
    infer_global,
    signature,
)
from numba.extending import (
    intrinsic,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
    register_model,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.libs import hstr_ext
from bodo.utils.typing import (
    BodoError,
    get_overload_const_int,
    get_overload_const_str,
    is_overload_constant_int,
    is_overload_constant_str,
)


# from bodo.utils.utils import unliteral_all
# TODO: resolve import conflict
def unliteral_all(args):
    return tuple(types.unliteral(a) for a in args)


ll.add_symbol("del_str", hstr_ext.del_str)
ll.add_symbol("unicode_to_utf8", hstr_ext.unicode_to_utf8)
ll.add_symbol("memcmp", hstr_ext.memcmp)
ll.add_symbol("int_to_hex", hstr_ext.int_to_hex)
ll.add_symbol("re_escape_length", hstr_ext.re_escape_length)
ll.add_symbol("re_escape_with_output", hstr_ext.re_escape_with_output)


string_type = types.unicode_type


@numba.njit
def contains_regex(e, in_str):  # pragma: no cover
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
def unicode_to_utf8_and_len(typingctx, str_typ):
    """convert unicode string to utf8 string and return its utf8 length.
    If input is ascii, just wrap its data and meminfo. Otherwise, allocate
    a new buffer and call encoder.
    """
    # Optional(string_type) means string or None. In this case, it is actually a string
    assert str_typ in (string_type, types.Optional(string_type)) or isinstance(
        str_typ, types.StringLiteral
    )
    ret_typ = types.Tuple([utf8_str_type, types.int64])

    def codegen(context, builder, sig, args):
        (str_in,) = args

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
                context.nrt.incref(builder, string_type, str_in)
                utf8_str.data = uni_str.data
                utf8_str.meminfo = uni_str.meminfo
                out_tup.f1 = uni_str.length
            # non-ascii case
            with orelse:
                # call utf8 encoder once to get the allocation size, then call again
                # to write to output buffer (TODO: avoid two calls?)
                fnty = lir.FunctionType(
                    lir.IntType(64),
                    [
                        lir.IntType(8).as_pointer(),
                        lir.IntType(8).as_pointer(),
                        lir.IntType(64),
                        lir.IntType(32),
                    ],
                )
                fn_encode = cgutils.get_or_insert_function(
                    builder.module, fnty, name="unicode_to_utf8"
                )
                null_ptr = context.get_constant_null(types.voidptr)
                utf8_len = builder.call(
                    fn_encode,
                    [null_ptr, uni_str.data, uni_str.length, uni_str.kind],
                )
                out_tup.f1 = utf8_len

                # add null padding character
                nbytes_val = builder.add(utf8_len, lir.Constant(lir.IntType(64), 1))
                utf8_str.meminfo = context.nrt.meminfo_alloc_aligned(
                    builder, size=nbytes_val, align=32
                )

                utf8_str.data = context.nrt.meminfo_data(builder, utf8_str.meminfo)
                builder.call(
                    fn_encode,
                    [utf8_str.data, uni_str.data, uni_str.length, uni_str.kind],
                )
                # set last character to NULL
                builder.store(
                    lir.Constant(lir.IntType(8), 0),
                    builder.gep(utf8_str.data, [utf8_len]),
                )

        out_tup.f0 = utf8_str._getvalue()

        return out_tup._getvalue()

    return ret_typ(string_type), codegen


@intrinsic
def re_escape_len(typingctx, str_typ):
    """Intrinsic to call into a C++ function that determines the length
    of the output string when calling re.escape.

    Args:
        typingctx (TypingContext): TypingContext required by the calling convention.
        in_str_typ (unicode_type): Input string that will be escaped

    Returns:
        types.int64: The number of elements in the output string.
    """
    assert types.unliteral(str_typ) == string_type, "str_typ must be a string"

    def codegen(context, builder, sig, args):
        (str_in,) = args
        str_struct = cgutils.create_struct_proxy(string_type)(
            context, builder, value=str_in
        )

        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(32),
            ],
        )
        fn_escape_length = cgutils.get_or_insert_function(
            builder.module, fnty, name="re_escape_length"
        )
        new_length = builder.call(
            fn_escape_length, [str_struct.data, str_struct.length, str_struct.kind]
        )
        return new_length

    return types.int64(string_type), codegen


@intrinsic
def re_escape_with_output(typingctx, in_str_typ, out_str_typ):
    """Intrinsic to call into a C++ function that implements
    re.escape. The output data is written to the buffer allocated
    in out_str

    Args:
        typingctx (TypingContext): TypingContext required by the calling convention.
        in_str_typ (unicode_type): Input string to escape
        out_str_typ (unicode_type): Output string used to store the result.
    """
    assert types.unliteral(in_str_typ) == string_type, "str_typ must be a string"
    assert types.unliteral(out_str_typ) == string_type, "str_typ must be a string"

    def codegen(context, builder, sig, args):
        (in_str, out_str) = args
        in_str_struct = cgutils.create_struct_proxy(string_type)(
            context, builder, value=in_str
        )
        out_str_struct = cgutils.create_struct_proxy(string_type)(
            context, builder, value=out_str
        )

        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
            ],
        )
        fn_re_escape_with_output = cgutils.get_or_insert_function(
            builder.module, fnty, name="re_escape_with_output"
        )
        builder.call(
            fn_re_escape_with_output,
            [
                in_str_struct.data,
                in_str_struct.length,
                out_str_struct.data,
                in_str_struct.kind,
            ],
        )

    return types.void(string_type, string_type), codegen


def unicode_to_utf8(s):  # pragma: no cover
    return s


@overload(unicode_to_utf8)
def overload_unicode_to_utf8(s):
    return lambda s: unicode_to_utf8_and_len(s)[0]  # pragma: no cover


def unicode_to_utf8_len(s):  # pragma: no cover
    return s


@overload(unicode_to_utf8_len)
def overload_unicode_to_utf8_len(s):
    """Obtain the length of a unicode string when encoded in
    utf8. Currently this requires converting to utf8.

    Args:
        s (types.unicode_type): String in unicode

    Returns:
        types.int64: length of the string as a utf8 string.
    """
    return lambda s: unicode_to_utf8_and_len(s)[1]  # pragma: no cover


@overload(max)
def overload_builtin_max(lhs, rhs):
    if lhs == types.unicode_type and rhs == types.unicode_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs > rhs else rhs

        return impl


@overload(min)
def overload_builtin_min(lhs, rhs):
    if lhs == types.unicode_type and rhs == types.unicode_type:

        def impl(lhs, rhs):  # pragma: no cover
            return lhs if lhs < rhs else rhs

        return impl


@intrinsic
def memcmp(typingctx, dest_t, src_t, count_t):
    """call memcmp() in C"""

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        memcmp_func = cgutils.get_or_insert_function(
            builder.module, fnty, name="memcmp"
        )
        return builder.call(
            memcmp_func,
            args,
        )

    return types.int32(types.voidptr, types.voidptr, types.intp), codegen


def int_to_str_len(n):  # pragma: no cover
    return len(str(n))


@overload(int_to_str_len)
def overload_int_to_str_len(n):
    """
    count the number of characters in integer n when converted to string
    """
    ten = n(10)

    def impl(n):  # pragma: no cover
        if n == 0:
            return 1  # "0"
        count = 0
        if n < 0:
            n = -n
            count += 1  # for "-"
        while n > 0:
            n = n // ten
            count += 1
        return count

    return impl


#######################  type for std string pointer  ########################
# Some support for std::string since it is used in some C++ extension code.


class StdStringType(types.Opaque):
    def __init__(self):
        super().__init__(name="StdStringType")


std_str_type = StdStringType()
register_model(StdStringType)(models.OpaqueModel)


del_str = types.ExternalFunction("del_str", types.void(std_str_type))
get_c_str = types.ExternalFunction("get_c_str", types.voidptr(std_str_type))


dummy_use = numba.njit(lambda a: None)


# not using no_unliteral=True to be able to handle string literal
@overload(int)
def int_str_overload(in_str, base=10):
    if in_str == string_type:
        if is_overload_constant_int(base) and get_overload_const_int(base) == 10:

            def _str_to_int_impl(in_str, base=10):  # pragma: no cover
                val = _str_to_int64(in_str._data, in_str._length)
                dummy_use(in_str)
                return val

            return _str_to_int_impl

        def _str_to_int_base_impl(in_str, base=10):  # pragma: no cover
            val = _str_to_int64_base(in_str._data, in_str._length, base)
            dummy_use(in_str)
            return val

        return _str_to_int_base_impl


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


ll.add_symbol("init_string_const", hstr_ext.init_string_const)
ll.add_symbol("get_c_str", hstr_ext.get_c_str)
ll.add_symbol("str_to_int64", hstr_ext.str_to_int64)
ll.add_symbol("str_to_uint64", hstr_ext.str_to_uint64)
ll.add_symbol("str_to_int64_base", hstr_ext.str_to_int64_base)
ll.add_symbol("str_to_float64", hstr_ext.str_to_float64)
ll.add_symbol("str_to_float32", hstr_ext.str_to_float32)
ll.add_symbol("get_str_len", hstr_ext.get_str_len)
ll.add_symbol("str_from_float32", hstr_ext.str_from_float32)
ll.add_symbol("str_from_float64", hstr_ext.str_from_float64)

get_std_str_len = types.ExternalFunction(
    "get_str_len", signature(types.intp, std_str_type)
)
init_string_from_chars = types.ExternalFunction(
    "init_string_const", std_str_type(types.voidptr, types.intp)
)

_str_to_int64 = types.ExternalFunction(
    "str_to_int64", signature(types.int64, types.voidptr, types.int64)
)
_str_to_uint64 = types.ExternalFunction(
    "str_to_uint64", signature(types.uint64, types.voidptr, types.int64)
)

_str_to_int64_base = types.ExternalFunction(
    "str_to_int64_base", signature(types.int64, types.voidptr, types.int64, types.int64)
)


def gen_unicode_to_std_str(context, builder, unicode_val):
    #
    uni_str = cgutils.create_struct_proxy(string_type)(
        context, builder, value=unicode_val
    )
    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer(), lir.IntType(64)]
    )
    fn = cgutils.get_or_insert_function(builder.module, fnty, name="init_string_const")
    return builder.call(fn, [uni_str.data, uni_str.length])


def gen_std_str_to_unicode(context, builder, std_str_val, del_str=False):
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def _std_str_to_unicode(std_str):  # pragma: no cover
        length = bodo.libs.str_ext.get_std_str_len(std_str)
        ret = numba.cpython.unicode._empty_string(kind, length, 1)
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


@intrinsic
def unicode_to_std_str(typingctx, unicode_t):
    def codegen(context, builder, sig, args):
        return gen_unicode_to_std_str(context, builder, args[0])

    return std_str_type(string_type), codegen


@intrinsic
def std_str_to_unicode(typingctx, unicode_t):
    def codegen(context, builder, sig, args):
        return gen_std_str_to_unicode(context, builder, args[0], True)

    return string_type(std_str_type), codegen


# string array type that is optimized for random access read/write
class RandomAccessStringArrayType(types.ArrayCompatible):
    def __init__(self):
        super().__init__(name="RandomAccessStringArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return string_type

    def copy(self):
        RandomAccessStringArrayType()


random_access_string_array = RandomAccessStringArrayType()


# store data as a list of strings
@register_model(RandomAccessStringArrayType)
class RandomAccessStringArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.List(string_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(RandomAccessStringArrayType, "data", "_data")


@intrinsic
def alloc_random_access_string_array(typingctx, n_t=None):
    def codegen(context, builder, sig, args):
        (nitems,) = args

        # alloc a list
        list_type = types.List(string_type)
        l = numba.cpython.listobj.ListInstance.allocate(
            context, builder, list_type, nitems
        )
        l.size = nitems

        str_arr = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        str_arr.data = l.value

        return str_arr._getvalue()

    return random_access_string_array(types.intp), codegen


@overload(operator.getitem, no_unliteral=True)
def random_access_str_arr_getitem(A, ind):
    if A != random_access_string_array:
        return

    if isinstance(ind, types.Integer):
        return lambda A, ind: A._data[ind]


@overload(operator.setitem)
def random_access_str_arr_setitem(A, idx, val):
    if A != random_access_string_array:
        return

    if isinstance(idx, types.Integer):
        assert val == string_type

        def impl_scalar(A, idx, val):  # pragma: no cover
            A._data[idx] = val

        return impl_scalar


@overload(len, no_unliteral=True)
def overload_str_arr_len(A):
    if A == random_access_string_array:
        return lambda A: len(A._data)


@overload_attribute(RandomAccessStringArrayType, "shape")
def overload_str_arr_shape(A):
    return lambda A: (len(A._data),)


def alloc_random_access_str_arr_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_random_access_string_array()"""
    assert len(args) == 1 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_str_ext_alloc_random_access_string_array = (
    alloc_random_access_str_arr_equiv
)


str_from_float32 = types.ExternalFunction(
    "str_from_float32", types.void(types.voidptr, types.float32)
)
str_from_float64 = types.ExternalFunction(
    "str_from_float64", types.void(types.voidptr, types.float64)
)


def float_to_str(s, v):  # pragma: no cover
    pass


@overload(float_to_str)
def float_to_str_overload(s, v):
    assert isinstance(v, types.Float)
    if v == types.float32:
        return lambda s, v: str_from_float32(s._data, v)  # pragma: no cover
    return lambda s, v: str_from_float64(s._data, v)  # pragma: no cover


@overload_method(types.Float, "__str__")
def float_str_overload(v):
    """support str(float) by preallocating the output string and calling snprintf() in C"""
    # TODO(ehsan): handle in Numba similar to str(int)
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def impl(v):  # pragma: no cover
        # Shortcut for 0
        if v == 0:
            return "0.0"
        # same formula as str(int) in Numba, plus 1 char for decimal and 6 precision
        # chars (default precision in C)
        # https://github.com/numba/numba/blob/0db8a2bcd0f53c0d0ad8a798432fb3f37f14af27/numba/cpython/unicode.py#L2391
        flag = 0
        inner_v = v
        if inner_v < 0:
            flag = 1
            inner_v = -inner_v
        if inner_v < 1:
            # Less than 1 produces a negative np.log value, so skip computation
            digits_len = 1
        else:
            digits_len = 1 + int(np.floor(np.log10(inner_v)))
        # possible values: - sign, digits before decimal place, decimal point,
        # 6 digits after decimal
        # NOTE: null character is added automatically by _malloc_string()
        length = flag + digits_len + 1 + 6
        s = numba.cpython.unicode._malloc_string(kind, 1, length, True)
        float_to_str(s, v)
        return s

    return impl


@overload(format, no_unliteral=True)
def overload_format(value, format_spec=""):
    """overload python's 'format' builtin function (using objmode if necessary)"""

    # fast path for common cases with no format specified, same as CPython
    # https://github.com/python/cpython/blob/e35dd556e1adb4fc8b83e5b75ac59e428a8b5460/Objects/abstract.c#L703
    # https://github.com/python/cpython/blob/e35dd556e1adb4fc8b83e5b75ac59e428a8b5460/Python/formatter_unicode.c#L1527
    if (
        is_overload_constant_str(format_spec)
        and get_overload_const_str(format_spec) == ""
    ):

        def impl_fast(value, format_spec=""):  # pragma: no cover
            return str(value)

        return impl_fast

    # use Python's format() in objmode
    def impl(value, format_spec=""):  # pragma: no cover
        with numba.objmode(res="string"):
            res = format(value, format_spec)
        return res

    return impl


@lower_cast(StdStringType, types.float64)
def cast_str_to_float64(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.DoubleType(), [lir.IntType(8).as_pointer()])
    fn = cgutils.get_or_insert_function(builder.module, fnty, name="str_to_float64")
    res = builder.call(fn, (val,))
    # Check if there was an error in the C++ code. If so, raise it.
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return res


@lower_cast(StdStringType, types.float32)
def cast_str_to_float32(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.FloatType(), [lir.IntType(8).as_pointer()])
    fn = cgutils.get_or_insert_function(builder.module, fnty, name="str_to_float32")
    res = builder.call(fn, (val,))
    # Check if there was an error in the C++ code. If so, raise it.
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return res


# XXX handle unicode until Numba supports float(str)
@lower_cast(string_type, types.float64)
def cast_unicode_str_to_float64(context, builder, fromty, toty, val):
    std_str = gen_unicode_to_std_str(context, builder, val)
    return cast_str_to_float64(context, builder, std_str_type, toty, std_str)


# XXX handle unicode until Numba supports float(str)
@lower_cast(string_type, types.float32)
def cast_unicode_str_to_float32(context, builder, fromty, toty, val):
    std_str = gen_unicode_to_std_str(context, builder, val)
    return cast_str_to_float32(context, builder, std_str_type, toty, std_str)


@lower_cast(string_type, types.int64)
@lower_cast(string_type, types.int32)
@lower_cast(string_type, types.int16)
@lower_cast(string_type, types.int8)
def cast_unicode_str_to_int64(context, builder, fromty, toty, val):
    # Support all signed integers with "str_to_int64", casting the output.
    uni_str = cgutils.create_struct_proxy(string_type)(context, builder, value=val)
    fnty = lir.FunctionType(
        lir.IntType(toty.bitwidth), [lir.IntType(8).as_pointer(), lir.IntType(64)]
    )
    fn = cgutils.get_or_insert_function(builder.module, fnty, name="str_to_int64")
    res = builder.call(fn, (uni_str.data, uni_str.length))
    # Check if there was an error in the C++ code. If so, raise it.
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return res


@lower_cast(string_type, types.uint64)
@lower_cast(string_type, types.uint32)
@lower_cast(string_type, types.uint16)
@lower_cast(string_type, types.uint8)
def cast_unicode_str_to_uint64(context, builder, fromty, toty, val):
    # Support all unsigned integers with "str_to_uint64", casting the output.
    uni_str = cgutils.create_struct_proxy(string_type)(context, builder, value=val)
    fnty = lir.FunctionType(
        lir.IntType(toty.bitwidth), [lir.IntType(8).as_pointer(), lir.IntType(64)]
    )
    fn = cgutils.get_or_insert_function(builder.module, fnty, name="str_to_uint64")
    res = builder.call(fn, (uni_str.data, uni_str.length))
    # Check if there was an error in the C++ code. If so, raise it.
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return res


@infer_getattr
class StringAttribute(AttributeTemplate):
    key = types.UnicodeType

    @bound_function("str.format", no_unliteral=True)
    def resolve_format(self, string_typ, args, kws):
        kws = dict(kws)
        # add dummy default value for kws to avoid errors
        arg_names = ", ".join(f"e{i}" for i in range(len(args)))
        if arg_names:
            arg_names += ", "
        kw_names = ", ".join(f"{a} = ''" for a in kws.keys())
        func_text = f"def format_stub(string, {arg_names} {kw_names}):\n"
        func_text += "    pass\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        format_stub = loc_vars["format_stub"]
        pysig = numba.core.utils.pysignature(format_stub)
        arg_types = (string_typ,) + args + tuple(kws.values())
        return signature(string_typ, arg_types).replace(pysig=pysig)


@numba.njit(cache=True)
def str_split(arr, pat, n):  # pragma: no cover
    """spits string array's elements into lists and creates an array of string arrays"""
    # numba.parfors.parfor.init_prange()
    is_regex = pat is not None and len(pat) > 1
    compiled_pat = None
    if is_regex:
        compiled_pat = re.compile(pat)
        if n == -1:
            n = 0
    else:
        if n == 0:
            n = -1
    l = len(arr)
    num_strs = 0
    num_chars = 0
    for i in numba.parfors.parfor.internal_prange(l):
        if bodo.libs.array_kernels.isna(arr, i):
            continue
        if is_regex:
            vals = compiled_pat.split(arr[i], maxsplit=n)
        # For usage in Series.str.split(). Behavior differs from split
        elif pat == "":
            vals = [""] + list(arr[i]) + [""]
        else:
            vals = arr[i].split(pat, n)
        num_strs += len(vals)
        for s in vals:
            num_chars += bodo.libs.str_arr_ext.get_utf8_size(s)

    out_arr = bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
        l, (num_strs, num_chars), bodo.libs.str_arr_ext.string_array_type
    )
    # XXX helper functions to establish aliasing between array and pointer
    # TODO: fix aliasing for getattr
    index_offsets = bodo.libs.array_item_arr_ext.get_offsets(out_arr)
    null_bitmap = bodo.libs.array_item_arr_ext.get_null_bitmap(out_arr)
    data = bodo.libs.array_item_arr_ext.get_data(out_arr)
    curr_ind = 0
    for j in numba.parfors.parfor.internal_prange(l):
        index_offsets[j] = curr_ind
        # set NA
        if bodo.libs.array_kernels.isna(arr, j):
            bodo.libs.int_arr_ext.set_bit_to_arr(null_bitmap, j, 0)
            continue
        bodo.libs.int_arr_ext.set_bit_to_arr(null_bitmap, j, 1)
        if is_regex:
            vals = compiled_pat.split(arr[j], maxsplit=n)
        # For usage in Series.str.split(). Behavior differs from split
        elif pat == "":
            vals = [""] + list(arr[j]) + [""]
        else:
            vals = arr[j].split(pat, n)
        n_str = len(vals)
        for k in range(n_str):
            s = vals[k]
            data[curr_ind] = s
            curr_ind += 1

    index_offsets[l] = curr_ind
    return out_arr


@numba.njit(cache=True)
def str_split_empty_n(arr, n):  # pragma: no cover
    """Used for pd.Series.str.split when pat is not provided, but n is and it is positive."""
    compiled_pat = re.compile("\\s+")

    # Do a pre-pass to calculate the exact number of strings and number of characters from
    # the inner string array of the final answer.
    l = len(arr)
    num_strs = 0
    num_chars = 0
    numba.parfors.parfor.init_prange()
    for i in numba.parfors.parfor.internal_prange(l):
        if bodo.libs.array_kernels.isna(arr, i):
            continue
        strs = 0
        chars = 0
        pruned_str = arr[i].strip()
        if not (bodo.libs.array_kernels.isna(arr, i) or pruned_str == 0):
            vals = compiled_pat.split(pruned_str, maxsplit=n)
            strs = len(vals)
            for s in vals:
                chars += bodo.libs.str_arr_ext.get_utf8_size(s)
        num_strs += strs
        num_chars += chars

    # Allocate the array item array where the inner array is the string
    # array with the specified size.
    out_arr = bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
        l, (num_strs, num_chars), bodo.libs.str_arr_ext.string_array_type
    )
    index_offsets = bodo.libs.array_item_arr_ext.get_offsets(out_arr)
    null_bitmap = bodo.libs.array_item_arr_ext.get_null_bitmap(out_arr)
    data = bodo.libs.array_item_arr_ext.get_data(out_arr)

    # Repeat the same logic as the first pass, writing the split up
    # string lists into the result array.
    curr_ind = 0
    for j in numba.parfors.parfor.internal_prange(l):
        index_offsets[j] = curr_ind
        if bodo.libs.array_kernels.isna(arr, j):
            bodo.libs.int_arr_ext.set_bit_to_arr(null_bitmap, j, 0)
            continue
        bodo.libs.int_arr_ext.set_bit_to_arr(null_bitmap, j, 1)
        pruned_str = arr[j].strip()
        if len(pruned_str) > 0:
            vals = compiled_pat.split(pruned_str, maxsplit=n)
            n_str = len(vals)
            for k in range(n_str):
                s = vals[k]
                data[curr_ind] = s
                curr_ind += 1

    index_offsets[l] = curr_ind
    return out_arr


@overload(hex)
def overload_hex(x):
    if isinstance(x, types.Integer):
        kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

        def impl(x):
            x = np.int64(x)
            # If int_val < 0 we have a leading -0x, else 0x
            if x < 0:
                header = "-0x"
                x = x * -1
            else:
                header = "0x"
            # Algorithm is written for unsigned 64-bit integers
            x = np.uint64(x)
            # Allocate the string. We know for any integer,
            # we need ceil(log16(x + 1)) numbers to store the result.
            # The exception is 0, which also needs 1.
            # TODO: Replace algorithm with a fast log 16
            if x == 0:
                int_len = 1
            else:
                int_len = fast_ceil_log2(x + 1)
                # Ceiling divide the total result by 4 to convert to log16
                int_len = (int_len + 3) // 4

            length = len(header) + int_len
            output = numba.cpython.unicode._empty_string(kind, length, 1)

            # Copy the header
            bodo.libs.str_arr_ext._memcpy(output._data, header._data, len(header), 1)
            int_to_hex(output, int_len, len(header), x)
            return output

        return impl


@register_jitable
def fast_ceil_log2(x):
    """
    Computes ceil(log2) for unsigned 64 bit integers.
    https://stackoverflow.com/questions/3272424/compute-fast-log-base-2-ceiling
    """
    # Add 1 if not currently a pow2 (ceil) or the value is 1
    total = 0 if ((x & (x - 1)) == 0) else 1
    # Create an array of mask, chunking the problem in half each mask.
    masks = [
        np.uint64(0xFFFFFFFF00000000),
        np.uint64(0x00000000FFFF0000),
        np.uint64(0x000000000000FF00),
        np.uint64(0x00000000000000F0),
        np.uint64(0x000000000000000C),
        np.uint64(0x0000000000000002),
    ]
    # Use the masks to compute the length
    log_add = 32
    for i in range(len(masks)):
        offset = 0 if ((x & masks[i]) == 0) else log_add
        total = total + offset
        x = x >> offset
        log_add = log_add >> 1
    return total


@intrinsic
def int_to_hex(typingctx, output, out_len, header_len, int_val):
    """Call C implementation of bytes_to_hex"""

    def codegen(context, builder, sig, args):
        (output, out_len, header_len, int_val) = args
        output_arr = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=output
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        hex_func = cgutils.get_or_insert_function(
            builder.module, fnty, name="int_to_hex"
        )
        # increment the arr ptr by the length of the header
        data_arr = builder.inttoptr(
            builder.add(
                builder.ptrtoint(output_arr.data, lir.IntType(64)),
                header_len,
            ),
            lir.IntType(8).as_pointer(),
        )
        builder.call(hex_func, (data_arr, out_len, int_val))

    return types.void(output, out_len, header_len, int_val), codegen


def alloc_empty_bytes_or_string_data(typ, kind, length, is_ascii=0):  # pragma: no cover
    """Function used to allocate empty data in functions that reuse string array and binary array."""


@overload(alloc_empty_bytes_or_string_data)
def overload_alloc_empty_bytes_or_string_data(typ, kind, length, is_ascii=0):
    typ = typ.instance_type if isinstance(typ, types.TypeRef) else typ
    if typ == bodo.types.bytes_type:
        return lambda typ, kind, length, is_ascii=0: np.empty(length, np.uint8)
    if typ == string_type:
        return (
            lambda typ, kind, length, is_ascii=0: numba.cpython.unicode._empty_string(
                kind, length, is_ascii
            )
        )  # pragma: no cover
    raise BodoError(f"Internal Error: Expected Bytes or String type, found {typ}")


def get_unicode_or_numpy_data(val):  # pragma: no cover
    """Function used to extract the underlying 'data' element of the
    model from both types.unicode and numpy arrays."""


@overload(get_unicode_or_numpy_data)
def overload_get_unicode_or_numpy_data(val):  # pragma: no cover
    if val == string_type:
        return lambda val: val._data  # pragma: no cover
    if isinstance(val, types.Array):
        return lambda val: val.ctypes  # pragma: no cover
    raise BodoError(f"Internal Error: Expected String or Numpy Array, found {val}")


@overload_method(types.UnicodeType, "removesuffix")
def overload_str_removesuffix(s, suffix):
    def impl(s, suffix):  # pragma: no cover
        if s.endswith(suffix):
            return s[: len(s) - len(suffix)]
        return s

    return impl


@overload_method(types.UnicodeType, "removeprefix")
def overload_str_removeprefix(s, prefix):
    def impl(s, prefix):  # pragma: no cover
        if s.startswith(prefix):
            return s[len(prefix) :]
        return s

    return impl
