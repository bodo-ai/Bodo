# Copyright (C) 2019 Bodo Inc. All rights reserved.
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
    ConcreteTemplate,
    bound_function,
    infer,
    infer_getattr,
    infer_global,
    signature,
)
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_builtin,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.libs import hstr_ext
from bodo.utils.typing import get_overload_const_int, is_overload_constant_int


# from bodo.utils.utils import unliteral_all
# TODO: resolve import conflict
def unliteral_all(args):
    return tuple(types.unliteral(a) for a in args)


ll.add_symbol("del_str", hstr_ext.del_str)
ll.add_symbol("unicode_to_utf8", hstr_ext.unicode_to_utf8)
ll.add_symbol("memcmp", hstr_ext.memcmp)


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
def unicode_to_utf8_and_len(typingctx, str_typ=None):
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
                fn_encode = builder.module.get_or_insert_function(
                    fnty, name="unicode_to_utf8"
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


def unicode_to_utf8(s):  # pragma: no cover
    return s


@overload(unicode_to_utf8)
def overload_unicode_to_utf8(s):
    return lambda s: unicode_to_utf8_and_len(s)[0]  # pragma: no cover


@intrinsic
def memcmp(typingctx, dest_t, src_t, count_t=None):
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
        memcmp_func = builder.module.get_or_insert_function(fnty, name="memcmp")
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
        super(StdStringType, self).__init__(name="StdStringType")


std_str_type = StdStringType()
register_model(StdStringType)(models.OpaqueModel)


del_str = types.ExternalFunction("del_str", types.void(std_str_type))
get_c_str = types.ExternalFunction("get_c_str", types.voidptr(std_str_type))


dummy_use = numba.njit(lambda a: None)


@overload(int, no_unliteral=True)
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
    fn = builder.module.get_or_insert_function(fnty, name="init_string_const")
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
def unicode_to_std_str(typingctx, unicode_t=None):
    def codegen(context, builder, sig, args):
        return gen_unicode_to_std_str(context, builder, args[0])

    return std_str_type(string_type), codegen


@intrinsic
def std_str_to_unicode(typingctx, unicode_t=None):
    def codegen(context, builder, sig, args):
        return gen_std_str_to_unicode(context, builder, args[0], True)

    return string_type(std_str_type), codegen


# string array type that is optimized for random access read/write
class RandomAccessStringArrayType(types.ArrayCompatible):
    def __init__(self):
        super(RandomAccessStringArrayType, self).__init__(
            name="RandomAccessStringArrayType()"
        )

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

    if isinstance(types.unliteral(ind), types.Integer):
        return lambda A, ind: A._data[ind]


@overload(operator.setitem)
def random_access_str_arr_setitem(A, idx, val):
    if A != random_access_string_array:
        return

    if isinstance(types.unliteral(idx), types.Integer):
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
    return args[0], []


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


@overload(str)
def float_str_overload(v):
    """support str(float) by preallocating the output string and calling sprintf() in C"""
    # TODO(ehsan): handle in Numba similar to str(int)
    if isinstance(v, types.Float):
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
            length = flag + digits_len + 1 + 6
            s = numba.cpython.unicode._malloc_string(kind, 1, length, True)
            float_to_str(s, v)
            return s

        return impl


@lower_cast(StdStringType, types.float64)
def cast_str_to_float64(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.DoubleType(), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_to_float64")
    return builder.call(fn, (val,))


@lower_cast(StdStringType, types.float32)
def cast_str_to_float32(context, builder, fromty, toty, val):
    fnty = lir.FunctionType(lir.FloatType(), [lir.IntType(8).as_pointer()])
    fn = builder.module.get_or_insert_function(fnty, name="str_to_float32")
    return builder.call(fn, (val,))


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


@infer_getattr
class StringAttribute(AttributeTemplate):
    key = types.UnicodeType

    @bound_function("str.format", no_unliteral=True)
    def resolve_format(self, string_typ, args, kws):
        kws = dict(kws)
        # add dummy default value for kws to avoid errors
        arg_names = ", ".join("e{}".format(i) for i in range(len(args)))
        if arg_names:
            arg_names += ", "
        kw_names = ", ".join("{} = ''".format(a) for a in kws.keys())
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
