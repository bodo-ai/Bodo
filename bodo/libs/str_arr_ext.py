# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Array implementation for string objects, which are usually immutable.
The characters are stored in a contingous data array, and an offsets array marks the
the individual strings. For example:
value:             ['a', 'bc', '', 'abc', None, 'bb']
data:              [a, b, c, a, b, c, b, b]
offsets:           [0, 1, 3, 3, 6, 6, 8]
"""
import operator
import decimal
import datetime
import warnings
import numpy as np
import pandas as pd
import numba
import bodo
from numba.core import types
from numba.core.typing.templates import (
    infer_global,
    AbstractTemplate,
    signature,
)
import numba.core.typing.typeof
from numba.extending import (
    typeof_impl,
    type_callable,
    models,
    register_model,
    NativeValue,
    make_attribute_wrapper,
    lower_builtin,
    box,
    unbox,
    intrinsic,
    overload_method,
    overload,
    overload_attribute,
    register_jitable,
)
from numba.core import cgutils
from bodo.libs.str_ext import string_type
from bodo.libs.list_str_arr_ext import list_string_array_type
from bodo.libs.list_item_arr_ext import ListItemArrayType
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.hiframes.datetime_date_ext import datetime_date_array_type, datetime_date_type
from bodo.utils.typing import (
    BodoWarning,
    is_list_like_index_type,
    is_overload_true,
    is_overload_none,
    parse_dtype,
    BodoError,
)
from numba.core.imputils import (
    impl_ret_new_ref,
    impl_ret_borrowed,
    iternext_impl,
    RefType,
)
import llvmlite.llvmpy.core as lc
import glob


# flag for creating pd.arrays.StringArray when boxing Bodo's native string array
# Off currently since Pandas still has issues with this new type (e.g. low performance,
# parquet write issues)
use_pd_string_array = False


char_typ = types.uint8
offset_typ = types.uint32

data_ctypes_type = types.ArrayCTypes(types.Array(char_typ, 1, "C"))
offset_ctypes_type = types.ArrayCTypes(types.Array(offset_typ, 1, "C"))


# type for pd.arrays.StringArray and ndarray with string object values
class StringArrayType(types.IterableType, types.ArrayCompatible):
    def __init__(self):
        super(StringArrayType, self).__init__(name="StringArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return string_type

    @property
    def iterator_type(self):
        return StringArrayIterator()

    def copy(self):
        return StringArrayType()


string_array_type = StringArrayType()


@typeof_impl.register(pd.arrays.StringArray)
def typeof_string_array(val, c):
    return string_array_type


class StringArrayPayloadType(types.Type):
    def __init__(self):
        super(StringArrayPayloadType, self).__init__(name="StringArrayPayloadType()")


str_arr_payload_type = StringArrayPayloadType()


# XXX: C equivalent in _bodo_common.h
@register_model(StringArrayPayloadType)
class StringArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("num_strings", types.int64),
            ("offsets", types.CPointer(offset_typ)),
            ("data", types.CPointer(char_typ)),
            ("null_bitmap", types.CPointer(char_typ)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(StringArrayType)
class StringArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("meminfo", types.MemInfoPointer(str_arr_payload_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


# dtype object for pd.StringDtype()
class StringDtype(types.Number):
    """
    Type class associated with pandas String dtype pd.StringDtype()
    """

    def __init__(self):
        super(StringDtype, self).__init__("StringDtype")


string_dtype = StringDtype()


register_model(StringDtype)(models.OpaqueModel)


@box(StringDtype)
def box_string_dtype(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    res = c.pyapi.call_method(pd_class_obj, "StringDtype", ())
    c.pyapi.decref(pd_class_obj)
    return res


@unbox(StringDtype)
def unbox_string_dtype(typ, val, c):
    return NativeValue(c.context.get_dummy_value())


typeof_impl.register(pd.StringDtype)(lambda a, b: string_dtype)
type_callable(pd.StringDtype)(lambda c: lambda: string_dtype)
lower_builtin(pd.StringDtype)(lambda c, b, s, a: c.get_dummy_value())


def create_binary_op_overload(op):
    na_fill = op == operator.ne

    def overload_string_array_binary_op(A, B):
        # both string array
        if A == string_array_type and B == string_array_type:

            def impl_both(A, B):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(A)
                out_arr = np.empty(n, np.bool_)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(
                        A, i
                    ) or bodo.libs.array_kernels.isna(B, i):
                        val = na_fill
                    else:
                        val = op(A[i], B[i])
                    out_arr[i] = val
                    # XXX assigning to out_arr indirectly since parfor fusion
                    # cannot handle branching properly here and doesn't remove
                    # out_arr. Example issue in test_agg_seq_str

                return out_arr

            return impl_both

        # left arg is string array
        if A == string_array_type and types.unliteral(B) == string_type:

            def impl_left(A, B):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(A)
                out_arr = np.empty(n, np.bool_)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(A, i):
                        val = na_fill
                    else:
                        val = op(A[i], B)
                    out_arr[i] = val

                return out_arr

            return impl_left

        # right arg is string array
        if types.unliteral(A) == string_type and B == string_array_type:

            def impl_right(A, B):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(B)
                out_arr = np.empty(n, np.bool_)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(B, i):
                        val = na_fill
                    else:
                        val = op(A, B[i])
                    out_arr[i] = val

                return out_arr

            return impl_right

    return overload_string_array_binary_op


def _install_binary_ops():
    # install comparison binary ops
    for op in (
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
    ):
        overload_impl = create_binary_op_overload(op)
        overload(op)(overload_impl)


_install_binary_ops()


@overload(operator.add, no_unliteral=True)
def overload_string_array_add(A, B):
    # both string array
    if A == string_array_type and B == string_array_type:

        def impl_both(A, B):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            l = len(A)
            num_chars = 0
            for i in numba.parfors.parfor.internal_prange(l):
                s = 0
                if not (
                    bodo.libs.array_kernels.isna(A, i)
                    or bodo.libs.array_kernels.isna(B, i)
                ):
                    s = bodo.libs.str_arr_ext.get_utf8_size(A[i] + B[i])
                num_chars += s

            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(A, j) or bodo.libs.array_kernels.isna(
                    B, j
                ):
                    out_arr[j] = ""
                    bodo.ir.join.setitem_arr_nan(out_arr, j)
                else:
                    out_arr[j] = A[j] + B[j]

            return out_arr

        return impl_both

    # left arg is string array
    if A == string_array_type and types.unliteral(B) == string_type:

        def impl_left(A, B):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            l = len(A)
            num_chars = 0
            for i in numba.parfors.parfor.internal_prange(l):
                s = 0
                if not bodo.libs.array_kernels.isna(A, i):
                    s = bodo.libs.str_arr_ext.get_utf8_size(A[i] + B)
                num_chars += s

            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(A, j):
                    out_arr[j] = ""
                    bodo.ir.join.setitem_arr_nan(out_arr, j)
                else:
                    out_arr[j] = A[j] + B

            return out_arr

        return impl_left

    # right arg is string array
    if types.unliteral(A) == string_type and B == string_array_type:

        def impl_right(A, B):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            l = len(B)
            num_chars = 0
            for i in numba.parfors.parfor.internal_prange(l):
                s = 0
                if not bodo.libs.array_kernels.isna(B, i):
                    s = bodo.libs.str_arr_ext.get_utf8_size(A + B[i])
                num_chars += s

            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(B, j):
                    out_arr[j] = ""
                    bodo.ir.join.setitem_arr_nan(out_arr, j)
                else:
                    out_arr[j] = A + B[j]

            return out_arr

        return impl_right


class StringArrayIterator(types.SimpleIteratorType):
    """
    Type class for iterators of string arrays.
    """

    def __init__(self):
        name = "iter(String)"
        yield_type = string_type
        super(StringArrayIterator, self).__init__(name, yield_type)


@register_model(StringArrayIterator)
class StrArrayIteratorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [
            ("index", types.EphemeralPointer(types.uintp)),
            ("array", string_array_type),
        ]
        super(StrArrayIteratorModel, self).__init__(dmm, fe_type, members)


lower_builtin("getiter", string_array_type)(numba.np.arrayobj.getiter_array)


@lower_builtin("iternext", StringArrayIterator)
@iternext_impl(RefType.NEW)
def iternext_str_array(context, builder, sig, args, result):
    [iterty] = sig.args
    [iter_arg] = args

    iterobj = context.make_helper(builder, iterty, value=iter_arg)
    len_sig = signature(types.intp, string_array_type)
    nitems = context.compile_internal(
        builder, lambda a: len(a), len_sig, [iterobj.array]
    )

    index = builder.load(iterobj.index)
    is_valid = builder.icmp(lc.ICMP_SLT, index, nitems)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        getitem_sig = signature(string_type, string_array_type, types.intp)
        value = context.compile_internal(
            builder, lambda a, i: a[i], getitem_sig, [iterobj.array, index]
        )
        result.yield_(value)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)


def _get_string_arr_payload(context, builder, str_arr_value):
    """get payload struct proxy for a string array value
    """
    string_array = context.make_helper(builder, string_array_type, str_arr_value)
    meminfo_data_ptr = context.nrt.meminfo_data(builder, string_array.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_data_ptr, context.get_data_type(str_arr_payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(str_arr_payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload


@intrinsic
def num_strings(typingctx, str_arr_typ=None):
    # None default to make IntelliSense happy
    assert is_str_arr_typ(str_arr_typ)

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_string_arr_payload(context, builder, in_str_arr)
        return payload.num_strings

    return types.int64(string_array_type), codegen


def _get_num_total_chars(builder, offsets, num_strings):
    """generate llvm code to get the total number of characters for string array using
    the last element of its offset array
    """
    return builder.zext(
        builder.load(builder.gep(offsets, [num_strings])), lir.IntType(64),
    )


@intrinsic
def num_total_chars(typingctx, str_arr_typ=None):
    # None default to make IntelliSense happy
    assert is_str_arr_typ(str_arr_typ)

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_string_arr_payload(context, builder, in_str_arr)
        return _get_num_total_chars(builder, payload.offsets, payload.num_strings)

    return types.uint64(string_array_type), codegen


@intrinsic
def get_offset_ptr(typingctx, str_arr_typ=None):
    assert is_str_arr_typ(str_arr_typ)

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        in_payload = _get_string_arr_payload(context, builder, in_str_arr)
        string_array = context.make_helper(builder, string_array_type, in_str_arr)
        # return string_array.offsets
        # # Create new ArrayCType structure
        ctinfo = context.make_helper(builder, offset_ctypes_type)
        ctinfo.data = builder.bitcast(in_payload.offsets, lir.IntType(32).as_pointer())
        ctinfo.meminfo = string_array.meminfo
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, offset_ctypes_type, res)

    return offset_ctypes_type(string_array_type), codegen


@intrinsic
def get_data_ptr(typingctx, str_arr_typ=None):
    assert is_str_arr_typ(str_arr_typ)

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_string_arr_payload(context, builder, in_str_arr)
        string_array = context.make_helper(builder, string_array_type, in_str_arr)

        # Create new ArrayCType structure
        ctinfo = context.make_helper(builder, data_ctypes_type)
        ctinfo.data = payload.data
        ctinfo.meminfo = string_array.meminfo
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, data_ctypes_type, res)

    return data_ctypes_type(string_array_type), codegen


@intrinsic
def get_data_ptr_ind(typingctx, str_arr_typ, int_t=None):
    assert is_str_arr_typ(str_arr_typ)

    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        payload = _get_string_arr_payload(context, builder, in_str_arr)
        string_array = context.make_helper(builder, string_array_type, in_str_arr)

        # Create new ArrayCType structure
        ctinfo = context.make_helper(builder, data_ctypes_type)
        ctinfo.data = builder.gep(payload.data, [ind])
        ctinfo.meminfo = string_array.meminfo
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, data_ctypes_type, res)

    return data_ctypes_type(string_array_type, types.intp), codegen


@intrinsic
def get_null_bitmap_ptr(typingctx, str_arr_typ=None):
    assert is_str_arr_typ(str_arr_typ)

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_string_arr_payload(context, builder, in_str_arr)
        string_array = context.make_helper(builder, string_array_type, in_str_arr)
        ctinfo = context.make_helper(builder, data_ctypes_type)
        ctinfo.data = payload.null_bitmap
        ctinfo.meminfo = string_array.meminfo
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, data_ctypes_type, res)

    return data_ctypes_type(string_array_type), codegen


@intrinsic
def getitem_str_offset(typingctx, str_arr_typ, ind_t=None):
    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        payload = _get_string_arr_payload(context, builder, in_str_arr)

        offsets = builder.bitcast(payload.offsets, lir.IntType(32).as_pointer())
        return builder.load(builder.gep(offsets, [ind]))

    return types.uint32(string_array_type, ind_t), codegen


# TODO: fix this for join
@intrinsic
def setitem_str_offset(typingctx, str_arr_typ, ind_t, val_t=None):
    def codegen(context, builder, sig, args):
        in_str_arr, ind, val = args
        payload = _get_string_arr_payload(context, builder, in_str_arr)

        offsets = builder.bitcast(payload.offsets, lir.IntType(32).as_pointer())
        builder.store(val, builder.gep(offsets, [ind]))
        return context.get_dummy_value()

    return types.void(string_array_type, ind_t, types.uint32), codegen


@intrinsic
def getitem_str_bitmap(typingctx, in_bitmap_typ, ind_t=None):
    def codegen(context, builder, sig, args):
        in_bitmap, ind = args
        if in_bitmap_typ == data_ctypes_type:
            ctinfo = context.make_helper(builder, data_ctypes_type, in_bitmap)
            in_bitmap = ctinfo.data
        return builder.load(builder.gep(in_bitmap, [ind]))

    return char_typ(in_bitmap_typ, ind_t), codegen


@intrinsic
def setitem_str_bitmap(typingctx, in_bitmap_typ, ind_t, val_t=None):
    def codegen(context, builder, sig, args):
        in_bitmap, ind, val = args
        if in_bitmap_typ == data_ctypes_type:
            ctinfo = context.make_helper(builder, data_ctypes_type, in_bitmap)
            in_bitmap = ctinfo.data
        builder.store(val, builder.gep(in_bitmap, [ind]))
        return context.get_dummy_value()

    return types.void(in_bitmap_typ, ind_t, char_typ), codegen


@intrinsic
def copy_str_arr_slice(typingctx, out_str_arr_typ, in_str_arr_typ, ind_t=None):
    """
    Copy a slice of input array (from the beginning) to output array.
    Precondition: output is allocated with enough room for data.
    """

    def codegen(context, builder, sig, args):
        out_str_arr, in_str_arr, ind = args

        in_payload = _get_string_arr_payload(context, builder, in_str_arr)
        out_payload = _get_string_arr_payload(context, builder, out_str_arr)

        in_offsets = builder.bitcast(in_payload.offsets, lir.IntType(32).as_pointer())
        out_offsets = builder.bitcast(out_payload.offsets, lir.IntType(32).as_pointer())

        ind_p1 = builder.add(ind, context.get_constant(types.intp, 1))
        cgutils.memcpy(builder, out_offsets, in_offsets, ind_p1)
        cgutils.memcpy(
            builder,
            out_payload.data,
            in_payload.data,
            builder.load(builder.gep(in_offsets, [ind])),
        )
        # n_bytes = (num_strings+sizeof(uint8_t)-1)/sizeof(uint8_t)
        ind_p7 = builder.add(ind, lir.Constant(lir.IntType(64), 7))
        n_bytes = builder.lshr(ind_p7, lir.Constant(lir.IntType(64), 3))
        # assuming rest of last byte is set to all ones (e.g. from prealloc)
        cgutils.memcpy(
            builder, out_payload.null_bitmap, in_payload.null_bitmap, n_bytes
        )
        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type, ind_t), codegen


@intrinsic
def copy_data(typingctx, str_arr_typ, out_str_arr_typ=None):
    # precondition: output is allocated with data the same size as input's data
    def codegen(context, builder, sig, args):
        out_str_arr, in_str_arr = args

        in_payload = _get_string_arr_payload(context, builder, in_str_arr)
        out_payload = _get_string_arr_payload(context, builder, out_str_arr)
        num_total_chars = _get_num_total_chars(
            builder, in_payload.offsets, in_payload.num_strings
        )

        cgutils.memcpy(
            builder, out_payload.data, in_payload.data, num_total_chars,
        )
        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type), codegen


@intrinsic
def copy_non_null_offsets(typingctx, str_arr_typ, out_str_arr_typ=None):
    # precondition: output is allocated with offset the size non-nulls in input
    def codegen(context, builder, sig, args):
        out_str_arr, in_str_arr = args

        in_payload = _get_string_arr_payload(context, builder, in_str_arr)
        out_payload = _get_string_arr_payload(context, builder, out_str_arr)

        n = in_payload.num_strings
        zero = context.get_constant(offset_typ, 0)
        curr_offset_ptr = cgutils.alloca_once_value(builder, zero)
        # XXX: assuming last offset is already set by allocate_string_array

        # for i in range(n)
        #   if not isna():
        #     out_offset[curr] = offset[i]
        with cgutils.for_range(builder, n) as loop:
            isna = lower_is_na(context, builder, in_payload.null_bitmap, loop.index)
            with cgutils.if_likely(builder, builder.not_(isna)):
                in_val = builder.load(builder.gep(in_payload.offsets, [loop.index]))
                curr_offset = builder.load(curr_offset_ptr)
                builder.store(in_val, builder.gep(out_payload.offsets, [curr_offset]))
                builder.store(
                    builder.add(
                        curr_offset, lir.Constant(context.get_data_type(offset_typ), 1)
                    ),
                    curr_offset_ptr,
                )

        # set last offset
        curr_offset = builder.load(curr_offset_ptr)
        in_val = builder.load(builder.gep(in_payload.offsets, [n]))
        builder.store(in_val, builder.gep(out_payload.offsets, [curr_offset]))
        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type), codegen


@intrinsic
def str_copy(typingctx, buff_arr_typ, ind_typ, str_typ, len_typ=None):
    def codegen(context, builder, sig, args):
        buff_arr, ind, str, len_str = args
        buff_arr = context.make_array(sig.args[0])(context, builder, buff_arr)
        ptr = builder.gep(buff_arr.data, [ind])
        cgutils.raw_memcpy(builder, ptr, str, len_str, 1)
        return context.get_dummy_value()

    return (
        types.void(
            types.Array(types.uint8, 1, "C"), types.intp, types.voidptr, types.intp
        ),
        codegen,
    )


@intrinsic
def str_copy_ptr(typingctx, ptr_typ, ind_typ, str_typ, len_typ=None):
    def codegen(context, builder, sig, args):
        ptr, ind, _str, len_str = args
        ptr = builder.gep(ptr, [ind])
        cgutils.raw_memcpy(builder, ptr, _str, len_str, 1)
        return context.get_dummy_value()

    return types.void(types.voidptr, types.intp, types.voidptr, types.intp), codegen


@numba.njit(no_cpython_wrapper=True)
def get_str_arr_item_length(A, i):  # pragma: no cover
    return np.int64(getitem_str_offset(A, i + 1) - getitem_str_offset(A, i))


@numba.njit(no_cpython_wrapper=True)
def get_str_arr_item_ptr(A, i):  # pragma: no cover
    return get_data_ptr_ind(A, getitem_str_offset(A, i))


@numba.njit(no_cpython_wrapper=True)
def get_str_null_bools(str_arr):  # pragma: no cover
    n = len(str_arr)
    null_bools = np.empty(n, np.bool_)
    for i in range(n):
        null_bools[i] = bodo.libs.array_kernels.isna(str_arr, i)
    return null_bools


# convert array to list of strings if it is StringArray
# just return it otherwise
def to_string_list(arr, str_null_bools=None):  # pragma: no cover
    return arr


@overload(to_string_list, no_unliteral=True)
def to_string_list_overload(data, str_null_bools=None):
    """if str_null_bools is True and data is tuple, output tuple contains
    an array of bools as null mask for each string array
    """
    # TODO: create a StringRandomWriteArray
    if is_str_arr_typ(data):

        def to_string_impl(data, str_null_bools=None):  # pragma: no cover
            n = len(data)
            l_str = []
            for i in range(n):
                l_str.append(data[i])
            return l_str

        return to_string_impl

    if isinstance(data, types.BaseTuple):
        count = data.count
        out = ["to_string_list(data[{}])".format(i) for i in range(count)]
        if is_overload_true(str_null_bools):
            out += [
                "get_str_null_bools(data[{}])".format(i)
                for i in range(count)
                if is_str_arr_typ(data.types[i])
            ]

        func_text = "def f(data, str_null_bools=None):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(out), "," if count == 1 else ""
        )  # single value needs comma to become tuple

        loc_vars = {}
        # print(func_text)
        exec(
            func_text,
            {
                "to_string_list": to_string_list,
                "get_str_null_bools": get_str_null_bools,
                "bodo": bodo,
            },
            loc_vars,
        )
        to_str_impl = loc_vars["f"]
        return to_str_impl

    return lambda data, str_null_bools=None: data


def cp_str_list_to_array(str_arr, str_list, str_null_bools=None):  # pragma: no cover
    return


@overload(cp_str_list_to_array, no_unliteral=True)
def cp_str_list_to_array_overload(str_arr, list_data, str_null_bools=None):
    """when str_arr is tuple, str_null_bools is a flag indicating whether
    list_data includes an extra bool array for each string array's null masks.
    When data is string array, str_null_bools is the null masks to apply.
    """
    if is_str_arr_typ(str_arr):
        if is_overload_none(str_null_bools):

            def cp_str_list_impl(
                str_arr, list_data, str_null_bools=None
            ):  # pragma: no cover
                n = len(list_data)
                for i in range(n):
                    _str = list_data[i]
                    str_arr[i] = _str

            return cp_str_list_impl
        else:

            def cp_str_list_impl_null(
                str_arr, list_data, str_null_bools=None
            ):  # pragma: no cover
                n = len(list_data)
                for i in range(n):
                    _str = list_data[i]
                    str_arr[i] = _str
                    if str_null_bools[i]:
                        str_arr_set_na(str_arr, i)
                    else:
                        str_arr_set_not_na(str_arr, i)

            return cp_str_list_impl_null

    if isinstance(str_arr, types.BaseTuple):
        count = str_arr.count

        str_ind = 0
        func_text = "def f(str_arr, list_data, str_null_bools=None):\n"
        for i in range(count):
            if is_overload_true(str_null_bools) and is_str_arr_typ(str_arr.types[i]):
                func_text += "  cp_str_list_to_array(str_arr[{}], list_data[{}], list_data[{}])\n".format(
                    i, i, count + str_ind
                )
                str_ind += 1
            else:
                func_text += "  cp_str_list_to_array(str_arr[{}], list_data[{}])\n".format(
                    i, i
                )
        func_text += "  return\n"

        loc_vars = {}
        # print(func_text)
        exec(func_text, {"cp_str_list_to_array": cp_str_list_to_array}, loc_vars)
        cp_str_impl = loc_vars["f"]
        return cp_str_impl

    return lambda str_arr, list_data, str_null_bools=None: None


def str_list_to_array(str_list):
    return str_list


@overload(str_list_to_array, no_unliteral=True)
def str_list_to_array_overload(str_list):
    """same as cp_str_list_to_array, except this call allocates output
    """
    if str_list == types.List(string_type):

        def str_list_impl(str_list):  # pragma: no cover
            n = len(str_list)
            n_char = 0
            for i in range(n):
                _str = str_list[i]
                n_char += get_utf8_size(_str)
            str_arr = pre_alloc_string_array(n, n_char)
            for i in range(n):
                _str = str_list[i]
                str_arr[i] = _str
            return str_arr

        return str_list_impl

    return lambda str_list: str_list


def is_str_arr_typ(typ):
    from bodo.hiframes.pd_series_ext import is_str_series_typ

    return typ == string_array_type or is_str_series_typ(typ)


@overload_method(StringArrayType, "copy", no_unliteral=True)
def str_arr_copy_overload(arr):
    def copy_impl(arr):  # pragma: no cover
        n = len(arr)
        n_chars = num_total_chars(arr)
        new_arr = pre_alloc_string_array(n, np.int64(n_chars))
        copy_str_arr_slice(new_arr, arr, n)
        return new_arr

    return copy_impl


@overload(len, no_unliteral=True)
def str_arr_len_overload(str_arr):
    if is_str_arr_typ(str_arr):

        def str_arr_len(str_arr):  # pragma: no cover
            return str_arr.size

        return str_arr_len


@overload_attribute(StringArrayType, "size")
def str_arr_size_overload(str_arr):
    return lambda str_arr: num_strings(str_arr)


@overload_attribute(StringArrayType, "shape")
def str_arr_shape_overload(str_arr):
    return lambda str_arr: (str_arr.size,)


from llvmlite import ir as lir
import llvmlite.binding as ll
from bodo.libs import hstr_ext

ll.add_symbol("get_str_len", hstr_ext.get_str_len)
ll.add_symbol("allocate_string_array", hstr_ext.allocate_string_array)
ll.add_symbol("setitem_string_array", hstr_ext.setitem_string_array)
ll.add_symbol("getitem_string_array", hstr_ext.getitem_string_array)
ll.add_symbol("getitem_string_array_std", hstr_ext.getitem_string_array_std)
ll.add_symbol("is_na", hstr_ext.is_na)
ll.add_symbol("string_array_from_sequence", hstr_ext.string_array_from_sequence)
ll.add_symbol("pd_array_from_string_array", hstr_ext.pd_array_from_string_array)
ll.add_symbol("np_array_from_string_array", hstr_ext.np_array_from_string_array)
ll.add_symbol("convert_len_arr_to_offset", hstr_ext.convert_len_arr_to_offset)
ll.add_symbol("set_string_array_range", hstr_ext.set_string_array_range)
ll.add_symbol("str_arr_to_int64", hstr_ext.str_arr_to_int64)
ll.add_symbol("str_arr_to_float64", hstr_ext.str_arr_to_float64)
ll.add_symbol("dtor_string_array", hstr_ext.dtor_string_array)
ll.add_symbol("get_utf8_size", hstr_ext.get_utf8_size)
ll.add_symbol("print_str_arr", hstr_ext.print_str_arr)

convert_len_arr_to_offset = types.ExternalFunction(
    "convert_len_arr_to_offset", types.void(types.voidptr, types.intp)
)


setitem_string_array = types.ExternalFunction(
    "setitem_string_array",
    types.void(
        types.CPointer(offset_typ),
        types.CPointer(char_typ),
        types.uint64,
        types.voidptr,
        types.intp,
        types.int32,
        types.int32,
        types.intp,
    ),
)
_get_utf8_size = types.ExternalFunction(
    "get_utf8_size", types.intp(types.voidptr, types.intp, types.int32)
)
_print_str_arr = types.ExternalFunction(
    "print_str_arr",
    types.void(
        types.uint64, types.uint64, types.CPointer(offset_typ), types.CPointer(char_typ)
    ),
)


def construct_string_array(context, builder):
    """Creates meminfo and sets dtor.
    """
    alloc_type = context.get_data_type(str_arr_payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = lir.FunctionType(lir.VoidType(), [llvoidptr, llsize, llvoidptr])
    dtor_fn = builder.module.get_or_insert_function(
        dtor_ftype, name="dtor_string_array"
    )

    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # Nullify all data
    # builder.store( cgutils.get_null_value(alloc_type),
    #             meminfo_data_ptr)
    return meminfo, meminfo_data_ptr


@register_jitable
def str_arr_from_sequence(in_seq):  # pragma: no cover
    n_strs = len(in_seq)
    total_chars = 0
    # get total number of chars
    for s in in_seq:
        total_chars += get_utf8_size(s)

    A = pre_alloc_string_array(n_strs, total_chars)
    for i in range(n_strs):
        A[i] = in_seq[i]

    return A


@intrinsic
def pre_alloc_string_array(typingctx, num_strs_typ, num_total_chars_typ=None):
    assert isinstance(num_strs_typ, types.Integer) and isinstance(
        num_total_chars_typ, types.Integer
    )

    def codegen(context, builder, sig, args):
        num_strs, num_total_chars = args
        meminfo, meminfo_data_ptr = construct_string_array(context, builder)

        str_arr_payload = cgutils.create_struct_proxy(str_arr_payload_type)(
            context, builder
        )
        extra_null_bytes = context.get_constant(types.int64, 0)

        # allocate string array
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(32).as_pointer().as_pointer(),
                lir.IntType(8).as_pointer().as_pointer(),
                lir.IntType(8).as_pointer().as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_alloc = builder.module.get_or_insert_function(
            fnty, name="allocate_string_array"
        )
        builder.call(
            fn_alloc,
            [
                str_arr_payload._get_ptr_by_name("offsets"),
                str_arr_payload._get_ptr_by_name("data"),
                str_arr_payload._get_ptr_by_name("null_bitmap"),
                num_strs,
                num_total_chars,
                extra_null_bytes,
            ],
        )

        str_arr_payload.num_strings = num_strs
        builder.store(str_arr_payload._getvalue(), meminfo_data_ptr)
        string_array = context.make_helper(builder, string_array_type)
        string_array.meminfo = meminfo
        ret = string_array._getvalue()
        return impl_ret_new_ref(context, builder, string_array_type, ret)

    return string_array_type(types.intp, types.intp), codegen


kBitmask = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)


# from SetBitTo() in Arrow
@numba.njit
def set_bit_to(bits, i, bit_is_set):  # pragma: no cover
    b_ind = i // 8
    byte = getitem_str_bitmap(bits, b_ind)
    byte ^= np.uint8(-np.uint8(bit_is_set) ^ byte) & kBitmask[i % 8]
    setitem_str_bitmap(bits, b_ind, byte)


@numba.njit
def get_bit_bitmap(bits, i):  # pragma: no cover
    return (getitem_str_bitmap(bits, i >> 3) >> (i & 0x07)) & 1


@numba.njit
def copy_nulls_range(out_str_arr, in_str_arr, out_start):  # pragma: no cover
    out_null_bitmap_ptr = get_null_bitmap_ptr(out_str_arr)
    in_null_bitmap_ptr = get_null_bitmap_ptr(in_str_arr)

    for j in range(len(in_str_arr)):
        bit = get_bit_bitmap(in_null_bitmap_ptr, j)
        set_bit_to(out_null_bitmap_ptr, out_start + j, bit)


@intrinsic
def set_string_array_range(
    typingctx, out_typ, in_typ, curr_str_typ, curr_chars_typ=None
):
    """
    Copy input string array to a range of output string array starting from
    curr_str_ind string index and curr_chars_ind character index.
    """
    assert is_str_arr_typ(out_typ) and is_str_arr_typ(in_typ)
    assert curr_str_typ == types.intp and curr_chars_typ == types.intp

    def codegen(context, builder, sig, args):
        out_arr, in_arr, curr_str_ind, curr_chars_ind = args

        # get input/output struct
        in_payload = _get_string_arr_payload(context, builder, in_arr)
        out_payload = _get_string_arr_payload(context, builder, out_arr)
        num_total_chars = _get_num_total_chars(
            builder, in_payload.offsets, in_payload.num_strings
        )

        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(32).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_alloc = builder.module.get_or_insert_function(
            fnty, name="set_string_array_range"
        )
        builder.call(
            fn_alloc,
            [
                out_payload.offsets,
                out_payload.data,
                in_payload.offsets,
                in_payload.data,
                curr_str_ind,
                curr_chars_ind,
                in_payload.num_strings,
                num_total_chars,
            ],
        )

        # copy nulls
        bt_typ = context.typing_context.resolve_value_type(copy_nulls_range)
        bt_sig = bt_typ.get_call_type(
            context.typing_context,
            (string_array_type, string_array_type, types.int64),
            {},
        )
        bt_impl = context.get_function(bt_typ, bt_sig)
        bt_impl(builder, (out_arr, in_arr, curr_str_ind))

        return context.get_dummy_value()

    sig = types.void(string_array_type, string_array_type, types.intp, types.intp)
    return sig, codegen


# box series calls this too
@box(StringArrayType)
def box_str_arr(typ, val, c):
    """
    """
    payload = _get_string_arr_payload(c.context, c.builder, val)

    box_fname = "np_array_from_string_array"
    if use_pd_string_array:
        box_fname = "pd_array_from_string_array"

    fnty = lir.FunctionType(
        c.context.get_argument_type(types.pyobject),
        [
            lir.IntType(64),
            lir.IntType(32).as_pointer(),
            lir.IntType(8).as_pointer(),
            lir.IntType(8).as_pointer(),
        ],
    )
    fn_get = c.builder.module.get_or_insert_function(fnty, name=box_fname)
    arr = c.builder.call(
        fn_get,
        [payload.num_strings, payload.offsets, payload.data, payload.null_bitmap,],
    )

    c.context.nrt.decref(c.builder, typ, val)
    return arr


@intrinsic
def str_arr_is_na(typingctx, str_arr_typ, ind_typ=None):
    # None default to make IntelliSense happy
    # reuse for list_string_array_type
    assert str_arr_typ in (string_array_type, list_string_array_type)

    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        if str_arr_typ == string_array_type:
            payload = _get_string_arr_payload(context, builder, in_str_arr)
        else:
            # TODO: refactor list_string_array_type case
            payload = context.make_helper(builder, str_arr_typ, in_str_arr)

        # (null_bitmap[i / 8] & kBitmask[i % 8]) == 0;
        byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
        bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
        byte = builder.load(builder.gep(payload.null_bitmap, [byte_ind], inbounds=True))
        ll_typ_mask = lir.ArrayType(lir.IntType(8), 8)
        mask_tup = cgutils.alloca_once_value(
            builder, lir.Constant(ll_typ_mask, (1, 2, 4, 8, 16, 32, 64, 128))
        )
        mask = builder.load(
            builder.gep(
                mask_tup, [lir.Constant(lir.IntType(64), 0), bit_ind], inbounds=True
            )
        )
        return builder.icmp_unsigned(
            "==", builder.and_(byte, mask), lir.Constant(lir.IntType(8), 0)
        )

    return types.bool_(str_arr_typ, types.intp), codegen


@intrinsic
def str_arr_set_na(typingctx, str_arr_typ, ind_typ=None):
    # None default to make IntelliSense happy
    # reuse for list_string_array_type
    assert str_arr_typ in (string_array_type, list_string_array_type)

    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        if str_arr_typ == string_array_type:
            payload = _get_string_arr_payload(context, builder, in_str_arr)
        else:
            # TODO: refactor list_string_array_type case
            payload = context.make_helper(builder, str_arr_typ, in_str_arr)

        # bits[i / 8] |= kBitmask[i % 8];
        byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
        bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
        byte_ptr = builder.gep(payload.null_bitmap, [byte_ind], inbounds=True)
        byte = builder.load(byte_ptr)
        ll_typ_mask = lir.ArrayType(lir.IntType(8), 8)
        mask_tup = cgutils.alloca_once_value(
            builder, lir.Constant(ll_typ_mask, (1, 2, 4, 8, 16, 32, 64, 128))
        )
        mask = builder.load(
            builder.gep(
                mask_tup, [lir.Constant(lir.IntType(64), 0), bit_ind], inbounds=True
            )
        )
        # flip all bits of mask e.g. 11111101
        mask = builder.xor(mask, lir.Constant(lir.IntType(8), -1))
        # unset masked bit
        builder.store(builder.and_(byte, mask), byte_ptr)
        return context.get_dummy_value()

    return types.void(str_arr_typ, types.intp), codegen


@intrinsic
def str_arr_set_not_na(typingctx, str_arr_typ, ind_typ=None):
    # None default to make IntelliSense happy
    assert is_str_arr_typ(str_arr_typ)

    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        payload = _get_string_arr_payload(context, builder, in_str_arr)

        # bits[i / 8] |= kBitmask[i % 8];
        byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
        bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
        byte_ptr = builder.gep(payload.null_bitmap, [byte_ind], inbounds=True)
        byte = builder.load(byte_ptr)
        ll_typ_mask = lir.ArrayType(lir.IntType(8), 8)
        mask_tup = cgutils.alloca_once_value(
            builder, lir.Constant(ll_typ_mask, (1, 2, 4, 8, 16, 32, 64, 128))
        )
        mask = builder.load(
            builder.gep(
                mask_tup, [lir.Constant(lir.IntType(64), 0), bit_ind], inbounds=True
            )
        )
        # set masked bit
        builder.store(builder.or_(byte, mask), byte_ptr)
        return context.get_dummy_value()

    return types.void(string_array_type, types.intp), codegen


@intrinsic
def set_null_bits(typingctx, str_arr_typ=None):
    assert is_str_arr_typ(str_arr_typ)

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_string_arr_payload(context, builder, in_str_arr)

        # n_bytes = (num_strings+sizeof(uint8_t)-1)/sizeof(uint8_t);
        n_bytes = builder.udiv(
            builder.add(payload.num_strings, lir.Constant(lir.IntType(64), 7)),
            lir.Constant(lir.IntType(64), 8),
        )
        cgutils.memset(builder, payload.null_bitmap, n_bytes, -1)
        return context.get_dummy_value()

    return types.none(string_array_type), codegen


@intrinsic
def move_str_arr_payload(typingctx, to_str_arr_typ, from_str_arr_typ=None):
    """Move string array payload from one array to another.
    """
    assert is_str_arr_typ(to_str_arr_typ) and is_str_arr_typ(from_str_arr_typ)

    def codegen(context, builder, sig, args):
        (to_str_arr, from_str_arr) = args

        # get payload pointers
        from_string_array = context.make_helper(
            builder, string_array_type, from_str_arr
        )
        from_meminfo_data_ptr = context.nrt.meminfo_data(
            builder, from_string_array.meminfo
        )
        from_meminfo_data_ptr = builder.bitcast(
            from_meminfo_data_ptr,
            context.get_data_type(str_arr_payload_type).as_pointer(),
        )

        to_string_array = context.make_helper(builder, string_array_type, to_str_arr)
        to_meminfo_data_void_ptr = context.nrt.meminfo_data(
            builder, to_string_array.meminfo
        )
        to_meminfo_data_ptr = builder.bitcast(
            to_meminfo_data_void_ptr,
            context.get_data_type(str_arr_payload_type).as_pointer(),
        )

        # delete existing data by calling destructor
        llvoidptr = context.get_value_type(types.voidptr)
        llsize = context.get_value_type(types.uintp)
        dtor_ftype = lir.FunctionType(lir.VoidType(), [llvoidptr, llsize, llvoidptr])
        dtor_fn = builder.module.get_or_insert_function(
            dtor_ftype, name="dtor_string_array"
        )
        # NOTE: passing zero and null for second and third argument, since not needed by
        # the destructor. This has to change if dependency to those arguments is
        # introduced.
        builder.call(
            dtor_fn,
            [
                to_meminfo_data_void_ptr,
                context.get_constant(types.int64, 0),
                context.get_constant_null(types.voidptr),
            ],
        )

        # copy payload
        builder.store(builder.load(from_meminfo_data_ptr), to_meminfo_data_ptr)

        # clear "from" array's payload, set to nulls to disable destructor
        builder.store(
            context.get_constant_null(str_arr_payload_type), from_meminfo_data_ptr
        )

        return context.get_dummy_value()

    return types.none(string_array_type, string_array_type), codegen


dummy_use = numba.njit(lambda a: None)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_utf8_size(s):
    if isinstance(s, types.StringLiteral):
        l = len(s.literal_value.encode())
        return lambda s: l

    def impl(s):  # pragma: no cover
        if s._is_ascii == 1:
            return len(s)
        n = _get_utf8_size(s._data, s._length, s._kind)
        dummy_use(s)
        return n

    return impl


@intrinsic
def setitem_str_arr_ptr(typingctx, str_arr_t, ind_t, ptr_t, len_t=None):
    def codegen(context, builder, sig, args):
        arr, ind, ptr, length = args
        payload = _get_string_arr_payload(context, builder, arr)
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(32).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(64),
            ],
        )
        fn_setitem = builder.module.get_or_insert_function(
            fnty, name="setitem_string_array"
        )
        # kind doesn't matter since input is ASCII
        kind = context.get_constant(types.int32, -1)
        is_ascii = context.get_constant(types.int32, 1)
        num_total_chars = _get_num_total_chars(
            builder, payload.offsets, payload.num_strings
        )
        builder.call(
            fn_setitem,
            [
                payload.offsets,
                payload.data,
                num_total_chars,
                builder.extract_value(ptr, 0),
                length,
                kind,
                is_ascii,
                ind,
            ],
        )
        return context.get_dummy_value()

    return types.void(str_arr_t, ind_t, ptr_t, len_t), codegen


def lower_is_na(context, builder, bull_bitmap, ind):
    fnty = lir.FunctionType(
        lir.IntType(1), [lir.IntType(8).as_pointer(), lir.IntType(64)]
    )
    fn_getitem = builder.module.get_or_insert_function(fnty, name="is_na")
    return builder.call(fn_getitem, [bull_bitmap, ind])


@intrinsic
def _memcpy(typingctx, dest_t, src_t, count_t, item_size_t=None):
    def codegen(context, builder, sig, args):
        dst, src, count, itemsize = args
        # buff_arr = context.make_array(sig.args[0])(context, builder, buff_arr)
        # ptr = builder.gep(buff_arr.data, [ind])
        cgutils.raw_memcpy(builder, dst, src, count, itemsize)
        return context.get_dummy_value()

    return types.void(types.voidptr, types.voidptr, types.intp, types.intp), codegen


@numba.njit
def print_str_arr(arr):  # pragma: no cover
    _print_str_arr(
        num_strings(arr), num_total_chars(arr), get_offset_ptr(arr), get_data_ptr(arr)
    )


@overload(operator.getitem, no_unliteral=True)
def str_arr_getitem_int(A, ind):
    if A != string_array_type:
        return

    if isinstance(types.unliteral(ind), types.Integer):
        # kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND
        def str_arr_getitem_impl(A, ind):  # pragma: no cover
            start_offset = getitem_str_offset(A, ind)
            end_offset = getitem_str_offset(A, ind + 1)
            length = end_offset - start_offset
            ptr = get_data_ptr_ind(A, start_offset)
            ret = decode_utf8(ptr, length)
            # ret = numba.cpython.unicode._empty_string(kind, length)
            # _memcpy(ret._data, ptr, length, 1)
            return ret

        return str_arr_getitem_impl

    # bool arr indexing
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def bool_impl(A, ind):  # pragma: no cover
            n = len(A)
            n_strs = 0
            n_chars = 0
            for i in range(n):
                if ind[i]:
                    n_strs += 1
                    n_chars += get_str_arr_item_length(A, i)
            out_arr = pre_alloc_string_array(n_strs, n_chars)
            str_ind = 0
            for i in range(n):
                if ind[i]:
                    _str = A[i]
                    out_arr[str_ind] = _str
                    # set NA
                    if str_arr_is_na(A, i):
                        str_arr_set_na(out_arr, str_ind)
                    str_ind += 1
            return out_arr

        return bool_impl

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def str_arr_arr_impl(A, ind):  # pragma: no cover
            n = len(ind)
            # get lengths
            n_strs = 0
            n_chars = 0
            for i in range(n):
                n_strs += 1
                n_chars += get_str_arr_item_length(A, ind[i])

            out_arr = pre_alloc_string_array(n_strs, n_chars)
            str_ind = 0
            for i in range(n):
                _str = A[ind[i]]
                out_arr[str_ind] = _str
                # set NA
                if str_arr_is_na(A, ind[i]):
                    str_arr_set_na(out_arr, str_ind)
                str_ind += 1
            return out_arr

        return str_arr_arr_impl

    # slice case
    if isinstance(ind, types.SliceType):

        def str_arr_slice_impl(A, ind):  # pragma: no cover
            n = len(A)
            slice_idx = numba.cpython.unicode._normalize_slice(ind, n)
            span = numba.cpython.unicode._slice_span(slice_idx)

            if slice_idx.step == 1:
                start_offset = getitem_str_offset(A, slice_idx.start)
                end_offset = getitem_str_offset(A, slice_idx.stop)
                n_chars = end_offset - start_offset
                new_arr = pre_alloc_string_array(span, np.int64(n_chars))
                # TODO: more efficient copy
                for i in range(span):
                    new_arr[i] = A[slice_idx.start + i]
                    # set NA
                    if str_arr_is_na(A, slice_idx.start + i):
                        str_arr_set_na(new_arr, i)
                return new_arr
            else:  # TODO: test
                # get number of chars
                n_chars = 0
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    n_chars += get_str_arr_item_length(A, i)
                new_arr = pre_alloc_string_array(span, np.int64(n_chars))
                # TODO: more efficient copy
                for i in range(span):
                    new_arr[i] = A[slice_idx.start + i * slice_idx.step]
                    # set NA
                    if str_arr_is_na(A, slice_idx.start + i * slice_idx.step):
                        str_arr_set_na(new_arr, i)
                return new_arr

        return str_arr_slice_impl


dummy_use = numba.njit(lambda a: None)


# TODO: support literals directly and turn on `no_unliteral=True`
#@overload(operator.setitem, no_unliteral=True)
@overload(operator.setitem)
def str_arr_setitem(A, idx, val):
    if A != string_array_type:
        return

    # scalar case
    if isinstance(types.unliteral(idx), types.Integer):
        assert val == string_type

        # XXX: setitem works only if value is same size as the previous value
        def impl_scalar(A, idx, val):  # pragma: no cover
            setitem_string_array(
                get_offset_ptr(A),
                get_data_ptr(A),
                num_total_chars(A),
                val._data,
                val._length,
                val._kind,
                val._is_ascii,
                idx,
            )
            # dummy use function to avoid decref of A
            # TODO: refcounting support for _offsets, ... to avoid this workaround
            dummy_use(A)
            dummy_use(val)

        return impl_scalar

    # TODO: other setitem cases


@overload_attribute(StringArrayType, "dtype")
def overload_str_arr_dtype(A):
    return lambda A: pd.StringDtype()


@overload_attribute(StringArrayType, "ndim")
def overload_str_arr_ndim(A):
    return lambda A: 1


@overload_method(StringArrayType, "astype", no_unliteral=True)
def overload_str_arr_astype(A, dtype, copy=True):

    # same dtype case
    if isinstance(dtype, types.Function) and dtype.key[0] == str:
        # no need to copy since our StringArray is immutable
        return lambda A, dtype, copy=True: A

    # numpy dtypes
    nb_dtype = parse_dtype(dtype)

    # TODO: support other dtypes if any
    # TODO: error checking
    if not isinstance(nb_dtype, (types.Float, types.Integer)):  # pragma: no cover
        raise BodoError("invalid dtype in StringArray.astype()")

    # NA positions are assigned np.nan for float output
    if isinstance(nb_dtype, types.Float):
        # TODO: raise error if conversion not possible
        def impl_float(A, dtype, copy=True):  # pragma: no cover
            numba.parfors.parfor.init_prange()  # TODO: test fusion
            n = len(A)
            B = np.empty(n, nb_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(A, i):
                    B[i] = np.nan
                else:
                    B[i] = float(A[i])
            return B

        return impl_float

    else:
        # int dtype doesn't support NAs
        # TODO: raise some form of error for NAs
        def impl_int(A, dtype, copy=True):  # pragma: no cover
            numba.parfors.parfor.init_prange()  # TODO: test fusion
            n = len(A)
            B = np.empty(n, nb_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                B[i] = int(A[i])
            return B

        return impl_int


@intrinsic
def decode_utf8(typingctx, ptr_t, len_t=None):
    def codegen(context, builder, sig, args):
        ptr, length = args

        pyapi = context.get_python_api(builder)
        unicode_obj = pyapi.string_from_string_and_size(ptr, length)
        str_val = pyapi.to_native_value(string_type, unicode_obj).value
        str_struct = cgutils.create_struct_proxy(string_type)(context, builder, str_val)
        # clear hash field due to Python-based shuffle hashing (#442)
        str_struct.hash = str_struct.hash.type(-1)
        pyapi.decref(unicode_obj)
        return str_struct._getvalue()

    return string_type(types.voidptr, types.intp), codegen


def get_arr_data_ptr(arr, ind):  # pragma: no cover
    return arr


@overload(get_arr_data_ptr, no_unliteral=True)
def overload_get_arr_data_ptr(arr, ind):
    """return data pointer for array 'arr' at index 'ind'
    currently only used in 'str_arr_item_to_numeric' for nullable int and numpy arrays
    """
    assert isinstance(types.unliteral(ind), types.Integer)

    # nullable int array
    if isinstance(arr, bodo.libs.int_arr_ext.IntegerArrayType):

        def impl_int(arr, ind):  # pragma: no cover
            return bodo.hiframes.split_impl.get_c_arr_ptr(arr._data.ctypes, ind)

        return impl_int

    # numpy case, TODO: other
    assert isinstance(arr, types.Array)

    def impl_np(arr, ind):  # pragma: no cover
        return bodo.hiframes.split_impl.get_c_arr_ptr(arr.ctypes, ind)

    return impl_np


@numba.njit(no_cpython_wrapper=True)
def str_arr_item_to_numeric(out_arr, out_ind, str_arr, ind):  # pragma: no cover
    return _str_arr_item_to_numeric(
        get_arr_data_ptr(out_arr, out_ind), str_arr, ind, out_arr.dtype,
    )


@intrinsic
def _str_arr_item_to_numeric(typingctx, out_ptr_t, str_arr_t, ind_t, out_dtype_t=None):
    assert str_arr_t == string_array_type
    assert ind_t == types.int64

    def codegen(context, builder, sig, args):
        # TODO: return tuple with value and error and avoid array arg?
        out_ptr, arr, ind, _dtype = args
        payload = _get_string_arr_payload(context, builder, arr)

        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                out_ptr.type,
                lir.IntType(32).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        fname = "str_arr_to_int64"
        if sig.args[3].dtype == types.float64:
            fname = "str_arr_to_float64"
        else:
            assert sig.args[3].dtype == types.int64
        # TODO: handle NA for float64 (use np.nan)
        fn_to_numeric = builder.module.get_or_insert_function(fnty, fname)
        return builder.call(
            fn_to_numeric, [out_ptr, payload.offsets, payload.data, ind]
        )

    return types.int32(out_ptr_t, string_array_type, types.int64, out_dtype_t), codegen


@unbox(StringArrayType)
def unbox_str_series(typ, val, c):
    """
    Unbox a Pandas String Series. We just redirect to StringArray implementation.
    """
    payload = cgutils.create_struct_proxy(str_arr_payload_type)(c.context, c.builder)
    string_array = c.context.make_helper(c.builder, typ)

    # function signature of string_array_from_sequence
    # we use void* instead of PyObject*
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64).as_pointer(),
            lir.IntType(32).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
            lir.IntType(8).as_pointer().as_pointer(),
        ],
    )
    fn = c.builder.module.get_or_insert_function(
        fnty, name="string_array_from_sequence"
    )
    c.builder.call(
        fn,
        [
            val,
            payload._get_ptr_by_name("num_strings"),
            payload._get_ptr_by_name("offsets"),
            payload._get_ptr_by_name("data"),
            payload._get_ptr_by_name("null_bitmap"),
        ],
    )

    # the raw data is now copied to payload
    # The native representation is a proxy to the payload, we need to
    # get a proxy and attach the payload and meminfo
    meminfo, meminfo_data_ptr = construct_string_array(c.context, c.builder)
    c.builder.store(payload._getvalue(), meminfo_data_ptr)

    string_array.meminfo = meminfo

    # cgutils.printf(c.builder, "unbox done\n")
    # FIXME how to check that the returned size is > 0?
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(string_array._getvalue(), is_error=is_error)


# TODO: array analysis and remove call for other functions


def pre_alloc_str_arr_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 2 and not kws
    return args[0], []


from numba.parfors.array_analysis import ArrayAnalysis

ArrayAnalysis._analyze_op_call_bodo_libs_str_arr_ext_pre_alloc_string_array = (
    pre_alloc_str_arr_equiv
)


#### glob support #####


@overload(glob.glob, no_unliteral=True)
def overload_glob_glob(pathname, recursive=False):
    def _glob_glob_impl(pathname, recursive=False):  # pragma: no cover
        with numba.objmode(l="list_str_type"):
            l = glob.glob(pathname, recursive=recursive)
        return l

    return _glob_glob_impl
