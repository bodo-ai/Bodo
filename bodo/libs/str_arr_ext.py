# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Array implementation for string objects, which are usually immutable.
The characters are stored in a contiguous data array, and an offsets array marks the
the individual strings. For example:
value:             ['a', 'bc', '', 'abc', None, 'bb']
data:              [a, b, c, a, b, c, b, b]
offsets:           [0, 1, 3, 3, 6, 6, 8]
"""
import glob
import operator
from enum import Enum

import llvmlite.binding as ll
import numba
import numba.core.typing.typeof
import numpy as np
import pandas as pd
import pyarrow as pa
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed, lower_constant
from numba.core.unsafe.bytes import memcpy_region
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)

import bodo
from bodo.libs import hstr_ext
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayPayloadType,
    ArrayItemArrayType,
    _get_array_item_arr_payload,
    define_array_item_dtor,
    np_offset_type,
    offset_type,
)
from bodo.libs.binary_arr_ext import (
    BinaryArrayType,
    binary_array_type,
    pre_alloc_binary_array,
)
from bodo.libs.primitive_arr_ext import PrimitiveArrayType
from bodo.libs.str_ext import memcmp, string_type, unicode_to_utf8_and_len
from bodo.utils.typing import (
    BodoArrayIterator,
    BodoError,
    assert_bodo_error,
    get_overload_const_int,
    is_list_like_index_type,
    is_overload_constant_int,
    is_overload_none,
    is_overload_true,
    is_str_arr_type,
    parse_dtype,
    raise_bodo_error,
)

ll.add_symbol("bool_arr_to_bitmap", hstr_ext.bool_arr_to_bitmap)

char_type = types.uint8
char_arr_type = PrimitiveArrayType(char_type)
offset_arr_type = types.Array(offset_type, 1, "C")
null_bitmap_arr_type = types.Array(types.uint8, 1, "C")

data_ctypes_type = types.ArrayCTypes(types.Array(char_type, 1, "C"))
offset_ctypes_type = types.ArrayCTypes(offset_arr_type)


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
        return BodoArrayIterator(self)

    def copy(self):
        return StringArrayType()


string_array_type = StringArrayType()


@typeof_impl.register(pd.arrays.StringArray)
def typeof_string_array(val, c):
    return string_array_type


@typeof_impl.register(pd.arrays.ArrowStringArray)
def typeof_pyarrow_string_array(val, c):
    # use dict-encoded type if input is dict-encoded (boxed from Bodo dict-encoded
    # array since pandas doesn't use dict-encoded yet)
    if pa.types.is_dictionary(val._pa_array.combine_chunks().type):
        return bodo.dict_str_arr_type
    return string_array_type


@register_model(BinaryArrayType)
@register_model(StringArrayType)
class StringArrayModel(models.StructModel):
    """Use array(uint8) array to store string array data"""

    def __init__(self, dmm, fe_type):
        array_item_data_type = ArrayItemArrayType(char_arr_type)
        members = [
            ("data", array_item_data_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(StringArrayType, "data", "_data")
make_attribute_wrapper(BinaryArrayType, "data", "_data")


lower_builtin("getiter", string_array_type)(numba.np.arrayobj.getiter_array)


@intrinsic
def init_str_arr(typingctx, data_typ=None):
    """create a new string array from input data array(char) array data"""
    assert isinstance(data_typ, ArrayItemArrayType) and data_typ.dtype == char_arr_type

    def codegen(context, builder, sig, args):
        (data_arr,) = args
        str_array = context.make_helper(builder, string_array_type)
        str_array.data = data_arr
        context.nrt.incref(builder, data_typ, data_arr)
        return str_array._getvalue()

    return string_array_type(data_typ), codegen


class StringDtype(types.Number):
    """
    dtype object for pd.StringDtype()
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
    """create an overload function for a string array comparison operator"""

    # TODO(ehsan): use more optimized implementation for dictionary-encoded arrays when
    # possible, e.g. dictionaries are compatible

    def overload_string_array_binary_op(lhs, rhs):
        # optimized paths for dictionary-encoded arrays
        opt_impl = bodo.libs.dict_arr_ext.get_binary_op_overload(op, lhs, rhs)
        if opt_impl is not None:
            return opt_impl

        # both string array
        if is_str_arr_type(lhs) and is_str_arr_type(rhs):

            def impl_both(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(
                        lhs, i
                    ) or bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                        continue

                    val = op(lhs[i], rhs[i])
                    out_arr[i] = val
                    # XXX assigning to out_arr indirectly since parfor fusion
                    # cannot handle branching properly here and doesn't remove
                    # out_arr. Example issue in test_agg_seq_str

                return out_arr

            return impl_both

        # left arg is string array
        if is_str_arr_type(lhs) and types.unliteral(rhs) == string_type:

            def impl_left(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(lhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                        continue

                    val = op(lhs[i], rhs)
                    out_arr[i] = val

                return out_arr

            return impl_left

        # right arg is string array
        if types.unliteral(lhs) == string_type and is_str_arr_type(rhs):

            def impl_right(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(rhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                        continue

                    val = op(lhs, rhs[i])
                    out_arr[i] = val

                return out_arr

            return impl_right

        raise_bodo_error(f"{op} operator not supported for data types {lhs} and {rhs}.")

    return overload_string_array_binary_op


def overload_add_operator_string_array(lhs, rhs):
    lhs_is_unicode_or_string_array = is_str_arr_type(lhs) or (
        isinstance(lhs, types.Array) and lhs.dtype == string_type
    )
    rhs_is_unicode_or_string_array = is_str_arr_type(rhs) or (
        isinstance(rhs, types.Array) and rhs.dtype == string_type
    )

    # both string arrays
    # Check that at least 1 arg is an actual string_array_type to avoid
    # conflict with Numba's overload.
    if (is_str_arr_type(lhs) and rhs_is_unicode_or_string_array) or (
        lhs_is_unicode_or_string_array and is_str_arr_type(rhs)
    ):

        def impl_both(lhs, rhs):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            l = len(lhs)

            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(lhs, j) or bodo.libs.array_kernels.isna(
                    rhs, j
                ):
                    out_arr[j] = ""
                    bodo.libs.array_kernels.setna(out_arr, j)
                else:
                    out_arr[j] = lhs[j] + rhs[j]

            return out_arr

        return impl_both

    # left arg is string array
    if is_str_arr_type(lhs) and types.unliteral(rhs) == string_type:

        def impl_left(lhs, rhs):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            l = len(lhs)

            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(lhs, j):
                    out_arr[j] = ""
                    bodo.libs.array_kernels.setna(out_arr, j)
                else:
                    out_arr[j] = lhs[j] + rhs

            return out_arr

        return impl_left

    # right arg is string array
    if types.unliteral(lhs) == string_type and is_str_arr_type(rhs):

        def impl_right(lhs, rhs):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            l = len(rhs)

            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(rhs, j):
                    out_arr[j] = ""
                    bodo.libs.array_kernels.setna(out_arr, j)
                else:
                    out_arr[j] = lhs + rhs[j]

            return out_arr

        return impl_right

    # TODO: raise bodo error


def overload_mul_operator_str_arr(lhs, rhs):
    # rhs is an integer
    if is_str_arr_type(lhs) and isinstance(rhs, types.Integer):

        def impl(lhs, rhs):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            l = len(lhs)

            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(lhs, j):
                    out_arr[j] = ""
                    bodo.libs.array_kernels.setna(out_arr, j)
                else:
                    out_arr[j] = lhs[j] * rhs

            return out_arr

        return impl

    # lhs is an integer
    if isinstance(lhs, types.Integer) and is_str_arr_type(rhs):

        def impl(lhs, rhs):  # pragma: no cover
            return rhs * lhs

        return impl


def _get_str_binary_arr_payload(context, builder, arr_value, arr_typ):
    """get payload struct proxy for a string/binary array's underlying array(item) array"""
    assert arr_typ == string_array_type or arr_typ == binary_array_type
    cur_array = context.make_helper(builder, arr_typ, arr_value)
    array_item_data_type = ArrayItemArrayType(char_arr_type)
    payload = _get_array_item_arr_payload(
        context, builder, array_item_data_type, cur_array.data
    )
    return payload


@intrinsic
def num_strings(typingctx, str_arr_typ=None):
    # None default to make IntelliSense happy
    assert str_arr_typ == string_array_type

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_str_binary_arr_payload(
            context, builder, in_str_arr, string_array_type
        )
        return payload.n_arrays

    return types.int64(string_array_type), codegen


def _get_num_total_chars(builder, offsets, num_strings):
    """generate llvm code to get the total number of characters for string array using
    the last element of its offset array
    """
    return builder.zext(
        builder.load(builder.gep(offsets, [num_strings])),
        lir.IntType(64),
    )


@numba.njit
def check_offsets(str_arr):  # pragma: no cover
    """Debugging function for checking offsets of a string array for out-of-bounds
    values.

    Args:
        str_arr (StringArray): input string array
    """
    offsets = bodo.libs.array_item_arr_ext.get_offsets(str_arr._data)
    n_chars = bodo.libs.str_arr_ext.num_total_chars(str_arr)
    for i in range(bodo.libs.array_item_arr_ext.get_n_arrays(str_arr._data)):
        if offsets[i] > n_chars or offsets[i + 1] - offsets[i] < 0:
            print("wrong offset found", i, offsets[i])
            break


@intrinsic
def num_total_chars(typingctx, in_arr_typ):
    assert in_arr_typ in [binary_array_type, string_array_type]

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, sig.args[0])
        offsets_ptr = builder.bitcast(
            context.nrt.meminfo_data(builder, payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )
        return _get_num_total_chars(builder, offsets_ptr, payload.n_arrays)

    return types.uint64(in_arr_typ), codegen


@intrinsic
def get_offset_ptr(typingctx, in_arr_typ):
    assert in_arr_typ in [binary_array_type, string_array_type]

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, sig.args[0])
        # # Create new ArrayCType structure
        ctinfo = context.make_helper(builder, offset_ctypes_type)
        ctinfo.data = builder.bitcast(
            context.nrt.meminfo_data(builder, payload.offsets),
            lir.IntType(offset_type.bitwidth).as_pointer(),
        )
        ctinfo.meminfo = payload.offsets
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, offset_ctypes_type, res)

    return offset_ctypes_type(in_arr_typ), codegen


def get_data_ptr_cg(context, builder, data_arr):
    """Codegen for getting data pointer of string array's char array.
    Handles primitive array's offset attribute.

    Args:
        context (BaseContext): codegen context
        builder (IRBuilder): codegen builder
        data_arr (struct proxy(PrimitiveArrayType)): input primitive array struct proxy

    Returns:
        ll_voidptr: data pointer of char array
    """
    meminfo_ptr = context.nrt.meminfo_data(builder, data_arr.meminfo)
    return builder.inttoptr(
        builder.add(
            builder.ptrtoint(meminfo_ptr, lir.IntType(64)), data_arr.meminfo_offset
        ),
        lir.IntType(8).as_pointer(),
    )


@intrinsic
def get_data_ptr(typingctx, in_arr_typ):
    assert in_arr_typ in [binary_array_type, string_array_type]

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, sig.args[0])
        data_arr = context.make_helper(builder, char_arr_type, payload.data)

        # Create new ArrayCType structure
        ctinfo = context.make_helper(builder, data_ctypes_type)
        ctinfo.data = get_data_ptr_cg(context, builder, data_arr)
        ctinfo.meminfo = data_arr.meminfo
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, data_ctypes_type, res)

    return data_ctypes_type(in_arr_typ), codegen


@intrinsic
def get_data_ptr_ind(typingctx, in_arr_typ, int_t):
    assert in_arr_typ in [binary_array_type, string_array_type]

    def codegen(context, builder, sig, args):
        in_arr, ind = args
        payload = _get_str_binary_arr_payload(context, builder, in_arr, sig.args[0])
        data_arr = context.make_helper(builder, char_arr_type, payload.data)

        # Create new ArrayCType structure
        ctinfo = context.make_helper(builder, data_ctypes_type)
        ctinfo.meminfo = data_arr.meminfo
        data_ptr = get_data_ptr_cg(context, builder, data_arr)
        ctinfo.data = builder.gep(data_ptr, [ind])
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, data_ctypes_type, res)

    return data_ctypes_type(in_arr_typ, types.intp), codegen


@intrinsic
def copy_single_char(typingctx, dst_ptr_t, dst_ind_t, src_ptr_t, src_ind_t):
    """copy a single character value from src_ptr[src_ind] to dst_ptr[dst_ind]"""

    def codegen(context, builder, sig, args):
        dst_ptr, dst_ind, src_ptr, src_ind = args
        dst = builder.bitcast(
            builder.gep(dst_ptr, [dst_ind]), lir.IntType(8).as_pointer()
        )
        src = builder.bitcast(
            builder.gep(src_ptr, [src_ind]), lir.IntType(8).as_pointer()
        )

        char_val = builder.load(src)
        builder.store(char_val, dst)
        return context.get_dummy_value()

    return types.void(types.voidptr, types.intp, types.voidptr, types.intp), codegen


@intrinsic
def get_null_bitmap_ptr(typingctx, in_arr_typ):
    assert in_arr_typ in [binary_array_type, string_array_type]

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, sig.args[0])

        ctinfo = context.make_helper(builder, data_ctypes_type)
        ctinfo.data = context.nrt.meminfo_data(builder, payload.null_bitmap)
        ctinfo.meminfo = payload.null_bitmap
        res = ctinfo._getvalue()
        return impl_ret_borrowed(context, builder, data_ctypes_type, res)

    return data_ctypes_type(in_arr_typ), codegen


@intrinsic
def getitem_str_offset(typingctx, in_arr_typ, ind_t):
    assert in_arr_typ in [binary_array_type, string_array_type]

    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, sig.args[0])
        offsets_ptr = builder.bitcast(
            context.nrt.meminfo_data(builder, payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )
        return builder.load(builder.gep(offsets_ptr, [ind]))

    return offset_type(in_arr_typ, ind_t), codegen


@intrinsic
def setitem_str_offset(typingctx, str_arr_typ, ind_t, val_t):
    """set offset value of string array.
    Equivalent to: get_offsets(str_arr._data)[ind] = val
    """
    assert str_arr_typ == string_array_type

    def codegen(context, builder, sig, args):
        in_str_arr, ind, val = args
        payload = _get_str_binary_arr_payload(
            context, builder, in_str_arr, string_array_type
        )
        offsets_ptr = builder.bitcast(
            context.nrt.meminfo_data(builder, payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )

        builder.store(val, builder.gep(offsets_ptr, [ind]))
        return context.get_dummy_value()

    return types.void(string_array_type, ind_t, offset_type), codegen


@intrinsic
def getitem_str_bitmap(typingctx, in_bitmap_typ, ind_t):
    def codegen(context, builder, sig, args):
        in_bitmap, ind = args
        if in_bitmap_typ == data_ctypes_type:
            ctinfo = context.make_helper(builder, data_ctypes_type, in_bitmap)
            in_bitmap = ctinfo.data
        return builder.load(builder.gep(in_bitmap, [ind]))

    return char_type(in_bitmap_typ, ind_t), codegen


@intrinsic
def setitem_str_bitmap(typingctx, in_bitmap_typ, ind_t, val_t):
    def codegen(context, builder, sig, args):
        in_bitmap, ind, val = args
        if in_bitmap_typ == data_ctypes_type:
            ctinfo = context.make_helper(builder, data_ctypes_type, in_bitmap)
            in_bitmap = ctinfo.data
        builder.store(val, builder.gep(in_bitmap, [ind]))
        return context.get_dummy_value()

    return types.void(in_bitmap_typ, ind_t, char_type), codegen


@intrinsic
def copy_str_arr_slice(typingctx, out_str_arr_typ, in_str_arr_typ, ind_t):
    """
    Copy a slice of input array (from the beginning) to output array.
    Precondition: output is allocated with enough room for data.
    """
    assert out_str_arr_typ == string_array_type and in_str_arr_typ == string_array_type

    def codegen(context, builder, sig, args):
        out_str_arr, in_str_arr, ind = args

        in_payload = _get_str_binary_arr_payload(
            context, builder, in_str_arr, string_array_type
        )
        out_payload = _get_str_binary_arr_payload(
            context, builder, out_str_arr, string_array_type
        )

        in_offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, in_payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )
        out_offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, out_payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )

        in_data_arr = context.make_helper(builder, char_arr_type, in_payload.data)
        out_data_arr = context.make_helper(builder, char_arr_type, out_payload.data)
        in_data = get_data_ptr_cg(context, builder, in_data_arr)
        out_data = get_data_ptr_cg(context, builder, out_data_arr)

        in_null_bitmap = context.nrt.meminfo_data(builder, in_payload.null_bitmap)
        out_null_bitmap = context.nrt.meminfo_data(builder, out_payload.null_bitmap)

        ind_p1 = builder.add(ind, context.get_constant(types.intp, 1))
        cgutils.memcpy(builder, out_offsets, in_offsets, ind_p1)
        cgutils.memcpy(
            builder,
            out_data,
            in_data,
            builder.load(builder.gep(in_offsets, [ind])),
        )
        # n_bytes = (num_strings + 7) // 8
        ind_p7 = builder.add(ind, lir.Constant(lir.IntType(64), 7))
        n_bytes = builder.lshr(ind_p7, lir.Constant(lir.IntType(64), 3))
        # assuming rest of last byte is set to all ones (e.g. from prealloc)
        cgutils.memcpy(builder, out_null_bitmap, in_null_bitmap, n_bytes)
        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type, ind_t), codegen


@intrinsic
def copy_data(typingctx, str_arr_typ, out_str_arr_typ):
    # precondition: output is allocated with data the same size as input's data
    assert str_arr_typ == string_array_type and out_str_arr_typ == string_array_type

    def codegen(context, builder, sig, args):
        out_str_arr, in_str_arr = args

        in_payload = _get_str_binary_arr_payload(
            context, builder, in_str_arr, string_array_type
        )
        out_payload = _get_str_binary_arr_payload(
            context, builder, out_str_arr, string_array_type
        )
        in_offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, in_payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )
        in_data_arr = context.make_helper(builder, char_arr_type, in_payload.data)
        out_data_arr = context.make_helper(builder, char_arr_type, out_payload.data)
        in_data = get_data_ptr_cg(context, builder, in_data_arr)
        out_data = get_data_ptr_cg(context, builder, out_data_arr)
        num_total_chars = _get_num_total_chars(builder, in_offsets, in_payload.n_arrays)

        cgutils.memcpy(
            builder,
            out_data,
            in_data,
            num_total_chars,
        )
        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type), codegen


@intrinsic
def copy_non_null_offsets(typingctx, str_arr_typ, out_str_arr_typ):
    # precondition: output is allocated with offset the size non-nulls in input
    assert str_arr_typ == string_array_type and out_str_arr_typ == string_array_type

    def codegen(context, builder, sig, args):
        out_str_arr, in_str_arr = args

        in_payload = _get_str_binary_arr_payload(
            context, builder, in_str_arr, string_array_type
        )
        out_payload = _get_str_binary_arr_payload(
            context, builder, out_str_arr, string_array_type
        )

        in_offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, in_payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )
        out_offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, out_payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )

        in_null_bitmap = context.nrt.meminfo_data(builder, in_payload.null_bitmap)

        n = in_payload.n_arrays
        zero = context.get_constant(offset_type, 0)
        curr_offset_ptr = cgutils.alloca_once_value(builder, zero)
        # XXX: assuming last offset is already set by allocate_string_array

        # for i in range(n)
        #   if not isna():
        #     out_offset[curr] = offset[i]
        with cgutils.for_range(builder, n) as loop:
            isna = lower_is_na(context, builder, in_null_bitmap, loop.index)
            with cgutils.if_likely(builder, builder.not_(isna)):
                in_val = builder.load(builder.gep(in_offsets, [loop.index]))
                curr_offset = builder.load(curr_offset_ptr)
                builder.store(in_val, builder.gep(out_offsets, [curr_offset]))
                builder.store(
                    builder.add(
                        curr_offset,
                        lir.Constant(context.get_value_type(offset_type), 1),
                    ),
                    curr_offset_ptr,
                )

        # set last offset
        curr_offset = builder.load(curr_offset_ptr)
        in_val = builder.load(builder.gep(in_offsets, [n]))
        builder.store(in_val, builder.gep(out_offsets, [curr_offset]))
        return context.get_dummy_value()

    return types.void(string_array_type, string_array_type), codegen


@intrinsic
def str_copy(typingctx, buff_arr_typ, ind_typ, str_typ, len_typ):
    def codegen(context, builder, sig, args):
        buff_arr, ind, str, len_str = args
        buff_arr = context.make_array(sig.args[0])(context, builder, buff_arr)
        ptr = builder.gep(buff_arr.data, [ind])
        cgutils.raw_memcpy(builder, ptr, str, len_str, 1)
        return context.get_dummy_value()

    return (
        types.void(null_bitmap_arr_type, types.intp, types.voidptr, types.intp),
        codegen,
    )


@intrinsic
def str_copy_ptr(typingctx, ptr_typ, ind_typ, str_typ, len_typ):
    def codegen(context, builder, sig, args):
        ptr, ind, _str, len_str = args
        ptr = builder.gep(ptr, [ind])
        cgutils.raw_memcpy(builder, ptr, _str, len_str, 1)
        return context.get_dummy_value()

    return types.void(types.voidptr, types.intp, types.voidptr, types.intp), codegen


@numba.generated_jit(nopython=True)
def get_str_arr_item_length(A, i):  # pragma: no cover
    """return the number of bytes in the string at index i.
    Note: may not be the same as the length of the string for non-ascii unicode.
    """
    if A == bodo.dict_str_arr_type:
        # For dictionary encoded arrays we recurse on the dictionary.
        def impl(A, i):  # pragma: no cover
            idx = A._indices[i]
            dict_arr = A._data
            return np.int64(
                getitem_str_offset(dict_arr, idx + 1)
                - getitem_str_offset(dict_arr, idx)
            )

        return impl
    else:
        return lambda A, i: np.int64(
            getitem_str_offset(A, i + 1) - getitem_str_offset(A, i)
        )  # pragma: no cover


@numba.njit(no_cpython_wrapper=True)
def get_str_arr_str_length(A, i):  # pragma: no cover
    """return length of string at index i of string array A.
    This avoids creating a new string object in the common case of ascii strings.
    Note: non-ascii unicode characters may have multiple bytes per character.
    """
    start = np.int64(getitem_str_offset(A, i))
    end = np.int64(getitem_str_offset(A, i + 1))
    l = end - start
    data_ptr = get_data_ptr_ind(A, start)
    for j in range(l):
        if bodo.hiframes.split_impl.getitem_c_arr(data_ptr, j) >= 128:
            # non-ascii case
            return len(A[i])

    return l


@numba.njit(no_cpython_wrapper=True)
def get_str_arr_item_ptr(A, i):  # pragma: no cover
    return get_data_ptr_ind(A, getitem_str_offset(A, i))


@numba.generated_jit(no_cpython_wrapper=True, nopython=True)
def get_str_arr_item_copy(B, j, A, i):  # pragma: no cover
    """copy string from A[i] to B[j] without creating intermediate string value.\
    This supports both copying from a string array -> string array and a dictionary
    encoded array to a string array.
    """
    if B != string_array_type:
        raise BodoError("get_str_arr_item_copy(): Output array must be a string array")
    if not is_str_arr_type(A):
        raise BodoError(
            "get_str_arr_item_copy(): Input array must be a string array or dictionary encoded array"
        )

    # Update the location of the string array + index for dict encoded
    # array input vs string array input
    if A == bodo.dict_str_arr_type:
        load_input_array = "in_str_arr = A._data"
        input_index = "input_index = A._indices[i]"
    else:
        load_input_array = "in_str_arr = A"
        input_index = "input_index = i"

    func_text = f"""def impl(B, j, A, i):
        if j == 0:
            setitem_str_offset(B, 0, 0)

        {load_input_array}
        {input_index}

        # set NA
        if bodo.libs.array_kernels.isna(A, i):
            str_arr_set_na(B, j)
            return
        else:
            str_arr_set_not_na(B, j)

        # get input array offsets
        in_start_offset = getitem_str_offset(in_str_arr, input_index)
        in_end_offset = getitem_str_offset(in_str_arr, input_index + 1)
        val_len = in_end_offset - in_start_offset

        # set output offset
        out_start_offset = getitem_str_offset(B, j)
        out_end_offset = out_start_offset + val_len
        setitem_str_offset(B, j + 1, out_end_offset)

        # copy data
        if val_len != 0:
            # ensure required space in output array
            data_arr = B._data
            bodo.libs.array_item_arr_ext.ensure_data_capacity(
                data_arr, np.int64(out_start_offset), np.int64(out_end_offset)
            )
            out_data_ptr = get_data_ptr(B).data
            in_data_ptr = get_data_ptr(in_str_arr).data
            memcpy_region(
                out_data_ptr,
                out_start_offset,
                in_data_ptr,
                in_start_offset,
                val_len,
                1,
            )"""
    loc_vars = {}
    exec(
        func_text,
        {
            "setitem_str_offset": setitem_str_offset,
            "memcpy_region": memcpy_region,
            "getitem_str_offset": getitem_str_offset,
            "str_arr_set_na": str_arr_set_na,
            "str_arr_set_not_na": str_arr_set_not_na,
            "get_data_ptr": get_data_ptr,
            "bodo": bodo,
            "np": np,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


@numba.njit(no_cpython_wrapper=True)
def get_str_null_bools(str_arr):  # pragma: no cover
    n = len(str_arr)
    null_bools = bodo.libs.bool_arr_ext.alloc_bool_array(n)
    for i in range(n):
        null_bools[i] = bodo.libs.array_kernels.isna(str_arr, i)
    return null_bools


# converts array to list of strings if it is StringArray
# and converts array to list of bytes if it is a BinaryArray
# just return it otherwise
def to_list_if_immutable_arr(arr, str_null_bools=None):  # pragma: no cover
    return arr


@overload(to_list_if_immutable_arr, no_unliteral=True)
def to_list_if_immutable_arr_overload(data, str_null_bools=None):
    """if str_null_bools is True and data is tuple, output tuple contains
    an array of bools as null mask for each string array
    """
    # TODO: create a StringRandomWriteArray
    if is_str_arr_type(data) or data == binary_array_type:

        def to_list_impl(data, str_null_bools=None):  # pragma: no cover
            n = len(data)
            l = []
            for i in range(n):
                l.append(data[i])
            return l

        return to_list_impl

    if isinstance(data, types.BaseTuple):
        count = data.count
        out = ["to_list_if_immutable_arr(data[{}])".format(i) for i in range(count)]
        if is_overload_true(str_null_bools):
            out += [
                "get_str_null_bools(data[{}])".format(i)
                for i in range(count)
                if is_str_arr_type(data.types[i]) or data.types[i] == binary_array_type
            ]

        func_text = "def f(data, str_null_bools=None):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(out), "," if count == 1 else ""
        )  # single value needs comma to become tuple

        loc_vars = {}
        exec(
            func_text,
            {
                "to_list_if_immutable_arr": to_list_if_immutable_arr,
                "get_str_null_bools": get_str_null_bools,
                "bodo": bodo,
            },
            loc_vars,
        )
        to_str_impl = loc_vars["f"]
        return to_str_impl

    return lambda data, str_null_bools=None: data  # pragma: no cover


def cp_str_list_to_array(str_arr, str_list, str_null_bools=None):  # pragma: no cover
    return


@overload(cp_str_list_to_array, no_unliteral=True)
def cp_str_list_to_array_overload(str_arr, list_data, str_null_bools=None):
    """when str_arr is tuple, str_null_bools is a flag indicating whether
    list_data includes an extra bool array for each string array's null masks.
    When data is string array, str_null_bools is the null masks to apply.
    """
    if str_arr == string_array_type:
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
            if (
                is_overload_true(str_null_bools)
                and str_arr.types[i] == string_array_type
            ):
                func_text += "  cp_str_list_to_array(str_arr[{}], list_data[{}], list_data[{}])\n".format(
                    i, i, count + str_ind
                )
                str_ind += 1
            else:
                func_text += (
                    "  cp_str_list_to_array(str_arr[{}], list_data[{}])\n".format(i, i)
                )
        func_text += "  return\n"

        loc_vars = {}
        exec(func_text, {"cp_str_list_to_array": cp_str_list_to_array}, loc_vars)
        cp_str_impl = loc_vars["f"]
        return cp_str_impl

    return lambda str_arr, list_data, str_null_bools=None: None  # pragma: no cover


def str_list_to_array(str_list):
    return str_list


@overload(str_list_to_array, no_unliteral=True)
def str_list_to_array_overload(str_list):
    """same as cp_str_list_to_array, except this call allocates output"""
    if isinstance(str_list, types.List) and str_list.dtype == bodo.string_type:

        def str_list_impl(str_list):  # pragma: no cover
            n = len(str_list)
            str_arr = pre_alloc_string_array(n, -1)
            for i in range(n):
                _str = str_list[i]
                str_arr[i] = _str
            return str_arr

        return str_list_impl

    return lambda str_list: str_list  # pragma: no cover


def get_num_total_chars(A):  # pragma: no cover
    pass


@overload(get_num_total_chars)
def overload_get_num_total_chars(A):
    """get total number of characters in a list(str) or string array"""
    if isinstance(A, types.List) and A.dtype == string_type:

        def str_list_impl(A):  # pragma: no cover
            n = len(A)
            n_char = 0
            for i in range(n):
                _str = A[i]
                n_char += get_utf8_size(_str)
            return n_char

        return str_list_impl

    assert A == string_array_type
    return lambda A: num_total_chars(A)  # pragma: no cover


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
    if str_arr == string_array_type:

        def str_arr_len(str_arr):  # pragma: no cover
            return str_arr.size

        return str_arr_len


@overload_attribute(StringArrayType, "size")
def str_arr_size_overload(str_arr):
    return lambda str_arr: len(str_arr._data)  # pragma: no cover


@overload_attribute(StringArrayType, "shape")
def str_arr_shape_overload(str_arr):
    return lambda str_arr: (str_arr.size,)  # pragma: no cover


@overload_attribute(StringArrayType, "nbytes")
def str_arr_nbytes_overload(str_arr):
    return lambda str_arr: str_arr._data.nbytes  # pragma: no cover


@overload_method(types.Array, "tolist", no_unliteral=True)
@overload_method(StringArrayType, "tolist", no_unliteral=True)
def overload_to_list(arr):
    return lambda arr: list(arr)  # pragma: no cover


import llvmlite.binding as ll
from llvmlite import ir as lir

from bodo.libs import array_ext, hstr_ext

ll.add_symbol("get_str_len", hstr_ext.get_str_len)
ll.add_symbol("setitem_string_array", hstr_ext.setitem_string_array)
ll.add_symbol("is_na", hstr_ext.is_na)
ll.add_symbol("string_array_from_sequence", array_ext.string_array_from_sequence)
ll.add_symbol("np_array_from_string_array", hstr_ext.np_array_from_string_array)
ll.add_symbol(
    "pd_pyarrow_array_from_string_array", hstr_ext.pd_pyarrow_array_from_string_array
)
ll.add_symbol("convert_len_arr_to_offset32", hstr_ext.convert_len_arr_to_offset32)
ll.add_symbol("convert_len_arr_to_offset", hstr_ext.convert_len_arr_to_offset)
ll.add_symbol("set_string_array_range", hstr_ext.set_string_array_range)
ll.add_symbol("str_arr_to_int64", hstr_ext.str_arr_to_int64)
ll.add_symbol("str_arr_to_float64", hstr_ext.str_arr_to_float64)
ll.add_symbol("get_utf8_size", hstr_ext.get_utf8_size)
ll.add_symbol("print_str_arr", hstr_ext.print_str_arr)
ll.add_symbol("inplace_int64_to_str", hstr_ext.inplace_int64_to_str)
ll.add_symbol("str_to_dict_str_array", hstr_ext.str_to_dict_str_array)

inplace_int64_to_str = types.ExternalFunction(
    "inplace_int64_to_str", types.void(types.voidptr, types.int64, types.int64)
)

convert_len_arr_to_offset32 = types.ExternalFunction(
    "convert_len_arr_to_offset32", types.void(types.voidptr, types.intp)
)

convert_len_arr_to_offset = types.ExternalFunction(
    "convert_len_arr_to_offset", types.void(types.voidptr, types.voidptr, types.intp)
)

setitem_string_array = types.ExternalFunction(
    "setitem_string_array",
    types.void(
        types.CPointer(offset_type),
        types.CPointer(char_type),
        types.uint64,
        types.voidptr,
        types.intp,
        offset_type,
        offset_type,
        types.intp,
    ),
)
_get_utf8_size = types.ExternalFunction(
    "get_utf8_size", types.intp(types.voidptr, types.intp, offset_type)
)
_print_str_arr = types.ExternalFunction(
    "print_str_arr",
    types.void(
        types.uint64,
        types.uint64,
        types.CPointer(offset_type),
        types.CPointer(char_type),
    ),
)


@numba.generated_jit(nopython=True)
def empty_str_arr(in_seq):  # pragma: no cover
    func_text = "def f(in_seq):\n"
    func_text += "    n_strs = len(in_seq)\n"
    func_text += "    A = pre_alloc_string_array(n_strs, -1)\n"
    func_text += "    return A\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "pre_alloc_string_array": pre_alloc_string_array,
        },
        loc_vars,
    )
    f = loc_vars["f"]
    return f


@numba.generated_jit(nopython=True)
def str_arr_from_sequence(in_seq):  # pragma: no cover
    """
    Converts sequence (e.g. list, tuple, etc.) into a string array
    """
    in_seq = types.unliteral(in_seq)
    if in_seq.dtype == bodo.bytes_type:
        alloc_fn = "pre_alloc_binary_array"
    else:
        alloc_fn = "pre_alloc_string_array"

    func_text = "def f(in_seq):\n"
    func_text += "    n_strs = len(in_seq)\n"
    func_text += f"    A = {alloc_fn}(n_strs, -1)\n"
    func_text += "    for i in range(n_strs):\n"
    func_text += "        A[i] = in_seq[i]\n"
    func_text += "    return A\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "pre_alloc_string_array": pre_alloc_string_array,
            "pre_alloc_binary_array": pre_alloc_binary_array,
        },
        loc_vars,
    )
    f = loc_vars["f"]
    return f


@intrinsic
def set_all_offsets_to_0(typingctx, arr_typ):
    """
    Set all the offsets of a string/binary array to 0. Useful for
    all null columns.
    """
    assert arr_typ in (
        string_array_type,
        binary_array_type,
    ), "set_all_offsets_to_0 requires a string or binary array"

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, sig.args[0])
        n_arrays_plus_1 = builder.add(
            payload.n_arrays, lir.Constant(lir.IntType(64), 1)
        )
        # 1byte = 8bits. So >> 3. Should be "4" since we use Int32s.
        bytes_per_offset_entry = builder.lshr(
            lir.Constant(lir.IntType(64), offset_type.bitwidth),
            lir.Constant(lir.IntType(64), 3),
        )
        # n_bytes = number of entries in offset table * bytes_per_offset_entry
        n_bytes = builder.mul(
            n_arrays_plus_1,
            bytes_per_offset_entry,
        )
        null_offsets_ptr = builder.bitcast(
            context.nrt.meminfo_data(builder, payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )
        cgutils.memset(builder, null_offsets_ptr, n_bytes, 0)
        return context.get_dummy_value()

    return types.none(arr_typ), codegen


@intrinsic
def set_bitmap_all_NA(typingctx, arr_typ):
    """
    Set all the bitmap of a string/binary array to 0. Useful for
    operations that have missing values as NA.

    Note: This function assumes the string is preallocated with its length.
    """

    assert arr_typ in (
        string_array_type,
        binary_array_type,
    ), "set_bitmap_all_NA requires a string or binary array"

    def codegen(context, builder, sig, args):
        (in_str_arr,) = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, sig.args[0])
        n_arrays = payload.n_arrays
        # We use 1 byte for every 8 entries, so ((x + 7) >> 3) to compute the ceil(x, 8).
        n_bytes = builder.lshr(
            builder.add(n_arrays, lir.Constant(lir.IntType(64), 7)),
            lir.Constant(lir.IntType(64), 3),
        )
        null_bitmap_ptr = context.nrt.meminfo_data(builder, payload.null_bitmap)

        # NA is represented with 0
        cgutils.memset(builder, null_bitmap_ptr, n_bytes, 0)
        return context.get_dummy_value()

    return types.none(arr_typ), codegen


@numba.njit
def pre_alloc_string_array(n_strs, n_chars):  # pragma: no cover
    """
    Wrapper for String Array Allocation with Pre- and Post- Processing
    Preprocessing: Converting Inputs to Numpy Types
    Postprocessing: Sets offsets to 0 if n_chars == 0

    n_strs: int = Number of Strings in Array
    n_chars: Optional[int] = Number of Chars per String, or None if Unknown
    """

    if n_chars is None:
        n_chars = -1
    str_arr = init_str_arr(
        bodo.libs.array_item_arr_ext.pre_alloc_array_item_array(
            np.int64(n_strs), (np.int64(n_chars),), char_arr_type
        )
    )
    # The call above only sets offsets[0] and offset[n_strs]
    # But in case of n_chars == 0, we need to set the whole
    # offset array to 0s.
    if n_chars == 0:
        set_all_offsets_to_0(str_arr)
    return str_arr


@register_jitable
def gen_na_str_array_lens(n_strs, total_len, len_arr):
    """
    Allocates a string array with initially all NA values,
    but sets the offsets with values based on the cumulative
    sum of the len_arr.
    """
    str_arr = pre_alloc_string_array(n_strs, total_len)
    set_bitmap_all_NA(str_arr)
    # Get the offsets array
    offsets = bodo.libs.array_item_arr_ext.get_offsets(str_arr._data)
    # Compute the cumsum to set the offsets
    curr_total = 0
    if total_len == 0:
        for i in range(len(offsets)):
            offsets[i] = 0
    else:
        n_elems = len(len_arr)
        for i in range(n_elems):
            offsets[i] = curr_total
            curr_total += len_arr[i]
        offsets[n_elems] = curr_total
    return str_arr


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
def set_string_array_range(typingctx, out_typ, in_typ, curr_str_typ, curr_chars_typ):
    """
    Copy input string/binary array to a range of output string/binary array starting from
    curr_str_ind string index and curr_chars_ind character index.
    """
    assert (out_typ == string_array_type and in_typ == string_array_type) or (
        out_typ == binary_array_type and in_typ == binary_array_type
    ), "set_string_array_range requires string or binary arrays"
    assert isinstance(curr_str_typ, types.Integer) and isinstance(
        curr_chars_typ, types.Integer
    ), "set_string_array_range requires integer indices"

    def codegen(context, builder, sig, args):
        out_arr, in_arr, curr_str_ind, curr_chars_ind = args

        # get input/output struct
        in_payload = _get_str_binary_arr_payload(
            context, builder, in_arr, string_array_type
        )
        out_payload = _get_str_binary_arr_payload(
            context, builder, out_arr, string_array_type
        )

        in_offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, in_payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )
        out_offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, out_payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )

        in_data_arr = context.make_helper(builder, char_arr_type, in_payload.data)
        out_data_arr = context.make_helper(builder, char_arr_type, out_payload.data)
        in_data = get_data_ptr_cg(context, builder, in_data_arr)
        out_data = get_data_ptr_cg(context, builder, out_data_arr)

        num_total_chars = _get_num_total_chars(builder, in_offsets, in_payload.n_arrays)

        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(offset_type.bitwidth).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(offset_type.bitwidth).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_alloc = cgutils.get_or_insert_function(
            builder.module, fnty, name="set_string_array_range"
        )
        builder.call(
            fn_alloc,
            [
                out_offsets,
                out_data,
                in_offsets,
                in_data,
                curr_str_ind,
                curr_chars_ind,
                in_payload.n_arrays,
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

    sig = types.void(out_typ, in_typ, types.intp, types.intp)
    return sig, codegen


@box(BinaryArrayType)
@box(StringArrayType)
def box_str_arr(typ, val, c):
    """box string array into numpy object array with string values"""
    assert typ in [binary_array_type, string_array_type]
    array_struct = c.context.make_helper(c.builder, typ, val)
    array_item_data_type = ArrayItemArrayType(char_arr_type)
    payload = _get_array_item_arr_payload(
        c.context, c.builder, array_item_data_type, array_struct.data
    )
    is_bytes = c.context.get_constant(types.int32, int(typ == binary_array_type))

    # box to Pandas ArrowStringArray to minimize boxing overhead
    if typ != binary_array_type:
        from bodo.libs.array import array_info_type, array_to_info_codegen

        arr_info = array_to_info_codegen(
            c.context, c.builder, array_info_type(typ), (val,)
        )
        fnty = lir.FunctionType(
            c.pyapi.pyobj,
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        box_fname = "pd_pyarrow_array_from_string_array"
        fn_get = cgutils.get_or_insert_function(c.builder.module, fnty, name=box_fname)
        arr = c.builder.call(
            fn_get,
            [
                arr_info,
            ],
        )
        c.context.nrt.decref(c.builder, typ, val)
        return arr

    fnty = lir.FunctionType(
        c.context.get_argument_type(types.pyobject),
        [
            lir.IntType(64),
            lir.IntType(offset_type.bitwidth).as_pointer(),
            lir.IntType(8).as_pointer(),
            lir.IntType(8).as_pointer(),
            lir.IntType(32),
        ],
    )
    box_fname = "np_array_from_string_array"
    fn_get = cgutils.get_or_insert_function(c.builder.module, fnty, name=box_fname)
    offsets_ptr = c.builder.bitcast(
        c.context.nrt.meminfo_data(c.builder, payload.offsets),
        c.context.get_data_type(offset_type).as_pointer(),
    )
    data_arr = c.context.make_helper(c.builder, char_arr_type, payload.data)
    data_ptr = get_data_ptr_cg(c.context, c.builder, data_arr)
    null_bitmap_ptr = c.context.nrt.meminfo_data(c.builder, payload.null_bitmap)
    arr = c.builder.call(
        fn_get,
        [
            payload.n_arrays,
            offsets_ptr,
            data_ptr,
            null_bitmap_ptr,
            is_bytes,
        ],
    )

    c.context.nrt.decref(c.builder, typ, val)
    return arr


@intrinsic
def str_arr_is_na(typingctx, str_arr_typ, ind_typ):
    # None default to make IntelliSense happy
    assert str_arr_typ in (
        string_array_type,
        binary_array_type,
    ), "str_arr_is_na: string/binary array expected"

    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, str_arr_typ)
        null_bitmap_ptr = context.nrt.meminfo_data(builder, payload.null_bitmap)

        # (null_bitmap[i / 8] & kBitmask[i % 8]) == 0;
        byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
        bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
        byte = builder.load(builder.gep(null_bitmap_ptr, [byte_ind], inbounds=True))
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
def str_arr_set_na(typingctx, str_arr_typ, ind_typ):
    # None default to make IntelliSense happy
    assert str_arr_typ in [
        string_array_type,
        binary_array_type,
    ], "str_arr_set_na: string/binary array expected"

    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, str_arr_typ)

        # bits[i / 8] |= kBitmask[i % 8];
        byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
        bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
        null_bitmap_ptr = context.nrt.meminfo_data(builder, payload.null_bitmap)

        offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )

        byte_ptr = builder.gep(null_bitmap_ptr, [byte_ind], inbounds=True)
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

        # NOTE: sometimes during construction, setna may be called before setting
        # the actual value (see struct array unboxing). setting the last offset can
        # make output of num_total_chars() invalid
        # TODO: refactor string array to avoid C code
        # if ind+1 != num_strings
        #   offsets[ind+1] = offsets[ind]
        ind_plus1 = builder.add(ind, lir.Constant(lir.IntType(64), 1))
        is_na_cond = builder.icmp_unsigned("!=", ind_plus1, payload.n_arrays)
        with builder.if_then(is_na_cond):
            builder.store(
                builder.load(builder.gep(offsets, [ind])),
                builder.gep(
                    offsets,
                    [ind_plus1],
                ),
            )

        return context.get_dummy_value()

    return types.void(str_arr_typ, types.intp), codegen


@intrinsic
def str_arr_set_not_na(typingctx, str_arr_typ, ind_typ):
    # None default to make IntelliSense happy
    assert str_arr_typ in [
        binary_array_type,
        string_array_type,
    ], "str_arr_set_not_na: string/binary array expected"

    def codegen(context, builder, sig, args):
        in_str_arr, ind = args
        payload = _get_str_binary_arr_payload(context, builder, in_str_arr, str_arr_typ)

        # bits[i / 8] |= kBitmask[i % 8];
        byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
        bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
        null_bitmap_ptr = context.nrt.meminfo_data(builder, payload.null_bitmap)
        byte_ptr = builder.gep(null_bitmap_ptr, [byte_ind], inbounds=True)
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

    return types.void(str_arr_typ, types.intp), codegen


@intrinsic(prefer_literal=True)
def set_null_bits_to_value(typingctx, arr_typ, value_typ):
    """
    Sets all the bits in the null bitmap of the string/binary array
    to the specified value.
    Setting them to 0 sets them to null and setting them
    to -1 sets them to not null.
    """
    assert (
        arr_typ == string_array_type or arr_typ == binary_array_type
    ) and is_overload_constant_int(value_typ)

    def codegen(context, builder, sig, args):
        (
            in_str_arr,
            value,
        ) = args
        payload = _get_str_binary_arr_payload(
            context, builder, in_str_arr, string_array_type
        )

        # n_bytes = (num_strings + 7) // 8;
        n_bytes = builder.udiv(
            builder.add(payload.n_arrays, lir.Constant(lir.IntType(64), 7)),
            lir.Constant(lir.IntType(64), 8),
        )
        null_bitmap_ptr = context.nrt.meminfo_data(builder, payload.null_bitmap)
        cgutils.memset(builder, null_bitmap_ptr, n_bytes, value)
        return context.get_dummy_value()

    return types.none(arr_typ, types.int8), codegen


def _get_str_binary_arr_data_payload_ptr(context, builder, str_arr):
    """get pointer to meminfo of string array's underlying data array"""
    string_array = context.make_helper(builder, string_array_type, str_arr)
    array_item_data_type = ArrayItemArrayType(char_arr_type)
    array_item_array = context.make_helper(
        builder, array_item_data_type, string_array.data
    )
    payload_type = ArrayItemArrayPayloadType(array_item_data_type)
    meminfo_void_ptr = context.nrt.meminfo_data(builder, array_item_array.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_void_ptr, context.get_value_type(payload_type).as_pointer()
    )
    return meminfo_data_ptr


@intrinsic
def move_str_binary_arr_payload(typingctx, to_arr_typ, from_arr_typ):
    """Move string/binary array payload from one array to another."""
    assert (to_arr_typ == string_array_type and from_arr_typ == string_array_type) or (
        to_arr_typ == binary_array_type and from_arr_typ == binary_array_type
    )

    def codegen(context, builder, sig, args):
        (to_arr, from_arr) = args

        # get payload pointers
        from_meminfo_data_ptr = _get_str_binary_arr_data_payload_ptr(
            context, builder, from_arr
        )
        to_meminfo_data_ptr = _get_str_binary_arr_data_payload_ptr(
            context, builder, to_arr
        )
        from_payload = _get_str_binary_arr_payload(
            context, builder, from_arr, sig.args[1]
        )
        to_payload = _get_str_binary_arr_payload(context, builder, to_arr, sig.args[0])

        array_item_data_type = ArrayItemArrayType(char_arr_type)
        payload_type = ArrayItemArrayPayloadType(array_item_data_type)

        # incref data of from_str_arr (not the meminfo, which is not copied)
        context.nrt.incref(builder, payload_type, from_payload._getvalue())

        # decref data of to_str_arr (not the meminfo, which is still used)
        context.nrt.decref(builder, payload_type, to_payload._getvalue())

        # copy payload
        builder.store(builder.load(from_meminfo_data_ptr), to_meminfo_data_ptr)

        return context.get_dummy_value()

    return types.none(to_arr_typ, from_arr_typ), codegen


dummy_use = numba.njit(lambda a: None)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_utf8_size(s):
    if isinstance(s, types.StringLiteral):
        l = len(s.literal_value.encode())
        return lambda s: l  # pragma: no cover

    def impl(s):  # pragma: no cover
        # s can be Optional
        if s is None:
            return 0
        s = bodo.utils.indexing.unoptional(s)
        if s._is_ascii == 1:
            return len(s)
        n = _get_utf8_size(s._data, s._length, s._kind)
        dummy_use(s)
        return n

    return impl


@intrinsic
def setitem_str_arr_ptr(typingctx, str_arr_t, ind_t, ptr_t, len_t):
    def codegen(context, builder, sig, args):
        arr, ind, ptr, length = args
        payload = _get_str_binary_arr_payload(context, builder, arr, sig.args[0])
        offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )
        data_arr = context.make_helper(builder, char_arr_type, payload.data)
        data = get_data_ptr_cg(context, builder, data_arr)
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(offset_type.bitwidth).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(64),
            ],
        )
        fn_setitem = cgutils.get_or_insert_function(
            builder.module, fnty, name="setitem_string_array"
        )
        # kind doesn't matter since input is ASCII
        kind = context.get_constant(types.int32, -1)
        is_ascii = context.get_constant(types.int32, 1)
        num_total_chars = _get_num_total_chars(builder, offsets, payload.n_arrays)
        builder.call(
            fn_setitem,
            [
                offsets,
                data,
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
    fn_getitem = cgutils.get_or_insert_function(builder.module, fnty, name="is_na")
    return builder.call(fn_getitem, [bull_bitmap, ind])


@intrinsic
def _memcpy(typingctx, dest_t, src_t, count_t, item_size_t):
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


def inplace_eq(A, i, val):  # pragma: no cover
    return A[i] == val


@overload(inplace_eq)
def inplace_eq_overload(A, ind, val):
    """compare string array element to a string value inplace, without creating a string
    value from the element (which incurrs allocation overhead).
    """

    # TODO(ehsan): support dict encoded array

    def impl(A, ind, val):  # pragma: no cover
        utf8_str, utf8_len = unicode_to_utf8_and_len(val)
        start_offset = getitem_str_offset(A, ind)
        end_offset = getitem_str_offset(A, ind + 1)
        arr_val_len = end_offset - start_offset
        if arr_val_len != utf8_len:
            return False
        ptr = get_data_ptr_ind(A, start_offset)
        return memcmp(ptr, utf8_str, utf8_len) == 0

    return impl


def str_arr_setitem_int_to_str(A, ind, value):
    A[ind] = str(value)


@overload(str_arr_setitem_int_to_str)
def overload_str_arr_setitem_int_to_str(A, ind, val):
    """
    Set string array element to string representation of an integer value
    """

    def impl(A, ind, val):  # pragma: no cover
        # get pointer to string position and its length
        start_offset = getitem_str_offset(A, ind)
        arr_val_len = bodo.libs.str_ext.int_to_str_len(val)
        required_capacity = start_offset + arr_val_len
        bodo.libs.array_item_arr_ext.ensure_data_capacity(
            A._data, start_offset, required_capacity
        )
        ptr = get_data_ptr_ind(A, start_offset)
        # convert integer to string and write to output string position inplace
        inplace_int64_to_str(ptr, arr_val_len, val)
        # set end offset of string element
        setitem_str_offset(A, ind + 1, start_offset + arr_val_len)
        str_arr_set_not_na(A, ind)

    return impl


@intrinsic
def inplace_set_NA_str(typingctx, ptr_typ):
    """
    Write "<NA>" (string representation of pd.NA) to string pointer
    """

    def codegen(context, builder, sig, args):
        (ptr,) = args
        na_str = context.insert_const_string(builder.module, "<NA>")
        na_str_len = lir.Constant(lir.IntType(64), len("<NA>"))
        cgutils.raw_memcpy(builder, ptr, na_str, na_str_len, 1)

    return types.none(types.voidptr), codegen


def str_arr_setitem_NA_str(A, ind):
    A[ind] = "<NA>"


@overload(str_arr_setitem_NA_str)
def overload_str_arr_setitem_NA_str(A, ind):
    """
    Set string array element to "<NA>" (string representation of pd.NA)
    """
    na_len = len("<NA>")

    def impl(A, ind):  # pragma: no cover
        start_offset = getitem_str_offset(A, ind)
        required_capacity = start_offset + na_len
        bodo.libs.array_item_arr_ext.ensure_data_capacity(
            A._data, start_offset, required_capacity
        )
        ptr = get_data_ptr_ind(A, start_offset)
        inplace_set_NA_str(ptr)
        # set end offset of string element
        setitem_str_offset(A, ind + 1, start_offset + na_len)
        str_arr_set_not_na(A, ind)

    return impl


@overload(operator.getitem, no_unliteral=True)
def str_arr_getitem_int(A, ind):
    if A != string_array_type:
        return

    if isinstance(ind, types.Integer):
        # kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND
        def str_arr_getitem_impl(A, ind):  # pragma: no cover
            if ind < 0:
                ind += A.size
            start_offset = getitem_str_offset(A, np.int64(ind))
            end_offset = getitem_str_offset(A, np.int64(ind) + 1)
            length = end_offset - start_offset
            ptr = get_data_ptr_ind(A, start_offset)
            ret = decode_utf8(ptr, length)
            # ret = numba.cpython.unicode._empty_string(kind, length)
            # _memcpy(ret._data, ptr, length, 1)
            return ret

        return str_arr_getitem_impl

    # bool arr indexing.
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def bool_impl(A, ind):  # pragma: no cover
            # convert potential Series to array
            ind = bodo.utils.conversion.coerce_to_array(ind)
            n = len(A)
            n_strs = 0
            n_chars = 0
            for i in range(n):
                if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                    n_strs += 1
                    n_chars += get_str_arr_item_length(A, i)

            out_arr = pre_alloc_string_array(n_strs, n_chars)
            out_data_ptr = get_data_ptr(out_arr).data
            in_data_ptr = get_data_ptr(A).data
            str_ind = 0
            curr_offset = 0
            setitem_str_offset(out_arr, 0, 0)
            for i in range(n):
                if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                    # copy buffers directly and avoid extra string buffer allocation
                    # which impacts performance significantly (see TPC-H Q1)
                    # _str = A[i]
                    # out_arr[str_ind] = _str
                    char_len = get_str_arr_item_length(A, i)
                    # optimize empty or null string case
                    if char_len == 0:
                        pass
                    # optimize single char case since common (~10% faster)
                    elif char_len == 1:
                        copy_single_char(
                            out_data_ptr,
                            curr_offset,
                            in_data_ptr,
                            getitem_str_offset(A, i),
                        )
                    else:
                        memcpy_region(
                            out_data_ptr,
                            curr_offset,
                            in_data_ptr,
                            getitem_str_offset(A, i),
                            char_len,
                            1,
                        )

                    curr_offset += char_len
                    setitem_str_offset(out_arr, str_ind + 1, curr_offset)
                    # set NA
                    if str_arr_is_na(A, i):
                        str_arr_set_na(out_arr, str_ind)
                    else:
                        str_arr_set_not_na(out_arr, str_ind)
                    str_ind += 1
            return out_arr

        return bool_impl

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def str_arr_arr_impl(A, ind):  # pragma: no cover
            # convert potential Series to array
            ind = bodo.utils.conversion.coerce_to_array(ind)
            n = len(ind)
            n_chars = 0
            for i in range(n):
                # NOTE: NA values have valid offsets with 0 data length
                # this is low overhead, no need for optimizing for duplicate values
                n_chars += get_str_arr_item_length(A, ind[i])

            out_arr = pre_alloc_string_array(n, n_chars)
            out_data_ptr = get_data_ptr(out_arr).data
            in_data_ptr = get_data_ptr(A).data
            curr_offset = 0
            setitem_str_offset(out_arr, 0, 0)
            for i in range(n):
                if bodo.libs.array_kernels.isna(ind, i):
                    raise ValueError(
                        "Cannot index with an integer indexer containing NA values"
                    )
                arr_ind = ind[i]
                # copy buffers directly and avoid extra string buffer allocation
                # which impacts performance significantly (see TPC-H Q1)
                # _str = A[ind[i]]
                # out_arr[i] = _str
                char_len = get_str_arr_item_length(A, arr_ind)
                # optimize empty or null string case
                if char_len == 0:
                    pass
                # optimize single char case since common (~10% faster)
                elif char_len == 1:
                    copy_single_char(
                        out_data_ptr,
                        curr_offset,
                        in_data_ptr,
                        getitem_str_offset(A, arr_ind),
                    )
                else:
                    memcpy_region(
                        out_data_ptr,
                        curr_offset,
                        in_data_ptr,
                        getitem_str_offset(A, arr_ind),
                        char_len,
                        1,
                    )

                curr_offset += char_len
                setitem_str_offset(out_arr, i + 1, curr_offset)
                # set NA
                if str_arr_is_na(A, arr_ind):
                    str_arr_set_na(out_arr, i)
                else:
                    str_arr_set_not_na(out_arr, i)
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
                new_arr = pre_alloc_string_array(span, -1)
                # TODO: more efficient copy
                for i in range(span):
                    new_arr[i] = A[slice_idx.start + i * slice_idx.step]
                    # set NA
                    if str_arr_is_na(A, slice_idx.start + i * slice_idx.step):
                        str_arr_set_na(new_arr, i)
                return new_arr

        return str_arr_slice_impl

    # This should be the only StringArray implementation.
    # We only expect to reach this case if more ind options are added.
    raise BodoError(
        f"getitem for StringArray with indexing type {ind} not supported."
    )  # pragma: no cover


dummy_use = numba.njit(lambda a: None)


# TODO: support literals directly and turn on `no_unliteral=True`
# @overload(operator.setitem, no_unliteral=True)
@overload(operator.setitem)
def str_arr_setitem(A, idx, val):
    if A != string_array_type:
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    typ_err_msg = (
        f"StringArray setitem with index {idx} and value {val} not supported yet."
    )

    # scalar case
    if isinstance(idx, types.Integer):
        if val != string_type:
            raise BodoError(typ_err_msg)

        # XXX: setitem works only if value is same size as the previous value
        # maximum number of bytes possible in UTF-8 is 4
        MAX_UTF8_BYTES = 4

        def impl_scalar(A, idx, val):  # pragma: no cover
            # make sure data array has enough space for new characters to be stored
            # if string value is not ASCII, assume maximum possible UTF-8 characters
            max_val_len = val._length if val._is_ascii else MAX_UTF8_BYTES * val._length
            data_arr = A._data
            start_offset = np.int64(getitem_str_offset(A, idx))
            required_capacity = start_offset + max_val_len
            bodo.libs.array_item_arr_ext.ensure_data_capacity(
                data_arr, start_offset, required_capacity
            )
            setitem_string_array(
                get_offset_ptr(A),
                get_data_ptr(A),
                required_capacity,
                val._data,
                val._length,
                val._kind,
                val._is_ascii,
                idx,
            )
            str_arr_set_not_na(A, idx)
            # TODO(ehsan): trim data array if done writing all values?
            # dummy use function to avoid decref of A
            # TODO: refcounting support for _offsets, ... to avoid this workaround
            dummy_use(A)
            dummy_use(val)

        return impl_scalar

    # slice case
    if isinstance(idx, types.SliceType):
        if val == string_array_type:

            def impl_slice(A, idx, val):  # pragma: no cover
                slice_idx = numba.cpython.unicode._normalize_slice(idx, len(A))
                start = slice_idx.start
                data_arr = A._data
                start_offset = np.int64(getitem_str_offset(A, start))
                required_capacity = start_offset + np.int64(num_total_chars(val))
                bodo.libs.array_item_arr_ext.ensure_data_capacity(
                    data_arr, start_offset, required_capacity
                )
                set_string_array_range(A, val, start, start_offset)
                # nulls of input and output arrays should match
                curr = 0
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    if str_arr_is_na(val, curr):
                        str_arr_set_na(A, i)
                    else:
                        str_arr_set_not_na(A, i)
                    curr += 1

            return impl_slice

        # slice with list
        elif isinstance(val, types.List) and val.dtype == string_type:

            def impl_slice_list(A, idx, val):  # pragma: no cover
                val_arr = str_list_to_array(val)
                A[idx] = val_arr

            return impl_slice_list

        # slice with scalar
        elif val == string_type:

            def impl_slice(A, idx, val):  # pragma: no cover
                slice_idx = numba.cpython.unicode._normalize_slice(idx, len(A))
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    A[i] = val

            return impl_slice

        else:
            raise BodoError(typ_err_msg)

    # set scalar value using bool index
    # NOTE: this changes the array inplace after construction
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        if val == string_type:

            def impl_bool_scalar(A, idx, val):  # pragma: no cover
                n = len(A)
                # NOTE: necessary to convert potential Series to array
                idx = bodo.utils.conversion.coerce_to_array(idx)
                out_arr = pre_alloc_string_array(n, -1)
                for i in numba.parfors.parfor.internal_prange(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        out_arr[i] = val
                    else:
                        if bodo.libs.array_kernels.isna(A, i):
                            out_arr[i] = ""
                            str_arr_set_na(out_arr, i)
                        else:
                            get_str_arr_item_copy(out_arr, i, A, i)

                move_str_binary_arr_payload(A, out_arr)

            return impl_bool_scalar

        elif val == string_array_type or (
            isinstance(val, types.Array) and isinstance(val.dtype, types.UnicodeCharSeq)
        ):

            def impl_bool_arr(A, idx, val):  # pragma: no cover
                n = len(A)
                # NOTE: necessary to convert potential Series to array
                idx = bodo.utils.conversion.coerce_to_array(
                    idx, use_nullable_array=True
                )
                out_arr = pre_alloc_string_array(n, -1)
                ind_count = 0
                for i in numba.parfors.parfor.internal_prange(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        if bodo.libs.array_kernels.isna(val, ind_count):
                            out_arr[i] = ""
                            str_arr_set_na(out_arr, ind_count)
                        else:
                            # Convert to a string to support UnicodeCharSeq
                            out_arr[i] = str(val[ind_count])
                        ind_count += 1
                    else:
                        if bodo.libs.array_kernels.isna(A, i):
                            out_arr[i] = ""
                            str_arr_set_na(out_arr, i)
                        else:
                            get_str_arr_item_copy(out_arr, i, A, i)

                move_str_binary_arr_payload(A, out_arr)

            return impl_bool_arr

        else:
            raise BodoError(typ_err_msg)

    # TODO: other setitem cases
    raise BodoError(typ_err_msg)


@overload_attribute(StringArrayType, "dtype")
def overload_str_arr_dtype(A):
    return lambda A: pd.StringDtype()  # pragma: no cover


@overload_attribute(StringArrayType, "ndim")
def overload_str_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_method(StringArrayType, "astype", no_unliteral=True)
def overload_str_arr_astype(A, dtype, copy=True):
    # If dtype is a string, force it to be a literal
    if dtype == types.unicode_type:
        raise_bodo_error(
            "StringArray.astype(): 'dtype' when passed as string must be a constant value"
        )

    # same dtype case with str. Here we opt to cause both dict
    # and regular string arrays maintain the same type.
    if isinstance(dtype, types.Function) and dtype.key[0] == str:
        # no need to copy since our StringArray is immutable
        return lambda A, dtype, copy=True: A  # pragma: no cover

    # numpy dtypes
    nb_dtype = parse_dtype(dtype, "StringArray.astype")

    if A == nb_dtype:
        # same dtype case when passing an array typeref.
        # no need to copy since our StringArray is immutable
        return lambda A, dtype, copy=True: A  # pragma: no cover

    # TODO: support other dtypes if any
    # TODO: error checking
    if not isinstance(nb_dtype, (types.Float, types.Integer)) and nb_dtype not in (
        types.bool_,
        bodo.libs.bool_arr_ext.boolean_dtype,
        bodo.dict_str_arr_type,
    ):  # pragma: no cover
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
    elif nb_dtype == types.bool_:

        def impl_bool(A, dtype, copy=True):  # pragma: no cover
            numba.parfors.parfor.init_prange()  # TODO: test fusion
            n = len(A)
            B = np.empty(n, nb_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(A, i):
                    B[i] = False
                else:
                    B[i] = bool(A[i])
            return B

        return impl_bool

    elif nb_dtype == bodo.libs.bool_arr_ext.boolean_dtype:

        def impl_bool(A, dtype, copy=True):  # pragma: no cover
            numba.parfors.parfor.init_prange()  # TODO: test fusion
            n = len(A)
            B = np.empty(n, nb_dtype)
            for i in numba.parfors.parfor.internal_prange(n):
                if bodo.libs.array_kernels.isna(A, i):
                    bodo.libs.array_kernels.setna(B, i)
                else:
                    B[i] = bool(A[i])
            return B

        return impl_bool

    elif nb_dtype == bodo.dict_str_arr_type:

        def impl_dict_str(A, dtype, copy=True):  # pragma: no cover
            return str_arr_to_dict_str_arr(A)

        return impl_dict_str

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


@numba.njit(no_cpython_wrapper=True)
def str_arr_to_dict_str_arr(A):  # pragma: no cover
    return str_arr_to_dict_str_arr_cpp(A)


@intrinsic
def str_arr_to_dict_str_arr_cpp(typingctx, str_arr_t):
    def codegen(context, builder, sig, args):
        (str_arr,) = args

        str_arr_info = bodo.libs.array.array_to_info_codegen(
            context, builder, bodo.libs.array.array_info_type(sig.args[0]), (str_arr,)
        )

        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="str_to_dict_str_array"
        )
        dict_array_info = builder.call(fn_tp, [str_arr_info])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

        dict_arr = bodo.libs.array.info_to_array_codegen(
            context,
            builder,
            sig.return_type(bodo.libs.array.array_info_type, sig.return_type),
            (dict_array_info, context.get_constant_null(sig.return_type)),
        )

        # delete output array_info
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(builder.module, fnty, name="delete_info")
        builder.call(fn_tp, [dict_array_info])

        return dict_arr

    assert (
        str_arr_t == bodo.string_array_type
    ), "str_arr_to_dict_str_arr: Input Array is not a Bodo String Array"

    sig = bodo.dict_str_arr_type(bodo.string_array_type)
    return sig, codegen


@intrinsic
def decode_utf8(typingctx, ptr_t, len_t):
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


def set_to_numeric_out_na_err(out_arr, out_ind, err_code):  # pragma: no cover
    pass


@overload(set_to_numeric_out_na_err)
def set_to_numeric_out_na_err_overload(out_arr, out_ind, err_code):
    """set NA to output of to_numeric() based on error code from C++ code."""
    # nullable int array
    if isinstance(out_arr, bodo.libs.int_arr_ext.IntegerArrayType):

        def impl_int(out_arr, out_ind, err_code):  # pragma: no cover
            bodo.libs.int_arr_ext.set_bit_to_arr(
                out_arr._null_bitmap, out_ind, 0 if err_code == -1 else 1
            )

        return impl_int

    # numpy case, TODO: other
    assert isinstance(out_arr, types.Array)

    if isinstance(out_arr.dtype, types.Float):

        def impl_np(out_arr, out_ind, err_code):  # pragma: no cover
            if err_code == -1:
                out_arr[out_ind] = np.nan

        return impl_np

    return lambda out_arr, out_ind, err_code: None  # pragma: no cover


@numba.njit(no_cpython_wrapper=True)
def str_arr_item_to_numeric(out_arr, out_ind, str_arr, ind):  # pragma: no cover
    err_code = _str_arr_item_to_numeric(
        get_arr_data_ptr(out_arr, out_ind),
        str_arr,
        ind,
        out_arr.dtype,
    )
    set_to_numeric_out_na_err(out_arr, out_ind, err_code)


@intrinsic
def _str_arr_item_to_numeric(typingctx, out_ptr_t, str_arr_t, ind_t, out_dtype_t):
    assert str_arr_t == string_array_type, "_str_arr_item_to_numeric: str arr expected"
    assert ind_t == types.int64, "_str_arr_item_to_numeric: integer index expected"

    def codegen(context, builder, sig, args):
        # TODO: return tuple with value and error and avoid array arg?
        out_ptr, arr, ind, _dtype = args
        payload = _get_str_binary_arr_payload(context, builder, arr, string_array_type)
        offsets = builder.bitcast(
            context.nrt.meminfo_data(builder, payload.offsets),
            context.get_data_type(offset_type).as_pointer(),
        )
        data_arr = context.make_helper(builder, char_arr_type, payload.data)
        data = get_data_ptr_cg(context, builder, data_arr)

        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                out_ptr.type,
                lir.IntType(offset_type.bitwidth).as_pointer(),
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
        fn_to_numeric = cgutils.get_or_insert_function(builder.module, fnty, fname)
        return builder.call(fn_to_numeric, [out_ptr, offsets, data, ind])

    return types.int32(out_ptr_t, string_array_type, types.int64, out_dtype_t), codegen


class MinOrMax(Enum):
    Min = 1
    Max = 2


def str_arr_min_max_seq(arr, min_or_max):  # pragma: no cover
    pass


# NOTE: no_unliteral=True is necessary for min_or_max argument to be constant
@overload(str_arr_min_max_seq, no_unliteral=True)
def overload_str_arr_min_max_seq(arr, min_or_max):
    """String array min/max sequential implementation"""
    # TODO: optimize for dictionary-encoded case
    assert is_str_arr_type(arr), "str_arr_min_max: string array expected"
    assert_bodo_error(
        is_overload_constant_int(min_or_max),
        "str_arr_min_max: min_or_max should be constant int",
    )

    min_or_max = get_overload_const_int(min_or_max)
    min_max_func = max if min_or_max == MinOrMax.Max.value else min

    def impl_str_arr_min_max(arr, min_or_max):  # pragma: no cover
        s = None
        for i in range(len(arr)):
            if not bodo.libs.array_kernels.isna(arr, i):
                v = arr[i]
                if s is None:
                    s = v
                else:
                    s = min_max_func(s, v)
        return s

    return impl_str_arr_min_max


def str_arr_min_max(arr, min_or_max, parallel=False):  # pragma: no cover
    pass


# NOTE: no_unliteral=True is necessary for min_or_max argument to be constant
@overload(str_arr_min_max, no_unliteral=True)
def overload_str_arr_min_max(arr, min_or_max, parallel=False):
    """String array min/max implementation"""
    # TODO: optimize for dictionary-encoded case
    assert is_str_arr_type(arr), "str_arr_min_max: string array expected"
    assert_bodo_error(
        is_overload_constant_int(min_or_max),
        "str_arr_min_max: min_or_max should be constant int",
    )

    def impl_str_arr_min_max(arr, min_or_max, parallel=False):  # pragma: no cover
        s = str_arr_min_max_seq(arr, min_or_max)

        if parallel:
            nchars = 0 if s is None else len(bodo.utils.indexing.unoptional(s))
            loc_val_arr = pre_alloc_string_array(1, nchars)
            loc_val_arr[0] = s
            vals_arr = bodo.allgatherv(loc_val_arr)
            return str_arr_min_max_seq(vals_arr, min_or_max)

        return s

    return impl_str_arr_min_max


@unbox(BinaryArrayType)
@unbox(StringArrayType)
def unbox_str_array(typ, val, c):
    """
    Unbox a numpy object array with string object values.
    """

    is_bytes = c.context.get_constant(types.int32, int(typ == binary_array_type))

    string_array = c.context.make_helper(c.builder, typ)
    array_item_data_type = ArrayItemArrayType(char_arr_type)
    array_item_array = c.context.make_helper(c.builder, array_item_data_type)

    # create payload type
    payload_type = ArrayItemArrayPayloadType(array_item_data_type)
    alloc_type = c.context.get_value_type(payload_type)
    alloc_size = c.context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_array_item_dtor(
        c.context, c.builder, array_item_data_type, payload_type
    )

    # create meminfo
    meminfo = c.context.nrt.meminfo_alloc_dtor(
        c.builder, c.context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = c.context.nrt.meminfo_data(c.builder, meminfo)
    meminfo_data_ptr = c.builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # alloc values in payload
    payload = cgutils.create_struct_proxy(payload_type)(c.context, c.builder)
    char_arr = cgutils.create_struct_proxy(char_arr_type)(c.context, c.builder)

    # TODO: check python errors
    # function signature: NRT_MemInfo* string_array_from_sequence(PyObject* obj)
    # returns meminfo for underlying array(item) array
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64).as_pointer(),
            lir.IntType(64).as_pointer(),
            c.context.get_value_type(types.MemInfoPointer(char_type)).as_pointer(),
            c.context.get_value_type(types.MemInfoPointer(offset_type)).as_pointer(),
            c.context.get_value_type(types.MemInfoPointer(types.uint8)).as_pointer(),
            lir.IntType(32),
        ],
    )
    fn = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="string_array_from_sequence"
    )
    c.builder.call(
        fn,
        [
            val,
            payload._get_ptr_by_name("n_arrays"),
            char_arr._get_ptr_by_name("length"),
            char_arr._get_ptr_by_name("meminfo"),
            payload._get_ptr_by_name("offsets"),
            payload._get_ptr_by_name("null_bitmap"),
            is_bytes,
        ],
    )

    char_arr.meminfo_offset = c.context.get_constant(types.int64, 0)
    payload.data = char_arr._getvalue()
    c.builder.store(payload._getvalue(), meminfo_data_ptr)
    array_item_array.meminfo = meminfo
    string_array.data = array_item_array._getvalue()

    # FIXME how to check that the returned size is > 0?
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(string_array._getvalue(), is_error=is_error)


@lower_constant(BinaryArrayType)
@lower_constant(StringArrayType)
def lower_constant_str_arr(context, builder, typ, pyval):
    """embed constant string array value by getting constant values for underlying
    array(item) data arrays.
    """
    # create offsets, chars, nulls arrays from Python array of string objects
    # converts all strings to utf8 bytes and appends to a char list to create char array
    n = len(pyval)
    curr_offset = 0
    offset_arr = np.empty(n + 1, np_offset_type)
    char_list = []
    nulls_arr = np.empty((n + 7) >> 3, np.uint8)
    for i, s in enumerate(pyval):
        offset_arr[i] = curr_offset
        is_na = pd.isna(s)
        bodo.libs.int_arr_ext.set_bit_to_arr(nulls_arr, i, int(not is_na))
        if is_na:
            continue
        str_chars = list(s.encode()) if isinstance(s, str) else list(s)
        char_list.extend(str_chars)
        curr_offset += len(str_chars)

    offset_arr[n] = curr_offset
    char_arr = np.array(char_list, np.uint8)

    # get lowered constants for required attributes
    n_const = context.get_constant(types.int64, n)

    # create constant meminfos for offsets and nulls
    # see here: https://github.com/Bodo-inc/Bodo/blob/90e1bcfa82d588f08e2d2dd27f9b28266015da42/bodo/numba_compat.py#L4445
    chars_const = cgutils.create_constant_array(
        lir.IntType(8), bytearray(char_arr.data)
    )
    chars_const = cgutils.global_constant(
        builder, ".const.str_array.chars", chars_const
    )

    offsets_const = cgutils.create_constant_array(
        lir.IntType(8), bytearray(offset_arr.data)
    )
    offsets_const = cgutils.global_constant(
        builder, ".const.str_array.offsets", offsets_const
    )

    nulls_const = cgutils.create_constant_array(
        lir.IntType(8), bytearray(nulls_arr.data)
    )
    nulls_const = cgutils.global_constant(
        builder, ".const.str_array.nullbitmap", nulls_const
    )

    minus_one = context.get_constant(types.int64, -1)
    null_ptr = context.get_constant_null(types.voidptr)
    offsets_meminfo = lir.Constant.literal_struct(
        [minus_one, null_ptr, null_ptr, offsets_const, minus_one]
    )
    offsets_meminfo = cgutils.global_constant(
        builder, ".const.offsets_meminfo", offsets_meminfo
    ).bitcast(cgutils.voidptr_t)
    nulls_meminfo = lir.Constant.literal_struct(
        [minus_one, null_ptr, null_ptr, nulls_const, minus_one]
    )
    nulls_meminfo = cgutils.global_constant(
        builder, ".const.nulls_meminfo", nulls_meminfo
    ).bitcast(cgutils.voidptr_t)
    chars_meminfo = lir.Constant.literal_struct(
        [minus_one, null_ptr, null_ptr, chars_const, minus_one]
    )
    chars_meminfo = cgutils.global_constant(
        builder, ".const.chars_meminfo", chars_meminfo
    ).bitcast(cgutils.voidptr_t)

    char_arr_const = lir.Constant.literal_struct(
        [
            context.get_constant(types.int64, len(char_arr)),
            chars_meminfo,
            context.get_constant(types.int64, 0),
        ]
    )

    # create array(item) data array

    # create a constant payload with the same data model as ArrayItemArrayPayloadType
    # "n_arrays", "data", "offsets", "null_bitmap"
    payload = lir.Constant.literal_struct(
        [n_const, char_arr_const, offsets_meminfo, nulls_meminfo]
    )
    payload = cgutils.global_constant(builder, ".const.payload", payload).bitcast(
        cgutils.voidptr_t
    )

    # create a constant meminfo with the same data model as Numba:
    # https://github.com/numba/numba/blob/0499b906a850af34f0e2fdcc6b3b3836cdc95297/numba/core/runtime/nrtdynmod.py#L14
    # https://github.com/numba/numba/blob/2776e1a7cf49aeb513e0319fe4a94a12836a995b/numba/core/runtime/nrt.c#L16
    # we set refcount=-1 to avoid calling the destructor (see _define_atomic_inc_dec
    # patch in numba_compat and test_constant_lowering_refcount)
    minus_one = context.get_constant(types.int64, -1)
    null_ptr = context.get_constant_null(types.voidptr)
    meminfo = lir.Constant.literal_struct(
        [minus_one, null_ptr, null_ptr, payload, minus_one]
    )
    meminfo = cgutils.global_constant(builder, ".const.meminfo", meminfo).bitcast(
        cgutils.voidptr_t
    )

    # literal struct with the same model as ArrayItemArrayType
    data_arr = lir.Constant.literal_struct([meminfo])
    # literal struct with the same model as StringArrayType
    string_array = lir.Constant.literal_struct([data_arr])
    return string_array


# TODO: array analysis and remove call for other functions


def pre_alloc_str_arr_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 2 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


from numba.parfors.array_analysis import ArrayAnalysis

ArrayAnalysis._analyze_op_call_bodo_libs_str_arr_ext_pre_alloc_string_array = (  # type: ignore
    pre_alloc_str_arr_equiv
)


#### glob support #####


@overload(glob.glob, no_unliteral=True)
def overload_glob_glob(pathname, recursive=False):
    def _glob_glob_impl(pathname, recursive=False):  # pragma: no cover
        with bodo.objmode(l="list_str_type"):
            l = glob.glob(pathname, recursive=recursive)
        return l

    return _glob_glob_impl
