# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Array implementation for list of string objects (e.g. from S.str.split()), which are
usually immutable.
The characters are stored in a contingous data array, and two arrays of offsets mark
the individual strings and lists. For example:
value:             [['a', 'bc'], ['a'], ['aaa', 'b', 'cc']]
data:              [a, b, c, a, a, a, a, b, c, c]
data_offsets:      [0, 1, 3, 4, 7, 8, 10]
index_offsets:     [0, 2, 3, 6]
"""
import operator
import numpy as np
import numba
import bodo
from numba import types
from numba.typing.templates import (
    infer_global,
    AbstractTemplate,
    infer,
    signature,
    AttributeTemplate,
    infer_getattr,
    bound_function,
)
import numba.typing.typeof
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
    lower_getattr,
    intrinsic,
    overload_method,
    overload,
    overload_attribute,
)
from numba import cgutils
from bodo.libs.str_ext import string_type
from numba.targets.imputils import (
    impl_ret_new_ref,
    impl_ret_borrowed,
    iternext_impl,
    RefType,
)
from bodo.utils.typing import is_list_like_index_type
import llvmlite.llvmpy.core as lc
from glob import glob
from bodo.utils.typing import is_overload_true, is_overload_none
from llvmlite import ir as lir
import llvmlite.binding as ll
from bodo.libs import hstr_ext

ll.add_symbol("list_string_array_from_sequence", hstr_ext.list_string_array_from_sequence)
ll.add_symbol("dtor_list_string_array", hstr_ext.dtor_list_string_array)
ll.add_symbol("np_array_from_list_string_array", hstr_ext.np_array_from_list_string_array)
ll.add_symbol("allocate_list_string_array", hstr_ext.allocate_list_string_array)


char_typ = types.uint8
offset_typ = types.uint32

data_ctypes_type = types.ArrayCTypes(types.Array(char_typ, 1, "C"))
offset_ctypes_type = types.ArrayCTypes(types.Array(offset_typ, 1, "C"))


class ListStringArrayType(types.ArrayCompatible):
    def __init__(self):
        super(ListStringArrayType, self).__init__(name="ListStringArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return types.List(string_type)

    def copy(self):
        return ListStringArrayType()


list_string_array_type = ListStringArrayType()


class ListStringArrayPayloadType(types.Type):
    def __init__(self):
        super(ListStringArrayPayloadType, self).__init__(name="ListStringArrayPayloadType()")


list_str_arr_payload_type = ListStringArrayPayloadType()


# XXX: C equivalent in _str_ext.cpp
@register_model(ListStringArrayPayloadType)
class ListStringArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.CPointer(char_typ)),
            ("data_offsets", types.CPointer(offset_typ)),
            ("index_offsets", types.CPointer(offset_typ)),
            ("null_bitmap", types.CPointer(char_typ)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


list_str_arr_model_members = [
    ("num_items", types.uint64),
    ("num_total_strings", types.uint64),
    ("num_total_chars", types.uint64),
    ("data", types.CPointer(char_typ)),
    ("data_offsets", types.CPointer(offset_typ)),
    ("index_offsets", types.CPointer(offset_typ)),
    ("null_bitmap", types.CPointer(char_typ)),
    ("meminfo", types.MemInfoPointer(list_str_arr_payload_type)),
]


@register_model(ListStringArrayType)
class ListStringArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        models.StructModel.__init__(self, dmm, fe_type, list_str_arr_model_members)


# XXX: should these be exposed?
make_attribute_wrapper(ListStringArrayType, "num_items", "_num_items")
make_attribute_wrapper(ListStringArrayType, "num_total_strings", "_num_total_strings")
make_attribute_wrapper(ListStringArrayType, "num_total_chars", "_num_total_chars")
make_attribute_wrapper(ListStringArrayType, "data", "_data")
make_attribute_wrapper(ListStringArrayType, "data_offsets", "_data_offsets")
make_attribute_wrapper(ListStringArrayType, "index_offsets", "_index_offsets")
make_attribute_wrapper(ListStringArrayType, "null_bitmap", "_null_bitmap")


def construct_list_string_array(context, builder):
    """Creates meminfo and sets dtor.
    """
    alloc_type = context.get_data_type(list_str_arr_payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    dtor_ftype = lir.FunctionType(lir.VoidType(), [llvoidptr, llsize, llvoidptr])
    dtor_fn = builder.module.get_or_insert_function(
        dtor_ftype, name="dtor_list_string_array"
    )

    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_data_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_data_ptr, alloc_type.as_pointer())

    return meminfo, meminfo_data_ptr


@unbox(ListStringArrayType)
def unbox_str_series(typ, val, c):
    """
    Unbox a numpy array with list of string data values.
    """
    payload = cgutils.create_struct_proxy(
        list_str_arr_payload_type)(c.context, c.builder)
    list_string_array = c.context.make_helper(c.builder, typ)

    # function signature of list_string_array_from_sequence
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),                    # obj
            lir.IntType(64).as_pointer(),                   # num_items (pointer)
            lir.IntType(8).as_pointer().as_pointer(),      # data
            lir.IntType(32).as_pointer().as_pointer(),       # data_offsets
            lir.IntType(32).as_pointer().as_pointer(),       # index_offsets
            lir.IntType(8).as_pointer().as_pointer(),       # null_bitmap
        ],
    )
    fn = c.builder.module.get_or_insert_function(
        fnty, name="list_string_array_from_sequence"
    )
    c.builder.call(
        fn,
        [
            val,
            list_string_array._get_ptr_by_name("num_items"),
            payload._get_ptr_by_name("data"),
            payload._get_ptr_by_name("data_offsets"),
            payload._get_ptr_by_name("index_offsets"),
            payload._get_ptr_by_name("null_bitmap"),
        ],
    )

    meminfo, meminfo_data_ptr = construct_list_string_array(c.context, c.builder)
    c.builder.store(payload._getvalue(), meminfo_data_ptr)

    list_string_array.meminfo = meminfo
    list_string_array.data = payload.data
    list_string_array.data_offsets = payload.data_offsets
    list_string_array.index_offsets = payload.index_offsets
    list_string_array.null_bitmap = payload.null_bitmap
    list_string_array.num_total_strings = c.builder.zext(
        c.builder.load(c.builder.gep(list_string_array.index_offsets,
        [list_string_array.num_items])),
        lir.IntType(64),
    )
    list_string_array.num_total_chars = c.builder.zext(
        c.builder.load(c.builder.gep(list_string_array.data_offsets,
        [list_string_array.num_total_strings])),
        lir.IntType(64),
    )

    # FIXME how to check that the returned size is > 0?
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(list_string_array._getvalue(), is_error=is_error)


@box(ListStringArrayType)
def box_list_str_arr(typ, val, c):
    """box packed native representation of list of string array into python objects
    """
    list_string_array = c.context.make_helper(c.builder, typ, val)

    fnty = lir.FunctionType(
        c.context.get_argument_type(types.pyobject),
        [
            lir.IntType(64),                # num_items
            lir.IntType(8).as_pointer(),    # data
            lir.IntType(32).as_pointer(),   # data_offsets
            lir.IntType(32).as_pointer(),   # index_offsets
            lir.IntType(8).as_pointer(),    # null_bitmap
        ],
    )
    fn_get = c.builder.module.get_or_insert_function(
        fnty, name="np_array_from_list_string_array"
    )
    arr = c.builder.call(
        fn_get,
        [
            list_string_array.num_items,
            list_string_array.data,
            list_string_array.data_offsets,
            list_string_array.index_offsets,
            list_string_array.null_bitmap,
        ],
    )

    return arr


@overload(len)
def overload_len_list_str_arr(A):
    if A == list_string_array_type:
        return lambda A: A._num_items


@overload_attribute(ListStringArrayType, "shape")
def overload_list_str_arr_shape(A):
    return lambda A: (A._num_items,)


@intrinsic
def pre_alloc_list_string_array(typingctx, num_lists_typ, num_strs_typ, num_chars_typ=None):
    assert isinstance(num_lists_typ, types.Integer) and isinstance(num_strs_typ, types.Integer) and isinstance(
        num_chars_typ, types.Integer
    )

    def codegen(context, builder, sig, args):
        num_lists, num_strs, num_chars = args
        meminfo, meminfo_data_ptr = construct_list_string_array(context, builder)

        list_str_arr_payload = cgutils.create_struct_proxy(list_str_arr_payload_type)(
            context, builder
        )
        extra_null_bytes = context.get_constant(types.int64, 0)

        # allocate string array
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer().as_pointer(),
                lir.IntType(32).as_pointer().as_pointer(),
                lir.IntType(32).as_pointer().as_pointer(),
                lir.IntType(8).as_pointer().as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn_alloc = builder.module.get_or_insert_function(
            fnty, name="allocate_list_string_array"
        )
        builder.call(
            fn_alloc,
            [
                list_str_arr_payload._get_ptr_by_name("data"),
                list_str_arr_payload._get_ptr_by_name("data_offsets"),
                list_str_arr_payload._get_ptr_by_name("index_offsets"),
                list_str_arr_payload._get_ptr_by_name("null_bitmap"),
                num_lists,
                num_strs,
                num_chars,
                extra_null_bytes,
            ],
        )

        builder.store(list_str_arr_payload._getvalue(), meminfo_data_ptr)
        list_string_array = context.make_helper(builder, list_string_array_type)
        list_string_array.num_items = num_lists
        list_string_array.num_total_strings = num_strs
        list_string_array.num_total_chars = num_chars
        list_string_array.data = list_str_arr_payload.data
        list_string_array.data_offsets = list_str_arr_payload.data_offsets
        list_string_array.index_offsets = list_str_arr_payload.index_offsets
        list_string_array.null_bitmap = list_str_arr_payload.null_bitmap
        list_string_array.meminfo = meminfo
        ret = list_string_array._getvalue()

        return impl_ret_new_ref(context, builder, list_string_array_type, ret)

    return list_string_array_type(types.intp, types.intp, types.intp), codegen


@overload(operator.getitem)
def list_str_arr_getitem_array(arr, ind):
    if arr != list_string_array_type:
        return

    if isinstance(ind, types.Integer):
        # XXX: cannot handle NA for scalar getitem since not type stable
        def list_str_arr_getitem_impl(arr, ind):
            l_start_offset = arr._index_offsets[ind]
            l_end_offset = arr._index_offsets[ind + 1]
            out = []
            for i in range(l_start_offset, l_end_offset):
                start_offset = arr._data_offsets[i]
                end_offset = arr._data_offsets[i + 1]
                length = end_offset - start_offset
                ptr = bodo.hiframes.split_impl.get_c_arr_ptr(arr._data, start_offset)
                out.append(bodo.libs.str_arr_ext.decode_utf8(ptr, length))
            return out

        return list_str_arr_getitem_impl

    # bool arr indexing
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(arr, ind):
            n = len(arr)
            if n != len(ind):
                raise IndexError(
                    "boolean index did not match indexed array along dimension 0"
                )
            n_lists = 0
            n_strs = 0
            n_chars = 0
            for i in range(n):
                if ind[i]:
                    n_lists += 1
                    l_start_offset = arr._index_offsets[i]
                    l_end_offset = arr._index_offsets[i + 1]
                    for j in range(l_start_offset, l_end_offset):
                        n_strs += 1
                        n_chars += arr._data_offsets[j + 1] - arr._data_offsets[j]
            out_arr = pre_alloc_list_string_array(n_lists, n_strs, n_chars)
            out_ind = 0
            curr_s_offset = 0
            curr_d_offset = 0
            for ii in range(n):
                if ind[ii]:
                    l_start_offset = arr._index_offsets[ii]
                    l_end_offset = arr._index_offsets[ii + 1]
                    n_str = l_end_offset - l_start_offset
                    str_ind = 0
                    for jj in range(l_start_offset, l_end_offset):
                        out_arr._data_offsets[curr_s_offset + str_ind] = curr_d_offset
                        n_char = arr._data_offsets[jj + 1] - arr._data_offsets[jj]
                        in_ptr = bodo.hiframes.split_impl.get_c_arr_ptr(arr._data, arr._data_offsets[jj])
                        out_ptr = bodo.hiframes.split_impl.get_c_arr_ptr(out_arr._data, curr_d_offset)
                        bodo.libs.str_arr_ext._memcpy(out_ptr, in_ptr, n_char, 1)
                        curr_d_offset += n_char
                        str_ind += 1

                    out_arr._index_offsets[out_ind] = curr_s_offset
                    curr_s_offset += n_str
                    # set NA
                    if bodo.libs.str_arr_ext.str_arr_is_na(arr, ii):
                        bodo.libs.str_arr_ext.str_arr_set_na(out_arr, out_ind)
                    out_ind += 1
            out_arr._index_offsets[out_ind] = curr_s_offset
            out_arr._data_offsets[curr_s_offset] = curr_d_offset
            return out_arr

        return impl_bool

    # ind arr indexing
    # TODO: avoid code duplication
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl_int(arr, ind):
            n = len(ind)
            n_lists = 0
            n_strs = 0
            n_chars = 0
            for k in range(n):
                i = ind[k]
                n_lists += 1
                l_start_offset = arr._index_offsets[i]
                l_end_offset = arr._index_offsets[i + 1]
                for j in range(l_start_offset, l_end_offset):
                    n_strs += 1
                    n_chars += arr._data_offsets[j + 1] - arr._data_offsets[j]
            out_arr = pre_alloc_list_string_array(n_lists, n_strs, n_chars)
            out_ind = 0
            curr_s_offset = 0
            curr_d_offset = 0
            for kk in range(n):
                ii = ind[kk]
                l_start_offset = arr._index_offsets[ii]
                l_end_offset = arr._index_offsets[ii + 1]
                n_str = l_end_offset - l_start_offset
                str_ind = 0
                for jj in range(l_start_offset, l_end_offset):
                    out_arr._data_offsets[curr_s_offset + str_ind] = curr_d_offset
                    n_char = arr._data_offsets[jj + 1] - arr._data_offsets[jj]
                    in_ptr = bodo.hiframes.split_impl.get_c_arr_ptr(arr._data, arr._data_offsets[jj])
                    out_ptr = bodo.hiframes.split_impl.get_c_arr_ptr(out_arr._data, curr_d_offset)
                    bodo.libs.str_arr_ext._memcpy(out_ptr, in_ptr, n_char, 1)
                    curr_d_offset += n_char
                    str_ind += 1

                out_arr._index_offsets[out_ind] = curr_s_offset
                curr_s_offset += n_str
                # set NA
                if bodo.libs.str_arr_ext.str_arr_is_na(arr, ii):
                    bodo.libs.str_arr_ext.str_arr_set_na(out_arr, out_ind)
                out_ind += 1
            out_arr._index_offsets[out_ind] = curr_s_offset
            out_arr._data_offsets[curr_s_offset] = curr_d_offset
            return out_arr

        return impl_int


    # slice case
    if isinstance(ind, types.SliceType):

        def impl_slice(arr, ind):
            n = len(arr)
            slice_idx = numba.unicode._normalize_slice(ind, n)
            # reusing integer array slicing above
            arr_ind = np.arange(slice_idx.start, slice_idx.stop, slice_idx.step)
            return arr[arr_ind]

        return impl_slice


@overload_method(ListStringArrayType, "copy")
def overload_list_str_arr_copy(A):
    def copy_impl(A):
        n_lists = A._num_items
        n_strs = A._num_total_strings
        n_chars = A._num_total_chars
        B = pre_alloc_list_string_array(n_lists, n_strs, n_chars)
        offset_typ_size = 4  # uint32 offsets
        bodo.libs.str_arr_ext._memcpy(B._index_offsets, A._index_offsets, n_lists + 1, offset_typ_size)
        bodo.libs.str_arr_ext._memcpy(B._data_offsets, A._data_offsets, n_strs + 1, offset_typ_size)
        bodo.libs.str_arr_ext._memcpy(B._data, A._data, n_chars, 1)
        n_bytes = (n_lists + 7) >> 3
        bodo.libs.str_arr_ext._memcpy(B._null_bitmap, A._null_bitmap, n_bytes, 1)
        return B
    return copy_impl
