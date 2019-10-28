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
from numba.targets.hashing import _Py_hash_t
import llvmlite.llvmpy.core as lc
from glob import glob
from bodo.utils.typing import is_overload_true, is_overload_none
from llvmlite import ir as lir
import llvmlite.binding as ll
from bodo.libs import hstr_ext

ll.add_symbol("list_string_array_from_sequence", hstr_ext.list_string_array_from_sequence)
ll.add_symbol("dtor_list_string_array", hstr_ext.dtor_list_string_array)


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
