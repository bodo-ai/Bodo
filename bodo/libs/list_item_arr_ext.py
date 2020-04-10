# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Array implementation for list of fixed size items, which are usually immutable.
Corresponds to Spark's ArrayType: https://spark.apache.org/docs/latest/sql-reference.html
Corresponds to Arrow's Variable-size List: https://arrow.apache.org/docs/format/Columnar.html

The values are stored in a contingous data array, while an offsets array marks the
individual lists. For example:
value:             [[1, 2], [3], None, [5, 4, 6], []]
data:              [1, 2, 3, 5, 4, 6]
offsets:           [0, 2, 3, 3, 6, 6]
"""
import operator
import numpy as np
import numba
import bodo

from numba import types
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

from bodo.utils.typing import is_list_like_index_type
from llvmlite import ir as lir
import llvmlite.binding as ll


# offset index types
offset_typ = types.uint32


class ListItemArrayType(types.ArrayCompatible):
    def __init__(self, elem_type):
        self.elem_type = elem_type
        super(ListItemArrayType, self).__init__(
            name="ListItemArrayType({})".format(elem_type)
        )

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return types.List(self.elem_type)

    def copy(self):
        return ListItemArrayType(self.elem_type)


class ListItemArrayPayloadType(types.Type):
    def __init__(self, list_type):
        self.list_type = list_type
        super(ListItemArrayPayloadType, self).__init__(
            name="ListItemArrayPayloadType({})".format(list_type)
        )


@register_model(ListItemArrayPayloadType)
class ListItemArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("n_lists", types.int64),
            ("data", types.Array(fe_type.list_type.elem_type, 1, "C")),
            ("offsets", types.Array(offset_typ, 1, "C")),
            ("null_bitmap", types.Array(types.uint8, 1, "C")),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(ListItemArrayType)
class ListItemArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        payload_type = ListItemArrayPayloadType(fe_type)
        members = [
            ("meminfo", types.MemInfoPointer(payload_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


def define_list_item_dtor(context, builder, list_item_type, payload_type):
    """
    Define destructor for list(item) array type if not already defined
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    fn = mod.get_or_insert_function(
        fnty, name=".dtor.list_item.{}".format(list_item_type.elem_type)
    )

    # End early if the dtor is already defined
    if not fn.is_declaration:
        return fn

    fn.linkage = "linkonce_odr"
    # Populate the dtor
    builder = lir.IRBuilder(fn.append_basic_block())
    base_ptr = fn.args[0]  # void*

    # get payload struct
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_data_helper(builder, payload_type, ref=payload_ptr)

    context.nrt.decref(
        builder, types.Array(list_item_type.elem_type, 1, "C"), payload.data
    )
    context.nrt.decref(builder, types.Array(offset_typ, 1, "C"), payload.offsets)
    context.nrt.decref(builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap)

    builder.ret_void()
    return fn


def construct_list_item_array(context, builder, list_item_type, n_lists, n_elems):
    """Creates meminfo and sets dtor, and allocates buffers for list(item) array
    """
    # create payload type
    payload_type = ListItemArrayPayloadType(list_item_type)
    alloc_type = context.get_data_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_list_item_dtor(context, builder, list_item_type, payload_type)

    # create meminfo
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_data_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_data_ptr, alloc_type.as_pointer())

    # alloc values in payload
    payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    payload.n_lists = n_lists

    # alloc data
    data = bodo.utils.utils._empty_nd_impl(
        context, builder, types.Array(list_item_type.elem_type, 1, "C"), [n_elems]
    )
    data_ptr = data.data
    payload.data = data._getvalue()

    # alloc offsets
    n_lists_plus_1 = builder.add(n_lists, lir.Constant(lir.IntType(64), 1))
    offsets = bodo.utils.utils._empty_nd_impl(
        context, builder, types.Array(offset_typ, 1, "C"), [n_lists_plus_1]
    )
    offsets_ptr = offsets.data
    payload.offsets = offsets._getvalue()

    # alloc null bitmap
    n_bitmask_bytes = builder.udiv(
        builder.add(n_lists, lir.Constant(lir.IntType(64), 7)),
        lir.Constant(lir.IntType(64), 8),
    )
    null_bitmap = bodo.utils.utils._empty_nd_impl(
        context, builder, types.Array(types.uint8, 1, "C"), [n_bitmask_bytes]
    )
    null_bitmap_ptr = null_bitmap.data
    payload.null_bitmap = null_bitmap._getvalue()

    builder.store(payload._getvalue(), meminfo_data_ptr)

    return meminfo, data_ptr, offsets_ptr, null_bitmap_ptr


@unbox(ListItemArrayType)
def unbox_list_item_array(typ, val, c):
    """
    Unbox a numpy array with list of data values.
    """
    from bodo.libs import array_ext

    ll.add_symbol(
        "count_total_elems_list_array", array_ext.count_total_elems_list_array
    )
    ll.add_symbol(
        "list_item_array_from_sequence", array_ext.list_item_array_from_sequence
    )
    n_lists = bodo.utils.utils.object_length(c, val)
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()],)
    fn_tp = c.builder.module.get_or_insert_function(
        fnty, name="count_total_elems_list_array"
    )
    n_elems = c.builder.call(fn_tp, [val])

    meminfo, data_ptr, offsets_ptr, null_bitmap_ptr = construct_list_item_array(
        c.context, c.builder, typ, n_lists, n_elems
    )
    ctype = bodo.utils.utils.numba_to_c_type(typ.elem_type)

    # function signature of list_item_array_from_sequence
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),  # obj
            lir.IntType(8).as_pointer(),  # data
            lir.IntType(32).as_pointer(),  # offsets
            lir.IntType(8).as_pointer(),  # null_bitmap
            lir.IntType(32),  # ctype
        ],
    )
    fn = c.builder.module.get_or_insert_function(
        fnty, name="list_item_array_from_sequence"
    )
    c.builder.call(
        fn,
        [
            val,
            c.builder.bitcast(data_ptr, lir.IntType(8).as_pointer()),
            offsets_ptr,
            null_bitmap_ptr,
            lir.Constant(lir.IntType(32), ctype),
        ],
    )

    list_item_array = c.context.make_helper(c.builder, typ)
    list_item_array.meminfo = meminfo

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(list_item_array._getvalue(), is_error=is_error)
