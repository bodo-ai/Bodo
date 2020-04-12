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
from numba.array_analysis import ArrayAnalysis
from numba.targets.imputils import impl_ret_borrowed

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


def _get_list_item_arr_payload(context, builder, arr_typ, arr):
    """get payload struct proxy for a list(item) value
    """
    list_item_array = context.make_helper(builder, arr_typ, arr)
    payload_type = ListItemArrayPayloadType(arr_typ)
    meminfo_data_ptr = context.nrt.meminfo_data(builder, list_item_array.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_data_ptr, context.get_data_type(payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload


@box(ListItemArrayType)
def box_list_item_arr(typ, val, c):
    """box packed native representation of list of item array into python objects
    """
    from bodo.libs import array_ext

    ll.add_symbol(
        "np_array_from_list_item_array", array_ext.np_array_from_list_item_array
    )

    payload = _get_list_item_arr_payload(c.context, c.builder, typ, val)

    ctype = bodo.utils.utils.numba_to_c_type(typ.elem_type)

    fnty = lir.FunctionType(
        c.context.get_argument_type(types.pyobject),
        [
            lir.IntType(64),  # num_lists
            lir.IntType(8).as_pointer(),  # data
            lir.IntType(32).as_pointer(),  # offsets
            lir.IntType(8).as_pointer(),  # null_bitmap
            lir.IntType(32),  # ctype
        ],
    )
    fn_get = c.builder.module.get_or_insert_function(
        fnty, name="np_array_from_list_item_array"
    )

    data_ptr = c.context.make_helper(
        c.builder, types.Array(typ.elem_type, 1, "C"), payload.data
    ).data
    offsets_ptr = c.context.make_helper(
        c.builder, types.Array(offset_typ, 1, "C"), payload.offsets
    ).data
    null_bitmap_ptr = c.context.make_helper(
        c.builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap
    ).data

    arr = c.builder.call(
        fn_get,
        [
            payload.n_lists,
            c.builder.bitcast(data_ptr, lir.IntType(8).as_pointer()),
            offsets_ptr,
            null_bitmap_ptr,
            lir.Constant(lir.IntType(32), ctype),
        ],
    )

    c.context.nrt.decref(c.builder, typ, val)
    return arr


@intrinsic
def pre_alloc_list_item_array(typingctx, num_lists_typ, num_values_typ, dtype_typ=None):
    assert (
        isinstance(num_lists_typ, types.Integer)
        and isinstance(num_values_typ, types.Integer)
        and isinstance(dtype_typ, types.DType)
    )
    list_item_type = ListItemArrayType(dtype_typ.dtype)

    def codegen(context, builder, sig, args):
        num_lists, num_values, _ = args
        meminfo, _, _, _ = construct_list_item_array(
            context, builder, list_item_type, num_lists, num_values
        )
        list_item_array = context.make_helper(builder, list_item_type)
        list_item_array.meminfo = meminfo
        return list_item_array._getvalue()

    return list_item_type(types.int64, types.int64, dtype_typ), codegen


def pre_alloc_list_item_array_equiv(
    self, scope, equiv_set, args, kws
):  # pragma: no cover
    """Array analysis function for pre_alloc_list_item_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 3 and not kws
    return args[0], []


ArrayAnalysis._analyze_op_call_bodo_libs_list_item_arr_ext_pre_alloc_list_item_array = (
    pre_alloc_list_item_array_equiv
)


@intrinsic
def get_offsets(typingctx, arr_typ=None):
    assert isinstance(arr_typ, ListItemArrayType)
    # TODO: alias analysis extension functions for get_offsets, etc.?

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_list_item_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.offsets)

    return types.Array(offset_typ, 1, "C")(arr_typ), codegen


@intrinsic
def get_data(typingctx, arr_typ=None):
    assert isinstance(arr_typ, ListItemArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_list_item_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.data)

    return types.Array(arr_typ.elem_type, 1, "C")(arr_typ), codegen


@intrinsic
def get_null_bitmap(typingctx, arr_typ=None):
    assert isinstance(arr_typ, ListItemArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_list_item_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.null_bitmap)

    return types.Array(types.uint8, 1, "C")(arr_typ), codegen


@intrinsic
def get_n_lists(typingctx, arr_typ=None):
    assert isinstance(arr_typ, ListItemArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_list_item_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.n_lists)

    return types.int64(arr_typ), codegen


@overload(len)
def overload_list_item_arr_len(A):
    if isinstance(A, ListItemArrayType):
        return lambda A: get_n_lists(A)


@overload_attribute(ListItemArrayType, "shape")
def overload_list_item_arr_shape(A):
    return lambda A: (get_n_lists(A),)


@overload_attribute(ListItemArrayType, "ndim")
def overload_list_item_arr_ndim(A):
    return lambda A: 1


@overload(operator.getitem)
def list_item_arr_getitem_array(arr, ind):
    if not isinstance(arr, ListItemArrayType):
        return

    if isinstance(ind, types.Integer):
        # returning [] if NA due to type stability issues
        # TODO: warning if value is NA?
        def list_item_arr_getitem_impl(arr, ind):  # pragma: no cover
            offsets = get_offsets(arr)
            data = get_data(arr)
            l_start_offset = offsets[ind]
            l_end_offset = offsets[ind + 1]
            out = []
            for i in range(l_start_offset, l_end_offset):
                out.append(data[i])
            return out

        return list_item_arr_getitem_impl

    # bool arr indexing
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(arr, ind):  # pragma: no cover
            n = len(arr)
            if n != len(ind):
                raise IndexError(
                    "boolean index did not match indexed array along dimension 0"
                )
            offsets = get_offsets(arr)
            data = get_data(arr)
            null_bitmap = get_null_bitmap(arr)

            # count the number of lists and value in output and allocate
            n_lists = 0
            n_values = 0
            for i in range(n):
                if ind[i]:
                    n_lists += 1
                    n_values += int(offsets[i + 1] - offsets[i])

            out_arr = pre_alloc_list_item_array(n_lists, n_values, data.dtype)
            out_offsets = get_offsets(out_arr)
            out_data = get_data(out_arr)
            out_null_bitmap = get_null_bitmap(out_arr)

            # write output
            out_ind = 0
            curr_offset = 0
            for ii in range(n):
                if ind[ii]:
                    l_start_offset = offsets[ii]
                    l_end_offset = offsets[ii + 1]
                    n_vals = int(l_end_offset - l_start_offset)
                    val_ind = 0
                    for jj in range(l_start_offset, l_end_offset):
                        out_data[curr_offset + val_ind] = data[jj]
                        val_ind += 1

                    out_offsets[out_ind] = curr_offset
                    curr_offset += n_vals
                    # set NA
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(null_bitmap, ii)
                    bodo.libs.int_arr_ext.set_bit_to_arr(out_null_bitmap, out_ind, bit)
                    out_ind += 1

            out_offsets[out_ind] = curr_offset
            return out_arr

        return impl_bool

    # ind arr indexing
    # TODO: avoid code duplication
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl_int(arr, ind):  # pragma: no cover
            offsets = get_offsets(arr)
            data = get_data(arr)
            null_bitmap = get_null_bitmap(arr)

            n = len(ind)
            n_lists = n
            n_values = 0
            for k in range(n):
                i = ind[k]
                n_values += int(offsets[i + 1] - offsets[i])

            out_arr = pre_alloc_list_item_array(n_lists, n_values, data.dtype)
            out_offsets = get_offsets(out_arr)
            out_data = get_data(out_arr)
            out_null_bitmap = get_null_bitmap(out_arr)

            out_ind = 0
            curr_offset = 0
            for kk in range(n):
                ii = ind[kk]
                l_start_offset = offsets[ii]
                l_end_offset = offsets[ii + 1]
                n_vals = int(l_end_offset - l_start_offset)
                val_ind = 0
                for jj in range(l_start_offset, l_end_offset):
                    out_data[curr_offset + val_ind] = data[jj]
                    val_ind += 1

                out_offsets[out_ind] = curr_offset
                curr_offset += n_vals
                # set NA
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(null_bitmap, ii)
                bodo.libs.int_arr_ext.set_bit_to_arr(out_null_bitmap, out_ind, bit)
                out_ind += 1

            out_offsets[out_ind] = curr_offset
            return out_arr

        return impl_int

    # slice case
    if isinstance(ind, types.SliceType):

        def impl_slice(arr, ind):  # pragma: no cover
            n = len(arr)
            slice_idx = numba.unicode._normalize_slice(ind, n)
            # reusing integer array slicing above
            arr_ind = np.arange(slice_idx.start, slice_idx.stop, slice_idx.step)
            return arr[arr_ind]

        return impl_slice


@overload_method(ListItemArrayType, "copy")
def overload_list_item_arr_copy(A):
    def copy_impl(A):  # pragma: no cover
        offsets = get_offsets(A)
        data = get_data(A)
        null_bitmap = get_null_bitmap(A)

        # allocate
        n = len(A)
        n_values = offsets[-1]
        out_arr = pre_alloc_list_item_array(n, n_values, data.dtype)
        out_offsets = get_offsets(out_arr)
        out_data = get_data(out_arr)
        out_null_bitmap = get_null_bitmap(out_arr)

        # copy input values to output values
        out_offsets[:] = offsets
        out_data[:] = data
        out_null_bitmap[:] = null_bitmap

        return out_arr

    return copy_impl
