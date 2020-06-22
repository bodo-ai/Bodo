# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Array implementation for variable-size array items.
Corresponds to Spark's ArrayType: https://spark.apache.org/docs/latest/sql-reference.html
Corresponds to Arrow's Variable-size List: https://arrow.apache.org/docs/format/Columnar.html

The values are stored in a contingous data array, while an offsets array marks the
individual arrays. For example:
value:             [[1, 2], [3], None, [5, 4, 6], []]
data:              [1, 2, 3, 5, 4, 6]
offsets:           [0, 2, 3, 3, 6, 6]
"""
import operator
import numpy as np
import numba
import bodo

from numba.core import types, cgutils
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
from numba.parfors.array_analysis import ArrayAnalysis
from numba.core.imputils import impl_ret_borrowed

from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.utils.typing import is_list_like_index_type, BodoError
from bodo.utils.cg_helpers import (
    set_bitmap_bit,
    pyarray_getitem,
    pyarray_setitem,
    list_check,
    is_na_value,
    get_bitmap_bit,
)
from llvmlite import ir as lir
import llvmlite.binding as ll

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import hdist
from bodo.libs import array_ext


ll.add_symbol("count_total_elems_list_array", array_ext.count_total_elems_list_array)
ll.add_symbol(
    "array_item_array_from_sequence", array_ext.array_item_array_from_sequence
)
ll.add_symbol(
    "np_array_from_array_item_array", array_ext.np_array_from_array_item_array
)


# offset index types
offset_typ = types.uint32


class ArrayItemArrayType(types.ArrayCompatible):
    def __init__(self, dtype):
        self.dtype = dtype
        super(ArrayItemArrayType, self).__init__(
            name="ArrayItemArrayType({})".format(dtype)
        )

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return ArrayItemArrayType(self.dtype)


class ArrayItemArrayPayloadType(types.Type):
    def __init__(self, list_type):
        self.list_type = list_type
        super(ArrayItemArrayPayloadType, self).__init__(
            name="ArrayItemArrayPayloadType({})".format(list_type)
        )


@register_model(ArrayItemArrayPayloadType)
class ArrayItemArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("n_arrays", types.int64),
            ("data", fe_type.list_type.dtype),
            ("offsets", types.Array(offset_typ, 1, "C")),
            ("null_bitmap", types.Array(types.uint8, 1, "C")),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(ArrayItemArrayType)
class ArrayItemArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        payload_type = ArrayItemArrayPayloadType(fe_type)
        members = [
            ("meminfo", types.MemInfoPointer(payload_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


def define_array_item_dtor(context, builder, array_item_type, payload_type):
    """
    Define destructor for array(item) array type if not already defined
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    fn = mod.get_or_insert_function(
        fnty, name=".dtor.array_item.{}".format(array_item_type.dtype)
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

    context.nrt.decref(builder, array_item_type.dtype, payload.data)
    context.nrt.decref(builder, types.Array(offset_typ, 1, "C"), payload.offsets)
    context.nrt.decref(builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap)

    builder.ret_void()
    return fn


def construct_array_item_array(context, builder, array_item_type, n_arrays, n_elems):
    """Creates meminfo and sets dtor, and allocates buffers for array(item) array
    """
    # create payload type
    payload_type = ArrayItemArrayPayloadType(array_item_type)
    alloc_type = context.get_data_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_array_item_dtor(context, builder, array_item_type, payload_type)

    # create meminfo
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # alloc values in payload
    payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    payload.n_arrays = n_arrays

    # alloc data
    data = bodo.utils.utils._empty_nd_impl(
        context, builder, types.Array(array_item_type.dtype.dtype, 1, "C"), [n_elems]
    )
    data_ptr = data.data
    payload.data = data._getvalue()

    # alloc offsets
    n_arrays_plus_1 = builder.add(n_arrays, lir.Constant(lir.IntType(64), 1))
    offsets = bodo.utils.utils._empty_nd_impl(
        context, builder, types.Array(offset_typ, 1, "C"), [n_arrays_plus_1]
    )
    offsets_ptr = offsets.data
    payload.offsets = offsets._getvalue()

    # alloc null bitmap
    n_bitmask_bytes = builder.udiv(
        builder.add(n_arrays, lir.Constant(lir.IntType(64), 7)),
        lir.Constant(lir.IntType(64), 8),
    )
    null_bitmap = bodo.utils.utils._empty_nd_impl(
        context, builder, types.Array(types.uint8, 1, "C"), [n_bitmask_bytes]
    )
    null_bitmap_ptr = null_bitmap.data
    payload.null_bitmap = null_bitmap._getvalue()

    builder.store(payload._getvalue(), meminfo_data_ptr)

    return meminfo, data_ptr, offsets_ptr, null_bitmap_ptr


def _unbox_array_item_array_generic(
    typ, val, c, n_arrays, data_ptr, offsets_ptr, null_bitmap_ptr
):
    """unbox array(item) array using generic Numba list unboxing to handle all item types
    that can be unboxed.
    """
    context = c.context
    builder = c.builder
    # TODO: error checking for pyapi calls
    # get pd.NA object to check for new NA kind
    mod_name = context.insert_const_string(builder.module, "pandas")
    pd_mod_obj = c.pyapi.import_module_noblock(mod_name)
    C_NA = c.pyapi.object_getattr_string(pd_mod_obj, "NA")

    # pseudocode for code generation:
    # curr_item_ind = 0
    # for i in range(len(A)):
    #   offsets[i] = curr_item_ind
    #   list_obj = A[i]
    #   if isna(list_obj):
    #     set null_bitmap i'th bit to 0
    #   else:
    #     set null_bitmap i'th bit to 1
    #     n_items = len(list_obj)
    #     if isinstance(list_obj, list):
    #        unbox(list_obj) and copy data to data_ptr
    #        curr_item_ind += n_items
    #     else:  # list_obj is ndarray
    #        unbox(list_obj) and copy data to data_ptr
    #        curr_item_ind += n_items
    # offsets[n] = curr_item_ind;

    list_type = typ.dtype
    # curr_item_ind = 0
    curr_item_ind = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(64), 0))
    item_size = context.get_abi_sizeof(context.get_data_type(list_type.dtype))
    # for each array
    with cgutils.for_range(builder, n_arrays) as loop:
        array_ind = loop.index
        item_ind = builder.load(curr_item_ind)
        # offsets[i] = curr_item_ind
        builder.store(
            builder.trunc(item_ind, lir.IntType(32)),
            builder.gep(offsets_ptr, [array_ind]),
        )
        # list_obj = A[i]
        list_obj = pyarray_getitem(builder, context, val, array_ind)
        # check for NA
        is_na = is_na_value(builder, context, list_obj, C_NA)
        is_na_cond = builder.icmp_unsigned("==", is_na, lir.Constant(is_na.type, 1))
        with builder.if_else(is_na_cond) as (then, orelse):
            # NA case
            with then:
                # set NA bit to 0
                set_bitmap_bit(builder, null_bitmap_ptr, array_ind, 0)
            # non-NA case
            with orelse:
                # set NA bit to 1
                set_bitmap_bit(builder, null_bitmap_ptr, array_ind, 1)
                # each item can be either a list or an array
                # check for list
                l_check = list_check(builder, context, list_obj)
                is_list = builder.icmp_unsigned(
                    "==", l_check, lir.Constant(l_check.type, 1)
                )
                with builder.if_else(is_list) as (list_case, array_case):
                    with list_case:
                        # unbox list
                        list_val = c.pyapi.to_native_value(list_type, list_obj).value
                        list_payload = numba.cpython.listobj.ListInstance(
                            context, builder, list_type, list_val
                        )
                        # copy list data
                        n_items = list_payload.size
                        dst = builder.gep(data_ptr, [item_ind])
                        cgutils.raw_memcpy(
                            builder, dst, list_payload.data, n_items, item_size
                        )
                        # NOTE: numba stores list meminfo inside the Python list
                        # objects, which we need to clean up
                        c.pyapi.object_set_private_data(
                            list_obj, context.get_constant_null(types.voidptr)
                        )
                        c.context.nrt.decref(builder, list_type, list_val)
                        c.pyapi.decref(list_obj)
                        # curr_item_ind += n_items
                        builder.store(builder.add(item_ind, n_items), curr_item_ind)
                    with array_case:
                        # unbox array
                        arr_typ = types.Array(list_type.dtype, 1, "C")
                        arr_val = c.pyapi.to_native_value(arr_typ, list_obj).value
                        arr = context.make_array(arr_typ)(context, builder, arr_val)
                        # copy array data
                        (n_items,) = cgutils.unpack_tuple(builder, arr.shape, count=1)
                        dst = builder.gep(data_ptr, [item_ind])
                        cgutils.raw_memcpy(builder, dst, arr.data, n_items, item_size)
                        c.context.nrt.decref(builder, arr_typ, arr_val)
                        c.pyapi.decref(list_obj)
                        # curr_item_ind += n_items
                        builder.store(builder.add(item_ind, n_items), curr_item_ind)

    # offsets[n] = curr_item_ind;
    builder.store(
        builder.trunc(builder.load(curr_item_ind), lir.IntType(32)),
        builder.gep(offsets_ptr, [n_arrays]),
    )

    c.pyapi.decref(pd_mod_obj)
    c.pyapi.decref(C_NA)


@unbox(ArrayItemArrayType)
def unbox_array_item_array(typ, val, c):
    """
    Unbox a numpy array with array of data values.
    """

    n_arrays = bodo.utils.utils.object_length(c, val)
    fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()],)
    fn_tp = c.builder.module.get_or_insert_function(
        fnty, name="count_total_elems_list_array"
    )
    n_elems = c.builder.call(fn_tp, [val])

    meminfo, data_ptr, offsets_ptr, null_bitmap_ptr = construct_array_item_array(
        c.context, c.builder, typ, n_arrays, n_elems
    )
    ctype = bodo.utils.utils.numba_to_c_type(typ.dtype.dtype)

    # use C unboxing when possible to avoid compilation and runtime overheads
    # otherwise, use generic llvm/Numba unboxing
    if typ.dtype.dtype in (
        types.int64,
        types.float64,
        types.bool_,
        datetime_date_type,
    ):

        # function signature of array_item_array_from_sequence
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
            fnty, name="array_item_array_from_sequence"
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

    else:
        _unbox_array_item_array_generic(
            typ, val, c, n_arrays, data_ptr, offsets_ptr, null_bitmap_ptr
        )

    array_item_array = c.context.make_helper(c.builder, typ)
    array_item_array.meminfo = meminfo

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(array_item_array._getvalue(), is_error=is_error)


def _get_array_item_arr_payload(context, builder, arr_typ, arr):
    """get payload struct proxy for a array(item) value
    """
    array_item_array = context.make_helper(builder, arr_typ, arr)
    payload_type = ArrayItemArrayPayloadType(arr_typ)
    meminfo_void_ptr = context.nrt.meminfo_data(builder, array_item_array.meminfo)
    meminfo_data_ptr = builder.bitcast(
        meminfo_void_ptr, context.get_data_type(payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload


def _box_array_item_array_generic(
    typ, c, n_arrays, data_ptr, offsets_ptr, null_bitmap_ptr
):
    """box array(item) array using generic Numba list boxing to handle all item types
    that can be boxed.
    """
    context = c.context
    builder = c.builder
    # TODO: error checking for pyapi calls

    # pseudocode for code generation:
    # out_arr = np.ndarray(n, np.object_)
    # curr_item_ind = 0
    # for i in range(n):
    #   if isna(A[i]):
    #     out_arr[i] = np.nan
    #   else:
    #     n_items = offsets[i + 1] - offsets[i]
    #     list_obj = list_new(n_items)
    #     for j in range(n_items):
    #        list_obj[j] = A.data[curr_item_ind]
    #        # curr_item_ind += 1
    #     A[i] = list_obj

    # create array of objects with num_items shape
    mod_name = context.insert_const_string(builder.module, "numpy")
    np_class_obj = c.pyapi.import_module_noblock(mod_name)
    dtype_obj = c.pyapi.object_getattr_string(np_class_obj, "object_")
    num_items_obj = c.pyapi.long_from_longlong(n_arrays)
    out_arr = c.pyapi.call_method(np_class_obj, "ndarray", (num_items_obj, dtype_obj))
    # get np.nan to set NA
    nan_obj = c.pyapi.object_getattr_string(np_class_obj, "nan")

    list_type = typ.dtype
    # curr_item_ind = 0
    curr_item_ind = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(64), 0))
    # for each array
    with cgutils.for_range(builder, n_arrays) as loop:
        array_ind = loop.index
        # check for NA
        na_bit = get_bitmap_bit(builder, null_bitmap_ptr, array_ind)
        is_na_cond = builder.icmp_unsigned(
            "==", na_bit, lir.Constant(lir.IntType(8), 0)
        )
        with builder.if_else(is_na_cond) as (then, orelse):
            # NA case
            with then:
                # A[i] = np.nan
                pyarray_setitem(builder, context, out_arr, array_ind, nan_obj)
            # non-NA case
            with orelse:
                # n_items = offsets[i + 1] - offsets[i]
                n_items = builder.sext(
                    builder.sub(
                        builder.load(
                            builder.gep(
                                offsets_ptr,
                                [
                                    builder.add(
                                        array_ind, lir.Constant(array_ind.type, 1)
                                    )
                                ],
                            )
                        ),
                        builder.load(builder.gep(offsets_ptr, [array_ind])),
                    ),
                    lir.IntType(64),
                )
                # create list obj
                list_obj = c.pyapi.list_new(n_items)
                # store items in list
                with cgutils.for_range(builder, n_items) as item_loop:
                    j = item_loop.index
                    item_ind = builder.load(curr_item_ind)
                    buff_ptr = builder.gep(data_ptr, [item_ind])
                    item_obj = c.pyapi.from_native_value(
                        list_type.dtype, builder.load(buff_ptr), c.env_manager
                    )
                    # steals reference to item_obj
                    c.pyapi.list_setitem(list_obj, j, item_obj)
                    # curr_item_ind += 1
                    builder.store(
                        builder.add(item_ind, lir.Constant(item_ind.type, 1)),
                        curr_item_ind,
                    )

                pyarray_setitem(builder, context, out_arr, array_ind, list_obj)
                c.pyapi.decref(list_obj)

    c.pyapi.decref(np_class_obj)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(num_items_obj)
    c.pyapi.decref(nan_obj)
    return out_arr


@box(ArrayItemArrayType)
def box_array_item_arr(typ, val, c):
    """box packed native representation of array of item array into python objects
    """

    payload = _get_array_item_arr_payload(c.context, c.builder, typ, val)

    data_ptr = c.context.make_helper(c.builder, typ.dtype, payload.data).data
    offsets_ptr = c.context.make_helper(
        c.builder, types.Array(offset_typ, 1, "C"), payload.offsets
    ).data
    null_bitmap_ptr = c.context.make_helper(
        c.builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap
    ).data

    # use C boxing when possible to avoid compilation and runtime overheads
    # otherwise, use generic llvm/Numba unboxing
    if typ.dtype.dtype in (
        types.int64,
        types.float64,
        types.bool_,
        datetime_date_type,
    ):
        ctype = bodo.utils.utils.numba_to_c_type(typ.dtype.dtype)

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
            fnty, name="np_array_from_array_item_array"
        )

        arr = c.builder.call(
            fn_get,
            [
                payload.n_arrays,
                c.builder.bitcast(data_ptr, lir.IntType(8).as_pointer()),
                offsets_ptr,
                null_bitmap_ptr,
                lir.Constant(lir.IntType(32), ctype),
            ],
        )

    else:
        arr = _box_array_item_array_generic(
            typ, c, payload.n_arrays, data_ptr, offsets_ptr, null_bitmap_ptr
        )

    c.context.nrt.decref(c.builder, typ, val)
    return arr


@intrinsic
def pre_alloc_array_item_array(
    typingctx, num_lists_typ, num_values_typ, dtype_typ=None
):
    assert (
        isinstance(num_lists_typ, types.Integer)
        and isinstance(num_values_typ, types.Integer)
        and isinstance(dtype_typ, (types.DType, types.NumberClass))
    )
    array_item_type = ArrayItemArrayType(
        bodo.hiframes.pd_series_ext._get_series_array_type(dtype_typ.dtype)
    )

    def codegen(context, builder, sig, args):
        num_lists, num_values, _ = args
        meminfo, _, _, _ = construct_array_item_array(
            context, builder, array_item_type, num_lists, num_values
        )
        array_item_array = context.make_helper(builder, array_item_type)
        array_item_array.meminfo = meminfo
        return array_item_array._getvalue()

    return array_item_type(types.int64, types.int64, dtype_typ), codegen


def pre_alloc_array_item_array_equiv(
    self, scope, equiv_set, loc, args, kws
):  # pragma: no cover
    """Array analysis function for pre_alloc_array_item_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 3 and not kws
    return args[0], []


ArrayAnalysis._analyze_op_call_bodo_libs_array_item_arr_ext_pre_alloc_array_item_array = (
    pre_alloc_array_item_array_equiv
)


@intrinsic
def get_offsets(typingctx, arr_typ=None):
    assert isinstance(arr_typ, ArrayItemArrayType)
    # TODO: alias analysis extension functions for get_offsets, etc.?

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_array_item_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.offsets)

    return types.Array(offset_typ, 1, "C")(arr_typ), codegen


@intrinsic
def get_data(typingctx, arr_typ=None):
    assert isinstance(arr_typ, ArrayItemArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_array_item_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.data)

    return (arr_typ.dtype)(arr_typ), codegen


@intrinsic
def get_null_bitmap(typingctx, arr_typ=None):
    assert isinstance(arr_typ, ArrayItemArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_array_item_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.null_bitmap)

    return types.Array(types.uint8, 1, "C")(arr_typ), codegen


@intrinsic
def get_n_arrays(typingctx, arr_typ=None):
    assert isinstance(arr_typ, ArrayItemArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_array_item_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.n_arrays)

    return types.int64(arr_typ), codegen


@overload(len, no_unliteral=True)
def overload_array_item_arr_len(A):
    if isinstance(A, ArrayItemArrayType):
        return lambda A: get_n_arrays(A)


@overload_attribute(ArrayItemArrayType, "shape")
def overload_array_item_arr_shape(A):
    return lambda A: (get_n_arrays(A),)


@overload_attribute(ArrayItemArrayType, "ndim")
def overload_array_item_arr_ndim(A):
    return lambda A: 1


@overload(operator.getitem, no_unliteral=True)
def array_item_arr_getitem_array(arr, ind):
    if not isinstance(arr, ArrayItemArrayType):
        return

    if isinstance(types.unliteral(ind), types.Integer):
        # returning [] if NA due to type stability issues
        # TODO: warning if value is NA?
        def array_item_arr_getitem_impl(arr, ind):  # pragma: no cover
            offsets = get_offsets(arr)
            data = get_data(arr)
            l_start_offset = offsets[ind]
            l_end_offset = offsets[ind + 1]
            out = []
            for i in range(l_start_offset, l_end_offset):
                out.append(data[i])
            return out

        return array_item_arr_getitem_impl

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
            n_arrays = 0
            n_values = 0
            for i in range(n):
                if ind[i]:
                    n_arrays += 1
                    n_values += int(offsets[i + 1] - offsets[i])

            out_arr = pre_alloc_array_item_array(n_arrays, n_values, data.dtype)
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
            n_arrays = n
            n_values = 0
            for k in range(n):
                i = ind[k]
                n_values += int(offsets[i + 1] - offsets[i])

            out_arr = pre_alloc_array_item_array(n_arrays, n_values, data.dtype)
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
            slice_idx = numba.cpython.unicode._normalize_slice(ind, n)
            # reusing integer array slicing above
            arr_ind = np.arange(slice_idx.start, slice_idx.stop, slice_idx.step)
            return arr[arr_ind]

        return impl_slice


@overload(operator.setitem)
def array_item_arr_setitem(A, idx, val):
    if not isinstance(A, ArrayItemArrayType):
        return

    # scalar case
    # NOTE: assuming that the array is being built and all previous elements are set
    # TODO: make sure array is being build
    if isinstance(types.unliteral(idx), types.Integer):
        assert isinstance(val, types.List)  # TODO: raise proper error

        def impl_scalar(A, idx, val):  # pragma: no cover
            offsets = get_offsets(A)
            data = get_data(A)
            null_bitmap = get_null_bitmap(A)
            if idx == 0:
                offsets[0] = 0

            n_items = len(val)
            offsets[idx + 1] = offsets[idx] + n_items
            data[offsets[idx] : offsets[idx + 1]] = val
            bodo.libs.int_arr_ext.set_bit_to_arr(null_bitmap, idx, 1)

        return impl_scalar

    raise BodoError(
        "only setitem with scalar index is currently supported for list arrays"
    )  # pragma: no cover


@overload_method(ArrayItemArrayType, "copy", no_unliteral=True)
def overload_array_item_arr_copy(A):
    def copy_impl(A):  # pragma: no cover
        offsets = get_offsets(A)
        data = get_data(A)
        null_bitmap = get_null_bitmap(A)

        # allocate
        n = len(A)
        n_values = offsets[-1]
        out_arr = pre_alloc_array_item_array(n, n_values, data.dtype)
        out_offsets = get_offsets(out_arr)
        out_data = get_data(out_arr)
        out_null_bitmap = get_null_bitmap(out_arr)

        # copy input values to output values
        out_offsets[:] = offsets
        out_data[:] = data
        out_null_bitmap[:] = null_bitmap

        return out_arr

    return copy_impl
