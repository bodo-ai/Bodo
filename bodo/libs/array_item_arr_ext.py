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
    count_array_inner_elems,
    seq_getitem,
    gen_allocate_array,
)
from bodo.utils.indexing import init_nested_counts, add_nested_counts
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
        assert bodo.utils.utils.is_array_typ(dtype, False)
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
    def __init__(self, array_type):
        self.array_type = array_type
        super(ArrayItemArrayPayloadType, self).__init__(
            name="ArrayItemArrayPayloadType({})".format(array_type)
        )


@register_model(ArrayItemArrayPayloadType)
class ArrayItemArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("n_arrays", types.int64),
            ("data", fe_type.array_type.dtype),
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
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_helper(builder, payload_type, ref=payload_ptr)

    context.nrt.decref(builder, array_item_type.dtype, payload.data)
    context.nrt.decref(builder, types.Array(offset_typ, 1, "C"), payload.offsets)
    context.nrt.decref(builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap)

    builder.ret_void()
    return fn


def construct_array_item_array(
    context, builder, array_item_type, n_arrays, n_elems, c=None
):
    """Creates meminfo and sets dtor, and allocates buffers for array(item) array
    """
    # create payload type
    payload_type = ArrayItemArrayPayloadType(array_item_type)
    alloc_type = context.get_value_type(payload_type)
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
    payload.data = gen_allocate_array(
        context, builder, array_item_type.dtype, n_elems, c
    )

    # alloc offsets
    n_arrays_plus_1 = builder.add(n_arrays, lir.Constant(lir.IntType(64), 1))
    offsets = bodo.utils.utils._empty_nd_impl(
        context, builder, types.Array(offset_typ, 1, "C"), [n_arrays_plus_1]
    )
    offsets_ptr = offsets.data
    # offsets[0] = 0
    builder.store(context.get_constant(types.int32, 0), offsets_ptr)
    # offsets[n] = n_elems
    builder.store(
        builder.trunc(builder.extract_value(n_elems, 0), lir.IntType(32)),
        builder.gep(offsets_ptr, [n_arrays]),
    )
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

    return meminfo, payload.data, offsets_ptr, null_bitmap_ptr


def _unbox_array_item_array_copy_data(
    typ, arr_obj, np_array_method, c, data_arr, item_ind, n_items
):
    """unbox 'arr_obj' and copy data to 'data_arr' at index 'item_ind'
    """
    context = c.context
    builder = c.builder

    # convert arr_obj to array if it is a list (list_check is needed since converting
    # pd arrays like IntergerArray can cause errors)
    # if isinstance(arr_obj, list):
    #   arr_obj = np.array(arr_obj)
    arr_obj_ptr = cgutils.alloca_once_value(builder, arr_obj)
    is_list = list_check(builder, context, arr_obj)
    is_list_cond = builder.icmp_unsigned("!=", is_list, lir.Constant(is_list.type, 0))
    with builder.if_then(is_list_cond):
        old_obj = builder.load(arr_obj_ptr)
        new_obj = c.pyapi.call(np_array_method, c.pyapi.tuple_pack([old_obj]))
        # TODO: this decref causes crashes for some reason in test_array_item_array.py
        # needs to be investigated to avoid object leaks
        # c.pyapi.decref(old_obj)
        builder.store(new_obj, arr_obj_ptr)
    arr_obj = builder.load(arr_obj_ptr)

    # unbox array
    arr_typ = typ.dtype
    arr_val = c.pyapi.to_native_value(arr_typ, arr_obj).value
    # copy array data
    sig = types.none(arr_typ, types.int64, types.int64, arr_typ)

    def copy_data(data_arr, item_ind, n_items, arr_val):
        data_arr[item_ind : item_ind + n_items] = arr_val

    _is_error, _res = c.pyapi.call_jit_code(
        copy_data, sig, [data_arr, item_ind, n_items, arr_val]
    )
    c.context.nrt.decref(builder, arr_typ, arr_val)


def _unbox_array_item_array_generic(
    typ, val, c, n_arrays, data_arr, offsets_ptr, null_bitmap_ptr
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

    # get np.array to convert list items to array
    mod_name = context.insert_const_string(builder.module, "numpy")
    np_mod_obj = c.pyapi.import_module_noblock(mod_name)
    np_array_method = c.pyapi.object_getattr_string(np_mod_obj, "asarray")

    zero32 = c.context.get_constant(types.int32, 0)
    builder.store(zero32, offsets_ptr)

    # pseudocode for code generation:
    # curr_item_ind = 0
    # for i in range(len(A)):
    #   offsets[i] = curr_item_ind
    #   arr_obj = A[i]
    #   if isna(arr_obj):
    #     set null_bitmap i'th bit to 0
    #   else:
    #     set null_bitmap i'th bit to 1
    #     n_items = len(arr_obj)
    #     if isinstance(arr_obj, list):
    #        unbox(arr_obj) and copy data to data_arr
    #        curr_item_ind += n_items
    #     else:  # arr_obj is ndarray
    #        unbox(arr_obj) and copy data to data_arr
    #        curr_item_ind += n_items
    # offsets[n] = curr_item_ind;

    # curr_item_ind = 0
    curr_item_ind = cgutils.alloca_once_value(
        builder, context.get_constant(types.int64, 0)
    )
    # for each array
    with cgutils.for_range(builder, n_arrays) as loop:
        array_ind = loop.index
        item_ind = builder.load(curr_item_ind)

        # offsets[i] = curr_item_ind
        builder.store(
            builder.trunc(item_ind, lir.IntType(32)),
            builder.gep(offsets_ptr, [array_ind]),
        )
        # arr_obj = A[i]
        arr_obj = seq_getitem(builder, context, val, array_ind)
        # set NA bit to 0
        set_bitmap_bit(builder, null_bitmap_ptr, array_ind, 0)

        # check for NA
        is_na = is_na_value(builder, context, arr_obj, C_NA)
        is_na_cond = builder.icmp_unsigned("!=", is_na, lir.Constant(is_na.type, 1))
        with builder.if_then(is_na_cond):
            # set NA bit to 1
            set_bitmap_bit(builder, null_bitmap_ptr, array_ind, 1)
            n_items = bodo.utils.utils.object_length(c, arr_obj)
            _unbox_array_item_array_copy_data(
                typ, arr_obj, np_array_method, c, data_arr, item_ind, n_items
            )
            # curr_item_ind += n_items
            builder.store(builder.add(item_ind, n_items), curr_item_ind)
        c.pyapi.decref(arr_obj)

    # offsets[n] = curr_item_ind;
    builder.store(
        builder.trunc(builder.load(curr_item_ind), lir.IntType(32)),
        builder.gep(offsets_ptr, [n_arrays]),
    )

    c.pyapi.decref(pd_mod_obj)
    c.pyapi.decref(C_NA)
    c.pyapi.decref(np_mod_obj)
    c.pyapi.decref(np_array_method)


@unbox(ArrayItemArrayType)
def unbox_array_item_array(typ, val, c):
    """
    Unbox a numpy array with array of data values.
    """
    handle_in_c = isinstance(typ.dtype, types.Array) and typ.dtype.dtype in (
        types.int64,
        types.float64,
        types.bool_,
        datetime_date_type,
    )

    n_arrays = bodo.utils.utils.object_length(c, val)

    if handle_in_c:
        fnty = lir.FunctionType(lir.IntType(64), [lir.IntType(8).as_pointer()],)
        fn_tp = c.builder.module.get_or_insert_function(
            fnty, name="count_total_elems_list_array"
        )
        n_elems = cgutils.pack_array(c.builder, [c.builder.call(fn_tp, [val])])
    else:
        n_elems = count_array_inner_elems(c, c.builder, c.context, val, typ)

    meminfo, data_arr, offsets_ptr, null_bitmap_ptr = construct_array_item_array(
        c.context, c.builder, typ, n_arrays, n_elems, c
    )

    # use C unboxing when possible to avoid compilation and runtime overheads
    # otherwise, use generic llvm/Numba unboxing
    if handle_in_c:
        ctype = bodo.utils.utils.numba_to_c_type(typ.dtype.dtype)
        data_ptr = c.context.make_array(typ.dtype)(c.context, c.builder, data_arr).data
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
            typ, val, c, n_arrays, data_arr, offsets_ptr, null_bitmap_ptr
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
        meminfo_void_ptr, context.get_value_type(payload_type).as_pointer()
    )
    payload = cgutils.create_struct_proxy(payload_type)(
        context, builder, builder.load(meminfo_data_ptr)
    )
    return payload


def _box_array_item_array_generic(
    typ, c, n_arrays, data_arr, offsets_ptr, null_bitmap_ptr
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
    #     arr_obj = list_new(n_items)
    #     for j in range(n_items):
    #        arr_obj[j] = A.data[curr_item_ind]
    #        # curr_item_ind += 1
    #     A[i] = arr_obj

    # create array of objects with num_items shape
    mod_name = context.insert_const_string(builder.module, "numpy")
    np_class_obj = c.pyapi.import_module_noblock(mod_name)
    dtype_obj = c.pyapi.object_getattr_string(np_class_obj, "object_")
    num_items_obj = c.pyapi.long_from_longlong(n_arrays)
    out_arr = c.pyapi.call_method(np_class_obj, "ndarray", (num_items_obj, dtype_obj))
    # get np.nan to set NA
    nan_obj = c.pyapi.object_getattr_string(np_class_obj, "nan")

    # curr_item_ind = 0
    curr_item_ind = cgutils.alloca_once_value(builder, lir.Constant(lir.IntType(64), 0))
    # for each array
    with cgutils.for_range(builder, n_arrays) as loop:
        array_ind = loop.index
        # A[i] = np.nan
        pyarray_setitem(builder, context, out_arr, array_ind, nan_obj)
        # check for NA
        na_bit = get_bitmap_bit(builder, null_bitmap_ptr, array_ind)
        not_na_cond = builder.icmp_unsigned(
            "!=", na_bit, lir.Constant(lir.IntType(8), 0)
        )
        with builder.if_then(not_na_cond):
            # n_items = offsets[i + 1] - offsets[i]
            n_items = builder.sext(
                builder.sub(
                    builder.load(
                        builder.gep(
                            offsets_ptr,
                            [builder.add(array_ind, lir.Constant(array_ind.type, 1))],
                        )
                    ),
                    builder.load(builder.gep(offsets_ptr, [array_ind])),
                ),
                lir.IntType(64),
            )
            # create array obj
            n_items_obj = c.pyapi.long_from_longlong(n_items)
            item_ind = builder.load(curr_item_ind)
            _is_error, arr_slice = c.pyapi.call_jit_code(
                lambda data_arr, item_ind, n_items: data_arr[
                    item_ind : item_ind + n_items
                ],
                (typ.dtype)(typ.dtype, types.int64, types.int64),
                [data_arr, item_ind, n_items],
            )
            builder.store(
                builder.add(item_ind, n_items), curr_item_ind,
            )
            arr_obj = c.pyapi.from_native_value(typ.dtype, arr_slice, c.env_manager)
            pyarray_setitem(builder, context, out_arr, array_ind, arr_obj)
            c.pyapi.decref(n_items_obj)
            c.pyapi.decref(arr_obj)

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
    data_arr = payload.data
    offsets_ptr = c.context.make_helper(
        c.builder, types.Array(offset_typ, 1, "C"), payload.offsets
    ).data
    null_bitmap_ptr = c.context.make_helper(
        c.builder, types.Array(types.uint8, 1, "C"), payload.null_bitmap
    ).data

    # use C boxing when possible to avoid compilation and runtime overheads
    # otherwise, use generic llvm/Numba unboxing
    if isinstance(typ.dtype, types.Array) and typ.dtype.dtype in (
        types.int64,
        types.float64,
        types.bool_,
        datetime_date_type,
    ):
        ctype = bodo.utils.utils.numba_to_c_type(typ.dtype.dtype)
        data_ptr = c.context.make_helper(c.builder, typ.dtype, data_arr).data

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
            typ, c, payload.n_arrays, data_arr, offsets_ptr, null_bitmap_ptr
        )

    c.context.nrt.decref(c.builder, typ, val)
    return arr


def lower_pre_alloc_array_item_array(context, builder, sig, args):
    array_item_type = sig.return_type
    num_arrs, num_values, _ = args
    meminfo, _, _, _ = construct_array_item_array(
        context, builder, array_item_type, num_arrs, num_values
    )
    array_item_array = context.make_helper(builder, array_item_type)
    array_item_array.meminfo = meminfo
    return array_item_array._getvalue()


@intrinsic
def pre_alloc_array_item_array(typingctx, num_arrs_typ, num_values_typ, dtype_typ=None):
    assert isinstance(num_arrs_typ, types.Integer)
    array_item_type = ArrayItemArrayType(dtype_typ.instance_type)
    return (
        array_item_type(types.int64, num_values_typ, dtype_typ),
        lower_pre_alloc_array_item_array,
    )


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
def init_array_item_array(
    typingctx, n_arrays_typ, data_type, offsets_typ, null_bitmap_typ=None
):
    """Create a ArrayItemArray with provided offsets, data and null bitmap values.
    """
    assert null_bitmap_typ == types.Array(types.uint8, 1, "C")

    def codegen(context, builder, signature, args):
        n_arrays, data, offsets, null_bitmap = args
        array_item_type = signature.return_type

        # TODO: refactor with construct_array_item_array
        # create payload type
        payload_type = ArrayItemArrayPayloadType(array_item_type)
        alloc_type = context.get_value_type(payload_type)
        alloc_size = context.get_abi_sizeof(alloc_type)

        # define dtor
        dtor_fn = define_array_item_dtor(
            context, builder, array_item_type, payload_type
        )

        # create meminfo
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder, context.get_constant(types.uintp, alloc_size), dtor_fn
        )
        meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
        meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

        # alloc values in payload
        payload = cgutils.create_struct_proxy(payload_type)(context, builder)
        payload.n_arrays = n_arrays
        payload.data = data
        payload.offsets = offsets
        payload.null_bitmap = null_bitmap
        builder.store(payload._getvalue(), meminfo_data_ptr)

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[1], data)
        context.nrt.incref(builder, signature.args[2], offsets)
        context.nrt.incref(builder, signature.args[3], null_bitmap)

        array_item_array = context.make_helper(builder, array_item_type)
        array_item_array.meminfo = meminfo
        return array_item_array._getvalue()

    ret_typ = ArrayItemArrayType(data_type)
    sig = ret_typ(types.int64, data_type, offsets_typ, null_bitmap_typ)
    return sig, codegen


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
        return payload.n_arrays

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
        # returning empty array if NA due to type stability issues
        # TODO: warning if value is NA?
        def array_item_arr_getitem_impl(arr, ind):  # pragma: no cover
            offsets = get_offsets(arr)
            data = get_data(arr)
            l_start_offset = offsets[ind]
            l_end_offset = offsets[ind + 1]
            return data[l_start_offset:l_end_offset]

        return array_item_arr_getitem_impl

    # bool arr indexing
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        data_arr_type = arr.dtype

        def impl_bool(arr, ind):  # pragma: no cover
            n = len(arr)
            if n != len(ind):
                raise IndexError(
                    "boolean index did not match indexed array along dimension 0"
                )
            null_bitmap = get_null_bitmap(arr)

            # count the number of lists and value in output and allocate
            n_arrays = 0
            nested_counts = init_nested_counts(data_arr_type)
            for i in range(n):
                if ind[i]:
                    n_arrays += 1
                    arr_item = arr[i]
                    nested_counts = add_nested_counts(nested_counts, arr_item)

            out_arr = pre_alloc_array_item_array(n_arrays, nested_counts, data_arr_type)
            out_null_bitmap = get_null_bitmap(out_arr)

            # write output
            out_ind = 0
            for ii in range(n):
                if ind[ii]:
                    out_arr[out_ind] = arr[ii]
                    # set NA
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(null_bitmap, ii)
                    bodo.libs.int_arr_ext.set_bit_to_arr(out_null_bitmap, out_ind, bit)
                    out_ind += 1

            return out_arr

        return impl_bool

    # ind arr indexing
    # TODO: avoid code duplication
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        data_arr_type = arr.dtype

        def impl_int(arr, ind):  # pragma: no cover
            null_bitmap = get_null_bitmap(arr)

            n = len(ind)
            n_arrays = n
            nested_counts = init_nested_counts(data_arr_type)
            for k in range(n):
                i = ind[k]
                arr_item = arr[i]
                nested_counts = add_nested_counts(nested_counts, arr_item)

            out_arr = pre_alloc_array_item_array(n_arrays, nested_counts, data_arr_type)
            out_null_bitmap = get_null_bitmap(out_arr)

            for kk in range(n):
                ii = ind[kk]
                out_arr[kk] = arr[ii]
                # set NA
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(null_bitmap, ii)
                bodo.libs.int_arr_ext.set_bit_to_arr(out_null_bitmap, kk, bit)

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

    # slice case (used in unboxing)
    if isinstance(idx, types.SliceType):

        def impl_slice(A, idx, val):  # pragma: no cover
            # get output arrays
            offsets = get_offsets(A)
            data = get_data(A)
            null_bitmap = get_null_bitmap(A)

            # get input arrays
            val_offsets = get_offsets(val)
            val_data = get_data(val)
            val_null_bitmap = get_null_bitmap(val)

            n = len(A)
            slice_idx = numba.cpython.unicode._normalize_slice(idx, n)
            start, stop = slice_idx.start, slice_idx.stop
            assert slice_idx.step == 1

            # copy data from first offset
            if start == 0:
                offsets[start] = 0
            start_offset = offsets[start]
            data[start_offset : start_offset + len(val_data)] = val_data

            # copy offsets (n+1 elements)
            offsets[start : stop + 1] = val_offsets + start_offset

            # copy null bits
            val_ind = 0
            for i in range(start, stop):
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(val_null_bitmap, val_ind)
                bodo.libs.int_arr_ext.set_bit_to_arr(null_bitmap, i, bit)
                val_ind += 1

        return impl_slice

    raise BodoError(
        "only setitem with scalar index is currently supported for list arrays"
    )  # pragma: no cover


@overload_method(ArrayItemArrayType, "copy", no_unliteral=True)
def overload_array_item_arr_copy(A):
    def copy_impl(A):  # pragma: no cover
        return init_array_item_array(
            len(A), get_data(A).copy(), get_offsets(A).copy(), get_null_bitmap(A).copy()
        )

    return copy_impl
