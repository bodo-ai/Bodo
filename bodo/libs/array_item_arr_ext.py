# Copyright (C) 2022 Bodo Inc. All rights reserved.
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

import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.extending import (
    box,
    intrinsic,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.libs import array_ext  # noqa: F401
from bodo.utils.cg_helpers import (
    gen_alloc_meminfo,
    gen_allocate_array,
    meminfo_to_np_arr,
)
from bodo.utils.indexing import add_nested_counts, init_nested_counts
from bodo.utils.typing import (
    BodoError,
    is_iterable_type,
    is_list_like_index_type,
    unwrap_typeref,
)

# offset index types
offset_type = types.uint64
np_offset_type = numba.np.numpy_support.as_dtype(offset_type)


class ArrayItemArrayType(types.ArrayCompatible):
    def __init__(self, dtype):
        assert bodo.utils.utils.is_array_typ(
            dtype, False
        ), "ArrayItemArrayType dtype should be an array type"
        self.dtype = dtype
        super(ArrayItemArrayType, self).__init__(name=f"ArrayItemArrayType({dtype})")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return ArrayItemArrayType(self.dtype)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


class ArrayItemArrayPayloadType(types.Type):
    def __init__(self, array_type):
        self.array_type = array_type
        super(ArrayItemArrayPayloadType, self).__init__(
            name=f"ArrayItemArrayPayloadType({array_type})"
        )

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@register_model(ArrayItemArrayPayloadType)
class ArrayItemArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("n_arrays", types.int64),
            # NOTE: data array may have more capacity than necessary
            # (similar to std::vector).
            # Use offsets[n] to get total number of elements instead of len(data_arr).
            ("data", fe_type.array_type.dtype),
            ("offsets", types.MemInfoPointer(offset_type)),
            ("null_bitmap", types.MemInfoPointer(types.uint8)),
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
    fn = cgutils.get_or_insert_function(
        mod, fnty, name=f".dtor.array_item.{array_item_type.dtype}"
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

    context.nrt.decref(builder, payload_type, builder.load(payload_ptr))

    builder.ret_void()
    return fn


def construct_array_item_array(
    context, builder, array_item_type, n_arrays, n_elems, dict_arr_ref, c=None
):
    """Creates meminfo and sets dtor, and allocates buffers for array(item) array"""
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

    # if number of data elements is unknown (-1), use same length as array for now
    # i.e: if n_elems[0] == -1: n_elems[0] = n_arrays
    n_counts = n_elems.type.count
    data_len = builder.extract_value(n_elems, 0)
    len_ptr = cgutils.alloca_once_value(builder, data_len)
    is_unknown_len = builder.icmp_signed(
        "==", data_len, lir.Constant(data_len.type, -1)
    )
    with builder.if_then(is_unknown_len):
        builder.store(n_arrays, len_ptr)
    n_elems = cgutils.pack_array(
        builder,
        [builder.load(len_ptr)]
        + [builder.extract_value(n_elems, i) for i in range(1, n_counts)],
    )

    # alloc data
    payload.data = gen_allocate_array(
        context, builder, array_item_type.dtype, n_elems, dict_arr_ref, c
    )

    # alloc offsets
    n_arrays_plus_1 = builder.add(n_arrays, lir.Constant(lir.IntType(64), 1))
    offsets_meminfo = gen_alloc_meminfo(context, builder, n_arrays_plus_1, offset_type)

    offsets_ptr = builder.bitcast(
        context.nrt.meminfo_data(builder, offsets_meminfo),
        context.get_data_type(offset_type).as_pointer(),
    )
    # offsets[0] = 0
    builder.store(context.get_constant(offset_type, 0), offsets_ptr)
    # offsets[n] = n_elems
    builder.store(
        builder.trunc(
            builder.extract_value(n_elems, 0), lir.IntType(offset_type.bitwidth)
        ),
        builder.gep(offsets_ptr, [n_arrays]),
    )
    payload.offsets = offsets_meminfo

    # alloc null bitmap
    n_bitmask_bytes = builder.udiv(
        builder.add(n_arrays, lir.Constant(lir.IntType(64), 7)),
        lir.Constant(lir.IntType(64), 8),
    )
    null_bitmap_meminfo = gen_alloc_meminfo(
        context, builder, n_bitmask_bytes, types.uint8
    )

    null_bitmap_ptr = context.nrt.meminfo_data(builder, null_bitmap_meminfo)
    payload.null_bitmap = null_bitmap_meminfo

    builder.store(payload._getvalue(), meminfo_data_ptr)

    return meminfo, payload.data, offsets_ptr, null_bitmap_ptr


@unbox(ArrayItemArrayType)
def unbox_array_item_array(typ, val, c):
    """
    Unbox a numpy array with array of data values.
    """
    return bodo.libs.array.unbox_nested_array(typ, val, c)


def _get_array_item_arr_payload(context, builder, arr_typ, arr):
    """get payload struct proxy for a array(item) value"""
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


@box(ArrayItemArrayType)
def box_array_item_arr(typ, val, c):
    """box packed native representation of array of item array into python objects"""
    return bodo.libs.array.box_nested_array(typ, val, c)


def lower_pre_alloc_array_item_array(context, builder, sig, args):
    array_item_type = sig.return_type
    num_arrs, num_values, dtype = args
    # make sure 'num_values' has -1 for all unknown alloc counts
    n_elem_alloc_counts = bodo.utils.transform.get_type_alloc_counts(
        array_item_type.dtype
    )
    n_elems_type = sig.args[1]
    if not isinstance(n_elems_type, types.UniTuple):
        num_values = cgutils.pack_array(
            builder,
            [lir.Constant(lir.IntType(64), -1) for _ in range(n_elem_alloc_counts)],
        )
    elif n_elems_type.count < n_elem_alloc_counts:
        num_values = cgutils.pack_array(
            builder,
            [builder.extract_value(num_values, i) for i in range(n_elems_type.count)]
            + [
                lir.Constant(lir.IntType(64), -1)
                for _ in range(n_elem_alloc_counts - n_elems_type.count)
            ],
        )
    dict_arr_ref = None if isinstance(sig.args[2], types.TypeRef) else dtype
    meminfo, _, _, _ = construct_array_item_array(
        context, builder, array_item_type, num_arrs, num_values, dict_arr_ref
    )
    array_item_array = context.make_helper(builder, array_item_type)
    array_item_array.meminfo = meminfo
    return array_item_array._getvalue()


@intrinsic
def pre_alloc_array_item_array(typingctx, num_arrs_typ, num_values_typ, dtype_typ):
    assert isinstance(
        num_arrs_typ, types.Integer
    ), "pre_alloc_array_item_array: num_arrs should be integer"

    array_item_type = ArrayItemArrayType(unwrap_typeref(dtype_typ))
    num_values_typ = types.unliteral(num_values_typ)  # avoid e.g. (int64, literal(0))
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
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_array_item_arr_ext_pre_alloc_array_item_array = (
    pre_alloc_array_item_array_equiv
)


def array_to_repeated_array_item_array(
    scalar_arr, length, data_arr_type
):  # pragma: no cover
    pass


@overload(array_to_repeated_array_item_array, no_unliteral=True)
def overload_array_to_repeated_array_item_array(scalar_arr, length, data_arr_type):
    """
    Create an ArrayItemArray of length `length` by repeating scalar_arr `length` times

    Args:
        scalar_arr (array): The array to be repeated
        length (int): Length of the output ArrayItemArray
        data_arr_type(types.Type): Type of scalar_arr
    Returns:
        An ArrayItemArray(date_arr_type) of length `length`

    Example:
        array_to_repeated_array_item_array([1, 2], 3, types.Array(types.int32, 1, "C")) ->
            [[1, 2], [1, 2], [1, 2]]
    """

    def impl(scalar_arr, length, data_arr_type):  # pragma: no cover
        nested_counts = init_nested_counts(data_arr_type)
        for i in range(length):
            nested_counts = add_nested_counts(nested_counts, scalar_arr)
        out_arr = pre_alloc_array_item_array(length, nested_counts, scalar_arr)
        for i in range(length):
            out_arr[i] = scalar_arr
        return out_arr

    return impl


def init_array_item_array_codegen(context, builder, signature, args):
    """codegen for initializing an array(item) array from individual data/offset/bitmap
    arrays.
    """
    n_arrays, data, offsets, null_bitmap = args
    array_item_type = signature.return_type

    # TODO: refactor with construct_array_item_array
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
    payload.data = data
    # unwrap meminfos from Numpy arrays to store in payload
    payload.offsets = context.make_array(types.Array(offset_type, 1, "C"))(
        context, builder, offsets
    ).meminfo
    payload.null_bitmap = context.make_array(types.Array(types.uint8, 1, "C"))(
        context, builder, null_bitmap
    ).meminfo
    payload_val = payload._getvalue()
    builder.store(payload_val, meminfo_data_ptr)

    # increase refcount of stored values
    context.nrt.incref(builder, payload_type, payload_val)

    array_item_array = context.make_helper(builder, array_item_type)
    array_item_array.meminfo = meminfo
    return array_item_array._getvalue()


@intrinsic
def init_array_item_array(
    typingctx, n_arrays_typ, data_type, offsets_typ, null_bitmap_typ
):
    """Create a ArrayItemArray with provided offsets, data and null bitmap values."""
    assert null_bitmap_typ == types.Array(
        types.uint8, 1, "C"
    ), "init_array_item_array: null_bitmap should be Numpy array of uint8 values"

    ret_typ = ArrayItemArrayType(data_type)
    sig = ret_typ(types.int64, data_type, offsets_typ, null_bitmap_typ)
    return sig, init_array_item_array_codegen


@intrinsic
def get_offsets(typingctx, arr_typ):
    assert isinstance(
        arr_typ, ArrayItemArrayType
    ), "get_offsets: ArrayItemArrayType expected"

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_array_item_arr_payload(context, builder, arr_typ, arr)
        n_arrays_plus_1 = builder.add(
            payload.n_arrays, lir.Constant(lir.IntType(64), 1)
        )
        offsets_arr = meminfo_to_np_arr(
            context,
            builder,
            payload.offsets,
            lir.Constant(lir.IntType(64), 0),
            n_arrays_plus_1,
            types.Array(offset_type, 1, "C"),
        )
        return impl_ret_borrowed(context, builder, sig.return_type, offsets_arr)

    return types.Array(offset_type, 1, "C")(arr_typ), codegen


@intrinsic
def get_offsets_ind(typingctx, arr_typ, ind_t):
    """get offsets[ind] without wrapping in Numpy array (can be ~2x faster)"""
    assert isinstance(arr_typ, ArrayItemArrayType)

    def codegen(context, builder, sig, args):
        (arr, ind) = args
        payload = _get_array_item_arr_payload(context, builder, arr_typ, arr)
        data_ptr = context.nrt.meminfo_data(builder, payload.offsets)
        offsets = builder.bitcast(
            data_ptr, lir.IntType(offset_type.bitwidth).as_pointer()
        )
        return builder.load(builder.gep(offsets, [ind]))

    return offset_type(arr_typ, types.int64), codegen


@intrinsic
def get_data(typingctx, arr_typ):
    """get underlying array used for storing data elements.
    NOTE: May have more capacity than necessary (similar to std::vector).
    Use offsets[n] to get total number of elements instead of len(data_arr).
    """
    assert isinstance(arr_typ, ArrayItemArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_array_item_arr_payload(context, builder, arr_typ, arr)
        return impl_ret_borrowed(context, builder, sig.return_type, payload.data)

    return (arr_typ.dtype)(arr_typ), codegen


@intrinsic
def get_null_bitmap(typingctx, arr_typ):
    assert isinstance(arr_typ, ArrayItemArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_array_item_arr_payload(context, builder, arr_typ, arr)
        n_bitmask_bytes = builder.udiv(
            builder.add(payload.n_arrays, lir.Constant(lir.IntType(64), 7)),
            lir.Constant(lir.IntType(64), 8),
        )
        null_bitmap_arr = meminfo_to_np_arr(
            context,
            builder,
            payload.null_bitmap,
            lir.Constant(lir.IntType(64), 0),
            n_bitmask_bytes,
            types.Array(types.uint8, 1, "C"),
        )
        return impl_ret_borrowed(context, builder, sig.return_type, null_bitmap_arr)

    return types.Array(types.uint8, 1, "C")(arr_typ), codegen


def alias_ext_single_array(lhs_name, args, alias_map, arg_aliases):
    """
    Aliasing function for offsets, data, and null bitmap.
    """
    assert len(args) == 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("get_offsets", "bodo.libs.array_item_arr_ext")
] = alias_ext_single_array


numba.core.ir_utils.alias_func_extensions[
    ("get_data", "bodo.libs.array_item_arr_ext")
] = alias_ext_single_array

numba.core.ir_utils.alias_func_extensions[
    ("get_null_bitmap", "bodo.libs.array_item_arr_ext")
] = alias_ext_single_array


@intrinsic
def get_n_arrays(typingctx, arr_typ):
    assert isinstance(arr_typ, ArrayItemArrayType)

    def codegen(context, builder, sig, args):
        (arr,) = args
        payload = _get_array_item_arr_payload(context, builder, arr_typ, arr)
        return payload.n_arrays

    return types.int64(arr_typ), codegen


@intrinsic
def replace_data_arr(typingctx, arr_typ, data_typ):
    """replace the underlying data array of array(item) array"""
    assert isinstance(arr_typ, ArrayItemArrayType) and data_typ == arr_typ.dtype

    def codegen(context, builder, sig, args):
        (arr, new_data) = args
        # replace data array in payload
        array_item_array = context.make_helper(builder, arr_typ, arr)
        payload_type = ArrayItemArrayPayloadType(arr_typ)
        meminfo_void_ptr = context.nrt.meminfo_data(builder, array_item_array.meminfo)
        meminfo_data_ptr = builder.bitcast(
            meminfo_void_ptr, context.get_value_type(payload_type).as_pointer()
        )
        payload = cgutils.create_struct_proxy(payload_type)(
            context, builder, builder.load(meminfo_data_ptr)
        )
        context.nrt.decref(builder, data_typ, payload.data)
        payload.data = new_data
        context.nrt.incref(builder, data_typ, new_data)
        builder.store(payload._getvalue(), meminfo_data_ptr)

    return types.none(arr_typ, data_typ), codegen


# Inlining at Numba level to work around caching issues for recursive functions.
# See https://bodo.atlassian.net/browse/BSE-3359
@numba.njit(no_cpython_wrapper=True, inline="always")
def ensure_data_capacity(arr, old_size, new_size):  # pragma: no cover
    """
    make sure the internal data array has enough space for 'new_size' number of elements
    'old_size' is the current number of valid elements (to enable data copy)
    """
    data = get_data(arr)
    old_capacity = len(data)
    if old_capacity < new_size:
        # double the capacity similar to std::vector
        new_capacity = max(2 * old_capacity, new_size)
        new_data = bodo.libs.array_kernels.resize_and_copy(data, old_size, new_capacity)
        replace_data_arr(arr, new_data)


# Inlining at Numba level to work around caching issues for recursive functions.
# See https://bodo.atlassian.net/browse/BSE-3359
@numba.njit(no_cpython_wrapper=True, inline="always")
def trim_excess_data(arr):  # pragma: no cover
    """
    make sure the internal data array doesn't have an extra data (to enable
    operations expecting the exact length). This assumes the array is fully
    allocated.
    """
    data = get_data(arr)
    offsets = get_offsets(arr)
    capacity = len(data)
    size = offsets[-1]
    if capacity != size:
        # Shrink the array to the size
        new_data = bodo.libs.array_kernels.resize_and_copy(data, size, size)
        replace_data_arr(arr, new_data)


@overload(len, no_unliteral=True)
def overload_array_item_arr_len(A):
    if isinstance(A, ArrayItemArrayType):
        return lambda A: get_n_arrays(A)  # pragma: no cover


@overload_attribute(ArrayItemArrayType, "shape")
def overload_array_item_arr_shape(A):
    return lambda A: (get_n_arrays(A),)  # pragma: no cover


@overload_attribute(ArrayItemArrayType, "dtype")
def overload_array_item_arr_dtype(A):
    return lambda A: np.object_  # pragma: no cover


@overload_attribute(ArrayItemArrayType, "ndim")
def overload_array_item_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(ArrayItemArrayType, "nbytes")
def overload_array_item_arr_nbytes(A):
    return (
        lambda A: get_data(A).nbytes + get_offsets(A).nbytes + get_null_bitmap(A).nbytes
    )  # pragma: no cover


@overload(operator.getitem, no_unliteral=True)
def array_item_arr_getitem_array(arr, ind):
    if not isinstance(arr, ArrayItemArrayType):
        return

    if isinstance(ind, types.Integer):
        # returning empty array if NA due to type stability issues
        # TODO: warning if value is NA?
        def array_item_arr_getitem_impl(arr, ind):  # pragma: no cover
            if ind < 0:
                ind += len(arr)
            offsets = get_offsets(arr)
            data = get_data(arr)
            l_start_offset = offsets[np.int64(ind)]
            l_end_offset = offsets[np.int64(ind) + 1]
            return data[l_start_offset:l_end_offset]

        return array_item_arr_getitem_impl

    # bool arr indexing.
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
                if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
                    n_arrays += 1
                    arr_item = arr[i]
                    nested_counts = add_nested_counts(nested_counts, arr_item)

            out_arr = pre_alloc_array_item_array(n_arrays, nested_counts, get_data(arr))
            out_null_bitmap = get_null_bitmap(out_arr)

            # write output
            out_ind = 0
            for j in range(n):
                if not bodo.libs.array_kernels.isna(ind, j) and ind[j]:
                    out_arr[out_ind] = arr[j]
                    # set NA
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(null_bitmap, j)
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

            out_arr = pre_alloc_array_item_array(n_arrays, nested_counts, get_data(arr))
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

    # This should be the only ArrayItemArray implementation.
    # We only expect to reach this case if more ind options are added.
    raise BodoError(
        f"getitem for ArrayItemArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem)
def array_item_arr_setitem(A, idx, val):
    if not isinstance(A, ArrayItemArrayType):
        return

    # scalar case
    # NOTE: assuming that the array is being built and all previous elements are set
    # TODO: make sure array is being build
    if isinstance(idx, types.Integer):

        def impl_scalar(A, idx, val):  # pragma: no cover
            offsets = get_offsets(A)
            null_bitmap = get_null_bitmap(A)
            if idx == 0:
                offsets[0] = 0

            n_items = len(val)
            required_capacity = offsets[idx] + n_items
            ensure_data_capacity(A, offsets[idx], required_capacity)
            data = get_data(A)
            offsets[idx + 1] = offsets[idx] + n_items
            data[offsets[idx] : offsets[idx + 1]] = val
            bodo.libs.int_arr_ext.set_bit_to_arr(null_bitmap, idx, 1)
            # Length of the data array has to be set properly after filling the data to
            # avoid violating assumptions in other parts of the code.
            # For example, length of string array is used when passed to C++.
            # See https://bodo.atlassian.net/browse/BSE-1915
            if idx == (get_n_arrays(A) - 1):
                trim_excess_data(A)

        return impl_scalar

    # setting a slice to individual element
    if isinstance(idx, types.SliceType) and A.dtype == val:

        def impl_slice_elem(A, idx, val):  # pragma: no cover
            slice_idx = numba.cpython.unicode._normalize_slice(idx, len(A))
            for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                A[i] = val

        return impl_slice_elem

    # slice case (used in unboxing)
    if isinstance(idx, types.SliceType) and is_iterable_type(val):

        def impl_slice(A, idx, val):  # pragma: no cover
            # TODO: set scalar_to_arr_len
            val = bodo.utils.conversion.coerce_to_array(val, use_nullable_array=True)
            # get output arrays
            offsets = get_offsets(A)
            null_bitmap = get_null_bitmap(A)

            # get input arrays
            val_offsets = get_offsets(val)
            val_data = get_data(val)
            val_null_bitmap = get_null_bitmap(val)
            # NOTE: need to avoid len(val_data) in case over-allocated
            val_data_len = np.int64(val_offsets[len(val)])

            n = len(A)
            slice_idx = numba.cpython.unicode._normalize_slice(idx, n)
            start, stop = slice_idx.start, slice_idx.stop
            assert slice_idx.step == 1

            # copy data from first offset
            if start == 0:
                offsets[start] = 0
            start_offset = offsets[start]
            required_capacity = start_offset + val_data_len
            ensure_data_capacity(A, start_offset, required_capacity)
            data = get_data(A)
            # NOTE: val_data could be over-allocated so :val_data_len is necessary
            data[start_offset : start_offset + val_data_len] = val_data[:val_data_len]

            # copy offsets (n+1 elements)
            # NOTE: val_data could be over-allocated so :(len(val)+1) is necessary
            offsets[start : stop + 1] = val_offsets[: (len(val) + 1)] + start_offset

            # copy null bits
            val_ind = 0
            for i in range(start, stop):
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(val_null_bitmap, val_ind)
                bodo.libs.int_arr_ext.set_bit_to_arr(null_bitmap, i, bit)
                val_ind += 1

            if stop == get_n_arrays(A):
                trim_excess_data(A)

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


def _convert_array_type(arr, new_arr_type):
    pass


@overload(_convert_array_type)
def overload_convert_array_type(arr, new_arr_type):
    """Convert array to match target type.
    Used in BodoSQL typically for nullability conversion.
    """
    if arr == unwrap_typeref(new_arr_type):
        return lambda arr, new_arr_type: arr  # pragma: no cover

    def impl(arr, new_arr_type):  # pragma: no cover
        new_arr = bodo.utils.utils.alloc_type(len(arr), new_arr_type, (-1,))
        # Reuse setitem element conversions
        new_arr[:] = arr
        return new_arr

    return impl


def sql_null_checking_scalar_conv_array(arg0, new_arr_type):  # pragma: no cover
    pass


@overload(sql_null_checking_scalar_conv_array)
def overload_sql_null_checking_scalar_conv_array(arg0, new_arr_type):
    """Convert scalar value of nested array to target array type for BodoSQL.
    See similar scalar conversion functions such as sql_null_checking_scalar_conv_int8.
    """

    if arg0 == types.none:
        return lambda arg0, new_arr_type: None  # pragma: no cover

    elif isinstance(arg0, types.Optional):

        def impl(arg0, new_arr_type):  # pragma: no cover
            if arg0 is None:
                return None
            else:
                arg0 = bodo.utils.indexing.unoptional(arg0)
                return _convert_array_type(arg0, new_arr_type)

        return impl

    else:
        return lambda arg0, new_arr_type: _convert_array_type(
            arg0, new_arr_type
        )  # pragma: no cover
