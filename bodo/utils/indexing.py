"""
Collection of utility functions for indexing implementation (getitem/setitem)
"""

import operator

import numba
import numpy as np
from numba.core import types
from numba.core.imputils import impl_ret_borrowed
from numba.extending import intrinsic, overload, register_jitable

import bodo
from bodo.utils.typing import BodoError


@register_jitable  # do not marke inline as it causes conversion to fail
def bitmap_size(n: int) -> int:  # pragma: no cover
    """Get the number of bytes necessary to store an n-bit bitmap."""
    return (n + 7) >> 3


@register_jitable
def get_dt64_bitmap_fill(dt64_val):  # pragma: no cover
    """Returns bitmap fill value for nullable dt64 arrays: 0s if null value else 1s."""
    return 0 if np.isnat(dt64_val) else 0xFF


@register_jitable
def get_new_null_mask_bool_index(old_mask, ind, n):  # pragma: no cover
    """create a new null bitmask for output of indexing using bool index 'ind'.
    'n' is the total number of elements in original array (not bytes).
    """
    n_bytes = bitmap_size(n)
    new_mask = np.empty(n_bytes, np.uint8)
    curr_bit = 0
    for i in range(len(ind)):
        if not bodo.libs.array_kernels.isna(ind, i) and ind[i]:
            bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, i)
            bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
            curr_bit += 1
    return new_mask


@register_jitable
def array_getitem_bool_index(A, ind):  # pragma: no cover
    """implements getitem with bool index for arrays that have a '_data' attribute and
    '_null_bitmap' attribute (e.g. int/bool/decimal/date).
    Covered by test_series_iloc_getitem_array_bool.
    """
    ind = bodo.utils.conversion.coerce_to_array(ind)
    old_mask = A._null_bitmap
    new_data = A._data[ind]
    n = len(new_data)
    new_mask = get_new_null_mask_bool_index(old_mask, ind, n)
    return new_data, new_mask


@register_jitable
def get_new_null_mask_int_index(old_mask, ind, n):  # pragma: no cover
    """create a new null bitmask for output of indexing using integer index 'ind'.
    'n' is the total number of elements in original array (not bytes).
    """
    n_bytes = bitmap_size(n)
    new_mask = np.empty(n_bytes, np.uint8)
    curr_bit = 0
    for i in range(len(ind)):
        bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, ind[i])
        bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
        curr_bit += 1
    return new_mask


@register_jitable
def array_getitem_int_index(A, ind):  # pragma: no cover
    """implements getitem with int index for arrays that have a '_data' attribute and
    '_null_bitmap' attribute (e.g. int/bool/decimal/date).
    Covered by test_series_iloc_getitem_array_int.
    """
    ind_t = bodo.utils.conversion.coerce_to_array(ind)
    old_mask = A._null_bitmap
    new_data = A._data[ind_t]
    n = len(new_data)
    new_mask = get_new_null_mask_int_index(old_mask, ind_t, n)
    return new_data, new_mask


@register_jitable
def get_new_null_mask_slice_index(old_mask, ind, n):  # pragma: no cover
    """create a new null bitmask for output of indexing using slice index 'ind'.
    'n' is the total number of elements in original array (not bytes).
    """
    slice_idx = numba.cpython.unicode._normalize_slice(ind, n)
    span = numba.cpython.unicode._slice_span(slice_idx)
    n_bytes = bitmap_size(span)
    new_mask = np.empty(n_bytes, np.uint8)
    curr_bit = 0
    for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
        bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(old_mask, i)
        bodo.libs.int_arr_ext.set_bit_to_arr(new_mask, curr_bit, bit)
        curr_bit += 1
    return new_mask


@register_jitable
def array_getitem_slice_index(A, ind):  # pragma: no cover
    """implements getitem with slice index for arrays that have a '_data' attribute and
    '_null_bitmap' attribute (e.g. int/bool/decimal/date).
    Covered by test_series_iloc_getitem_slice.
    """
    n = len(A._data)
    old_mask = A._null_bitmap
    new_data = np.ascontiguousarray(A._data[ind])
    new_mask = get_new_null_mask_slice_index(old_mask, ind, n)
    return new_data, new_mask


def array_setitem_int_index(A, idx, val):  # pragma: no cover
    return


@overload(array_setitem_int_index, no_unliteral=True)
def array_setitem_int_index_overload(A, idx, val):  # pragma: no cover
    """implements setitem with int index for arrays that have a '_data' attribute and
    '_null_bitmap' attribute (e.g. int/bool/decimal/date). The value is assumed to be
    another array of same type.
    Covered by test_series_iloc_setitem_list_int.
    """
    if bodo.utils.utils.is_array_typ(val) or bodo.utils.typing.is_iterable_type(val):

        def impl_arr(A, idx, val):  # pragma: no cover
            val = bodo.utils.conversion.coerce_to_array(val, use_nullable_array=True)
            n = len(val._data)
            for i in range(n):
                A._data[idx[i]] = val._data[i]
                bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(val._null_bitmap, i)
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx[i], bit)

        return impl_arr

    if bodo.utils.typing.is_scalar_type(val):

        def impl_scalar(A, idx, val):  # pragma: no cover
            for i in idx:
                A._data[i] = val
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

        return impl_scalar

    # Safeguard against gaps in Array setitem to avoid failing in compilation.
    raise BodoError(
        f"setitem not supported for {A} with value {val}"
    )  # pragma: no cover


def array_setitem_bool_index(A, idx, val):  # pragma: no cover
    A[idx] = val


@overload(array_setitem_bool_index, no_unliteral=True)
def array_setitem_bool_index_overload(A, idx, val):
    """implements setitem with bool index for arrays that have a '_data' attribute and
    '_null_bitmap' attribute (e.g. int/bool/decimal/date). The value is assumed to be
    another array of same type.
    Covered by test_series_iloc_setitem_list_bool.
    """
    if bodo.utils.utils.is_array_typ(val) or bodo.utils.typing.is_iterable_type(val):

        def impl_arr(A, idx, val):  # pragma: no cover
            idx = bodo.utils.conversion.coerce_to_array(idx)
            val = bodo.utils.conversion.coerce_to_array(val, use_nullable_array=True)
            n = len(idx)
            val_ind = 0
            for i in range(n):
                if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                    A._data[i] = val._data[val_ind]
                    bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(
                        val._null_bitmap, val_ind
                    )
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, bit)
                    val_ind += 1

        return impl_arr

    if bodo.utils.typing.is_scalar_type(val):

        def impl_scalar(A, idx, val):  # pragma: no cover
            idx = bodo.utils.conversion.coerce_to_array(idx)
            n = len(idx)
            val_ind = 0
            for i in range(n):
                if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                    A._data[i] = val
                    bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)
                    val_ind += 1

        return impl_scalar

    # Safeguard against gaps in Array setitem to avoid failing in compilation.
    raise BodoError(
        f"setitem not supported for {A} with value {val}"
    )  # pragma: no cover


@register_jitable
def setitem_slice_index_null_bits(dst_bitmap, src_bitmap, idx, n):  # pragma: no cover
    """set null bits for setitem with slice index for nullable arrays."""
    slice_idx = numba.cpython.unicode._normalize_slice(idx, n)
    val_ind = 0
    for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
        bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(src_bitmap, val_ind)
        bodo.libs.int_arr_ext.set_bit_to_arr(dst_bitmap, i, bit)
        val_ind += 1


def array_setitem_slice_index(A, idx, val):  # pragma: no cover
    return


@overload(array_setitem_slice_index, no_unliteral=True)
def array_setitem_slice_index_overload(A, idx, val):  # pragma: no cover
    """implements setitem with slice index for arrays that have a '_data' attribute and
    '_null_bitmap' attribute (e.g. int/bool/decimal/date). The value is assumed to be
    another array of same type or a scalar.
    Covered by test_series_iloc_setitem_slice.
    """
    if bodo.utils.utils.is_array_typ(val) or bodo.utils.typing.is_iterable_type(val):

        def impl_arr(A, idx, val):  # pragma: no cover
            val = bodo.utils.conversion.coerce_to_array(
                val,
                use_nullable_array=True,
            )
            n = len(A._data)
            # using setitem directly instead of copying in loop since
            # Array setitem checks for memory overlap and copies source
            A._data[idx] = val._data
            # XXX: conservative copy of bitmap in case there is overlap
            # TODO: check for overlap and copy only if necessary
            src_bitmap = val._null_bitmap.copy()
            setitem_slice_index_null_bits(A._null_bitmap, src_bitmap, idx, n)

        return impl_arr

    if bodo.utils.typing.is_scalar_type(val):

        def impl_scalar(A, idx, val):  # pragma: no cover
            slice_idx = numba.cpython.unicode._normalize_slice(idx, len(A))
            for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                A._data[i] = val
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, 1)

        return impl_scalar

    # Safeguard against gaps in Array setitem to avoid failing in compilation.
    raise BodoError(
        f"setitem not supported for {A} with value {val}"
    )  # pragma: no cover


def untuple_if_one_tuple(v):
    return v


@overload(untuple_if_one_tuple)
def untuple_if_one_tuple_overload(v):
    """if 'v' is a single element tuple, return 'v[0]' to avoid unnecessary tuple value."""
    if isinstance(v, types.BaseTuple) and len(v.types) == 1:
        return lambda v: v[0]  # pragma: no cover

    return lambda v: v


def init_nested_counts(arr_typ):  # pragma: no cover
    return (0,)


@overload(init_nested_counts)
def overload_init_nested_counts(arr_typ):
    """initialize nested counts for counting nested elements in array of type 'arr_typ'.
    E.g. array(array(int)) will return (0, 0)
    """
    arr_typ = arr_typ.instance_type
    if (
        isinstance(arr_typ, bodo.libs.array_item_arr_ext.ArrayItemArrayType)
        or arr_typ == bodo.string_array_type
    ):
        data_arr_typ = arr_typ.dtype
        return lambda arr_typ: (0,) + init_nested_counts(
            data_arr_typ
        )  # pragma: no cover

    if bodo.utils.utils.is_array_typ(arr_typ, False) or arr_typ == bodo.string_type:
        return lambda arr_typ: (0,)  # pragma: no cover

    return lambda arr_typ: ()  # pragma: no cover


def add_nested_counts(nested_counts, arr_item):  # pragma: no cover
    return (0,)


@overload(add_nested_counts)
def overload_add_nested_counts(nested_counts, arr_item):
    """add nested counts of elements in 'arr_item', which could be array(item) array or
    regular array, to nested counts. For example, [[1, 2, 3], [2]] will add (2, 4)
    """
    from bodo.libs.str_arr_ext import get_utf8_size

    arr_item = arr_item.type if isinstance(arr_item, types.Optional) else arr_item

    # array(array)
    if isinstance(arr_item, bodo.libs.array_item_arr_ext.ArrayItemArrayType):
        return lambda nested_counts, arr_item: (
            nested_counts[0] + len(arr_item),
        ) + add_nested_counts(
            nested_counts[1:], bodo.libs.array_item_arr_ext.get_data(arr_item)
        )  # pragma: no cover

    # list is similar to array
    if isinstance(arr_item, types.List):
        return lambda nested_counts, arr_item: add_nested_counts(
            nested_counts, bodo.utils.conversion.coerce_to_array(arr_item)
        )  # pragma: no cover

    # string array
    if arr_item == bodo.string_array_type:
        return lambda nested_counts, arr_item: (
            nested_counts[0] + len(arr_item),
            nested_counts[1]
            + np.int64(bodo.libs.str_arr_ext.num_total_chars(arr_item)),
        )  # pragma: no cover

    # other arrays
    if bodo.utils.utils.is_array_typ(arr_item, False):
        return lambda nested_counts, arr_item: (
            nested_counts[0] + len(arr_item),
        )  # pragma: no cover

    # string
    if arr_item == bodo.string_type:
        return lambda nested_counts, arr_item: (
            nested_counts[0] + get_utf8_size(arr_item),
        )  # pragma: no cover

    return lambda nested_counts, arr_item: ()  # pragma: no cover


@overload(operator.setitem)
def none_optional_setitem_overload(A, idx, val):
    if not bodo.utils.utils.is_array_typ(A, False):
        return  # pragma: no cover

    elif val == types.none:
        if isinstance(idx, types.Integer):
            return lambda A, idx, val: bodo.libs.array_kernels.setna(
                A, idx
            )  # pragma: no cover

        elif bodo.utils.typing.is_list_like_index_type(idx) and isinstance(
            idx.dtype, types.Integer
        ):

            def setitem_none_int_arr(A, idx, val):  # pragma: no cover
                idx = bodo.utils.conversion.coerce_to_array(idx)
                for i in idx:
                    bodo.libs.array_kernels.setna(A, i)

            return setitem_none_int_arr

        elif (
            bodo.utils.typing.is_list_like_index_type(idx) and idx.dtype == types.bool_
        ):
            # Handle string array specially because we need to copy the data
            if A == bodo.string_array_type:

                def string_arr_impl(A, idx, val):  # pragma: no cover
                    n = len(A)
                    # NOTE: necessary to convert potential Series to array
                    idx = bodo.utils.conversion.coerce_to_array(idx)
                    out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
                    for i in numba.parfors.parfor.internal_prange(n):
                        if idx[i] or bodo.libs.array_kernels.isna(A, i):
                            out_arr[i] = ""
                            bodo.libs.str_arr_ext.str_arr_set_na(out_arr, i)
                        else:
                            out_arr[i] = A[i]  # TODO(ehsan): copy inplace

                    bodo.libs.str_arr_ext.move_str_binary_arr_payload(A, out_arr)

                return string_arr_impl

            def setitem_none_bool_arr(A, idx, val):  # pragma: no cover
                idx = bodo.utils.conversion.coerce_to_array(idx)
                n = len(idx)
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        bodo.libs.array_kernels.setna(A, i)

            return setitem_none_bool_arr

        elif isinstance(idx, types.SliceType):

            def setitem_none_slice(A, idx, val):  # pragma: no cover
                n = len(A)
                slice_idx = numba.cpython.unicode._normalize_slice(idx, n)
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    bodo.libs.array_kernels.setna(A, i)

            return setitem_none_slice

        raise BodoError(
            f"setitem for {A} with indexing type {idx} and None value not supported."
        )  # pragma: no cover

    elif isinstance(val, types.optional):
        if isinstance(idx, types.Integer):

            def impl_optional(A, idx, val):  # pragma: no cover
                if val is None:
                    bodo.libs.array_kernels.setna(A, idx)
                else:
                    A[idx] = bodo.utils.indexing.unoptional(val)

            return impl_optional

        elif bodo.utils.typing.is_list_like_index_type(idx) and isinstance(
            idx.dtype, types.Integer
        ):

            def setitem_optional_int_arr(A, idx, val):  # pragma: no cover
                idx = bodo.utils.conversion.coerce_to_array(idx)
                for i in idx:
                    if val is None:
                        bodo.libs.array_kernels.setna(A, i)
                        continue
                    A[i] = bodo.utils.indexing.unoptional(val)

            return setitem_optional_int_arr

        elif (
            bodo.utils.typing.is_list_like_index_type(idx) and idx.dtype == types.bool_
        ):
            # Handle string array specially because we need to copy the data
            if A == bodo.string_array_type:

                def string_arr_impl(A, idx, val):
                    if val is None:
                        A[idx] = None
                    else:
                        A[idx] = bodo.utils.indexing.unoptional(val)

                return string_arr_impl

            def setitem_optional_bool_arr(A, idx, val):  # pragma: no cover
                idx = bodo.utils.conversion.coerce_to_array(idx)
                n = len(idx)
                for i in range(n):
                    if not bodo.libs.array_kernels.isna(idx, i) and idx[i]:
                        if val is None:
                            bodo.libs.array_kernels.setna(A, i)
                            continue
                        A[i] = bodo.utils.indexing.unoptional(val)

            return setitem_optional_bool_arr

        elif isinstance(idx, types.SliceType):

            def setitem_optional_slice(A, idx, val):  # pragma: no cover
                n = len(A)
                slice_idx = numba.cpython.unicode._normalize_slice(idx, n)
                for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
                    if val is None:
                        bodo.libs.array_kernels.setna(A, i)
                        continue
                    A[i] = bodo.utils.indexing.unoptional(val)

            return setitem_optional_slice

        raise BodoError(
            f"setitem for {A} with indexing type {idx} and optional value not supported."
        )  # pragma: no cover


@intrinsic
def unoptional(typingctx, val_t=None):
    """Return value inside Optional type assuming that it is not None"""
    # just return input if not Optional
    if not isinstance(val_t, types.Optional):
        return val_t(val_t), lambda c, b, s, args: impl_ret_borrowed(
            c, b, val_t, args[0]
        )

    def codegen(context, builder, signature, args):  # pragma: no cover
        optval = context.make_helper(builder, val_t, args[0])
        # TODO: check optval.valid bit to be True
        out_val = optval.data
        context.nrt.incref(builder, val_t.type, out_val)
        return out_val

    return val_t.type(val_t), codegen


def scalar_optional_getitem(A, idx):  # pragma: no cover
    pass


@overload(scalar_optional_getitem)
def overload_scalar_optional_getitem(A, idx):
    """An implementation of getitem that returns None if the array is NULL. This is
    used by BodoSQL to select individual elements with correction optional type support
    inside CASE statements.

    Args:
        A (types.Type): Input Array
        idx (types.Integer): Index to fetch in the array.

    Raises:
        BodoError: The index is not the correct type.

    Returns:
        types.Type: A[idx] if the data isn't null and otherwise None.
    """
    if not isinstance(idx, types.Integer):
        raise BodoError("scalar_optional_getitem(): Can only select a single element")

    if bodo.utils.typing.is_nullable(A):
        # If the array type is nullable then have an optional return type.
        def impl(A, idx):  # pragma: no cover
            if bodo.libs.array_kernels.isna(A, idx):
                return None
            else:
                return bodo.utils.conversion.box_if_dt64(A[idx])

        return impl
    else:
        # If the data isn't nullable we don't need to return an optional type.
        return lambda A, idx: bodo.utils.conversion.box_if_dt64(
            A[idx]
        )  # pragma: no cover
