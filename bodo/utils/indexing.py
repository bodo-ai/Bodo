# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Collection of utility functions for indexing implementation (getitem/setitem)
"""
import numpy as np

import numba
from numba.extending import register_jitable, overload
import bodo


@register_jitable
def get_new_null_mask_bool_index(old_mask, ind, n):  # pragma: no cover
    """create a new null bitmask for output of indexing using bool index 'ind'.
    'n' is the total number of elements in original array (not bytes).
    """
    n_bytes = (n + 7) >> 3
    new_mask = np.empty(n_bytes, np.uint8)
    curr_bit = 0
    for i in range(len(ind)):
        if ind[i]:
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
    ind = bodo.utils.conversion.coerce_to_ndarray(ind)
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
    n_bytes = (n + 7) >> 3
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
    ind_t = bodo.utils.conversion.coerce_to_ndarray(ind)
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
    n_bytes = (span + 7) >> 3
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


@register_jitable
def array_setitem_int_index(A, idx, val):  # pragma: no cover
    """implements setitem with int index for arrays that have a '_data' attribute and
    '_null_bitmap' attribute (e.g. int/bool/decimal/date). The value is assumed to be
    another array of same type.
    Covered by test_series_iloc_setitem_list_int.
    """
    val = bodo.utils.conversion.coerce_to_array(val, use_nullable_array=True)
    n = len(val._data)
    for i in range(n):
        A._data[idx[i]] = val._data[i]
        bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(val._null_bitmap, i)
        bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx[i], bit)


@register_jitable
def array_setitem_bool_index(A, idx, val):  # pragma: no cover
    """implements setitem with bool index for arrays that have a '_data' attribute and
    '_null_bitmap' attribute (e.g. int/bool/decimal/date). The value is assumed to be
    another array of same type.
    Covered by test_series_iloc_setitem_list_bool.
    """
    val = bodo.utils.conversion.coerce_to_array(val, use_nullable_array=True)
    n = len(idx)
    val_ind = 0
    for i in range(n):
        if idx[i]:
            A._data[i] = val._data[val_ind]
            bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(val._null_bitmap, val_ind)
            bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, bit)
            val_ind += 1


@register_jitable
def array_setitem_slice_index(A, idx, val):  # pragma: no cover
    """implements setitem with slice index for arrays that have a '_data' attribute and
    '_null_bitmap' attribute (e.g. int/bool/decimal/date). The value is assumed to be
    another array of same type.
    Covered by test_series_iloc_setitem_slice.
    """
    val = bodo.utils.conversion.coerce_to_array(val, use_nullable_array=True)
    n = len(A._data)
    # using setitem directly instead of copying in loop since
    # Array setitem checks for memory overlap and copies source
    A._data[idx] = val._data
    # XXX: conservative copy of bitmap in case there is overlap
    # TODO: check for overlap and copy only if necessary
    src_bitmap = val._null_bitmap.copy()
    slice_idx = numba.cpython.unicode._normalize_slice(idx, n)
    val_ind = 0
    for i in range(slice_idx.start, slice_idx.stop, slice_idx.step):
        bit = bodo.libs.int_arr_ext.get_bit_bitmap_arr(src_bitmap, val_ind)
        bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, i, bit)
        val_ind += 1


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

    if isinstance(arr_item, bodo.libs.array_item_arr_ext.ArrayItemArrayType):
        return lambda nested_counts, arr_item: (
            nested_counts[0] + len(arr_item),
        ) + add_nested_counts(
            nested_counts[1:], bodo.libs.array_item_arr_ext.get_data(arr_item)
        )  # pragma: no cover

    if arr_item == bodo.string_array_type:
        return lambda nested_counts, arr_item: (
            nested_counts[0] + len(arr_item),
            np.int64(bodo.libs.str_arr_ext.num_total_chars(arr_item)),
        )  # pragma: no cover

    if bodo.utils.utils.is_array_typ(arr_item, False):
        return lambda nested_counts, arr_item: (
            nested_counts[0] + len(arr_item),
        )  # pragma: no cover

    if arr_item == bodo.string_type:
        return lambda nested_counts, arr_item: (
            nested_counts[0] + get_utf8_size(arr_item),
        )  # pragma: no cover

    return lambda nested_counts, arr_item: ()  # pragma: no cover
