# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""
Collection of utility functions for indexing implementation (getitem/setitem)
"""
import numpy as np

import numba
from numba.extending import register_jitable
import bodo


@register_jitable
def get_new_null_mask_bool_index(old_mask, ind, n):
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
def get_new_null_mask_int_index(old_mask, ind, n):
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
