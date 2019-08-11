"""Nullable integer array corresponding to Pandas IntegerArray.
However, nulls are stored in bit arrays similar to Arrow's arrays.
"""
import operator
import pandas as pd
import numpy as np
import numba
import bodo
from numba import types
from numba import cgutils
from numba.extending import (typeof_impl, type_callable, models,
    register_model, NativeValue, make_attribute_wrapper, lower_builtin, box,
    unbox, lower_getattr, intrinsic, overload_method, overload,
    overload_attribute)
from bodo.libs.str_arr_ext import kBitmask
from llvmlite import ir as lir


class IntegerArrayType(types.ArrayCompatible):
    def __init__(self, dtype):
        self.dtype = dtype
        super(IntegerArrayType, self).__init__(
            name='IntegerArrayType({})'.format(dtype))

    @property
    def as_array(self):
        return types.Array(self.dtype, 1, 'C')

    def copy(self):
        return IntegerArrayType(self.dtype)


# store data and nulls as regular numpy arrays without payload machineray
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
@register_model(IntegerArrayType)
class IntegerArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', types.Array(fe_type.dtype, 1, 'C')),
            ('null_bitmap', types.Array(types.uint8, 1, 'C')),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IntegerArrayType, 'data', '_data')
make_attribute_wrapper(IntegerArrayType, 'null_bitmap', '_null_bitmap')


@typeof_impl.register(pd.arrays.IntegerArray)
def _typeof_pd_int_array(val, c):
    bitwidth = 8 * val.dtype.itemsize
    kind = '' if val.dtype.kind == 'i' else 'u'
    dtype = getattr(types, '{}int{}'.format(kind, bitwidth))
    return IntegerArrayType(dtype)


@numba.extending.register_jitable
def mask_arr_to_bitmap(mask_arr):
    n = len(mask_arr)
    n_bytes = (n + 7) >> 3
    bit_arr = np.empty(n_bytes, np.uint8)
    for i in range(n):
        b_ind = i // 8
        bit_arr[b_ind] ^= np.uint8(
            -np.uint8(not mask_arr[i]) ^ bit_arr[b_ind]) & kBitmask[i % 8]

    return bit_arr


@unbox(IntegerArrayType)
def unbox_int_array(typ, obj, c):
    """
    Convert a pd.arrays.IntegerArray object to a native IntegerArray structure.
    """
    int_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    data_obj = c.pyapi.object_getattr_string(obj, "_data")
    int_arr.data = c.pyapi.to_native_value(
        types.Array(typ.dtype, 1, 'C'), data_obj).value

    mask_arr_obj = c.pyapi.object_getattr_string(obj, "_mask")
    mask_arr = c.pyapi.to_native_value(
        types.Array(types.bool_, 1, 'C'), mask_arr_obj).value

    bt_typ = c.context.typing_context.resolve_value_type(mask_arr_to_bitmap)
    bt_sig = bt_typ.get_call_type(
        c.context.typing_context, (types.Array(types.bool_, 1, 'C'),), {})
    bt_impl = c.context.get_function(bt_typ, bt_sig)
    int_arr.null_bitmap = bt_impl(c.builder, (mask_arr,))

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(int_arr._getvalue(), is_error=is_error)
